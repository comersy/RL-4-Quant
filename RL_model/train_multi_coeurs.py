"""
GRU-PPO Multi-Core Training Script

Architecture parallèle style IMPALA :
  - NUM_WORKERS workers collectent des épisodes en parallèle
  - 1 learner central fait les gradient updates
  - Le learner affiche les SPS globaux toutes les N updates
"""

import argparse
import os
import sys
import time
from collections import deque, namedtuple
from datetime import datetime
from pathlib import Path

# Allow running this file directly (`python RL_model/train_multi_coeurs.py`):
# put the project root on sys.path so `envs` / `data` are importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from envs.env import EPISODE_STEPS, MAX_PORTFOLIO, SPOT_HISTORY, OptionsEnv

# ============================================================================
# Config — changer NUM_WORKERS ici
# ============================================================================

NUM_WORKERS = 10  # ← sur 16 cœurs : 8 workers + 1 learner + overhead OS

NUM_EPISODES = 100_000

CONFIG = {
    "num_workers": NUM_WORKERS,
    "gru_hidden_size": 256,
    "encoder_hidden_sizes": [256, 128],
    "actor_hidden_sizes": [128, 64],
    "critic_hidden_sizes": [128, 64],
    "learning_rate": 2e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ppo_epochs": 2,        # 4 -> 2 : moitié moins de passes par update
    "ppo_clip": 0.15,
    "batch_size": 16,       # 256 -> 16 : update ~6s au lieu de ~200s (non vectorisé)
    "buffer_capacity": 300,
    "episode_length": EPISODE_STEPS,   # = EPISODE_DAYS * TRADES_PER_DAY
    "init_random_positions": True,     # portefeuille de depart aleatoire (anti-overfit)
    "entropy_coef": 0.07,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "tensorboard_log_dir": "runs/gru_ppo_multicore",
    "print_every": 10,  # affiche SPS toutes les N updates du learner
}

# ============================================================================
# Structures
# ============================================================================

Transition = namedtuple("Transition", ("observation", "action", "reward", "done", "value", "old_log_prob"))
Episode = namedtuple("Episode", ("transitions", "returns", "advantages"))


# ============================================================================
# Réseaux
# ============================================================================


class RunningNorm(nn.Module):
    """Online observation normalizer (Welford). Stats live in buffers so they are
    broadcast to the workers together with the network weights and frozen during
    the PPO update. See RL_model/train.py for the rationale."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_count = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10.0, 10.0)


class SetEncoder(nn.Module):
    """Permutation-invariant (DeepSets) encoder. See RL_model/train.py for the
    full rationale. Slices the flat obs into blocks, encodes the options/positions
    sets token-by-token with a shared MLP, masks padding, then mean+max pools.
    Per-block normalization is built in; option/position norms are shared across
    slots to preserve permutation invariance."""

    OPT_FEATS = 6
    POS_FEATS = 8

    def __init__(self, obs_dim, hidden_sizes, spot_history=SPOT_HISTORY,
                 max_portfolio=MAX_PORTFOLIO, token_dim=32):
        super().__init__()
        self.spot_history = spot_history
        self.max_portfolio = max_portfolio
        n_opt_vals = obs_dim - 2 - spot_history - max_portfolio * self.POS_FEATS - 2
        assert n_opt_vals > 0 and n_opt_vals % self.OPT_FEATS == 0, \
            f"obs_dim {obs_dim} incompatible with expected env layout"
        self.max_options = n_opt_vals // self.OPT_FEATS

        self.i_hist = 2
        self.i_opt = self.i_hist + spot_history
        self.i_pos = self.i_opt + self.max_options * self.OPT_FEATS
        self.i_pnl = self.i_pos + max_portfolio * self.POS_FEATS

        self.scalar_norm = RunningNorm(4)
        self.hist_norm = RunningNorm(spot_history)
        self.opt_norm = RunningNorm(self.OPT_FEATS)
        self.pos_norm = RunningNorm(self.POS_FEATS)

        def token_mlp(in_f):
            return nn.Sequential(nn.Linear(in_f, token_dim), nn.ReLU(),
                                 nn.Linear(token_dim, token_dim), nn.ReLU())

        self.opt_mlp = token_mlp(self.OPT_FEATS)
        self.pos_mlp = token_mlp(self.POS_FEATS)
        self.hist_mlp = nn.Sequential(nn.Linear(spot_history, 2 * token_dim), nn.ReLU())

        concat_dim = 4 + 2 * token_dim + 2 * token_dim + 2 * token_dim
        layers, prev = [], concat_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.proj = nn.Sequential(*layers)

    @staticmethod
    def _masked_pool(tokens, mask):
        mask_f = mask.unsqueeze(-1)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (tokens * mask_f).sum(dim=1) / count
        masked = tokens.masked_fill(mask_f == 0, torch.finfo(tokens.dtype).min)
        mx = masked.max(dim=1).values
        no_valid = (mask.sum(dim=1) == 0).unsqueeze(-1)
        mean = torch.where(no_valid, torch.zeros_like(mean), mean)
        mx = torch.where(no_valid, torch.zeros_like(mx), mx)
        return torch.cat([mean, mx], dim=-1)

    def forward(self, obs, update_stats=False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        scalars_raw = torch.cat([obs[:, 0:2], obs[:, self.i_pnl:self.i_pnl + 2]], dim=-1)
        hist_raw = obs[:, self.i_hist:self.i_opt]
        opt_raw = obs[:, self.i_opt:self.i_pos].reshape(-1, self.max_options, self.OPT_FEATS)
        pos_raw = obs[:, self.i_pos:self.i_pnl].reshape(-1, self.max_portfolio, self.POS_FEATS)
        opt_mask = (opt_raw.abs().sum(dim=-1) > 0).float()
        pos_mask = (pos_raw.abs().sum(dim=-1) > 0).float()

        if update_stats:
            self.scalar_norm.update(scalars_raw)
            self.hist_norm.update(hist_raw)
            valid_opt = opt_raw[opt_mask.bool()]
            valid_pos = pos_raw[pos_mask.bool()]
            if valid_opt.numel() > 0:
                self.opt_norm.update(valid_opt)
            if valid_pos.numel() > 0:
                self.pos_norm.update(valid_pos)

        scalars = self.scalar_norm(scalars_raw)
        hist = self.hist_mlp(self.hist_norm(hist_raw))
        opt_tok = self.opt_mlp(self.opt_norm(opt_raw))
        pos_tok = self.pos_mlp(self.pos_norm(pos_raw))
        opt_pool = self._masked_pool(opt_tok, opt_mask)
        pos_pool = self._masked_pool(pos_tok, pos_mask)
        return self.proj(torch.cat([scalars, hist, opt_pool, pos_pool], dim=-1))


class GRUCell(nn.Module):
    """GRU avec learned adaptive decay (α_t)."""

    def __init__(self, input_size, hidden_size, decay_hidden=[128, 64]):
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        layers, prev = [], hidden_size + input_size
        for h in decay_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.decay_net = nn.Sequential(*layers)
        self.decay_out = nn.Linear(prev, 1)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        alpha = torch.sigmoid(self.decay_out(self.decay_net(torch.cat([h, x], -1))))
        h_new = alpha * self.gru(x, h) + (1 - alpha) * h
        return h_new, alpha


class ActorNetwork(nn.Module):
    """Same hierarchical head as RL_model/train.py: Categorical action_type and
    call_or_put, Normal strike/quantity, Categorical maturity."""

    def __init__(self, gru_hidden, actor_hidden, max_maturity=252):
        super().__init__()
        self.max_maturity = max_maturity
        layers, prev = [], gru_hidden
        for h in actor_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.common = nn.Sequential(*layers)
        self.action_type_logits = nn.Linear(prev, 3)
        self.call_or_put = nn.Linear(prev, 2)
        self.str_mu   = nn.Linear(prev, 1)
        self.str_sig  = nn.Linear(prev, 1)
        self.mat_logits = nn.Linear(prev, max_maturity)
        self.qty_mu   = nn.Linear(prev, 1)
        self.qty_sig  = nn.Linear(prev, 1)

    def forward(self, h):
        x = self.common(h)
        at_dist = Categorical(logits=self.action_type_logits(x))
        at = at_dist.sample()
        lp_at = at_dist.log_prob(at)

        cop_dist = Categorical(logits=self.call_or_put(x))
        cop = cop_dist.sample()
        lp_cop = cop_dist.log_prob(cop)

        str_dist = Normal(self.str_mu(x), F.softplus(self.str_sig(x)) + 0.01)
        strike = str_dist.rsample()
        lp_str = str_dist.log_prob(strike).sum(-1)

        mat_dist = Categorical(logits=self.mat_logits(x))
        mat = mat_dist.sample()
        lp_mat = mat_dist.log_prob(mat)

        qty_dist = Normal(self.qty_mu(x), F.softplus(self.qty_sig(x)) + 0.01)
        qty = qty_dist.rsample()
        lp_qty = qty_dist.log_prob(qty).sum(-1)

        log_prob = lp_at + (at == 1).float() * (lp_cop + lp_str + lp_mat + lp_qty)
        return {
            "action_type": at,
            "call_or_put": cop,
            "strike": strike,
            "maturity": mat.float() + 1,
            "quantity_signed": qty,
            "log_prob": log_prob,
        }

    def evaluate(self, h, action):
        """Re-evaluate a stored action under the current policy (for PPO ratio)."""
        x = self.common(h)
        at_dist = Categorical(logits=self.action_type_logits(x))
        at = action["action_type"].long().reshape(-1)
        lp_at = at_dist.log_prob(at)

        cop_dist = Categorical(logits=self.call_or_put(x))
        lp_cop = cop_dist.log_prob(action["call_or_put"].long().reshape(-1))

        str_dist = Normal(self.str_mu(x), F.softplus(self.str_sig(x)) + 0.01)
        lp_str = str_dist.log_prob(action["strike"]).sum(-1)

        mat_dist = Categorical(logits=self.mat_logits(x))
        mat_idx = (action["maturity"].reshape(-1) - 1).long().clamp_(0, self.max_maturity - 1)
        lp_mat = mat_dist.log_prob(mat_idx)

        qty_dist = Normal(self.qty_mu(x), F.softplus(self.qty_sig(x)) + 0.01)
        lp_qty = qty_dist.log_prob(action["quantity_signed"]).sum(-1)

        is_trade = (at == 1).float()
        log_prob = lp_at + is_trade * (lp_cop + lp_str + lp_mat + lp_qty)
        entropy = at_dist.entropy() + is_trade * (
            cop_dist.entropy() + str_dist.entropy().sum(-1) + mat_dist.entropy() + qty_dist.entropy().sum(-1)
        )
        return log_prob, entropy


class CriticNetwork(nn.Module):
    def __init__(self, gru_hidden, critic_hidden):
        super().__init__()
        layers, prev = [], gru_hidden
        for h in critic_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net = nn.Sequential(*layers)
        self.value = nn.Linear(prev, 1)

    def forward(self, h):
        return self.value(self.net(h))


# ============================================================================
# Utils réseau
# ============================================================================


def build_networks(obs_dim, config):
    enc_dim = config["encoder_hidden_sizes"][-1]
    return {
        # SetEncoder owns observation normalization (broadcast with its weights).
        "encoder": SetEncoder(obs_dim, config["encoder_hidden_sizes"]),
        "gru":     GRUCell(enc_dim, config["gru_hidden_size"]),
        "actor":   ActorNetwork(config["gru_hidden_size"], config["actor_hidden_sizes"]),
        "critic":  CriticNetwork(config["gru_hidden_size"], config["critic_hidden_sizes"]),
    }


def load_weights(nets, state_dicts):
    for k, net in nets.items():
        net.load_state_dict(state_dicts[k])


def stored_action_to_tensors(action, device):
    """Convert a stored (numpy) action dict into batched tensors for evaluate()."""
    def to_tensor(value, shape):
        arr = np.atleast_1d(np.asarray(value, dtype=np.float32))
        return torch.as_tensor(arr, dtype=torch.float32, device=device).reshape(shape)

    return {
        "action_type": to_tensor(action["action_type"], (1,)),
        "call_or_put": to_tensor(action["call_or_put"], (1,)),
        "strike": to_tensor(action["strike"], (1, 1)),
        "maturity": to_tensor(action["maturity"], (1,)),
        "quantity_signed": to_tensor(action["quantity_signed"], (1, 1)),
    }


def _scalar(v, default=0.0):
    try:
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        return float(v.item() if hasattr(v, "item") else v)
    except Exception:
        return default


# ============================================================================
# GAE
# ============================================================================


def compute_gae(rewards, values, dones, gamma, lam):
    adv, gae = [], 0
    for t in reversed(range(len(rewards))):
        nv = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv.insert(0, gae)
    adv = np.array(adv)
    return adv, adv + np.array(values)


# ============================================================================
# Worker process
# ============================================================================


def worker_fn(worker_id, obs_dim, config, episode_queue, weight_queue, stop_event):
    """Collecte des épisodes et les envoie au learner."""
    torch.set_num_threads(1)
    device = torch.device("cpu")
    nets = build_networks(obs_dim, config)
    for net in nets.values():
        net.to(device).eval()

    env = OptionsEnv(init_random_positions=config.get("init_random_positions", False))
    ep_len = config["episode_length"]
    gamma, lam = config["gamma"], config["gae_lambda"]

    while not stop_event.is_set():
        # Sync poids si dispo
        while not weight_queue.empty():
            try:
                load_weights(nets, weight_queue.get_nowait())
            except Exception:
                pass

        obs, _ = env.reset()
        transitions, rewards, values, dones = [], [], [], []
        h = None

        for _ in range(ep_len):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                enc = nets["encoder"](obs_t)  # norm stats are owned by the learner
                h, _ = nets["gru"](enc, h)
                act = nets["actor"](h)
                val = nets["critic"](h)

            act_np = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                      for k, v in act.items()}
            obs2, reward, done, trunc, _ = env.step(act_np)

            rewards.append(reward)
            values.append(_scalar(val))
            dones.append(done or trunc)
            transitions.append(Transition(obs, act_np, reward, done or trunc,
                                          _scalar(val), _scalar(act_np["log_prob"])))
            obs = obs2
            if done or trunc:
                break

        adv, ret = compute_gae(rewards, values, dones, gamma, lam)
        episode_queue.put(Episode(transitions, ret.tolist(), adv.tolist()))


# ============================================================================
# Buffer
# ============================================================================


class EpisodeBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, ep):
        self.buf.append(ep)

    def sample(self, n):
        idx = np.random.choice(len(self.buf), size=min(n, len(self.buf)), replace=False)
        return [list(self.buf)[i] for i in idx]

    def __len__(self):
        return len(self.buf)


# ============================================================================
# PPO update
# ============================================================================


def ppo_update(nets, optimizer, batch, config, device):
    total_pl = total_vl = total_el = total_dr = n = 0
    params = [p for net in nets.values() for p in net.parameters()]
    eps = 1e-6
    for _ in range(config["ppo_epochs"]):
        for ep in batch:
            if len(ep.transitions) == 0:
                continue

            # BPTT over the whole episode: keep the hidden state attached and
            # back-propagate once per sequence so the GRU/decay/encoder learn.
            h = None
            pls, vls, ents, decs = [], [], [], []
            for t_idx, tr in enumerate(ep.transitions):
                obs   = torch.FloatTensor(tr.observation).unsqueeze(0).to(device)
                old_lp = torch.FloatTensor([float(tr.old_log_prob)]).to(device)
                adv   = torch.FloatTensor([ep.advantages[t_idx]]).to(device)
                ret   = torch.FloatTensor([[ep.returns[t_idx]]]).to(device)

                enc = nets["encoder"](obs, update_stats=False)  # frozen stats during update
                h, alpha = nets["gru"](enc, h)
                log_prob, entropy = nets["actor"].evaluate(h, stored_action_to_tensors(tr.action, device))
                val = nets["critic"](h)

                ratio = torch.exp(log_prob - old_lp)
                pls.append(-torch.min(ratio * adv,
                                      torch.clamp(ratio, 1 - config["ppo_clip"],
                                                  1 + config["ppo_clip"]) * adv).mean())
                vls.append(F.mse_loss(val, ret))
                ents.append(entropy.mean())
                decs.append((-(alpha * torch.log(alpha + eps)
                               + (1 - alpha) * torch.log(1 - alpha + eps))).mean())

            pl = torch.stack(pls).mean()
            vl = torch.stack(vls).mean()
            el = -config["entropy_coef"] * torch.stack(ents).mean()
            dr = 0.01 * torch.stack(decs).mean()

            optimizer.zero_grad()
            (pl + vl * config["vf_coef"] + el + dr).backward()
            torch.nn.utils.clip_grad_norm_(params, config["max_grad_norm"])
            optimizer.step()

            total_pl += pl.item(); total_vl += vl.item()
            total_el += el.item(); total_dr += dr.item()
            n += 1

    if n == 0:
        return {}
    return {"policy_loss": total_pl/n, "value_loss": total_vl/n,
            "entropy_loss": total_el/n, "decay_reg": total_dr/n}


# ============================================================================
# Learner (process principal)
# ============================================================================


def learner_loop(obs_dim, config, episode_queue, weight_queues, num_episodes):
    device = torch.device(config["device"])
    nets = build_networks(obs_dim, config)
    for net in nets.values():
        net.to(device).train()

    optimizer = optim.Adam(
        [p for net in nets.values() for p in net.parameters()],
        lr=config["learning_rate"])

    buffer = EpisodeBuffer(config["buffer_capacity"])
    updates = 0
    episodes_recv = 0
    total_steps = 0

    # TensorBoard
    writer = None
    log_dir = config.get("tensorboard_log_dir")
    if log_dir:
        from torch.utils.tensorboard import SummaryWriter
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard → {log_dir}")

    # broadcast poids initiaux
    init_w = {k: {pk: pv.cpu() for pk, pv in v.state_dict().items()}
              for k, v in nets.items()}
    for wq in weight_queues:
        wq.put(init_w)

    t_start = time.time()
    t_last = t_start
    steps_last = 0
    print_every = config.get("print_every", 10)

    # Live progress: one self-refreshing line (\r) like SB3, plus a committed
    # snapshot every `print_every` updates. We also track the running mean of
    # episode returns (= episode P&L, since reward is terminal only).
    ep_returns = deque(maxlen=100)
    last_return = float("nan")
    last_loss = float("nan")
    live_last_t = t_start
    live_last_steps = 0
    LIVE_EVERY = 0.5  # seconds between live-line refreshes

    def status_line():
        now = time.time()
        inst_sps = (total_steps - live_last_steps) / max(now - live_last_t, 1e-6)
        rew = float(np.mean(ep_returns)) if ep_returns else float("nan")
        return (
            f"ep {episodes_recv:>7,}/{num_episodes} | upd {updates:>5} | "
            f"steps {total_steps:>11,} | {inst_sps:6.0f} sps | "
            f"buf {len(buffer):>3}/{config['batch_size']} | "
            f"P&L last {last_return:+8.3f}  μ100 {rew:+8.3f} | loss {last_loss:+.4f}"
        )

    while episodes_recv < num_episodes:
        # Drainer la queue
        drained = 0
        while not episode_queue.empty() and drained < 32:
            try:
                ep = episode_queue.get_nowait()
                buffer.push(ep)
                episodes_recv += 1
                total_steps += len(ep.transitions)
                last_return = float(sum(t.reward for t in ep.transitions))  # episode P&L
                ep_returns.append(last_return)
                if writer is not None:
                    writer.add_scalar("rollout/ep_return", last_return, episodes_recv)
                drained += 1
                # Update normalization stats once per fresh episode (the learner's
                # encoder is the source of truth broadcast to the workers).
                if ep.transitions:
                    obs_batch = torch.as_tensor(
                        np.asarray([t.observation for t in ep.transitions], dtype=np.float32),
                        device=device,
                    )
                    with torch.no_grad():
                        nets["encoder"](obs_batch, update_stats=True)
            except Exception:
                break

        # Live status line, refreshed in place (throttled)
        if time.time() - live_last_t >= LIVE_EVERY:
            print("\r" + status_line().ljust(110), end="", flush=True)
            live_last_t = time.time()
            live_last_steps = total_steps

        if drained == 0:
            time.sleep(0.01)
            continue

        if len(buffer) < config["batch_size"]:
            continue

        batch = buffer.sample(config["batch_size"])
        metrics = ppo_update(nets, optimizer, batch, config, device)
        updates += 1
        if metrics:
            last_loss = metrics.get("policy_loss", last_loss)

        # Broadcast nouveaux poids
        new_w = {k: {pk: pv.cpu() for pk, pv in v.state_dict().items()}
                 for k, v in nets.items()}
        for wq in weight_queues:
            while not wq.empty():
                try: wq.get_nowait()
                except Exception: pass
            wq.put(new_w)

        # TensorBoard — chaque update
        if writer is not None and metrics:
            writer.add_scalar("loss/policy",  metrics["policy_loss"],  updates)
            writer.add_scalar("loss/value",   metrics["value_loss"],   updates)
            writer.add_scalar("loss/entropy", metrics["entropy_loss"], updates)
            writer.add_scalar("loss/decay",   metrics["decay_reg"],    updates)
            writer.add_scalar("train/episodes_received", episodes_recv, updates)
            writer.add_scalar("train/total_steps", total_steps, updates)
            if ep_returns:
                writer.add_scalar("rollout/ep_return_mean", float(np.mean(ep_returns)), updates)

        if updates % print_every == 0:
            now = time.time()
            sps_window = (total_steps - steps_last) / max(now - t_last, 1e-6)
            # Commit a permanent line (overwrites the live line, then newline)
            print("\r" + status_line().ljust(110))
            if writer is not None:
                writer.add_scalar("train/sps", sps_window, updates)
                writer.flush()
            t_last = now
            steps_last = total_steps

    if writer is not None:
        writer.close()
    print("\n" + f"Done. {updates} updates, {total_steps:,} steps, "
          f"{total_steps/(time.time()-t_start):.0f} sps avg")


# ============================================================================
# Point d'entrée
# ============================================================================


def train(config=None, num_episodes=NUM_EPISODES):
    run_config = (config or CONFIG).copy()
    # Fresh timestamped sub-directory per run so TensorBoard curves don't mix.
    base_log_dir = run_config.get("tensorboard_log_dir")
    if base_log_dir:
        run_config["tensorboard_log_dir"] = os.path.join(
            base_log_dir, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    num_workers = run_config["num_workers"]

    env_tmp = OptionsEnv()
    obs_dim = env_tmp.observation_space.shape[0]
    del env_tmp

    print(f"device: {run_config['device']} | workers: {num_workers} | obs_dim: {obs_dim}")

    mp.set_start_method("spawn", force=True)

    episode_queue = mp.Queue(maxsize=512)
    weight_queues = [mp.Queue(maxsize=2) for _ in range(num_workers)]
    stop_event    = mp.Event()

    procs = []
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_fn,
            args=(wid, obs_dim, run_config, episode_queue, weight_queues[wid], stop_event),
            daemon=True,
        )
        p.start()
        procs.append(p)

    print(f"{num_workers} workers started")

    try:
        learner_loop(obs_dim, run_config, episode_queue, weight_queues, num_episodes)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_event.set()
        for p in procs:
            p.join(timeout=5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--workers",  type=int, default=NUM_WORKERS)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = CONFIG.copy()
    cfg["num_workers"] = args.workers
    if args.no_tensorboard:
        cfg["tensorboard_log_dir"] = None
    if args.smoke_test:
        cfg.update({
            "encoder_hidden_sizes": [32, 16], "gru_hidden_size": 16,
            "actor_hidden_sizes": [16], "critic_hidden_sizes": [16],
            "episode_length": 2, "batch_size": 1, "buffer_capacity": 4,
            "ppo_epochs": 1, "num_workers": 2, "print_every": 1,
            "tensorboard_log_dir": None,
        })
    train(cfg, args.episodes)


if __name__ == "__main__":
    main()