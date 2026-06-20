"""
GRU-PPO Multi-Core Training Script

Architecture parallèle style IMPALA :
  - NUM_WORKERS workers collectent des épisodes en parallèle
  - 1 learner central fait les gradient updates
  - Le learner affiche les SPS globaux toutes les N updates
"""

import argparse
import time
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

from envs.env import EPISODE_DAYS, OptionsEnv

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
    "ppo_epochs": 4,
    "ppo_clip": 0.15,
    "batch_size": 256,
    "buffer_capacity": 300,
    "episode_length": 150,
    "entropy_coef": 0.07,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "tensorboard_log_dir": "runs/gru_ppo_multicore",
    "print_every": 50,  # affiche SPS toutes les N updates du learner
}

# ============================================================================
# Structures
# ============================================================================

Transition = namedtuple("Transition", ("observation", "action", "reward", "done", "value", "old_log_prob"))
Episode = namedtuple("Episode", ("transitions", "returns", "advantages"))


# ============================================================================
# Réseaux
# ============================================================================


class FCEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        layers, prev = [], obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
    def __init__(self, gru_hidden, actor_hidden, max_maturity=252):
        super().__init__()
        self.max_maturity = max_maturity
        layers, prev = [], gru_hidden
        for h in actor_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.common = nn.Sequential(*layers)
        self.action_type_logits = nn.Linear(prev, 3)
        self.cop_mu   = nn.Linear(prev, 1)
        self.cop_sig  = nn.Linear(prev, 1)
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

        def cont(mu_l, sig_l, tanh=False):
            mu = torch.tanh(mu_l(x)) if tanh else mu_l(x)
            sig = F.softplus(sig_l(x)) + 0.01
            dist = Normal(mu, sig)
            s = torch.tanh(dist.rsample()) if tanh else dist.rsample()
            return s, dist.log_prob(s).sum(-1)

        cop, lp_cop = cont(self.cop_mu, self.cop_sig, tanh=True)
        strike, lp_str = cont(self.str_mu, self.str_sig)
        mat_dist = Categorical(logits=self.mat_logits(x))
        mat = mat_dist.sample()
        lp_mat = mat_dist.log_prob(mat)
        qty, lp_qty = cont(self.qty_mu, self.qty_sig)

        log_prob = lp_at + (at == 1).float() * (lp_cop + lp_str + lp_mat + lp_qty)
        return {
            "action_type": at,
            "call_or_put": cop,
            "strike": strike,
            "maturity": mat.float() + 1,
            "quantity_signed": qty,
            "log_prob": log_prob,
        }


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
        "encoder": FCEncoder(obs_dim, config["encoder_hidden_sizes"]),
        "gru":     GRUCell(enc_dim, config["gru_hidden_size"]),
        "actor":   ActorNetwork(config["gru_hidden_size"], config["actor_hidden_sizes"]),
        "critic":  CriticNetwork(config["gru_hidden_size"], config["critic_hidden_sizes"]),
    }


def load_weights(nets, state_dicts):
    for k, net in nets.items():
        net.load_state_dict(state_dicts[k])


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

    env = OptionsEnv()
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
                enc = nets["encoder"](obs_t)
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
    for _ in range(config["ppo_epochs"]):
        for ep in batch:
            for t_idx, tr in enumerate(ep.transitions):
                obs   = torch.FloatTensor(tr.observation).unsqueeze(0).to(device)
                old_lp = torch.FloatTensor([float(tr.old_log_prob)]).to(device)
                adv   = torch.FloatTensor([ep.advantages[t_idx]]).to(device)
                ret   = torch.FloatTensor([[ep.returns[t_idx]]]).to(device)

                enc = nets["encoder"](obs)
                h, alpha = nets["gru"](enc)
                act = nets["actor"](h)
                val = nets["critic"](h)

                ratio = torch.exp(act["log_prob"] - old_lp)
                pl = -torch.min(ratio * adv,
                                torch.clamp(ratio, 1 - config["ppo_clip"],
                                            1 + config["ppo_clip"]) * adv).mean()
                vl = F.mse_loss(val, ret)
                el = -config["entropy_coef"] * act["log_prob"].mean()
                eps = 1e-6
                dr = 0.01 * (-(alpha * torch.log(alpha + eps)
                               + (1 - alpha) * torch.log(1 - alpha + eps))).mean()

                optimizer.zero_grad()
                (pl + vl * config["vf_coef"] + el + dr).backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for net in nets.values() for p in net.parameters()],
                    config["max_grad_norm"])
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
    print_every = config.get("print_every", 50)

    while episodes_recv < num_episodes:
        # Drainer la queue
        drained = 0
        while not episode_queue.empty() and drained < 32:
            try:
                ep = episode_queue.get_nowait()
                buffer.push(ep)
                episodes_recv += 1
                total_steps += len(ep.transitions)
                drained += 1
            except Exception:
                break

        if drained == 0:
            time.sleep(0.01)
            continue

        if len(buffer) < config["batch_size"]:
            continue

        batch = buffer.sample(config["batch_size"])
        metrics = ppo_update(nets, optimizer, batch, config, device)
        updates += 1

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

        if updates % print_every == 0:
            now = time.time()
            elapsed_total = now - t_start
            elapsed_window = now - t_last
            sps_window = (total_steps - steps_last) / max(elapsed_window, 1e-6)
            sps_total  = total_steps / max(elapsed_total, 1e-6)
            print(
                f"update {updates:6d} | "
                f"ep {episodes_recv:7d}/{num_episodes} | "
                f"steps {total_steps:9,d} | "
                f"sps {sps_window:6.0f} (avg {sps_total:6.0f}) | "
                f"loss {metrics.get('policy_loss', 0):.4f}"
            )
            if writer is not None:
                writer.add_scalar("train/sps", sps_window, updates)
                writer.flush()
            t_last = now
            steps_last = total_steps

    if writer is not None:
        writer.close()
    print(f"Done. {updates} updates, {total_steps:,} steps, "
          f"{total_steps/(time.time()-t_start):.0f} sps avg")


# ============================================================================
# Point d'entrée
# ============================================================================


def train(config=None, num_episodes=NUM_EPISODES):
    run_config = (config or CONFIG).copy()
    run_config["episode_length"] = min(run_config["episode_length"], EPISODE_DAYS)
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