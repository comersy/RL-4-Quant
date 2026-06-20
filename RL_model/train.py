"""
GRU-PPO Training Script

Trains a Proximal Policy Optimization agent with GRU memory to learn options trading strategies.

PPO handles mixed discrete/continuous actions naturally without the complexity of SAC.

The action space is hierarchical:
  - Primary action (actor outputs 3 logits): 0 = do nothing, 1 = trade, 2 = close positions
  - If trading (action = 1):
    - call_or_put: binary : 0 (call) or 1 (put)
    - strike: continuous (unbounded, relative to spot)
    - maturity: discrete [1, T_remaining]
    - quantity_signed: continuous (positive=long/buy, negative=short/sell)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from collections import deque, namedtuple
from datetime import datetime
import os
import sys
from pathlib import Path

# Allow running this file directly (`python RL_model/train.py`): put the project
# root on sys.path so `envs` / `data` are importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.env import EPISODE_STEPS, MAX_PORTFOLIO, SPOT_HISTORY, OptionsEnv

# ============================================================================
# Configuration
# ============================================================================

NUM_EPISODES = 100_000

CONFIG = {
    "gru_hidden_size": 256,
    "encoder_hidden_sizes": [256, 128],
    "actor_hidden_sizes": [128, 64],
    "critic_hidden_sizes": [128, 64],
    "learning_rate": 2e-4,
    "gamma": 0.99,  # discount factor
    "gae_lambda": 0.95,  # GAE lambda
    "ppo_epochs": 4,  # number of PPO epochs per update
    "ppo_clip": 0.15,  # PPO clipping coefficient
    "batch_size": 256,  # episode batch size for update
    "buffer_capacity": 300,  # number of episodes to store
    "episode_length": EPISODE_STEPS,  # = EPISODE_DAYS * TRADES_PER_DAY
    "entropy_coef": 0.07,  # entropy regularization
    "vf_coef": 0.5,  # value function loss coefficient
    "max_grad_norm": 0.5,  # gradient clipping
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "tensorboard_log_dir": "runs/gru_ppo",
}

print(f"Using device: {CONFIG['device']}")

# ============================================================================
# Experience Storage
# ============================================================================

Transition = namedtuple("Transition", ("observation", "action", "reward", "done", "value", "old_log_prob"))
Episode = namedtuple("Episode", ("transitions", "returns", "advantages"))


class EpisodeBuffer:
    """
    Stores complete episodes for GRU training.
    GRU agents need full sequences, not random single transitions.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, episode):
        """episode is a list of Transition named tuples"""
        self.buffer.append(episode)

    def sample(self, batch_size):
        """Sample random complete episodes"""
        episodes = list(self.buffer)
        sample_size = min(batch_size, len(episodes))
        indices = np.random.choice(len(episodes), size=sample_size, replace=False)
        return [episodes[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Networks: Encoder, GRU, Actor, Critic
# ============================================================================


class RunningNorm(nn.Module):
    """
    Online observation normalizer (Welford running mean/variance).

    Raw observations contain BTC spot/strikes (~1e4-1e5), P&L, IV, etc. Feeding
    those magnitudes straight into a Linear+ReLU encoder makes training unstable.
    Statistics are updated only during rollout (get_action) and frozen during the
    PPO update so the normalization is consistent across a minibatch. The stats
    live in buffers, so they are saved/loaded with the model and never receive
    gradients.
    """

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
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10.0, 10.0)


class SetEncoder(nn.Module):
    """
    Permutation-invariant (DeepSets) encoder for the flat OptionsEnv observation.

    The flat vector is sliced back into its semantic blocks:
        [spot, episode_day], spot_history(365),
        options(max_options x 6), positions(max_portfolio x 8),
        [realized_pnl, unrealized_pnl]

    Each *set* block (options, positions) is encoded token-by-token with a SHARED
    MLP, padded rows are masked out, then mean+max pooled into a fixed-size vector.
    This removes the dependence on option ordering and on the huge zero-padding,
    which a flat MLP handled very poorly.

    Observations are normalized online (RunningNorm) before encoding; the padding
    mask is derived from the RAW observation so normalization can't hide it.
    """

    OPT_FEATS = 6
    POS_FEATS = 8

    def __init__(self, obs_dim, hidden_sizes, spot_history=SPOT_HISTORY,
                 max_portfolio=MAX_PORTFOLIO, token_dim=32):
        super().__init__()
        self.spot_history = spot_history
        self.max_portfolio = max_portfolio
        self.token_dim = token_dim

        n_opt_vals = obs_dim - 2 - spot_history - max_portfolio * self.POS_FEATS - 2
        assert n_opt_vals > 0 and n_opt_vals % self.OPT_FEATS == 0, \
            f"obs_dim {obs_dim} incompatible with expected env layout"
        self.max_options = n_opt_vals // self.OPT_FEATS

        # Block offsets inside the flat vector
        self.i_hist = 2
        self.i_opt = self.i_hist + spot_history
        self.i_pos = self.i_opt + self.max_options * self.OPT_FEATS
        self.i_pnl = self.i_pos + max_portfolio * self.POS_FEATS

        # Per-block normalization. Crucially the option/position norms are
        # SHARED across slots (dim = features per token, not per-slot), otherwise
        # two identical options in different slots would be normalized differently
        # and the encoding would stop being permutation-invariant.
        self.scalar_norm = RunningNorm(4)                 # market(2) + pnl(2)
        self.hist_norm = RunningNorm(spot_history)
        self.opt_norm = RunningNorm(self.OPT_FEATS)       # shared over option slots
        self.pos_norm = RunningNorm(self.POS_FEATS)       # shared over position slots

        def token_mlp(in_features):
            return nn.Sequential(
                nn.Linear(in_features, token_dim), nn.ReLU(),
                nn.Linear(token_dim, token_dim), nn.ReLU(),
            )

        self.opt_mlp = token_mlp(self.OPT_FEATS)
        self.pos_mlp = token_mlp(self.POS_FEATS)
        self.hist_mlp = nn.Sequential(nn.Linear(spot_history, 2 * token_dim), nn.ReLU())

        # scalars(4) + hist(2d) + opt pool(2d) + pos pool(2d)
        concat_dim = 4 + 2 * token_dim + 2 * token_dim + 2 * token_dim
        layers, prev = [], concat_dim
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(prev, hidden_size), nn.ReLU()]
            prev = hidden_size
        self.proj = nn.Sequential(*layers)

    @staticmethod
    def _masked_pool(tokens, mask):
        """tokens: (B, N, d), mask: (B, N) in {0,1} → (B, 2d) [mean ‖ max]."""
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

        # Slice blocks from the RAW observation.
        scalars_raw = torch.cat([obs[:, 0:2], obs[:, self.i_pnl:self.i_pnl + 2]], dim=-1)
        hist_raw = obs[:, self.i_hist:self.i_opt]
        opt_raw = obs[:, self.i_opt:self.i_pos].reshape(-1, self.max_options, self.OPT_FEATS)
        pos_raw = obs[:, self.i_pos:self.i_pnl].reshape(-1, self.max_portfolio, self.POS_FEATS)

        # Padding masks (a padded row is all zeros; a real one has strike > 0).
        opt_mask = (opt_raw.abs().sum(dim=-1) > 0).float()
        pos_mask = (pos_raw.abs().sum(dim=-1) > 0).float()

        if update_stats:
            self.scalar_norm.update(scalars_raw)
            self.hist_norm.update(hist_raw)
            valid_opt = opt_raw[opt_mask.bool()]   # only real options feed the shared stats
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

        feat = torch.cat([scalars, hist, opt_pool, pos_pool], dim=-1)
        return self.proj(feat)


class GRUCell(nn.Module): #Il faudrait peut-être passer par un tenseur pour alpha plutot qu'un scalaire? 
    """
    GRU recurrent module with LEARNED ADAPTIVE DECAY.
    
    Instead of vanilla GRU with fixed gates, this implementation learns
    a context-dependent "forget factor" α_t that controls memory retention.
    
    The decay network decides: α_t = σ(MLP([h_{t-1}, s_t]))
    where:
      - α_t ≈ 1 → preserve memory (stable regime, maintain long-term context)
      - α_t ≈ 0 → reset memory (regime shift, market shock)
    
    This allows the agent to adaptively "forget" outdated market states
    when needed, making memory context-aware rather than blind replay.
    """

    def __init__(self, input_size, hidden_size, decay_hidden_sizes=[128, 64]):
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        
        # Decay network: learns how much to remember
        # Input: [h_{t-1}, s_t] (previous hidden state + current observation)
        # Output: α_t ∈ [0, 1] (forget factor)
        decay_layers = []
        prev_size = hidden_size + input_size
        for decay_hidden_size in decay_hidden_sizes:
            decay_layers.append(nn.Linear(prev_size, decay_hidden_size))
            decay_layers.append(nn.ReLU())
            prev_size = decay_hidden_size
        
        self.decay_net = nn.Sequential(*decay_layers)
        self.decay_output = nn.Linear(prev_size, 1)  # Single scalar α_t

    def forward(self, x, h=None):
        """
        Forward pass through GRU with learned decay.
        
        Args:
            x: (batch_size, input_size) - current observation at timestep t
            h: (batch_size, hidden_size) - previous hidden state h_{t-1}
        
        Returns:
            h_new: (batch_size, hidden_size) - updated hidden state
            alpha: (batch_size, 1) - learned decay factor (for logging/analysis)
        """
        if h is None:
            batch_size = x.shape[0]
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Compute adaptive decay factor α_t = σ(DecayNet([h_{t-1}, s_t]))
        decay_input = torch.cat([h, x], dim=-1)
        decay_logit = self.decay_output(self.decay_net(decay_input))
        alpha = torch.sigmoid(decay_logit)  # α_t ∈ [0, 1]

        # Standard GRU update: h_t = GRU(x_t, h_{t-1})
        h_raw = self.gru(x, h)

        # Apply learned decay: blend old memory with new state
        # h_new = α_t * h_raw + (1 - α_t) * h_{t-1}
        # When α_t=1: full new state (reset memory)
        # When α_t=0: keep old state (preserve memory)
        h_new = alpha * h_raw + (1 - alpha) * h

        return h_new, alpha


class ActorNetwork(nn.Module):
    """
    Actor outputs all action distribution parameters:
    - action_type: 3 logits (do nothing, trade, close positions)
    - call_or_put: binary [0, 1]
    - strike: continuous (unbounded)
    - maturity: logits for Categorical [1, T]
    - quantity_signed: continuous (unbounded, negative=short/sell, positive=long/buy)
    """

    def __init__(self, gru_hidden_size, actor_hidden_sizes, max_maturity=252):
        super().__init__()
        self.max_maturity = max_maturity

        prev_size = gru_hidden_size
        layers = []
        for hidden_size in actor_hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        self.common = nn.Sequential(*layers)

        # Action type: do nothing, trade, close
        self.action_type_logits = nn.Linear(prev_size, 3)

        # Call or put
        self.call_or_put = nn.Linear(prev_size, 2) 
        # Strike
        self.strike_mu = nn.Linear(prev_size, 1)
        self.strike_sigma = nn.Linear(prev_size, 1)

        # Maturity
        self.maturity_logits = nn.Linear(prev_size, max_maturity)

        # Quantity (signed: positive=long/buy, negative=short/sell)
        self.quantity_mu = nn.Linear(prev_size, 1)
        self.quantity_sigma = nn.Linear(prev_size, 1)

    def forward(self, h_gru):
        """
        h_gru: (batch_size, gru_hidden_size)

        Returns: dict with action samples and their log_probs
        """
        x = self.common(h_gru)

        # Action type (Categorical)
        action_type_logits = self.action_type_logits(x)
        action_type_dist = Categorical(logits=action_type_logits)
        action_type = action_type_dist.sample()
        log_prob_action_type = action_type_dist.log_prob(action_type)
        call_or_put_logits = self.call_or_put(x)
        call_or_put_dist = Categorical(logits=call_or_put_logits)
        call_or_put = call_or_put_dist.sample()
        log_prob_call_or_put = call_or_put_dist.log_prob(call_or_put)

        # Continuous actions (Normal distributions)
        strike_mu = self.strike_mu(x)
        strike_sigma = F.softplus(self.strike_sigma(x)) + 0.01
        strike_dist = Normal(strike_mu, strike_sigma)
        strike = strike_dist.rsample()
        log_prob_strike = strike_dist.log_prob(strike).sum(dim=-1)

        maturity_logits = self.maturity_logits(x)
        maturity_dist = Categorical(logits=maturity_logits)
        maturity = maturity_dist.sample()
        log_prob_maturity = maturity_dist.log_prob(maturity)

        # Quantity signed (positive=long, negative=short)
        quantity_mu = self.quantity_mu(x)
        quantity_sigma = F.softplus(self.quantity_sigma(x)) + 0.01
        quantity_dist = Normal(quantity_mu, quantity_sigma)
        quantity = quantity_dist.rsample()
        log_prob_quantity = quantity_dist.log_prob(quantity).sum(dim=-1)

        # Total log prob (only conditioned on action_type)
        log_prob = log_prob_action_type + (
            (action_type == 1).float() * (
                log_prob_call_or_put + log_prob_strike + log_prob_maturity +
                log_prob_quantity
            )
        )

        return {
            "action_type": action_type,
            "call_or_put": call_or_put,
            "strike": strike,
            "maturity": maturity.float() + 1,
            "quantity_signed": quantity,
            "log_prob": log_prob,
        }

    def evaluate(self, h_gru, action):
        """
        Re-evaluate a *stored* action under the current policy.

        This is what PPO actually needs: log_prob(stored_action | current_policy)
        so that ratio = exp(new_log_prob - old_log_prob) compares the same action.
        The forward() path re-samples a fresh action, which would make the ratio
        meaningless. Returns (log_prob, entropy), both shaped (batch,).
        """
        x = self.common(h_gru)

        action_type_dist = Categorical(logits=self.action_type_logits(x))
        action_type = action["action_type"].long().reshape(-1)
        log_prob_action_type = action_type_dist.log_prob(action_type)

        call_or_put_dist = Categorical(logits=self.call_or_put(x))
        log_prob_call_or_put = call_or_put_dist.log_prob(action["call_or_put"].long().reshape(-1))

        strike_mu = self.strike_mu(x)
        strike_sigma = F.softplus(self.strike_sigma(x)) + 0.01
        strike_dist = Normal(strike_mu, strike_sigma)
        log_prob_strike = strike_dist.log_prob(action["strike"]).sum(dim=-1)

        maturity_dist = Categorical(logits=self.maturity_logits(x))
        maturity_idx = (action["maturity"].reshape(-1) - 1).long().clamp_(0, self.max_maturity - 1)
        log_prob_maturity = maturity_dist.log_prob(maturity_idx)

        quantity_mu = self.quantity_mu(x)
        quantity_sigma = F.softplus(self.quantity_sigma(x)) + 0.01
        quantity_dist = Normal(quantity_mu, quantity_sigma)
        log_prob_quantity = quantity_dist.log_prob(action["quantity_signed"]).sum(dim=-1)

        is_trade = (action_type == 1).float()
        log_prob = log_prob_action_type + is_trade * (
            log_prob_call_or_put + log_prob_strike + log_prob_maturity + log_prob_quantity
        )
        entropy = action_type_dist.entropy() + is_trade * (
            call_or_put_dist.entropy()
            + strike_dist.entropy().sum(dim=-1)
            + maturity_dist.entropy()
            + quantity_dist.entropy().sum(dim=-1)
        )
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Value network: V(s) estimates state value for advantage calculation
    """

    def __init__(self, gru_hidden_size, critic_hidden_sizes):
        super().__init__()

        prev_size = gru_hidden_size
        layers = []
        for hidden_size in critic_hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        self.net = nn.Sequential(*layers)
        self.value = nn.Linear(prev_size, 1)

    def forward(self, h_gru):
        """
        h_gru: (batch_size, gru_hidden_size)

        Returns: value (batch_size, 1)
        """
        x = self.net(h_gru)
        return self.value(x)


# ============================================================================
# GRU-PPO Agent
# ============================================================================


class GRUPPOAgent: # L'agent ne connait pas le prix a laquelle il achete les options...
    """
    GRU-PPO Agent with Learned Adaptive Decay Memory.
    
    Architecture:
      1. Encoder: project observations to latent space
      2. GRU (with learned decay): accumulate memory with context-aware forgetting
      3. Actor: output hierarchical actions
      4. Critic: estimate value for advantage computation
    
    The key innovation is the decay network inside GRUCell: instead of 
    vanilla recurrence, α_t = σ(DecayNet([h_{t-1}, s_t])) allows the 
    agent to learn WHEN to forget vs. when to remember.
    """
    
    def __init__(self, obs_dim, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.obs_dim = obs_dim

        # Networks
        # SetEncoder owns the observation normalization (RunningNorm) and turns
        # the flat padded vector into a permutation-invariant latent.
        self.encoder = SetEncoder(obs_dim, config["encoder_hidden_sizes"]).to(self.device)
        
        # GRU with learned decay (replaces vanilla GRU)
        encoded_dim = config["encoder_hidden_sizes"][-1]
        self.gru = GRUCell(encoded_dim, config["gru_hidden_size"]).to(self.device)
        
        self.actor = ActorNetwork(config["gru_hidden_size"], config["actor_hidden_sizes"]).to(self.device)
        self.critic = CriticNetwork(config["gru_hidden_size"], config["critic_hidden_sizes"]).to(self.device)

        # Single optimizer for all networks (including decay network inside GRU)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.gru.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters()),
            lr=config["learning_rate"]
        )

    def get_action(self, obs, h_gru=None):
        """
        Get action from policy for a single timestep.
        
        Args:
            obs: numpy array (obs_dim,) or torch tensor
            h_gru: previous GRU hidden state or None (initializes to zero)
        
        Returns:
            action_dict: dict with action samples [numpy]
            h_gru_new: updated GRU hidden state (for next timestep)
            value: scalar value estimate
        """
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).to(self.device)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            # Encode observation (updates running normalization stats on fresh data)
            encoded = self.encoder(obs, update_stats=True)
            
            # GRU forward with learned decay
            # Returns: h_gru_new, alpha (alpha is just for monitoring)
            h_gru_new, alpha = self.gru(encoded, h_gru)

            # Actor samples actions from policy
            action_dict = self.actor(h_gru_new)
            
            # Critic estimates current state value
            value = self.critic(h_gru_new)

            # Convert to numpy for environment interaction
            action_dict_np = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                              for k, v in action_dict.items()}

            return action_dict_np, h_gru_new, value.item()

    def _stored_action_to_tensors(self, action):
        """Convert a stored (numpy) action dict into batched tensors for evaluate()."""
        def to_tensor(value, shape):
            arr = np.atleast_1d(np.asarray(value, dtype=np.float32))
            return torch.as_tensor(arr, dtype=torch.float32, device=self.device).reshape(shape)

        return {
            "action_type": to_tensor(action["action_type"], (1,)),
            "call_or_put": to_tensor(action["call_or_put"], (1,)),
            "strike": to_tensor(action["strike"], (1, 1)),
            "maturity": to_tensor(action["maturity"], (1,)),
            "quantity_signed": to_tensor(action["quantity_signed"], (1, 1)),
        }

    def train_step(self, episodes_batch, config):
        """
        PPO training step on a batch of complete episodes with GAE.
        
        The learned decay α_t is trained end-to-end:
        when the policy gradient improves, so does the decay network.
        
        Args:
            episodes_batch: list of Episode objects (transitions, returns, advantages)
            config: configuration dict
        
        Returns:
            metrics: dict with loss values for monitoring
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_decay_reg = 0.0  # optional: regularize decay to prevent collapse
        num_updates = 0

        params = (
            list(self.encoder.parameters())
            + list(self.gru.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )

        # PPO epochs: multiple passes over the same batch
        for epoch in range(config["ppo_epochs"]):
            for episode in episodes_batch:
                if len(episode.transitions) == 0:
                    continue

                # One full pass over the episode WITHOUT detaching the hidden
                # state, so gradients flow back through time (BPTT) into the GRU,
                # the decay network and the encoder. Per-step losses are collected
                # and back-propagated once at the end of the sequence.
                h_gru = None
                policy_losses, value_losses, entropy_terms, decay_terms = [], [], [], []

                for t_idx, transition in enumerate(episode.transitions):
                    obs = torch.FloatTensor(transition.observation).unsqueeze(0).to(self.device)
                    old_log_prob = torch.FloatTensor([_as_scalar(transition.old_log_prob)]).to(self.device)
                    advantage = torch.FloatTensor([episode.advantages[t_idx]]).to(self.device)
                    ret = torch.FloatTensor([[episode.returns[t_idx]]]).to(self.device)

                    # 1. Encode (frozen norm stats during the update) → 2. GRU (attached for BPTT)
                    encoded = self.encoder(obs, update_stats=False)
                    h_gru, alpha_t = self.gru(encoded, h_gru)

                    # 3. Re-evaluate the STORED action under the current policy
                    action = self._stored_action_to_tensors(transition.action)
                    log_prob, entropy = self.actor.evaluate(h_gru, action)

                    # 4. Critic estimates value
                    value = self.critic(h_gru)

                    # ===== Policy Loss (PPO clipping) =====
                    ratio = torch.exp(log_prob - old_log_prob)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - config["ppo_clip"],
                                        1.0 + config["ppo_clip"]) * advantage
                    policy_losses.append(-torch.min(surr1, surr2).mean())

                    # ===== Value Function Loss =====
                    value_losses.append(F.mse_loss(value, ret))

                    # ===== Entropy (true policy entropy, encourages exploration) =====
                    entropy_terms.append(entropy.mean())

                    # ===== Decay Regularization (prevent α collapse to 0 or 1) =====
                    eps = 1e-6
                    decay_terms.append(
                        -(alpha_t * torch.log(alpha_t + eps)
                          + (1 - alpha_t) * torch.log(1 - alpha_t + eps)).mean()
                    )

                # ===== Aggregate over the sequence and back-propagate once =====
                policy_loss = torch.stack(policy_losses).mean()
                value_loss = torch.stack(value_losses).mean()
                entropy_loss = -config["entropy_coef"] * torch.stack(entropy_terms).mean()
                decay_reg = 0.01 * torch.stack(decay_terms).mean()

                total_loss = (policy_loss
                              + value_loss * config["vf_coef"]
                              + entropy_loss
                              + decay_reg)

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, config["max_grad_norm"])
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_decay_reg += decay_reg.item()
                num_updates += 1

        # Return averaged metrics for monitoring
        if num_updates > 0:
            return {
                "policy_loss": total_policy_loss / num_updates,
                "value_loss": total_value_loss / num_updates,
                "entropy_loss": total_entropy_loss / num_updates,
                "decay_reg": total_decay_reg / num_updates,
            }
        return {}

    def save(self, path):
        """Save all network weights."""
        torch.save({
            "encoder": self.encoder.state_dict(),  # includes RunningNorm buffers
            "gru": self.gru.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        """Load all network weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.gru.load_state_dict(checkpoint["gru"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])


# ============================================================================
# Training Loop
# ============================================================================


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """
    Compute Generalized Advantage Estimation (GAE)

    rewards: list of rewards
    values: list of value estimates
    dones: list of done flags
    gamma: discount factor
    gae_lambda: GAE lambda parameter

    Returns: advantages (list), returns (list)
    """
    advantages = []
    gae = 0
    next_value = 0

    for t in reversed(range(len(rewards))):
        done = dones[t]
        if t == len(rewards) - 1:
            next_value = 0  # Bootstrap with 0 at episode end
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - done) - values[t]
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        advantages.insert(0, gae)

    advantages = np.array(advantages)
    returns = advantages + np.array(values)

    return advantages, returns


def _make_tensorboard_writer(config):
    """Create a TensorBoard writer when tensorboard_log_dir is configured."""
    log_dir = config.get("tensorboard_log_dir")
    if not log_dir:
        return None

    from torch.utils.tensorboard import SummaryWriter

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _as_scalar(value, default=0.0):
    """Convert tensors/arrays/scalars into a Python float for logging."""
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return default


def _action_type(action_dict):
    return int(_as_scalar(np.atleast_1d(action_dict.get("action_type", 0))[0]))


def _portfolio_metrics(env):
    realized_pnl = _as_scalar(getattr(env, "realized_pnl", 0.0))
    unrealized_pnl = _as_scalar(getattr(env, "unrealized_pnl", 0.0))
    portfolio = getattr(env, "portfolio", [])
    num_positions = len(portfolio) if portfolio is not None else 0

    num_long = 0
    num_short = 0
    for position in portfolio or []:
        if position.get("is_short", False):
            num_short += 1
        else:
            num_long += 1

    return {
        "pnl/realized": realized_pnl,
        "pnl/unrealized": unrealized_pnl,
        "pnl/total": realized_pnl + unrealized_pnl,
        "portfolio/open_positions": num_positions,
        "portfolio/long_positions": num_long,
        "portfolio/short_positions": num_short,
    }


def _log_scalars(writer, metrics, step):
    if writer is None:
        return
    for name, value in metrics.items():
        writer.add_scalar(name, value, step)


def train(env, agent, config, num_episodes=NUM_EPISODES, writer=None):
    """
    Main PPO training loop

    env: RL environment
    agent: GRUPPOAgent
    config: config dict
    num_episodes: number of episodes to train
    """
    buffer = EpisodeBuffer(config["buffer_capacity"])
    owns_writer = writer is None
    writer = writer or _make_tensorboard_writer(config)

    try:
        for episode_num in range(num_episodes):
            obs, info = env.reset()
            episode_transitions = []
            episode_rewards = []
            episode_values = []
            episode_dones = []

            h_gru = None
            episode_reward = 0.0
            action_counts = {"hold": 0, "trade": 0, "close": 0}

            for step in range(config["episode_length"]):
                # Get action and value from agent
                action_dict, h_gru, value = agent.get_action(obs, h_gru)
                action_type = _action_type(action_dict)
                if action_type == 1:
                    action_counts["trade"] += 1
                elif action_type == 2:
                    action_counts["close"] += 1
                else:
                    action_counts["hold"] += 1

                # Step environment
                obs_next, reward, done, truncated, info = env.step(action_dict)

                episode_reward += reward
                episode_rewards.append(reward)
                episode_values.append(value)
                episode_dones.append(done or truncated)

                # Store transition with old log prob
                episode_transitions.append(Transition(
                    obs,
                    action_dict,
                    reward,
                    done or truncated,
                    value,
                    action_dict["log_prob"]  # store log_prob as numpy
                ))

                obs = obs_next
                env_step = episode_num * config["episode_length"] + step + 1

                if done or truncated:
                    break

            # Compute advantages and returns using GAE
            advantages, returns = compute_gae(
                episode_rewards,
                episode_values,
                episode_dones,
                config["gamma"],
                config["gae_lambda"]
            )

            # Create episode with advantages and returns
            episode = Episode(episode_transitions, returns.tolist(), advantages.tolist())
            buffer.push(episode)

            global_step = episode_num + 1
            print(
                f"Episode {global_step}/{num_episodes} | "
                f"Steps: {len(episode_transitions)} | Reward: {episode_reward:.2f}"
            )

            episode_metrics = _portfolio_metrics(env)
            _log_scalars(writer, episode_metrics, global_step)

            if writer is not None:
                final_metrics = _portfolio_metrics(env)
                writer.add_scalar("episode/reward", episode_reward, global_step)
                writer.add_scalar("episode/steps", len(episode_transitions), global_step)

                writer.add_scalar("episode/pnl_total", final_metrics["pnl/total"], global_step)
                writer.add_scalar("episode/pnl_realized", final_metrics["pnl/realized"], global_step)
                writer.add_scalar("episode/pnl_unrealized", final_metrics["pnl/unrealized"], global_step)

                writer.add_scalar("episode/num_positions", final_metrics["portfolio/open_positions"], global_step)
                writer.add_scalar("episode/long_positions", final_metrics["portfolio/long_positions"], global_step)
                writer.add_scalar("episode/short_positions", final_metrics["portfolio/short_positions"], global_step)

                writer.add_scalar("episode/hold_actions", action_counts["hold"], global_step)
                writer.add_scalar("episode/trade_actions", action_counts["trade"], global_step)
                writer.add_scalar("episode/close_actions", action_counts["close"], global_step)
                writer.add_scalar("buffer/size", len(buffer), global_step)

            # Train on batch of episodes if buffer is full
            if len(buffer) >= config["batch_size"]:
                episodes_batch = buffer.sample(config["batch_size"])
                metrics = agent.train_step(episodes_batch, config)
                if metrics:
                    print(f"  Training metrics: {metrics}")
                    if writer is not None:
                        for name, value in metrics.items():
                            writer.add_scalar(f"loss/{name}", value, global_step)

            if writer is not None:
                writer.flush()
    finally:
        if owns_writer and writer is not None:
            writer.close()

    return agent, buffer


def build_env_and_agent(config=None, init_random_positions=False):
    """Build the default project environment and matching GRU-PPO agent."""


    run_config = (config or CONFIG).copy()
    # episode_length already reflects EPISODE_STEPS (or a smoke-test override).

    env = OptionsEnv(init_random_positions=init_random_positions)
    agent = GRUPPOAgent(obs_dim=env.observation_space.shape[0], config=run_config)
    return env, agent, run_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train GRU-PPO on OptionsEnv.")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes to train.")
    parser.add_argument(
        "--log-dir",
        default=CONFIG["tensorboard_log_dir"],
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional path for saving the trained agent checkpoint.",
    )
    parser.add_argument(
        "--init-random-positions",
        action="store_true",
        help="Start episodes with random positions. Useful for evaluation/stress tests.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny model and two environment steps to verify wiring quickly.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_config = CONFIG.copy()
    run_config["tensorboard_log_dir"] = None if args.no_tensorboard else args.log_dir

    if args.smoke_test:
        run_config.update(
            {
                "encoder_hidden_sizes": [32, 16],
                "gru_hidden_size": 16,
                "actor_hidden_sizes": [16],
                "critic_hidden_sizes": [16],
                "episode_length": 2,
                "batch_size": 1,
                "buffer_capacity": 2,
                "ppo_epochs": 1,
                "tensorboard_log_dir": None,
            }
        )

    # Fresh timestamped sub-directory per run so TensorBoard curves don't mix.
    if run_config.get("tensorboard_log_dir"):
        run_config["tensorboard_log_dir"] = os.path.join(
            run_config["tensorboard_log_dir"], datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    env, agent, run_config = build_env_and_agent(
        config=run_config,
        init_random_positions=args.init_random_positions,
    )

    print("GRU-PPO Training Script Initialized")
    print(f"Observation dim: {env.observation_space.shape[0]}")
    print(f"Episode length: {run_config['episode_length']}")
    print(f"TensorBoard log dir: {run_config['tensorboard_log_dir']}")
    print(f"Training episodes: {args.episodes}")

    agent, buffer = train(env, agent, run_config, num_episodes=args.episodes)

    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        agent.save(args.save_path)
        print(f"Saved checkpoint to {args.save_path}")

    return agent, buffer


if __name__ == "__main__":
    main()
