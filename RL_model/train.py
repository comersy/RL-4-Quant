"""
GRU-PPO Training Script

Trains a Proximal Policy Optimization agent with GRU memory to learn options trading strategies.

PPO handles mixed discrete/continuous actions naturally without the complexity of SAC.

The action space is hierarchical:
  - Primary action (actor outputs 3 logits): 0 = do nothing, 1 = trade, 2 = close positions
  - If trading (action = 1):
    - call_or_put: continuous [0-1] → rounded to 0 (call) or 1 (put)
    - strike: continuous (unbounded, relative to spot)
    - maturity: discrete [1, T_remaining]
    - quantity_signed: continuous (positive=long/buy, negative=short/sell)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from collections import deque, namedtuple
import os
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "gru_hidden_size": 256,
    "encoder_hidden_sizes": [256, 128],
    "actor_hidden_sizes": [128, 64],
    "critic_hidden_sizes": [128, 64],
    "learning_rate": 3e-4,
    "gamma": 0.99,  # discount factor
    "gae_lambda": 0.95,  # GAE lambda
    "ppo_epochs": 4,  # number of PPO epochs per update
    "ppo_clip": 0.2,  # PPO clipping coefficient
    "batch_size": 32,  # episode batch size for update
    "buffer_capacity": 100,  # number of episodes to store
    "episode_length": 252,
    "entropy_coef": 0.01,  # entropy regularization
    "vf_coef": 0.5,  # value function loss coefficient
    "max_grad_norm": 0.5,  # gradient clipping
    "device": "cuda" if torch.cuda.is_available() else "cpu",
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
        batch = np.random.choice(episodes, size=min(batch_size, len(episodes)), replace=False)
        return batch

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Networks: Encoder, GRU, Actor, Critic
# ============================================================================


class FCEncoder(nn.Module):
    """Fully connected encoder: raw observation → latent"""

    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


class GRUCell(nn.Module):
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
    - call_or_put: continuous [0-1]
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
        self.call_or_put_mu = nn.Linear(prev_size, 1)
        self.call_or_put_sigma = nn.Linear(prev_size, 1)

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

        # Continuous actions (Normal distributions)
        call_or_put_mu = torch.tanh(self.call_or_put_mu(x))
        call_or_put_sigma = F.softplus(self.call_or_put_sigma(x)) + 0.01
        call_or_put_dist = Normal(call_or_put_mu, call_or_put_sigma)
        call_or_put = torch.tanh(call_or_put_dist.rsample())
        log_prob_call_or_put = call_or_put_dist.log_prob(call_or_put).sum(dim=-1)

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


class GRUPPOAgent:
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
        self.encoder = FCEncoder(obs_dim, config["encoder_hidden_sizes"]).to(self.device)
        
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

            # Encode observation
            encoded = self.encoder(obs)
            
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

        # PPO epochs: multiple passes over the same batch
        for epoch in range(config["ppo_epochs"]):
            for episode in episodes_batch:
                if len(episode.transitions) == 0:
                    continue

                for t_idx, transition in enumerate(episode.transitions):
                    obs = torch.FloatTensor(transition.observation).unsqueeze(0).to(self.device)
                    reward = torch.FloatTensor([transition.reward]).to(self.device)
                    old_log_prob = torch.FloatTensor([transition.old_log_prob]).to(self.device)
                    advantage = torch.FloatTensor([episode.advantages[t_idx]]).to(self.device)
                    ret = torch.FloatTensor([episode.returns[t_idx]]).to(self.device)

                    # Forward pass:
                    # 1. Encode observation
                    encoded = self.encoder(obs)
                    
                    # 2. GRU with learned decay (α_t is computed and used internally)
                    h_gru, alpha_t = self.gru(encoded)
                    
                    # 3. Actor outputs action distribution
                    action_dict = self.actor(h_gru)
                    
                    # 4. Critic estimates value
                    value = self.critic(h_gru)

                    # ===== Policy Loss (PPO clipping) =====
                    # Compare new policy log_prob with old to measure policy change
                    ratio = torch.exp(action_dict["log_prob"] - old_log_prob)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - config["ppo_clip"],
                                       1.0 + config["ppo_clip"]) * advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # ===== Entropy Bonus (exploration) =====
                    # Encourage diverse actions (especially useful in early training)
                    entropy_loss = -config["entropy_coef"] * action_dict["log_prob"].mean()

                    # ===== Value Function Loss =====
                    # Train critic to predict returns accurately
                    value_loss = F.mse_loss(value, ret)

                    # ===== Decay Regularization (optional) =====
                    # Prevent decay from collapsing to 0 or 1
                    # Use entropy of alpha distribution: -E[α log α + (1-α) log(1-α)]
                    eps = 1e-6
                    decay_entropy = -(alpha_t * torch.log(alpha_t + eps) + 
                                      (1 - alpha_t) * torch.log(1 - alpha_t + eps)).mean()
                    decay_reg = 0.01 * decay_entropy  # light regularization

                    # ===== Total Loss =====
                    total_loss = (policy_loss + 
                                 value_loss * config["vf_coef"] + 
                                 entropy_loss + 
                                 decay_reg)

                    # ===== Backward & Optimize =====
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) +
                        list(self.gru.parameters()) +
                        list(self.actor.parameters()) +
                        list(self.critic.parameters()),
                        config["max_grad_norm"]
                    )
                    self.optimizer.step()

                    # Accumulate metrics
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
            "encoder": self.encoder.state_dict(),
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


def train(env, agent, config, num_episodes=10):
    """
    Main PPO training loop

    env: RL environment
    agent: GRUPPOAgent
    config: config dict
    num_episodes: number of episodes to train
    """
    buffer = EpisodeBuffer(config["buffer_capacity"])

    for episode_num in range(num_episodes):
        obs, info = env.reset()
        episode_transitions = []
        episode_rewards = []
        episode_values = []
        episode_dones = []

        h_gru = None
        episode_reward = 0.0

        for step in range(config["episode_length"]):
            # Get action and value from agent
            action_dict, h_gru, value = agent.get_action(obs, h_gru)

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

        print(f"Episode {episode_num + 1}/{num_episodes} | Steps: {len(episode_transitions)} | Reward: {episode_reward:.2f}")

        # Train on batch of episodes if buffer is full
        if len(buffer) >= config["batch_size"]:
            episodes_batch = buffer.sample(config["batch_size"])
            metrics = agent.train_step(episodes_batch, config)
            if metrics:
                print(f"  Training metrics: {metrics}")

    return agent, buffer


if __name__ == "__main__":
    print("GRU-PPO Training Script Initialized")
    print(f"Configuration: {CONFIG}")
    print("\nTo start training, import this module and use the train() function with your environment.")
    print("\nExample:")
    print("  from RL_model.train import GRUPPOAgent, train")
    print("  from envs.env import YourEnvironment")
    print("  env = YourEnvironment()")
    print("  agent = GRUPPOAgent(obs_dim=env.observation_space.shape[0], config=CONFIG)")
    print("  agent, buffer = train(env, agent, CONFIG, num_episodes=100)")
