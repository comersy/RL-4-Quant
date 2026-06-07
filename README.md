# RL-4-Quant 

> A Reinforcement Learning agent that learns to trade options — calls and puts — on a simulated (then real) underlying asset, purely from market signals. No predefined rules. No hardcoded strategies. Just an agent that learns to profit.

---

## Table of Contents

- [Overview](#overview)
- [Core Idea](#core-idea)
- [Architecture](#architecture)
  - [Environment](#environment)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Reward Function](#reward-function)
- [Algorithm — GRU-PPO](#algorithm--gru-ppo)
  - [Why PPO with Learned Decay Memory](#why-ppo-with-learned-decay-memory)
  - [Why GRU with Learned Adaptive Decay](#why-gru-with-learned-adaptive-decay)
  - [Full Architecture](#full-architecture)
- [Current Implementation Status](#current-implementation-status)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [What's Next](#whats-next)

---

## Overview

RL-4-Quant is a **proof of concept** for training a reinforcement learning agent to build options trading strategies on a single underlying asset. The agent can buy or sell calls and puts at various strikes and maturities, and must learn to maximize its total P&L over a trading episode.

The core hypothesis: **the more realistic and information-rich the observations, the better the agent's strategies**. This is why the project is structured around progressive phases — starting from minimal simulated data and gradually moving toward real market history enriched with macro, fundamental, and alternative data.

---

## Core Idea

- The agent receives a **fixed starting budget** at the beginning of each episode
- It steps through **150 calendar days** of real Deribit options market data
- At each day it can **do nothing, or place an options trade** (call/put, strike, maturity, quantity)
- Options are priced from **real market data** (or Black-Scholes as fallback when data is sparse)
- The agent receives a **single reward at the end of the episode**, equal to total P&L (realized + unrealized)
- The underlying asset (BTC) prices come from **real historical data** (May 2023 — May 2026)

The agent is **not given any predefined trading rules**. It must discover by itself when to buy protective puts, when to sell covered calls, when to go directional, when to hold across multiple days, and when to stay flat. The **sparse reward and long memory requirement** force it to learn deep temporal patterns.

---

## Architecture

### Environment

**Data Source**: Real historical options trading data from Deribit (BTC options)
- **Date range**: May 19, 2023 — May 18, 2026 (3 years, ~1,100 trading days)
- **Format**: Daily market snapshots stored in `data/raw/YYYY-MM-DD/`
  - `meta.json`: BTC spot price at end of day
  - `trades.csv`: all option trades (instrument, price, IV, volume)

**Episode Structure**
- **Episode length**: 150 calendar days
- **Reset mechanism**: 
  - Selects a random start date between day 365 and day (total_days - 151)
  - Ensures 365 days of spot history are available (for observation)
  - Guarantees 150 days of future data (for trading)
- **Daily progression**: Agent steps through consecutive days, one at a time
  - Environment loads actual market data from files for that day
  - If option pricing data unavailable, falls back to **Black-Scholes** (σ = last known IV or 50% default)

**Position Management**
- Agent can hold multiple simultaneous options positions (calls and puts, various strikes/maturities)
- Positions can extend **beyond the 150-day episode** — no forced expiration
- At episode end, remaining positions are valued at **mark-to-market** (actual prices or Black-Scholes)
- This forces the agent to learn maturity management, not just position opening

### Action Space

At each timestep, the agent outputs a **hierarchical action space**:

| Parameter | Type | Range |
|-----------|------|-------|
| **Primary decision** | Discrete (0-2) | 0 = do nothing, 1 = trade, 2 = close positions |
| **IF action_type = 1 (trade):** | | |
| `call_or_put` | Continuous → [0, 1] | 0 = call, 1 = put |
| `strike` | Continuous | Unbounded (relative to current spot, e.g. moneyness) |
| `maturity` | Discrete | [1, T_remaining] (days) |
| `quantity_signed` | Continuous | Unbounded; **positive = long/buy, negative = short/sell** |

This design is **cleaner than separating buy/sell**: the sign of `quantity_signed` directly encodes both the direction and magnitude, reducing redundancy and letting the agent learn to use leverage naturally.

### Observation Space

A flat float32 vector includes:

**Market State** (2 values)
- Current BTC spot price (USD)
- Day index within episode (0–89)

**Price History** (365 values)
- Past 365 days of BTC spot prices (for trend/volatility context)

**Tradable Options** (500 × 6 = 3,000 values, padded)
For each available option on the day:
- Call/put flag (1.0 or 0.0)
- Strike price
- Days to expiration
- Mid-price (BTC or USD)
- Implied volatility (%)
- Volume traded

**Open Positions** (10,000 × 8 = 80,000 values, padded)
For each position in agent's portfolio:
- Call/put flag
- Strike price
- Days remaining to expiry (can be negative if expired; still mark-to-market)
- Quantity (absolute value)
- Is-short flag (1.0 if short, 0.0 if long)
- Entry price
- Current mark-to-market price
- Unrealized P&L on position

**Portfolio Summary** (2 values)
- Total realized P&L (from closed positions)
- Total unrealized P&L (from open positions)

**Total observation dimension**: ~86,000 values

### Reward Function

Reward is provided **only at the end of the episode** (day 150):

```
R_final = Σ(realized P&L from closed positions) + Σ(unrealized P&L from open positions)
```

This approach:
- **Gives credit for mature decisions**: Positions that extend beyond the episode are marked-to-market, so the agent receives signal even if it doesn't liquidate
- **Encourages holding vs. panic selling**: The agent learns that profitable positions should be held through the episode
- **Simple and interpretable**: Total episodic profit, nothing more
- **Sparse reward forces memory**: No intermediary rewards, agent depends on GRU to link actions to delayed consequences

The **GRU's learned decay mechanism** is critical here — it helps the agent remember key market states and connect long-term consequences to earlier decisions.

---

## Algorithm — GRU-PPO

### Why PPO with Learned Decay Memory

**Proximal Policy Optimization (PPO)** combined with a learned decay GRU is ideal for this problem:

- **Continuous + discrete actions**: PPO handles mixed action spaces elegantly (discrete action_type, continuous strike/maturity/quantity)
- **Stable training**: PPO clipping prevents catastrophic policy collapses, important for financial environments with noisy rewards
- **Memory via GRU**: Long-term dependencies are critical (positions held 30+ days, multi-step strategies)
- **Learned adaptation**: The **decay network** allows the agent to learn WHEN to remember vs. when to forget, adapting to market regimes
- **Sample efficiency with limited data**: 3 years (1,100 days, ~7 episodes if non-overlapping) is tight — PPO is more sample-efficient than value-based methods
- **End-to-end learning**: Gradients flow through the decay network, training it jointly with the policy

### Why GRU with Learned Adaptive Decay

A memoryless agent sees only the current state of the market. It cannot:
- Remember positions it opened 30 days ago
- Recognize that it's been on a losing streak and should reduce risk
- Build multi-leg strategies over time (e.g. sell a call now, buy a put in 2 weeks)
- Track the evolution of its own P&L trajectory

The **GRU hidden state `h_t`** encodes the agent's full memory since the start of the episode. Compared to LSTM, GRU is more computationally efficient while retaining the ability to capture long-term dependencies.

**Key innovation: Learned Decay Network**

Instead of a fixed GRU with constant gating, the agent learns **how much to remember** at each timestep via a **decay network**:

$$\\alpha_t = \\sigma(\\text{DecayNet}([h_{t-1}, s_t]))$$

Where:
- `h_{t-1}` → previous GRU hidden state (accumulated memory)
- `s_t` → current market/portfolio state (observation)
- `α_t` ∈ [0, 1] → adaptive forget factor
  - α_t ≈ 1 → preserve memory (stable market, holding long-term positions)
  - α_t ≈ 0 → reset memory (regime shift, liquidation event)

**Decay Network Architecture** (implemented inside `GRUCell`):

```
Input: [h_{t-1}, s_t]  (size: gru_hidden_size + obs_dim)
       ↓
MLP Layer 1: Linear(hidden + obs_dim → 128) + ReLU
       ↓
MLP Layer 2: Linear(128 → 64) + ReLU
       ↓
Output Layer: Linear(64 → 1)
       ↓
Sigmoid  →  α_t ∈ [0, 1]
```

Then the decay is applied to the GRU hidden state:

$$h_t = \\alpha_t \\odot h_\\text{raw} + (1 - \\alpha_t) \\odot h_{t-1}$$

This is a **learned blend** between the new GRU state and previous memory. The network trains end-to-end with PPO: when the policy gradient improves, so does decay.

**Benefit**: The GRU becomes **context-aware**. When volatility spikes or the market regime changes, the decay network can reduce α_t and let old memories fade. When in a trend, it keeps α_t high. This is far more expressive than a fixed GRU.

Since the reward is **sparse (only at episode end)**, recurrent memory with learned adaptive decay is essential for connecting multi-step decisions to delayed final consequences.

### Full Architecture

```
╔════════════════════════════════════════════════════════════════════════════╗
║                      GRU-PPO with Learned Decay Memory                     ║
╚════════════════════════════════════════════════════════════════════════════╝

Observation (t)
      │
      ▼
┌──────────────────────────┐
│    FC Encoder            │   Raw obs → latent representation
│  [obs_dim → 256 → 128]   │   Normalizes and projects observation
└────────────┬─────────────┘
             │ encoded_t
             ▼
    ╔════════════════════════════════════════════════════════════════╗
    ║          GRU Cell with Learned Adaptive Decay                  ║
    ║                                                                ║
    ║  1. Standard GRU:                                              ║
    ║     h_raw = GRU_cell(encoded_t, h_{t-1})                       ║
    ║                                                                ║
    ║  2. Decay Network (MLP):                                       ║
    ║     decay_input = concat([h_{t-1}, encoded_t])                 ║
    ║     α_t = σ(MLP_decay(decay_input))    ← learns when to forget ║
    ║                                                                ║
    ║  3. Adaptive Blend:                                            ║
    ║     h_t = α_t ⊙ h_raw + (1 - α_t) ⊙ h_{t-1}                    ║
    ║     │         └─ new state  │         └─ preserve memory       ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
             │ h_t (context-aware memory)
             │
             ├─────────────────────────────┬─────────────────────────┐
             ▼                             ▼                         ▼
    ┌──────────────────┐        ┌──────────────────┐    ┌──────────────────┐
    │      Actor       │        │   Critic (V)     │    │ (For monitoring) │
    │   [256 → 128]    │        │  [256 → 128]     │    │  alpha ∈ [0, 1]  │
    │                  │        │                  │    └──────────────────┘
    │  Outputs:        │        │  Outputs:        │
    │  • action_type   │        │  • state_value   │
    │    (3 logits)    │        │                  │
    │  • call_or_put   │        │  Uses: MSE loss  │
    │    (Normal)      │        │  with returns    │
    │  • strike        │        │                  │
    │    (Normal)      │        │                  │
    │  • maturity      │        │                  │
    │    (Categorical) │        │                  │
    │  • quantity_     │        │                  │
    │    signed        │        │                  │
    │    (Normal)      │        │                  │
    │                  │        │                  │
    │ PPO Loss:        │        │                  │
    │ -min(r·adv,      │        │                  │
    │  clip·adv)       │        │                  │
    └──────────────────┘        └──────────────────┘
             │                          │
             └──────────┬───────────────┘
                        ▼
              ╔═══════════════════════════════╗
              ║  Gradient flows back through  ║
              ║  ENTIRE network including     ║
              ║  decay network (α_t)          ║
              ╚═══════════════════════════════╝
```

**Key Components:**

1. **FC Encoder** → Projects raw observation to latent space
2. **GRU Cell with Decay Network** → The core innovation
   - Standard `GRUCell` produces `h_raw`
   - **Decay MLP** learns context-dependent forget factor `α_t`
   - **Adaptive blend** mixes new information with old memory
3. **Actor** → Outputs hierarchical actions with learnable log_probs
4. **Critic** → Estimates value for advantage calculation
5. **Decay Regularization** → Prevents α_t from collapsing

**Data Flow:**
- Observation → Encoder → [h_prev, encoded] → Decay MLP → α_t
- GRUCell(encoded, h_prev) → h_raw
- h_new = α_t × h_raw + (1-α_t) × h_prev → Actor/Critic → Actions & Values

**Training:**
- PPO clipping on policy gradients
- MSE on value function
- Decay entropy regularization (optional but recommended)
- Full backprop through decay network — it learns end-to-end!

**Hierarchical action structure**: The actor first predicts the primary action (do nothing, trade, close). Only if the prediction is "trade" does it output the detailed trade parameters (call/put, strike, maturity, buy/sell, quantity).

---

## Current Implementation Status

✅ **Completed**
- Real Deribit BTC options data (May 2023 — May 2026)
- GRU-PPO agent with learned adaptive decay memory
- 150-day episodic training with random resets
- Black-Scholes fallback for sparse option data
- Mark-to-market valuation of open positions
- Sparse reward (episode-end total P&L signal)
- Continuous hierarchical action space with signed quantities
- End-to-end training with decay network gradient flow

🔄 **Potential Extensions**
- Add technical indicators (RSI, MACD, Bollinger Bands) to observation
- Implement portfolio Greeks (delta, gamma, vega, theta) computation
- Add risk-free rate and macro indicators to observation
- Multi-asset extension: options on multiple underlyings
- Online learning: agent continues updating on live market data
- Alternative data: sentiment scores, macro event indicators
- Curriculum learning: start with trending/easy markets, add choppy regimes
- Hyperparameter optimization: decay network architecture, learning rate scheduling

---

## Key Financial Indicators

The following indicators are included across phases and why they matter for options trading:

| Indicator | Why it matters for options |
|-----------|---------------------------|
| **Realized volatility (5d/20d/60d)** | Direct input to Black-Scholes; drives option pricing |
| **Vol-of-vol** | Signals regime changes; affects when to buy/sell vol |
| **RSI** | Overbought/oversold signals → directional bias for calls/puts |
| **MACD** | Trend strength and reversals |
| **Bollinger Bands** | Volatility expansion/contraction → breakout or mean-reversion plays |
| **Skewness of returns** | Tail risk indicator → when to buy protective puts |
| **Kurtosis** | Fat tail detection → affects vol surface |
| **Delta (portfolio)** | Net directional exposure — is the agent hedged or directional? |
| **Gamma (portfolio)** | Sensitivity to large moves — convexity risk |
| **Vega (portfolio)** | Sensitivity to volatility changes |
| **Theta (portfolio)** | Time decay cost of current positions |
| **VIX / implied vol** | Market fear gauge; vol premium over realized → sell or buy vol |
| **Risk-free rate** | Direct Black-Scholes input; affects call/put pricing asymmetry |
| **Beta vs market** | Systematic vs idiosyncratic risk decomposition |

---

## Tech Stack

```
Python 3.10+
├── torch (2.0+)           # PPO agent, GRU, decay network
├── gymnasium              # official RL environment API
├── numpy                  # numerical computing
├── scipy                  # Black-Scholes pricing, statistics
├── pandas                 # data manipulation
├── matplotlib             # plotting results
└── requests               # API calls (for potential live data)
```

---

## Project Structure

```
RL-4-Quant/
├── README.md
├── requirements.txt
├── .venv/                         # Python virtual environment
│
├── data/
│   ├── loader.py                  # Load daily Deribit snapshots from raw/
│   ├── download.py                # Future: download from Deribit API
│   └── raw/                       # Daily snapshots (2023-05-19 to 2026-05-18)
│       ├── 2023-05-19/
│       │   ├── meta.json          # {"spot": 26890.92}
│       │   └── trades.csv         # instrument_name, price, iv, amount
│       ├── 2023-05-20/
│       └── ... (1096 days total)
│
├── envs/
│   ├── env.py                     # OptionsEnv: Gymnasium-compatible environment
│   │                              # - 150-day episodes, random reset
│   │                              # - Real Deribit data, Black-Scholes fallback
│   │                              # - Mark-to-market open positions
│   │                              # - Continuous hierarchical actions
│   ├── pricing.py                 # Black-Scholes call/put pricing
│   └── __init__.py
│
├── RL_model/
│   ├── train.py                   # Complete GRU-PPO training
│   │                              # - GRUCell with learned decay network
│   │                              # - Actor (hierarchical actions)
│   │                              # - Critic (value function)
│   │                              # - Training loop with GAE, PPO clipping
│   └── __init__.py
│
├── test_env_train_compatibility.py # Verify environment ↔ agent integration
├── test_random_positions.py        # Test random position initialization
└── test_debug_integration.py       # Debug agent-env interaction
```

**Key Files:**

- **[envs/env.py](envs/env.py)**: The core environment
  - Loads real Deribit data from `data/raw/`
  - Handles 150-day episodes with random resets (guarantees 150-day horizon)
  - Returns flat observation vector (~86k values)
  - Accepts continuous hierarchical actions
  - Reward: episode-end realized P&L + unrealized P&L

- **[RL_model/train.py](RL_model/train.py)**: The learning agent
  - `GRUCell`: GRU with learned adaptive decay network (MLP)
  - `ActorNetwork`: Outputs hierarchical actions (action_type, call_or_put, strike, maturity, quantity_signed)
  - `CriticNetwork`: Outputs state value for GAE calculation
  - `GRUPPOAgent`: Full training loop with PPO clipping, entropy bonus, decay regularization

- **[data/loader.py](data/loader.py)**: Data loading
  - `list_available_days()`: Get sorted list of all available dates
  - `load_day(date_str)`: Load spot + options data for a single day
  - `max_options_per_day()`: Get maximum options count for padding


---

## Getting Started

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run compatibility tests
python test_env_train_compatibility.py

# 4. Train the agent
#    (See RL_model/train.py for training configuration)
python -c "
from RL_model.train import GRUPPOAgent
from envs.env import OptionsEnv

config = {
    'device': 'cpu',
    'encoder_hidden_sizes': [128, 64],
    'gru_hidden_size': 64,
    'actor_hidden_sizes': [64, 32],
    'critic_hidden_sizes': [64, 32],
    'learning_rate': 3e-4,
}

env = OptionsEnv()
agent = GRUPPOAgent(obs_dim=env.observation_space.shape[0], config=config)
# Training loop (see RL_model/train.py for full implementation)
"

# 5. Evaluate on random initial positions (tests generalization)
python -c "
from RL_model.train import GRUPPOAgent
from envs.env import OptionsEnv

env_eval = OptionsEnv(init_random_positions=True)
# Run evaluation episodes
"
```

---

## What's Next

1. **Run training**: Start with the configuration in [RL_model/train.py](RL_model/train.py). The agent will learn to trade options using real Deribit data.

2. **Monitor learning**: Log episode returns, realized P&L, unrealized P&L, action distributions, and decay factors over time.

3. **Evaluate robustness**: Test the trained agent with `init_random_positions=True` to verify it can handle starting in unfamiliar portfolio states (not just empty).

4. **Extend observations**: Add technical indicators (RSI, MACD), portfolio Greeks (delta, gamma), or macro indicators to the observation vector.

5. **Experiment with decay**: Try different decay network architectures (deeper MLP, attention-based) to see if the agent learns more nuanced memory patterns.

6. **Out-of-sample validation**: Evaluate the agent on held-out date ranges (e.g., train on 2023-2025, test on 2026 data) to assess generalization.

---

**Questions or issues?** Check the [compatibility tests](test_env_train_compatibility.py) or review the [environment documentation](envs/env.py).
