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
- [Algorithm — LSTM-SAC](#algorithm--lstm-sac)
  - [Why SAC](#why-sac)
  - [Why LSTM](#why-lstm)
  - [Full Architecture](#full-architecture)
- [Roadmap — Progressive Phases](#roadmap--progressive-phases)
  - [Phase 1 — Baseline (Simulated, Minimal Obs)](#phase-1--baseline-simulated-minimal-obs)
  - [Phase 2 — Technical Indicators](#phase-2--technical-indicators)
  - [Phase 3 — Macro & Fundamentals](#phase-3--macro--fundamentals)
  - [Phase 4 — Alternative Data & Real History](#phase-4--alternative-data--real-history)
- [Key Financial Indicators](#key-financial-indicators)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [What's Next](#whats-next)

---

## Overview

RL-4-Quant is a **proof of concept** for training a reinforcement learning agent to build options trading strategies on a single underlying asset. The agent operates with a fixed budget, can buy or sell calls and puts at various strikes and maturities, and must learn to maximize its total P&L over a trading episode.

The core hypothesis: **the more realistic and information-rich the observations, the better the agent's strategies**. This is why the project is structured around progressive phases — starting from minimal simulated data and gradually moving toward real market history enriched with macro, fundamental, and alternative data.

---

## Core Idea

- The agent receives a **fixed starting budget** at the beginning of each episode
- It steps through **252 trading days** (1 simulated year)
- At each step it can **do nothing, or place an options trade** (call/put, buy/sell)
- Options are priced using **Black-Scholes** with realized volatility from the simulated path
- The agent receives a **reward every quarter (every 84 days)**, equal to total P&L since episode start
- The underlying asset price evolves via **Geometric Brownian Motion (GBM)** in early phases, and real historical data in later phases

The agent is **not given any predefined trading rules**. It must discover by itself when to buy protective puts, when to sell covered calls, when to go directional, and when to stay flat.

---

## Architecture

### Environment

- **Episode length**: 252 days (1 trading year)
- **Underlying**: simulated via GBM (S₀, σ daily) → then real historical prices
- **Options pricing**: Black-Scholes, using realized volatility of the current simulated path
- **Budget**: fixed at episode start, tracks P&L across all open positions
- **Positions**: the agent can hold multiple simultaneous options positions

At the start of each episode, the full price path for the year is pre-simulated. This gives the agent access to a price history from day 1. The underlying then advances day by day regardless of the agent's actions.

### Action Space

At each timestep, the agent outputs a **continuous action vector** that is discretized before being sent to the environment:

| Parameter | Type | Range |
|-----------|------|-------|
| `action_type` | Continuous → discretized | 0 = nothing, 1 = buy call, 2 = buy put, 3 = sell call, 4 = sell put |
| `strike` | Continuous | Unbounded (relative to current spot, e.g. moneyness) |
| `maturity` | Continuous → discretized | 1 to T_remaining (days) |
| `quantity` | Continuous → discretized integer | 1 to N_max |

SAC outputs all four values as continuous numbers. Post-processing handles discretization (argmax for type, rounding for maturity and quantity) before the environment receives the action. This preserves the benefits of continuous action optimization while respecting the discrete nature of certain parameters.

### Observation Space

Observations are structured progressively across phases (see [Roadmap](#roadmap--progressive-phases)). At full capacity (Phase 4), the observation vector includes:

**Market state**
- Current spot price + normalized price history (5d, 10d, 20d windows)
- Daily log-returns + rolling statistics

**Volatility**
- Realized volatility (5d, 20d, 60d windows)
- Volatility of volatility (vol-of-vol)
- VIX / implied volatility (Phase 3+)

**Technical indicators** *(Phase 2+)*
- RSI (14d)
- MACD + signal line
- Bollinger Bands (upper, lower, %B)
- Momentum (10d, 20d)
- Skewness and kurtosis of recent returns

**Options & Greeks** *(Phase 2+)*
- Delta, Gamma, Vega, Theta of current portfolio
- Portfolio-level net delta / net gamma exposure

**Portfolio state**
- Current budget / remaining cash
- Current total P&L
- Open positions: strike, maturity remaining, type, quantity (for each open leg)
- Days remaining in episode

**Macro & Fundamentals** *(Phase 3+)*
- Risk-free rate (direct input to Black-Scholes)
- Sector index performance
- P/E ratio, beta vs market

**Alternative data** *(Phase 4)*
- NLP sentiment score from news / financial social media
- Competitor price correlations
- Macro releases: CPI, Fed rate decisions, PMI

### Reward Function

The primary reward is **quarterly P&L** (every 84 days). However, a pure sparse reward creates a severe credit assignment problem — the agent cannot easily link an action at day 3 to a gain at day 84.

The shaped reward is:

```
R_total = R_pnl                          # main signal: quarterly P&L
        + λ₁ × R_action_bonus            # small bonus per trade executed (fights passivity)
        - λ₂ × R_drawdown_penalty        # penalty if drawdown exceeds threshold
        - λ₃ × R_concentration_penalty   # penalty for over-concentration on a single trade
```

**`R_action_bonus`** is the critical component to prevent the agent from converging to a do-nothing policy. A small positive reward is given for each trade executed, and a small negative reward is applied if the agent goes 5+ consecutive days without any action.

---

## Algorithm — LSTM-SAC

### Why SAC

**Soft Actor-Critic (SAC)** is the right algorithm for this problem because:

- **Continuous action space**: SAC is designed for continuous actions. The strike is unbounded and continuous — discretizing it coarsely would lose critical information about moneyness
- **Maximum entropy objective**: SAC maximizes both reward *and* policy entropy. This prevents the agent from collapsing to a single repetitive strategy and naturally encourages exploration of diverse strikes, maturities, and trade types
- **Sample efficiency**: far more sample-efficient than PPO on continuous spaces — important when each episode is 252 steps and training is expensive
- **Stability**: SAC's twin critic architecture and temperature-tuned entropy make it significantly more stable than DDPG on noisy financial environments

### Why LSTM

A memoryless agent sees only the current state of the market. It cannot:
- Remember positions it opened 30 days ago
- Recognize that it's been on a losing streak and should reduce risk
- Build multi-leg strategies over time (e.g. sell a call now, buy a put in 2 weeks)
- Track the evolution of its own P&L trajectory

The **LSTM hidden state `h_t`** encodes the agent's full memory since the start of the episode. At each step, `h_t` is updated with the new observation and passed to both the Actor and the Critic. This gives the agent a recurrent memory of:
- Its own action history
- The sequence of P&L realizations
- The price path it has lived through
- Its current open positions and their aging

Since the reward is **sparse and quarterly**, recurrent memory is not optional — it is essential for the agent to connect its current decisions to their future consequences.

### Full Architecture

```
Observation (t)
      │
      ▼
┌─────────────┐
│ FC Encoder  │   Linear layers — normalizes and projects raw obs
│ [256 → 128] │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    LSTM     │   Recurrent layer — full episode memory
│ hidden=256  │   h_t, c_t = LSTM(encoded_obs_t, h_{t-1}, c_{t-1})
└──────┬──────┘
       │  h_t
       ├────────────────────────┐
       ▼                        ▼
┌─────────────┐        ┌─────────────────┐
│    Actor    │        │  Twin Critics   │   Q₁(s,a) and Q₂(s,a)
│ [128 → 64]  │        │  [128 → 64]     │   min(Q₁, Q₂) → anti-overestimation
│ outputs:    │        └─────────────────┘
│ μ, σ → a   │
│ [type, K,  │
│  T, qty]   │
└─────────────┘
```

**LSTM mode**: full recurrent (Option B) — `h_{t-1}` is passed at every step so the hidden state accumulates information from the very start of the episode. This is necessary given the sparse quarterly reward.

**Replay buffer**: stores full episode sequences, not individual transitions. LSTM training requires sequential context — sampling random transitions would break the temporal structure and corrupt the hidden state.

---

## Roadmap — Progressive Phases

### Phase 1 — Baseline (Simulated, Minimal Obs)

**Goal**: establish that the agent can learn *something* before adding complexity.

- Underlying: GBM with fixed S₀ and σ
- Observations: spot price, price history (5d/20d), realized vol, P&L, budget, days remaining, open positions
- Algorithm: LSTM-SAC, basic shaped reward
- Data: 100% simulated, generated fresh each episode
- Baseline to beat: a rule-based agent that buys calls when RSI < 30

This phase validates the environment, the reward shaping, and the LSTM-SAC pipeline before any real data is introduced.

---

### Phase 2 — Technical Indicators

**Goal**: give the agent richer market signals to make more informed decisions.

- Add RSI, MACD, Bollinger Bands, momentum, skewness/kurtosis to observations
- Add portfolio Greeks (delta, gamma, vega, theta) to observations
- Introduce **curriculum learning**: start with trending/easy markets, progressively add choppy/mean-reverting regimes
- Upgrade GBM to **stochastic volatility** (Heston model) for more realistic price paths
- Begin logging Tensorboard metrics: entropy, Q-values, action distributions, P&L per episode

---

### Phase 3 — Macro & Fundamentals

**Goal**: move toward a realistic market environment.

- Add risk-free rate, VIX / implied vol, sector indices, P/E, beta to observations
- Switch from pure GBM to **real historical price data** (via yfinance or equivalent) for the underlying
- Introduce **SAC temperature tuning** — auto-adjust entropy coefficient as the agent becomes more confident
- Evaluate on out-of-sample historical periods to check for overfitting

---

### Phase 4 — Alternative Data & Real History

**Goal**: maximum environment realism, full information agent.

- Add NLP sentiment scores (news, financial social media) to observations
- Add competitor / correlated asset prices
- Add macro event indicators (CPI release, Fed decisions, earnings dates)
- Explore **multi-asset extension**: agent trades options on a small basket of stocks
- Consider **online learning**: agent continues updating from live market data

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
Python 3.11
├── gymnasium              # custom options trading environment
├── stable-baselines3      # SAC base implementation
├── sb3-contrib            # RecurrentSAC / LSTM support
├── torch                  # custom LSTM-SAC architecture
├── numpy / pandas         # data handling
├── scipy                  # Black-Scholes pricing, statistics
├── yfinance               # real historical data (Phase 3+)
├── ta / ta-lib            # technical indicators (Phase 2+)
└── tensorboard            # training monitoring (mandatory from day 1)
```

---

## Project Structure

```
rl-4-quant/
├── README.md
├── requirements.txt
│
├── env/
│   ├── options_env.py          # core Gym environment
│   ├── gbm_simulator.py        # GBM / Heston price simulation
│   ├── black_scholes.py        # BS pricing + Greeks
│   └── reward.py               # shaped reward function
│
├── agent/
│   ├── lstm_sac.py             # LSTM-SAC full architecture
│   ├── actor.py                # Actor network (μ, σ outputs)
│   ├── critic.py               # Twin Critic networks
│   └── replay_buffer.py        # sequential episode replay buffer
│
├── observations/
│   ├── phase1.py               # minimal obs builder
│   ├── phase2.py               # + technical indicators + Greeks
│   ├── phase3.py               # + macro + fundamentals
│   └── phase4.py               # + alternative data
│
├── training/
│   ├── train.py                # main training loop
│   ├── evaluate.py             # out-of-sample evaluation
│   └── curriculum.py           # curriculum learning schedule
│
├── baselines/
│   └── rule_based_agent.py     # RSI-based baseline to beat
│
└── logs/
    └── tensorboard/            # training metrics
```

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/your-handle/rl-4-quant.git
cd rl-4-quant

# Install dependencies
pip install -r requirements.txt

# Run Phase 1 training
python training/train.py --phase 1 --episodes 5000

# Monitor training
tensorboard --logdir logs/tensorboard
```

---
