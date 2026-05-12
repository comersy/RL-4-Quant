# RL-4-Quant

Trying to implement a RL algorithm to create options trading strategies.

## Idea

Train a RL agent to trade options (calls and puts) on a simulated underlying asset. The agent learns which strategies to build — buying or selling calls and puts at different strikes and maturities — purely from price signals, with no predefined rules.

## Environment

At the start of each episode, a full year of underlying prices (252 days) is simulated using Geometric Brownian Motion (GBM) from a starting price and a daily volatility. This gives the agent a price history to work with from day one.

The agent then steps through the episode day by day. At each step it observes the current state of the market and decides what to do. The underlying keeps advancing regardless of the agent's actions.

Options are priced using Black-Scholes, with the realized volatility of the simulated path as the volatility input (no implied vol).

## Action Space

At each step the agent can:
- **Do nothing** — skip the day
- **Buy a call**
- **Buy a put**
- **Sell a call**
- **Sell a put**

Each trade is defined by:
- `type_option` : discrete — {buy call, buy put, sell call, sell put}
- `strike` : continuous, unbounded
- `maturite` : discrete, in days, between 1 and the number of days remaining in the episode
- `quantite` : discrete integer (positive = buy, negative = sell)

## Reward

The agent receives a reward every quarter (every 84 days), equal to its total P&L since the start of the episode. No intermediate reward.
