"""
RL environment for BTC options trading on real data.

=============================================================================
OVERVIEW
=============================================================================
Each episode lasts 90 calendar days. The agent starts at a random day in the
dataset (with at least 90 days of data ahead) and trades day by day.

At each step the agent can place up to MAX_TRADES trades from the options
actually traded that day on Deribit. The data is real historical data
downloaded via data/download.py.

-----------------------------------------------------------------------------
OBSERVATION SPACE
-----------------------------------------------------------------------------
Flat float32 vector:

  Market info:
    [0]              : current BTC spot (USD)
    [1]              : day index inside the episode (0..89)
    [2 : 367]        : spot history of the LAST 365 days (USD)
                        (episode always starts after at least 365 days of data)

  Option grid (today's tradable options, padded to self.MAX_OPTIONS):
    per option: [option_type, strike, days_to_expiry, price_btc, iv, volume]
    -> shape: (self.MAX_OPTIONS x 6,)
    Empty slots filled with zeros.

  Portfolio (up to MAX_PORTFOLIO open positions):
    per position: [option_type, strike, days_to_expiry, quantity, is_short,
                   entry_price, current_price, unrealized_pnl]
    -> shape: (MAX_PORTFOLIO x 8,)

  Scalar state:
    realized_pnl, unrealized_pnl

-----------------------------------------------------------------------------
ACTION SPACE
-----------------------------------------------------------------------------
The agent outputs MAX_TRADES slots. Each slot is:

  [type_action, option_index, quantite, close_index]

  type_action  : Discrete(4)
    0 = do nothing
    1 = buy   (uses option_index, quantite)
    2 = sell  (uses option_index, quantite)
    3 = close (uses close_index)

  option_index : Discrete(self.MAX_OPTIONS)  - index in today's option list
  quantite     : Discrete(MAX_QTY)           - contracts, >= 1
  close_index  : Discrete(MAX_PORTFOLIO)

-----------------------------------------------------------------------------
REWARD
-----------------------------------------------------------------------------
Reward = 0 every day except on the final day (day 89) where it equals the
total realized P&L of the episode.

-----------------------------------------------------------------------------
PRICING
-----------------------------------------------------------------------------
- All transactions (buy / sell / close) use the option's "price" field
  (last trade of the day in BTC).
- Mark-to-market uses today's price if the option traded that day, otherwise
  Black-Scholes fallback with last known IV.
- Expired options settle at intrinsic payoff: max(S-K, 0) for calls,
  max(K-S, 0) for puts (paid in BTC).
=============================================================================
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime

from data.loader   import list_available_days, load_day, max_options_per_day
from envs.pricing  import black_scholes


# ── Constants ─────────────────────────────────────────────────────────────────

EPISODE_DAYS  = 90        # length of an episode in calendar days
SPOT_HISTORY  = 365       # days of past spot shown in observation
MAX_PORTFOLIO = 10000     # max open positions
MAX_QTY       = 100       # max contracts per trade
MAX_TRADES    = 500       # max trades per step


class OptionsEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # load all available days from disk
        self.all_days = list_available_days()
        if len(self.all_days) < EPISODE_DAYS + SPOT_HISTORY:
            raise ValueError(
                f"Not enough data: need at least {EPISODE_DAYS + SPOT_HISTORY} days, "
                f"got {len(self.all_days)}"
            )

        # episode state (set by reset)
        self.start_day_idx   = 0
        self.current_day_idx = 0
        self.episode_day     = 0
        self.current_data    = None
        self.spot_history    = []
        self.portfolio       = []
        self.realized_pnl    = 0.0
        self.unrealized_pnl  = 0.0
        self.today_options   = []
        self.MAX_OPTIONS     = max_options_per_day()   # computed from data

        # action space
        single_trade = spaces.Dict({
            "type_action":  spaces.Discrete(4),
            "option_index": spaces.Discrete(self.MAX_OPTIONS),
            "quantite":     spaces.Discrete(MAX_QTY),
            "close_index":  spaces.Discrete(MAX_PORTFOLIO),
        })
        self.action_space = spaces.Tuple(tuple(single_trade for _ in range(MAX_TRADES)))

        # observation space (flat float32 vector)
        obs_size = (
            2                            # spot, episode_day
            + SPOT_HISTORY               # past spots
            + self.MAX_OPTIONS * 6       # tradable options
            + MAX_PORTFOLIO * 8          # portfolio
            + 2                          # realized + unrealized pnl
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.all_days) - EPISODE_DAYS - 1
        min_start = SPOT_HISTORY
        self.start_day_idx   = int(self.np_random.integers(min_start, max_start + 1))
        self.current_day_idx = self.start_day_idx
        self.episode_day     = 0

        self.portfolio      = []
        self.realized_pnl   = 0.0
        self.unrealized_pnl = 0.0

        self.spot_history = []
        for i in range(self.start_day_idx - SPOT_HISTORY, self.start_day_idx):
            data = load_day(self.all_days[i])
            self.spot_history.append(data["spot"] if data["spot"] is not None else 0.0)

        self._load_current_day()
        return self._get_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        for slot in action:
            self._execute_trade(slot)

        self.current_day_idx += 1
        self.episode_day     += 1
        self._load_current_day()
        self._mark_to_market()

        terminated = self.episode_day >= EPISODE_DAYS - 1
        reward     = self.realized_pnl if terminated else 0.0
        return self._get_obs(), float(reward), terminated, False, {}

    # ── Day loading ───────────────────────────────────────────────────────────

    def _load_current_day(self):
        day_str           = self.all_days[self.current_day_idx]
        self.current_data = load_day(day_str)

        if self.current_data["spot"] is not None:
            self.spot_history.append(self.current_data["spot"])
            self.spot_history = self.spot_history[-SPOT_HISTORY:]

        today              = datetime.strptime(day_str, "%Y-%m-%d")
        self.today_options = [
            o for o in self.current_data["options"]
            if o["expiry"].date() > today.date()
        ]

    # ── Trades ────────────────────────────────────────────────────────────────

    def _execute_trade(self, slot: dict):
        t = int(slot["type_action"])
        if t == 0:
            return

        if t == 3:
            idx = int(slot["close_index"])
            if idx < len(self.portfolio):
                self._close(idx)
            return

        opt_idx = int(slot["option_index"])
        if opt_idx >= len(self.today_options):
            return
        opt = self.today_options[opt_idx]

        qty        = max(1, int(slot["quantite"]))
        side_short = (t == 2)
        price      = opt["price"]

        if side_short:
            self.realized_pnl += price * qty
        else:
            self.realized_pnl -= price * qty

        self.portfolio.append({
            "instrument":  opt["instrument"],
            "strike":      opt["strike"],
            "expiry":      opt["expiry"],
            "option_type": opt["option_type"],
            "quantity":    qty,
            "is_short":    side_short,
            "entry_price": price,
            "price":       price,
            "iv_last":     opt["iv"] if opt["iv"] > 0 else 50.0,
            "stale":       False,
        })

    def _close(self, idx: int):
        p     = self.portfolio[idx]
        price = p["price"]
        if p["is_short"]:
            self.realized_pnl -= price * p["quantity"]
        else:
            self.realized_pnl += price * p["quantity"]
        self.portfolio.pop(idx)

    # ── Mark to market ────────────────────────────────────────────────────────

    def _mark_to_market(self):
        opts_by_instr = {o["instrument"]: o for o in self.current_data["options"]}
        spot          = self.current_data["spot"]
        if spot is None:
            spot = self.spot_history[-1] if self.spot_history else 0.0
        day_today     = datetime.strptime(self.all_days[self.current_day_idx], "%Y-%m-%d")
        unrealized    = 0.0

        for p in self.portfolio:
            if p["expiry"] <= day_today:
                payoff_usd = (max(spot - p["strike"], 0.0) if p["option_type"] == "call"
                              else max(p["strike"] - spot, 0.0))
                payoff_btc = payoff_usd / spot if spot > 0 else 0.0
                sign       = -1 if p["is_short"] else 1
                self.realized_pnl += sign * payoff_btc * p["quantity"]
                p["expired"] = True
                continue

            o = opts_by_instr.get(p["instrument"])
            if o is not None:
                p["price"]   = o["price"]
                p["iv_last"] = o["iv"] if o["iv"] > 0 else p["iv_last"]
                p["stale"]   = False
            else:
                days_left = (p["expiry"] - day_today).days
                T         = max(days_left, 1) / 365
                sigma     = p["iv_last"] / 100
                price_usd = black_scholes(
                    S=spot, K=p["strike"], T=T, r=0.0,
                    sigma=sigma, option_type=p["option_type"],
                )
                p["price"] = price_usd / spot if spot > 0 else 0.0
                p["stale"] = True

            sign        = -1 if p["is_short"] else 1
            unrealized += sign * (p["price"] - p["entry_price"]) * p["quantity"]

        self.portfolio      = [p for p in self.portfolio if not p.get("expired")]
        self.unrealized_pnl = unrealized

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        spot      = self.current_data["spot"] if self.current_data["spot"] is not None else 0.0
        day_today = datetime.strptime(self.all_days[self.current_day_idx], "%Y-%m-%d")

        hist = np.array(self.spot_history[-SPOT_HISTORY:], dtype=np.float32)

        opt_vec = np.zeros((self.MAX_OPTIONS, 6), dtype=np.float32)
        for i, o in enumerate(self.today_options[:self.MAX_OPTIONS]):
            days_to_exp = (o["expiry"] - day_today).days
            opt_vec[i] = [
                1.0 if o["option_type"] == "call" else 0.0,
                o["strike"],
                days_to_exp,
                o["price"],
                o["iv"],
                o["volume"],
            ]

        port_vec = np.zeros((MAX_PORTFOLIO, 8), dtype=np.float32)
        for i, p in enumerate(self.portfolio[:MAX_PORTFOLIO]):
            days_to_exp = (p["expiry"] - day_today).days
            unrealized  = (-1 if p["is_short"] else 1) * (p["price"] - p["entry_price"]) * p["quantity"]
            port_vec[i] = [
                1.0 if p["option_type"] == "call" else 0.0,
                p["strike"],
                days_to_exp,
                p["quantity"],
                1.0 if p["is_short"] else 0.0,
                p["entry_price"],
                p["price"],
                unrealized,
            ]

        obs = np.concatenate([
            [spot, float(self.episode_day)],
            hist,
            opt_vec.flatten(),
            port_vec.flatten(),
            [self.realized_pnl, self.unrealized_pnl],
        ])
        return obs.astype(np.float32)