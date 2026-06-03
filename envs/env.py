"""
RL environment for BTC options trading on real Deribit data.

=============================================================================
OVERVIEW
=============================================================================
Each episode lasts 150 calendar days. The agent starts at a random day in the
dataset, with at least 365 days of past spot history behind and 150 days of
data ahead. The agent then trades day by day with continuous decision-making.

=============================================================================
ACTION SPACE (from train.py / README)
=============================================================================
At each step, the agent outputs a single hierarchical action:

{
  "action_type": int ∈ {0, 1, 2}
    0 = do nothing
    1 = place new trade (use call_or_put, strike, maturity, quantity_signed)
    2 = close a position

  "call_or_put": float ∈ [-1, 1]
    Rescale to [0, 1]: 0 = CALL, 1 = PUT

  "strike": float (unbounded)
    Moneyness relative to spot. The environment finds the closest available
    option that matches this strike.

  "maturity": int ∈ [1, T_remaining]
    Days to expiry. Environment finds the closest option with this maturity.

  "quantity_signed": float (unbounded)
    Absolute value = number of contracts
    Sign: positive = LONG (buy), negative = SHORT (sell)

  "log_prob": float (for PPO training, ignored by env)
}

=============================================================================
OBSERVATION SPACE
=============================================================================
Flat float32 vector:

  [0]              : current BTC spot (USD)
  [1]              : day index inside the episode (0..149)
  [2:367]          : spot history of LAST 365 days (USD)

  [367:...]        : Current tradable options (padded to MAX_OPTIONS)
                     Per option: [call=1/put=0, strike, days_to_expiry,
                                  price_btc, iv, volume]

  [...+MAX_PORTFOLIO*8]:
                     Current open positions (padded to MAX_PORTFOLIO)
                     Per position: [call=1/put=0, strike, days_to_expiry,
                                    quantity, is_short, entry_price,
                                    current_price, unrealized_pnl]

  [-2]             : realized P&L (USD)
  [-1]             : unrealized P&L (USD)

=============================================================================
REWARD
=============================================================================
Reward is 0 every day, except on the final day (day ) where it equals the
total realized P&L from all trades during the episode.

=============================================================================
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime

from data.loader import list_available_days, load_day, max_options_per_day
from envs.pricing import black_scholes


# ── Constants ─────────────────────────────────────────────────────────────────

EPISODE_DAYS  = 150       # length of an episode in calendar days
SPOT_HISTORY  = 365       # days of past spot shown in observation
MAX_PORTFOLIO = 100     # max open positions to track


class OptionsEnv(gym.Env):
    """
    Options trading environment with continuous hierarchical action space.
    
    The agent receives continuous parameters (strike, maturity, quantity_signed)
    and the environment matches them to actual available options.
    """

    metadata = {"render_modes": []}

    def __init__(self, init_random_positions=False):
        super().__init__()

        # Load all available days from disk
        self.all_days = list_available_days()
        if len(self.all_days) < EPISODE_DAYS + SPOT_HISTORY:
            raise ValueError(
                f"Not enough data: need at least {EPISODE_DAYS + SPOT_HISTORY} days, "
                f"got {len(self.all_days)}"
            )

        # Get max options from data
        self.MAX_OPTIONS = max_options_per_day()
        if self.MAX_OPTIONS == 0:
            self.MAX_OPTIONS = 500  # fallback

        # Feature flag: initialize episodes with random positions (for testing/evaluation only)
        self.init_random_positions = init_random_positions

        # Episode state
        self.start_day_idx = 0
        self.current_day_idx = 0
        self.episode_day = 0
        self.current_data = None
        self.spot_history = []
        self.portfolio = []
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.today_options = []

        # Action space: continuous dict matching train.py
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(3),
            "call_or_put": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "strike": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "maturity": spaces.Box(low=1, high=150, shape=(1,), dtype=np.float32),
            "quantity_signed": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "log_prob": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })

        # Observation space
        obs_size = (
            2                            # spot, episode_day
            + SPOT_HISTORY               # historical spot prices
            + self.MAX_OPTIONS * 6       # tradable options
            + MAX_PORTFOLIO * 8          # open positions
            + 2                          # realized + unrealized P&L
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset environment to a new episode.
        
        Initializes with random positions to force generalization and avoid
        overfitting on limited 3-year dataset. Each episode has different
        starting conditions (0-5 random positions).
        """
        super().reset(seed=seed)

        max_start = len(self.all_days) - EPISODE_DAYS - 1
        min_start = SPOT_HISTORY
        self.start_day_idx = int(self.np_random.integers(min_start, max_start + 1))
        self.current_day_idx = self.start_day_idx
        self.episode_day = 0

        self.portfolio = []
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

        # Load spot history
        self.spot_history = []
        for i in range(self.start_day_idx - SPOT_HISTORY, self.start_day_idx):
            data = load_day(self.all_days[i])
            self.spot_history.append(data["spot"] if data["spot"] is not None else 0.0)

        self._load_current_day()
        
        # Initialize with random starting positions only if enabled
        # (For testing/evaluation, not for training)
        if self.init_random_positions:
            self._initialize_random_positions()
        
        return self._get_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        """Execute one trading day."""
        self._execute_action(action)

        self.current_day_idx += 1
        self.episode_day += 1
        self._load_current_day()
        self._mark_to_market()

        terminated = self.episode_day >= EPISODE_DAYS - 1
        
        # Reward = realized P&L + unrealized P&L at end of episode
        # This gives coherent signal on maturity choices (even if beyond episode)
        if terminated:
            reward = float(self.realized_pnl) + float(self.unrealized_pnl)
        else:
            reward = 0.0

        return self._get_obs(), reward, terminated, False, {}

    # ── Random Position Initialization ────────────────────────────────────────

    def _initialize_random_positions(self, prob_position=0.6, max_positions=5):
        """
        Initialize episode with random open positions.
        
        This forces the agent to learn position management, not just opening.
        Each episode has 0-max_positions random initial positions with:
        - Random call/put type
        - Random strike (moneyness relative to spot)
        - Random maturity (1-60 days)
        - Random quantity (long or short, 0.5-2.5 contracts)
        - Entry price set to current market price (so unrealized P&L = 0 initially)
        """
        if self.today_options is None or len(self.today_options) == 0:
            return
        
        num_init_positions = self.np_random.integers(0, max_positions + 1)
        
        for _ in range(num_init_positions):
            # Random option type
            option_type = "call" if self.np_random.random() > 0.5 else "put"
            
            # Random strike (moneyness: -20% to +20% relative to spot)
            moneyness_factor = self.np_random.uniform(0.8, 1.2)
            strike_target = self.spot * moneyness_factor
            
            # Random maturity (1-180 days) - some positions extend beyond episode
            maturity_target = self.np_random.integers(1, 181)
            
            # Random quantity (0.5-2.5 contracts, can be short)
            quantity = self.np_random.uniform(0.5, 2.5)
            if self.np_random.random() > 0.5:
                quantity = -quantity  # Short position
            
            # Find closest matching option
            match = self._find_closest_option(option_type, strike_target, maturity_target)
            if match is None:
                continue
            
            option, strike, maturity = match
            
            # Create position with entry_price = current price
            # (so unrealized P&L starts at 0, agent must manage it)
            position = {
                "instrument": option["instrument"],
                "strike": strike,
                "expiry": option["expiry"],
                "maturity": maturity,
                "option_type": option_type,
                "quantity": abs(quantity),
                "is_short": quantity < 0,
                "entry_price": option["price"],
                "price": option["price"],
                "iv_last": option["iv"] if option["iv"] > 0 else 50.0,
                "stale": False,
            }
            
            self.portfolio.append(position)

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

    @property
    def spot(self):
        """Get current BTC spot price."""
        if self.current_data and self.current_data["spot"] is not None:
            return self.current_data["spot"]
        return self.spot_history[-1] if self.spot_history else 0.0


    # ── Action Execution ────────────────────────────────────────────────────

    def _to_python_scalar(self, x):
        """Safely convert array or scalar to Python scalar (float/int)."""
        if hasattr(x, 'item'):  # numpy scalar / array
            return x.item()
        return x

    def _execute_action(self, action):
        """
        Execute action from GRU-PPO agent.
        
        action: dict with keys:
            - "action_type": scalar (0/1/2)
            - "call_or_put": scalar or array in [-1, 1]
            - "strike": scalar or array
            - "maturity": scalar or array in [1, 150]
            - "quantity_signed": scalar or array
            - "log_prob": scalar or array (ignored)
        """
        action_type = int(self._to_python_scalar(np.atleast_1d(action.get("action_type", 0))[0]))

        if action_type == 1:  # Trade
            call_or_put = float(self._to_python_scalar(np.atleast_1d(action.get("call_or_put", 0.5))[0]))
            strike = float(self._to_python_scalar(np.atleast_1d(action.get("strike", self.spot))[0]))
            maturity = int(self._to_python_scalar(np.atleast_1d(action.get("maturity", 30))[0]))
            maturity = max(1, maturity)  # Allow any maturity (including beyond 150-day episode)
            quantity_signed = float(self._to_python_scalar(np.atleast_1d(action.get("quantity_signed", 0.0))[0]))

            option_type = "call" if call_or_put > 0 else "put"
            self._open_position(option_type, strike, maturity, quantity_signed)

        elif action_type == 2:  # Close position
            self._close_position_by_maturity()


    # ── Option Matching ────────────────────────────────────────────────────

    def _find_closest_option(self, option_type, strike_target, maturity_target):
        """
        Find closest option to target strike and maturity from today_options.
        
        Returns: (option, strike, maturity) or None if no options available.
        """
        candidates = [
            opt for opt in self.today_options
            if opt["option_type"] == option_type
        ]

        if not candidates:
            return None

        day_today = datetime.strptime(self.all_days[self.current_day_idx], "%Y-%m-%d")

        def score(opt):
            strike_dist = abs(opt["strike"] - strike_target) / (self.spot + 1e-8)
            maturity_dist = abs((opt["expiry"] - day_today).days - maturity_target) / 90.0
            return strike_dist + maturity_dist

        best_opt = min(candidates, key=score)
        return (best_opt, best_opt["strike"], (best_opt["expiry"] - day_today).days)

    # ── Position Management ────────────────────────────────────────────────

    def _open_position(self, option_type, strike_target, maturity_target, quantity_signed):
        """Open a new position or add to existing one."""
        if quantity_signed == 0.0:
            return

        match = self._find_closest_option(option_type, strike_target, maturity_target)
        if match is None:
            return

        option, strike, maturity = match

        position = {
            "instrument": option["instrument"],
            "strike": strike,
            "expiry": option["expiry"],
            "maturity": maturity,
            "option_type": option_type,
            "quantity": abs(quantity_signed),
            "is_short": quantity_signed < 0,
            "entry_price": option["price"],
            "price": option["price"],
            "iv_last": option["iv"] if option["iv"] > 0 else 50.0,
            "stale": False,
        }

        self.portfolio.append(position)

    def _close_position_by_maturity(self):
        """Close oldest positions by maturity."""
        day_today = datetime.strptime(self.all_days[self.current_day_idx], "%Y-%m-%d")
        
        remaining = []
        for p in self.portfolio:
            if p["expiry"] <= day_today:
                self._realize_pnl(p)
            else:
                remaining.append(p)
        
        self.portfolio = remaining

    def _realize_pnl(self, position):
        """Realize profit/loss for a closed position."""
        entry = position["entry_price"]
        quantity = position["quantity"]
        
        spot = self.current_data["spot"] if self.current_data["spot"] is not None else self.spot_history[-1]
        
        # Intrinsic value at expiry
        strike = position["strike"]
        if position["option_type"] == "call":
            payoff = max(spot - strike, 0.0)
        else:
            payoff = max(strike - spot, 0.0)
        
        if position["is_short"]:
            pnl = (entry - payoff) * quantity
        else:
            pnl = (payoff - entry) * quantity
        
        self.realized_pnl += pnl

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