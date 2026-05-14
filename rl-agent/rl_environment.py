import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.underlying import Underlying
from envs.pricing import black_scholes


# =============================================================================
# OptionsEnv — RL Environment for Options Trading
# =============================================================================
#
# OVERVIEW
# --------
# At the start of each episode, a full year of underlying prices (252 days) is
# simulated using GBM. This gives the agent historical context from day one.
# The agent then trades day by day, indefinitely (no episode length limit).
# Each day the agent can place up to MAX_TRADES trades before time advances.
#
# -----------------------------------------------------------------------------
# OBSERVATION SPACE — what the agent sees each day
# -----------------------------------------------------------------------------
#
# The observation is a flat float32 vector:
#
#   Market info:
#     [0]          : current spot price (float)
#     [1]          : current trading day index (int, starts at 0)
#     [2]          : realized annual vol over the full history so far (float)
#     [3 : 255]    : price history of the underlying, last 252 values (float x 252)
#     [255 : 255+H]: P&L history, one value per past trading day (float x H)
#
#   Portfolio info (up to MAX_PORTFOLIO positions, padded with 0):
#     per position: [option_type, strike, maturite_restante, quantite, is_short,
#                    entry_price, last_price, unrealized_pnl]
#     → shape: (MAX_PORTFOLIO x 8,)
#
#   Scalar constants visible to the agent:
#     - commission per contract : 0.65 (fixed, always visible)
#
# -----------------------------------------------------------------------------
# ACTION SPACE — what the agent can do each day
# -----------------------------------------------------------------------------
#
# The agent outputs MAX_TRADES trade slots. Each slot is a fixed-size vector:
#
#   [type_action, strike, maturite, quantite, close_index]
#
#   type_action : Discrete(6)
#     0 = do nothing          → all other fields ignored
#     1 = buy call            → uses strike (float), maturite (int), quantite (int)
#     2 = buy put             → uses strike (float), maturite (int), quantite (int)
#     3 = sell call           → uses strike (float), maturite (int), quantite (int)
#     4 = sell put            → uses strike (float), maturite (int), quantite (int)
#     5 = close position      → uses close_index (int), points to portfolio position
#
#   strike      : Box(-inf, inf, float32)  — desired strike price
#   maturite    : Discrete(252)            — days to expiry, clipped to days available
#   quantite    : Discrete(MAX_QTY)        — number of contracts (>= 1)
#   close_index : Discrete(MAX_PORTFOLIO)  — index of position to close
#
# Up to MAX_TRADES slots per step. Slots with type_action=0 are ignored.
# All trades in a step happen on the same day before time advances.
#
# -----------------------------------------------------------------------------
# REWARD
# -----------------------------------------------------------------------------
#
# Reward = total P&L, given every 84 trading days (quarterly).
# Reward = 0 on all other days.
# P&L includes: mark-to-market gains/losses + broker commissions (0.65/contract).
# Short positions are exercised immediately if ITM (American-style for shorts).
#
# =============================================================================

COMMISSION    = 0.65   # broker fee per contract
MAX_TRADES    = 15     # max trades per day
MAX_PORTFOLIO = 50     # max open positions at once
MAX_QTY       = 100    # max contracts per trade


class OptionsEnv(gym.Env):

    def __init__(self, S0: float = 100.0, daily_vol: float = 0.01, r: float = 0.0):
        super().__init__()

        self.S0        = S0
        self.daily_vol = daily_vol
        self.r         = r

        self.underlying    = Underlying(S0=S0, daily_vol=daily_vol)
        self.history_prices = []  # 252 prices simulated at reset
        self.live_prices    = []  # prices generated day by day during trading
        self.portfolio      = []
        self.pnl            = 0.0
        self.pnl_history    = []
        self.trading_day    = 0

        # ── Action space ──────────────────────────────────────────────────────
        # MAX_TRADES slots, each slot is a Dict
        single_trade = spaces.Dict({
            "type_action":  spaces.Discrete(6),
            "strike":       spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "maturite":     spaces.Discrete(252),
            "quantite":     spaces.Discrete(MAX_QTY),
            "close_index":  spaces.Discrete(MAX_PORTFOLIO),
        })
        self.action_space = spaces.Tuple(tuple(single_trade for _ in range(MAX_TRADES)))

        # ── Observation space ─────────────────────────────────────────────────
        # scalar info + price history + pnl history + portfolio
        obs_size = (
            3                      # spot, day, realized_vol
            + 252                  # price history (padded)
            + 252                  # pnl history (padded, one per trading day max shown)
            + MAX_PORTFOLIO * 8    # portfolio positions
            + 1                    # commission constant
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # simulate 1 full year of history upfront
        self.underlying = Underlying(S0=self.S0, daily_vol=self.daily_vol)
        self.underlying.simulate(251)
        self.history_prices = list(self.underlying.prices)  # 252 prices

        # trading starts from the last historical price
        self.live_prices = [self.history_prices[-1]]
        self.portfolio   = []
        self.pnl         = 0.0
        self.pnl_history = []
        self.trading_day = 0

        return self._get_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: tuple):
        # 1. Process all trade slots for today
        for trade in action:
            self._execute_trade(trade)

        # 2. Check early exercise of short positions
        self._check_early_exercise()

        # 3. Advance underlying by one day
        S    = self.live_prices[-1]
        Z    = np.random.standard_normal()
        S_new = S * np.exp(-0.5 * self.daily_vol ** 2 + self.daily_vol * Z)
        self.live_prices.append(S_new)
        self.trading_day += 1

        # 4. Mark to market
        self.pnl += self._mark_to_market()

        # 5. Remove expired options (T=0 → payoff already settled)
        self.portfolio = [p for p in self.portfolio if not p.get("expired")]
        for p in self.portfolio:
            p["maturite"] -= 1

        # 6. Record P&L
        self.pnl_history.append(self.pnl)

        # 7. Quarterly reward
        reward     = self.pnl if self.trading_day % 84 == 0 else 0.0
        terminated = False  # no fixed end — agent trades indefinitely
        truncated  = False

        return self._get_obs(), reward, terminated, truncated, {}

    # ── Trade execution ───────────────────────────────────────────────────────

    def _execute_trade(self, trade: dict):
        type_action = int(trade["type_action"])

        if type_action == 0:
            return  # do nothing

        if type_action == 5:
            # close a position by index
            idx = int(trade["close_index"])
            if idx < len(self.portfolio):
                self._close_position(idx)
            return

        # buy/sell call/put
        option_type_map = {1: "call", 2: "put", 3: "call", 4: "put"}
        is_short_map    = {1: False,  2: False, 3: True,   4: True}

        option_type = option_type_map[type_action]
        is_short    = is_short_map[type_action]

        strike   = float(trade["strike"][0])
        maturite = max(1, int(trade["maturite"]))
        quantite = max(1, int(trade["quantite"]))

        S     = self.live_prices[-1]
        sigma = self._realized_vol()
        T     = maturite / 252
        price = black_scholes(S=S, K=strike, T=T, r=self.r, sigma=sigma, option_type=option_type)

        # deduct commission
        self.pnl -= COMMISSION * quantite

        self.portfolio.append({
            "option_type": option_type,
            "strike":      strike,
            "maturite":    maturite,
            "quantite":    quantite,
            "is_short":    is_short,
            "entry_price": price,
            "last_price":  price,
            "expired":     False,
        })

    def _close_position(self, idx: int):
        p         = self.portfolio[idx]
        direction = -1 if p["is_short"] else 1
        # realize at last marked price
        self.pnl -= COMMISSION * p["quantite"]  # commission on close too
        self.portfolio.pop(idx)

    def _check_early_exercise(self):
        """Short positions are exercised immediately if ITM."""
        S = self.live_prices[-1]
        for p in self.portfolio:
            if not p["is_short"]:
                continue
            if p["option_type"] == "call" and S > p["strike"]:
                payoff   = (S - p["strike"]) * p["quantite"]
                self.pnl -= payoff - p["last_price"] * p["quantite"]
                p["expired"] = True
            elif p["option_type"] == "put" and S < p["strike"]:
                payoff   = (p["strike"] - S) * p["quantite"]
                self.pnl -= payoff - p["last_price"] * p["quantite"]
                p["expired"] = True
        self.portfolio = [p for p in self.portfolio if not p.get("expired")]

    def _mark_to_market(self) -> float:
        daily_pnl = 0.0
        S     = self.live_prices[-1]
        sigma = self._realized_vol()

        for p in self.portfolio:
            if p["maturite"] <= 0:
                # expiry payoff
                if p["option_type"] == "call":
                    payoff = max(S - p["strike"], 0.0)
                else:
                    payoff = max(p["strike"] - S, 0.0)
                direction  = -1 if p["is_short"] else 1
                daily_pnl += direction * (payoff - p["last_price"]) * p["quantite"]
                p["last_price"] = payoff
                p["expired"]    = True
            else:
                T         = p["maturite"] / 252
                new_price = black_scholes(S=S, K=p["strike"], T=T, r=self.r,
                                          sigma=sigma, option_type=p["option_type"])
                direction  = -1 if p["is_short"] else 1
                daily_pnl += direction * (new_price - p["last_price"]) * p["quantite"]
                p["last_price"] = new_price

        return daily_pnl

    # ── Observation ───────────────────────────────────────────────────────────

    def _realized_vol(self) -> float:
        all_p = self.history_prices + self.live_prices[1:]
        if len(all_p) < 2:
            return 0.01
        lr = np.diff(np.log(all_p))
        return max(float(np.std(lr)) * np.sqrt(252), 0.001)

    def get_option_price(self, strike: float, maturite: int, option_type: str) -> float:
        """Public method: agent can query any option price before acting."""
        S     = self.live_prices[-1]
        sigma = self._realized_vol()
        T     = max(maturite, 1) / 252
        return black_scholes(S=S, K=strike, T=T, r=self.r, sigma=sigma, option_type=option_type)

    def _get_obs(self) -> np.ndarray:
        S     = self.live_prices[-1]
        sigma = self._realized_vol()

        # price history padded to 252
        all_prices = self.history_prices + self.live_prices[1:]
        price_hist = np.zeros(252, dtype=np.float32)
        price_hist[-min(len(all_prices), 252):] = all_prices[-252:]

        # pnl history padded to 252
        pnl_hist = np.zeros(252, dtype=np.float32)
        pnl_hist[-min(len(self.pnl_history), 252):] = self.pnl_history[-252:]

        # portfolio padded to MAX_PORTFOLIO x 8
        port_vec = np.zeros((MAX_PORTFOLIO, 8), dtype=np.float32)
        for i, p in enumerate(self.portfolio[:MAX_PORTFOLIO]):
            unrealized = (-1 if p["is_short"] else 1) * (p["last_price"] - p["entry_price"]) * p["quantite"]
            port_vec[i] = [
                1 if p["option_type"] == "call" else 0,  # option type
                p["strike"],
                p["maturite"],
                p["quantite"],
                1 if p["is_short"] else 0,
                p["entry_price"],
                p["last_price"],
                unrealized,
            ]

        obs = np.concatenate([
            [S, float(self.trading_day), sigma],  # scalars
            price_hist,                            # price history
            pnl_hist,                              # pnl history
            port_vec.flatten(),                    # portfolio
            [COMMISSION],                          # commission constant
        ])
        return obs.astype(np.float32)