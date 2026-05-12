import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.underlying import Underlying
from envs.pricing import black_scholes


class OptionsEnv(gym.Env):
    """
    RL environment for options trading.

    The agent trades options on a simulated underlying (GBM) over 252 days.
    At each step the agent can buy/sell a call or put, or do nothing.
    The reward is the daily P&L of the portfolio.

    Action space (Dict):
        do_nothing  : Discrete(2)         – 1 = skip the day
        type_option : Discrete(4)         – 0=buy call, 1=buy put, 2=sell call, 3=sell put
        strike      : Box(-inf, inf)      – strike price (continuous)
        maturite    : Discrete(252)       – days to expiry (1 to days remaining)
        quantite    : Discrete(2*Q+1)     – signed integer quantity, centered at 0

    Observation space:
        - current underlying price
        - realized vol (annualized)
        - days remaining in episode
        - current portfolio P&L
    """

    MAX_QUANTITY = 10  # agent can trade between -MAX_QUANTITY and +MAX_QUANTITY

    def __init__(self, S0: float = 100.0, daily_vol: float = 0.01, r: float = 0.0):
        super().__init__()

        self.S0 = S0
        self.daily_vol = daily_vol
        self.r = r  # risk-free rate, fixed for the whole environment

        self.underlying = Underlying(S0=S0, daily_vol=daily_vol)

        # --- Action space ---
        self.action_space = spaces.Dict({
            "do_nothing":  spaces.Discrete(2),
            "type_option": spaces.Discrete(4),   # 0=buy call, 1=buy put, 2=sell call, 3=sell put
            "strike":      spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "maturite":    spaces.Discrete(252),  # clipped to days remaining at step time
            "quantite":    spaces.Discrete(2 * self.MAX_QUANTITY + 1),  # centered: 0 = flat
        })

        # --- Observation space ---
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, 252.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        self.portfolio: list[dict] = []  # list of open option positions
        self.pnl: float = 0.0
        self.day: int = 0

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.underlying.reset()
        self.portfolio = []
        self.pnl = 0.0
        self.day = 0
        return self._get_obs(), {}

    def step(self, action: dict):
        # 1. Advance the underlying by one day
        self.underlying.step()
        self.day += 1
        days_remaining = 252 - self.day

        # 2. Price all open positions before and after the step, compute daily P&L
        daily_pnl = self._mark_to_market()
        self.pnl += daily_pnl

        # 3. Process action unless do_nothing
        if action["do_nothing"] == 0 and days_remaining > 0:
            self._execute_action(action, days_remaining)

        # 4. Remove expired options
        self.portfolio = [p for p in self.portfolio if p["maturite"] > 0]

        # 5. Decrease remaining maturity of each position by 1 day
        for p in self.portfolio:
            p["maturite"] -= 1

        terminated = self.day >= 252
        return self._get_obs(), daily_pnl, terminated, False, {}

    # ------------------------------------------------------------------

    def _execute_action(self, action: dict, days_remaining: int):
        option_type_map = {0: "call", 1: "put", 2: "call", 3: "put"}
        is_short_map    = {0: False,   1: False,  2: True,   3: True}

        type_idx    = int(action["type_option"])
        option_type = option_type_map[type_idx]
        is_short    = is_short_map[type_idx]

        strike   = float(action["strike"][0])
        maturite = int(np.clip(action["maturite"], 1, days_remaining))  # can't exceed days remaining
        quantite = int(action["quantite"]) - self.MAX_QUANTITY           # re-center around 0

        if quantite == 0:
            return  # effectively do nothing

        S     = self.underlying.current_price
        sigma = self.underlying.realized_vol() * np.sqrt(252)  # annualize
        T     = maturite / 252                                  # convert days to years
        price = black_scholes(S=S, K=strike, T=T, r=self.r, sigma=sigma, option_type=option_type)

        self.portfolio.append({
            "option_type": option_type,
            "strike":      strike,
            "maturite":    maturite,  # days remaining
            "quantite":    quantite,
            "is_short":    is_short,
            "entry_price": price,
            "last_price":  price,
        })

    def _mark_to_market(self) -> float:
        """Reprice all positions and return the daily P&L."""
        daily_pnl = 0.0
        S     = self.underlying.current_price
        sigma = self.underlying.realized_vol() * np.sqrt(252)

        for p in self.portfolio:
            T = p["maturite"] / 252
            new_price = black_scholes(S=S, K=p["strike"], T=T, r=self.r, sigma=sigma, option_type=p["option_type"])
            price_change = new_price - p["last_price"]

            # short positions lose when price goes up
            daily_pnl += (-1 if p["is_short"] else 1) * price_change * p["quantite"]
            p["last_price"] = new_price

        return daily_pnl

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.underlying.current_price,
            self.underlying.realized_vol() * np.sqrt(252),
            float(252 - self.day),
            self.pnl,
        ], dtype=np.float32)