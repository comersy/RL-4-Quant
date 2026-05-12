import numpy as np


class Underlying:
    """
    Simulates an underlying asset price path using Geometric Brownian Motion.

    Parameters
    ----------
    S0        : starting price
    daily_vol : daily volatility (e.g. 0.01 for 1%)
    mu        : daily drift (default 0, i.e. pure random walk)
    """

    def __init__(self, S0: float, daily_vol: float, mu: float = 0.0):
        self.S0 = S0
        self.daily_vol = daily_vol
        self.mu = mu  # daily drift, 0 = pure random walk
        self.reset()

    def reset(self):
        self.prices = [self.S0]
        self.current_price = self.S0
        self.t = 0

    def step(self) -> float:
        # GBM: S_{t+1} = S_t * exp((mu - 0.5*sigma^2) + sigma*Z)
        Z = np.random.standard_normal()
        self.current_price *= np.exp((self.mu - 0.5 * self.daily_vol ** 2) + self.daily_vol * Z)
        self.prices.append(self.current_price)
        self.t += 1
        return self.current_price

    def simulate(self, n_days: int) -> np.ndarray:
        for _ in range(n_days):
            self.step()
        return np.array(self.prices)

    def annual_vol(self) -> float:
        return self.daily_vol * np.sqrt(252)

    def realized_vol(self) -> float:
        if len(self.prices) < 2:
            return 0.0
        log_returns = np.diff(np.log(self.prices))
        return float(np.std(log_returns))

    def __repr__(self):
        return f"Underlying(S0={self.S0}, daily_vol={self.daily_vol}, mu={self.mu}, t={self.t}, S={self.current_price:.4f})"