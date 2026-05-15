import numpy as np
from scipy.stats import norm


def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")