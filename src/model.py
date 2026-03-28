"""
Control-theoretic pairs trading model.

Implements the feedback control law from the paper:
    h(t) = -k * (s(t) - μ)

where h is the investment allocation, k is the control gain,
s is the current spread, and μ is the long-term mean.
"""
import numpy as np
import pandas as pd
from src.ou_process import estimate_ou_params


class ControlTrader:
    """
    Pairs trading strategy using feedback control on a mean-reverting spread.

    The control law allocates inversely proportional to the spread deviation:
    when the spread is above its mean, short the spread; when below, go long.

    Args:
        kappa: OU mean-reversion speed.
        mu: OU long-term mean.
        sigma: OU volatility.
        k: Control gain (scaling factor for allocation).
    """

    def __init__(self, kappa: float, mu: float, sigma: float, k: float = 1.0):
        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma
        self.k = k

    def get_allocation(self, spread_value: float) -> float:
        """
        Compute the allocation based on current spread value.

        h = -k * (spread_value - mu)

        Positive h => long the spread (spread is below mean).
        Negative h => short the spread (spread is above mean).

        Args:
            spread_value: Current value of the spread.

        Returns:
            Allocation signal h.
        """
        return -self.k * (spread_value - self.mu)

    def run_backtest(self, spread: np.ndarray, dt: float = 1 / 252) -> dict:
        """
        Run a simple backtest on a spread time series.

        Portfolio return at each step:
            r[t] = h[t] * ds[t]

        where ds[t] = s[t+1] - s[t] and h[t] = get_allocation(s[t]).

        Args:
            spread: Array of spread values.
            dt: Time step.

        Returns:
            Dict with 'allocations', 'returns', 'cumulative_pnl', 'portfolio_value'.
        """
        n = len(spread) - 1
        allocations = np.array([self.get_allocation(spread[i]) for i in range(n)])
        ds = np.diff(spread)
        pnl = allocations * ds

        cumulative_pnl = np.cumsum(pnl)
        # Portfolio value starting at 1.0
        portfolio_value = 1.0 + cumulative_pnl

        return {
            "allocations": allocations,
            "returns": pnl,
            "cumulative_pnl": cumulative_pnl,
            "portfolio_value": portfolio_value,
        }

    @classmethod
    def from_data(cls, spread: np.ndarray, k: float = 1.0, dt: float = 1 / 252) -> "ControlTrader":
        """
        Create a ControlTrader by estimating OU parameters from data.

        Args:
            spread: Historical spread values.
            k: Control gain.
            dt: Time step.

        Returns:
            ControlTrader instance with estimated parameters.
        """
        params = estimate_ou_params(spread, dt)
        return cls(kappa=params["kappa"], mu=params["mu"], sigma=params["sigma"], k=k)
