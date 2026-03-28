"""
Ornstein-Uhlenbeck process simulation and parameter estimation.

The OU process models mean-reverting spreads:
    ds = κ(μ - s)dt + σdW

where κ = mean-reversion speed, μ = long-term mean, σ = volatility.
"""
import numpy as np
import pandas as pd


def generate_ou_path(
    kappa: float,
    mu: float,
    sigma: float,
    s0: float,
    dt: float = 1 / 252,
    n_steps: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a sample path of an Ornstein-Uhlenbeck process using Euler-Maruyama.

    Args:
        kappa: Mean-reversion speed.
        mu: Long-term mean level.
        sigma: Volatility of the process.
        s0: Initial value of the spread.
        dt: Time step (default: 1/252 for daily).
        n_steps: Number of time steps to simulate.
        seed: Random seed for reproducibility.

    Returns:
        Array of length n_steps+1 containing the OU path (including s0).
    """
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rng = np.random.default_rng(seed)
    path = np.empty(n_steps + 1)
    path[0] = s0
    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        dW = rng.standard_normal()
        path[i + 1] = path[i] + kappa * (mu - path[i]) * dt + sigma * sqrt_dt * dW

    return path


def estimate_ou_params(spread: np.ndarray, dt: float = 1 / 252) -> dict:
    """
    Estimate OU process parameters from observed spread data using OLS regression.

    Regresses ds on s to recover κ, μ, σ from:
        s[t+1] - s[t] = κ(μ - s[t])dt + σ√dt ε

    Args:
        spread: Array of spread values.
        dt: Time step between observations.

    Returns:
        Dict with keys 'kappa', 'mu', 'sigma'.
    """
    s = spread[:-1]
    ds = np.diff(spread)

    # OLS: ds = a + b*s + residual => b = -κ*dt, a = κ*μ*dt
    n = len(s)
    s_mean = s.mean()
    ds_mean = ds.mean()

    b = np.sum((s - s_mean) * (ds - ds_mean)) / np.sum((s - s_mean) ** 2)
    a = ds_mean - b * s_mean

    kappa = max(-b / dt, 1e-8)  # ensure positive
    mu = a / (kappa * dt) if kappa * dt > 1e-12 else s_mean

    residuals = ds - (a + b * s)
    sigma = np.std(residuals, ddof=1) / np.sqrt(dt)

    return {"kappa": kappa, "mu": mu, "sigma": sigma}
