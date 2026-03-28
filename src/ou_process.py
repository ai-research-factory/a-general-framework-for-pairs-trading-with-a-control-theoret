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


def estimate_ou_params_rolling(
    spread: np.ndarray,
    window: int = 252,
    dt: float = 1 / 252,
) -> pd.DataFrame:
    """
    Estimate OU parameters on a rolling window basis.

    For each time step t >= window, estimates κ, μ, σ using the preceding
    `window` observations of the spread.

    Args:
        spread: Array of spread values.
        window: Lookback window size (default: 252 ~ 1 year).
        dt: Time step between observations.

    Returns:
        DataFrame with columns ['kappa', 'mu', 'sigma'] indexed from
        position `window` to `len(spread)-1`. Values at index i are
        estimated from spread[i-window:i].
    """
    if window < 10:
        raise ValueError("window must be >= 10 for meaningful estimation")
    n = len(spread)
    if n <= window:
        raise ValueError(f"spread length ({n}) must exceed window ({window})")

    kappas = np.empty(n - window)
    mus = np.empty(n - window)
    sigmas = np.empty(n - window)

    for i in range(window, n):
        params = estimate_ou_params(spread[i - window : i], dt=dt)
        kappas[i - window] = params["kappa"]
        mus[i - window] = params["mu"]
        sigmas[i - window] = params["sigma"]

    return pd.DataFrame(
        {"kappa": kappas, "mu": mus, "sigma": sigmas},
        index=np.arange(window, n),
    )
