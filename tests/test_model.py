"""Tests for ControlTrader and OU process."""
import numpy as np
import pytest
from src.ou_process import generate_ou_path, estimate_ou_params, estimate_ou_params_rolling
from src.model import ControlTrader, run_rolling_backtest


class TestGenerateOUPath:
    def test_output_length(self):
        path = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=1.0, n_steps=100, seed=42)
        assert len(path) == 101

    def test_starts_at_s0(self):
        path = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=2.5, n_steps=50, seed=42)
        assert path[0] == 2.5

    def test_reproducible_with_seed(self):
        p1 = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=0.0, n_steps=100, seed=42)
        p2 = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=0.0, n_steps=100, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_mean_reversion(self):
        """With strong mean-reversion, the path should stay near mu."""
        path = generate_ou_path(kappa=50.0, mu=1.0, sigma=0.1, s0=1.0, n_steps=5000, seed=42)
        assert abs(path[1000:].mean() - 1.0) < 0.1

    def test_negative_kappa_raises(self):
        with pytest.raises(ValueError, match="kappa"):
            generate_ou_path(kappa=-1.0, mu=0.0, sigma=0.5, s0=0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            generate_ou_path(kappa=1.0, mu=0.0, sigma=-0.5, s0=0.0)


class TestEstimateOUParams:
    def test_recovers_parameters(self):
        """Estimated params should be close to true params for a long path."""
        path = generate_ou_path(kappa=5.0, mu=1.0, sigma=0.5, s0=1.0, n_steps=10000, seed=42)
        params = estimate_ou_params(path)
        assert abs(params["kappa"] - 5.0) < 2.0
        assert abs(params["mu"] - 1.0) < 0.2
        assert abs(params["sigma"] - 0.5) < 0.2


class TestControlTrader:
    def test_allocation_at_mean(self):
        """At the mean, allocation should be zero."""
        trader = ControlTrader(kappa=5.0, mu=1.0, sigma=0.5, k=1.0)
        assert trader.get_allocation(1.0) == 0.0

    def test_allocation_above_mean(self):
        """Above mean => negative allocation (short spread)."""
        trader = ControlTrader(kappa=5.0, mu=1.0, sigma=0.5, k=1.0)
        assert trader.get_allocation(2.0) < 0.0

    def test_allocation_below_mean(self):
        """Below mean => positive allocation (long spread)."""
        trader = ControlTrader(kappa=5.0, mu=1.0, sigma=0.5, k=1.0)
        assert trader.get_allocation(0.0) > 0.0

    def test_allocation_linear_in_k(self):
        """Allocation scales linearly with control gain k."""
        t1 = ControlTrader(kappa=5.0, mu=0.0, sigma=0.5, k=1.0)
        t2 = ControlTrader(kappa=5.0, mu=0.0, sigma=0.5, k=2.0)
        assert t2.get_allocation(1.0) == 2.0 * t1.get_allocation(1.0)

    def test_allocation_formula(self):
        """h = -k * (spread - mu)."""
        trader = ControlTrader(kappa=5.0, mu=3.0, sigma=0.5, k=2.0)
        assert trader.get_allocation(5.0) == pytest.approx(-2.0 * (5.0 - 3.0))

    def test_backtest_output_shape(self):
        spread = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        trader = ControlTrader(kappa=5.0, mu=1.0, sigma=0.5, k=1.0)
        result = trader.run_backtest(spread)
        assert len(result["allocations"]) == 4
        assert len(result["returns"]) == 4
        assert len(result["portfolio_value"]) == 4

    def test_backtest_portfolio_starts_near_one(self):
        spread = np.array([1.0, 1.0, 1.0])  # no movement
        trader = ControlTrader(kappa=5.0, mu=1.0, sigma=0.5, k=1.0)
        result = trader.run_backtest(spread)
        np.testing.assert_array_almost_equal(result["portfolio_value"], [1.0, 1.0])

    def test_from_data(self):
        spread = generate_ou_path(kappa=5.0, mu=1.0, sigma=0.5, s0=1.0, n_steps=1000, seed=42)
        trader = ControlTrader.from_data(spread, k=0.5)
        assert trader.k == 0.5
        assert abs(trader.mu - 1.0) < 0.5

    def test_mean_reverting_spread_positive_pnl(self):
        """On a mean-reverting spread, the strategy should produce positive PnL."""
        spread = generate_ou_path(kappa=10.0, mu=0.0, sigma=0.3, s0=0.0, n_steps=5000, seed=42)
        trader = ControlTrader(kappa=10.0, mu=0.0, sigma=0.3, k=1.0)
        result = trader.run_backtest(spread)
        assert result["cumulative_pnl"][-1] > 0


class TestEstimateOUParamsRolling:
    def test_output_shape(self):
        """Rolling output should have n - window rows."""
        path = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=0.0, n_steps=500, seed=42)
        result = estimate_ou_params_rolling(path, window=100)
        assert len(result) == 500 + 1 - 100  # n_steps+1 - window = 401
        assert set(result.columns) == {"kappa", "mu", "sigma"}

    def test_index_starts_at_window(self):
        path = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=0.0, n_steps=500, seed=42)
        result = estimate_ou_params_rolling(path, window=100)
        assert result.index[0] == 100
        assert result.index[-1] == 500

    def test_recovers_stable_parameters(self):
        """On a long OU path, rolling estimates should cluster near true values."""
        path = generate_ou_path(kappa=5.0, mu=1.0, sigma=0.5, s0=1.0, n_steps=3000, seed=42)
        result = estimate_ou_params_rolling(path, window=500)
        # Mean of rolling estimates should be near true values
        assert abs(result["kappa"].mean() - 5.0) < 3.0
        assert abs(result["mu"].mean() - 1.0) < 0.3
        assert abs(result["sigma"].mean() - 0.5) < 0.2

    def test_small_window_raises(self):
        path = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=0.0, n_steps=100, seed=42)
        with pytest.raises(ValueError, match="window must be >= 10"):
            estimate_ou_params_rolling(path, window=5)

    def test_short_spread_raises(self):
        path = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=0.0, n_steps=50, seed=42)
        with pytest.raises(ValueError, match="spread length"):
            estimate_ou_params_rolling(path, window=100)

    def test_all_kappas_positive(self):
        path = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=0.0, n_steps=500, seed=42)
        result = estimate_ou_params_rolling(path, window=100)
        assert (result["kappa"] > 0).all()

    def test_all_sigmas_positive(self):
        path = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=0.0, n_steps=500, seed=42)
        result = estimate_ou_params_rolling(path, window=100)
        assert (result["sigma"] > 0).all()


class TestRunRollingBacktest:
    def test_output_keys(self):
        spread = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=0.0, n_steps=500, seed=42)
        result = run_rolling_backtest(spread, ou_window=100, k=1.0)
        assert "allocations" in result
        assert "returns" in result
        assert "pnl" in result
        assert "portfolio_value" in result
        assert "rolling_params" in result

    def test_output_lengths(self):
        spread = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=0.0, n_steps=500, seed=42)
        result = run_rolling_backtest(spread, ou_window=100, k=1.0)
        expected_len = 501 - 100 - 1  # n - window - 1 = 400
        assert len(result["allocations"]) == expected_len
        assert len(result["returns"]) == expected_len
        assert len(result["portfolio_value"]) == expected_len

    def test_positive_pnl_on_mean_reverting_spread(self):
        """Strategy with rolling params should profit on a strongly mean-reverting spread."""
        spread = generate_ou_path(kappa=10.0, mu=0.0, sigma=0.3, s0=0.0, n_steps=3000, seed=42)
        result = run_rolling_backtest(spread, ou_window=252, k=1.0)
        assert result["portfolio_value"][-1] > 1.0

    def test_reduced_allocation_when_not_mean_reverting(self):
        """When spread is a random walk, estimated kappa should be lower than for OU."""
        rng = np.random.default_rng(42)
        # Compare: random walk vs strongly mean-reverting OU
        rw_spread = np.cumsum(rng.standard_normal(600)) * 0.01
        ou_spread = generate_ou_path(kappa=10.0, mu=0.0, sigma=0.3, s0=0.0, n_steps=599, seed=42)
        rw_result = run_rolling_backtest(rw_spread, ou_window=100, k=1.0)
        ou_result = run_rolling_backtest(ou_spread, ou_window=100, k=1.0)
        # OU should have higher kappa estimates on average
        assert ou_result["rolling_params"]["kappa"].mean() > rw_result["rolling_params"]["kappa"].mean()
