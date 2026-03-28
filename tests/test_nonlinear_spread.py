"""Tests for Phase 8: Alternative spread function evaluation."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.ou_process import generate_ou_path
from src.data_loader import DataLoader
from src.run_backtest import evaluate_spread_function, run_phase8, COST_SCENARIOS
from src.backtest import TransactionCostModel


@pytest.fixture
def mock_pair_data():
    """Create mock pair price data for two cointegrated assets."""
    np.random.seed(42)
    n = 1500
    dates = pd.bdate_range("2015-01-01", periods=n)
    log_p2 = np.cumsum(np.random.normal(0.0002, 0.01, n))
    beta = 1.2
    ou = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.3, s0=0.0, n_steps=n - 1, seed=42)
    log_p1 = beta * log_p2 + ou

    return pd.DataFrame(
        {"TICK1": np.exp(log_p1), "TICK2": np.exp(log_p2)},
        index=dates,
    )


@pytest.fixture
def loader():
    return DataLoader()


# --- Log Ratio Spread Tests ---


class TestLogRatioSpread:
    def test_output_columns(self, loader, mock_pair_data):
        result = loader.calculate_log_ratio_spread(mock_pair_data, "TICK1", "TICK2")
        assert "spread" in result.columns
        assert "log_price1" in result.columns
        assert "log_price2" in result.columns
        assert "beta" in result.columns

    def test_beta_is_one(self, loader, mock_pair_data):
        result = loader.calculate_log_ratio_spread(mock_pair_data, "TICK1", "TICK2")
        assert (result["beta"] == 1.0).all()

    def test_spread_equals_log_ratio(self, loader, mock_pair_data):
        result = loader.calculate_log_ratio_spread(mock_pair_data, "TICK1", "TICK2")
        expected = np.log(mock_pair_data["TICK1"]) - np.log(mock_pair_data["TICK2"])
        np.testing.assert_allclose(result["spread"].values, expected.values, atol=1e-10)

    def test_same_length_as_input(self, loader, mock_pair_data):
        result = loader.calculate_log_ratio_spread(mock_pair_data, "TICK1", "TICK2")
        assert len(result) == len(mock_pair_data)


# --- Bounded (Logistic) Spread Tests ---


class TestBoundedSpread:
    def test_output_bounded(self, loader, mock_pair_data):
        result = loader.calculate_bounded_spread(mock_pair_data, "TICK1", "TICK2", alpha=10.0)
        spread = result["spread"].values
        assert np.all(spread > -1.0)
        assert np.all(spread < 1.0)

    def test_different_alpha_values(self, loader, mock_pair_data):
        r1 = loader.calculate_bounded_spread(mock_pair_data, "TICK1", "TICK2", alpha=5.0)
        r2 = loader.calculate_bounded_spread(mock_pair_data, "TICK1", "TICK2", alpha=20.0)
        # Higher alpha should produce more extreme (closer to -1/+1) values
        assert r2["spread"].std() >= r1["spread"].std()

    def test_zero_alpha_gives_zero_spread(self, loader, mock_pair_data):
        result = loader.calculate_bounded_spread(mock_pair_data, "TICK1", "TICK2", alpha=0.0)
        # logistic(0) = 0.5, so 2*0.5 - 1 = 0
        np.testing.assert_allclose(result["spread"].values, 0.0, atol=1e-10)

    def test_spread_is_centered(self, loader, mock_pair_data):
        result = loader.calculate_bounded_spread(mock_pair_data, "TICK1", "TICK2", alpha=10.0)
        # The standardization should center the spread near 0
        assert abs(result["spread"].mean()) < 0.3

    def test_same_length_as_input(self, loader, mock_pair_data):
        result = loader.calculate_bounded_spread(mock_pair_data, "TICK1", "TICK2")
        assert len(result) == len(mock_pair_data)


# --- Power Spread Tests ---


class TestPowerSpread:
    def test_sign_preserved(self, loader, mock_pair_data):
        linear = loader.calculate_spread(mock_pair_data, "TICK1", "TICK2")
        power = loader.calculate_power_spread(mock_pair_data, "TICK1", "TICK2", power=0.5)
        # Sign should be preserved
        linear_sign = np.sign(linear["spread"].values)
        power_sign = np.sign(power["spread"].values)
        np.testing.assert_array_equal(linear_sign, power_sign)

    def test_power_one_equals_linear(self, loader, mock_pair_data):
        linear = loader.calculate_spread(mock_pair_data, "TICK1", "TICK2")
        power1 = loader.calculate_power_spread(mock_pair_data, "TICK1", "TICK2", power=1.0)
        np.testing.assert_allclose(
            power1["spread"].values, linear["spread"].values, atol=1e-10
        )

    def test_sub_linear_compresses_large_deviations(self, loader, mock_pair_data):
        linear = loader.calculate_spread(mock_pair_data, "TICK1", "TICK2")
        power05 = loader.calculate_power_spread(mock_pair_data, "TICK1", "TICK2", power=0.5)
        # For values where |z| > 1, p=0.5 compresses; for |z| < 1, it amplifies.
        # The max absolute deviation should be compressed relative to the linear case.
        linear_max = np.abs(linear["spread"].values).max()
        power_max = np.abs(power05["spread"].values).max()
        if linear_max > 1.0:
            assert power_max < linear_max

    def test_super_linear_amplifies(self, loader, mock_pair_data):
        linear = loader.calculate_spread(mock_pair_data, "TICK1", "TICK2")
        power2 = loader.calculate_power_spread(mock_pair_data, "TICK1", "TICK2", power=2.0)
        # Super-linear (p>1) should have different std
        assert power2["spread"].std() != linear["spread"].std()

    def test_same_length_as_input(self, loader, mock_pair_data):
        result = loader.calculate_power_spread(mock_pair_data, "TICK1", "TICK2", power=0.5)
        assert len(result) == len(mock_pair_data)


# --- evaluate_spread_function Tests ---


class TestEvaluateSpreadFunction:
    def test_returns_expected_keys(self, loader, mock_pair_data):
        spread_df = loader.calculate_spread(mock_pair_data, "TICK1", "TICK2")
        result = evaluate_spread_function(
            "test_linear", spread_df,
            ou_window=252, k=0.25, kappa_threshold=1.0,
            n_splits=3,
        )
        assert "spreadName" in result
        assert "fullBacktest" in result
        assert "walkForward" in result
        assert "rollingParams" in result
        assert result["spreadName"] == "test_linear"

    def test_walk_forward_has_correct_windows(self, loader, mock_pair_data):
        spread_df = loader.calculate_spread(mock_pair_data, "TICK1", "TICK2")
        result = evaluate_spread_function(
            "test", spread_df,
            ou_window=252, k=0.25, kappa_threshold=1.0,
            n_splits=3,
        )
        assert result["walkForward"]["windows"] <= 3
        assert result["walkForward"]["positiveWindows"] <= result["walkForward"]["windows"]

    def test_full_backtest_metrics(self, loader, mock_pair_data):
        spread_df = loader.calculate_spread(mock_pair_data, "TICK1", "TICK2")
        result = evaluate_spread_function(
            "test", spread_df,
            ou_window=252, k=0.25, kappa_threshold=1.0,
            n_splits=3,
        )
        fb = result["fullBacktest"]
        assert "grossSharpe" in fb
        assert "netSharpe" in fb
        assert "annualReturn" in fb
        assert "maxDrawdown" in fb
        assert fb["maxDrawdown"] <= 0  # drawdown is always negative

    def test_different_spreads_different_results(self, loader, mock_pair_data):
        linear_df = loader.calculate_spread(mock_pair_data, "TICK1", "TICK2")
        ratio_df = loader.calculate_log_ratio_spread(mock_pair_data, "TICK1", "TICK2")

        r1 = evaluate_spread_function("linear", linear_df, n_splits=3)
        r2 = evaluate_spread_function("ratio", ratio_df, n_splits=3)

        # Different spread definitions should produce different results
        assert r1["fullBacktest"]["grossSharpe"] != r2["fullBacktest"]["grossSharpe"]


# --- run_phase8 Integration Tests ---


class TestRunPhase8:
    def _make_mock_loader(self, mock_pair_data):
        mock_loader = MagicMock()
        mock_loader.download_pair_data.return_value = mock_pair_data

        real_loader = DataLoader()

        def mock_calc_spread(pair_data, t1, t2):
            return real_loader.calculate_spread(pair_data, "TICK1", "TICK2")

        def mock_log_ratio(pair_data, t1, t2):
            return real_loader.calculate_log_ratio_spread(pair_data, "TICK1", "TICK2")

        def mock_bounded(pair_data, t1, t2, alpha=10.0):
            return real_loader.calculate_bounded_spread(pair_data, "TICK1", "TICK2", alpha=alpha)

        def mock_power(pair_data, t1, t2, power=0.5):
            return real_loader.calculate_power_spread(pair_data, "TICK1", "TICK2", power=power)

        mock_loader.calculate_spread.side_effect = mock_calc_spread
        mock_loader.calculate_log_ratio_spread.side_effect = mock_log_ratio
        mock_loader.calculate_bounded_spread.side_effect = mock_bounded
        mock_loader.calculate_power_spread.side_effect = mock_power
        return mock_loader

    @patch("src.run_backtest.DataLoader")
    def test_phase8_returns_metrics_json(self, MockDataLoader, mock_pair_data):
        MockDataLoader.return_value = self._make_mock_loader(mock_pair_data)
        output = run_phase8()
        assert "metrics_json" in output
        metrics = output["metrics_json"]
        assert "sharpeRatio" in metrics
        assert "walkForward" in metrics
        assert "customMetrics" in metrics

    @patch("src.run_backtest.DataLoader")
    def test_phase8_evaluates_all_spreads(self, MockDataLoader, mock_pair_data):
        MockDataLoader.return_value = self._make_mock_loader(mock_pair_data)
        output = run_phase8()
        cm = output["metrics_json"]["customMetrics"]
        assert cm["spreadFunctionsEvaluated"] == 6
        assert "spreadResults" in cm
        assert "linear" in cm["spreadResults"]
        assert "log_ratio" in cm["spreadResults"]
        assert "bounded_a10" in cm["spreadResults"]
        assert "power_p05" in cm["spreadResults"]

    @patch("src.run_backtest.DataLoader")
    def test_phase8_has_ranking(self, MockDataLoader, mock_pair_data):
        MockDataLoader.return_value = self._make_mock_loader(mock_pair_data)
        output = run_phase8()
        cm = output["metrics_json"]["customMetrics"]
        assert "ranking" in cm
        assert "bestSpread" in cm["ranking"]
        assert "worstSpread" in cm["ranking"]
        assert "byAvgOosSharpe" in cm["ranking"]
