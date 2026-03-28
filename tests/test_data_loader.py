"""Tests for DataLoader and data pipeline."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.data_loader import DataLoader


@pytest.fixture
def sample_pair_data():
    """Create sample pair price data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    p1 = 20.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    p2 = 30.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    return pd.DataFrame({"EWA": p1, "EWC": p2}, index=dates)


class TestCalculateSpread:
    def test_spread_has_no_nan(self, sample_pair_data):
        loader = DataLoader()
        result = loader.calculate_spread(sample_pair_data, "EWA", "EWC")
        assert not result.isna().any().any()

    def test_spread_columns_exist(self, sample_pair_data):
        loader = DataLoader()
        result = loader.calculate_spread(sample_pair_data, "EWA", "EWC")
        assert "spread" in result.columns
        assert "beta" in result.columns
        assert "log_price1" in result.columns
        assert "log_price2" in result.columns

    def test_spread_length_matches_input(self, sample_pair_data):
        loader = DataLoader()
        result = loader.calculate_spread(sample_pair_data, "EWA", "EWC")
        assert len(result) == len(sample_pair_data)

    def test_beta_is_scalar(self, sample_pair_data):
        loader = DataLoader()
        result = loader.calculate_spread(sample_pair_data, "EWA", "EWC")
        beta_values = result["beta"].unique()
        assert len(beta_values) == 1

    def test_spread_formula(self, sample_pair_data):
        """Verify spread = log(p1) - beta * log(p2)."""
        loader = DataLoader()
        result = loader.calculate_spread(sample_pair_data, "EWA", "EWC")
        beta = result["beta"].iloc[0]
        expected = np.log(sample_pair_data["EWA"]) - beta * np.log(sample_pair_data["EWC"])
        pd.testing.assert_series_equal(result["spread"], expected, check_names=False)

    def test_spread_is_mean_reverting_for_cointegrated_pair(self):
        """For truly cointegrated prices, spread should have limited variance."""
        np.random.seed(123)
        n = 1000
        # Generate cointegrated pair: P2 random walk, P1 = 0.8*P2 + stationary noise
        log_p2 = np.cumsum(np.random.normal(0, 0.01, n)) + np.log(30)
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = 0.9 * noise[i - 1] + np.random.normal(0, 0.005)
        log_p1 = 0.8 * log_p2 + noise + np.log(20)

        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        pair = pd.DataFrame({"A": np.exp(log_p1), "B": np.exp(log_p2)}, index=dates)

        loader = DataLoader()
        result = loader.calculate_spread(pair, "A", "B")
        # Beta should be close to 0.8
        assert abs(result["beta"].iloc[0] - 0.8) < 0.1
        # Spread std should be small (mean-reverting)
        assert result["spread"].std() < 0.1


class TestCalculateRollingSpread:
    @pytest.fixture
    def cointegrated_pair(self):
        """Create a cointegrated pair with 500 data points."""
        np.random.seed(42)
        n = 500
        log_p2 = np.cumsum(np.random.normal(0, 0.01, n)) + np.log(30)
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = 0.9 * noise[i - 1] + np.random.normal(0, 0.005)
        log_p1 = 0.8 * log_p2 + noise + np.log(20)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({"A": np.exp(log_p1), "B": np.exp(log_p2)}, index=dates)

    def test_rolling_spread_length(self, cointegrated_pair):
        loader = DataLoader()
        result = loader.calculate_rolling_spread(cointegrated_pair, "A", "B", window=100)
        assert len(result) == 500 - 100

    def test_rolling_spread_no_nan(self, cointegrated_pair):
        loader = DataLoader()
        result = loader.calculate_rolling_spread(cointegrated_pair, "A", "B", window=100)
        assert not result["spread"].isna().any()
        assert not result["beta"].isna().any()

    def test_rolling_spread_columns(self, cointegrated_pair):
        loader = DataLoader()
        result = loader.calculate_rolling_spread(cointegrated_pair, "A", "B", window=100)
        for col in ["spread", "beta", "log_price1", "log_price2"]:
            assert col in result.columns

    def test_rolling_beta_varies(self, cointegrated_pair):
        """Rolling beta should vary over time (unlike static)."""
        loader = DataLoader()
        result = loader.calculate_rolling_spread(cointegrated_pair, "A", "B", window=100)
        assert result["beta"].nunique() > 1

    def test_rolling_spread_formula(self, cointegrated_pair):
        """spread[i] = log(p1[i]) - beta[i] * log(p2[i])."""
        loader = DataLoader()
        result = loader.calculate_rolling_spread(cointegrated_pair, "A", "B", window=100)
        expected = result["log_price1"] - result["beta"] * result["log_price2"]
        pd.testing.assert_series_equal(result["spread"], expected, check_names=False)
