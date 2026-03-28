"""Tests for Phase 7 multi-pair robustness testing."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.ou_process import generate_ou_path
from src.run_backtest import evaluate_pair, run_phase7, COST_SCENARIOS
from src.backtest import TransactionCostModel


@pytest.fixture
def mock_pair_data():
    """Create mock pair price data for two cointegrated assets."""
    np.random.seed(42)
    n = 1500
    dates = pd.bdate_range("2015-01-01", periods=n)
    # Asset 2 follows random walk
    log_p2 = np.cumsum(np.random.normal(0.0002, 0.01, n))
    # Asset 1 = beta * asset2 + mean-reverting residual
    beta = 1.2
    ou = generate_ou_path(kappa=5.0, mu=0.0, sigma=0.3, s0=0.0, n_steps=n - 1, seed=42)
    log_p1 = beta * log_p2 + ou

    return pd.DataFrame(
        {"TICK1": np.exp(log_p1), "TICK2": np.exp(log_p2)},
        index=dates,
    )


@pytest.fixture
def mock_spread_df(mock_pair_data):
    """Create spread DataFrame from mock pair data."""
    from src.data_loader import DataLoader
    loader = DataLoader()
    return loader.calculate_spread(mock_pair_data, "TICK1", "TICK2")


def _make_mock_loader(mock_pair_data):
    """Create a mock DataLoader that returns mock_pair_data for any pair."""
    mock_loader = MagicMock()
    mock_loader.download_pair_data.return_value = mock_pair_data

    from src.data_loader import DataLoader
    real_loader = DataLoader()

    def mock_calc_spread(pair_data, t1, t2):
        return real_loader.calculate_spread(pair_data, "TICK1", "TICK2")

    mock_loader.calculate_spread.side_effect = mock_calc_spread
    return mock_loader


class TestEvaluatePair:
    def test_returns_expected_keys(self, mock_pair_data):
        """evaluate_pair returns all expected top-level keys."""
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = evaluate_pair("TICK1", "TICK2", ou_window=126, k=0.25,
                                   kappa_threshold=1.0, n_splits=3)

        expected_keys = {"pair", "ticker1", "ticker2", "dataPoints", "dateRange",
                         "hedgeRatio", "fullBacktest", "walkForward",
                         "rollingParams", "wf_results_raw", "description"}
        # description is added by run_phase7, not by evaluate_pair directly
        actual_keys = set(result.keys())
        assert actual_keys >= (expected_keys - {"description"})

    def test_full_backtest_metrics(self, mock_pair_data):
        """Full backtest contains all required metric fields."""
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = evaluate_pair("TICK1", "TICK2", ou_window=126, k=0.25,
                                   kappa_threshold=1.0, n_splits=3)

        fb = result["fullBacktest"]
        assert "grossSharpe" in fb
        assert "netSharpe" in fb
        assert "annualReturn" in fb
        assert "maxDrawdown" in fb
        assert "hitRate" in fb
        assert "totalCosts" in fb
        assert "costReturnRatio" in fb
        assert "turnover" in fb

    def test_net_sharpe_leq_gross(self, mock_pair_data):
        """Net Sharpe should be at most equal to gross Sharpe."""
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = evaluate_pair("TICK1", "TICK2", ou_window=126, k=0.25,
                                   kappa_threshold=1.0, n_splits=3)

        fb = result["fullBacktest"]
        assert fb["netSharpe"] <= fb["grossSharpe"] + 0.01

    def test_walk_forward_structure(self, mock_pair_data):
        """Walk-forward results contain correct structure."""
        n_splits = 3
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = evaluate_pair("TICK1", "TICK2", ou_window=126, k=0.25,
                                   kappa_threshold=1.0, n_splits=n_splits)

        wf = result["walkForward"]
        assert "windows" in wf
        assert "positiveWindows" in wf
        assert "avgOosSharpe" in wf
        assert "windowDetails" in wf
        assert wf["positiveWindows"] <= wf["windows"]

    def test_rolling_params_structure(self, mock_pair_data):
        """Rolling parameter stats are present and valid."""
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = evaluate_pair("TICK1", "TICK2", ou_window=126, k=0.25,
                                   kappa_threshold=1.0, n_splits=3)

        rp = result["rollingParams"]
        assert "activeDaysPct" in rp
        assert 0 <= rp["activeDaysPct"] <= 100
        for param in ["kappa", "mu", "sigma"]:
            assert "mean" in rp[param]
            assert "std" in rp[param]

    def test_pair_name_correct(self, mock_pair_data):
        """Pair name matches input tickers."""
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = evaluate_pair("TICK1", "TICK2", ou_window=126, k=0.25,
                                   kappa_threshold=1.0, n_splits=3)

        assert result["pair"] == "TICK1/TICK2"
        assert result["ticker1"] == "TICK1"
        assert result["ticker2"] == "TICK2"

    def test_positive_sharpe_on_cointegrated_pair(self, mock_pair_data):
        """Strategy should produce positive gross Sharpe on cointegrated data."""
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = evaluate_pair("TICK1", "TICK2", ou_window=126, k=0.5,
                                   kappa_threshold=0.5, n_splits=3)

        assert result["fullBacktest"]["grossSharpe"] > 0

    def test_custom_cost_model(self, mock_pair_data):
        """Custom cost model is applied correctly."""
        cost_model = TransactionCostModel(fee_bps=20.0, slippage_bps=10.0)
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = evaluate_pair("TICK1", "TICK2", ou_window=126, k=0.25,
                                   kappa_threshold=1.0, n_splits=3,
                                   cost_model=cost_model)

        # Higher costs should reduce net returns relative to gross
        fb = result["fullBacktest"]
        gap = fb["grossSharpe"] - fb["netSharpe"]
        assert gap >= 0  # net <= gross with positive costs


class TestRunPhase7:
    def test_returns_expected_keys(self, mock_pair_data):
        """run_phase7 returns metrics_json and pair_results."""
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = run_phase7()

        assert "metrics_json" in result
        assert "pair_results" in result

    def test_metrics_json_schema(self, mock_pair_data):
        """metrics_json follows ARF standard schema."""
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = run_phase7()

        m = result["metrics_json"]
        assert "sharpeRatio" in m
        assert "annualReturn" in m
        assert "maxDrawdown" in m
        assert "hitRate" in m
        assert "totalTrades" in m
        assert "transactionCosts" in m
        assert "walkForward" in m
        assert "customMetrics" in m

    def test_custom_metrics_has_cross_pair(self, mock_pair_data):
        """customMetrics contains cross-pair results and aggregate stats."""
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = run_phase7()

        cm = result["metrics_json"]["customMetrics"]
        assert cm["phase"] == "Phase 7 - Multi-Pair Robustness Testing"
        assert "crossPairResults" in cm
        assert "aggregateStats" in cm
        assert "pairsEvaluated" in cm
        assert cm["pairsEvaluated"] == 4

    def test_aggregate_stats_keys(self, mock_pair_data):
        """Aggregate stats contain all expected fields."""
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = run_phase7()

        agg = result["metrics_json"]["customMetrics"]["aggregateStats"]
        expected = {"meanNetSharpe", "medianNetSharpe", "stdNetSharpe",
                    "meanAvgOosSharpe", "meanAnnualReturn", "meanMaxDrawdown",
                    "totalPositiveWindows", "totalWindows", "positiveWindowRate"}
        assert set(agg.keys()) == expected

    def test_params_match_phase6_optimized(self, mock_pair_data):
        """Phase 7 uses optimized params from Phase 6."""
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = run_phase7()

        params = result["metrics_json"]["customMetrics"]["params"]
        assert params["ouWindow"] == 252
        assert params["k"] == 0.25
        assert params["kappaThreshold"] == 1.0

    def test_four_pairs_evaluated(self, mock_pair_data):
        """All four pairs are evaluated."""
        with patch("src.run_backtest.DataLoader") as MockDL:
            MockDL.return_value = _make_mock_loader(mock_pair_data)
            result = run_phase7()

        assert len(result["pair_results"]) == 4
        pair_names = [r["pair"] for r in result["pair_results"]]
        assert "EWA/EWC" in pair_names
        assert "GLD/SLV" in pair_names
        assert "TLT/IEF" in pair_names
        assert "XOM/CVX" in pair_names

    def test_error_handling_continues(self, mock_pair_data):
        """If one pair fails, other pairs are still evaluated."""
        call_count = [0]

        def side_effect(t1, t2, start, end):
            call_count[0] += 1
            if call_count[0] <= 2:  # First pair (2 tickers)
                return mock_pair_data
            if call_count[0] <= 4:  # Second pair raises
                raise ValueError("Test error")
            return mock_pair_data

        with patch("src.run_backtest.DataLoader") as MockDL:
            mock_loader = _make_mock_loader(mock_pair_data)
            mock_loader.download_pair_data.side_effect = side_effect
            MockDL.return_value = mock_loader
            result = run_phase7()

        # Should still have 4 entries (some may have errors)
        assert len(result["pair_results"]) == 4
