"""Tests for Phase 6 hyperparameter optimization."""
import numpy as np
import pandas as pd
import pytest
from src.ou_process import generate_ou_path
from src.optimize import (
    OptimizationConfig,
    evaluate_params,
    optimize_walk_forward,
    run_sensitivity_analysis,
)
from src.backtest import BacktestConfig, TransactionCostModel


@pytest.fixture
def ou_spread():
    """Generate a mean-reverting OU spread for testing."""
    return generate_ou_path(kappa=5.0, mu=0.0, sigma=0.5, s0=0.0,
                            n_steps=2000, seed=42)


@pytest.fixture
def ou_spread_df(ou_spread):
    """OU spread as a DataFrame with DatetimeIndex."""
    dates = pd.bdate_range("2010-01-01", periods=len(ou_spread))
    return pd.DataFrame({"spread": ou_spread}, index=dates)


class TestOptimizationConfig:
    def test_default_values(self):
        cfg = OptimizationConfig()
        assert 252 in cfg.ou_windows
        assert 1.0 in cfg.k_values
        assert 0.5 in cfg.kappa_thresholds
        assert cfg.target_metric == "netSharpe"

    def test_custom_values(self):
        cfg = OptimizationConfig(
            ou_windows=[100, 200],
            k_values=[0.5, 1.0],
            kappa_thresholds=[0.5],
        )
        assert cfg.ou_windows == [100, 200]
        assert cfg.k_values == [0.5, 1.0]
        assert len(cfg.kappa_thresholds) == 1

    def test_default_cost_model(self):
        cfg = OptimizationConfig()
        assert cfg.cost_model.fee_bps == 10.0
        assert cfg.cost_model.slippage_bps == 5.0


class TestEvaluateParams:
    def test_returns_expected_keys(self, ou_spread):
        result = evaluate_params(ou_spread, ou_window=126, k=1.0)
        expected_keys = {"grossSharpe", "netSharpe", "annualReturn",
                         "maxDrawdown", "hitRate", "totalTrades", "turnover"}
        assert set(result.keys()) == expected_keys

    def test_short_spread_returns_zeros(self):
        short = np.array([0.0, 0.1, 0.2])
        result = evaluate_params(short, ou_window=126, k=1.0)
        assert result["grossSharpe"] == 0.0
        assert result["totalTrades"] == 0

    def test_positive_sharpe_on_ou(self, ou_spread):
        result = evaluate_params(ou_spread, ou_window=252, k=1.0)
        assert result["grossSharpe"] > 0

    def test_net_sharpe_less_than_gross(self, ou_spread):
        cost_model = TransactionCostModel(fee_bps=10, slippage_bps=5)
        result = evaluate_params(ou_spread, ou_window=252, k=1.0,
                                 cost_model=cost_model)
        assert result["netSharpe"] <= result["grossSharpe"]

    def test_higher_k_changes_turnover(self, ou_spread):
        r1 = evaluate_params(ou_spread, ou_window=252, k=0.5)
        r2 = evaluate_params(ou_spread, ou_window=252, k=2.0)
        # Higher k means larger positions, different turnover
        assert r1["turnover"] != r2["turnover"]

    def test_zero_cost_net_equals_gross(self, ou_spread):
        cost_model = TransactionCostModel(fee_bps=0, slippage_bps=0, eta=0)
        result = evaluate_params(ou_spread, ou_window=252, k=1.0,
                                 cost_model=cost_model)
        assert abs(result["netSharpe"] - result["grossSharpe"]) < 0.01


class TestOptimizeWalkForward:
    def test_returns_expected_keys(self, ou_spread_df):
        cfg = OptimizationConfig(
            ou_windows=[126, 252],
            k_values=[0.5, 1.0],
            kappa_thresholds=[0.5],
            inner_n_splits=3,
            min_train_size=252,
        )
        from src.backtest import BacktestConfig
        bc = BacktestConfig(n_splits=3, min_train_size=252)
        result = optimize_walk_forward(ou_spread_df, cfg, bc)
        assert "best_params" in result
        assert "window_results" in result
        assert "grid_summary" in result
        assert "default_comparison" in result

    def test_best_params_has_required_fields(self, ou_spread_df):
        cfg = OptimizationConfig(
            ou_windows=[126, 252],
            k_values=[0.5, 1.0],
            kappa_thresholds=[0.5],
            inner_n_splits=3,
            min_train_size=252,
        )
        bc = BacktestConfig(n_splits=3, min_train_size=252)
        result = optimize_walk_forward(ou_spread_df, cfg, bc)
        bp = result["best_params"]
        assert "ouWindow" in bp
        assert "k" in bp
        assert "kappaThreshold" in bp
        assert bp["ouWindow"] in cfg.ou_windows
        assert bp["k"] in cfg.k_values

    def test_grid_summary_sorted_descending(self, ou_spread_df):
        cfg = OptimizationConfig(
            ou_windows=[126, 252],
            k_values=[0.5, 1.0],
            kappa_thresholds=[0.5],
            inner_n_splits=3,
            min_train_size=252,
        )
        bc = BacktestConfig(n_splits=3, min_train_size=252)
        result = optimize_walk_forward(ou_spread_df, cfg, bc)
        scores = [g["meanTrainScore"] for g in result["grid_summary"]]
        assert scores == sorted(scores, reverse=True)

    def test_window_results_have_oos_metrics(self, ou_spread_df):
        cfg = OptimizationConfig(
            ou_windows=[126, 252],
            k_values=[0.5, 1.0],
            kappa_thresholds=[0.5],
            inner_n_splits=3,
            min_train_size=252,
        )
        bc = BacktestConfig(n_splits=3, min_train_size=252)
        result = optimize_walk_forward(ou_spread_df, cfg, bc)
        for wr in result["window_results"]:
            assert "oosMetrics" in wr
            assert "defaultOosMetrics" in wr
            assert "bestParams" in wr
            assert "netSharpe" in wr["oosMetrics"]

    def test_default_comparison_structure(self, ou_spread_df):
        cfg = OptimizationConfig(
            ou_windows=[126, 252],
            k_values=[0.5, 1.0],
            kappa_thresholds=[0.5],
            inner_n_splits=3,
            min_train_size=252,
        )
        bc = BacktestConfig(n_splits=3, min_train_size=252)
        result = optimize_walk_forward(ou_spread_df, cfg, bc)
        comp = result["default_comparison"]
        assert "optimized" in comp
        assert "paperDefault" in comp
        assert "avgOosSharpe" in comp["optimized"]
        assert "positiveWindows" in comp["optimized"]


class TestSensitivityAnalysis:
    def test_returns_all_parameters(self, ou_spread):
        result = run_sensitivity_analysis(ou_spread)
        assert "ouWindow" in result
        assert "k" in result
        assert "kappaThreshold" in result

    def test_ou_window_sensitivity_length(self, ou_spread):
        result = run_sensitivity_analysis(ou_spread)
        # Should have results for all ou_windows that fit
        assert len(result["ouWindow"]) >= 3

    def test_k_sensitivity_varies(self, ou_spread):
        result = run_sensitivity_analysis(ou_spread)
        sharpes = [r["grossSharpe"] for r in result["k"]]
        # Different k values should produce different results
        assert len(set(sharpes)) > 1

    def test_kappa_threshold_sensitivity(self, ou_spread):
        result = run_sensitivity_analysis(ou_spread)
        # Higher kappa threshold means fewer trading days
        trades_low = None
        trades_high = None
        for r in result["kappaThreshold"]:
            if r["kappaThreshold"] == 0.0:
                trades_low = r["totalTrades"]
            elif r["kappaThreshold"] == 5.0:
                trades_high = r["totalTrades"]
        if trades_low is not None and trades_high is not None:
            assert trades_high <= trades_low

    def test_custom_base_params(self, ou_spread):
        result = run_sensitivity_analysis(
            ou_spread, base_ou_window=126, base_k=0.5, base_kappa_threshold=1.0
        )
        assert len(result["ouWindow"]) > 0
        assert len(result["k"]) > 0
