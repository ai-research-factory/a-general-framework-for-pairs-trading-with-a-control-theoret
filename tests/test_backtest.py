"""Tests for ARF backtest framework, including Phase 5 transaction cost model."""
import numpy as np
import pandas as pd
import pytest
from src.backtest import (
    BacktestConfig,
    BacktestResult,
    TransactionCostModel,
    WalkForwardValidator,
    apply_transaction_costs,
    calculate_costs,
    compute_metrics,
    generate_metrics_json,
)


@pytest.fixture
def sample_returns():
    """Sample daily returns series."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.02, 500))


@pytest.fixture
def sample_positions():
    """Sample position series with varying allocation sizes."""
    np.random.seed(42)
    pos = np.cumsum(np.random.normal(0, 0.1, 500))
    return pd.Series(pos)


class TestTransactionCostModel:
    def test_default_values(self):
        model = TransactionCostModel()
        assert model.fee_bps == 10.0
        assert model.slippage_bps == 5.0
        assert model.eta == 0.0

    def test_custom_values(self):
        model = TransactionCostModel(fee_bps=5.0, slippage_bps=3.0, eta=0.002)
        assert model.fee_bps == 5.0
        assert model.slippage_bps == 3.0
        assert model.eta == 0.002


class TestApplyTransactionCosts:
    def test_zero_cost_model_returns_gross(self, sample_returns, sample_positions):
        model = TransactionCostModel(fee_bps=0, slippage_bps=0, eta=0)
        result = apply_transaction_costs(sample_returns, sample_positions, model)
        pd.testing.assert_series_equal(result["net_returns"], sample_returns)

    def test_net_returns_less_than_gross(self, sample_returns, sample_positions):
        model = TransactionCostModel(fee_bps=10, slippage_bps=5, eta=0)
        result = apply_transaction_costs(sample_returns, sample_positions, model)
        assert result["net_returns"].sum() < sample_returns.sum()

    def test_costs_non_negative(self, sample_returns, sample_positions):
        model = TransactionCostModel(fee_bps=10, slippage_bps=5, eta=0.001)
        result = apply_transaction_costs(sample_returns, sample_positions, model)
        assert (result["total_costs"] >= 0).all()
        assert (result["fee_costs"] >= 0).all()
        assert (result["slippage_costs"] >= 0).all()
        assert (result["impact_costs"] >= 0).all()

    def test_cost_decomposition_sums(self, sample_returns, sample_positions):
        model = TransactionCostModel(fee_bps=10, slippage_bps=5, eta=0.001)
        result = apply_transaction_costs(sample_returns, sample_positions, model)
        total = result["fee_costs"] + result["slippage_costs"] + result["impact_costs"]
        pd.testing.assert_series_equal(result["total_costs"], total)

    def test_fee_slippage_proportional_to_turnover(self, sample_returns, sample_positions):
        model = TransactionCostModel(fee_bps=10, slippage_bps=5, eta=0)
        result = apply_transaction_costs(sample_returns, sample_positions, model)
        turnover = result["turnover"]
        expected_fee = (10.0 / 10000) * turnover
        expected_slip = (5.0 / 10000) * turnover
        pd.testing.assert_series_equal(result["fee_costs"], expected_fee)
        pd.testing.assert_series_equal(result["slippage_costs"], expected_slip)

    def test_impact_uses_sqrt(self, sample_returns, sample_positions):
        eta = 0.005
        model = TransactionCostModel(fee_bps=0, slippage_bps=0, eta=eta)
        result = apply_transaction_costs(sample_returns, sample_positions, model)
        turnover = result["turnover"]
        mask = turnover > model.min_trade_size
        expected_impact = pd.Series(0.0, index=turnover.index)
        expected_impact[mask] = eta * np.sqrt(turnover[mask])
        pd.testing.assert_series_equal(result["impact_costs"], expected_impact)

    def test_higher_fees_mean_higher_costs(self, sample_returns, sample_positions):
        low = TransactionCostModel(fee_bps=5, slippage_bps=3, eta=0)
        high = TransactionCostModel(fee_bps=15, slippage_bps=10, eta=0)
        r_low = apply_transaction_costs(sample_returns, sample_positions, low)
        r_high = apply_transaction_costs(sample_returns, sample_positions, high)
        assert r_high["total_costs"].sum() > r_low["total_costs"].sum()

    def test_impact_increases_costs(self, sample_returns, sample_positions):
        no_impact = TransactionCostModel(fee_bps=10, slippage_bps=5, eta=0)
        with_impact = TransactionCostModel(fee_bps=10, slippage_bps=5, eta=0.001)
        r1 = apply_transaction_costs(sample_returns, sample_positions, no_impact)
        r2 = apply_transaction_costs(sample_returns, sample_positions, with_impact)
        assert r2["total_costs"].sum() > r1["total_costs"].sum()

    def test_summary_keys(self, sample_returns, sample_positions):
        model = TransactionCostModel(fee_bps=10, slippage_bps=5, eta=0.001)
        result = apply_transaction_costs(sample_returns, sample_positions, model)
        expected_keys = {
            "totalCosts", "totalTurnover", "numTrades", "avgCostPerTrade",
            "feeCosts", "slippageCosts", "impactCosts", "costReturnRatio",
        }
        assert set(result["summary"].keys()) == expected_keys

    def test_no_trades_zero_costs(self):
        """Flat position = no trades = no costs."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.005])
        positions = pd.Series([1.0, 1.0, 1.0, 1.0])
        model = TransactionCostModel(fee_bps=10, slippage_bps=5, eta=0.001)
        result = apply_transaction_costs(returns, positions, model)
        assert result["summary"]["totalCosts"] == 0.0
        assert result["summary"]["numTrades"] == 0


class TestCalculateCostsBackcompat:
    """Test backward compatibility of the original calculate_costs function."""

    def test_matches_simple_model(self, sample_returns, sample_positions):
        config = BacktestConfig(fee_bps=10, slippage_bps=5)
        net_old = calculate_costs(sample_returns, sample_positions, config)
        model = TransactionCostModel(fee_bps=10, slippage_bps=5, eta=0)
        result_new = apply_transaction_costs(sample_returns, sample_positions, model)
        pd.testing.assert_series_equal(net_old, result_new["net_returns"])


class TestComputeMetrics:
    def test_positive_sharpe_for_positive_returns(self):
        returns = pd.Series([0.01] * 100)
        m = compute_metrics(returns)
        assert m["sharpeRatio"] > 0

    def test_empty_returns(self):
        m = compute_metrics(pd.Series([], dtype=float))
        assert m["sharpeRatio"] == 0.0

    def test_hit_rate_all_positive(self):
        returns = pd.Series([0.01, 0.02, 0.005])
        m = compute_metrics(returns)
        assert m["hitRate"] == 1.0


class TestGenerateMetricsJson:
    def test_empty_results(self):
        config = BacktestConfig()
        m = generate_metrics_json([], config)
        assert m["sharpeRatio"] == 0.0
        assert m["walkForward"]["windows"] == 0

    def test_with_cost_model(self):
        config = BacktestConfig()
        cost_model = TransactionCostModel(fee_bps=5, slippage_bps=3, eta=0.001)
        m = generate_metrics_json([], config, cost_model=cost_model)
        assert m["transactionCosts"]["feeBps"] == 5.0
        assert m["transactionCosts"]["slippageBps"] == 3.0
