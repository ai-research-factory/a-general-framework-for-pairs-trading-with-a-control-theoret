"""
ARF Standard Backtest Framework
Walk-forward validation with transaction cost accounting.

Phase 5 adds a realistic transaction cost model with:
- Proportional commission fees (bps on trade notional)
- Linear slippage (bps on trade notional)
- Square-root market impact: impact = eta * sqrt(|turnover|)
- Detailed per-trade cost breakdown
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    fee_bps: float = 10.0       # Transaction fee in basis points
    slippage_bps: float = 5.0   # Slippage in basis points
    train_ratio: float = 0.7    # Train window ratio for walk-forward
    n_splits: int = 10          # Number of walk-forward windows
    gap: int = 1                # Gap between train and test (prevent leakage)
    min_train_size: int = 252   # Minimum training samples (~1 year daily)


@dataclass
class BacktestResult:
    """Results from a single walk-forward window."""
    window: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    gross_sharpe: float = 0.0
    net_sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    hit_rate: float = 0.0
    pnl_series: Optional[pd.Series] = field(default=None, repr=False)


@dataclass
class TransactionCostModel:
    """
    Realistic transaction cost model with three components:

    1. Proportional fee: commission charged as bps on trade notional (|Δh|)
    2. Linear slippage: market bid-ask spread cost as bps on trade notional
    3. Market impact: square-root model η * sqrt(|Δh|), capturing temporary
       price impact from large trades (Almgren & Chriss, 2000)

    Total cost per step = (fee_bps + slippage_bps)/10000 * |Δh| + eta * sqrt(|Δh|)
    """
    fee_bps: float = 10.0          # Proportional commission (basis points)
    slippage_bps: float = 5.0      # Bid-ask spread slippage (basis points)
    eta: float = 0.0               # Market impact coefficient (0 = no impact)
    min_trade_size: float = 1e-8   # Ignore trades smaller than this


def apply_transaction_costs(
    returns: pd.Series,
    positions: pd.Series,
    cost_model: TransactionCostModel,
) -> dict:
    """
    Apply realistic transaction costs to a returns series.

    Args:
        returns: Gross returns series (percentage or PnL)
        positions: Position/allocation series (continuous values)
        cost_model: TransactionCostModel with fee, slippage, and impact params

    Returns:
        Dict with:
            'net_returns': Returns after all costs
            'total_costs': Total cost series
            'fee_costs': Commission fee component
            'slippage_costs': Slippage component
            'impact_costs': Market impact component
            'turnover': Absolute position changes per step
            'summary': Dict with aggregate cost statistics
    """
    turnover = positions.diff().abs().fillna(0)

    # Proportional costs: (fee + slippage) * |Δposition|
    prop_rate = (cost_model.fee_bps + cost_model.slippage_bps) / 10000
    fee_costs = (cost_model.fee_bps / 10000) * turnover
    slippage_costs = (cost_model.slippage_bps / 10000) * turnover

    # Square-root market impact: eta * sqrt(|Δposition|)
    # Only applied to trades above min_trade_size
    trade_mask = turnover > cost_model.min_trade_size
    impact_costs = pd.Series(0.0, index=turnover.index)
    if cost_model.eta > 0:
        impact_costs[trade_mask] = cost_model.eta * np.sqrt(turnover[trade_mask])

    total_costs = fee_costs + slippage_costs + impact_costs
    net_returns = returns - total_costs

    # Aggregate statistics
    n_trades = int(trade_mask.sum())
    total_cost_sum = float(total_costs.sum())
    total_turnover = float(turnover.sum())
    avg_cost_per_trade = total_cost_sum / n_trades if n_trades > 0 else 0.0

    return {
        "net_returns": net_returns,
        "total_costs": total_costs,
        "fee_costs": fee_costs,
        "slippage_costs": slippage_costs,
        "impact_costs": impact_costs,
        "turnover": turnover,
        "summary": {
            "totalCosts": round(total_cost_sum, 6),
            "totalTurnover": round(total_turnover, 4),
            "numTrades": n_trades,
            "avgCostPerTrade": round(avg_cost_per_trade, 6),
            "feeCosts": round(float(fee_costs.sum()), 6),
            "slippageCosts": round(float(slippage_costs.sum()), 6),
            "impactCosts": round(float(impact_costs.sum()), 6),
            "costReturnRatio": round(
                total_cost_sum / float(returns.sum()) if float(returns.sum()) != 0 else 0.0, 4
            ),
        },
    }


class WalkForwardValidator:
    """
    Walk-forward out-of-sample validation.

    Usage:
        validator = WalkForwardValidator(config)
        for train_idx, test_idx in validator.split(df):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            # Train model on train_df, evaluate on test_df
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def split(self, data: pd.DataFrame):
        """Generate train/test index pairs for walk-forward validation."""
        n = len(data)
        cfg = self.config
        test_size = max(1, (n - cfg.min_train_size) // cfg.n_splits)

        for i in range(cfg.n_splits):
            test_end = n - (cfg.n_splits - 1 - i) * test_size
            test_start = test_end - test_size
            train_end = test_start - cfg.gap
            train_start = max(0, int(train_end * (1 - cfg.train_ratio))) if cfg.train_ratio < 1.0 else 0

            if train_end - train_start < cfg.min_train_size:
                continue
            if test_start >= test_end:
                continue

            yield (
                list(range(train_start, train_end)),
                list(range(test_start, test_end)),
            )


def calculate_costs(returns: pd.Series, positions: pd.Series, config: BacktestConfig) -> pd.Series:
    """
    Calculate transaction costs from position changes.

    Args:
        returns: Gross returns series
        positions: Position series (-1, 0, 1 or continuous)
        config: Backtest configuration with fee/slippage settings

    Returns:
        Net returns after costs
    """
    trades = positions.diff().abs().fillna(0)
    cost_per_trade = (config.fee_bps + config.slippage_bps) / 10000
    costs = trades * cost_per_trade
    return returns - costs


def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> dict:
    """
    Compute standard performance metrics from a returns series.

    Args:
        returns: Daily (or periodic) returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily, 365 for crypto)

    Returns:
        Dict with sharpeRatio, annualReturn, maxDrawdown, hitRate, totalTrades
    """
    if len(returns) == 0:
        return {"sharpeRatio": 0.0, "annualReturn": 0.0, "maxDrawdown": 0.0, "hitRate": 0.0}

    excess = returns - risk_free_rate / periods_per_year
    sharpe = float(np.sqrt(periods_per_year) * excess.mean() / excess.std()) if excess.std() > 0 else 0.0

    cumulative = (1 + returns).cumprod()
    final_value = cumulative.iloc[-1]
    if final_value > 0 and np.isfinite(final_value):
        annual_return = float(final_value ** (periods_per_year / len(returns)) - 1)
    else:
        # Fallback: use mean return annualized
        annual_return = float(returns.mean() * periods_per_year)

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    hit_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

    return {
        "sharpeRatio": round(sharpe, 4),
        "annualReturn": round(annual_return, 4),
        "maxDrawdown": round(max_drawdown, 4),
        "hitRate": round(hit_rate, 4),
    }


def generate_metrics_json(
    results: list[BacktestResult],
    config: BacktestConfig,
    custom_metrics: Optional[dict] = None,
    cost_model: Optional[TransactionCostModel] = None,
) -> dict:
    """
    Generate ARF-standard metrics.json from walk-forward results.

    Args:
        results: List of BacktestResult from each window
        config: Backtest configuration
        custom_metrics: Optional paper-specific metrics
        cost_model: Optional TransactionCostModel (uses config bps if None)

    Returns:
        Dict matching ARF metrics.json schema
    """
    fee_bps = cost_model.fee_bps if cost_model else config.fee_bps
    slip_bps = cost_model.slippage_bps if cost_model else config.slippage_bps

    if not results:
        return {
            "sharpeRatio": 0.0, "annualReturn": 0.0, "maxDrawdown": 0.0,
            "hitRate": 0.0, "totalTrades": 0,
            "transactionCosts": {"feeBps": fee_bps, "slippageBps": slip_bps, "netSharpe": 0.0},
            "walkForward": {"windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0},
            "customMetrics": custom_metrics or {},
        }

    net_sharpes = [r.net_sharpe for r in results]
    positive_windows = sum(1 for s in net_sharpes if s > 0)

    return {
        "sharpeRatio": round(float(np.mean([r.gross_sharpe for r in results])), 4),
        "annualReturn": round(float(np.mean([r.annual_return for r in results])), 4),
        "maxDrawdown": round(float(min(r.max_drawdown for r in results)), 4),
        "hitRate": round(float(np.mean([r.hit_rate for r in results])), 4),
        "totalTrades": sum(r.total_trades for r in results),
        "transactionCosts": {
            "feeBps": fee_bps,
            "slippageBps": slip_bps,
            "netSharpe": round(float(np.mean(net_sharpes)), 4),
        },
        "walkForward": {
            "windows": len(results),
            "positiveWindows": positive_windows,
            "avgOosSharpe": round(float(np.mean(net_sharpes)), 4),
        },
        "customMetrics": custom_metrics or {},
    }
