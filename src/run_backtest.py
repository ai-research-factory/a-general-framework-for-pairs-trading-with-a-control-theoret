"""
Walk-forward backtest with rolling OU parameter estimation on real data.
Phase 5: Enhanced transaction cost model with detailed cost breakdown.

Usage:
    python -m src.run_backtest
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import DataLoader
from src.model import run_rolling_backtest
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


# Standard cost scenarios for comparison
COST_SCENARIOS = {
    "zero": TransactionCostModel(fee_bps=0.0, slippage_bps=0.0, eta=0.0),
    "low": TransactionCostModel(fee_bps=5.0, slippage_bps=3.0, eta=0.0),
    "base": TransactionCostModel(fee_bps=10.0, slippage_bps=5.0, eta=0.0),
    "high": TransactionCostModel(fee_bps=15.0, slippage_bps=10.0, eta=0.0),
    "base_impact": TransactionCostModel(fee_bps=10.0, slippage_bps=5.0, eta=0.001),
}


def run_walk_forward(
    ticker1: str = "EWA",
    ticker2: str = "EWC",
    start_date: str = "2000-01-01",
    end_date: str = "2026-03-28",
    ou_window: int = 252,
    k: float = 1.0,
    use_rolling_hedge: bool = False,
    hedge_window: int = 252,
    config: BacktestConfig | None = None,
    cost_model: TransactionCostModel | None = None,
) -> dict:
    """
    Run walk-forward backtest with rolling OU parameters on real pair data.

    Args:
        cost_model: TransactionCostModel for detailed cost accounting.
            If None, uses base scenario (10 bps fee + 5 bps slippage).

    Returns:
        Dict with 'metrics_json', 'results', 'full_backtest',
        'spread_df', 'cost_breakdown', 'scenario_comparison' keys.
    """
    config = config or BacktestConfig()
    cost_model = cost_model or COST_SCENARIOS["base"]
    loader = DataLoader()

    # Load or download pair data
    print(f"Loading {ticker1}/{ticker2} data...")
    pair_data = loader.download_pair_data(ticker1, ticker2, start_date, end_date)
    print(f"  Raw data: {len(pair_data)} trading days, {pair_data.index[0].date()} to {pair_data.index[-1].date()}")

    if use_rolling_hedge:
        print(f"Computing rolling spread (hedge window={hedge_window})...")
        spread_df = loader.calculate_rolling_spread(pair_data, ticker1, ticker2, window=hedge_window)
        print(f"  Spread data: {len(spread_df)} points after rolling window")
    else:
        print("Computing static spread (full-sample hedge ratio)...")
        spread_df = loader.calculate_spread(pair_data, ticker1, ticker2)
        print(f"  Spread data: {len(spread_df)} points, beta={spread_df['beta'].iloc[0]:.4f}")

    spread_values = spread_df["spread"].values

    # Run full rolling backtest for overall metrics
    print(f"Running full rolling backtest (OU window={ou_window}, k={k})...")
    full_result = run_rolling_backtest(spread_values, ou_window=ou_window, k=k)
    print(f"  Tradeable period: {len(full_result['returns'])} days")
    print(f"  Final portfolio value (gross): {full_result['portfolio_value'][-1]:.4f}")

    # Apply enhanced cost model to full backtest
    full_returns = pd.Series(full_result["returns"])
    full_positions = pd.Series(full_result["allocations"])
    full_cost_result = apply_transaction_costs(full_returns, full_positions, cost_model)
    full_gross_metrics = compute_metrics(full_returns)
    full_net_metrics = compute_metrics(full_cost_result["net_returns"])

    print(f"  Cost model: {cost_model.fee_bps} bps fee + {cost_model.slippage_bps} bps slippage"
          f" + eta={cost_model.eta}")
    print(f"  Total costs: {full_cost_result['summary']['totalCosts']:.4f}")
    print(f"  Gross Sharpe: {full_gross_metrics['sharpeRatio']:.4f}, "
          f"Net Sharpe: {full_net_metrics['sharpeRatio']:.4f}")

    # Run scenario comparison on full backtest
    print("\n=== Cost Scenario Comparison (Full Backtest) ===")
    scenario_comparison = {}
    for name, scenario in COST_SCENARIOS.items():
        sc_result = apply_transaction_costs(full_returns, full_positions, scenario)
        sc_metrics = compute_metrics(sc_result["net_returns"])
        scenario_comparison[name] = {
            "feeBps": scenario.fee_bps,
            "slippageBps": scenario.slippage_bps,
            "eta": scenario.eta,
            "netSharpe": sc_metrics["sharpeRatio"],
            "annualReturn": sc_metrics["annualReturn"],
            "maxDrawdown": sc_metrics["maxDrawdown"],
            "totalCosts": sc_result["summary"]["totalCosts"],
            "costReturnRatio": sc_result["summary"]["costReturnRatio"],
        }
        print(f"  {name:15s}: Sharpe={sc_metrics['sharpeRatio']:+.4f}, "
              f"Return={sc_metrics['annualReturn']:+.4f}, "
              f"Costs={sc_result['summary']['totalCosts']:.4f}")

    # Walk-forward validation with enhanced costs
    print(f"\nRunning walk-forward validation ({config.n_splits} windows)...")
    validator = WalkForwardValidator(config)
    wf_results = []

    for train_idx, test_idx in validator.split(spread_df):
        window_num = len(wf_results) + 1
        train_spread = spread_df.iloc[train_idx]["spread"].values
        test_spread = spread_df.iloc[test_idx]["spread"].values

        full_segment = np.concatenate([train_spread[-ou_window:], test_spread])
        bt = run_rolling_backtest(full_segment, ou_window=ou_window, k=k)

        returns_series = pd.Series(bt["returns"])
        positions_series = pd.Series(bt["allocations"])

        # Use enhanced cost model
        cost_result = apply_transaction_costs(returns_series, positions_series, cost_model)
        gross_metrics = compute_metrics(returns_series)
        net_metrics = compute_metrics(cost_result["net_returns"])

        total_trades = cost_result["summary"]["numTrades"]

        train_dates = spread_df.index[train_idx]
        test_dates = spread_df.index[test_idx]

        result = BacktestResult(
            window=window_num,
            train_start=str(train_dates[0].date()),
            train_end=str(train_dates[-1].date()),
            test_start=str(test_dates[0].date()),
            test_end=str(test_dates[-1].date()),
            gross_sharpe=gross_metrics["sharpeRatio"],
            net_sharpe=net_metrics["sharpeRatio"],
            annual_return=net_metrics["annualReturn"],
            max_drawdown=net_metrics["maxDrawdown"],
            total_trades=total_trades,
            hit_rate=net_metrics["hitRate"],
        )
        wf_results.append(result)
        print(f"  Window {window_num}: test {result.test_start} to {result.test_end}, "
              f"gross Sharpe={result.gross_sharpe:.4f}, net Sharpe={result.net_sharpe:.4f}, "
              f"trades={total_trades}")

    # Rolling parameter stats
    rp = full_result["rolling_params"]
    kappa_mask = rp["kappa"] >= 0.5
    rp_active = rp[kappa_mask]

    custom_metrics = {
        "phase": "Phase 5 - Transaction Cost Model",
        "pair": f"{ticker1}/{ticker2}",
        "dataSource": "ARF Data API",
        "hedgeMethod": "rolling" if use_rolling_hedge else "static",
        "hedgeWindow": hedge_window if use_rolling_hedge else "full-sample",
        "ouWindow": ou_window,
        "controlGain": k,
        "kappaThreshold": 0.5,
        "dataPoints": len(spread_df),
        "tradeableDays": len(full_result["returns"]),
        "activeTradingDays": int(kappa_mask.sum()),
        "dateRange": f"{spread_df.index[0].date()} to {spread_df.index[-1].date()}",
        "finalPortfolioValue": round(float(full_result["portfolio_value"][-1]), 4),
        "costModel": {
            "feeBps": cost_model.fee_bps,
            "slippageBps": cost_model.slippage_bps,
            "eta": cost_model.eta,
        },
        "costBreakdown": full_cost_result["summary"],
        "fullBacktest": {
            "grossSharpe": full_gross_metrics["sharpeRatio"],
            "netSharpe": full_net_metrics["sharpeRatio"],
            "annualReturn": full_net_metrics["annualReturn"],
            "maxDrawdown": full_net_metrics["maxDrawdown"],
        },
        "scenarioComparison": scenario_comparison,
        "rollingParamStats_activeOnly": {
            "kappa": {"mean": round(float(rp_active["kappa"].mean()), 4), "std": round(float(rp_active["kappa"].std()), 4)},
            "mu": {"mean": round(float(rp_active["mu"].mean()), 4), "std": round(float(rp_active["mu"].std()), 4)},
            "sigma": {"mean": round(float(rp_active["sigma"].mean()), 4), "std": round(float(rp_active["sigma"].std()), 4)},
        },
    }

    metrics_json = generate_metrics_json(wf_results, config, custom_metrics, cost_model)

    return {
        "metrics_json": metrics_json,
        "results": wf_results,
        "full_backtest": full_result,
        "spread_df": spread_df,
        "cost_breakdown": full_cost_result,
        "scenario_comparison": scenario_comparison,
    }


def main():
    output = run_walk_forward()
    metrics = output["metrics_json"]

    # Save metrics
    report_dir = Path("reports/cycle_5")
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {report_dir / 'metrics.json'}")

    # Print summary
    print("\n=== Walk-Forward Results (Phase 5) ===")
    wf = metrics["walkForward"]
    print(f"Windows: {wf['windows']}, Positive: {wf['positiveWindows']}, Avg OOS Sharpe: {wf['avgOosSharpe']:.4f}")
    print(f"Overall Sharpe (gross): {metrics['sharpeRatio']:.4f}")
    tc = metrics["transactionCosts"]
    print(f"Overall Sharpe (net): {tc['netSharpe']:.4f}")
    print(f"Annual Return: {metrics['annualReturn']:.4f}")
    print(f"Max Drawdown: {metrics['maxDrawdown']:.4f}")
    print(f"Hit Rate: {metrics['hitRate']:.4f}")
    print(f"Total Trades: {metrics['totalTrades']}")

    # Print cost breakdown
    cm = metrics["customMetrics"]
    print(f"\n=== Cost Breakdown ===")
    cb = cm["costBreakdown"]
    print(f"Total costs: {cb['totalCosts']:.4f}")
    print(f"  Fees: {cb['feeCosts']:.4f}")
    print(f"  Slippage: {cb['slippageCosts']:.4f}")
    print(f"  Impact: {cb['impactCosts']:.4f}")
    print(f"Cost/Return ratio: {cb['costReturnRatio']:.4f}")
    print(f"Num trades: {cb['numTrades']}")
    print(f"Avg cost/trade: {cb['avgCostPerTrade']:.6f}")


if __name__ == "__main__":
    main()
