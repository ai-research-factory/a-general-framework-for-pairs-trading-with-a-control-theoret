"""
Walk-forward backtest with rolling OU parameter estimation on real data.

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
    WalkForwardValidator,
    calculate_costs,
    compute_metrics,
    generate_metrics_json,
)


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
) -> dict:
    """
    Run walk-forward backtest with rolling OU parameters on real pair data.

    Args:
        use_rolling_hedge: If True, use rolling hedge ratio (no look-ahead bias).
            If False, use pre-computed static spread from Phase 2.

    Returns:
        Dict with 'metrics_json', 'results', 'full_backtest' keys.
    """
    config = config or BacktestConfig()
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
        # Use static hedge ratio (paper's approach for Phase 3)
        print("Computing static spread (full-sample hedge ratio)...")
        spread_df = loader.calculate_spread(pair_data, ticker1, ticker2)
        print(f"  Spread data: {len(spread_df)} points, beta={spread_df['beta'].iloc[0]:.4f}")

    spread_values = spread_df["spread"].values

    # Run full rolling backtest for overall metrics
    print(f"Running full rolling backtest (OU window={ou_window}, k={k})...")
    full_result = run_rolling_backtest(spread_values, ou_window=ou_window, k=k)
    print(f"  Tradeable period: {len(full_result['returns'])} days")
    print(f"  Final portfolio value: {full_result['portfolio_value'][-1]:.4f}")

    # Walk-forward validation
    print(f"Running walk-forward validation ({config.n_splits} windows)...")
    validator = WalkForwardValidator(config)
    wf_results = []

    for train_idx, test_idx in validator.split(spread_df):
        window_num = len(wf_results) + 1
        train_spread = spread_df.iloc[train_idx]["spread"].values
        test_spread = spread_df.iloc[test_idx]["spread"].values

        # Full segment for rolling estimation: we need ou_window before test starts
        # Use train data to estimate initial params, then run on test
        full_segment = np.concatenate([train_spread[-ou_window:], test_spread])
        bt = run_rolling_backtest(full_segment, ou_window=ou_window, k=k)

        returns_series = pd.Series(bt["returns"])
        positions_series = pd.Series(bt["allocations"])
        net_returns = calculate_costs(returns_series, positions_series, config)

        gross_metrics = compute_metrics(returns_series)
        net_metrics = compute_metrics(net_returns)

        total_trades = int((positions_series.diff().abs() > 1e-10).sum())

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
              f"gross Sharpe={result.gross_sharpe:.4f}, net Sharpe={result.net_sharpe:.4f}")

    # Compute overall metrics from full backtest
    full_returns = pd.Series(full_result["returns"])
    full_positions = pd.Series(full_result["allocations"])
    full_net_returns = calculate_costs(full_returns, full_positions, config)
    full_gross_metrics = compute_metrics(full_returns)
    full_net_metrics = compute_metrics(full_net_returns)

    # Rolling parameter stats (filtered to windows where kappa >= threshold)
    rp = full_result["rolling_params"]
    kappa_mask = rp["kappa"] >= 0.5
    rp_active = rp[kappa_mask]

    custom_metrics = {
        "phase": "Phase 3 - Rolling OU Parameter Estimation",
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
        "fullBacktest": {
            "grossSharpe": full_gross_metrics["sharpeRatio"],
            "netSharpe": full_net_metrics["sharpeRatio"],
            "annualReturn": full_net_metrics["annualReturn"],
            "maxDrawdown": full_net_metrics["maxDrawdown"],
        },
        "rollingParamStats_activeOnly": {
            "kappa": {"mean": round(float(rp_active["kappa"].mean()), 4), "std": round(float(rp_active["kappa"].std()), 4)},
            "mu": {"mean": round(float(rp_active["mu"].mean()), 4), "std": round(float(rp_active["mu"].std()), 4)},
            "sigma": {"mean": round(float(rp_active["sigma"].mean()), 4), "std": round(float(rp_active["sigma"].std()), 4)},
        },
    }

    metrics_json = generate_metrics_json(wf_results, config, custom_metrics)

    return {
        "metrics_json": metrics_json,
        "results": wf_results,
        "full_backtest": full_result,
        "spread_df": spread_df,
    }


def main():
    output = run_walk_forward()
    metrics = output["metrics_json"]

    # Save metrics
    report_dir = Path("reports/cycle_3")
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {report_dir / 'metrics.json'}")

    # Print summary
    print("\n=== Walk-Forward Results ===")
    wf = metrics["walkForward"]
    print(f"Windows: {wf['windows']}, Positive: {wf['positiveWindows']}, Avg OOS Sharpe: {wf['avgOosSharpe']:.4f}")
    print(f"Overall Sharpe (gross): {metrics['sharpeRatio']:.4f}")
    tc = metrics["transactionCosts"]
    print(f"Overall Sharpe (net): {tc['netSharpe']:.4f}")
    print(f"Annual Return: {metrics['annualReturn']:.4f}")
    print(f"Max Drawdown: {metrics['maxDrawdown']:.4f}")
    print(f"Hit Rate: {metrics['hitRate']:.4f}")
    print(f"Total Trades: {metrics['totalTrades']}")


if __name__ == "__main__":
    main()
