"""
Walk-forward backtest with rolling OU parameter estimation on real data.
Phase 7: Multi-pair robustness testing with optimized parameters.

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
from src.optimize import (
    OptimizationConfig,
    optimize_walk_forward,
    run_sensitivity_analysis,
    evaluate_params,
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


def run_phase6(
    ticker1: str = "EWA",
    ticker2: str = "EWC",
    start_date: str = "2000-01-01",
    end_date: str = "2026-03-28",
) -> dict:
    """
    Phase 6: Hyperparameter optimization.

    1. Load data and compute spread
    2. Run grid search optimization with walk-forward CV
    3. Run sensitivity analysis on full data
    4. Compare optimized vs paper default using outer walk-forward
    5. Generate metrics and save reports
    """
    loader = DataLoader()
    cost_model = COST_SCENARIOS["base"]

    # Load data
    print("=" * 60)
    print("Phase 6: Hyperparameter Optimization")
    print("=" * 60)
    print(f"\nLoading {ticker1}/{ticker2} data...")
    pair_data = loader.download_pair_data(ticker1, ticker2, start_date, end_date)
    print(f"  Raw data: {len(pair_data)} trading days")

    print("Computing static spread...")
    spread_df = loader.calculate_spread(pair_data, ticker1, ticker2)
    spread_values = spread_df["spread"].values
    print(f"  Spread data: {len(spread_df)} points")

    # Step 1: Walk-forward hyperparameter optimization
    print("\n" + "=" * 60)
    print("Step 1: Walk-Forward Hyperparameter Optimization")
    print("=" * 60)

    opt_config = OptimizationConfig(
        # Paper-near neighborhoods per reproduction mode rules
        ou_windows=[126, 189, 252, 315, 378],
        k_values=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        kappa_thresholds=[0.25, 0.5, 1.0],
        target_metric="netSharpe",
        inner_n_splits=5,
        min_train_size=504,
        cost_model=cost_model,
    )

    wf_config = BacktestConfig(
        n_splits=5,
        min_train_size=504,
        fee_bps=cost_model.fee_bps,
        slippage_bps=cost_model.slippage_bps,
    )

    opt_results = optimize_walk_forward(spread_df, opt_config, wf_config)

    # Step 2: Sensitivity analysis on full data
    print("\n" + "=" * 60)
    print("Step 2: Sensitivity Analysis")
    print("=" * 60)

    best = opt_results["best_params"]
    sensitivity = run_sensitivity_analysis(
        spread_values,
        base_ou_window=best["ouWindow"],
        base_k=best["k"],
        base_kappa_threshold=best["kappaThreshold"],
        cost_model=cost_model,
    )

    # Print sensitivity results
    for param_name, param_results in sensitivity.items():
        print(f"\n  {param_name} sensitivity:")
        for r in param_results:
            val = r[param_name]
            print(f"    {param_name}={val}: grossSharpe={r['grossSharpe']:.4f}, "
                  f"netSharpe={r['netSharpe']:.4f}, turnover={r['turnover']:.1f}")

    # Step 3: Final evaluation with optimized params (outer walk-forward)
    print("\n" + "=" * 60)
    print("Step 3: Final Evaluation (Optimized vs Default)")
    print("=" * 60)

    ou_w_opt = best["ouWindow"]
    k_opt = best["k"]
    kappa_th_opt = best["kappaThreshold"]

    # Run full backtest with optimized params
    print(f"\nOptimized params: ou_window={ou_w_opt}, k={k_opt}, kappa_th={kappa_th_opt}")
    opt_full = run_rolling_backtest(spread_values, ou_window=ou_w_opt, k=k_opt,
                                    kappa_threshold=kappa_th_opt)
    opt_returns = pd.Series(opt_full["returns"])
    opt_positions = pd.Series(opt_full["allocations"])
    opt_cost = apply_transaction_costs(opt_returns, opt_positions, cost_model)
    opt_gross = compute_metrics(opt_returns)
    opt_net = compute_metrics(opt_cost["net_returns"])

    # Run full backtest with paper defaults
    print(f"Paper defaults: ou_window=252, k=1.0, kappa_th=0.5")
    def_full = run_rolling_backtest(spread_values, ou_window=252, k=1.0, kappa_threshold=0.5)
    def_returns = pd.Series(def_full["returns"])
    def_positions = pd.Series(def_full["allocations"])
    def_cost = apply_transaction_costs(def_returns, def_positions, cost_model)
    def_gross = compute_metrics(def_returns)
    def_net = compute_metrics(def_cost["net_returns"])

    print(f"\n  {'Metric':<20} {'Default':>12} {'Optimized':>12} {'Delta':>12}")
    print(f"  {'-'*56}")
    for metric_name in ["sharpeRatio", "annualReturn", "maxDrawdown", "hitRate"]:
        d = def_net[metric_name]
        o = opt_net[metric_name]
        delta = o - d
        print(f"  {metric_name:<20} {d:>12.4f} {o:>12.4f} {delta:>+12.4f}")

    print(f"  {'grossSharpe':<20} {def_gross['sharpeRatio']:>12.4f} "
          f"{opt_gross['sharpeRatio']:>12.4f} "
          f"{opt_gross['sharpeRatio'] - def_gross['sharpeRatio']:>+12.4f}")
    print(f"  {'totalCosts':<20} {def_cost['summary']['totalCosts']:>12.4f} "
          f"{opt_cost['summary']['totalCosts']:>12.4f} "
          f"{opt_cost['summary']['totalCosts'] - def_cost['summary']['totalCosts']:>+12.4f}")
    print(f"  {'costReturnRatio':<20} {def_cost['summary']['costReturnRatio']:>12.4f} "
          f"{opt_cost['summary']['costReturnRatio']:>12.4f}")
    print(f"  {'turnover':<20} {def_cost['summary']['totalTurnover']:>12.1f} "
          f"{opt_cost['summary']['totalTurnover']:>12.1f}")

    # Step 4: Walk-forward with optimized params for metrics.json
    print("\n" + "=" * 60)
    print("Step 4: Walk-Forward Evaluation with Optimized Params")
    print("=" * 60)

    outer_config = BacktestConfig(n_splits=10, min_train_size=252)
    output_opt = run_walk_forward(
        ticker1=ticker1, ticker2=ticker2,
        start_date=start_date, end_date=end_date,
        ou_window=ou_w_opt, k=k_opt,
        config=outer_config, cost_model=cost_model,
    )

    # Build custom metrics for Phase 6
    rp = opt_full["rolling_params"]
    kappa_mask = rp["kappa"] >= kappa_th_opt
    rp_active = rp[kappa_mask]

    custom_metrics = {
        "phase": "Phase 6 - Hyperparameter Optimization",
        "pair": f"{ticker1}/{ticker2}",
        "dataSource": "ARF Data API",
        "dataPoints": len(spread_df),
        "dateRange": f"{spread_df.index[0].date()} to {spread_df.index[-1].date()}",
        "optimization": {
            "method": "grid_search_walk_forward_cv",
            "targetMetric": opt_config.target_metric,
            "gridSize": len(opt_config.ou_windows) * len(opt_config.k_values) * len(opt_config.kappa_thresholds),
            "innerCvSplits": opt_config.inner_n_splits,
            "searchSpace": {
                "ouWindows": opt_config.ou_windows,
                "kValues": opt_config.k_values,
                "kappaThresholds": opt_config.kappa_thresholds,
            },
        },
        "bestParams": {
            "ouWindow": ou_w_opt,
            "k": k_opt,
            "kappaThreshold": kappa_th_opt,
            "meanTrainScore": best["meanTrainScore"],
        },
        "paperDefault": {
            "ouWindow": 252,
            "k": 1.0,
            "kappaThreshold": 0.5,
        },
        "fullBacktest": {
            "optimized": {
                "grossSharpe": opt_gross["sharpeRatio"],
                "netSharpe": opt_net["sharpeRatio"],
                "annualReturn": opt_net["annualReturn"],
                "maxDrawdown": opt_net["maxDrawdown"],
                "hitRate": opt_net["hitRate"],
                "totalCosts": opt_cost["summary"]["totalCosts"],
                "costReturnRatio": opt_cost["summary"]["costReturnRatio"],
                "turnover": opt_cost["summary"]["totalTurnover"],
            },
            "paperDefault": {
                "grossSharpe": def_gross["sharpeRatio"],
                "netSharpe": def_net["sharpeRatio"],
                "annualReturn": def_net["annualReturn"],
                "maxDrawdown": def_net["maxDrawdown"],
                "hitRate": def_net["hitRate"],
                "totalCosts": def_cost["summary"]["totalCosts"],
                "costReturnRatio": def_cost["summary"]["costReturnRatio"],
                "turnover": def_cost["summary"]["totalTurnover"],
            },
        },
        "walkForwardComparison": opt_results["default_comparison"],
        "windowDetails": opt_results["window_results"],
        "topGridResults": opt_results["grid_summary"][:10],
        "sensitivity": {
            param: [
                {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
                for r in results
            ]
            for param, results in sensitivity.items()
        },
        "costModel": {
            "feeBps": cost_model.fee_bps,
            "slippageBps": cost_model.slippage_bps,
            "eta": cost_model.eta,
        },
        "rollingParamStats_activeOnly": {
            "kappa": {"mean": round(float(rp_active["kappa"].mean()), 4),
                      "std": round(float(rp_active["kappa"].std()), 4)},
            "mu": {"mean": round(float(rp_active["mu"].mean()), 4),
                   "std": round(float(rp_active["mu"].std()), 4)},
            "sigma": {"mean": round(float(rp_active["sigma"].mean()), 4),
                      "std": round(float(rp_active["sigma"].std()), 4)},
        },
    }

    # Generate final metrics JSON using walk-forward results with optimized params
    wf_results = output_opt["results"]
    metrics_json = generate_metrics_json(wf_results, outer_config, custom_metrics, cost_model)

    return {
        "metrics_json": metrics_json,
        "opt_results": opt_results,
        "sensitivity": sensitivity,
        "wf_output": output_opt,
    }


def evaluate_pair(
    ticker1: str,
    ticker2: str,
    ou_window: int = 252,
    k: float = 0.25,
    kappa_threshold: float = 1.0,
    start_date: str = "2000-01-01",
    end_date: str = "2026-03-28",
    cost_model: TransactionCostModel | None = None,
    n_splits: int = 10,
) -> dict:
    """
    Evaluate a single pair with walk-forward validation using given parameters.

    Returns dict with pair name, full-backtest metrics, walk-forward results,
    rolling parameter stats, and cost breakdown.
    """
    cost_model = cost_model or COST_SCENARIOS["base"]
    loader = DataLoader()
    pair_name = f"{ticker1}/{ticker2}"

    print(f"\n{'='*60}")
    print(f"Evaluating pair: {pair_name}")
    print(f"{'='*60}")

    # Load data
    pair_data = loader.download_pair_data(ticker1, ticker2, start_date, end_date)
    print(f"  Data: {len(pair_data)} trading days, "
          f"{pair_data.index[0].date()} to {pair_data.index[-1].date()}")

    spread_df = loader.calculate_spread(pair_data, ticker1, ticker2)
    spread_values = spread_df["spread"].values
    print(f"  Spread: {len(spread_df)} points, beta={spread_df['beta'].iloc[0]:.4f}")

    # Full rolling backtest
    full_result = run_rolling_backtest(
        spread_values, ou_window=ou_window, k=k, kappa_threshold=kappa_threshold
    )
    full_returns = pd.Series(full_result["returns"])
    full_positions = pd.Series(full_result["allocations"])
    full_cost_result = apply_transaction_costs(full_returns, full_positions, cost_model)
    full_gross = compute_metrics(full_returns)
    full_net = compute_metrics(full_cost_result["net_returns"])

    print(f"  Full backtest: gross Sharpe={full_gross['sharpeRatio']:.4f}, "
          f"net Sharpe={full_net['sharpeRatio']:.4f}")
    print(f"  Annual return (net): {full_net['annualReturn']:.4f}, "
          f"MDD: {full_net['maxDrawdown']:.4f}")
    print(f"  Turnover: {full_cost_result['summary']['totalTurnover']:.1f}, "
          f"Cost/Return: {full_cost_result['summary']['costReturnRatio']:.4f}")

    # Walk-forward validation
    wf_config = BacktestConfig(n_splits=n_splits, min_train_size=252)
    validator = WalkForwardValidator(wf_config)
    wf_results = []

    for train_idx, test_idx in validator.split(spread_df):
        window_num = len(wf_results) + 1
        train_spread = spread_df.iloc[train_idx]["spread"].values
        test_spread = spread_df.iloc[test_idx]["spread"].values

        full_segment = np.concatenate([train_spread[-ou_window:], test_spread])
        bt = run_rolling_backtest(
            full_segment, ou_window=ou_window, k=k, kappa_threshold=kappa_threshold
        )

        returns_series = pd.Series(bt["returns"])
        positions_series = pd.Series(bt["allocations"])

        cost_result = apply_transaction_costs(returns_series, positions_series, cost_model)
        gross_metrics = compute_metrics(returns_series)
        net_metrics = compute_metrics(cost_result["net_returns"])

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
            total_trades=cost_result["summary"]["numTrades"],
            hit_rate=net_metrics["hitRate"],
        )
        wf_results.append(result)

    pos_windows = sum(1 for r in wf_results if r.net_sharpe > 0)
    avg_oos_sharpe = float(np.mean([r.net_sharpe for r in wf_results])) if wf_results else 0.0
    print(f"  Walk-forward: {pos_windows}/{len(wf_results)} positive windows, "
          f"avg OOS net Sharpe={avg_oos_sharpe:.4f}")

    # Rolling parameter stats (active days only)
    rp = full_result["rolling_params"]
    kappa_mask = rp["kappa"] >= kappa_threshold
    rp_active = rp[kappa_mask]
    active_pct = float(kappa_mask.sum()) / len(rp) * 100

    return {
        "pair": pair_name,
        "ticker1": ticker1,
        "ticker2": ticker2,
        "dataPoints": len(spread_df),
        "dateRange": f"{spread_df.index[0].date()} to {spread_df.index[-1].date()}",
        "hedgeRatio": round(float(spread_df["beta"].iloc[0]), 4),
        "fullBacktest": {
            "grossSharpe": full_gross["sharpeRatio"],
            "netSharpe": full_net["sharpeRatio"],
            "annualReturn": full_net["annualReturn"],
            "maxDrawdown": full_net["maxDrawdown"],
            "hitRate": full_net["hitRate"],
            "totalCosts": full_cost_result["summary"]["totalCosts"],
            "costReturnRatio": full_cost_result["summary"]["costReturnRatio"],
            "turnover": full_cost_result["summary"]["totalTurnover"],
        },
        "walkForward": {
            "windows": len(wf_results),
            "positiveWindows": pos_windows,
            "avgOosSharpe": round(avg_oos_sharpe, 4),
            "windowDetails": [
                {
                    "window": r.window,
                    "testStart": r.test_start,
                    "testEnd": r.test_end,
                    "grossSharpe": r.gross_sharpe,
                    "netSharpe": r.net_sharpe,
                    "annualReturn": r.annual_return,
                    "maxDrawdown": r.max_drawdown,
                    "totalTrades": r.total_trades,
                }
                for r in wf_results
            ],
        },
        "rollingParams": {
            "activeDaysPct": round(active_pct, 1),
            "kappa": {
                "mean": round(float(rp_active["kappa"].mean()), 4) if len(rp_active) > 0 else 0.0,
                "std": round(float(rp_active["kappa"].std()), 4) if len(rp_active) > 0 else 0.0,
            },
            "mu": {
                "mean": round(float(rp_active["mu"].mean()), 4) if len(rp_active) > 0 else 0.0,
                "std": round(float(rp_active["mu"].std()), 4) if len(rp_active) > 0 else 0.0,
            },
            "sigma": {
                "mean": round(float(rp_active["sigma"].mean()), 4) if len(rp_active) > 0 else 0.0,
                "std": round(float(rp_active["sigma"].std()), 4) if len(rp_active) > 0 else 0.0,
            },
        },
        "wf_results_raw": wf_results,
    }


def run_phase7(
    start_date: str = "2000-01-01",
    end_date: str = "2026-03-28",
) -> dict:
    """
    Phase 7: Multi-pair robustness testing.

    Tests the optimized strategy (from Phase 6) on multiple asset pairs to
    evaluate robustness. Uses the same parameters across all pairs:
        ou_window=252, k=0.25, kappa_threshold=1.0

    Pairs tested:
        - EWA/EWC: Country ETFs (Australia/Canada) — original pair
        - GLD/SLV: Precious metals ETFs (Gold/Silver)
        - TLT/IEF: Treasury bond ETFs (20Y/7-10Y)
        - XOM/CVX: Energy majors (Exxon/Chevron)
    """
    cost_model = COST_SCENARIOS["base"]

    # Optimized parameters from Phase 6
    ou_window = 252
    k = 0.25
    kappa_threshold = 1.0

    pairs = [
        ("EWA", "EWC", "Country ETFs (Australia/Canada)"),
        ("GLD", "SLV", "Precious Metals ETFs (Gold/Silver)"),
        ("TLT", "IEF", "Treasury Bond ETFs (20Y/7-10Y)"),
        ("XOM", "CVX", "Energy Majors (Exxon/Chevron)"),
    ]

    print("=" * 60)
    print("Phase 7: Multi-Pair Robustness Testing")
    print("=" * 60)
    print(f"Parameters: ou_window={ou_window}, k={k}, kappa_threshold={kappa_threshold}")
    print(f"Cost model: {cost_model.fee_bps} bps fee + {cost_model.slippage_bps} bps slippage")
    print(f"Pairs: {len(pairs)}")

    pair_results = []
    for ticker1, ticker2, description in pairs:
        try:
            result = evaluate_pair(
                ticker1, ticker2,
                ou_window=ou_window, k=k, kappa_threshold=kappa_threshold,
                start_date=start_date, end_date=end_date,
                cost_model=cost_model,
            )
            result["description"] = description
            pair_results.append(result)
        except Exception as e:
            print(f"\n  ERROR evaluating {ticker1}/{ticker2}: {e}")
            pair_results.append({
                "pair": f"{ticker1}/{ticker2}",
                "description": description,
                "error": str(e),
            })

    # Cross-pair comparison summary
    print("\n" + "=" * 60)
    print("Cross-Pair Comparison Summary")
    print("=" * 60)

    successful = [r for r in pair_results if "error" not in r]

    print(f"\n  {'Pair':<12} {'Net Sharpe':>11} {'Ann. Ret':>10} {'MDD':>8} "
          f"{'WF +/-':>8} {'Avg OOS':>9} {'Active%':>9}")
    print(f"  {'-'*67}")

    for r in successful:
        fb = r["fullBacktest"]
        wf = r["walkForward"]
        rp = r["rollingParams"]
        print(f"  {r['pair']:<12} {fb['netSharpe']:>11.4f} {fb['annualReturn']:>10.4f} "
              f"{fb['maxDrawdown']:>8.4f} {wf['positiveWindows']:>3}/{wf['windows']:<4} "
              f"{wf['avgOosSharpe']:>9.4f} {rp['activeDaysPct']:>8.1f}%")

    # Aggregate metrics across all successful pairs
    if successful:
        all_net_sharpes = [r["fullBacktest"]["netSharpe"] for r in successful]
        all_oos_sharpes = [r["walkForward"]["avgOosSharpe"] for r in successful]
        all_ann_returns = [r["fullBacktest"]["annualReturn"] for r in successful]
        all_mdds = [r["fullBacktest"]["maxDrawdown"] for r in successful]
        total_pos = sum(r["walkForward"]["positiveWindows"] for r in successful)
        total_win = sum(r["walkForward"]["windows"] for r in successful)

        print(f"\n  {'AVERAGE':<12} {np.mean(all_net_sharpes):>11.4f} "
              f"{np.mean(all_ann_returns):>10.4f} {np.mean(all_mdds):>8.4f} "
              f"{total_pos:>3}/{total_win:<4} {np.mean(all_oos_sharpes):>9.4f}")
        print(f"  {'MEDIAN':<12} {np.median(all_net_sharpes):>11.4f} "
              f"{np.median(all_ann_returns):>10.4f} {np.median(all_mdds):>8.4f}")
        print(f"  {'STD':<12} {np.std(all_net_sharpes):>11.4f} "
              f"{np.std(all_ann_returns):>10.4f} {np.std(all_mdds):>8.4f}")

    # Build primary metrics from the original pair (EWA/EWC) for backward compatibility
    primary = next((r for r in successful if r["pair"] == "EWA/EWC"), successful[0] if successful else None)
    primary_wf = primary["wf_results_raw"] if primary else []

    # Build cross-pair summary for customMetrics
    cross_pair_summary = []
    for r in successful:
        entry = {k: v for k, v in r.items() if k != "wf_results_raw"}
        cross_pair_summary.append(entry)

    custom_metrics = {
        "phase": "Phase 7 - Multi-Pair Robustness Testing",
        "dataSource": "ARF Data API",
        "params": {
            "ouWindow": ou_window,
            "k": k,
            "kappaThreshold": kappa_threshold,
        },
        "costModel": {
            "feeBps": cost_model.fee_bps,
            "slippageBps": cost_model.slippage_bps,
            "eta": cost_model.eta,
        },
        "pairsEvaluated": len(pairs),
        "pairsSuccessful": len(successful),
        "crossPairResults": cross_pair_summary,
        "aggregateStats": {
            "meanNetSharpe": round(float(np.mean(all_net_sharpes)), 4) if successful else 0.0,
            "medianNetSharpe": round(float(np.median(all_net_sharpes)), 4) if successful else 0.0,
            "stdNetSharpe": round(float(np.std(all_net_sharpes)), 4) if successful else 0.0,
            "meanAvgOosSharpe": round(float(np.mean(all_oos_sharpes)), 4) if successful else 0.0,
            "meanAnnualReturn": round(float(np.mean(all_ann_returns)), 4) if successful else 0.0,
            "meanMaxDrawdown": round(float(np.mean(all_mdds)), 4) if successful else 0.0,
            "totalPositiveWindows": total_pos if successful else 0,
            "totalWindows": total_win if successful else 0,
            "positiveWindowRate": round(total_pos / total_win, 4) if successful and total_win > 0 else 0.0,
        },
    }

    wf_config = BacktestConfig(n_splits=10, min_train_size=252)
    metrics_json = generate_metrics_json(primary_wf, wf_config, custom_metrics, cost_model)

    return {
        "metrics_json": metrics_json,
        "pair_results": pair_results,
    }


def main():
    output = run_phase7()
    metrics = output["metrics_json"]

    # Save metrics
    report_dir = Path("reports/cycle_7")
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {report_dir / 'metrics.json'}")

    # Print summary
    print("\n=== Phase 7: Multi-Pair Robustness Results ===")
    cm = metrics["customMetrics"]
    agg = cm["aggregateStats"]
    print(f"Pairs tested: {cm['pairsEvaluated']}, Successful: {cm['pairsSuccessful']}")
    print(f"Mean Net Sharpe (full): {agg['meanNetSharpe']:.4f}")
    print(f"Median Net Sharpe (full): {agg['medianNetSharpe']:.4f}")
    print(f"Mean Avg OOS Sharpe: {agg['meanAvgOosSharpe']:.4f}")
    print(f"Positive WF windows: {agg['totalPositiveWindows']}/{agg['totalWindows']} "
          f"({agg['positiveWindowRate']:.1%})")

    wf = metrics["walkForward"]
    print(f"\nPrimary pair walk-forward:")
    print(f"  Windows: {wf['windows']}, Positive: {wf['positiveWindows']}, "
          f"Avg OOS Sharpe: {wf['avgOosSharpe']:.4f}")
    print(f"  Gross Sharpe: {metrics['sharpeRatio']:.4f}")
    tc = metrics["transactionCosts"]
    print(f"  Net Sharpe: {tc['netSharpe']:.4f}")


if __name__ == "__main__":
    main()
