"""
Phase 6: Hyperparameter optimization for the control-theoretic pairs trading strategy.

Optimizes the OU lookback window and control gain using walk-forward
cross-validation on in-sample data only. Per reproduction mode rules,
the search grid is centered on the paper's default parameters.

Key hyperparameters:
    - ou_window: Lookback period for OU parameter estimation (paper default: 252)
    - k: Control gain scaling factor (paper default: 1.0)
    - kappa_threshold: Minimum mean-reversion speed to trade (paper default: 0.5)
"""
import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.model import run_rolling_backtest
from src.backtest import (
    BacktestConfig,
    TransactionCostModel,
    WalkForwardValidator,
    apply_transaction_costs,
    compute_metrics,
)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    # Search grids (paper-near neighborhoods)
    ou_windows: list[int] = None
    k_values: list[float] = None
    kappa_thresholds: list[float] = None
    # Optimization target
    target_metric: str = "netSharpe"  # optimize net Sharpe by default
    # Walk-forward settings for inner CV
    inner_n_splits: int = 5
    min_train_size: int = 504  # 2 years minimum for inner CV
    # Cost model for evaluation
    cost_model: TransactionCostModel = None

    def __post_init__(self):
        if self.ou_windows is None:
            # Paper default 252 (~1 year), near neighbors
            self.ou_windows = [126, 189, 252, 315, 378]
        if self.k_values is None:
            # Paper default 1.0, near neighbors
            self.k_values = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        if self.kappa_thresholds is None:
            # Paper default 0.5, near neighbors
            self.kappa_thresholds = [0.25, 0.5, 1.0]
        if self.cost_model is None:
            self.cost_model = TransactionCostModel(fee_bps=10.0, slippage_bps=5.0, eta=0.0)


def evaluate_params(
    spread: np.ndarray,
    ou_window: int,
    k: float,
    kappa_threshold: float = 0.5,
    cost_model: TransactionCostModel = None,
) -> dict:
    """
    Evaluate a single parameter combination on a spread series.

    Returns dict with gross and net metrics.
    """
    cost_model = cost_model or TransactionCostModel()

    if len(spread) <= ou_window + 10:
        return {
            "grossSharpe": 0.0,
            "netSharpe": 0.0,
            "annualReturn": 0.0,
            "maxDrawdown": 0.0,
            "hitRate": 0.0,
            "totalTrades": 0,
            "turnover": 0.0,
        }

    result = run_rolling_backtest(
        spread, ou_window=ou_window, k=k, kappa_threshold=kappa_threshold
    )

    returns = pd.Series(result["returns"])
    positions = pd.Series(result["allocations"])

    gross_metrics = compute_metrics(returns)
    cost_result = apply_transaction_costs(returns, positions, cost_model)
    net_metrics = compute_metrics(cost_result["net_returns"])

    return {
        "grossSharpe": gross_metrics["sharpeRatio"],
        "netSharpe": net_metrics["sharpeRatio"],
        "annualReturn": net_metrics["annualReturn"],
        "maxDrawdown": net_metrics["maxDrawdown"],
        "hitRate": net_metrics["hitRate"],
        "totalTrades": cost_result["summary"]["numTrades"],
        "turnover": cost_result["summary"]["totalTurnover"],
    }


def optimize_walk_forward(
    spread_df: pd.DataFrame,
    opt_config: OptimizationConfig = None,
    backtest_config: BacktestConfig = None,
) -> dict:
    """
    Optimize hyperparameters using walk-forward cross-validation.

    For each walk-forward window:
      1. Use the training set to evaluate all parameter combinations
      2. Select the best parameters based on in-sample net Sharpe
      3. Apply best parameters to the test set for out-of-sample evaluation

    Also evaluates the paper's default parameters on each test set for comparison.

    Args:
        spread_df: DataFrame with 'spread' column and DatetimeIndex.
        opt_config: OptimizationConfig with search grids.
        backtest_config: BacktestConfig for walk-forward splits.

    Returns:
        Dict with optimization results including:
            - 'best_params': Overall best parameters across all windows
            - 'window_results': Per-window results (train/test metrics)
            - 'grid_summary': Aggregated performance for each parameter combo
            - 'default_comparison': Paper default vs optimized performance
    """
    opt_config = opt_config or OptimizationConfig()
    backtest_config = backtest_config or BacktestConfig(
        n_splits=opt_config.inner_n_splits,
        min_train_size=opt_config.min_train_size,
    )

    spread_values = spread_df["spread"].values
    validator = WalkForwardValidator(backtest_config)

    # Build parameter grid
    param_grid = list(itertools.product(
        opt_config.ou_windows,
        opt_config.k_values,
        opt_config.kappa_thresholds,
    ))

    print(f"Optimization grid: {len(param_grid)} combinations "
          f"({len(opt_config.ou_windows)} windows × "
          f"{len(opt_config.k_values)} k values × "
          f"{len(opt_config.kappa_thresholds)} thresholds)")

    # Track per-combo aggregated scores across all windows
    grid_scores = {combo: [] for combo in param_grid}
    window_results = []
    paper_default = (252, 1.0, 0.5)

    window_num = 0
    for train_idx, test_idx in validator.split(spread_df):
        window_num += 1
        train_spread = spread_values[train_idx[0]:train_idx[-1] + 1]
        test_spread = spread_values[test_idx[0]:test_idx[-1] + 1]

        train_dates = spread_df.index[train_idx]
        test_dates = spread_df.index[test_idx]

        print(f"\n  Window {window_num}: train {train_dates[0].date()} to "
              f"{train_dates[-1].date()}, test {test_dates[0].date()} to "
              f"{test_dates[-1].date()}")

        # Evaluate all parameter combos on training data
        best_train_score = -np.inf
        best_combo = paper_default
        combo_results = {}

        for combo in param_grid:
            ou_w, k_val, kappa_th = combo
            if len(train_spread) <= ou_w + 10:
                continue

            train_metrics = evaluate_params(
                train_spread, ou_w, k_val, kappa_th, opt_config.cost_model
            )
            score = train_metrics[opt_config.target_metric]
            combo_results[combo] = train_metrics
            grid_scores[combo].append(score)

            if score > best_train_score:
                best_train_score = score
                best_combo = combo

        # Evaluate best params on test data (out-of-sample)
        ou_w_best, k_best, kappa_th_best = best_combo
        # Need enough context for OU estimation on test set
        context_start = max(0, test_idx[0] - ou_w_best)
        test_with_context = spread_values[context_start:test_idx[-1] + 1]
        oos_metrics = evaluate_params(
            test_with_context, ou_w_best, k_best, kappa_th_best, opt_config.cost_model
        )

        # Also evaluate paper defaults on test data for comparison
        context_start_def = max(0, test_idx[0] - 252)
        test_with_context_def = spread_values[context_start_def:test_idx[-1] + 1]
        default_metrics = evaluate_params(
            test_with_context_def, 252, 1.0, 0.5, opt_config.cost_model
        )

        window_results.append({
            "window": window_num,
            "trainStart": str(train_dates[0].date()),
            "trainEnd": str(train_dates[-1].date()),
            "testStart": str(test_dates[0].date()),
            "testEnd": str(test_dates[-1].date()),
            "bestParams": {
                "ouWindow": ou_w_best,
                "k": k_best,
                "kappaThreshold": kappa_th_best,
            },
            "trainScore": round(best_train_score, 4),
            "oosMetrics": {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in oos_metrics.items()},
            "defaultOosMetrics": {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in default_metrics.items()},
        })

        print(f"    Best params: ou_window={ou_w_best}, k={k_best}, "
              f"kappa_th={kappa_th_best}")
        print(f"    Train {opt_config.target_metric}: {best_train_score:.4f}")
        print(f"    OOS net Sharpe: {oos_metrics['netSharpe']:.4f} "
              f"(default: {default_metrics['netSharpe']:.4f})")

    # Aggregate grid performance
    grid_summary = []
    for combo, scores in grid_scores.items():
        if not scores:
            continue
        ou_w, k_val, kappa_th = combo
        grid_summary.append({
            "ouWindow": ou_w,
            "k": k_val,
            "kappaThreshold": kappa_th,
            "meanTrainScore": round(float(np.mean(scores)), 4),
            "stdTrainScore": round(float(np.std(scores)), 4),
            "minTrainScore": round(float(np.min(scores)), 4),
            "maxTrainScore": round(float(np.max(scores)), 4),
            "nWindows": len(scores),
        })

    # Sort by mean train score descending
    grid_summary.sort(key=lambda x: x["meanTrainScore"], reverse=True)

    # Overall best parameters (by mean train score)
    if grid_summary:
        top = grid_summary[0]
        overall_best = {
            "ouWindow": top["ouWindow"],
            "k": top["k"],
            "kappaThreshold": top["kappaThreshold"],
            "meanTrainScore": top["meanTrainScore"],
        }
    else:
        overall_best = {"ouWindow": 252, "k": 1.0, "kappaThreshold": 0.5, "meanTrainScore": 0.0}

    # Comparison: optimized vs paper default OOS performance
    opt_oos_sharpes = [w["oosMetrics"]["netSharpe"] for w in window_results]
    def_oos_sharpes = [w["defaultOosMetrics"]["netSharpe"] for w in window_results]

    comparison = {
        "optimized": {
            "avgOosSharpe": round(float(np.mean(opt_oos_sharpes)), 4) if opt_oos_sharpes else 0.0,
            "positiveWindows": sum(1 for s in opt_oos_sharpes if s > 0),
            "totalWindows": len(opt_oos_sharpes),
        },
        "paperDefault": {
            "avgOosSharpe": round(float(np.mean(def_oos_sharpes)), 4) if def_oos_sharpes else 0.0,
            "positiveWindows": sum(1 for s in def_oos_sharpes if s > 0),
            "totalWindows": len(def_oos_sharpes),
        },
    }

    print(f"\n=== Optimization Summary ===")
    print(f"Overall best params: ou_window={overall_best['ouWindow']}, "
          f"k={overall_best['k']}, kappa_th={overall_best['kappaThreshold']}")
    print(f"Optimized avg OOS Sharpe: {comparison['optimized']['avgOosSharpe']:.4f} "
          f"({comparison['optimized']['positiveWindows']}/{comparison['optimized']['totalWindows']} positive)")
    print(f"Default avg OOS Sharpe: {comparison['paperDefault']['avgOosSharpe']:.4f} "
          f"({comparison['paperDefault']['positiveWindows']}/{comparison['paperDefault']['totalWindows']} positive)")

    return {
        "best_params": overall_best,
        "window_results": window_results,
        "grid_summary": grid_summary,
        "default_comparison": comparison,
    }


def run_sensitivity_analysis(
    spread: np.ndarray,
    base_ou_window: int = 252,
    base_k: float = 1.0,
    base_kappa_threshold: float = 0.5,
    cost_model: TransactionCostModel = None,
) -> dict:
    """
    Run one-at-a-time sensitivity analysis around base parameters.

    Varies each parameter while holding others at base values to understand
    marginal effects.

    Returns:
        Dict with sensitivity results for each parameter.
    """
    cost_model = cost_model or TransactionCostModel()

    results = {}

    # Sensitivity to ou_window
    ou_windows = [63, 126, 189, 252, 315, 378, 504]
    ou_results = []
    for ow in ou_windows:
        if len(spread) <= ow + 10:
            continue
        m = evaluate_params(spread, ow, base_k, base_kappa_threshold, cost_model)
        ou_results.append({"ouWindow": ow, **m})
    results["ouWindow"] = ou_results

    # Sensitivity to k
    k_values = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    k_results = []
    for k in k_values:
        m = evaluate_params(spread, base_ou_window, k, base_kappa_threshold, cost_model)
        k_results.append({"k": k, **m})
    results["k"] = k_results

    # Sensitivity to kappa_threshold
    kappa_vals = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    kappa_results = []
    for kt in kappa_vals:
        m = evaluate_params(spread, base_ou_window, base_k, kt, cost_model)
        kappa_results.append({"kappaThreshold": kt, **m})
    results["kappaThreshold"] = kappa_results

    return results
