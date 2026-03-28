# Open Questions

## Phase 1

1. **Optimal control gain k**: The paper derives an optimal control law but the exact relationship between k and the OU parameters (κ, σ) needs further investigation. In Phase 6, k should be optimized as a function of estimated parameters.

2. **Parameter estimation bias**: The OLS estimator for κ is known to have a downward bias for small samples. For the rolling-window estimation in Phase 3, the lookback window size will affect estimation quality.

3. **Spread definition generality**: Phase 1 uses a simple linear spread. The paper's framework allows arbitrary (potentially non-linear) spread functions, which will be explored in Phase 8.

4. **Portfolio value model simplicity**: The current backtest uses additive PnL (h * ds) rather than multiplicative returns. For large allocations, this could diverge from realistic portfolio dynamics. This should be addressed when moving to real data.

## Phase 2

5. **Adjusted close vs. close**: The ARF Data API returns "close" prices rather than "adjusted close". For ETFs like EWA/EWC, this may not properly account for dividend adjustments, which could introduce small errors in the log-price spread calculation.

6. **Static hedge ratio**: The current implementation estimates a single hedge ratio (beta) over the entire sample period. This introduces look-ahead bias for backtesting. Phase 3 will implement rolling estimation to address this.

7. **Data start date**: Requested data from 2000-01-01, but ARF API returns from 2000-03-28 (the maximum lookback appears to be ~26 years). The paper does not specify an exact start date, so this should be sufficient.

## Phase 3

8. **Rolling vs static hedge ratio trade-off**: Rolling hedge ratio eliminates look-ahead bias but introduces non-stationarity in the spread, causing the strategy to lose money. The static hedge ratio produces better results but has look-ahead bias. A Kalman filter-based dynamic beta estimation could resolve this trade-off.

9. **Kappa threshold for regime detection**: The paper's theoretical guarantee assumes a mean-reverting spread (positive kappa). We use kappa >= 0.5 as a heuristic threshold to filter non-mean-reverting periods. The paper does not specify a regime-switching mechanism. More rigorous approaches (e.g., ADF test p-value) could be explored.

10. **Recent performance degradation (2021-2026)**: Walk-forward windows 8-9 show negative Sharpe ratios, suggesting the EWA/EWC cointegration relationship may have weakened post-COVID. This motivates multi-pair testing (Phase 7) and regime analysis (Phase 9).

11. **OLS kappa estimation edge cases**: When the spread is not mean-reverting, the OLS estimator produces kappa near zero and mu can become very large (a/(kappa*dt) blows up). The current fallback uses the sample mean when kappa*dt < 1e-12. A more robust estimator (MLE) could handle these edge cases better.
