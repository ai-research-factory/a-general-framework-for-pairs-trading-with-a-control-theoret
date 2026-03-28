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

## Phase 5

12. **High cost/return ratio (0.75)**: Transaction costs consume ~75% of gross alpha at the base scenario (15 bps total). This raises the question of whether the continuous feedback control law is practical for real trading, or whether a discrete-signal variant (trade only when deviation exceeds a threshold) would reduce turnover and improve net performance.

13. **Market impact model calibration**: The square-root impact coefficient eta=0.001 renders the strategy deeply unprofitable. The appropriate value of eta depends on the assets' liquidity (ADV, bid-ask spread). For liquid ETFs like EWA/EWC, a proper calibration against historical trade data or VWAP benchmarks would be needed.

14. **Break-even cost sensitivity**: The strategy breaks even at ~20 bps total cost. Retail commission-free brokers (0 bps fee) with tight ETF spreads (2-3 bps slippage) could make this viable, but the paper does not discuss implementability at different cost levels.

15. **Turnover reduction strategies**: The strategy trades on 98.5% of days. Possible improvements: (a) increase kappa threshold to trade only in strongly mean-reverting periods, (b) add a minimum position change threshold, (c) reduce control gain k. These should be explored in Phase 6.

## Phase 6

16. **Overfitting risk in walk-forward optimization**: Inner CV (4 windows) selects optimized params with avg OOS Sharpe 0.13, while paper default achieves 0.36 on the same inner windows. However, the 9-window outer evaluation shows similar performance (0.43 vs 0.42). This discrepancy suggests the inner CV window count may be too small for robust parameter selection. More windows or a different selection criterion (e.g., worst-window Sharpe) could help.

17. **Kappa threshold at 0.0 produces extreme turnover**: When kappa_threshold=0.0, turnover explodes to ~5 billion due to non-mean-reverting periods generating huge mu estimates and consequently massive allocations. This reveals a fragility in the OU estimator when kappa → 0 (mu = a/(kappa*dt) blows up). A cap on allocation magnitude would be a practical safeguard.

18. **Optimal k much lower than paper default**: The optimization consistently selects k=0.25 over k=1.0, a 4x reduction. This aligns with the Phase 5 finding that transaction costs dominate. The paper's theoretical analysis assumes frictionless markets where higher k increases expected growth. In practice, the cost of continuous adjustment overwhelms the marginal alpha from larger positions.

19. **Diminishing returns from longer OU windows**: Sensitivity analysis shows net Sharpe plateau at 0.58-0.62 for windows 252-504 days. Longer windows reduce noise but lose responsiveness to regime changes. The paper's 252-day default is near-optimal.

20. **Post-2021 EWA/EWC degradation is parameter-independent**: Walk-forward windows 8-9 show negative Sharpe regardless of parameter choice. No combination in the search grid produces positive OOS performance in 2021-2026, confirming that the cointegration breakdown is structural rather than a tuning issue.

## Phase 7

21. **Pair-specific parameter optimization**: The Phase 6 optimized parameters (k=0.25, kappa_threshold=1.0) were tuned on EWA/EWC. Would pair-specific optimization improve performance for GLD/SLV, TLT/IEF, or XOM/CVX? The current negative results may partly reflect parameter mismatch rather than fundamental unsuitability.

22. **Static vs rolling hedge ratio impact on non-EWA/EWC pairs**: All four pairs use static (full-sample) hedge ratios, introducing look-ahead bias. For pairs with time-varying relationships (e.g., GLD/SLV during monetary policy shifts), rolling hedge ratios may better capture the cointegration structure and improve performance.

23. **Kappa threshold calibration per pair**: All pairs show >91% active days despite kappa_threshold=1.0, meaning the filter is not selective enough. TLT/IEF has kappa=5.8 (active) but sigma=0.07, too low for profitable trading after costs. A composite filter (e.g., kappa * sigma > threshold) could better identify tradeable regimes.

24. **Pair pre-selection via cointegration testing**: The paper's framework assumes the spread is mean-reverting, but 3 of 4 tested pairs fail to produce positive net returns. A systematic pair selection step — e.g., requiring ADF test p-value < 0.05 on a rolling basis — could filter out unsuitable pairs before capital allocation.

## Phase 8

25. **Bounded spread generalization to other pairs**: The bounded_a5 spread improved EWA/EWC OOS Sharpe by 82% over linear (0.98 vs 0.54). Would this improvement hold for non-EWA/EWC pairs where the linear spread already fails? If the bounded transformation only helps already-profitable pairs, its practical value is limited.

26. **Optimal alpha derivation from OU parameters**: Is there a theoretical relationship between the optimal logistic alpha parameter and the OU process parameters (kappa, sigma)? Currently alpha=5 was found empirically. A principled derivation could make this a self-tuning parameter.

27. **Bounded spread overfitting risk**: The bounded spread's higher OOS Sharpe (0.98) uses the same walk-forward framework as the linear baseline. However, the standardization step (mean/std of the linear spread) uses full-sample statistics, introducing mild look-ahead bias. A rolling standardization should be tested.

28. **Spread function ensemble**: Would combining allocations from multiple spread functions (e.g., 50% linear + 50% bounded) provide diversification benefits? The correlation structure between different spread-derived signals is unknown.
