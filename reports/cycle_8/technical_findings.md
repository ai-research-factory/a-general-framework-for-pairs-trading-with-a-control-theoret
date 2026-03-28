# Phase 8: Alternative Spread Function Evaluation

## Summary

Phase 8 validates the paper's claim that its control-theoretic framework is general and applicable to arbitrary spread definitions. We implemented three non-linear spread functions in addition to the baseline linear log-price spread, and evaluated all six variants on EWA/EWC using the Phase 6 optimized parameters (ou_window=252, k=0.25, kappa_threshold=1.0).

## Spread Functions Tested

| # | Name | Definition | Description |
|---|------|-----------|-------------|
| 1 | **linear** (baseline) | `s = log(P1) - beta * log(P2)` | Standard OLS hedge ratio spread |
| 2 | **log_ratio** | `s = log(P1/P2)` | Fixed beta=1, no estimation needed |
| 3 | **bounded_a10** | `s = logistic(10 * z_std)` | Logistic transform, alpha=10 |
| 4 | **bounded_a5** | `s = logistic(5 * z_std)` | Logistic transform, alpha=5 |
| 5 | **power_p05** | `s = sign(z) * |z|^0.5` | Square-root compression |
| 6 | **power_p15** | `s = sign(z) * |z|^1.5` | Super-linear amplification |

Where `z = log(P1) - beta * log(P2)` is the standard linear spread, and `z_std` is its standardized form.

## Results

### Full Backtest Comparison

| Spread | Gross Sharpe | Net Sharpe | Ann. Return | Max DD | WF +/- | Avg OOS Sharpe |
|--------|-------------|-----------|-------------|--------|--------|----------------|
| bounded_a5 | 0.7932 | 0.7352 | 5.59% | -10.49% | 8/9 | **0.9823** |
| power_p15 | 1.0681 | 0.6970 | 0.99% | -3.70% | 7/9 | 0.7319 |
| bounded_a10 | 0.6737 | 0.6326 | 6.54% | -12.35% | 5/9 | 0.6304 |
| linear | 1.1015 | 0.5778 | 0.58% | -3.25% | 6/9 | 0.5374 |
| log_ratio | 1.1028 | 0.5948 | 0.57% | -2.90% | 6/9 | 0.5293 |
| power_p05 | 1.0983 | 0.1369 | 0.07% | -3.22% | 6/9 | -0.0353 |

### Ranking by Out-of-Sample Net Sharpe

1. **bounded_a5**: 0.9823 (82% improvement over linear baseline)
2. **power_p15**: 0.7319 (36% improvement)
3. **bounded_a10**: 0.6304 (17% improvement)
4. **linear**: 0.5374 (baseline)
5. **log_ratio**: 0.5293 (-2%)
6. **power_p05**: -0.0353 (-107%)

## Key Findings

### 1. Non-linear spreads can outperform the linear baseline

The bounded spread with alpha=5 achieves the highest walk-forward performance (avg OOS Sharpe 0.98 vs. 0.54 for linear), with 8/9 windows positive. This supports the paper's generality claim.

### 2. Bounding is the most effective transformation

Both logistic-bounded spreads outperform the linear baseline on OOS metrics. The bounded transformation compresses extreme deviations, which has two effects:
- Reduces allocation size during large deviations (natural position sizing)
- Makes the OU parameter estimates more stable by keeping the spread in a fixed range

### 3. Lower alpha (gentler bounding) works better

bounded_a5 (alpha=5) outperforms bounded_a10 (alpha=10). With alpha=10, the logistic function saturates too quickly, losing information about deviation magnitude. Alpha=5 provides a better balance between bounding and information preservation.

### 4. Square-root power (p=0.5) destroys profitability

The power_p05 spread amplifies small deviations (|z|^0.5 > |z| for |z| < 1), causing the OU estimator to find spurious mean-reversion in noise. This leads to excessive trading on small, non-meaningful spread movements.

### 5. Super-linear power (p=1.5) improves performance

power_p15 amplifies large deviations while suppressing small ones. This acts as a natural noise filter, only generating significant allocation when the spread has moved meaningfully. The 36% improvement in OOS Sharpe supports this interpretation.

### 6. Gross vs net divergence reveals cost sensitivity

The bounded spreads show higher annual returns but also higher max drawdowns. Their higher gross-net spread (bounded_a5: 0.79 gross -> 0.74 net) suggests they trade more efficiently than the linear spread (1.10 gross -> 0.58 net, where costs erode 47% of alpha).

### 7. Log ratio (beta=1) performs similarly to linear

Forcing beta=1 barely changes results (0.53 vs 0.54 OOS Sharpe), suggesting that for EWA/EWC the hedge ratio is naturally close to 1.0 and precise estimation adds minimal value.

## Implications for the Paper's Claims

The paper claims its framework works with "arbitrary, potentially non-linear, spread definitions." Our results **partially support this claim**:

- **Supported**: The framework produces positive expected growth with multiple non-linear spread functions, confirming generality.
- **Nuanced**: Not all non-linear functions improve performance. The choice of transformation matters significantly (bounded_a5 at 0.98 vs. power_p05 at -0.04 OOS Sharpe).
- **Important caveat**: These results are labeled `implementation-improvement` since the paper does not specify these exact non-linear functions. The finding that bounded spreads outperform is an empirical contribution beyond strict paper reproduction.

## Open Questions (Q25-Q28)

25. Would the bounded spread improvement hold for non-EWA/EWC pairs (where linear spread already fails)?
26. Is there an optimal alpha for the bounded spread that can be derived theoretically from OU parameters?
27. Does the bounded spread's better OOS performance survive a longer out-of-sample period, or is it overfitting to the particular structure of EWA/EWC?
28. Would combining spread functions (ensemble of linear + bounded) provide diversification benefits?
