# Phase 7 Technical Findings: Multi-Pair Robustness Testing

## Summary

Tested the optimized control-theoretic pairs trading strategy (from Phase 6) on four asset pairs to evaluate robustness. Parameters held constant across all pairs: ou_window=252, k=0.25, kappa_threshold=1.0 with base cost model (10 bps fee + 5 bps slippage).

## Pairs Tested

| Pair    | Description                        | Data Points | Date Range              |
|---------|-------------------------------------|-------------|-------------------------|
| EWA/EWC | Country ETFs (Australia/Canada)    | 6,539       | 2000-03-28 to 2026-03-27 |
| GLD/SLV | Precious Metals ETFs (Gold/Silver) | 5,010       | 2006-04-28 to 2026-03-27 |
| TLT/IEF | Treasury Bond ETFs (20Y/7-10Y)     | 5,954       | 2002-07-30 to 2026-03-27 |
| XOM/CVX | Energy Majors (Exxon/Chevron)      | 6,539       | 2000-03-28 to 2026-03-27 |

## Cross-Pair Results

### Full Backtest Metrics

| Pair    | Gross Sharpe | Net Sharpe | Ann. Return | Max DD   | Turnover | Cost/Return |
|---------|-------------|------------|-------------|----------|----------|-------------|
| EWA/EWC | 1.10        | 0.58       | +0.58%      | -3.25%   | 87.5     | 0.48        |
| GLD/SLV | -0.14       | -0.59      | -0.69%      | -12.52%  | 65.8     | -3.23       |
| TLT/IEF | -0.10       | -1.14      | -0.66%      | -13.87%  | 91.0     | -10.61      |
| XOM/CVX | 0.39        | -0.30      | -0.25%      | -10.06%  | 94.0     | 1.78        |

### Walk-Forward Validation

| Pair    | Positive/Total | Avg OOS Sharpe | Active Days % |
|---------|---------------|----------------|---------------|
| EWA/EWC | 6/9           | +0.54          | 97.5%         |
| GLD/SLV | 3/9           | -0.56          | 91.3%         |
| TLT/IEF | 0/9           | -1.22          | 93.4%         |
| XOM/CVX | 2/9           | -0.41          | 94.6%         |

### Aggregate Statistics

| Metric               | Value  |
|----------------------|--------|
| Mean Net Sharpe      | -0.36  |
| Median Net Sharpe    | -0.44  |
| Std Net Sharpe       | 0.62   |
| Total Positive WF    | 11/36 (30.6%) |
| Mean Avg OOS Sharpe  | -0.41  |

## Key Findings

### 1. Strategy is pair-dependent, not universally robust

Only EWA/EWC produces a positive net Sharpe ratio. The other three pairs show negative performance after transaction costs. This indicates the strategy's profitability depends critically on the specific pair's cointegration properties, not just on the general framework.

### 2. Gross vs net performance reveals cost sensitivity

XOM/CVX has a positive gross Sharpe (0.39) but negative net Sharpe (-0.30), confirming that transaction costs remain the dominant factor even at the reduced control gain of k=0.25. The control law generates continuous position adjustments, creating persistent cost drag.

### 3. TLT/IEF: poorest performer despite high mean-reversion signal

The treasury pair shows 93.4% active days (kappa > 1.0) but the worst net Sharpe (-1.14). The spread has very low volatility (sigma=0.07, lowest of all pairs), meaning OU-estimated "mean reversion" produces tiny absolute price movements insufficient to cover even small transaction costs.

### 4. GLD/SLV: unstable cointegration relationship

The gold/silver ratio shows intermittent mean-reversion (3/9 windows positive) with no consistent pattern. The 2018-2022 period was particularly poor, likely reflecting the decoupling of precious metal relative prices during monetary policy shifts.

### 5. XOM/CVX: early-period strength, recent degradation

The energy pair was profitable in windows 2-3 (2006-2011, coinciding with the energy price supercycle) but has been consistently unprofitable since 2016. This mirrors the EWA/EWC post-2021 degradation pattern, suggesting structural shifts in pair relationships over time.

### 6. High active-day percentages across all pairs

All pairs show >91% active trading days (kappa exceeds threshold), meaning the kappa filter does not effectively screen out non-mean-reverting regimes for these pairs. The kappa threshold may need pair-specific calibration.

## Rolling OU Parameter Comparison

| Pair    | Kappa (mean) | Kappa (std) | Mu (mean) | Sigma (mean) |
|---------|-------------|-------------|-----------|--------------|
| EWA/EWC | 12.0        | 8.8         | -0.98     | 0.170        |
| GLD/SLV | 6.1         | 4.2         | 2.26      | 0.164        |
| TLT/IEF | 5.8         | 3.8         | -1.20     | 0.070        |
| XOM/CVX | 7.5         | 5.3         | 0.91      | 0.135        |

EWA/EWC has the highest mean-reversion speed (kappa=12.0) and highest spread volatility (sigma=0.17), both favorable for the control strategy.

## Implications for the Paper

1. **The paper's theoretical guarantee** (positive expected log growth under mean-reverting spread) holds for EWA/EWC but not universally, because:
   - The guarantee assumes true mean-reversion; estimated parameters introduce noise
   - Transaction costs negate small theoretical gains on low-volatility spreads
   - Pair relationships are non-stationary over 20+ year horizons

2. **Pair selection matters more than parameter optimization**: The difference between the best pair (EWA/EWC, +0.58 net Sharpe) and worst pair (TLT/IEF, -1.14) far exceeds the Phase 6 optimization improvement (+0.34 Sharpe from default to optimized on EWA/EWC).

3. **The framework is not "universally applicable"**: While the paper presents a general framework, practical profitability requires pairs with strong, persistent cointegration and sufficient spread volatility.

## Open Questions (Phase 7)

- Q21: Would pair-specific parameter optimization improve results for non-EWA/EWC pairs?
- Q22: Is the static hedge ratio a limiting factor? Rolling hedge ratios may better capture time-varying relationships for some pairs.
- Q23: Should the kappa threshold be calibrated per-pair based on spread volatility?
- Q24: Can a pair selection filter (e.g., cointegration test p-value) pre-screen viable pairs before deploying the strategy?
