# Cycle 5 Technical Findings: Transaction Cost Model

## Summary

Phase 5 introduces a realistic, multi-component transaction cost model into the backtest engine. The previous implementation used a simple flat bps cost on position changes. The new model decomposes costs into three components: proportional commission fees, linear slippage, and square-root market impact.

## Implementation

### TransactionCostModel (`src/backtest.py`)

New dataclass with three cost components:

1. **Proportional fee**: Commission charged as basis points on trade notional (|Δh|)
2. **Linear slippage**: Bid-ask spread cost as basis points on trade notional
3. **Market impact**: Square-root model `η × √(|Δh|)`, capturing temporary price impact from large trades (Almgren & Chriss, 2000)

Total cost per step: `(fee_bps + slippage_bps) / 10000 × |Δh| + η × √(|Δh|)`

### apply_transaction_costs() function

Returns a detailed breakdown:
- Net returns after all costs
- Per-step cost decomposition (fee, slippage, impact)
- Turnover series
- Aggregate summary statistics (total costs, cost/return ratio, avg cost per trade)

### Backward Compatibility

The original `calculate_costs()` function is preserved. Tests confirm that `apply_transaction_costs` with eta=0 produces identical results to the legacy function.

## Results: Cost Scenario Comparison (Full Backtest, EWA/EWC)

| Scenario | Fee (bps) | Slip (bps) | eta | Net Sharpe | Annual Return | Max DD | Total Costs | Cost/Return |
|----------|-----------|------------|-----|------------|---------------|--------|-------------|-------------|
| Zero | 0 | 0 | 0 | 0.95 | 3.17% | -6.9% | 0.000 | 0.00 |
| Low | 5 | 3 | 0 | 0.57 | 1.87% | -12.2% | 0.316 | 0.40 |
| **Base** | **10** | **5** | **0** | **0.24** | **0.75%** | **-19.4%** | **0.593** | **0.75** |
| High | 15 | 10 | 0 | -0.23 | -0.84% | -37.1% | 0.988 | 1.25 |
| Base+Impact | 10 | 5 | 0.001 | -1.36 | -4.58% | -69.6% | 1.946 | 2.46 |

## Walk-Forward Results (Base Cost Model: 10 bps fee + 5 bps slippage)

| Window | Period | Gross Sharpe | Net Sharpe | Trades |
|--------|--------|-------------|------------|--------|
| 1 | 2003-10 to 2006-04 | 1.37 | 0.79 | 626 |
| 2 | 2006-04 to 2008-10 | 1.37 | 0.96 | 626 |
| 3 | 2008-10 to 2011-04 | 2.00 | 1.62 | 626 |
| 4 | 2011-04 to 2013-10 | 0.38 | -0.43 | 591 |
| 5 | 2013-10 to 2016-04 | 1.24 | 0.68 | 626 |
| 6 | 2016-04 to 2018-10 | 1.85 | 1.06 | 626 |
| 7 | 2018-10 to 2021-03 | 1.79 | 1.39 | 626 |
| 8 | 2021-04 to 2023-10 | -0.00 | -0.92 | 618 |
| 9 | 2023-10 to 2026-03 | -0.74 | -1.36 | 576 |

- **Avg OOS Sharpe (gross)**: 1.03
- **Avg OOS Sharpe (net)**: 0.42
- **Positive windows**: 6/9 (67%)

## Cost Breakdown (Full Backtest)

- Total turnover: 395.17 (sum of |Δposition| over all days)
- Number of trades: 6,192
- Average cost per trade: 0.000096
- Fee costs: 0.3952 (66.7% of total)
- Slippage costs: 0.1976 (33.3% of total)
- **Cost/Return ratio: 0.748** — costs consume ~75% of gross returns

## Key Observations

1. **Transaction costs are the dominant factor**: The strategy's gross Sharpe of 0.95 drops to 0.24 net of base costs (10+5 bps). The cost/return ratio of 0.75 means 75% of gross alpha is consumed by costs.

2. **Strategy is unprofitable at institutional cost levels**: At 25 bps total (high scenario), the strategy turns negative. This suggests the strategy's edge is thin relative to realistic trading costs.

3. **Market impact is devastating**: Even a small market impact coefficient (eta=0.001) makes the strategy deeply unprofitable (Sharpe -1.36). This is because the strategy trades continuously every day with varying allocation sizes — the square-root impact on high-turnover positions compounds rapidly.

4. **Break-even cost analysis**: The strategy breaks even at approximately 20 bps total cost (interpolating between base and high scenarios). Below 8 bps total, the strategy maintains a Sharpe above 0.5.

5. **Turnover is very high**: 6,192 trades over 6,286 days means the strategy trades on 98.5% of all days. The continuous nature of the feedback control law means positions change almost every day, generating significant costs.

6. **Post-2021 degradation persists**: Windows 8-9 remain unprofitable regardless of cost scenario, confirming the EWA/EWC cointegration weakening is a structural issue, not a cost issue.

## Implications for Future Phases

- **Phase 6 (hyperparameter optimization)**: Reducing control gain k or increasing the kappa threshold could lower turnover and improve net performance. A turnover penalty in the objective function may be warranted.
- **Phase 7 (multiple pairs)**: Some pairs may have stronger mean-reversion, yielding better gross alpha to absorb costs.
- Cost sensitivity analysis should be a standard part of all future evaluations.
