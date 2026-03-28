# Cycle 2 Technical Findings: Real Data Pipeline

## Summary

Implemented the real data pipeline for fetching EWA/EWC pair data from the ARF Data API and computing the log-price spread with an OLS-estimated hedge ratio.

## Components Implemented

1. **DataLoader** (`src/data_loader.py`): Class for fetching and processing pair data.
   - `download_pair_data()`: Fetches daily OHLCV from ARF Data API with local CSV caching.
   - `calculate_spread()`: Computes log-price spread using OLS hedge ratio: `spread = log(P1) - beta * log(P2)`.

2. **Data Preparation Script** (`scripts/prepare_data.py`): End-to-end pipeline that fetches EWA/EWC data, computes spread, validates for NaN, and saves to parquet.

3. **Tests** (`tests/test_data_loader.py`): 6 tests covering spread computation correctness, NaN absence, column structure, and cointegration properties.

## EWA/EWC Spread Statistics

| Metric | Value |
|--------|-------|
| Data points | 6,539 trading days |
| Date range | 2000-03-28 to 2026-03-27 |
| Hedge ratio (beta) | 1.1570 |
| Spread mean | -0.9936 |
| Spread std | 0.1198 |
| NaN count | 0 |

## Key Observations

1. **Data availability**: The ARF Data API provides ~26 years of daily data for EWA/EWC when requesting maximum period. The API returns data from 2000-03-28 onward.

2. **Hedge ratio**: The OLS hedge ratio of 1.157 indicates that for every unit of EWA, approximately 1.157 units of EWC are needed to form a hedged position. This is consistent with the close economic relationship between Australian and Canadian equity markets.

3. **Spread properties**: The spread has a standard deviation of ~0.12 around a mean of -0.99. The relatively tight distribution suggests potential mean-reverting behavior suitable for the OU model in Phase 3.

4. **Static vs. rolling beta**: The current implementation uses a single hedge ratio estimated over the entire sample. Phase 3 will implement rolling estimation of both the hedge ratio and OU parameters, which is more realistic for production use.

## Limitations

- Hedge ratio is estimated over the full sample (look-ahead bias for backtesting purposes). This will be addressed with rolling estimation in Phase 3.
- The ARF Data API returns "close" prices rather than "adjusted close". For ETFs like EWA/EWC, this may not account for dividend adjustments.
- No walk-forward metrics computed yet (Phase 4).
- No transaction cost analysis on real data yet (Phase 5).
