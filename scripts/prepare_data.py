"""
Prepare EWA/EWC spread data for pairs trading analysis.
Fetches data from ARF Data API, computes spread, saves to parquet.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import DataLoader


def main():
    ticker1 = "EWA"
    ticker2 = "EWC"
    start_date = "2000-01-01"
    end_date = "2026-03-28"

    loader = DataLoader(cache_dir="data/raw")

    print(f"Downloading {ticker1}/{ticker2} data from {start_date} to {end_date}...")
    pair_data = loader.download_pair_data(ticker1, ticker2, start_date, end_date)
    print(f"  Retrieved {len(pair_data)} trading days")
    print(f"  Date range: {pair_data.index[0]} to {pair_data.index[-1]}")

    print("Computing log-price spread with OLS hedge ratio...")
    spread_data = loader.calculate_spread(pair_data, ticker1, ticker2)
    beta = spread_data["beta"].iloc[0]
    print(f"  Hedge ratio (beta): {beta:.4f}")
    print(f"  Spread mean: {spread_data['spread'].mean():.4f}")
    print(f"  Spread std:  {spread_data['spread'].std():.4f}")

    # Verify no NaN
    nan_count = spread_data.isna().sum().sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values found in spread data!")
        spread_data = spread_data.dropna()
        print(f"  After dropna: {len(spread_data)} rows")

    assert not spread_data.isna().any().any(), "Spread data contains NaN values"

    # Save to parquet
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker1}_{ticker2}_spread.parquet"
    spread_data.to_parquet(output_path)
    print(f"Saved spread data to {output_path}")
    print(f"  Shape: {spread_data.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
