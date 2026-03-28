"""
Data loader for pairs trading framework.
Fetches market data from ARF Data API and computes log-price spreads.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
from io import StringIO

API_BASE = "https://ai.1s.xyz/api/data/ohlcv"


class DataLoader:
    """Loads pair data from ARF Data API and computes spreads."""

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_ticker(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data for a single ticker from ARF Data API, with local caching."""
        import requests

        cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file, parse_dates=["timestamp"], index_col="timestamp")
            return df

        # Calculate period from date range
        from datetime import datetime
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days
        years = max(1, days // 365)
        period = f"{years}y"

        url = f"{API_BASE}?ticker={ticker}&interval=1d&period={period}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        df = pd.read_csv(StringIO(resp.text), parse_dates=["timestamp"], index_col="timestamp")
        # Filter to requested date range
        df = df.loc[start_date:end_date]
        # Cache locally
        df.to_csv(cache_file)
        return df

    def download_pair_data(
        self, ticker1: str, ticker2: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Download adjusted close data for a pair of tickers.

        Args:
            ticker1: First ticker symbol (e.g. 'EWA')
            ticker2: Second ticker symbol (e.g. 'EWC')
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'

        Returns:
            DataFrame with columns [ticker1, ticker2] containing close prices,
            indexed by timestamp.
        """
        df1 = self._fetch_ticker(ticker1, start_date, end_date)
        df2 = self._fetch_ticker(ticker2, start_date, end_date)

        pair = pd.DataFrame({
            ticker1: df1["close"],
            ticker2: df2["close"],
        })
        # Align on common dates and drop any NaN rows
        pair = pair.dropna()
        return pair

    def calculate_spread(
        self, pair_data: pd.DataFrame, ticker1: str, ticker2: str
    ) -> pd.DataFrame:
        """
        Calculate the log-price spread using linear regression hedge ratio.

        spread = log(price1) - beta * log(price2)

        Args:
            pair_data: DataFrame with columns [ticker1, ticker2] of close prices
            ticker1: Column name for first asset
            ticker2: Column name for second asset

        Returns:
            DataFrame with columns: ticker1, ticker2, log_price1, log_price2,
            beta, spread
        """
        log_p1 = np.log(pair_data[ticker1])
        log_p2 = np.log(pair_data[ticker2])

        # Estimate hedge ratio via OLS: log(P1) = alpha + beta * log(P2)
        reg = LinearRegression()
        reg.fit(log_p2.values.reshape(-1, 1), log_p1.values)
        beta = float(reg.coef_[0])

        spread = log_p1 - beta * log_p2

        result = pair_data.copy()
        result["log_price1"] = log_p1
        result["log_price2"] = log_p2
        result["beta"] = beta
        result["spread"] = spread
        return result

    def calculate_rolling_spread(
        self, pair_data: pd.DataFrame, ticker1: str, ticker2: str, window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate the log-price spread using a rolling-window hedge ratio.

        At each time step t, beta is estimated from the preceding `window`
        observations, eliminating look-ahead bias from the static method.

        Args:
            pair_data: DataFrame with columns [ticker1, ticker2] of close prices
            ticker1: Column name for first asset
            ticker2: Column name for second asset
            window: Lookback window for hedge ratio estimation (default: 252)

        Returns:
            DataFrame starting from index `window` with columns:
            ticker1, ticker2, log_price1, log_price2, beta, spread
        """
        log_p1 = np.log(pair_data[ticker1])
        log_p2 = np.log(pair_data[ticker2])

        n = len(pair_data)
        betas = np.full(n, np.nan)
        spreads = np.full(n, np.nan)

        reg = LinearRegression()
        for i in range(window, n):
            x = log_p2.values[i - window : i].reshape(-1, 1)
            y = log_p1.values[i - window : i]
            reg.fit(x, y)
            b = float(reg.coef_[0])
            betas[i] = b
            spreads[i] = log_p1.values[i] - b * log_p2.values[i]

        result = pair_data.copy()
        result["log_price1"] = log_p1
        result["log_price2"] = log_p2
        result["beta"] = betas
        result["spread"] = spreads

        # Drop rows where rolling estimation hasn't started yet
        result = result.iloc[window:].copy()
        return result
