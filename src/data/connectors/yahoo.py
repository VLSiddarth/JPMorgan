"""
Yahoo Finance Market Data Connector

Production-ready wrapper around yfinance for:
- Historical prices (single & bulk)
- Latest prices
- Basic info

Designed to be used across:
- Streamlit app (app.py)
- FastAPI API (api.py)
- Notebooks (data exploration, backtest, factors)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterable, Union

import logging
import time
from datetime import datetime
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class YahooMarketDataConfig:
    """Configuration for YahooMarketData connector."""
    max_retries: int = 3
    retry_backoff: float = 1.0  # seconds
    timeout: int = 10           # not directly used by yfinance, but kept for future HTTP session support


class YahooMarketData:
    """
    Production-grade wrapper around yfinance.

    Exposed methods (used by your project):
    - get_history(...)
    - get_history_bulk(...)
    - get_last_price(...)
    - get_multiple_last_prices(...)
    """

    def __init__(self, config: Optional[YahooMarketDataConfig] = None) -> None:
        self.config = config or YahooMarketDataConfig()

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------
    def _download_with_retry(
        self,
        tickers: Union[str, List[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """
        Internal helper with retry logic around yfinance.download.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.info(
                    "YahooMarketData: downloading tickers=%s, start=%s, end=%s, interval=%s (attempt %d)",
                    tickers, start, end, interval, attempt
                )

                data = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    progress=False,
                    threads=True,
                )

                # When multiple tickers: yfinance returns multi-index columns (e.g. ('Adj Close', 'AAPL'))
                if isinstance(tickers, (list, tuple, set)):
                    if isinstance(data.columns, pd.MultiIndex):
                        data = data["Adj Close"].copy()
                    else:
                        # fallback if not multiindex
                        pass
                else:
                    # Single ticker: use Adj Close if available
                    if isinstance(data.columns, pd.MultiIndex):
                        data = data["Adj Close"].copy()
                    elif "Adj Close" in data.columns:
                        data = data["Adj Close"].copy()

                # Ensure we have DataFrame
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=tickers if isinstance(tickers, str) else "price")

                # Drop all-NaN rows
                data = data.dropna(how="all")

                if data.empty:
                    logger.warning("YahooMarketData: empty data returned for %s", tickers)

                return data

            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                logger.warning(
                    "YahooMarketData: error downloading %s on attempt %d/%d: %s",
                    tickers, attempt, self.config.max_retries, exc
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff)

        logger.error("YahooMarketData: failed to download data for %s after retries", tickers)
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Failed to download data for {tickers!r}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def get_history(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.Series:
        """
        Get historical adjusted close price for a single ticker.

        Parameters
        ----------
        ticker : str
            Symbol (e.g. '^STOXX50E', '^GSPC', 'EXSA.DE').
        start : str, optional
            Start date 'YYYY-MM-DD'. Default: 2020-01-01 if not provided.
        end : str, optional
            End date 'YYYY-MM-DD'. Default: today if not provided.
        interval : str
            '1d', '1wk', '1mo', etc.

        Returns
        -------
        pandas.Series
            Adjusted close prices indexed by date.
        """
        if start is None:
            start = "2020-01-01"
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        df = self._download_with_retry(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
        )

        # For single ticker, yfinance often returns a DataFrame with a single column
        if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
            series = df.iloc[:, 0]
            series.name = ticker
            return series

        # Fallback
        return df.squeeze()

    def get_history_bulk(
        self,
        tickers: Iterable[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Get historical adjusted close prices for multiple tickers.

        Parameters
        ----------
        tickers : iterable of str
            List of tickers.
        start, end, interval : see get_history

        Returns
        -------
        pandas.DataFrame
            Columns = tickers, index = date.
        """
        tickers_list = list(tickers)
        if not tickers_list:
            raise ValueError("tickers must be a non-empty list")

        if start is None:
            start = "2020-01-01"
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        df = self._download_with_retry(
            tickers=tickers_list,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
        )

        # Ensure columns are plain ticker symbols
        if isinstance(df.columns, pd.MultiIndex):
            # If multiindex, we already selected 'Adj Close' in _download_with_retry
            # so columns should be second level = tickers
            df.columns = [str(col) for col in df.columns]

        return df

    def get_last_price(self, ticker: str) -> Optional[float]:
        """
        Get the last available close/price for a single ticker.

        Returns
        -------
        float or None
        """
        try:
            data = self.get_history(ticker=ticker, start=None, end=None)
            if isinstance(data, pd.Series) and not data.empty:
                return float(data.iloc[-1])
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("YahooMarketData: failed to get last price for %s: %s", ticker, exc)
            return None

    def get_multiple_last_prices(self, tickers: Iterable[str]) -> Dict[str, Optional[float]]:
        """
        Get last prices for multiple tickers.

        Returns
        -------
        dict
            {ticker: price or None}
        """
        prices: Dict[str, Optional[float]] = {}
        tickers_list = list(tickers)
        if not tickers_list:
            return prices

        try:
            df = self.get_history_bulk(tickers_list, start=None, end=None)
            for t in tickers_list:
                if t in df.columns and not df[t].dropna().empty:
                    prices[t] = float(df[t].dropna().iloc[-1])
                else:
                    prices[t] = None
        except Exception as exc:  # noqa: BLE001
            logger.error("YahooMarketData: failed to get multiple last prices: %s", exc)
            # best-effort: attempt individually
            for t in tickers_list:
                prices[t] = self.get_last_price(t)

        return prices

    def get_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic info for a ticker (name, sector, etc.).

        Returns
        -------
        dict
        """
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            return dict(info)
        except Exception as exc:  # noqa: BLE001
            logger.error("YahooMarketData: failed to fetch info for %s: %s", ticker, exc)
            return {}


if __name__ == "__main__":
    # Simple manual test
    logging.basicConfig(level=logging.INFO)
    client = YahooMarketData()

    print("Testing single ticker history...")
    sxxp = client.get_history("^STOXX50E", start="2023-01-01")
    print(sxxp.tail())

    print("\nTesting bulk history...")
    bulk = client.get_history_bulk(["^STOXX50E", "^GSPC"], start="2023-01-01")
    print(bulk.tail())

    print("\nTesting last price...")
    print("Last price ^STOXX50E:", client.get_last_price("^STOXX50E"))
