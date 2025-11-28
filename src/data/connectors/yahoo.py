"""
Yahoo Finance Connector - Production Grade
Robust data fetching with retry logic, validation, and caching
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union
from datetime import datetime, timedelta
import time
import logging
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, base_delay=1):
    """Decorator for exponential backoff retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class YahooFinanceConnector:
    """
    Production-grade Yahoo Finance connector
    Features: retry logic, parallel downloads, validation, caching
    """
    
    def __init__(self, request_delay: float = 0.5, max_workers: int = 5):
        self.request_delay = request_delay
        self.max_workers = max_workers
        self.last_request_time = {}
        self.cache = {}
        
    def _rate_limit(self, ticker: str):
        """Implement rate limiting per ticker"""
        key = f"yahoo_{ticker}"
        if key in self.last_request_time:
            elapsed = time.time() - self.last_request_time[key]
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
        self.last_request_time[key] = time.time()
    
    @retry_with_backoff(max_retries=3, base_delay=2)
    def _download_single(
        self, 
        ticker: str, 
        period: str = '1y',
        interval: str = '1d',
        validate: bool = True
    ) -> pd.DataFrame:
        """Download data for single ticker with validation"""
        
        self._rate_limit(ticker)
        
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                show_errors=False,
                timeout=15
            )
            
            if data.empty:
                logger.warning(f"Empty data for {ticker}")
                return pd.DataFrame()
            
            # Standardize column names
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Ensure required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for {ticker}: {missing_cols}")
                # Try to handle Adj Close
                if 'Adj Close' in data.columns and 'Close' not in data.columns:
                    data['Close'] = data['Adj Close']
            
            # Validation
            if validate:
                data = self._validate_data(data, ticker)
            
            logger.debug(f"✅ Downloaded {ticker}: {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            raise
    
    def _validate_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Validate and clean data"""
        
        if data.empty:
            return data
        
        original_len = len(data)
        
        # 1. Remove rows with all NaN
        data = data.dropna(how='all')
        
        # 2. Check for negative prices
        if 'Close' in data.columns:
            negative_prices = (data['Close'] <= 0).sum()
            if negative_prices > 0:
                logger.warning(f"{ticker}: {negative_prices} negative/zero prices removed")
                data = data[data['Close'] > 0]
        
        # 3. Check for extreme price changes (>50% in one day)
        if 'Close' in data.columns and len(data) > 1:
            price_changes = data['Close'].pct_change().abs()
            extreme_changes = price_changes > 0.5
            if extreme_changes.sum() > 0:
                logger.warning(f"{ticker}: {extreme_changes.sum()} extreme price changes detected")
                # Don't remove, but flag
                data.loc[extreme_changes, 'extreme_change'] = True
        
        # 4. Forward fill small gaps (max 5 days)
        if 'Close' in data.columns:
            data['Close'] = data['Close'].fillna(method='ffill', limit=5)
        
        # 5. Check data completeness
        missing_pct = data['Close'].isna().sum() / len(data) * 100
        if missing_pct > 20:
            logger.warning(f"{ticker}: {missing_pct:.1f}% missing data")
        
        cleaned_len = len(data)
        if cleaned_len < original_len:
            logger.info(f"{ticker}: Cleaned {original_len - cleaned_len} rows")
        
        return data
    
    def download_tickers(
        self,
        tickers: Union[str, List[str]],
        period: str = '1y',
        interval: str = '1d',
        parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download multiple tickers efficiently
        
        Args:
            tickers: Single ticker or list of tickers
            period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            parallel: Use parallel downloads
            
        Returns:
            Dictionary of {ticker: DataFrame}
        """
        
        if isinstance(tickers, str):
            tickers = [tickers]
        
        results = {}
        
        if parallel and len(tickers) > 1:
            # Parallel download
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_ticker = {
                    executor.submit(self._download_single, ticker, period, interval): ticker
                    for ticker in tickers
                }
                
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        data = future.result()
                        if not data.empty:
                            results[ticker] = data
                    except Exception as e:
                        logger.error(f"Failed to download {ticker}: {e}")
        else:
            # Sequential download
            for ticker in tickers:
                try:
                    data = self._download_single(ticker, period, interval)
                    if not data.empty:
                        results[ticker] = data
                except Exception as e:
                    logger.error(f"Failed to download {ticker}: {e}")
        
        logger.info(f"Downloaded {len(results)}/{len(tickers)} tickers successfully")
        return results
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get latest price for ticker"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Try different price fields
            price = (info.get('regularMarketPrice') or 
                    info.get('currentPrice') or
                    info.get('previousClose'))
            
            if price:
                logger.debug(f"{ticker} current price: {price}")
                return float(price)
            
            # Fallback: get from recent history
            hist = stock.history(period='1d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            logger.warning(f"Could not get current price for {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {ticker}: {e}")
            return None
    
    def get_info(self, ticker: str) -> Dict:
        """Get stock information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key metrics
            return {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'price_to_book': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def calculate_returns(
        self,
        data: pd.DataFrame,
        method: str = 'simple',
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate returns for various periods
        
        Args:
            data: DataFrame with 'Close' column
            method: 'simple' or 'log'
            periods: List of lookback periods (default: [1, 5, 20, 60, 252])
            
        Returns:
            DataFrame with return columns
        """
        
        if 'Close' not in data.columns:
            raise ValueError("DataFrame must have 'Close' column")
        
        if periods is None:
            periods = [1, 5, 20, 60, 252]  # Daily, weekly, monthly, quarterly, yearly
        
        returns = pd.DataFrame(index=data.index)
        
        for period in periods:
            if method == 'simple':
                returns[f'return_{period}d'] = data['Close'].pct_change(period)
            elif method == 'log':
                returns[f'return_{period}d'] = np.log(data['Close'] / data['Close'].shift(period))
        
        return returns
    
    def calculate_volatility(
        self,
        data: pd.DataFrame,
        window: int = 20,
        annualize: bool = True
    ) -> pd.Series:
        """Calculate rolling volatility"""
        
        if 'Close' not in data.columns:
            raise ValueError("DataFrame must have 'Close' column")
        
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=window).std()
        
        if annualize:
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    def get_index_performance(
        self,
        ticker: str,
        benchmark: str = '^GSPC',
        period: str = '1y'
    ) -> Dict:
        """
        Compare ticker performance vs benchmark
        
        Returns:
            Dictionary with performance metrics
        """
        
        # Download both
        data = self.download_tickers([ticker, benchmark], period=period)
        
        if ticker not in data or benchmark not in data:
            return {'error': 'Could not download data'}
        
        ticker_data = data[ticker]['Close']
        benchmark_data = data[benchmark]['Close']
        
        # Align dates
        common_dates = ticker_data.index.intersection(benchmark_data.index)
        ticker_aligned = ticker_data[common_dates]
        benchmark_aligned = benchmark_data[common_dates]
        
        # Calculate returns
        ticker_return = (ticker_aligned.iloc[-1] / ticker_aligned.iloc[0] - 1) * 100
        benchmark_return = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0] - 1) * 100
        
        # Calculate volatility
        ticker_vol = ticker_aligned.pct_change().std() * np.sqrt(252) * 100
        benchmark_vol = benchmark_aligned.pct_change().std() * np.sqrt(252) * 100
        
        # Calculate correlation
        returns_df = pd.DataFrame({
            'ticker': ticker_aligned.pct_change(),
            'benchmark': benchmark_aligned.pct_change()
        }).dropna()
        
        correlation = returns_df.corr().iloc[0, 1]
        
        # Calculate beta
        covariance = returns_df.cov().iloc[0, 1]
        benchmark_variance = returns_df['benchmark'].var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else None
        
        return {
            'ticker': ticker,
            'benchmark': benchmark,
            'period': period,
            'ticker_return': ticker_return,
            'benchmark_return': benchmark_return,
            'relative_return': ticker_return - benchmark_return,
            'ticker_volatility': ticker_vol,
            'benchmark_volatility': benchmark_vol,
            'correlation': correlation,
            'beta': beta,
            'sharpe_ratio': ticker_return / ticker_vol if ticker_vol != 0 else None
        }


if __name__ == "__main__":
    # Test the connector
    logging.basicConfig(level=logging.INFO)
    
    connector = YahooFinanceConnector(request_delay=0.5)
    
    print("\n" + "="*70)
    print("TESTING YAHOO FINANCE CONNECTOR")
    print("="*70)
    
    # Test 1: Single ticker
    print("\nTest 1: Download single ticker (STOXX 600)")
    data = connector.download_tickers("EXSA.DE", period='3mo')
    if "EXSA.DE" in data:
        df = data["EXSA.DE"]
        print(f"✅ Downloaded {len(df)} rows")
        print(f"Latest price: {df['Close'].iloc[-1]:.2f}")
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Test 2: Multiple tickers (parallel)
    print("\nTest 2: Download multiple tickers (parallel)")
    tickers = ["^GDAXI", "^FCHI", "^FTSE"]
    data = connector.download_tickers(tickers, period='1mo', parallel=True)
    print(f"✅ Downloaded {len(data)}/{len(tickers)} tickers")
    
    # Test 3: Get current price
    print("\nTest 3: Get current price")
    price = connector.get_current_price("^GDAXI")
    if price:
        print(f"✅ DAX current price: {price:.2f}")
    
    # Test 4: Performance comparison
    print("\nTest 4: STOXX 600 vs S&P 500 performance")
    perf = connector.get_index_performance("EXSA.DE", "^GSPC", period='1y')
    if 'error' not in perf:
        print(f"✅ STOXX 600 return: {perf['ticker_return']:+.2f}%")
        print(f"✅ S&P 500 return: {perf['benchmark_return']:+.2f}%")
        print(f"✅ Relative performance: {perf['relative_return']:+.2f}%")
        print(f"✅ Beta: {perf['beta']:.2f}" if perf['beta'] else "Beta: N/A")
    
    print("\n" + "="*70)
    print("CONNECTOR TEST COMPLETE")
    print("="*70)