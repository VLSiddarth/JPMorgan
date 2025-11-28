"""
===============================================================================
utils/logger.py - Custom Logging Configuration
===============================================================================
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: blue + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def setup_logging(log_dir: str = "logs", 
                 level: int = logging.INFO,
                 log_to_file: bool = True) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        log_to_file: Whether to log to file
        
    Returns:
        Configured logger
    """
    # Create logs directory
    if log_to_file:
        Path(log_dir).mkdir(exist_ok=True)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime('%Y%m%d')
        file_handler = logging.FileHandler(
            f"{log_dir}/app_{timestamp}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


"""
===============================================================================
utils/math_utils.py - Mathematical Utilities
===============================================================================
"""

import numpy as np
import pandas as pd
from typing import Union, List
from scipy import stats


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series
    
    Args:
        prices: Price series
        method: 'simple' or 'log'
        
    Returns:
        Returns series
    """
    if method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


def annualize_return(returns: Union[float, pd.Series], 
                     periods: int = 252) -> Union[float, pd.Series]:
    """
    Annualize returns
    
    Args:
        returns: Returns (decimal)
        periods: Periods per year (252 for daily, 12 for monthly)
        
    Returns:
        Annualized return
    """
    if isinstance(returns, pd.Series):
        return (1 + returns).prod() ** (periods / len(returns)) - 1
    else:
        return (1 + returns) ** periods - 1


def annualize_volatility(returns: pd.Series, periods: int = 252) -> float:
    """
    Annualize volatility
    
    Args:
        returns: Returns series
        periods: Periods per year
        
    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods)


def sharpe_ratio(returns: pd.Series, 
                risk_free_rate: float = 0.02,
                periods: int = 252) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods: Periods per year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods
    return np.sqrt(periods) * excess_returns.mean() / returns.std()


def sortino_ratio(returns: pd.Series,
                 risk_free_rate: float = 0.02,
                 periods: int = 252) -> float:
    """
    Calculate Sortino ratio (downside deviation)
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods: Periods per year
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods)
    
    if downside_std == 0:
        return np.inf
    
    return annualize_return(excess_returns, periods) / downside_std


def max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        returns: Returns series
        
    Returns:
        Maximum drawdown (negative)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    """
    Calculate Calmar ratio (return / max drawdown)
    
    Args:
        returns: Returns series
        periods: Periods per year
        
    Returns:
        Calmar ratio
    """
    ann_return = annualize_return(returns, periods)
    max_dd = abs(max_drawdown(returns))
    
    if max_dd == 0:
        return np.inf
    
    return ann_return / max_dd


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: Returns series
        confidence: Confidence level
        
    Returns:
        VaR (negative value)
    """
    return np.percentile(returns, (1 - confidence) * 100)


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Conditional VaR (CVaR/Expected Shortfall)
    
    Args:
        returns: Returns series
        confidence: Confidence level
        
    Returns:
        CVaR (negative value)
    """
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()


def rolling_sharpe(returns: pd.Series, 
                  window: int = 252,
                  risk_free_rate: float = 0.02) -> pd.Series:
    """
    Calculate rolling Sharpe ratio
    
    Args:
        returns: Returns series
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Rolling Sharpe ratio series
    """
    excess = returns - risk_free_rate / 252
    return (excess.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)


def beta(asset_returns: pd.Series, 
        market_returns: pd.Series) -> float:
    """
    Calculate beta
    
    Args:
        asset_returns: Asset returns
        market_returns: Market returns
        
    Returns:
        Beta coefficient
    """
    covariance = np.cov(asset_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance if market_variance != 0 else 0


def information_ratio(returns: pd.Series, 
                     benchmark_returns: pd.Series) -> float:
    """
    Calculate Information Ratio
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Information ratio
    """
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std()
    
    if tracking_error == 0:
        return 0
    
    return active_returns.mean() / tracking_error * np.sqrt(252)


"""
===============================================================================
utils/date_utils.py - Date Utilities
===============================================================================
"""

from datetime import datetime, timedelta
import pandas as pd
from typing import List, Tuple


def get_trading_days(start_date: datetime, 
                    end_date: datetime,
                    exclude_weekends: bool = True) -> List[datetime]:
    """
    Get list of trading days between dates
    
    Args:
        start_date: Start date
        end_date: End date
        exclude_weekends: Whether to exclude weekends
        
    Returns:
        List of trading days
    """
    dates = pd.date_range(start_date, end_date, freq='D')
    
    if exclude_weekends:
        dates = dates[dates.dayofweek < 5]  # Monday=0, Friday=4
    
    return dates.tolist()


def get_quarter_dates(year: int, quarter: int) -> Tuple[datetime, datetime]:
    """
    Get start and end dates for a quarter
    
    Args:
        year: Year
        quarter: Quarter (1-4)
        
    Returns:
        (start_date, end_date) tuple
    """
    quarter_starts = {
        1: (1, 1),
        2: (4, 1),
        3: (7, 1),
        4: (10, 1)
    }
    
    quarter_ends = {
        1: (3, 31),
        2: (6, 30),
        3: (9, 30),
        4: (12, 31)
    }
    
    start_month, start_day = quarter_starts[quarter]
    end_month, end_day = quarter_ends[quarter]
    
    start_date = datetime(year, start_month, start_day)
    end_date = datetime(year, end_month, end_day)
    
    return start_date, end_date


def get_month_end_dates(start_date: datetime, 
                       end_date: datetime) -> List[datetime]:
    """
    Get month-end dates between two dates
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of month-end dates
    """
    dates = pd.date_range(start_date, end_date, freq='M')
    return dates.tolist()


def get_year_to_date_range() -> Tuple[datetime, datetime]:
    """
    Get year-to-date date range
    
    Returns:
        (start_date, end_date) for current year
    """
    today = datetime.now()
    start_date = datetime(today.year, 1, 1)
    return start_date, today


def add_business_days(start_date: datetime, days: int) -> datetime:
    """
    Add business days to a date
    
    Args:
        start_date: Starting date
        days: Number of business days to add
        
    Returns:
        Resulting date
    """
    current = start_date
    added = 0
    
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday-Friday
            added += 1
    
    return current


def format_date_for_api(date: datetime, format_type: str = 'iso') -> str:
    """
    Format date for API requests
    
    Args:
        date: Date to format
        format_type: 'iso', 'us', 'eu'
        
    Returns:
        Formatted date string
    """
    formats = {
        'iso': '%Y-%m-%d',
        'us': '%m/%d/%Y',
        'eu': '%d/%m/%Y'
    }
    
    return date.strftime(formats.get(format_type, formats['iso']))


"""
===============================================================================
utils/decorators.py - Utility Decorators
===============================================================================
"""

import time
import functools
from typing import Callable, Any


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution
    
    Usage:
        @timer
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        
    Usage:
        @retry(max_attempts=3, delay=2.0)
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        logging.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    logging.warning(f"{func.__name__} attempt {attempt} failed: {e}. Retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def cache_result(ttl: int = 3600):
    """
    Decorator to cache function results
    
    Args:
        ttl: Time to live in seconds
        
    Usage:
        @cache_result(ttl=1800)
        def expensive_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = time.time()
            
            if key in cache and (current_time - cache_times[key]) < ttl:
                logging.debug(f"Returning cached result for {func.__name__}")
                return cache[key]
            
            result = func(*args, **kwargs)
            cache[key] = result
            cache_times[key] = current_time
            return result
        
        return wrapper
    return decorator


def validate_types(**type_checks):
    """
    Decorator to validate function argument types
    
    Usage:
        @validate_types(x=int, y=float)
        def add(x, y):
            return x + y
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate types
            for param_name, expected_type in type_checks.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"{func.__name__}: {param_name} must be {expected_type}, "
                            f"got {type(value)}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_calls(func: Callable) -> Callable:
    """
    Decorator to log function calls
    
    Usage:
        @log_calls
        def my_function(x, y):
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} returned {result}")
        return result
    return wrapper


def deprecated(reason: str = ""):
    """
    Decorator to mark functions as deprecated
    
    Args:
        reason: Reason for deprecation
        
    Usage:
        @deprecated("Use new_function instead")
        def old_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warning_msg = f"{func.__name__} is deprecated"
            if reason:
                warning_msg += f": {reason}"
            logging.warning(warning_msg)
            return func(*args, **kwargs)
        return wrapper
    return decorator