"""
Mathematical & Quant Utilities

Core quantitative helpers used across the JPMorgan European Equity Dashboard:

- Return calculations (simple/log)
- Annualized returns & volatility
- Sharpe / Sortino ratios
- Max drawdown & drawdown series
- CAGR
- Beta / alpha vs benchmark
- Correlation / covariance helpers
- Rolling statistics
- Basic VaR / Expected Shortfall (CVaR)

These are low-level building blocks; higher-level logic lives in
src/analytics/risk and src/analytics/backtest.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import]

logger = logging.getLogger(__name__)

Number = float | int | np.number


# -----------------------------------------------------------------------------
# Basic return & annualization helpers
# -----------------------------------------------------------------------------

def compute_returns_from_prices(
    prices: pd.Series | pd.DataFrame,
    method: str = "simple",
) -> pd.Series | pd.DataFrame:
    """
    Compute returns from a price series or DataFrame of prices.

    Args:
        prices: Price series (index = dates) or DataFrame (columns = assets).
        method: 'simple' for pct_change, 'log' for log returns.

    Returns:
        Series/DataFrame of returns with same shape (first value NaN).
    """
    if prices is None or len(prices) == 0:
        return prices

    if method not in {"simple", "log"}:
        raise ValueError("method must be 'simple' or 'log'")

    if method == "simple":
        return prices.pct_change()
    else:
        return np.log(prices / prices.shift(1))


def annualize_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Annualize a series of periodic returns.

    Uses geometric compounding.

    Args:
        returns: Periodic returns (daily, weekly, etc.).
        periods_per_year: Number of periods per year (default 252 trading days).

    Returns:
        Annualized return as decimal (0.10 = 10%).
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")

    cumulative = float((1.0 + r).prod())
    n_periods = len(r)
    if n_periods == 0 or cumulative <= 0:
        return float("nan")

    years = n_periods / periods_per_year
    if years <= 0:
        return float("nan")

    return cumulative ** (1.0 / years) - 1.0


def annualize_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Annualize volatility from periodic returns.

    Args:
        returns: Periodic returns.
        periods_per_year: Number of periods per year.

    Returns:
        Annualized volatility.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")

    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def cagr(
    prices: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR) from price series.

    Args:
        prices: Price series.
        periods_per_year: Number of observations per year.

    Returns:
        CAGR as decimal.
    """
    p = prices.dropna()
    if p.empty:
        return float("nan")

    start = float(p.iloc[0])
    end = float(p.iloc[-1])
    if start <= 0:
        return float("nan")

    n_periods = len(p) - 1
    if n_periods <= 0:
        return 0.0

    years = n_periods / periods_per_year
    if years <= 0:
        return float("nan")

    return (end / start) ** (1.0 / years) - 1.0


# -----------------------------------------------------------------------------
# Risk-adjusted metrics
# -----------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Periodic returns.
        risk_free_rate: Annual risk-free rate (decimal).
        periods_per_year: Number of periods per year.

    Returns:
        Sharpe ratio.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")

    # Convert annual RF to per-period
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_per_period

    if excess.std(ddof=1) == 0:
        return float("nan")

    mean_excess = float(excess.mean())
    vol = float(excess.std(ddof=1))

    return (mean_excess * periods_per_year) / (vol * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sortino ratio (only downside volatility).

    Args:
        returns: Periodic returns.
        target_return: Target per-period return (default 0).
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of periods per year.

    Returns:
        Sortino ratio.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")

    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_per_period

    downside = excess[excess < target_return]
    if downside.empty:
        return float("inf")

    downside_std = float(downside.std(ddof=1))
    if downside_std == 0.0:
        return float("inf")

    mean_excess = float(excess.mean())
    annual_excess = mean_excess * periods_per_year
    annual_downside = downside_std * np.sqrt(periods_per_year)

    return annual_excess / annual_downside


# -----------------------------------------------------------------------------
# Drawdown metrics
# -----------------------------------------------------------------------------

def drawdown_series(prices_or_equity: pd.Series) -> pd.Series:
    """
    Compute drawdown series from price or equity curve.

    Args:
        prices_or_equity: Series of prices or equity values.

    Returns:
        Series of drawdown (0 to negative values).
    """
    s = prices_or_equity.dropna()
    if s.empty:
        return s

    running_max = s.cummax()
    dd = (s - running_max) / running_max
    return dd


def max_drawdown(prices_or_equity: pd.Series) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Compute maximum drawdown and its start/end dates.

    Args:
        prices_or_equity: Series of prices or equity values.

    Returns:
        (max_drawdown, peak_date, trough_date)
        max_drawdown is negative (e.g. -0.25 for -25%).
    """
    s = prices_or_equity.dropna()
    if s.empty:
        return float("nan"), None, None

    running_max = s.cummax()
    dd = (s - running_max) / running_max

    trough_idx = dd.idxmin()
    max_dd = float(dd.loc[trough_idx])

    try:
        peak_idx = s.loc[:trough_idx].idxmax()
    except Exception:
        peak_idx = None

    return max_dd, peak_idx, trough_idx


# -----------------------------------------------------------------------------
# Beta / alpha & correlation helpers
# -----------------------------------------------------------------------------

def beta_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute CAPM beta and alpha (annualized) versus a benchmark.

    Args:
        portfolio_returns: Portfolio periodic returns.
        benchmark_returns: Benchmark periodic returns.
        periods_per_year: Number of periods per year.

    Returns:
        dict with 'beta' and 'alpha_annual'.
    """
    df = pd.DataFrame(
        {
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }
    ).dropna()

    if len(df) < 10:
        return {"beta": float("nan"), "alpha_annual": float("nan")}

    cov = float(df["portfolio"].cov(df["benchmark"]))
    var_b = float(df["benchmark"].var(ddof=1))
    if var_b == 0:
        beta = float("nan")
    else:
        beta = cov / var_b

    # Regression alpha (per period)
    slope, intercept, _, _, _ = stats.linregress(df["benchmark"], df["portfolio"])
    # slope ~ beta, intercept ~ alpha per period
    alpha_period = float(intercept)
    alpha_annual = (1.0 + alpha_period) ** periods_per_year - 1.0

    return {"beta": beta, "alpha_annual": alpha_annual}


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix of returns.

    Args:
        returns: DataFrame of aligned returns (columns = assets).

    Returns:
        Correlation matrix DataFrame.
    """
    r = returns.dropna(how="all")
    if r.empty:
        return pd.DataFrame()

    return r.corr()


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute covariance matrix of returns.

    Args:
        returns: DataFrame of returns.

    Returns:
        Covariance matrix.
    """
    r = returns.dropna(how="all")
    if r.empty:
        return pd.DataFrame()

    return r.cov()


# -----------------------------------------------------------------------------
# Rolling statistics
# -----------------------------------------------------------------------------

def rolling_volatility(
    returns: pd.Series,
    window: int,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Rolling annualized volatility.

    Args:
        returns: Periodic returns.
        window: Rolling window length.
        periods_per_year: Number of periods per year.

    Returns:
        Rolling annualized volatility series.
    """
    r = returns.dropna()
    if r.empty:
        return r

    roll_std = r.rolling(window).std(ddof=1)
    return roll_std * np.sqrt(periods_per_year)


def rolling_sharpe(
    returns: pd.Series,
    window: int,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Rolling Sharpe ratio (annualized).

    Args:
        returns: Periodic returns.
        window: Window length.
        risk_free_rate: Annual RF rate.
        periods_per_year: Periods per year.

    Returns:
        Series of rolling Sharpe ratios.
    """
    r = returns.dropna()
    if r.empty:
        return r

    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_per_period

    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std(ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (roll_mean * periods_per_year) / (roll_std * np.sqrt(periods_per_year))

    return sharpe


# -----------------------------------------------------------------------------
# Basic VaR / CVaR (for quick calculations)
# -----------------------------------------------------------------------------

def value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Basic Value at Risk (VaR) calculation.

    Args:
        returns: Periodic returns.
        confidence_level: Confidence level (e.g. 0.95).
        method: 'historical' or 'parametric'.

    Returns:
        VaR as positive loss number (e.g. 0.03 = 3% loss).
    """
    r = returns.dropna()
    if len(r) < 30:
        return float("nan")

    alpha = 1.0 - confidence_level

    if method == "historical":
        var = -np.percentile(r, alpha * 100.0)
    elif method == "parametric":
        mean = float(r.mean())
        std = float(r.std(ddof=1))
        z = stats.norm.ppf(alpha)
        var = -(mean + z * std)
    else:
        raise ValueError("method must be 'historical' or 'parametric'")

    return float(max(var, 0.0))


def expected_shortfall(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Basic Expected Shortfall (CVaR).

    Args:
        returns: Periodic returns.
        confidence_level: Confidence level.

    Returns:
        CVaR as positive expected loss beyond VaR.
    """
    r = returns.dropna()
    if len(r) < 30:
        return float("nan")

    alpha = 1.0 - confidence_level
    var_threshold = np.percentile(r, alpha * 100.0)

    tail_losses = r[r <= var_threshold]
    if tail_losses.empty:
        return float("nan")

    cvar = -float(tail_losses.mean())
    return max(cvar, 0.0)


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    np.random.seed(42)
    r = pd.Series(np.random.normal(0.0005, 0.01, len(idx)), index=idx)
    b = pd.Series(np.random.normal(0.0004, 0.009, len(idx)), index=idx)

    logger.info("Annualized return: %.2f%%", annualize_return(r) * 100)
    logger.info("Annualized vol: %.2f%%", annualize_volatility(r) * 100)
    logger.info("Sharpe: %.2f", sharpe_ratio(r, risk_free_rate=0.02))
    logger.info("Sortino: %.2f", sortino_ratio(r, risk_free_rate=0.02))

    prices = (1.0 + r).cumprod()
    mdd, peak, trough = max_drawdown(prices)
    logger.info("Max drawdown: %.2f%%, peak=%s, trough=%s", mdd * 100, peak, trough)

    ba = beta_alpha(r, b)
    logger.info("Beta: %.3f, Alpha (annual): %.2f%%", ba["beta"], ba["alpha_annual"] * 100)

    var_95 = value_at_risk(r, 0.95)
    es_95 = expected_shortfall(r, 0.95)
    logger.info("VaR 95: %.2f%%, ES 95: %.2f%%", var_95 * 100, es_95 * 100)
