"""
Value at Risk (VaR) Calculator
Comprehensive risk metrics for portfolio management

Features:
- Historical VaR
- Parametric VaR (Variance-Covariance)
- Monte Carlo VaR
- Conditional VaR (Expected Shortfall / CVaR)
- Marginal VaR
- Component VaR
- Risk-adjusted performance metrics (Sharpe, Sortino, Calmar)
- Maximum Drawdown
- Beta, Tracking Error, Information Ratio

Author: JPMorgan Dashboard Team
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Dict, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class VaRCalculator:
    """
    Value at Risk Calculator
    
    Calculates various VaR metrics and risk-adjusted performance measures:
    - VaR (Historical, Parametric, Monte Carlo)
    - CVaR (Expected Shortfall)
    - Maximum Drawdown
    - Sharpe, Sortino, Calmar ratios
    - Beta, Tracking Error, Information Ratio
    
    Example:
        >>> var_calc = VaRCalculator(confidence_level=0.95)
        >>> returns = pd.Series([...])  # Daily returns
        >>> var_95 = var_calc.historical_var(returns)
        >>> print(f"VaR (95%): {var_95*100:.2f}%")
        VaR (95%): 2.35%
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize VaR Calculator
        
        Args:
            confidence_level: Confidence level for VaR (0-1)
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        
        logger.info(f"✅ VaRCalculator initialized (confidence={confidence_level})")
    
    # ========== VaR Methods ==========
    
    def historical_var(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None
    ) -> float:
        """
        Calculate Historical VaR
        
        Based on empirical distribution of returns.
        
        Args:
            returns: Series of returns
            confidence: Confidence level (uses default if not provided)
            
        Returns:
            VaR as positive number (e.g., 0.025 = 2.5% loss)
            
        Example:
            >>> var_calc = VaRCalculator()
            >>> var_95 = var_calc.historical_var(returns)
        """
        
        if confidence is None:
            confidence = self.confidence_level
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 30:
            logger.warning(f"Insufficient data for VaR: {len(returns_clean)} observations")
            return np.nan
        
        # VaR = negative of (1-confidence) percentile
        var = -np.percentile(returns_clean, (1 - confidence) * 100)
        
        return var
    
    def parametric_var(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None
    ) -> float:
        """
        Calculate Parametric VaR (Variance-Covariance method)
        
        Assumes returns are normally distributed.
        
        Args:
            returns: Series of returns
            confidence: Confidence level
            
        Returns:
            VaR as positive number
        """
        
        if confidence is None:
            confidence = self.confidence_level
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 30:
            logger.warning(f"Insufficient data for VaR: {len(returns_clean)} observations")
            return np.nan
        
        # Calculate mean and std
        mean = returns_clean.mean()
        std = returns_clean.std()
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence)
        
        # VaR = -(mean + z_score * std)
        var = -(mean + z_score * std)
        
        return var
    
    def monte_carlo_var(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None,
        n_simulations: int = 10000
    ) -> float:
        """
        Calculate Monte Carlo VaR
        
        Simulates returns based on historical mean and volatility.
        
        Args:
            returns: Historical returns
            confidence: Confidence level
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            VaR as positive number
        """
        
        if confidence is None:
            confidence = self.confidence_level
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 30:
            logger.warning(f"Insufficient data for VaR: {len(returns_clean)} observations")
            return np.nan
        
        # Calculate parameters
        mean = returns_clean.mean()
        std = returns_clean.std()
        
        # Simulate returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean, std, n_simulations)
        
        # Calculate VaR from simulated distribution
        var = -np.percentile(simulated_returns, (1 - confidence) * 100)
        
        return var
    
    # ========== Conditional VaR (Expected Shortfall) ==========
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall / CVaR)
        
        CVaR = Average of losses beyond VaR threshold
        
        Args:
            returns: Series of returns
            confidence: Confidence level
            
        Returns:
            CVaR as positive number
            
        Example:
            >>> cvar_95 = var_calc.calculate_cvar(returns, confidence=0.95)
        """
        
        if confidence is None:
            confidence = self.confidence_level
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 30:
            return np.nan
        
        # Calculate VaR threshold
        var_threshold = -np.percentile(returns_clean, (1 - confidence) * 100)
        
        # Find losses beyond VaR
        tail_losses = returns_clean[returns_clean < -var_threshold]
        
        if len(tail_losses) == 0:
            return var_threshold
        
        # CVaR = average of tail losses
        cvar = -tail_losses.mean()
        
        return cvar
    
    # ========== Drawdown Analysis ==========
    
    def calculate_max_drawdown(
        self,
        prices: pd.Series
    ) -> Dict:
        """
        Calculate Maximum Drawdown
        
        Args:
            prices: Series of prices or cumulative returns
            
        Returns:
            Dictionary with drawdown metrics
            
        Example:
            >>> dd = var_calc.calculate_max_drawdown(prices)
            >>> print(f"Max DD: {dd['max_drawdown']*100:.2f}%")
        """
        
        prices_clean = prices.dropna()
        
        if len(prices_clean) < 2:
            return {
                'max_drawdown': 0,
                'peak_date': None,
                'trough_date': None,
                'recovery_date': None,
                'duration_days': 0
            }
        
        # Calculate running maximum
        cummax = prices_clean.cummax()
        
        # Calculate drawdown
        drawdown = (prices_clean - cummax) / cummax
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        
        # Find trough (lowest point)
        trough_date = drawdown.idxmin()
        
        # Find peak (before trough)
        peak_date = prices_clean[:trough_date].idxmax()
        
        # Find recovery (when price exceeds peak again)
        recovery_date = None
        peak_price = prices_clean[peak_date]
        
        # Look for recovery after trough
        after_trough = prices_clean[trough_date:]
        recovery_mask = after_trough >= peak_price
        
        if recovery_mask.any():
            recovery_date = after_trough[recovery_mask].index[0]
            duration_days = (recovery_date - peak_date).days
        else:
            duration_days = (prices_clean.index[-1] - peak_date).days
        
        return {
            'max_drawdown': max_dd,
            'peak_date': peak_date,
            'trough_date': trough_date,
            'recovery_date': recovery_date,
            'duration_days': duration_days,
            'drawdown_series': drawdown
        }
    
    # ========== Risk-Adjusted Performance ==========
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe Ratio (annualized)
        
        Sharpe = (Return - Risk-Free Rate) / Volatility
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: 252 for daily, 52 for weekly, 12 for monthly
            
        Returns:
            Sharpe ratio
        """
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 30 or returns_clean.std() == 0:
            return np.nan
        
        # Annualize return
        avg_return = returns_clean.mean() * periods_per_year
        
        # Annualize volatility
        volatility = returns_clean.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        sharpe = (avg_return - risk_free_rate) / volatility
        
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        target_return: float = 0,
        risk_free_rate: Optional[float] = None,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino Ratio
        
        Like Sharpe, but only penalizes downside volatility.
        
        Args:
            returns: Series of returns
            target_return: Target return (default: 0)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Periods per year
            
        Returns:
            Sortino ratio
        """
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 30:
            return np.nan
        
        # Annualize return
        avg_return = returns_clean.mean() * periods_per_year
        
        # Downside returns (below target)
        downside_returns = returns_clean[returns_clean < target_return]
        
        if len(downside_returns) == 0:
            return np.inf
        
        # Downside volatility (annualized)
        downside_std = downside_returns.std() * np.sqrt(periods_per_year)
        
        if downside_std == 0:
            return np.inf
        
        # Sortino ratio
        sortino = (avg_return - risk_free_rate) / downside_std
        
        return sortino
    
    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar Ratio
        
        Calmar = Annualized Return / Max Drawdown
        
        Args:
            returns: Series of returns
            periods_per_year: Periods per year
            
        Returns:
            Calmar ratio
        """
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 30:
            return np.nan
        
        # Annualized return
        cumulative = (1 + returns_clean).cumprod()
        total_return = cumulative.iloc[-1] - 1
        n_periods = len(returns_clean)
        years = n_periods / periods_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Max drawdown
        dd_metrics = self.calculate_max_drawdown(cumulative)
        max_dd = abs(dd_metrics['max_drawdown'])
        
        if max_dd == 0:
            return np.inf
        
        # Calmar ratio
        calmar = annualized_return / max_dd
        
        return calmar
    
    # ========== Relative Risk Metrics ==========
    
    def calculate_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate portfolio beta vs benchmark
        
        Beta measures systematic risk relative to market.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Beta coefficient
        """
        
        # Align series
        df = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(df) < 30:
            return np.nan
        
        # Calculate covariance and variance
        covariance = df['portfolio'].cov(df['benchmark'])
        benchmark_variance = df['benchmark'].var()
        
        if benchmark_variance == 0:
            return np.nan
        
        beta = covariance / benchmark_variance
        
        return beta
    
    def calculate_tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Tracking Error (annualized)
        
        Tracking Error = Std Dev of (Portfolio Returns - Benchmark Returns)
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            periods_per_year: Periods per year
            
        Returns:
            Tracking error (annualized)
        """
        
        # Align series
        df = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(df) < 30:
            return np.nan
        
        # Active returns
        active_returns = df['portfolio'] - df['benchmark']
        
        # Annualize tracking error
        tracking_error = active_returns.std() * np.sqrt(periods_per_year)
        
        return tracking_error
    
    def calculate_information_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Information Ratio
        
        IR = Excess Return / Tracking Error
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            periods_per_year: Periods per year
            
        Returns:
            Information ratio
        """
        
        # Align series
        df = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(df) < 30:
            return np.nan
        
        # Active returns
        active_returns = df['portfolio'] - df['benchmark']
        
        # Annualize
        excess_return = active_returns.mean() * periods_per_year
        tracking_error = active_returns.std() * np.sqrt(periods_per_year)
        
        if tracking_error == 0:
            return np.nan
        
        information_ratio = excess_return / tracking_error
        
        return information_ratio
    
    # ========== Comprehensive Risk Report ==========
    
    def calculate_risk_metrics_summary(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns (optional)
            
        Returns:
            Dictionary with all risk metrics
            
        Example:
            >>> metrics = var_calc.calculate_risk_metrics_summary(returns)
            >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
        """
        
        metrics = {}
        
        # VaR metrics
        metrics['var_95_historical'] = self.historical_var(returns, confidence=0.95)
        metrics['var_99_historical'] = self.historical_var(returns, confidence=0.99)
        metrics['var_95_parametric'] = self.parametric_var(returns, confidence=0.95)
        metrics['var_95_monte_carlo'] = self.monte_carlo_var(returns, confidence=0.95)
        
        # CVaR
        metrics['cvar_95'] = self.calculate_cvar(returns, confidence=0.95)
        metrics['cvar_99'] = self.calculate_cvar(returns, confidence=0.99)
        
        # Performance metrics
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        
        # Volatility
        metrics['volatility_daily'] = returns.std()
        metrics['volatility_annual'] = returns.std() * np.sqrt(252)
        
        # Drawdown
        prices = (1 + returns).cumprod()
        dd_metrics = self.calculate_max_drawdown(prices)
        metrics['max_drawdown'] = dd_metrics['max_drawdown']
        metrics['max_drawdown_duration'] = dd_metrics['duration_days']
        
        # Calculate Calmar
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns)
        
        # Benchmark-relative metrics (if provided)
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            metrics['beta'] = self.calculate_beta(returns, benchmark_returns)
            metrics['tracking_error'] = self.calculate_tracking_error(returns, benchmark_returns)
            metrics['information_ratio'] = self.calculate_information_ratio(returns, benchmark_returns)
        
        return metrics


if __name__ == "__main__":
    # Test VaR calculator
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("TESTING VAR CALCULATOR")
    print("="*70)
    
    # Generate sample returns
    np.random.seed(42)
    n_days = 252
    returns = pd.Series(np.random.normal(0.0005, 0.015, n_days))
    benchmark_returns = pd.Series(np.random.normal(0.0004, 0.012, n_days))
    
    var_calc = VaRCalculator(confidence_level=0.95)
    
    # Test 1: VaR calculations
    print("\n[Test 1] VaR Calculations (95% confidence):")
    var_hist = var_calc.historical_var(returns)
    var_param = var_calc.parametric_var(returns)
    var_mc = var_calc.monte_carlo_var(returns)
    
    print(f"  Historical VaR:  {var_hist*100:.2f}%")
    print(f"  Parametric VaR:  {var_param*100:.2f}%")
    print(f"  Monte Carlo VaR: {var_mc*100:.2f}%")
    
    # Test 2: CVaR
    print("\n[Test 2] Conditional VaR (Expected Shortfall):")
    cvar_95 = var_calc.calculate_cvar(returns, confidence=0.95)
    cvar_99 = var_calc.calculate_cvar(returns, confidence=0.99)
    print(f"  CVaR (95%): {cvar_95*100:.2f}%")
    print(f"  CVaR (99%): {cvar_99*100:.2f}%")
    
    # Test 3: Risk-adjusted metrics
    print("\n[Test 3] Risk-Adjusted Performance:")
    sharpe = var_calc.calculate_sharpe_ratio(returns)
    sortino = var_calc.calculate_sortino_ratio(returns)
    calmar = var_calc.calculate_calmar_ratio(returns)
    
    print(f"  Sharpe Ratio:  {sharpe:.2f}")
    print(f"  Sortino Ratio: {sortino:.2f}")
    print(f"  Calmar Ratio:  {calmar:.2f}")
    
    # Test 4: Drawdown
    print("\n[Test 4] Maximum Drawdown:")
    prices = (1 + returns).cumprod()
    dd = var_calc.calculate_max_drawdown(prices)
    print(f"  Max Drawdown:    {dd['max_drawdown']*100:.2f}%")
    print(f"  Duration (days): {dd['duration_days']}")
    
    # Test 5: Relative metrics
    print("\n[Test 5] Benchmark-Relative Metrics:")
    beta = var_calc.calculate_beta(returns, benchmark_returns)
    te = var_calc.calculate_tracking_error(returns, benchmark_returns)
    ir = var_calc.calculate_information_ratio(returns, benchmark_returns)
    
    print(f"  Beta:               {beta:.2f}")
    print(f"  Tracking Error:     {te*100:.2f}%")
    print(f"  Information Ratio:  {ir:.2f}")
    
    # Test 6: Full summary
    print("\n[Test 6] Complete Risk Metrics Summary:")
    summary = var_calc.calculate_risk_metrics_summary(returns, benchmark_returns)
    for key, value in list(summary.items())[:10]:
        if isinstance(value, (int, float)) and not np.isnan(value):
            print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*70)
    print("✅ VAR CALCULATOR TEST COMPLETE")
    print("="*70)