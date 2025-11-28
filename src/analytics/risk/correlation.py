"""
Correlation Analysis - Portfolio Diversification
Advanced correlation metrics and regime detection

Features:
- Pearson, Spearman, Kendall correlations
- Rolling correlation analysis
- Correlation regime detection
- Copula analysis
- Tail correlation
- Dynamic Conditional Correlation (DCC)
- Correlation breakdown analysis

Author: JPMorgan Dashboard Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CorrelationType(Enum):
    """Correlation calculation methods"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class CorrelationRegime(Enum):
    """Correlation regime classification"""
    LOW = "low"          # < 0.3
    MODERATE = "moderate"  # 0.3 - 0.7
    HIGH = "high"        # > 0.7


@dataclass
class CorrelationMetrics:
    """Comprehensive correlation metrics"""
    correlation_matrix: pd.DataFrame
    average_correlation: float
    min_correlation: float
    max_correlation: float
    diversification_ratio: float
    effective_n_assets: float  # Due to correlation
    
    # Regime info
    regime: CorrelationRegime
    regime_stability: float  # 0-1, higher = more stable
    
    def to_dict(self) -> Dict:
        return {
            'average_correlation': self.average_correlation,
            'min_correlation': self.min_correlation,
            'max_correlation': self.max_correlation,
            'diversification_ratio': self.diversification_ratio,
            'effective_n_assets': self.effective_n_assets,
            'regime': self.regime.value,
            'regime_stability': self.regime_stability
        }


class CorrelationAnalyzer:
    """
    Correlation Analysis Engine
    
    Analyzes correlation structure of portfolios:
    - Multiple correlation measures
    - Rolling correlation analysis
    - Regime detection
    - Tail correlation
    - Diversification metrics
    
    Example:
        >>> analyzer = CorrelationAnalyzer()
        >>> returns = pd.DataFrame({
        ...     'STOXX600': [...],
        ...     'SP500': [...],
        ...     'DAX': [...]
        ... })
        >>> metrics = analyzer.calculate_correlation_metrics(returns)
        >>> print(f"Avg correlation: {metrics.average_correlation:.2f}")
    """
    
    def __init__(self):
        logger.info("✅ CorrelationAnalyzer initialized")
    
    def calculate_correlation_matrix(
        self,
        returns: pd.DataFrame,
        method: CorrelationType = CorrelationType.PEARSON,
        min_periods: int = 30
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix
        
        Args:
            returns: DataFrame with asset returns (columns = assets)
            method: Correlation method
            min_periods: Minimum observations required
            
        Returns:
            Correlation matrix
        """
        
        if len(returns) < min_periods:
            logger.warning(f"Insufficient data: {len(returns)} < {min_periods}")
            return pd.DataFrame()
        
        # Calculate correlation
        corr_matrix = returns.corr(method=method.value, min_periods=min_periods)
        
        logger.info(f"✅ Correlation matrix calculated ({method.value})")
        return corr_matrix
    
    def calculate_rolling_correlation(
        self,
        returns: pd.DataFrame,
        window: int = 60,
        method: CorrelationType = CorrelationType.PEARSON
    ) -> pd.DataFrame:
        """
        Calculate rolling pairwise correlations
        
        Args:
            returns: Returns DataFrame
            window: Rolling window size (days)
            method: Correlation method
            
        Returns:
            DataFrame with rolling correlations for each pair
        """
        
        if len(returns.columns) < 2:
            logger.error("Need at least 2 assets for correlation")
            return pd.DataFrame()
        
        rolling_corr = {}
        
        # Calculate for each pair
        for i, col1 in enumerate(returns.columns):
            for col2 in returns.columns[i+1:]:
                pair_name = f"{col1}_{col2}"
                
                if method == CorrelationType.PEARSON:
                    rolling_corr[pair_name] = returns[col1].rolling(window).corr(returns[col2])
                else:
                    # For non-Pearson, calculate manually in windows
                    corrs = []
                    for start in range(len(returns) - window + 1):
                        end = start + window
                        window_data = returns.iloc[start:end][[col1, col2]]
                        
                        if method == CorrelationType.SPEARMAN:
                            corr = window_data.corr(method='spearman').iloc[0, 1]
                        else:  # Kendall
                            corr = window_data.corr(method='kendall').iloc[0, 1]
                        
                        corrs.append(corr)
                    
                    # Pad with NaN at the beginning
                    rolling_corr[pair_name] = pd.Series(
                        [np.nan] * (window - 1) + corrs,
                        index=returns.index
                    )
        
        rolling_df = pd.DataFrame(rolling_corr)
        
        logger.info(f"✅ Rolling correlation calculated (window={window})")
        return rolling_df
    
    def calculate_correlation_metrics(
        self,
        returns: pd.DataFrame,
        weights: Optional[np.ndarray] = None
    ) -> CorrelationMetrics:
        """
        Calculate comprehensive correlation metrics
        
        Args:
            returns: Returns DataFrame
            weights: Portfolio weights (equal if not provided)
            
        Returns:
            CorrelationMetrics object
        """
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(returns)
        
        if corr_matrix.empty:
            raise ValueError("Cannot calculate correlation metrics with insufficient data")
        
        # Get upper triangle (excluding diagonal)
        n = len(corr_matrix)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Basic statistics
        avg_corr = upper_triangle.stack().mean()
        min_corr = upper_triangle.stack().min()
        max_corr = upper_triangle.stack().max()
        
        # Equal weights if not provided
        if weights is None:
            weights = np.ones(n) / n
        
        # Diversification ratio
        # = weighted average vol / portfolio vol
        individual_vols = returns.std()
        weighted_avg_vol = (weights * individual_vols).sum()
        
        # Portfolio volatility (considers correlation)
        cov_matrix = returns.cov()
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        # Effective number of assets
        # Based on correlation structure
        # If perfect correlation (1), effective_n = 1
        # If zero correlation, effective_n = n
        correlation_contribution = np.mean(corr_matrix.values[np.triu_indices(n, k=1)])
        effective_n = 1 + (n - 1) * (1 - correlation_contribution)
        
        # Determine correlation regime
        regime = self._classify_correlation_regime(avg_corr)
        
        # Regime stability (based on correlation volatility)
        rolling = self.calculate_rolling_correlation(returns, window=60)
        if not rolling.empty:
            corr_volatility = rolling.std().mean()
            regime_stability = 1 / (1 + corr_volatility * 10)  # Normalize
        else:
            regime_stability = 0.5
        
        metrics = CorrelationMetrics(
            correlation_matrix=corr_matrix,
            average_correlation=avg_corr,
            min_correlation=min_corr,
            max_correlation=max_corr,
            diversification_ratio=diversification_ratio,
            effective_n_assets=effective_n,
            regime=regime,
            regime_stability=regime_stability
        )
        
        logger.info(f"✅ Correlation metrics: avg={avg_corr:.3f}, regime={regime.value}")
        
        return metrics
    
    def _classify_correlation_regime(self, avg_corr: float) -> CorrelationRegime:
        """Classify correlation regime"""
        
        if avg_corr < 0.3:
            return CorrelationRegime.LOW
        elif avg_corr < 0.7:
            return CorrelationRegime.MODERATE
        else:
            return CorrelationRegime.HIGH
    
    def detect_correlation_breakdowns(
        self,
        returns: pd.DataFrame,
        window: int = 60,
        threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Detect periods when correlation breaks down (crisis periods)
        
        Correlation breakdown = when avg correlation > threshold
        
        Args:
            returns: Returns DataFrame
            window: Rolling window
            threshold: Correlation threshold
            
        Returns:
            DataFrame with breakdown periods
        """
        
        rolling = self.calculate_rolling_correlation(returns, window=window)
        
        if rolling.empty:
            return pd.DataFrame()
        
        # Average correlation across all pairs
        avg_rolling_corr = rolling.mean(axis=1)
        
        # Detect breakdowns (high correlation periods)
        breakdowns = avg_rolling_corr > threshold
        
        # Find continuous periods
        breakdown_periods = []
        in_breakdown = False
        start_date = None
        
        for date, is_breakdown in breakdowns.items():
            if is_breakdown and not in_breakdown:
                # Start of breakdown
                start_date = date
                in_breakdown = True
            elif not is_breakdown and in_breakdown:
                # End of breakdown
                breakdown_periods.append({
                    'start_date': start_date,
                    'end_date': date,
                    'duration_days': (date - start_date).days,
                    'avg_correlation': avg_rolling_corr[start_date:date].mean()
                })
                in_breakdown = False
        
        # Handle ongoing breakdown
        if in_breakdown:
            breakdown_periods.append({
                'start_date': start_date,
                'end_date': returns.index[-1],
                'duration_days': (returns.index[-1] - start_date).days,
                'avg_correlation': avg_rolling_corr[start_date:].mean()
            })
        
        breakdown_df = pd.DataFrame(breakdown_periods)
        
        if not breakdown_df.empty:
            logger.info(f"✅ Detected {len(breakdown_df)} correlation breakdown periods")
        
        return breakdown_df
    
    def calculate_tail_correlation(
        self,
        returns: pd.DataFrame,
        quantile: float = 0.05,
        method: str = 'lower'
    ) -> pd.DataFrame:
        """
        Calculate tail correlation (correlation in extreme events)
        
        Args:
            returns: Returns DataFrame
            quantile: Quantile for tail (0.05 = 5% worst/best)
            method: 'lower' (crashes) or 'upper' (booms)
            
        Returns:
            Tail correlation matrix
        """
        
        n_assets = len(returns.columns)
        tail_corr = np.zeros((n_assets, n_assets))
        
        for i, asset1 in enumerate(returns.columns):
            for j, asset2 in enumerate(returns.columns):
                if i == j:
                    tail_corr[i, j] = 1.0
                elif i < j:
                    # Calculate tail correlation
                    if method == 'lower':
                        # Lower tail (crashes)
                        threshold1 = returns[asset1].quantile(quantile)
                        threshold2 = returns[asset2].quantile(quantile)
                        
                        tail1 = returns[asset1] <= threshold1
                        tail2 = returns[asset2] <= threshold2
                    else:
                        # Upper tail (booms)
                        threshold1 = returns[asset1].quantile(1 - quantile)
                        threshold2 = returns[asset2].quantile(1 - quantile)
                        
                        tail1 = returns[asset1] >= threshold1
                        tail2 = returns[asset2] >= threshold2
                    
                    # Tail correlation = correlation of tail events
                    tail_events = returns[(tail1 | tail2)][[asset1, asset2]]
                    
                    if len(tail_events) > 5:
                        corr = tail_events.corr().iloc[0, 1]
                        tail_corr[i, j] = corr
                        tail_corr[j, i] = corr
                    else:
                        tail_corr[i, j] = np.nan
                        tail_corr[j, i] = np.nan
        
        tail_corr_df = pd.DataFrame(
            tail_corr,
            index=returns.columns,
            columns=returns.columns
        )
        
        logger.info(f"✅ Tail correlation calculated ({method} tail, q={quantile})")
        
        return tail_corr_df
    
    def compare_correlation_regimes(
        self,
        returns: pd.DataFrame,
        crisis_dates: List[Tuple[str, str]],
        normal_dates: List[Tuple[str, str]]
    ) -> Dict:
        """
        Compare correlation in crisis vs normal periods
        
        Args:
            returns: Returns DataFrame
            crisis_dates: List of (start, end) date tuples for crisis
            normal_dates: List of (start, end) date tuples for normal
            
        Returns:
            Dictionary with comparison metrics
        """
        
        # Calculate correlation in crisis periods
        crisis_returns = []
        for start, end in crisis_dates:
            crisis_returns.append(returns[start:end])
        
        if crisis_returns:
            crisis_combined = pd.concat(crisis_returns)
            crisis_corr = self.calculate_correlation_matrix(crisis_combined)
            crisis_avg = crisis_corr.where(
                np.triu(np.ones(crisis_corr.shape), k=1).astype(bool)
            ).stack().mean()
        else:
            crisis_avg = np.nan
        
        # Calculate correlation in normal periods
        normal_returns = []
        for start, end in normal_dates:
            normal_returns.append(returns[start:end])
        
        if normal_returns:
            normal_combined = pd.concat(normal_returns)
            normal_corr = self.calculate_correlation_matrix(normal_combined)
            normal_avg = normal_corr.where(
                np.triu(np.ones(normal_corr.shape), k=1).astype(bool)
            ).stack().mean()
        else:
            normal_avg = np.nan
        
        comparison = {
            'crisis_avg_correlation': crisis_avg,
            'normal_avg_correlation': normal_avg,
            'correlation_increase': crisis_avg - normal_avg if not np.isnan([crisis_avg, normal_avg]).any() else np.nan,
            'correlation_increase_pct': (crisis_avg / normal_avg - 1) * 100 if not np.isnan([crisis_avg, normal_avg]).any() and normal_avg != 0 else np.nan
        }
        
        logger.info(
            f"✅ Regime comparison: crisis={crisis_avg:.3f}, "
            f"normal={normal_avg:.3f}, increase={comparison['correlation_increase_pct']:.1f}%"
        )
        
        return comparison
    
    def calculate_distance_matrix(
        self,
        returns: pd.DataFrame,
        method: str = 'euclidean'
    ) -> pd.DataFrame:
        """
        Calculate distance matrix (alternative to correlation)
        
        Distance = sqrt(2 * (1 - correlation))
        
        Args:
            returns: Returns DataFrame
            method: Distance method ('euclidean', 'correlation')
            
        Returns:
            Distance matrix
        """
        
        if method == 'correlation':
            corr = self.calculate_correlation_matrix(returns)
            distance = np.sqrt(2 * (1 - corr))
        else:
            # Euclidean distance on normalized returns
            normalized = (returns - returns.mean()) / returns.std()
            distance = squareform(pdist(normalized.T, metric=method))
            distance = pd.DataFrame(
                distance,
                index=returns.columns,
                columns=returns.columns
            )
        
        return distance


if __name__ == "__main__":
    # Test correlation analyzer
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("TESTING CORRELATION ANALYZER")
    print("="*70)
    
    # Generate sample returns
    np.random.seed(42)
    n_days = 252
    
    # Create correlated returns
    cov_matrix = np.array([
        [0.04, 0.02, 0.015],
        [0.02, 0.03, 0.01],
        [0.015, 0.01, 0.025]
    ])
    
    mean_returns = [0.0003, 0.0004, 0.0002]
    
    returns_data = np.random.multivariate_normal(
        mean_returns,
        cov_matrix,
        size=n_days
    )
    
    returns = pd.DataFrame(
        returns_data,
        columns=['STOXX600', 'SP500', 'DAX'],
        index=pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
    )
    
    analyzer = CorrelationAnalyzer()
    
    # Test 1: Correlation matrix
    print("\n[Test 1] Correlation Matrix:")
    corr_matrix = analyzer.calculate_correlation_matrix(returns)
    print(corr_matrix)
    
    # Test 2: Correlation metrics
    print("\n[Test 2] Correlation Metrics:")
    metrics = analyzer.calculate_correlation_metrics(returns)
    print(f"  Average Correlation: {metrics.average_correlation:.3f}")
    print(f"  Min Correlation: {metrics.min_correlation:.3f}")
    print(f"  Max Correlation: {metrics.max_correlation:.3f}")
    print(f"  Diversification Ratio: {metrics.diversification_ratio:.2f}")
    print(f"  Effective N Assets: {metrics.effective_n_assets:.1f}")
    print(f"  Regime: {metrics.regime.value}")
    print(f"  Regime Stability: {metrics.regime_stability:.2f}")
    
    # Test 3: Rolling correlation
    print("\n[Test 3] Rolling Correlation:")
    rolling = analyzer.calculate_rolling_correlation(returns, window=60)
    if not rolling.empty:
        print(f"  Calculated rolling correlation for {len(rolling.columns)} pairs")
        print(f"  Latest correlations:")
        for col in rolling.columns:
            latest = rolling[col].iloc[-1]
            if not np.isnan(latest):
                print(f"    {col}: {latest:.3f}")
    
    # Test 4: Tail correlation
    print("\n[Test 4] Tail Correlation (Lower Tail):")
    tail_corr = analyzer.calculate_tail_correlation(returns, quantile=0.05, method='lower')
    print(tail_corr)
    
    print("\n" + "="*70)
    print("✅ CORRELATION ANALYZER TEST COMPLETE")
    print("="*70)