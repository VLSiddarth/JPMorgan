"""
Portfolio Optimization Module
Implements various portfolio optimization strategies including Mean-Variance,
Risk Parity, Black-Litterman, and Maximum Sharpe
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy import linalg
import cvxpy as cp

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    metrics: Dict[str, float]
    success: bool
    message: str


class PortfolioOptimizer:
    """Main portfolio optimization engine"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate
        
    def optimize_mean_variance(self, 
                              returns: pd.DataFrame,
                              target_return: Optional[float] = None,
                              target_risk: Optional[float] = None,
                              constraints: Optional[Dict] = None) -> OptimizationResult:
        """
        Mean-Variance optimization (Markowitz)
        
        Args:
            returns: Historical returns DataFrame
            target_return: Target return (optional)
            target_risk: Target risk/volatility (optional)
            constraints: Additional constraints
            
        Returns:
            Optimization result
        """
        try:
            assets = returns.columns.tolist()
            n_assets = len(assets)
            
            # Calculate expected returns and covariance
            mu = returns.mean() * 252  # Annualized
            cov = returns.cov() * 252  # Annualized
            
            # Setup optimization
            w = cp.Variable(n_assets)
            port_return = mu.values @ w
            port_risk = cp.quad_form(w, cov.values)
            
            # Base constraints
            constraints_list = [
                cp.sum(w) == 1,  # Fully invested
                w >= 0  # Long only
            ]
            
            # Add custom constraints
            if constraints:
                if 'max_weight' in constraints:
                    constraints_list.append(w <= constraints['max_weight'])
                if 'min_weight' in constraints:
                    constraints_list.append(w >= constraints['min_weight'])
                if 'sector_limits' in constraints:
                    # Add sector constraints if provided
                    pass
            
            # Objective function
            if target_return is not None:
                # Minimize risk for target return
                constraints_list.append(port_return >= target_return)
                objective = cp.Minimize(port_risk)
            elif target_risk is not None:
                # Maximize return for target risk
                constraints_list.append(port_risk <= target_risk**2)
                objective = cp.Maximize(port_return)
            else:
                # Maximize Sharpe ratio
                objective = cp.Maximize(port_return - self.risk_free_rate)
                constraints_list.append(port_risk <= 1)
            
            # Solve
            problem = cp.Problem(objective, constraints_list)
            problem.solve()
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                return OptimizationResult(
                    weights={},
                    expected_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    metrics={},
                    success=False,
                    message=f"Optimization failed: {problem.status}"
                )
            
            # Extract results
            weights_array = w.value
            weights_dict = {asset: float(weight) for asset, weight in zip(assets, weights_array)}
            
            # Calculate metrics
            exp_return = float(port_return.value)
            volatility = np.sqrt(float(port_risk.value))
            sharpe = (exp_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=exp_return,
                volatility=volatility,
                sharpe_ratio=sharpe,
                metrics={
                    'diversification_ratio': self._calc_diversification_ratio(weights_array, cov.values)
                },
                success=True,
                message="Optimization successful"
            )
            
        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}")
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                metrics={},
                success=False,
                message=str(e)
            )
    
    def optimize_max_sharpe(self, returns: pd.DataFrame,
                           constraints: Optional[Dict] = None) -> OptimizationResult:
        """
        Maximum Sharpe Ratio optimization
        
        Args:
            returns: Historical returns DataFrame
            constraints: Additional constraints
            
        Returns:
            Optimization result
        """
        try:
            assets = returns.columns.tolist()
            n_assets = len(assets)
            
            mu = returns.mean() * 252
            cov = returns.cov() * 252
            
            def neg_sharpe(weights):
                port_return = np.dot(weights, mu)
                port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
                return -(port_return - self.risk_free_rate) / port_vol
            
            # Constraints
            cons = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Fully invested
            ]
            
            # Bounds
            bounds = tuple((0, 1) for _ in range(n_assets))  # Long only
            
            if constraints and 'max_weight' in constraints:
                bounds = tuple((0, constraints['max_weight']) for _ in range(n_assets))
            
            # Initial guess (equal weight)
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(neg_sharpe, x0, method='SLSQP', 
                            bounds=bounds, constraints=cons)
            
            if not result.success:
                return OptimizationResult(
                    weights={},
                    expected_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    metrics={},
                    success=False,
                    message=result.message
                )
            
            weights_dict = {asset: float(weight) for asset, weight in zip(assets, result.x)}
            exp_return = float(np.dot(result.x, mu))
            volatility = float(np.sqrt(np.dot(result.x, np.dot(cov, result.x))))
            sharpe = -float(result.fun)
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=exp_return,
                volatility=volatility,
                sharpe_ratio=sharpe,
                metrics={
                    'optimization_method': 'max_sharpe',
                    'iterations': result.nit
                },
                success=True,
                message="Optimization successful"
            )
            
        except Exception as e:
            logger.error(f"Max Sharpe optimization failed: {e}")
            return OptimizationResult(
                weights={}, expected_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, metrics={}, success=False, message=str(e)
            )
    
    def optimize_risk_parity(self, returns: pd.DataFrame) -> OptimizationResult:
        """
        Risk Parity optimization - equal risk contribution
        
        Args:
            returns: Historical returns DataFrame
            
        Returns:
            Optimization result
        """
        try:
            assets = returns.columns.tolist()
            n_assets = len(assets)
            
            cov = returns.cov() * 252
            
            def risk_contrib_diff(weights):
                """Minimize difference in risk contributions"""
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
                marginal_contrib = np.dot(cov, weights)
                risk_contrib = weights * marginal_contrib / portfolio_vol
                target_risk = portfolio_vol / n_assets
                return np.sum((risk_contrib - target_risk) ** 2)
            
            # Constraints
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = tuple((0.01, 1) for _ in range(n_assets))
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(risk_contrib_diff, x0, method='SLSQP',
                            bounds=bounds, constraints=cons)
            
            if not result.success:
                return OptimizationResult(
                    weights={}, expected_return=0.0, volatility=0.0,
                    sharpe_ratio=0.0, metrics={}, success=False,
                    message=result.message
                )
            
            weights_dict = {asset: float(weight) for asset, weight in zip(assets, result.x)}
            
            mu = returns.mean() * 252
            exp_return = float(np.dot(result.x, mu))
            volatility = float(np.sqrt(np.dot(result.x, np.dot(cov, result.x))))
            sharpe = (exp_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Calculate risk contributions
            marginal_contrib = np.dot(cov, result.x)
            risk_contrib = result.x * marginal_contrib / volatility
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=exp_return,
                volatility=volatility,
                sharpe_ratio=sharpe,
                metrics={
                    'risk_contributions': {asset: float(rc) 
                                          for asset, rc in zip(assets, risk_contrib)},
                    'optimization_method': 'risk_parity'
                },
                success=True,
                message="Risk parity optimization successful"
            )
            
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return OptimizationResult(
                weights={}, expected_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, metrics={}, success=False, message=str(e)
            )
    
    def optimize_min_variance(self, returns: pd.DataFrame,
                             constraints: Optional[Dict] = None) -> OptimizationResult:
        """
        Minimum Variance optimization
        
        Args:
            returns: Historical returns DataFrame
            constraints: Additional constraints
            
        Returns:
            Optimization result
        """
        try:
            assets = returns.columns.tolist()
            n_assets = len(assets)
            
            cov = returns.cov() * 252
            
            def portfolio_variance(weights):
                return np.dot(weights, np.dot(cov, weights))
            
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            if constraints and 'max_weight' in constraints:
                bounds = tuple((0, constraints['max_weight']) for _ in range(n_assets))
            
            x0 = np.array([1/n_assets] * n_assets)
            
            result = minimize(portfolio_variance, x0, method='SLSQP',
                            bounds=bounds, constraints=cons)
            
            if not result.success:
                return OptimizationResult(
                    weights={}, expected_return=0.0, volatility=0.0,
                    sharpe_ratio=0.0, metrics={}, success=False,
                    message=result.message
                )
            
            weights_dict = {asset: float(weight) for asset, weight in zip(assets, result.x)}
            
            mu = returns.mean() * 252
            exp_return = float(np.dot(result.x, mu))
            volatility = float(np.sqrt(result.fun))
            sharpe = (exp_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=exp_return,
                volatility=volatility,
                sharpe_ratio=sharpe,
                metrics={'optimization_method': 'min_variance'},
                success=True,
                message="Minimum variance optimization successful"
            )
            
        except Exception as e:
            logger.error(f"Min variance optimization failed: {e}")
            return OptimizationResult(
                weights={}, expected_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, metrics={}, success=False, message=str(e)
            )
    
    def optimize_black_litterman(self,
                                returns: pd.DataFrame,
                                market_caps: Dict[str, float],
                                views: Dict[str, float],
                                view_confidence: float = 0.25) -> OptimizationResult:
        """
        Black-Litterman optimization with investor views
        
        Args:
            returns: Historical returns DataFrame
            market_caps: Market capitalizations for equilibrium weights
            views: Expected returns views {asset: expected_return}
            view_confidence: Confidence in views (0-1)
            
        Returns:
            Optimization result
        """
        try:
            assets = returns.columns.tolist()
            n_assets = len(assets)
            
            # Market-cap weighted equilibrium
            total_mcap = sum(market_caps.values())
            w_eq = np.array([market_caps.get(asset, 0) / total_mcap for asset in assets])
            
            # Covariance and implied returns
            cov = returns.cov() * 252
            risk_aversion = 2.5
            pi = risk_aversion * np.dot(cov, w_eq)  # Implied equilibrium returns
            
            # Views matrix
            P = np.zeros((len(views), n_assets))
            Q = np.zeros(len(views))
            
            for i, (asset, view_return) in enumerate(views.items()):
                if asset in assets:
                    P[i, assets.index(asset)] = 1
                    Q[i] = view_return
            
            # View uncertainty (Omega)
            tau = 0.025  # Scaling factor
            omega = np.diag(np.diag(P @ (tau * cov) @ P.T)) / view_confidence
            
            # Black-Litterman expected returns
            M_inverse = linalg.inv(linalg.inv(tau * cov) + P.T @ linalg.inv(omega) @ P)
            mu_bl = M_inverse @ (linalg.inv(tau * cov) @ pi + P.T @ linalg.inv(omega) @ Q)
            
            # Optimize with BL returns
            w = cp.Variable(n_assets)
            port_return = mu_bl @ w
            port_risk = cp.quad_form(w, cov.values)
            
            objective = cp.Maximize(port_return - 0.5 * risk_aversion * port_risk)
            constraints = [cp.sum(w) == 1, w >= 0]
            
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                return OptimizationResult(
                    weights={}, expected_return=0.0, volatility=0.0,
                    sharpe_ratio=0.0, metrics={}, success=False,
                    message=f"BL optimization failed: {problem.status}"
                )
            
            weights_dict = {asset: float(weight) for asset, weight in zip(assets, w.value)}
            exp_return = float(np.dot(w.value, mu_bl))
            volatility = float(np.sqrt(np.dot(w.value, np.dot(cov, w.value))))
            sharpe = (exp_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            return OptimizationResult(
                weights=weights_dict,
                expected_return=exp_return,
                volatility=volatility,
                sharpe_ratio=sharpe,
                metrics={
                    'optimization_method': 'black_litterman',
                    'views_applied': len(views),
                    'view_confidence': view_confidence
                },
                success=True,
                message="Black-Litterman optimization successful"
            )
            
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            return OptimizationResult(
                weights={}, expected_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, metrics={}, success=False, message=str(e)
            )
    
    def _calc_diversification_ratio(self, weights: np.ndarray, 
                                   cov_matrix: np.ndarray) -> float:
        """Calculate portfolio diversification ratio"""
        try:
            weighted_vols = weights * np.sqrt(np.diag(cov_matrix))
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return float(np.sum(weighted_vols) / portfolio_vol)
        except:
            return 0.0
    
    def compare_strategies(self, returns: pd.DataFrame,
                          strategies: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple optimization strategies
        
        Args:
            returns: Historical returns DataFrame
            strategies: List of strategy names to compare
            
        Returns:
            Comparison DataFrame
        """
        if strategies is None:
            strategies = ['max_sharpe', 'min_variance', 'risk_parity']
        
        results = []
        
        for strategy in strategies:
            if strategy == 'max_sharpe':
                result = self.optimize_max_sharpe(returns)
            elif strategy == 'min_variance':
                result = self.optimize_min_variance(returns)
            elif strategy == 'risk_parity':
                result = self.optimize_risk_parity(returns)
            else:
                continue
            
            if result.success:
                results.append({
                    'strategy': strategy,
                    'expected_return': result.expected_return,
                    'volatility': result.volatility,
                    'sharpe_ratio': result.sharpe_ratio
                })
        
        return pd.DataFrame(results)