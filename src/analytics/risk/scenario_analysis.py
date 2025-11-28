"""
Scenario Analysis - Portfolio Stress Testing
Advanced scenario modeling for European equity portfolios

Features:
- Historical scenario replay (2008 crisis, COVID, European debt crisis)
- Custom scenario builder
- Factor-based scenario construction
- Monte Carlo scenario generation
- Multi-asset class scenarios
- Tail risk analysis

Author: JPMorgan Dashboard Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Scenario classification"""
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    MONTE_CARLO = "monte_carlo"
    FACTOR_BASED = "factor_based"


class AssetClass(Enum):
    """Asset classes for scenario impact"""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    CURRENCY = "currency"
    COMMODITY = "commodity"


@dataclass
class ScenarioImpact:
    """Impact parameters for a scenario"""
    asset_class: AssetClass
    return_shock: float  # Expected return impact (e.g., -0.30 for -30%)
    volatility_multiplier: float = 1.0  # Vol scaling factor
    correlation_shift: float = 0.0  # Change in correlation (+/- 0.5)
    duration_days: int = 30
    
    def to_dict(self) -> Dict:
        return {
            'asset_class': self.asset_class.value,
            'return_shock': self.return_shock,
            'volatility_multiplier': self.volatility_multiplier,
            'correlation_shift': self.correlation_shift,
            'duration_days': self.duration_days
        }


@dataclass
class Scenario:
    """Complete scenario definition"""
    name: str
    description: str
    scenario_type: ScenarioType
    probability: float  # Estimated probability (0-1)
    impacts: Dict[str, ScenarioImpact]  # Asset class -> impact
    macro_conditions: Dict[str, float] = field(default_factory=dict)
    
    # Market conditions
    vix_level: Optional[float] = None
    credit_spread_widening: Optional[float] = None  # bps
    yield_curve_shift: Optional[float] = None  # bps
    
    def get_equity_shock(self) -> float:
        """Get equity return shock"""
        if AssetClass.EQUITY.value in self.impacts:
            return self.impacts[AssetClass.EQUITY.value].return_shock
        return 0.0
    
    def get_duration(self) -> int:
        """Get scenario duration in days"""
        if self.impacts:
            return list(self.impacts.values())[0].duration_days
        return 30
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'scenario_type': self.scenario_type.value,
            'probability': self.probability,
            'impacts': {k: v.to_dict() for k, v in self.impacts.items()},
            'macro_conditions': self.macro_conditions,
            'vix_level': self.vix_level,
            'credit_spread_widening': self.credit_spread_widening
        }


@dataclass
class ScenarioResult:
    """Results from scenario analysis"""
    scenario: Scenario
    portfolio_value_initial: float
    portfolio_value_final: float
    portfolio_loss: float
    portfolio_return: float
    
    # Risk metrics
    max_drawdown: float
    var_95: float
    cvar_95: float
    sharpe_ratio: float
    
    # Time series
    equity_curve: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    
    # Contribution analysis
    position_contributions: Optional[Dict[str, float]] = None
    sector_contributions: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        result = {
            'scenario_name': self.scenario.name,
            'initial_value': self.portfolio_value_initial,
            'final_value': self.portfolio_value_final,
            'loss': self.portfolio_loss,
            'return_pct': self.portfolio_return * 100,
            'max_drawdown': self.max_drawdown * 100,
            'var_95': self.var_95 * 100,
            'cvar_95': self.cvar_95 * 100,
            'sharpe_ratio': self.sharpe_ratio
        }
        
        if self.position_contributions:
            result['position_contributions'] = self.position_contributions
        if self.sector_contributions:
            result['sector_contributions'] = self.sector_contributions
            
        return result


class ScenarioAnalyzer:
    """
    Scenario Analysis Engine
    
    Analyzes portfolio performance under various stress scenarios:
    - Historical crisis replays
    - Custom hypothetical scenarios
    - Monte Carlo simulation
    - Factor-based stress tests
    
    Example:
        >>> analyzer = ScenarioAnalyzer()
        >>> result = analyzer.run_scenario(
        ...     portfolio_value=1_000_000,
        ...     positions={'STOXX600': 0.6, 'SP500': 0.4},
        ...     scenario='2008_financial_crisis'
        ... )
        >>> print(f"Loss: {result.portfolio_loss:,.0f}")
    """
    
    def __init__(self):
        self.scenarios = self._initialize_scenarios()
        logger.info(f"✅ ScenarioAnalyzer initialized with {len(self.scenarios)} scenarios")
    
    def _initialize_scenarios(self) -> Dict[str, Scenario]:
        """Initialize predefined scenarios"""
        
        scenarios = {}
        
        # ========== Historical Scenarios ==========
        
        # 2008 Financial Crisis
        scenarios['2008_financial_crisis'] = Scenario(
            name='2008 Financial Crisis',
            description='Global financial meltdown, credit freeze, Lehman collapse',
            scenario_type=ScenarioType.HISTORICAL,
            probability=0.01,  # 1% annual probability
            impacts={
                AssetClass.EQUITY.value: ScenarioImpact(
                    AssetClass.EQUITY,
                    return_shock=-0.45,  # -45% equity crash
                    volatility_multiplier=3.5,
                    correlation_shift=0.3,  # Increased correlation
                    duration_days=120
                ),
                AssetClass.FIXED_INCOME.value: ScenarioImpact(
                    AssetClass.FIXED_INCOME,
                    return_shock=0.15,  # Flight to quality
                    volatility_multiplier=2.0,
                    duration_days=120
                )
            },
            macro_conditions={
                'gdp_shock': -3.5,  # -3.5% GDP
                'unemployment_rise': 5.0,  # +5pp unemployment
                'inflation_change': -2.0
            },
            vix_level=80.0,
            credit_spread_widening=600.0,  # +600 bps
            yield_curve_shift=-200.0  # Flight to safety
        )
        
        # COVID-19 Crash (2020)
        scenarios['covid_crash'] = Scenario(
            name='COVID-19 Pandemic Crash',
            description='Rapid pandemic-induced market crash, unprecedented vol spike',
            scenario_type=ScenarioType.HISTORICAL,
            probability=0.02,  # 2% (once in 50 years)
            impacts={
                AssetClass.EQUITY.value: ScenarioImpact(
                    AssetClass.EQUITY,
                    return_shock=-0.35,  # -35% in 30 days
                    volatility_multiplier=5.0,
                    correlation_shift=0.4,
                    duration_days=30
                ),
                AssetClass.CURRENCY.value: ScenarioImpact(
                    AssetClass.CURRENCY,
                    return_shock=-0.05,  # EUR weakens
                    volatility_multiplier=2.5,
                    duration_days=30
                )
            },
            vix_level=85.0,
            credit_spread_widening=400.0
        )
        
        # European Sovereign Debt Crisis (2011-2012)
        scenarios['european_debt_crisis'] = Scenario(
            name='European Sovereign Debt Crisis',
            description='Peripheral EU bond yields spike, fragmentation risk',
            scenario_type=ScenarioType.HISTORICAL,
            probability=0.03,  # 3%
            impacts={
                AssetClass.EQUITY.value: ScenarioImpact(
                    AssetClass.EQUITY,
                    return_shock=-0.25,  # -25%
                    volatility_multiplier=2.5,
                    duration_days=180
                ),
                AssetClass.FIXED_INCOME.value: ScenarioImpact(
                    AssetClass.FIXED_INCOME,
                    return_shock=-0.15,  # Peripheral bonds crash
                    volatility_multiplier=3.0,
                    duration_days=180
                )
            },
            macro_conditions={
                'gdp_shock': -2.0,
                'unemployment_rise': 3.0
            },
            credit_spread_widening=500.0
        )
        
        # ========== Hypothetical Scenarios ==========
        
        # US-EU Trade War
        scenarios['trade_war'] = Scenario(
            name='US-EU Trade War',
            description='Comprehensive tariffs, supply chain disruption',
            scenario_type=ScenarioType.HYPOTHETICAL,
            probability=0.05,  # 5%
            impacts={
                AssetClass.EQUITY.value: ScenarioImpact(
                    AssetClass.EQUITY,
                    return_shock=-0.20,  # -20%
                    volatility_multiplier=2.0,
                    duration_days=90
                )
            },
            macro_conditions={
                'gdp_shock': -1.5,
                'inflation_change': 1.0  # Tariff inflation
            },
            vix_level=35.0
        )
        
        # China Hard Landing
        scenarios['china_hard_landing'] = Scenario(
            name='China Economic Hard Landing',
            description='Chinese GDP growth <3%, property crisis deepens',
            scenario_type=ScenarioType.HYPOTHETICAL,
            probability=0.10,  # 10%
            impacts={
                AssetClass.EQUITY.value: ScenarioImpact(
                    AssetClass.EQUITY,
                    return_shock=-0.18,  # -18%
                    volatility_multiplier=1.8,
                    duration_days=60
                ),
                AssetClass.COMMODITY.value: ScenarioImpact(
                    AssetClass.COMMODITY,
                    return_shock=-0.25,  # Commodities crash
                    volatility_multiplier=2.0,
                    duration_days=60
                )
            },
            macro_conditions={
                'china_gdp': 2.5,
                'global_growth_impact': -0.5
            }
        )
        
        # ECB Policy Error
        scenarios['ecb_policy_error'] = Scenario(
            name='ECB Premature Tightening',
            description='ECB raises rates too fast, triggers recession',
            scenario_type=ScenarioType.HYPOTHETICAL,
            probability=0.08,  # 8%
            impacts={
                AssetClass.EQUITY.value: ScenarioImpact(
                    AssetClass.EQUITY,
                    return_shock=-0.15,  # -15%
                    volatility_multiplier=1.5,
                    duration_days=45
                ),
                AssetClass.FIXED_INCOME.value: ScenarioImpact(
                    AssetClass.FIXED_INCOME,
                    return_shock=-0.08,  # Bond sell-off
                    volatility_multiplier=1.8,
                    duration_days=45
                )
            },
            macro_conditions={
                'rate_hike': 150,  # +150 bps
                'gdp_shock': -1.0
            }
        )
        
        # Flash Crash
        scenarios['flash_crash'] = Scenario(
            name='Flash Crash',
            description='Algorithmic trading cascade, liquidity evaporation',
            scenario_type=ScenarioType.HYPOTHETICAL,
            probability=0.05,  # 5% annually
            impacts={
                AssetClass.EQUITY.value: ScenarioImpact(
                    AssetClass.EQUITY,
                    return_shock=-0.10,  # -10% intraday
                    volatility_multiplier=10.0,  # Extreme vol
                    duration_days=1
                )
            },
            vix_level=60.0
        )
        
        # Stagflation
        scenarios['stagflation'] = Scenario(
            name='Stagflation Scenario',
            description='High inflation + low growth, policy dilemma',
            scenario_type=ScenarioType.HYPOTHETICAL,
            probability=0.12,  # 12%
            impacts={
                AssetClass.EQUITY.value: ScenarioImpact(
                    AssetClass.EQUITY,
                    return_shock=-0.12,  # -12%
                    volatility_multiplier=1.6,
                    duration_days=180
                ),
                AssetClass.FIXED_INCOME.value: ScenarioImpact(
                    AssetClass.FIXED_INCOME,
                    return_shock=-0.10,  # Bonds sell off
                    volatility_multiplier=1.5,
                    duration_days=180
                )
            },
            macro_conditions={
                'inflation': 6.0,  # 6% inflation
                'gdp_shock': -0.5
            }
        )
        
        return scenarios
    
    def list_scenarios(self) -> List[str]:
        """List all available scenarios"""
        return list(self.scenarios.keys())
    
    def get_scenario(self, scenario_name: str) -> Optional[Scenario]:
        """Get scenario by name"""
        return self.scenarios.get(scenario_name)
    
    def create_custom_scenario(
        self,
        name: str,
        description: str,
        equity_shock: float,
        duration_days: int = 30,
        volatility_mult: float = 2.0,
        probability: float = 0.05
    ) -> Scenario:
        """
        Create custom scenario
        
        Args:
            name: Scenario name
            description: Description
            equity_shock: Equity return shock (-0.30 for -30%)
            duration_days: Duration in days
            volatility_mult: Volatility multiplier
            probability: Probability (0-1)
            
        Returns:
            Custom Scenario object
        """
        
        scenario = Scenario(
            name=name,
            description=description,
            scenario_type=ScenarioType.HYPOTHETICAL,
            probability=probability,
            impacts={
                AssetClass.EQUITY.value: ScenarioImpact(
                    AssetClass.EQUITY,
                    return_shock=equity_shock,
                    volatility_multiplier=volatility_mult,
                    duration_days=duration_days
                )
            }
        )
        
        return scenario
    
    def run_scenario(
        self,
        scenario: Union[str, Scenario],
        portfolio_value: float,
        positions: Optional[Dict[str, float]] = None,
        baseline_vol: float = 0.15
    ) -> ScenarioResult:
        """
        Run scenario analysis
        
        Args:
            scenario: Scenario name or Scenario object
            portfolio_value: Initial portfolio value
            positions: Position weights (ticker -> weight)
            baseline_vol: Baseline annual volatility
            
        Returns:
            ScenarioResult with detailed impact
        """
        
        # Get scenario
        if isinstance(scenario, str):
            scenario_obj = self.get_scenario(scenario)
            if not scenario_obj:
                raise ValueError(f"Unknown scenario: {scenario}")
        else:
            scenario_obj = scenario
        
        # Get equity impact
        equity_impact = scenario_obj.get_equity_shock()
        duration = scenario_obj.get_duration()
        
        # Get volatility multiplier
        vol_mult = 1.0
        if AssetClass.EQUITY.value in scenario_obj.impacts:
            vol_mult = scenario_obj.impacts[AssetClass.EQUITY.value].volatility_multiplier
        
        # Simulate price path
        daily_returns = self._simulate_scenario_returns(
            expected_return=equity_impact,
            duration=duration,
            baseline_vol=baseline_vol,
            vol_mult=vol_mult
        )
        
        # Calculate portfolio path
        equity_curve = (1 + daily_returns).cumprod() * portfolio_value
        
        # Final value
        final_value = equity_curve.iloc[-1]
        loss = portfolio_value - final_value
        portfolio_return = final_value / portfolio_value - 1
        
        # Calculate risk metrics
        max_dd = self._calculate_max_drawdown(equity_curve / portfolio_value - 1)
        var_95 = -np.percentile(daily_returns, 5)
        
        # CVaR (average of losses beyond VaR)
        tail_losses = daily_returns[daily_returns < -var_95]
        cvar_95 = -tail_losses.mean() if len(tail_losses) > 0 else var_95
        
        # Sharpe ratio (assuming 0% risk-free rate for scenario)
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        result = ScenarioResult(
            scenario=scenario_obj,
            portfolio_value_initial=portfolio_value,
            portfolio_value_final=final_value,
            portfolio_loss=loss,
            portfolio_return=portfolio_return,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            equity_curve=equity_curve,
            daily_returns=daily_returns
        )
        
        logger.info(
            f"✅ Scenario '{scenario_obj.name}': "
            f"Loss = ${loss:,.0f} ({portfolio_return*100:.1f}%)"
        )
        
        return result
    
    def _simulate_scenario_returns(
        self,
        expected_return: float,
        duration: int,
        baseline_vol: float,
        vol_mult: float
    ) -> pd.Series:
        """
        Simulate daily returns for scenario
        
        Args:
            expected_return: Total expected return over period
            duration: Days
            baseline_vol: Baseline annual vol
            vol_mult: Volatility multiplier
            
        Returns:
            Series of daily returns
        """
        
        # Daily expected return (compound)
        daily_return = (1 + expected_return) ** (1 / duration) - 1
        
        # Daily volatility
        daily_vol = baseline_vol / np.sqrt(252) * vol_mult
        
        # Generate returns with mean drift
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(daily_return, daily_vol, duration)
        
        return pd.Series(returns)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def run_all_scenarios(
        self,
        portfolio_value: float,
        positions: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Run all predefined scenarios
        
        Args:
            portfolio_value: Portfolio value
            positions: Position weights
            
        Returns:
            DataFrame with all scenario results
        """
        
        results = []
        
        for scenario_name in self.list_scenarios():
            try:
                result = self.run_scenario(
                    scenario_name,
                    portfolio_value,
                    positions
                )
                results.append(result.to_dict())
            except Exception as e:
                logger.error(f"Error running scenario {scenario_name}: {e}")
        
        df = pd.DataFrame(results)
        
        # Sort by loss (worst first)
        df = df.sort_values('loss', ascending=False)
        
        return df
    
    def monte_carlo_scenarios(
        self,
        portfolio_value: float,
        n_scenarios: int = 1000,
        time_horizon: int = 30,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Generate Monte Carlo stress scenarios
        
        Args:
            portfolio_value: Portfolio value
            n_scenarios: Number of simulations
            time_horizon: Days
            confidence_level: Confidence level
            
        Returns:
            Dictionary with MC results
        """
        
        logger.info(f"Running {n_scenarios} Monte Carlo scenarios...")
        
        # Simulate scenarios with varying severity
        losses = []
        
        for i in range(n_scenarios):
            # Random scenario parameters
            shock = np.random.uniform(-0.50, 0.20)  # -50% to +20%
            vol_mult = np.random.uniform(1.0, 5.0)
            
            # Create temp scenario
            temp_scenario = self.create_custom_scenario(
                name=f"MC_{i}",
                description="Monte Carlo scenario",
                equity_shock=shock,
                duration_days=time_horizon,
                volatility_mult=vol_mult
            )
            
            result = self.run_scenario(temp_scenario, portfolio_value)
            losses.append(result.portfolio_loss)
        
        losses = np.array(losses)
        
        # Calculate statistics
        var = np.percentile(losses, confidence_level * 100)
        cvar = losses[losses >= var].mean()
        
        return {
            'n_scenarios': n_scenarios,
            'time_horizon': time_horizon,
            'var': var,
            'cvar': cvar,
            'worst_case': losses.max(),
            'best_case': losses.min(),
            'mean_loss': losses.mean(),
            'median_loss': np.median(losses),
            'loss_distribution': losses
        }


if __name__ == "__main__":
    # Test scenario analyzer
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("TESTING SCENARIO ANALYZER")
    print("="*70)
    
    analyzer = ScenarioAnalyzer()
    
    # Test 1: List scenarios
    print("\n[Test 1] Available Scenarios:")
    for i, name in enumerate(analyzer.list_scenarios(), 1):
        scenario = analyzer.get_scenario(name)
        print(f"  {i}. {scenario.name}")
        print(f"     Probability: {scenario.probability*100:.1f}%")
        print(f"     Equity Shock: {scenario.get_equity_shock()*100:.1f}%")
    
    # Test 2: Run single scenario
    print("\n[Test 2] 2008 Financial Crisis Scenario:")
    result = analyzer.run_scenario(
        '2008_financial_crisis',
        portfolio_value=1_000_000
    )
    print(f"  Initial Value: ${result.portfolio_value_initial:,.0f}")
    print(f"  Final Value: ${result.portfolio_value_final:,.0f}")
    print(f"  Loss: ${result.portfolio_loss:,.0f} ({result.portfolio_return*100:.1f}%)")
    print(f"  Max Drawdown: {result.max_drawdown*100:.1f}%")
    print(f"  VaR (95%): {result.var_95*100:.2f}%")
    print(f"  CVaR (95%): {result.cvar_95*100:.2f}%")
    
    # Test 3: Run all scenarios
    print("\n[Test 3] All Scenarios Summary:")
    all_results = analyzer.run_all_scenarios(portfolio_value=1_000_000)
    print(all_results[['scenario_name', 'loss', 'return_pct', 'max_drawdown']].to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ SCENARIO ANALYZER TEST COMPLETE")
    print("="*70)