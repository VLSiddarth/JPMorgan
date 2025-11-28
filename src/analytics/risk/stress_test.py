"""
Stress Testing Module
Portfolio stress testing under extreme scenarios

Features:
- Predefined crisis scenarios (2008, COVID, Euro crisis, Flash crash)
- Custom scenario builder
- Multi-factor stress tests
- Reverse stress testing
- Portfolio impact analysis
- Sector-level stress testing

Author: JPMorgan Dashboard Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Stress test scenario definition"""
    name: str
    description: str
    equity_shock: float  # Expected equity return (-0.40 = -40%)
    vol_multiplier: float  # Volatility scaling factor
    correlation_increase: float  # Correlation shift (+0.2 = +20pp)
    duration_days: int
    spread_widening: Optional[float] = None  # bps for bond spreads
    vix_level: Optional[float] = None
    probability: float = 0.01  # Annual probability
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StressTestResult:
    """Results from stress test"""
    scenario_name: str
    initial_value: float
    stressed_value: float
    loss: float
    loss_pct: float
    duration_days: int
    
    # Additional metrics
    max_intraday_loss: Optional[float] = None
    recovery_time: Optional[int] = None
    sharpe_ratio: Optional[float] = None
    
    # Sector breakdown
    sector_losses: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        result = {
            'scenario': self.scenario_name,
            'initial_value': self.initial_value,
            'stressed_value': self.stressed_value,
            'loss': self.loss,
            'loss_pct': self.loss_pct,
            'duration_days': self.duration_days
        }
        
        if self.max_intraday_loss is not None:
            result['max_intraday_loss'] = self.max_intraday_loss
        if self.recovery_time is not None:
            result['recovery_time'] = self.recovery_time
        if self.sharpe_ratio is not None:
            result['sharpe_ratio'] = self.sharpe_ratio
        if self.sector_losses:
            result['sector_losses'] = self.sector_losses
            
        return result


class StressTest:
    """
    Stress Testing Engine
    
    Performs portfolio stress testing under extreme scenarios:
    - Historical crisis replays
    - Hypothetical scenarios
    - Custom stress tests
    - Reverse stress testing
    
    Example:
        >>> stress = StressTest()
        >>> result = stress.run_scenario(
        ...     portfolio_value=1_000_000,
        ...     scenario='2008_financial_crisis'
        ... )
        >>> print(f"Loss: ${result.loss:,.0f}")
        Loss: $400,000
    """
    
    def __init__(self):
        self.scenarios = self._define_scenarios()
        logger.info(f"✅ StressTest initialized with {len(self.scenarios)} scenarios")
    
    def _define_scenarios(self) -> Dict[str, StressScenario]:
        """Define predefined stress scenarios"""
        
        return {
            '2008_financial_crisis': StressScenario(
                name='2008 Financial Crisis',
                description='Global financial meltdown, Lehman collapse, credit freeze',
                equity_shock=-0.45,  # -45%
                vol_multiplier=3.5,
                correlation_increase=0.3,
                duration_days=120,
                spread_widening=600.0,
                vix_level=80.0,
                probability=0.01  # 1% per year (once in 100 years)
            ),
            
            'covid_crash': StressScenario(
                name='COVID-19 Crash',
                description='Pandemic-induced rapid market crash',
                equity_shock=-0.35,  # -35%
                vol_multiplier=5.0,
                correlation_increase=0.4,
                duration_days=30,
                spread_widening=400.0,
                vix_level=85.0,
                probability=0.02  # 2% per year
            ),
            
            'european_sovereign_crisis': StressScenario(
                name='European Sovereign Debt Crisis',
                description='Peripheral EU bond yields spike, fragmentation fears',
                equity_shock=-0.28,  # -28%
                vol_multiplier=2.5,
                correlation_increase=0.25,
                duration_days=180,
                spread_widening=500.0,
                vix_level=45.0,
                probability=0.03  # 3% per year
            ),
            
            'flash_crash': StressScenario(
                name='Flash Crash',
                description='Algorithmic trading cascade, sudden liquidity evaporation',
                equity_shock=-0.10,  # -10% intraday
                vol_multiplier=10.0,
                correlation_increase=0.5,
                duration_days=1,
                vix_level=60.0,
                probability=0.05  # 5% per year
            ),
            
            'stagflation': StressScenario(
                name='Stagflation',
                description='High inflation + low growth, policy dilemma',
                equity_shock=-0.15,  # -15%
                vol_multiplier=1.8,
                correlation_increase=0.15,
                duration_days=180,
                spread_widening=200.0,
                vix_level=35.0,
                probability=0.10  # 10% per year
            ),
            
            'geopolitical_shock': StressScenario(
                name='Major Geopolitical Event',
                description='Military conflict, cyber attack, or political crisis',
                equity_shock=-0.20,  # -20%
                vol_multiplier=2.5,
                correlation_increase=0.3,
                duration_days=60,
                spread_widening=300.0,
                vix_level=50.0,
                probability=0.08  # 8% per year
            ),
            
            'china_hard_landing': StressScenario(
                name='China Hard Landing',
                description='Chinese GDP growth <3%, property crisis deepens',
                equity_shock=-0.18,  # -18%
                vol_multiplier=2.0,
                correlation_increase=0.2,
                duration_days=90,
                vix_level=40.0,
                probability=0.12  # 12% per year
            ),
            
            'ecb_policy_error': StressScenario(
                name='ECB Policy Mistake',
                description='Premature tightening triggers recession',
                equity_shock=-0.12,  # -12%
                vol_multiplier=1.6,
                correlation_increase=0.15,
                duration_days=60,
                spread_widening=150.0,
                vix_level=32.0,
                probability=0.10  # 10% per year
            )
        }
    
    def list_scenarios(self) -> List[str]:
        """List all available stress test scenarios"""
        return list(self.scenarios.keys())
    
    def get_scenario(self, scenario_name: str) -> Optional[StressScenario]:
        """Get scenario details"""
        return self.scenarios.get(scenario_name)
    
    def run_scenario(
        self,
        portfolio_value: float,
        scenario: Union[str, StressScenario],
        positions: Optional[Dict[str, float]] = None,
        simulate_path: bool = False
    ) -> StressTestResult:
        """
        Run stress test for a scenario
        
        Args:
            portfolio_value: Initial portfolio value
            scenario: Scenario name or StressScenario object
            positions: Position breakdown (optional, for sector analysis)
            simulate_path: If True, simulate daily price path
            
        Returns:
            StressTestResult with detailed impact
            
        Example:
            >>> stress = StressTest()
            >>> result = stress.run_scenario(
            ...     portfolio_value=1_000_000,
            ...     scenario='2008_financial_crisis'
            ... )
            >>> print(f"Loss: ${result.loss:,.0f} ({result.loss_pct:.1f}%)")
        """
        
        # Get scenario
        if isinstance(scenario, str):
            scenario_obj = self.get_scenario(scenario)
            if not scenario_obj:
                raise ValueError(f"Unknown scenario: {scenario}")
        else:
            scenario_obj = scenario
        
        # Calculate final value based on shock
        stressed_value = portfolio_value * (1 + scenario_obj.equity_shock)
        loss = portfolio_value - stressed_value
        loss_pct = (loss / portfolio_value) * 100
        
        # Optional: Simulate daily path
        max_intraday_loss = None
        sharpe = None
        
        if simulate_path:
            daily_returns = self._simulate_path(
                scenario_obj.equity_shock,
                scenario_obj.duration_days,
                scenario_obj.vol_multiplier
            )
            
            # Calculate max intraday loss
            cumulative = (1 + daily_returns).cumprod()
            max_drawdown = (cumulative - 1).min()
            max_intraday_loss = abs(max_drawdown) * portfolio_value
            
            # Sharpe ratio during stress
            if daily_returns.std() > 0:
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        # Sector breakdown (if positions provided)
        sector_losses = None
        if positions:
            sector_losses = {
                sector: weight * loss 
                for sector, weight in positions.items()
            }
        
        result = StressTestResult(
            scenario_name=scenario_obj.name,
            initial_value=portfolio_value,
            stressed_value=stressed_value,
            loss=loss,
            loss_pct=loss_pct,
            duration_days=scenario_obj.duration_days,
            max_intraday_loss=max_intraday_loss,
            sharpe_ratio=sharpe,
            sector_losses=sector_losses
        )
        
        logger.info(
            f"✅ Stress test '{scenario_obj.name}': "
            f"Loss = ${loss:,.0f} ({loss_pct:.1f}%)"
        )
        
        return result
    
    def _simulate_path(
        self,
        total_return: float,
        days: int,
        vol_multiplier: float,
        baseline_vol: float = 0.15
    ) -> pd.Series:
        """
        Simulate daily return path for stress scenario
        
        Args:
            total_return: Total expected return over period
            days: Number of days
            vol_multiplier: Volatility scaling
            baseline_vol: Baseline annual volatility
            
        Returns:
            Series of daily returns
        """
        
        # Daily drift to achieve total return
        daily_drift = (1 + total_return) ** (1 / days) - 1
        
        # Daily volatility
        daily_vol = (baseline_vol / np.sqrt(252)) * vol_multiplier
        
        # Generate returns
        np.random.seed(42)  # For reproducibility in testing
        returns = np.random.normal(daily_drift, daily_vol, days)
        
        return pd.Series(returns)
    
    def run_all_scenarios(
        self,
        portfolio_value: float,
        positions: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Run all predefined scenarios
        
        Args:
            portfolio_value: Portfolio value
            positions: Position breakdown
            
        Returns:
            DataFrame with all scenario results
            
        Example:
            >>> stress = StressTest()
            >>> results = stress.run_all_scenarios(1_000_000)
            >>> print(results[['scenario', 'loss', 'loss_pct']])
        """
        
        results = []
        
        for scenario_name in self.list_scenarios():
            try:
                result = self.run_scenario(
                    portfolio_value,
                    scenario_name,
                    positions
                )
                results.append(result.to_dict())
            except Exception as e:
                logger.error(f"Error running scenario {scenario_name}: {e}")
        
        df = pd.DataFrame(results)
        
        # Sort by loss (worst first)
        if not df.empty:
            df = df.sort_values('loss', ascending=False)
        
        logger.info(f"✅ Completed stress tests for {len(df)} scenarios")
        
        return df
    
    def custom_stress_test(
        self,
        portfolio_value: float,
        equity_shock: float,
        duration_days: int = 30,
        vol_multiplier: float = 2.0,
        name: str = "Custom Scenario"
    ) -> StressTestResult:
        """
        Run custom stress test
        
        Args:
            portfolio_value: Portfolio value
            equity_shock: Equity return shock (-0.30 for -30%)
            duration_days: Duration
            vol_multiplier: Volatility multiplier
            name: Scenario name
            
        Returns:
            StressTestResult
        """
        
        custom_scenario = StressScenario(
            name=name,
            description="Custom user-defined scenario",
            equity_shock=equity_shock,
            vol_multiplier=vol_multiplier,
            correlation_increase=0.2,
            duration_days=duration_days
        )
        
        return self.run_scenario(portfolio_value, custom_scenario)
    
    def reverse_stress_test(
        self,
        portfolio_value: float,
        max_acceptable_loss: float,
        duration_days: int = 30
    ) -> Dict:
        """
        Reverse stress test: find shock that causes max acceptable loss
        
        Args:
            portfolio_value: Portfolio value
            max_acceptable_loss: Maximum loss tolerable
            duration_days: Scenario duration
            
        Returns:
            Dictionary with implied shock and scenario details
            
        Example:
            >>> stress = StressTest()
            >>> result = stress.reverse_stress_test(
            ...     portfolio_value=1_000_000,
            ...     max_acceptable_loss=200_000
            ... )
            >>> print(f"Implied shock: {result['implied_shock']*100:.1f}%")
        """
        
        # Calculate implied shock
        implied_shock = -max_acceptable_loss / portfolio_value
        
        # Find closest scenario
        closest_scenario = None
        min_diff = float('inf')
        
        for scenario_name, scenario in self.scenarios.items():
            diff = abs(scenario.equity_shock - implied_shock)
            if diff < min_diff:
                min_diff = diff
                closest_scenario = scenario_name
        
        return {
            'max_acceptable_loss': max_acceptable_loss,
            'implied_shock': implied_shock,
            'implied_shock_pct': implied_shock * 100,
            'closest_scenario': closest_scenario,
            'closest_scenario_shock': self.scenarios[closest_scenario].equity_shock * 100
        }
    
    def comparative_stress_test(
        self,
        portfolio_values: Dict[str, float],
        scenario: str
    ) -> pd.DataFrame:
        """
        Compare multiple portfolios under same stress
        
        Args:
            portfolio_values: Dict of portfolio_name -> value
            scenario: Scenario name
            
        Returns:
            DataFrame comparing portfolio stress results
        """
        
        results = []
        
        for portfolio_name, value in portfolio_values.items():
            result = self.run_scenario(value, scenario)
            result_dict = result.to_dict()
            result_dict['portfolio'] = portfolio_name
            results.append(result_dict)
        
        df = pd.DataFrame(results)
        
        return df


if __name__ == "__main__":
    # Test stress test module
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("TESTING STRESS TEST MODULE")
    print("="*70)
    
    stress = StressTest()
    
    # Test 1: List scenarios
    print("\n[Test 1] Available Scenarios:")
    for i, scenario_name in enumerate(stress.list_scenarios(), 1):
        scenario = stress.get_scenario(scenario_name)
        print(f"  {i}. {scenario.name}")
        print(f"     Shock: {scenario.equity_shock*100:.1f}%, Duration: {scenario.duration_days} days")
    
    # Test 2: Run single scenario
    print("\n[Test 2] 2008 Financial Crisis Scenario:")
    result = stress.run_scenario(
        portfolio_value=1_000_000,
        scenario='2008_financial_crisis',
        simulate_path=True
    )
    print(f"  Initial Value: ${result.initial_value:,.0f}")
    print(f"  Stressed Value: ${result.stressed_value:,.0f}")
    print(f"  Loss: ${result.loss:,.0f} ({result.loss_pct:.1f}%)")
    print(f"  Max Intraday Loss: ${result.max_intraday_loss:,.0f}")
    
    # Test 3: Run all scenarios
    print("\n[Test 3] All Scenarios Summary:")
    all_results = stress.run_all_scenarios(portfolio_value=1_000_000)
    print(all_results[['scenario', 'loss', 'loss_pct', 'duration_days']].to_string(index=False))
    
    # Test 4: Custom stress test
    print("\n[Test 4] Custom Stress Test (-25% shock):")
    custom_result = stress.custom_stress_test(
        portfolio_value=1_000_000,
        equity_shock=-0.25,
        duration_days=60,
        name="Custom -25% Shock"
    )
    print(f"  Loss: ${custom_result.loss:,.0f}")
    
    # Test 5: Reverse stress test
    print("\n[Test 5] Reverse Stress Test (Max loss: $150k):")
    reverse_result = stress.reverse_stress_test(
        portfolio_value=1_000_000,
        max_acceptable_loss=150_000
    )
    print(f"  Implied Shock: {reverse_result['implied_shock_pct']:.1f}%")
    print(f"  Closest Scenario: {reverse_result['closest_scenario']}")
    
    print("\n" + "="*70)
    print("✅ STRESS TEST MODULE TEST COMPLETE")
    print("="*70)