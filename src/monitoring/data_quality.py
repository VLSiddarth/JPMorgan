"""
===============================================================================
monitoring/data_quality.py - Data Quality Monitoring
===============================================================================
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DataQualityIssue(Enum):
    """Types of data quality issues"""
    MISSING_DATA = "missing_data"
    STALE_DATA = "stale_data"
    OUTLIER = "outlier"
    INCONSISTENT = "inconsistent"
    DUPLICATE = "duplicate"
    INVALID_RANGE = "invalid_range"


@dataclass
class QualityCheck:
    """Data quality check result"""
    check_name: str
    passed: bool
    issue_type: Optional[DataQualityIssue]
    severity: str  # 'low', 'medium', 'high'
    details: str
    affected_records: int
    timestamp: datetime


class DataQualityMonitor:
    """Monitor data quality and detect issues"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data quality monitor
        
        Args:
            config: Configuration for quality checks
        """
        self.config = config or self._default_config()
        self.quality_reports: List[QualityCheck] = []
        
    def _default_config(self) -> Dict:
        """Default quality check configuration"""
        return {
            'max_missing_pct': 5.0,  # Max % of missing data
            'staleness_hours': 24,    # Hours before data is stale
            'outlier_std': 5.0,       # Standard deviations for outliers
            'duplicate_check': True,
            'range_checks': {}        # Custom range checks per column
        }
    
    def run_all_checks(self, df: pd.DataFrame, 
                      data_source: str = "unknown") -> Dict[str, Any]:
        """
        Run all data quality checks
        
        Args:
            df: DataFrame to check
            data_source: Name of data source
            
        Returns:
            Quality report dictionary
        """
        checks = [
            self.check_missing_data(df),
            self.check_staleness(df),
            self.check_duplicates(df),
            self.check_outliers(df),
            self.check_data_ranges(df)
        ]
        
        # Filter out None results
        checks = [c for c in checks if c is not None]
        
        # Store checks
        self.quality_reports.extend(checks)
        
        # Generate summary
        passed_checks = sum(1 for c in checks if c.passed)
        total_checks = len(checks)
        
        return {
            'data_source': data_source,
            'timestamp': datetime.now().isoformat(),
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'quality_score': (passed_checks / total_checks * 100) if total_checks > 0 else 0,
            'checks': [self._check_to_dict(c) for c in checks],
            'summary': self._generate_summary(checks)
        }
    
    def check_missing_data(self, df: pd.DataFrame) -> QualityCheck:
        """Check for missing data"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        
        threshold = self.config['max_missing_pct']
        passed = missing_pct <= threshold
        
        if not passed:
            severity = 'high' if missing_pct > threshold * 2 else 'medium'
        else:
            severity = 'low'
        
        return QualityCheck(
            check_name="missing_data",
            passed=passed,
            issue_type=DataQualityIssue.MISSING_DATA if not passed else None,
            severity=severity,
            details=f"Missing data: {missing_pct:.2f}% (threshold: {threshold}%)",
            affected_records=int(missing_cells),
            timestamp=datetime.now()
        )
    
    def check_staleness(self, df: pd.DataFrame) -> Optional[QualityCheck]:
        """Check if data is stale (too old)"""
        if 'timestamp' not in df.columns and 'date' not in df.columns:
            return None
        
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        
        try:
            latest_date = pd.to_datetime(df[date_col]).max()
            hours_old = (datetime.now() - latest_date).total_seconds() / 3600
            
            threshold = self.config['staleness_hours']
            passed = hours_old <= threshold
            
            severity = 'high' if hours_old > threshold * 2 else 'medium'
            
            return QualityCheck(
                check_name="staleness",
                passed=passed,
                issue_type=DataQualityIssue.STALE_DATA if not passed else None,
                severity=severity if not passed else 'low',
                details=f"Data age: {hours_old:.1f} hours (threshold: {threshold}h)",
                affected_records=len(df),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"Staleness check failed: {e}")
            return None
    
    def check_duplicates(self, df: pd.DataFrame) -> QualityCheck:
        """Check for duplicate rows"""
        if not self.config['duplicate_check']:
            return None
        
        duplicates = df.duplicated().sum()
        total_rows = len(df)
        duplicate_pct = (duplicates / total_rows * 100) if total_rows > 0 else 0
        
        passed = duplicates == 0
        severity = 'high' if duplicate_pct > 5 else 'medium' if duplicate_pct > 1 else 'low'
        
        return QualityCheck(
            check_name="duplicates",
            passed=passed,
            issue_type=DataQualityIssue.DUPLICATE if not passed else None,
            severity=severity,
            details=f"Duplicate rows: {duplicates} ({duplicate_pct:.2f}%)",
            affected_records=int(duplicates),
            timestamp=datetime.now()
        )
    
    def check_outliers(self, df: pd.DataFrame) -> QualityCheck:
        """Check for outliers in numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        total_outliers = 0
        outlier_details = []
        
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((df[col] - mean) / std)
            outliers = (z_scores > self.config['outlier_std']).sum()
            
            if outliers > 0:
                total_outliers += outliers
                outlier_details.append(f"{col}: {outliers} outliers")
        
        passed = total_outliers == 0
        severity = 'medium' if total_outliers > len(df) * 0.05 else 'low'
        
        details = f"Total outliers: {total_outliers}"
        if outlier_details:
            details += f" ({', '.join(outlier_details[:3])})"
        
        return QualityCheck(
            check_name="outliers",
            passed=passed,
            issue_type=DataQualityIssue.OUTLIER if not passed else None,
            severity=severity,
            details=details,
            affected_records=int(total_outliers),
            timestamp=datetime.now()
        )
    
    def check_data_ranges(self, df: pd.DataFrame) -> Optional[QualityCheck]:
        """Check if data is within expected ranges"""
        range_checks = self.config.get('range_checks', {})
        
        if not range_checks:
            return None
        
        violations = 0
        violation_details = []
        
        for col, (min_val, max_val) in range_checks.items():
            if col not in df.columns:
                continue
            
            below_min = (df[col] < min_val).sum()
            above_max = (df[col] > max_val).sum()
            
            col_violations = below_min + above_max
            if col_violations > 0:
                violations += col_violations
                violation_details.append(f"{col}: {col_violations} out of range")
        
        passed = violations == 0
        severity = 'high' if violations > len(df) * 0.1 else 'medium'
        
        details = f"Range violations: {violations}"
        if violation_details:
            details += f" ({', '.join(violation_details[:3])})"
        
        return QualityCheck(
            check_name="data_ranges",
            passed=passed,
            issue_type=DataQualityIssue.INVALID_RANGE if not passed else None,
            severity=severity if not passed else 'low',
            details=details,
            affected_records=int(violations),
            timestamp=datetime.now()
        )
    
    def _check_to_dict(self, check: QualityCheck) -> Dict:
        """Convert QualityCheck to dictionary"""
        return {
            'check_name': check.check_name,
            'passed': check.passed,
            'issue_type': check.issue_type.value if check.issue_type else None,
            'severity': check.severity,
            'details': check.details,
            'affected_records': check.affected_records,
            'timestamp': check.timestamp.isoformat()
        }
    
    def _generate_summary(self, checks: List[QualityCheck]) -> Dict:
        """Generate summary of quality checks"""
        issues_by_type = {}
        issues_by_severity = {'low': 0, 'medium': 0, 'high': 0}
        
        for check in checks:
            if not check.passed:
                if check.issue_type:
                    issue_name = check.issue_type.value
                    issues_by_type[issue_name] = issues_by_type.get(issue_name, 0) + 1
                
                issues_by_severity[check.severity] += 1
        
        return {
            'issues_by_type': issues_by_type,
            'issues_by_severity': issues_by_severity,
            'total_issues': sum(issues_by_severity.values())
        }
    
    def get_quality_trend(self, days: int = 7) -> pd.DataFrame:
        """
        Get quality score trend over time
        
        Args:
            days: Number of days to look back
            
        Returns:
            DataFrame with quality scores over time
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_checks = [c for c in self.quality_reports if c.timestamp > cutoff]
        
        if not recent_checks:
            return pd.DataFrame()
        
        # Group by date
        data = []
        for check in recent_checks:
            data.append({
                'date': check.timestamp.date(),
                'check_name': check.check_name,
                'passed': check.passed
            })
        
        df = pd.DataFrame(data)
        
        # Calculate daily quality score
        quality_by_date = df.groupby('date').agg({
            'passed': lambda x: (x.sum() / len(x) * 100)
        }).reset_index()
        quality_by_date.columns = ['date', 'quality_score']
        
        return quality_by_date


"""
===============================================================================
Complete Integration Example
===============================================================================
"""

from datetime import datetime
import pandas as pd
import numpy as np

# Example usage of all modules together
def main_example():
    """Example demonstrating complete system integration"""
    
    # 1. Setup logging
    from utils.logger import setup_logging
    logger = setup_logging(log_dir="logs", level=logging.INFO)
    logger.info("Starting JPMorgan Analytics System")
    
    # 2. Initialize monitoring systems
    from monitoring.alerts import AlertManager, AlertSeverity, AlertType
    from monitoring.health_check import HealthMonitor
    from monitoring.data_quality import DataQualityMonitor
    
    alert_manager = AlertManager()
    health_monitor = HealthMonitor()
    data_quality = DataQualityMonitor()
    
    # 3. Run health checks
    logger.info("Running system health checks...")
    health_status = health_monitor.check_all()
    health_report = health_monitor.get_health_report()
    
    print(f"System Health: {health_report['overall_status']}")
    print(f"Healthy Components: {health_report['summary']['healthy']}")
    print(f"Issues: {health_report['summary']['unhealthy'] + health_report['summary']['degraded']}")
    
    # 4. Load and validate data
    logger.info("Loading market data...")
    
    # Sample data (in production, load from your connectors)
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-11-28', freq='D')
    
    market_data = pd.DataFrame({
        'date': dates,
        'SPY': np.random.randn(len(dates)).cumsum() + 100,
        'EuroStoxx': np.random.randn(len(dates)).cumsum() + 95,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Run data quality checks
    quality_report = data_quality.run_all_checks(market_data, "market_data")
    print(f"\nData Quality Score: {quality_report['quality_score']:.1f}%")
    print(f"Failed Checks: {quality_report['failed_checks']}")
    
    # 5. Portfolio optimization
    logger.info("Running portfolio optimization...")
    from portfolio.optimizer import PortfolioOptimizer
    
    # Calculate returns
    returns = market_data[['SPY', 'EuroStoxx']].pct_change().dropna()
    
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Compare strategies
    strategies_comparison = optimizer.compare_strategies(returns)
    print("\nPortfolio Optimization Results:")
    print(strategies_comparison)
    
    # Run max Sharpe optimization
    max_sharpe_result = optimizer.optimize_max_sharpe(returns)
    
    if max_sharpe_result.success:
        print(f"\nOptimal Portfolio (Max Sharpe):")
        for asset, weight in max_sharpe_result.weights.items():
            print(f"  {asset}: {weight*100:.1f}%")
        print(f"Expected Return: {max_sharpe_result.expected_return*100:.2f}%")
        print(f"Volatility: {max_sharpe_result.volatility*100:.2f}%")
        print(f"Sharpe Ratio: {max_sharpe_result.sharpe_ratio:.2f}")
    
    # 6. Check for alerts
    logger.info("Checking for market alerts...")
    
    # Simulate price checks
    alert = alert_manager.check_price_threshold(
        asset="EuroStoxx",
        current_price=market_data['EuroStoxx'].iloc[-1],
        previous_price=market_data['EuroStoxx'].iloc[-2]
    )
    
    if alert:
        print(f"\nAlert Generated: {alert.title}")
        print(f"Severity: {alert.severity.value}")
        print(f"Message: {alert.message}")
    
    # Get alert summary
    alert_summary = alert_manager.get_alert_summary()
    print(f"\nActive Alerts Summary: {alert_summary}")
    
    # 7. Generate reports
    logger.info("Generating PDF reports...")
    from reporting.pdf_generator import PDFReportGenerator
    
    pdf_gen = PDFReportGenerator()
    
    # Prepare portfolio data for report
    portfolio_data = {
        'portfolio_name': 'European Growth Strategy',
        'metrics': {
            'total_return': 0.125,
            'ytd_return': 0.085,
            'volatility': 0.18,
            'sharpe_ratio': max_sharpe_result.sharpe_ratio,
            'max_drawdown': -0.12,
            'var_95': -0.025
        },
        'holdings': pd.DataFrame({
            'Asset': ['SPY', 'EuroStoxx'],
            'Weight': [f"{w*100:.1f}%" for w in max_sharpe_result.weights.values()],
            'Value': ['$500k', '$500k']
        }),
        'returns': returns['EuroStoxx'],
        'allocation': max_sharpe_result.weights
    }
    
    success = pdf_gen.generate_portfolio_report(
        portfolio_data,
        "data/exports/portfolio_report.pdf"
    )
    
    if success:
        print("\n✓ Portfolio report generated successfully")
    
    # 8. Summary
    logger.info("Analysis complete")
    print("\n" + "="*60)
    print("JPMorgan Analytics System - Analysis Complete")
    print("="*60)
    print(f"System Health: {health_report['overall_status']}")
    print(f"Data Quality: {quality_report['quality_score']:.1f}%")
    print(f"Optimal Sharpe: {max_sharpe_result.sharpe_ratio:.2f}")
    print(f"Active Alerts: {len(alert_manager.active_alerts)}")
    print("="*60)


if __name__ == "__main__":
    main_example()