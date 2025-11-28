"""
Enhanced Alert System for Market Monitoring
Monitors thresholds, generates alerts, and sends notifications
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import yaml
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts"""
    PRICE_THRESHOLD = "price_threshold"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    CORRELATION_BREAK = "correlation_break"
    RISK_LIMIT = "risk_limit"
    DATA_QUALITY = "data_quality"
    SYSTEM_HEALTH = "system_health"
    SIGNAL_GENERATED = "signal_generated"


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    asset: Optional[str] = None
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        data = asdict(self)
        data['alert_type'] = self.alert_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AlertManager:
    """Main alert management system"""
    
    def __init__(self, config_path: str = "config/thresholds.yml"):
        """
        Initialize alert manager
        
        Args:
            config_path: Path to thresholds configuration file
        """
        self.config_path = config_path
        self.thresholds = self._load_thresholds()
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.cooldown_periods: Dict[str, datetime] = {}
        
    def _load_thresholds(self) -> Dict:
        """Load alert thresholds from config"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load thresholds: {e}")
            return self._get_default_thresholds()
    
    def _get_default_thresholds(self) -> Dict:
        """Default threshold configuration"""
        return {
            'price_change': {
                'warning': 5.0,  # 5% change
                'critical': 10.0  # 10% change
            },
            'volatility': {
                'warning': 30.0,  # 30% annualized
                'critical': 50.0  # 50% annualized
            },
            'volume': {
                'warning_multiplier': 2.0,  # 2x average volume
                'critical_multiplier': 5.0  # 5x average volume
            },
            'var': {
                'warning': 2.0,  # 2% portfolio VaR
                'critical': 5.0  # 5% portfolio VaR
            },
            'correlation': {
                'break_threshold': 0.3  # 30% correlation change
            },
            'cooldown_minutes': 60  # Minimum time between duplicate alerts
        }
    
    def check_price_threshold(self, asset: str, current_price: float, 
                            previous_price: float) -> Optional[Alert]:
        """
        Check if price change exceeds thresholds
        
        Args:
            asset: Asset identifier
            current_price: Current price
            previous_price: Previous price for comparison
            
        Returns:
            Alert if threshold breached, None otherwise
        """
        if previous_price == 0:
            return None
            
        pct_change = abs((current_price - previous_price) / previous_price * 100)
        
        thresholds = self.thresholds['price_change']
        severity = None
        
        if pct_change >= thresholds['critical']:
            severity = AlertSeverity.CRITICAL
        elif pct_change >= thresholds['warning']:
            severity = AlertSeverity.WARNING
        
        if severity and self._check_cooldown(f"price_{asset}"):
            direction = "increased" if current_price > previous_price else "decreased"
            alert = Alert(
                alert_id=self._generate_alert_id(),
                timestamp=datetime.now(),
                alert_type=AlertType.PRICE_THRESHOLD,
                severity=severity,
                title=f"Price Alert: {asset}",
                message=f"{asset} price {direction} by {pct_change:.2f}%",
                asset=asset,
                metric_name="price_change",
                current_value=pct_change,
                threshold_value=thresholds[severity.value],
                metadata={
                    'current_price': current_price,
                    'previous_price': previous_price,
                    'direction': direction
                }
            )
            self._register_alert(alert)
            return alert
        
        return None
    
    def check_volatility_spike(self, asset: str, current_vol: float, 
                              historical_vol: float) -> Optional[Alert]:
        """
        Check for volatility spikes
        
        Args:
            asset: Asset identifier
            current_vol: Current volatility (annualized %)
            historical_vol: Historical average volatility
            
        Returns:
            Alert if spike detected
        """
        vol_increase = current_vol - historical_vol
        
        thresholds = self.thresholds['volatility']
        severity = None
        
        if current_vol >= thresholds['critical']:
            severity = AlertSeverity.CRITICAL
        elif current_vol >= thresholds['warning']:
            severity = AlertSeverity.WARNING
        
        if severity and self._check_cooldown(f"vol_{asset}"):
            alert = Alert(
                alert_id=self._generate_alert_id(),
                timestamp=datetime.now(),
                alert_type=AlertType.VOLATILITY_SPIKE,
                severity=severity,
                title=f"Volatility Spike: {asset}",
                message=f"{asset} volatility at {current_vol:.1f}% (avg: {historical_vol:.1f}%)",
                asset=asset,
                metric_name="volatility",
                current_value=current_vol,
                threshold_value=thresholds[severity.value],
                metadata={
                    'historical_volatility': historical_vol,
                    'increase': vol_increase
                }
            )
            self._register_alert(alert)
            return alert
        
        return None
    
    def check_volume_anomaly(self, asset: str, current_volume: float, 
                           avg_volume: float) -> Optional[Alert]:
        """
        Check for unusual volume
        
        Args:
            asset: Asset identifier
            current_volume: Current trading volume
            avg_volume: Average volume
            
        Returns:
            Alert if anomaly detected
        """
        if avg_volume == 0:
            return None
            
        volume_multiplier = current_volume / avg_volume
        
        thresholds = self.thresholds['volume']
        severity = None
        
        if volume_multiplier >= thresholds['critical_multiplier']:
            severity = AlertSeverity.CRITICAL
        elif volume_multiplier >= thresholds['warning_multiplier']:
            severity = AlertSeverity.WARNING
        
        if severity and self._check_cooldown(f"volume_{asset}"):
            alert = Alert(
                alert_id=self._generate_alert_id(),
                timestamp=datetime.now(),
                alert_type=AlertType.VOLUME_ANOMALY,
                severity=severity,
                title=f"Volume Anomaly: {asset}",
                message=f"{asset} volume {volume_multiplier:.1f}x average",
                asset=asset,
                metric_name="volume",
                current_value=current_volume,
                threshold_value=avg_volume * thresholds[f'{severity.value}_multiplier'],
                metadata={
                    'average_volume': avg_volume,
                    'multiplier': volume_multiplier
                }
            )
            self._register_alert(alert)
            return alert
        
        return None
    
    def check_risk_limit(self, metric_name: str, current_value: float, 
                        limit: float, portfolio: str = "main") -> Optional[Alert]:
        """
        Check if risk metric exceeds limit
        
        Args:
            metric_name: Name of risk metric (VaR, CVaR, etc.)
            current_value: Current metric value
            limit: Risk limit
            portfolio: Portfolio identifier
            
        Returns:
            Alert if limit breached
        """
        breach_pct = (current_value / limit - 1) * 100
        
        if breach_pct > 0:
            severity = AlertSeverity.CRITICAL if breach_pct > 25 else AlertSeverity.WARNING
            
            if self._check_cooldown(f"risk_{portfolio}_{metric_name}"):
                alert = Alert(
                    alert_id=self._generate_alert_id(),
                    timestamp=datetime.now(),
                    alert_type=AlertType.RISK_LIMIT,
                    severity=severity,
                    title=f"Risk Limit Breach: {metric_name}",
                    message=f"{metric_name} at {current_value:.2f}% exceeds limit of {limit:.2f}%",
                    asset=portfolio,
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold_value=limit,
                    metadata={
                        'breach_percentage': breach_pct,
                        'portfolio': portfolio
                    }
                )
                self._register_alert(alert)
                return alert
        
        return None
    
    def check_correlation_break(self, asset1: str, asset2: str, 
                               current_corr: float, historical_corr: float) -> Optional[Alert]:
        """
        Check for correlation breakdown
        
        Args:
            asset1: First asset
            asset2: Second asset
            current_corr: Current correlation
            historical_corr: Historical correlation
            
        Returns:
            Alert if correlation breaks
        """
        corr_change = abs(current_corr - historical_corr)
        threshold = self.thresholds['correlation']['break_threshold']
        
        if corr_change >= threshold and self._check_cooldown(f"corr_{asset1}_{asset2}"):
            severity = AlertSeverity.WARNING
            
            alert = Alert(
                alert_id=self._generate_alert_id(),
                timestamp=datetime.now(),
                alert_type=AlertType.CORRELATION_BREAK,
                severity=severity,
                title=f"Correlation Break: {asset1}/{asset2}",
                message=f"Correlation changed from {historical_corr:.2f} to {current_corr:.2f}",
                asset=f"{asset1}_{asset2}",
                metric_name="correlation",
                current_value=current_corr,
                threshold_value=historical_corr,
                metadata={
                    'asset1': asset1,
                    'asset2': asset2,
                    'change': corr_change
                }
            )
            self._register_alert(alert)
            return alert
        
        return None
    
    def create_custom_alert(self, title: str, message: str, 
                          severity: AlertSeverity, alert_type: AlertType,
                          **kwargs) -> Alert:
        """
        Create a custom alert
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            alert_type: Alert type
            **kwargs: Additional alert fields
            
        Returns:
            Created alert
        """
        alert = Alert(
            alert_id=self._generate_alert_id(),
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            **kwargs
        )
        self._register_alert(alert)
        return alert
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None,
                         alert_type: Optional[AlertType] = None) -> List[Alert]:
        """
        Get active alerts with optional filtering
        
        Args:
            severity: Filter by severity
            alert_type: Filter by type
            
        Returns:
            List of matching active alerts
        """
        alerts = self.active_alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if acknowledged successfully
        """
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert and move to history
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if resolved successfully
        """
        for i, alert in enumerate(self.active_alerts):
            if alert.alert_id == alert_id:
                alert.resolved = True
                self.alert_history.append(alert)
                self.active_alerts.pop(i)
                logger.info(f"Alert {alert_id} resolved")
                return True
        return False
    
    def get_alert_summary(self) -> Dict[str, int]:
        """
        Get summary of active alerts by severity
        
        Returns:
            Dictionary with counts by severity
        """
        summary = {severity.value: 0 for severity in AlertSeverity}
        
        for alert in self.active_alerts:
            summary[alert.severity.value] += 1
        
        return summary
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"ALERT_{timestamp}"
    
    def _check_cooldown(self, key: str) -> bool:
        """
        Check if cooldown period has passed for alert type
        
        Args:
            key: Cooldown key
            
        Returns:
            True if cooldown passed
        """
        cooldown_minutes = self.thresholds.get('cooldown_minutes', 60)
        
        if key in self.cooldown_periods:
            last_alert = self.cooldown_periods[key]
            if datetime.now() - last_alert < timedelta(minutes=cooldown_minutes):
                return False
        
        self.cooldown_periods[key] = datetime.now()
        return True
    
    def _register_alert(self, alert: Alert):
        """Register new alert"""
        self.active_alerts.append(alert)
        logger.warning(f"Alert generated: {alert.title} - {alert.message}")
    
    def clear_old_alerts(self, days: int = 30):
        """
        Clear old alerts from history
        
        Args:
            days: Number of days to retain
        """
        cutoff = datetime.now() - timedelta(days=days)
        self.alert_history = [a for a in self.alert_history if a.timestamp > cutoff]
        logger.info(f"Cleared alerts older than {days} days")


class NotificationService:
    """Service for sending alert notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize notification service
        
        Args:
            config: Notification configuration
        """
        self.config = config
        self.email_config = config.get('email', {})
        self.slack_config = config.get('slack', {})
        
    def send_email_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """
        Send alert via email
        
        Args:
            alert: Alert to send
            recipients: Email recipients
            
        Returns:
            True if sent successfully
        """
        if not self.email_config.get('enabled', False):
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_address']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(self.email_config['smtp_host'], 
                            self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], 
                           self.email_config['password'])
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format alert as HTML email body"""
        color = {
            AlertSeverity.INFO: '#17a2b8',
            AlertSeverity.WARNING: '#ffc107',
            AlertSeverity.CRITICAL: '#dc3545',
            AlertSeverity.EMERGENCY: '#8b0000'
        }[alert.severity]
        
        html = f"""
        <html>
        <body>
            <div style="background-color: {color}; color: white; padding: 10px;">
                <h2>{alert.title}</h2>
            </div>
            <div style="padding: 20px;">
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Type:</strong> {alert.alert_type.value}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Message:</strong> {alert.message}</p>
        """
        
        if alert.asset:
            html += f"<p><strong>Asset:</strong> {alert.asset}</p>"
        if alert.current_value is not None:
            html += f"<p><strong>Current Value:</strong> {alert.current_value:.2f}</p>"
        if alert.threshold_value is not None:
            html += f"<p><strong>Threshold:</strong> {alert.threshold_value:.2f}</p>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html