"""
Alerts Module
Email alert system for dashboard signals
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class AlertSystem:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = os.getenv('SMTP_EMAIL', '')
        self.sender_password = os.getenv('SMTP_PASSWORD', '')
        
    def check_alerts(self, data):
        """Check for alert conditions"""
        alerts = []
        
        # Extract metrics
        rel_perf = data['indices'].get('relative_performance', 0)
        spread = data['macro'].get('fr_de_spread', 68)
        pe_gap = data['valuations'].get('pe_gap', 7.5)
        
        sector_perf = data.get('sectors', {})
        fin_perf = sector_perf.get('Financials', 0)
        tech_perf = sector_perf.get('Technology', 0)
        
        # Alert 1: Strong outperformance
        if rel_perf > 5:
            alerts.append({
                'type': 'BULLISH',
                'priority': 'HIGH',
                'title': '🚀 Europe Strongly Outperforming',
                'message': f"STOXX 600 outperforming S&P 500 by {rel_perf:.2f}% over 3 months.",
                'action': 'Consider taking partial profits or maintaining overweight position.',
                'metric': f"{rel_perf:.2f}%"
            })
        
        # Alert 2: Strong underperformance (buying opportunity)
        if rel_perf < -7:
            alerts.append({
                'type': 'OPPORTUNITY',
                'priority': 'HIGH',
                'title': '💰 Europe at Attractive Entry Levels',
                'message': f"STOXX 600 underperforming by {abs(rel_perf):.2f}%. Valuation gap at {pe_gap:.1f}x.",
                'action': 'Consider adding to Europe overweight position.',
                'metric': f"{rel_perf:.2f}%"
            })
        
        # Alert 3: Fragmentation risk
        if spread > 85:
            alerts.append({
                'type': 'RISK',
                'priority': 'CRITICAL',
                'title': '⚠️ EU Fragmentation Risk Critical',
                'message': f"France-Germany 10Y spread at {spread:.0f}bps (above 85bps danger zone).",
                'action': 'Review European exposure immediately. Consider hedges or position reduction.',
                'metric': f"{spread:.0f}bps"
            })
        elif spread > 80:
            alerts.append({
                'type': 'WARNING',
                'priority': 'MEDIUM',
                'title': '⚠️ Fragmentation Risk Elevated',
                'message': f"France-Germany spread at {spread:.0f}bps (above 80bps threshold).",
                'action': 'Monitor closely. Review hedging strategy.',
                'metric': f"{spread:.0f}bps"
            })
        
        # Alert 4: Sector rotation
        if fin_perf - tech_perf > 7:
            alerts.append({
                'type': 'OPPORTUNITY',
                'priority': 'MEDIUM',
                'title': '🔄 Strong Value Rotation Signal',
                'message': f"Financials outperforming Technology by {fin_perf - tech_perf:.1f}%.",
                'action': 'Rotate portfolio from Growth to Value sectors.',
                'metric': f"+{fin_perf - tech_perf:.1f}%"
            })
        
        # Alert 5: Valuation gap widening
        if pe_gap > 9:
            alerts.append({
                'type': 'OPPORTUNITY',
                'priority': 'LOW',
                'title': '📊 Valuation Gap Widening',
                'message': f"EU-US P/E gap at {pe_gap:.1f}x (above 9x threshold).",
                'action': 'Europe becoming increasingly attractive on valuation.',
                'metric': f"{pe_gap:.1f}x"
            })
        
        return alerts
    
    def send_alert_email(self, alerts, recipient_email):
        """Send alert email"""
        if not alerts:
            print("ℹ️ No alerts to send")
            return False
        
        if not self.sender_email or not self.sender_password:
            print("⚠️ Email credentials not configured")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"📊 JPM EU Thesis Alert: {len(alerts)} Signal(s) - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Create HTML email body
            html_body = self._create_html_email(alerts)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"✅ Alert email sent to {recipient_email}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error sending email: {e}")
            return False
    
    def _create_html_email(self, alerts):
        """Create HTML email body"""
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .header { background-color: #003366; color: white; padding: 20px; text-align: center; }
                .alert { border-left: 4px solid; padding: 15px; margin: 20px 0; }
                .alert-critical { border-color: #dc3545; background-color: #f8d7da; }
                .alert-high { border-color: #ffc107; background-color: #fff3cd; }
                .alert-medium { border-color: #0dcaf0; background-color: #cff4fc; }
                .alert-low { border-color: #198754; background-color: #d1e7dd; }
                .metric { font-size: 24px; font-weight: bold; color: #003366; }
                .action { background-color: #e7f3ff; padding: 10px; margin-top: 10px; border-radius: 5px; }
                .footer { text-align: center; color: #666; margin-top: 30px; padding: 20px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 JPM European Equity Thesis Dashboard</h1>
                <p>Real-Time Alert System</p>
            </div>
        """
        
        # Group alerts by priority
        priority_order = {'CRITICAL': 1, 'HIGH': 2, 'MEDIUM': 3, 'LOW': 4}
        sorted_alerts = sorted(alerts, key=lambda x: priority_order.get(x['priority'], 5))
        
        for alert in sorted_alerts:
            # Determine CSS class based on priority
            css_class = f"alert-{alert['priority'].lower()}"
            
            html += f"""
            <div class="alert {css_class}">
                <h2>{alert['title']}</h2>
                <div class="metric">{alert['metric']}</div>
                <p><strong>Signal:</strong> {alert['message']}</p>
                <div class="action">
                    <strong>💡 Recommended Action:</strong> {alert['action']}
                </div>
            </div>
            """
        
        html += f"""
            <div class="footer">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>This is an automated alert from your JPM EU Thesis Dashboard</p>
                <p style="font-size: 12px; color: #999;">
                    Disclaimer: For informational purposes only. Not investment advice.
                </p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def send_test_alert(self, recipient_email):
        """Send a test alert"""
        test_alerts = [{
            'type': 'INFO',
            'priority': 'LOW',
            'title': '✅ Test Alert - System Operational',
            'message': 'This is a test alert from your JPM EU Thesis Dashboard.',
            'action': 'No action required. Your alert system is working correctly.',
            'metric': '100%'
        }]
        
        return self.send_alert_email(test_alerts, recipient_email)


if __name__ == "__main__":
    # Test alert system
    alert_system = AlertSystem()
    
    test_data = {
        'indices': {'relative_performance': -8.5},
        'macro': {'fr_de_spread': 72},
        'valuations': {'pe_gap': 9.2},
        'sectors': {'Financials': 10.5, 'Technology': 2.1}
    }
    
    alerts = alert_system.check_alerts(test_data)
    print(f"\n✅ Generated {len(alerts)} alerts:")
    for alert in alerts:
        print(f"  • {alert['title']} [{alert['priority']}]")