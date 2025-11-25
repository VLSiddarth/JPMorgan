"""
Signal Generator Module
Generates live trade signals based on thesis metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime

class SignalGenerator:
    def __init__(self):
        self.signals = []
        
        # Define thresholds
        self.thresholds = {
            'relative_perf_strong_buy': -5,
            'relative_perf_buy': 0,
            'relative_perf_take_profit': 5,
            'pe_gap_wide': 9,
            'pe_gap_narrow': 7,
            'spread_danger': 85,
            'spread_warning': 80,
            'spread_safe': 40,
            'credit_impulse_target': 3.0,
            'eps_growth_target': 12.0
        }
    
    def generate_signals(self, data):
        """Generate all trading signals"""
        self.signals = []
        
        # Extract key metrics
        rel_perf = data['indices'].get('relative_performance', 0)
        pe_gap = data['valuations'].get('pe_gap', 7.5)
        spread = data['macro'].get('fr_de_spread', 68)
        credit_impulse = data.get('credit_impulse', 3.2)
        eps_growth = data.get('eps_growth_2026', 12.5)
        
        sector_perf = data.get('sectors', {})
        
        # Signal 1: Relative Value Entry
        if rel_perf < self.thresholds['relative_perf_strong_buy'] and pe_gap > self.thresholds['pe_gap_wide']:
            self.signals.append({
                'type': 'STRONG_BUY',
                'title': '🚀 Strong Buy Signal: Europe Deeply Oversold',
                'message': f"Europe underperforming by {abs(rel_perf):.1f}% with P/E gap at {pe_gap:.1f}x",
                'action': 'Add +5-7% Europe overweight position',
                'conviction': 'HIGH',
                'target_allocation': 'Overweight +7%',
                'timeframe': '3-6 months',
                'timestamp': datetime.now()
            })
        
        elif rel_perf < self.thresholds['relative_perf_buy']:
            self.signals.append({
                'type': 'BUY',
                'title': '💰 Buy Signal: Europe at Attractive Levels',
                'message': f"Europe underperforming by {abs(rel_perf):.1f}%, valuation support strong",
                'action': 'Add +3-5% Europe overweight',
                'conviction': 'MEDIUM',
                'target_allocation': 'Overweight +5%',
                'timeframe': '3-6 months',
                'timestamp': datetime.now()
            })
        
        # Signal 2: Take Profits
        if rel_perf > self.thresholds['relative_perf_take_profit']:
            self.signals.append({
                'type': 'TAKE_PROFIT',
                'title': '✅ Take Profit: Thesis Working',
                'message': f"Europe outperforming by {rel_perf:.1f}%, consider trimming positions",
                'action': 'Reduce overweight by 2-3%, lock in gains',
                'conviction': 'MEDIUM',
                'target_allocation': 'Reduce to +3% overweight',
                'timeframe': 'Immediate',
                'timestamp': datetime.now()
            })
        
        # Signal 3: Sector Rotation
        if sector_perf:
            fin_perf = sector_perf.get('Financials', 0)
            tech_perf = sector_perf.get('Technology', 0)
            
            if fin_perf - tech_perf > 5:
                self.signals.append({
                    'type': 'ROTATE',
                    'title': '🔄 Sector Rotation: Value Outperforming',
                    'message': f"Financials outperforming Tech by {fin_perf - tech_perf:.1f}%",
                    'action': 'Rotate from Growth to Value sectors (Financials, Industrials)',
                    'conviction': 'HIGH',
                    'target_allocation': 'Financials OW +8%, Tech UW -3%',
                    'timeframe': '1-3 months',
                    'timestamp': datetime.now()
                })
        
        # Signal 4: Risk Alert - Fragmentation
        if spread > self.thresholds['spread_danger']:
            self.signals.append({
                'type': 'RISK_ALERT',
                'title': '⚠️ RISK ALERT: EU Fragmentation Critical',
                'message': f"France-Germany spread at {spread:.0f}bps (above {self.thresholds['spread_danger']}bps danger zone)",
                'action': 'Reduce Europe exposure, hedge with periphery shorts',
                'conviction': 'HIGH',
                'target_allocation': 'Cut to neutral, add hedges',
                'timeframe': 'Immediate',
                'timestamp': datetime.now()
            })
        
        elif spread > self.thresholds['spread_warning']:
            self.signals.append({
                'type': 'WARNING',
                'title': '⚠️ Warning: Fragmentation Risk Elevated',
                'message': f"France-Germany spread at {spread:.0f}bps (above {self.thresholds['spread_warning']}bps threshold)",
                'action': 'Monitor closely, review hedges',
                'conviction': 'MEDIUM',
                'target_allocation': 'Maintain position, add protection',
                'timeframe': '1-2 weeks',
                'timestamp': datetime.now()
            })
        
        # Signal 5: Fundamental Tailwinds
        if eps_growth > self.thresholds['eps_growth_target'] and credit_impulse > self.thresholds['credit_impulse_target']:
            self.signals.append({
                'type': 'FUNDAMENTAL_BULLISH',
                'title': '📈 Strong Fundamentals: Thesis Supported',
                'message': f"EPS growth at {eps_growth:.1f}%, Credit impulse at {credit_impulse:.1f}%",
                'action': 'Hold overweight positions, fundamentals supportive',
                'conviction': 'HIGH',
                'target_allocation': 'Maintain overweight',
                'timeframe': '6-12 months',
                'timestamp': datetime.now()
            })
        
        return self.signals
    
    def get_overall_recommendation(self):
        """Get overall portfolio recommendation"""
        if not self.signals:
            return {
                'recommendation': 'HOLD',
                'allocation': 'Neutral',
                'summary': 'Maintain current positions'
            }
        
        # Count signal types
        buy_signals = sum(1 for s in self.signals if s['type'] in ['BUY', 'STRONG_BUY'])
        sell_signals = sum(1 for s in self.signals if s['type'] in ['TAKE_PROFIT', 'RISK_ALERT'])
        
        if buy_signals > sell_signals:
            return {
                'recommendation': 'BUY',
                'allocation': 'Overweight +5-7%',
                'summary': f'{buy_signals} bullish signals detected'
            }
        elif sell_signals > buy_signals:
            return {
                'recommendation': 'REDUCE',
                'allocation': 'Neutral to Underweight',
                'summary': f'{sell_signals} risk/profit-taking signals detected'
            }
        else:
            return {
                'recommendation': 'HOLD',
                'allocation': 'Current allocation',
                'summary': 'Mixed signals, maintain positions'
            }
    
    def format_signals_for_display(self):
        """Format signals for dashboard display"""
        if not self.signals:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.signals)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df[['title', 'message', 'action', 'conviction', 'timestamp']]


if __name__ == "__main__":
    # Test signal generator
    test_data = {
        'indices': {'relative_performance': -6.5},
        'valuations': {'pe_gap': 9.2},
        'macro': {'fr_de_spread': 72},
        'sectors': {'Financials': 8.5, 'Technology': 2.1},
        'credit_impulse': 3.4,
        'eps_growth_2026': 13.2
    }
    
    generator = SignalGenerator()
    signals = generator.generate_signals(test_data)
    
    print(f"\n✅ Generated {len(signals)} signals:")
    for signal in signals:
        print(f"\n{signal['title']}")
        print(f"  Action: {signal['action']}")
        print(f"  Conviction: {signal['conviction']}")