"""
Helper utilities for the dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime

def format_percentage(value, decimals=2):
    """Format value as percentage"""
    return f"{value:.{decimals}f}%"

def format_currency(value, currency='€', decimals=0):
    """Format value as currency"""
    return f"{currency}{value:,.{decimals}f}"

def calculate_cagr(start_value, end_value, years):
    """Calculate Compound Annual Growth Rate"""
    if years <= 0 or start_value <= 0:
        return 0
    return ((end_value / start_value) ** (1/years) - 1) * 100

def get_performance_color(value, reverse=False):
    """Get color based on performance (green for positive, red for negative)"""
    if reverse:
        return 'red' if value > 0 else 'green'
    return 'green' if value > 0 else 'red'

def normalize_to_100(series):
    """Normalize a series to base 100"""
    if len(series) == 0:
        return series
    return (series / series.iloc[0]) * 100

def calculate_rolling_return(prices, window=20):
    """Calculate rolling returns"""
    return prices.pct_change(window) * 100

def get_date_range_string(start, end):
    """Get formatted date range string"""
    return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"

def sanitize_ticker(ticker):
    """Clean and validate ticker symbol"""
    return ticker.strip().upper()

def load_static_data(filename):
    """Load static data files (CSV, JSON)"""
    try:
        if filename.endswith('.csv'):
            return pd.read_csv(f'data/{filename}')
        elif filename.endswith('.json'):
            return pd.read_json(f'data/{filename}')
    except Exception as e:
        print(f"⚠️ Error loading {filename}: {e}")
        return None

class PerformanceTracker:
    """Track and compare performance metrics"""
    def __init__(self):
        self.metrics = {}
    
    def add_metric(self, name, value, target=None):
        """Add a performance metric"""
        self.metrics[name] = {
            'value': value,
            'target': target,
            'timestamp': datetime.now()
        }
    
    def check_targets(self):
        """Check which metrics are meeting targets"""
        results = {}
        for name, metric in self.metrics.items():
            if metric['target'] is not None:
                results[name] = metric['value'] >= metric['target']
        return results
    
    def get_summary(self):
        """Get summary of all metrics"""
        return pd.DataFrame(self.metrics).T