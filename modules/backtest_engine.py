"""
Backtest Engine Module
Backtests the JPM thesis strategy
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class BacktestEngine:
    def __init__(self, start_date='2020-01-01'):
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.results = {}
    
    def fetch_historical_data(self):
        """Fetch historical price data"""
        print("📊 Fetching historical data for backtest...")
        
        # Fetch STOXX 600 and S&P 500
        stoxx = yf.download('^STOXX', start=self.start_date, end=self.end_date, progress=False)
        sp500 = yf.download('^GSPC', start=self.start_date, end=self.end_date, progress=False)
        
        return stoxx, sp500
    
    def calculate_signals(self, stoxx, sp500, lookback=20):
        """
        Calculate trading signals based on relative performance
        Strategy:
        - BUY when Europe underperforms by >10% (20-day rolling)
        - SELL when Europe outperforms by >5%
        - Otherwise HOLD (50% exposure)
        """
        # Calculate relative performance
        stoxx_norm = stoxx['Close'] / stoxx['Close'].iloc[0]
        sp500_norm = sp500['Close'] / sp500['Close'].iloc[0]
        
        rel_perf = (stoxx_norm / sp500_norm - 1) * 100
        rel_perf_rolling = rel_perf.rolling(lookback).mean()
        
        # Generate signals
        signals = pd.Series(0.5, index=stoxx.index)  # Default 50% exposure
        
        signals[rel_perf_rolling < -10] = 1.0  # 100% long when oversold
        signals[rel_perf_rolling > 5] = 0.0    # 0% when overbought (take profit)
        
        return signals, rel_perf_rolling
    
    def calculate_returns(self, stoxx, signals):
        """Calculate strategy returns"""
        # Daily returns of STOXX 600
        daily_returns = stoxx['Close'].pct_change()
        
        # Strategy returns = signal * asset returns
        strategy_returns = signals.shift(1) * daily_returns  # Shift to avoid lookahead bias
        
        # Buy and hold returns
        buyhold_returns = daily_returns
        
        return strategy_returns, buyhold_returns
    
    def calculate_metrics(self, returns):
        """Calculate performance metrics"""
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized return
        years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': volatility
        }
    
    def run_backtest(self):
        """Run full backtest"""
        print("🚀 Running backtest...")
        
        # Fetch data
        stoxx, sp500 = self.fetch_historical_data()
        
        if stoxx.empty or sp500.empty:
            print("⚠️ Error: Could not fetch historical data")
            return None
        
        # Generate signals
        signals, rel_perf = self.calculate_signals(stoxx, sp500)
        
        # Calculate returns
        strategy_returns, buyhold_returns = self.calculate_returns(stoxx, signals)
        
        # Calculate metrics
        strategy_metrics = self.calculate_metrics(strategy_returns.dropna())
        buyhold_metrics = self.calculate_metrics(buyhold_returns.dropna())
        
        # Calculate equity curves
        strategy_equity = (1 + strategy_returns).cumprod()
        buyhold_equity = (1 + buyhold_returns).cumprod()
        
        self.results = {
            'strategy_metrics': strategy_metrics,
            'buyhold_metrics': buyhold_metrics,
            'strategy_equity': strategy_equity,
            'buyhold_equity': buyhold_equity,
            'signals': signals,
            'relative_performance': rel_perf,
            'dates': stoxx.index
        }
        
        print("✅ Backtest complete!")
        return self.results
    
    def generate_summary(self):
        """Generate backtest summary"""
        if not self.results:
            return "No backtest results available"
        
        strat = self.results['strategy_metrics']
        bh = self.results['buyhold_metrics']
        
        summary = f"""
        📊 BACKTEST SUMMARY ({self.start_date} to {self.end_date})
        
        Strategy Performance:
        • Total Return: {strat['total_return']*100:.2f}%
        • Annualized Return: {strat['annualized_return']*100:.2f}%
        • Sharpe Ratio: {strat['sharpe_ratio']:.2f}
        • Max Drawdown: {strat['max_drawdown']*100:.2f}%
        • Win Rate: {strat['win_rate']*100:.1f}%
        • Volatility: {strat['volatility']*100:.2f}%
        
        Buy & Hold (STOXX 600):
        • Total Return: {bh['total_return']*100:.2f}%
        • Annualized Return: {bh['annualized_return']*100:.2f}%
        • Sharpe Ratio: {bh['sharpe_ratio']:.2f}
        • Max Drawdown: {bh['max_drawdown']*100:.2f}%
        
        Alpha: {(strat['total_return'] - bh['total_return'])*100:.2f}%
        """
        
        return summary


if __name__ == "__main__":
    # Test backtest
    engine = BacktestEngine(start_date='2020-01-01')
    results = engine.run_backtest()
    
    if results:
        print(engine.generate_summary())