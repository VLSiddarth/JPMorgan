"""
Factor Analysis Module
Analyzes portfolio factor exposures
"""

import yfinance as yf
import pandas as pd
import numpy as np

class FactorAnalyzer:
    def __init__(self):
        self.factors = ['Value', 'Momentum', 'Quality', 'Size', 'Low Volatility']
    
    def calculate_factor_exposures(self, tickers):
        """Calculate factor exposures for a list of tickers"""
        exposures = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period='1y')
                
                if hist.empty:
                    continue
                
                exposure = self._calculate_single_stock_factors(info, hist)
                exposure['ticker'] = ticker
                exposures.append(exposure)
                
            except Exception as e:
                print(f"⚠️ Error analyzing {ticker}: {e}")
                continue
        
        return pd.DataFrame(exposures)
    
    def _calculate_single_stock_factors(self, info, hist):
        """Calculate factor scores for a single stock"""
        factors = {}
        
        # Value Factor (inverse P/E and P/B)
        pe = info.get('trailingPE', None)
        pb = info.get('priceToBook', None)
        
        if pe and pe > 0:
            factors['Value'] = min(100, (1 / pe) * 100)  # Normalize
        else:
            factors['Value'] = 50
        
        if pb and pb > 0:
            factors['Value'] = (factors['Value'] + min(100, (1 / pb) * 50)) / 2
        
        # Momentum Factor (6-month and 12-month returns)
        if len(hist) >= 126:  # ~6 months
            mom_6m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-126] - 1) * 100
            factors['Momentum'] = min(100, max(0, mom_6m * 2 + 50))  # Scale
        else:
            factors['Momentum'] = 50
        
        # Quality Factor (ROE, Profit Margin)
        roe = info.get('returnOnEquity', None)
        profit_margin = info.get('profitMargins', None)
        
        if roe:
            factors['Quality'] = min(100, max(0, roe * 100 + 50))
        else:
            factors['Quality'] = 50
        
        if profit_margin:
            factors['Quality'] = (factors['Quality'] + min(100, profit_margin * 200 + 50)) / 2
        
        # Size Factor (Market Cap)
        market_cap = info.get('marketCap', None)
        if market_cap:
            # Inverse size score (small = higher score)
            size_score = max(0, 100 - np.log10(market_cap) * 10)
            factors['Size'] = size_score
        else:
            factors['Size'] = 50
        
        # Low Volatility Factor
        if len(hist) >= 30:
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            factors['Low Volatility'] = max(0, 100 - volatility)  # Inverse
        else:
            factors['Low Volatility'] = 50
        
        return factors
    
    def calculate_portfolio_exposure(self, tickers):
        """Calculate average factor exposure for a portfolio"""
        exposures_df = self.calculate_factor_exposures(tickers)
        
        if exposures_df.empty:
            return {factor: 50 for factor in self.factors}
        
        # Calculate mean exposure across portfolio
        portfolio_exposure = {}
        for factor in self.factors:
            portfolio_exposure[factor] = exposures_df[factor].mean()
        
        return portfolio_exposure
    
    def compare_to_benchmark(self, portfolio_tickers, benchmark_tickers):
        """Compare portfolio factor exposure to benchmark"""
        portfolio_exp = self.calculate_portfolio_exposure(portfolio_tickers)
        benchmark_exp = self.calculate_portfolio_exposure(benchmark_tickers)
        
        comparison = {}
        for factor in self.factors:
            comparison[factor] = {
                'portfolio': portfolio_exp.get(factor, 50),
                'benchmark': benchmark_exp.get(factor, 50),
                'difference': portfolio_exp.get(factor, 50) - benchmark_exp.get(factor, 50)
            }
        
        return comparison
    
    def generate_factor_summary(self, portfolio_tickers):
        """Generate summary of portfolio factor exposures"""
        print("📊 Analyzing factor exposures...")
        
        exposures = self.calculate_portfolio_exposure(portfolio_tickers)
        
        summary = "🎯 FACTOR EXPOSURE ANALYSIS\n\n"
        
        for factor, score in exposures.items():
            # Interpret the score
            if score > 70:
                interpretation = "High exposure"
            elif score > 55:
                interpretation = "Above average"
            elif score > 45:
                interpretation = "Neutral"
            elif score > 30:
                interpretation = "Below average"
            else:
                interpretation = "Low exposure"
            
            summary += f"{factor}: {score:.1f}/100 - {interpretation}\n"
        
        return summary


if __name__ == "__main__":
    # Test factor analyzer
    analyzer = FactorAnalyzer()
    
    test_tickers = ['SIE.DE', 'SAP.DE', 'UCG.MI']
    summary = analyzer.generate_factor_summary(test_tickers)
    print(summary)