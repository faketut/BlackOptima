#!/usr/bin/env python3
"""
Demo script for BlackOptima Pro
This script demonstrates the key features without running the full Streamlit app
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def demo_portfolio_optimization():
    """Demonstrate portfolio optimization with sample data"""
    print("🎯 BlackOptima Pro - Portfolio Optimization Demo")
    print("=" * 50)
    
    # Sample portfolio
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B"]
    budget = 750000
    
    print(f"📊 Portfolio Configuration:")
    print(f"   - Tickers: {', '.join(tickers)}")
    print(f"   - Budget: ${budget:,.0f}")
    print(f"   - Strategy: Black-Litterman Optimization")
    print()
    
    # Fetch real data
    print("📡 Fetching market data...")
    try:
        data = yf.download(tickers, period="1y")['Adj Close']
        returns = data.pct_change().dropna()
        
        print(f"✅ Data fetched successfully")
        print(f"   - Data period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   - Number of observations: {len(returns)}")
        print()
        
        # Calculate basic statistics
        annual_returns = returns.mean() * 252
        annual_vols = returns.std() * np.sqrt(252)
        
        print("📈 Asset Statistics:")
        stats_df = pd.DataFrame({
            'Annual Return': annual_returns.round(4),
            'Annual Volatility': annual_vols.round(4),
            'Sharpe Ratio': (annual_returns / annual_vols).round(3)
        })
        print(stats_df.to_string())
        print()
        
        # Simulate optimization (using equal weights for demo)
        print("⚙️ Running Portfolio Optimization...")
        weights = np.ones(len(tickers)) / len(tickers)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, annual_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_vol
        
        print("📊 Portfolio Results:")
        print(f"   - Expected Annual Return: {portfolio_return:.2%}")
        print(f"   - Expected Annual Volatility: {portfolio_vol:.2%}")
        print(f"   - Sharpe Ratio: {sharpe_ratio:.3f}")
        print()
        
        # Create allocation table
        print("💼 Portfolio Allocation:")
        allocation_data = []
        for i, ticker in enumerate(tickers):
            shares = int((budget * weights[i]) / data[ticker].iloc[-1])
            value = shares * data[ticker].iloc[-1]
            allocation_data.append({
                'Ticker': ticker,
                'Weight': f"{weights[i]:.1%}",
                'Shares': shares,
                'Price': f"${data[ticker].iloc[-1]:.2f}",
                'Value': f"${value:,.0f}"
            })
        
        allocation_df = pd.DataFrame(allocation_data)
        print(allocation_df.to_string(index=False))
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def demo_risk_analysis():
    """Demonstrate risk analysis features"""
    print("🔍 Risk Analysis Demo")
    print("=" * 30)
    
    # Create sample correlation matrix
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(252, 5) * 0.02,
        columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    )
    
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    print("📊 Correlation Matrix:")
    print(corr_matrix.round(3).to_string())
    print()
    
    # Calculate risk metrics
    weights = np.ones(5) / 5
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns.cov() * 252, weights)))
    
    print("📈 Risk Metrics:")
    print(f"   - Portfolio Volatility: {portfolio_vol:.2%}")
    print(f"   - Individual Asset Volatilities:")
    for i, ticker in enumerate(returns.columns):
        vol = returns[ticker].std() * np.sqrt(252)
        print(f"     {ticker}: {vol:.2%}")
    print()
    
    return True

def demo_backtesting():
    """Demonstrate backtesting functionality"""
    print("📈 Backtesting Demo")
    print("=" * 25)
    
    # Create sample price data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price movements
    prices = pd.DataFrame({
        'AAPL': 150 + np.cumsum(np.random.randn(len(dates)) * 0.8),
        'MSFT': 300 + np.cumsum(np.random.randn(len(dates)) * 1.2),
        'GOOGL': 2500 + np.cumsum(np.random.randn(len(dates)) * 15)
    }, index=dates)
    
    # Simulate backtest
    initial_capital = 750000
    weights = np.array([0.4, 0.35, 0.25])
    
    # Monthly rebalancing
    monthly_dates = prices.resample('M').last().index
    portfolio_values = [initial_capital]
    
    for i in range(1, len(monthly_dates)):
        returns = (prices.loc[monthly_dates[i]] / prices.loc[monthly_dates[i-1]]) - 1
        new_value = portfolio_values[-1] * (1 + np.dot(weights, returns))
        portfolio_values.append(new_value)
    
    # Calculate performance metrics
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    annual_return = (1 + total_return) ** (12 / len(monthly_dates)) - 1
    
    print("📊 Backtest Results:")
    print(f"   - Initial Capital: ${initial_capital:,.0f}")
    print(f"   - Final Value: ${portfolio_values[-1]:,.0f}")
    print(f"   - Total Return: {total_return:.2%}")
    print(f"   - Annualized Return: {annual_return:.2%}")
    print(f"   - Number of Rebalancing: {len(monthly_dates)}")
    print()
    
    return True

def demo_ai_views():
    """Demonstrate AI views generation"""
    print("🤖 AI-Generated Views Demo")
    print("=" * 30)
    
    # Sample market commentary
    commentary = """
    Apple continues to show strong growth in services and wearables. 
    Microsoft's cloud business is expanding rapidly with Azure growth. 
    Google faces regulatory challenges but maintains strong search dominance.
    Amazon's e-commerce business is recovering while AWS remains strong.
    Tesla's production challenges are being resolved with new factories.
    """
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    print("📰 Market Commentary:")
    print(commentary.strip())
    print()
    
    # Simulate AI views
    views = [
        {'ticker': 'AAPL', 'sentiment': 'Positive', 'expected_return': 0.08, 'confidence': 0.75},
        {'ticker': 'MSFT', 'sentiment': 'Positive', 'expected_return': 0.12, 'confidence': 0.80},
        {'ticker': 'GOOGL', 'sentiment': 'Neutral', 'expected_return': 0.05, 'confidence': 0.60},
        {'ticker': 'AMZN', 'sentiment': 'Positive', 'expected_return': 0.10, 'confidence': 0.70},
        {'ticker': 'TSLA', 'sentiment': 'Positive', 'expected_return': 0.15, 'confidence': 0.65}
    ]
    
    print("🎯 AI-Generated Views:")
    for view in views:
        print(f"   {view['ticker']}: {view['sentiment']} ({view['expected_return']:.1%} expected return, {view['confidence']:.0%} confidence)")
    print()
    
    return True

def main():
    """Run the complete demo"""
    print("🚀 BlackOptima Pro - Complete Demo")
    print("=" * 50)
    print()
    
    demos = [
        ("Portfolio Optimization", demo_portfolio_optimization),
        ("Risk Analysis", demo_risk_analysis),
        ("Backtesting", demo_backtesting),
        ("AI Views", demo_ai_views)
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
            print("-" * 50)
            print()
        except Exception as e:
            print(f"❌ {name} demo failed: {str(e)}")
            print()
    
    print("🎉 Demo completed successfully!")
    print("\n💡 To run the full interactive application:")
    print("   streamlit run main.py")
    print("   streamlit run dashboard.py")

if __name__ == "__main__":
    main() 