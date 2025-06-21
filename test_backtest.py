#!/usr/bin/env python3
"""
Test script to verify the backtesting fix
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

def test_backtest_fix():
    """Test the backtesting functionality"""
    print("🧪 Testing Backtesting Fix")
    print("=" * 40)
    
    # Test parameters
    tickers = ["AAPL", "MSFT", "GOOGL"]
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    initial_capital = 750000
    
    print(f"📊 Test Configuration:")
    print(f"   - Tickers: {', '.join(tickers)}")
    print(f"   - Start Date: {start_date.strftime('%Y-%m-%d')}")
    print(f"   - End Date: {end_date.strftime('%Y-%m-%d')}")
    print(f"   - Initial Capital: ${initial_capital:,.0f}")
    print()
    
    # Test data fetching
    print("📡 Testing Data Fetching...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print("❌ No data returned")
            return False
        
        print(f"✅ Data fetched successfully")
        print(f"   - Data shape: {data.shape}")
        print(f"   - Columns: {list(data.columns)}")
        
        # Check if it's multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            print("   - Multi-level columns detected")
            try:
                prices = data.xs('Adj Close', axis=1, level=1)
                print(f"   - Extracted Adj Close data: {prices.shape}")
            except KeyError:
                try:
                    prices = data.xs('Close', axis=1, level=1)
                    print(f"   - Extracted Close data: {prices.shape}")
                except KeyError:
                    print("❌ Could not extract price data")
                    return False
        else:
            print("   - Single-level columns")
            if 'Adj Close' in data.columns:
                prices = data[['Adj Close']]
                print(f"   - Using Adj Close data: {prices.shape}")
            elif 'Close' in data.columns:
                prices = data[['Close']]
                print(f"   - Using Close data: {prices.shape}")
            else:
                print("❌ No price columns found")
                return False
        
        # Clean data
        prices = prices.dropna()
        print(f"   - Clean data shape: {prices.shape}")
        
        if prices.empty:
            print("❌ No data after cleaning")
            return False
        
        print("✅ Data processing successful")
        print()
        
        # Test backtest simulation
        print("📈 Testing Backtest Simulation...")
        
        # Create sample weights
        weights = np.array([0.4, 0.35, 0.25])
        
        # Monthly rebalancing
        monthly_dates = prices.resample('M').last().index
        print(f"   - Rebalancing dates: {len(monthly_dates)}")
        
        if len(monthly_dates) < 2:
            print("❌ Insufficient data points for backtesting")
            return False
        
        # Simulate backtest
        portfolio_values = [initial_capital]
        
        for i in range(1, len(monthly_dates)):
            prev_date = monthly_dates[i-1]
            curr_date = monthly_dates[i]
            
            # Get period data
            period_data = prices.loc[prev_date:curr_date]
            
            if len(period_data) > 1:
                returns = (period_data.iloc[-1] / period_data.iloc[0]) - 1
                
                # Align weights with available data
                available_tickers = returns.index
                aligned_weights = []
                
                for ticker in available_tickers:
                    if ticker in tickers:
                        ticker_idx = tickers.index(ticker)
                        if ticker_idx < len(weights):
                            aligned_weights.append(weights[ticker_idx])
                        else:
                            aligned_weights.append(0)
                    else:
                        aligned_weights.append(0)
                
                # Normalize weights
                if sum(aligned_weights) > 0:
                    aligned_weights = np.array(aligned_weights) / sum(aligned_weights)
                    portfolio_return = np.dot(aligned_weights, returns.values)
                    new_value = portfolio_values[-1] * (1 + portfolio_return)
                    portfolio_values.append(new_value)
                else:
                    portfolio_values.append(portfolio_values[-1])
            else:
                portfolio_values.append(portfolio_values[-1])
        
        # Calculate results
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_return = (1 + total_return) ** (12 / len(monthly_dates)) - 1
        
        print("✅ Backtest simulation successful")
        print(f"   - Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
        print(f"   - Total Return: {total_return:.2%}")
        print(f"   - Annualized Return: {annual_return:.2%}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_backtest_fix()
    if success:
        print("🎉 Backtesting fix verification successful!")
    else:
        print("⚠️ Backtesting fix verification failed!") 