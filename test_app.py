#!/usr/bin/env python3
"""
Test script for BlackOptima Pro application
This script tests the core functionality without running the full Streamlit app
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_data_fetching():
    """Test data fetching functionality"""
    print("🧪 Testing data fetching...")
    
    # Test tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    try:
        # Fetch data
        data = yf.download(test_tickers, period="1mo")
        if not data.empty:
            print("✅ Data fetching successful")
            print(f"   - Fetched data for {len(test_tickers)} tickers")
            print(f"   - Data shape: {data['Adj Close'].shape}")
            return True
        else:
            print("❌ Data fetching failed - empty data")
            return False
    except Exception as e:
        print(f"❌ Data fetching failed: {str(e)}")
        return False

def test_returns_calculation():
    """Test returns calculation"""
    print("\n🧪 Testing returns calculation...")
    
    try:
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        # Generate random price data
        prices = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'MSFT': 200 + np.cumsum(np.random.randn(len(dates)) * 0.3),
            'GOOGL': 150 + np.cumsum(np.random.randn(len(dates)) * 0.4)
        }, index=dates)
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        print("✅ Returns calculation successful")
        print(f"   - Returns shape: {returns.shape}")
        print(f"   - Mean returns: {returns.mean().round(4).to_dict()}")
        return True
    except Exception as e:
        print(f"❌ Returns calculation failed: {str(e)}")
        return False

def test_optimization_classes():
    """Test optimization classes"""
    print("\n🧪 Testing optimization classes...")
    
    try:
        # Create sample returns data
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, 5) * 0.02,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        )
        
        # Test BlackLittermanOptimizer
        from main import BlackLittermanOptimizer
        bl_optimizer = BlackLittermanOptimizer(returns, risk_aversion=2.5)
        weights = bl_optimizer.optimize()
        
        print("✅ BlackLittermanOptimizer test successful")
        print(f"   - Weights sum: {weights.sum():.4f}")
        print(f"   - Max weight: {weights.max():.4f}")
        
        # Test RiskParityOptimizer
        from main import RiskParityOptimizer
        rp_optimizer = RiskParityOptimizer(returns)
        weights_rp = rp_optimizer.optimize()
        
        print("✅ RiskParityOptimizer test successful")
        print(f"   - Weights sum: {weights_rp.sum():.4f}")
        print(f"   - Max weight: {weights_rp.max():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Optimization classes test failed: {str(e)}")
        return False

def test_metrics_calculation():
    """Test portfolio metrics calculation"""
    print("\n🧪 Testing metrics calculation...")
    
    try:
        # Create sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.02)
        
        from main import calculate_portfolio_metrics
        metrics = calculate_portfolio_metrics(returns)
        
        print("✅ Metrics calculation successful")
        print(f"   - Total Return: {metrics['Total Return']:.4f}")
        print(f"   - Annualized Return: {metrics['Annualized Return']:.4f}")
        print(f"   - Volatility: {metrics['Volatility']:.4f}")
        print(f"   - Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Metrics calculation failed: {str(e)}")
        return False

def test_ai_views():
    """Test AI views generation"""
    print("\n🧪 Testing AI views generation...")
    
    try:
        from main import generate_ai_views
        
        tickers = ["AAPL", "MSFT", "GOOGL"]
        text_input = "Apple showing strong growth. Microsoft facing challenges. Google performing well."
        
        views = generate_ai_views(text_input, tickers)
        
        print("✅ AI views generation successful")
        print(f"   - Generated {len(views)} views")
        if views:
            print(f"   - Sample view: {views[0]}")
        
        return True
    except Exception as e:
        print(f"❌ AI views generation failed: {str(e)}")
        return False

def test_backtest_engine():
    """Test backtest engine"""
    print("\n🧪 Testing backtest engine...")
    
    try:
        from main import BacktestEngine
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        prices = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'MSFT': 200 + np.cumsum(np.random.randn(len(dates)) * 0.3)
        }, index=dates)
        
        # Mock the _fetch_data method
        class MockBacktestEngine(BacktestEngine):
            def _fetch_data(self):
                return prices
        
        # Test backtest
        engine = MockBacktestEngine(['AAPL', 'MSFT'], datetime(2024, 1, 1), datetime(2024, 12, 31))
        weights = np.array([0.6, 0.4])
        
        performance = engine.run_backtest(weights, 'M')
        
        print("✅ Backtest engine test successful")
        print(f"   - Performance data shape: {performance.shape}")
        print(f"   - Final portfolio value: ${performance['Portfolio_Value'].iloc[-1]:,.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Backtest engine test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting BlackOptima Pro Tests\n")
    
    tests = [
        test_data_fetching,
        test_returns_calculation,
        test_optimization_classes,
        test_metrics_calculation,
        test_ai_views,
        test_backtest_engine
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {str(e)}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 