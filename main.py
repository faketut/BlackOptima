import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy import linalg
import json
import io
import base64
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="BlackOptima Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class BlackLittermanOptimizer:
    def __init__(self, returns, market_caps=None, risk_aversion=2.5):
        self.returns = returns
        self.market_caps = market_caps
        self.risk_aversion = risk_aversion
        self.cov_matrix = self._calculate_covariance_matrix()
        self.equilibrium_returns = self._calculate_equilibrium_returns()
        
    def _calculate_covariance_matrix(self):
        """Calculate Ledoit-Wolf shrunk covariance matrix"""
        sample_cov = self.returns.cov().values
        n, p = self.returns.shape
        
        # Shrinkage target (identity matrix scaled by average variance)
        target = np.eye(p) * np.trace(sample_cov) / p
        
        # Shrinkage intensity (simplified Ledoit-Wolf)
        shrinkage = min(1.0, 0.1 + 0.9 * (p / n))
        
        return (1 - shrinkage) * sample_cov + shrinkage * target
    
    def _calculate_equilibrium_returns(self):
        """Calculate equilibrium returns using CAPM"""
        if self.market_caps is None:
            # Equal weights if no market caps provided
            weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
        else:
            weights = self.market_caps / self.market_caps.sum()
        
        return self.risk_aversion * np.dot(self.cov_matrix, weights)
    
    def optimize(self, views=None, view_confidences=None, constraints=None):
        """Optimize portfolio using Black-Litterman model"""
        mu = self.equilibrium_returns.copy()
        
        if views is not None and view_confidences is not None:
            # Incorporate views
            P = np.eye(len(mu))  # Simplified: each view is about one asset
            Q = np.array(views)
            Omega = np.diag(1.0 / np.array(view_confidences))
            
            # Black-Litterman formula
            tau = 0.025  # Scaling factor
            M1 = linalg.inv(tau * self.cov_matrix)
            M2 = np.dot(P.T, np.dot(linalg.inv(Omega), P))
            M3 = np.dot(linalg.inv(tau * self.cov_matrix), mu)
            M4 = np.dot(P.T, np.dot(linalg.inv(Omega), Q))
            
            mu = np.dot(linalg.inv(M1 + M2), M3 + M4)
        
        # Optimization
        n_assets = len(mu)
        
        def objective(weights):
            portfolio_return = np.dot(weights, mu)
            portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
            return -portfolio_return + 0.5 * self.risk_aversion * portfolio_variance
        
        # Constraints
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if constraints:
            for constraint in constraints:
                constraints_list.append(constraint)
        
        # Bounds
        bounds = [(0, 0.1) for _ in range(n_assets)]  # Max 10% per asset
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
        
        return result.x if result.success else x0

class RiskParityOptimizer:
    def __init__(self, returns):
        self.returns = returns
        self.cov_matrix = returns.cov().values
        
    def optimize(self):
        """Optimize portfolio using Risk Parity approach"""
        n_assets = len(self.returns.columns)
        
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            return np.sum((contrib - contrib.mean()) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.01, 0.1) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(risk_parity_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0

class BacktestEngine:
    def __init__(self, tickers, start_date, end_date, initial_capital=750000):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = self._fetch_data()
        
    def _fetch_data(self):
        """Fetch historical data for backtesting with robust error handling"""
        try:
            # Ensure tickers is a list
            if isinstance(self.tickers, str):
                tickers_list = [self.tickers]
            else:
                tickers_list = self.tickers
            
            # Filter out empty tickers
            valid_tickers = [t.strip().upper() for t in tickers_list if t.strip()]
            
            if not valid_tickers:
                st.error("No valid tickers provided for backtesting")
                return pd.DataFrame()
            
            st.info(f"Fetching backtest data for: {', '.join(valid_tickers)}")
            
            # Download data with multiple attempts
            data = None
            for attempt in range(3):
                try:
                    data = yf.download(
                        valid_tickers, 
                        start=self.start_date, 
                        end=self.end_date,
                        progress=False,
                        threads=False
                    )
                    if data is not None and not data.empty:
                        break
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        st.error(f"Failed to fetch data after 3 attempts: {str(e)}")
                        return pd.DataFrame()
                    time.sleep(1)  # Wait before retry
            
            if data is None or data.empty:
                st.error("No data returned from yfinance for backtesting")
                return pd.DataFrame()
            
            # Handle different data structures
            prices = None
            
            if len(valid_tickers) == 1:
                # Single ticker case
                if 'Adj Close' in data.columns:
                    prices = data[['Adj Close']].copy()
                    prices.columns = valid_tickers
                elif 'Close' in data.columns:
                    prices = data[['Close']].copy()
                    prices.columns = valid_tickers
                else:
                    st.error("Neither 'Adj Close' nor 'Close' columns found in data")
                    return pd.DataFrame()
            else:
                # Multiple tickers case
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-level columns - check the structure
                    # yfinance returns (price_type, ticker) format
                    try:
                        # Try to get Adj Close first
                        prices = data.xs('Adj Close', axis=1, level=0)
                    except KeyError:
                        try:
                            # Fallback to Close
                            prices = data.xs('Close', axis=1, level=0)
                        except KeyError:
                            st.error("Could not extract price data from multi-level columns")
                            return pd.DataFrame()
                else:
                    # Single level columns - this might happen with certain data structures
                    if 'Adj Close' in data.columns:
                        prices = data[['Adj Close']].copy()
                        prices.columns = valid_tickers[:1]  # Take first ticker
                    elif 'Close' in data.columns:
                        prices = data[['Close']].copy()
                        prices.columns = valid_tickers[:1]
                    else:
                        st.error("Price columns not found in data")
                        return pd.DataFrame()
            
            # Clean the data
            prices = prices.dropna()
            
            if prices.empty:
                st.error("No valid price data after cleaning")
                return pd.DataFrame()
            
            # Ensure we have enough data points
            if len(prices) < 10:
                st.warning(f"Limited data points ({len(prices)}) for backtesting")
            
            st.success(f"Successfully fetched {len(prices)} data points for backtesting")
            return prices
            
        except Exception as e:
            st.error(f"Error fetching backtest data: {str(e)}")
            return pd.DataFrame()
    
    def run_backtest(self, weights, rebalancing_freq='M'):
        """Run backtest with given weights"""
        if self.data.empty:
            st.error("No data available for backtesting")
            return None
            
        try:
            # Validate weights
            if len(weights) != len(self.tickers):
                st.error(f"Weight count ({len(weights)}) doesn't match ticker count ({len(self.tickers)})")
                return None
            
            # Resample data based on rebalancing frequency
            if rebalancing_freq == 'M':
                rebalance_dates = self.data.resample('M').last().index
            else:
                rebalance_dates = self.data.resample('Q').last().index
            
            if len(rebalance_dates) < 2:
                st.error("Insufficient data points for backtesting")
                return None
                
            portfolio_values = []
            current_capital = self.initial_capital
            
            for i, date in enumerate(rebalance_dates):
                if i == 0:
                    # Initial allocation
                    portfolio_values.append(current_capital)
                else:
                    # Calculate portfolio value
                    prev_date = rebalance_dates[i-1]
                    
                    # Get returns for the period
                    period_data = self.data.loc[prev_date:date]
                    if len(period_data) > 1:
                        returns = (period_data.iloc[-1] / period_data.iloc[0]) - 1
                        
                        # Align weights with available data
                        available_tickers = returns.index
                        aligned_weights = []
                        
                        for ticker in available_tickers:
                            if ticker in self.tickers:
                                ticker_idx = self.tickers.index(ticker)
                                if ticker_idx < len(weights):
                                    aligned_weights.append(weights[ticker_idx])
                                else:
                                    aligned_weights.append(0)
                            else:
                                aligned_weights.append(0)
                        
                        # Normalize weights to sum to 1
                        if sum(aligned_weights) > 0:
                            aligned_weights = np.array(aligned_weights) / sum(aligned_weights)
                            portfolio_return = np.dot(aligned_weights, returns.values)
                            current_capital = current_capital * (1 + portfolio_return)
                    
                    portfolio_values.append(current_capital)
            
            # Create performance DataFrame
            performance = pd.DataFrame({
                'Date': rebalance_dates,
                'Portfolio_Value': portfolio_values
            })
            performance['Returns'] = performance['Portfolio_Value'].pct_change()
            
            return performance
            
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            return None

def fetch_stock_data(tickers, period='1y', currency='CAD'):
    """Fetch stock data using yfinance with robust error handling"""
    try:
        # Ensure tickers is a list
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Filter out empty tickers
        valid_tickers = [t.strip().upper() for t in tickers if t.strip()]
        
        if not valid_tickers:
            st.error("No valid tickers provided")
            return None, None, None
        
        # Download data with error handling
        st.info(f"Fetching data for: {', '.join(valid_tickers)}")
        
        # Use progress bar for user feedback
        progress_bar = st.progress(0)
        progress_bar.progress(0.2)
        
        # Download with multiple attempts
        data = None
        for attempt in range(3):
            try:
                # Use threads=False to avoid threading issues
                data = yf.download(
                    valid_tickers, 
                    period=period, 
                    progress=False,
                    threads=False,
                    group_by='ticker' if len(valid_tickers) > 1 else None
                )
                if data is not None and not data.empty:
                    break
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise e
                time.sleep(1)  # Wait before retry
        
        progress_bar.progress(0.5)
        
        if data is None or data.empty:
            st.error("No data returned from yfinance")
            return None, None, None
        
        # Handle different data structures
        prices = None
        if len(valid_tickers) == 1:
            # Single ticker case
            if 'Adj Close' in data.columns:
                prices = data[['Adj Close']].copy()
                prices.columns = valid_tickers
            elif isinstance(data.index, pd.MultiIndex):
                # Multi-index case (shouldn't happen with single ticker, but just in case)
                prices = data.xs('Adj Close', axis=1, level=1)
            else:
                # Fallback to Close if Adj Close not available
                if 'Close' in data.columns:
                    prices = data[['Close']].copy()
                    prices.columns = valid_tickers
                else:
                    st.error("Neither 'Adj Close' nor 'Close' columns found")
                    return None, None, None
        else:
            # Multiple tickers case
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-level columns (ticker, price_type)
                try:
                    prices = data.xs('Adj Close', axis=1, level=1)
                except KeyError:
                    try:
                        prices = data.xs('Close', axis=1, level=1)
                    except KeyError:
                        st.error("Could not extract price data from multi-level columns")
                        return None, None, None
            else:
                # Single level columns - this means we have only one ticker despite the list
                if 'Adj Close' in data.columns:
                    prices = data[['Adj Close']].copy()
                    prices.columns = valid_tickers[:1]  # Take first ticker
                elif 'Close' in data.columns:
                    prices = data[['Close']].copy()
                    prices.columns = valid_tickers[:1]
                else:
                    st.error("Price columns not found in data")
                    return None, None, None
        
        progress_bar.progress(0.7)
        
        # Clean the data
        prices = prices.dropna()
        
        if prices.empty:
            st.error("No valid price data after cleaning")
            return None, None, None
        
        # Get current prices and currencies with robust error handling
        current_prices = {}
        currencies = {}
        failed_tickers = []
        
        for ticker in prices.columns:
            try:
                stock = yf.Ticker(ticker)
                
                # Try multiple methods to get current price
                current_price = None
                ticker_currency = currency
                
                # Method 1: Try info
                try:
                    info = stock.info
                    if info and isinstance(info, dict):
                        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                        ticker_currency = info.get('currency', currency)
                except:
                    pass
                
                # Method 2: Try fast_info (newer yfinance feature)
                if current_price is None:
                    try:
                        fast_info = stock.fast_info
                        if hasattr(fast_info, 'last_price'):
                            current_price = fast_info.last_price
                    except:
                        pass
                
                # Method 3: Use last available price from our data
                if current_price is None or pd.isna(current_price):
                    current_price = prices[ticker].iloc[-1]
                
                # Method 4: Try recent history if still no price
                if current_price is None or pd.isna(current_price):
                    recent = stock.history(period='5d')
                    if not recent.empty:
                        current_price = recent['Close'].iloc[-1]
                
                current_prices[ticker] = float(current_price) if current_price is not None else 0.0
                currencies[ticker] = ticker_currency
                
            except Exception as e:
                failed_tickers.append(ticker)
                # Use last available price as fallback
                current_prices[ticker] = float(prices[ticker].iloc[-1]) if not prices[ticker].empty else 0.0
                currencies[ticker] = currency
        
        progress_bar.progress(1.0)
        progress_bar.empty()
        
        if failed_tickers:
            st.warning(f"Could not fetch detailed info for: {', '.join(failed_tickers)}. Using last available prices.")
        
        # Final validation
        valid_current_prices = {k: v for k, v in current_prices.items() if v > 0}
        if not valid_current_prices:
            st.error("No valid current prices obtained")
            return None, None, None
        
        st.success(f"Successfully fetched data for {len(prices.columns)} securities")
        return prices, current_prices, currencies
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.error("Please try the following:")
        st.error("1. Check if ticker symbols are valid (e.g., AAPL, MSFT, GOOGL)")
        st.error("2. Ensure you have internet connection")
        st.error("3. Try reducing the number of tickers")
        st.error("4. Try a different time period")
        return None, None, None

def calculate_portfolio_metrics(returns, benchmark_returns=None):
    """Calculate portfolio performance metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['Total Return'] = (1 + returns).prod() - 1
    metrics['Annualized Return'] = (1 + returns).prod() ** (252 / len(returns)) - 1
    metrics['Volatility'] = returns.std() * np.sqrt(252)
    metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Volatility']
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    metrics['Max Drawdown'] = drawdown.min()
    
    # Benchmark comparison
    if benchmark_returns is not None:
        # Align returns by common dates to avoid length mismatch
        if hasattr(returns, 'index') and hasattr(benchmark_returns, 'index'):
            # Find common dates
            common_dates = returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 0:
                aligned_returns = returns.loc[common_dates]
                aligned_benchmark = benchmark_returns.loc[common_dates]
                
                # Ensure both are the same length
                min_length = min(len(aligned_returns), len(aligned_benchmark))
                if min_length > 0:
                    aligned_returns = aligned_returns.iloc[:min_length]
                    aligned_benchmark = aligned_benchmark.iloc[:min_length]
                    
                    benchmark_cumulative = (1 + aligned_benchmark).prod() - 1
                    metrics['Alpha'] = metrics['Total Return'] - benchmark_cumulative
                    
                    # Beta calculation with aligned data
                    try:
                        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                        benchmark_variance = np.var(aligned_benchmark)
                        metrics['Beta'] = covariance / benchmark_variance if benchmark_variance != 0 else 0
                    except Exception as e:
                        st.warning(f"Could not calculate Beta due to data alignment issues: {str(e)}")
                        metrics['Beta'] = 0
                else:
                    metrics['Alpha'] = 0
                    metrics['Beta'] = 0
            else:
                metrics['Alpha'] = 0
                metrics['Beta'] = 0
        else:
            # Fallback for non-indexed data
            try:
                min_length = min(len(returns), len(benchmark_returns))
                if min_length > 0:
                    aligned_returns = returns[:min_length]
                    aligned_benchmark = benchmark_returns[:min_length]
                    
                    benchmark_cumulative = (1 + aligned_benchmark).prod() - 1
                    metrics['Alpha'] = metrics['Total Return'] - benchmark_cumulative
                    
                    covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                    benchmark_variance = np.var(aligned_benchmark)
                    metrics['Beta'] = covariance / benchmark_variance if benchmark_variance != 0 else 0
                else:
                    metrics['Alpha'] = 0
                    metrics['Beta'] = 0
            except Exception as e:
                st.warning(f"Could not calculate Beta: {str(e)}")
                metrics['Alpha'] = 0
                metrics['Beta'] = 0
    
    return metrics

def generate_ai_views(text_input, tickers):
    """Simulate AI-generated views (placeholder for actual LLM integration)"""
    # This is a simplified simulation - in production, you'd integrate with GPT/Claude API
    views = []
    
    # Simple keyword-based view generation
    positive_keywords = ['bullish', 'growth', 'positive', 'buy', 'strong', 'outperform']
    negative_keywords = ['bearish', 'decline', 'negative', 'sell', 'weak', 'underperform']
    
    text_lower = text_input.lower()
    
    for ticker in tickers:
        if ticker.lower() in text_lower:
            confidence = 0.5  # Default confidence
            expected_return = 0.0
            
            # Check for positive sentiment
            if any(keyword in text_lower for keyword in positive_keywords):
                expected_return = np.random.uniform(0.01, 0.05)
                confidence = 0.7
            
            # Check for negative sentiment
            elif any(keyword in text_lower for keyword in negative_keywords):
                expected_return = np.random.uniform(-0.05, -0.01)
                confidence = 0.6
            
            if expected_return != 0:
                views.append({
                    'ticker': ticker,
                    'expected_excess_return': expected_return,
                    'confidence': confidence
                })
    
    return views

def main():
    # Main header
    st.markdown('<div class="main-header">📈 BlackOptima Pro</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Intelligent Portfolio Optimization Platform</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Data Input Section
    st.sidebar.subheader("📊 Data Input")
    
    # File upload or manual input
    input_method = st.sidebar.radio("Input Method", ["Manual Input", "Upload CSV"])
    
    tickers = []
    if input_method == "Manual Input":
        ticker_input = st.sidebar.text_area("Enter Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,TSLA")
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV with tickers", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'ticker' in df.columns:
                tickers = df['ticker'].tolist()
            elif 'Ticker' in df.columns:
                tickers = df['Ticker'].tolist()
            else:
                st.error("CSV must contain a 'ticker' or 'Ticker' column")
    
    # Investment parameters
    st.sidebar.subheader("💰 Investment Parameters")
    budget = st.sidebar.number_input("Investment Budget", value=750000, min_value=1000, step=1000)
    currency = st.sidebar.selectbox("Currency", ["CAD", "USD", "Auto-detect"])
    risk_aversion = st.sidebar.slider("Risk Aversion Coefficient", 0.1, 10.0, 2.5, 0.1)
    
    # Analysis period
    st.sidebar.subheader("📅 Analysis Period")
    period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
    period = st.sidebar.selectbox("Analysis Period", period_options, index=3)
    
    # Strategy selection
    st.sidebar.subheader("🎯 Optimization Strategy")
    strategy = st.sidebar.selectbox("Strategy", ["Black-Litterman", "Risk Parity"])
    
    # ESG Constraints
    st.sidebar.subheader("🌱 ESG Constraints")
    esg_enabled = st.sidebar.checkbox("Enable ESG Constraints")
    esg_min_allocation = 0
    if esg_enabled:
        esg_min_allocation = st.sidebar.slider("Minimum ESG Allocation (%)", 0, 50, 15) / 100
    
    # AI Views Section
    st.sidebar.subheader("🤖 AI-Generated Views")
    ai_views_enabled = st.sidebar.checkbox("Enable AI Views")
    ai_views = []
    if ai_views_enabled:
        view_text = st.sidebar.text_area("Market Commentary/News", 
                                       "Microsoft showing strong growth in cloud services. Tesla facing production challenges.")
        if st.sidebar.button("Generate Views"):
            ai_views = generate_ai_views(view_text, tickers)
            st.sidebar.success(f"Generated {len(ai_views)} views")
    
    # Main content area
    if not tickers:
        st.info("👆 Please configure your portfolio in the sidebar to get started.")
        return
    
    # Limit to 22 stocks as per requirements
    if len(tickers) > 22:
        st.warning("Portfolio limited to 22 stocks as per optimization constraints.")
        tickers = tickers[:22]
    
    # Fetch data
    with st.spinner("📡 Fetching market data..."):
        prices, current_prices, currencies = fetch_stock_data(tickers, period, currency)
    
    if prices is None:
        st.error("Failed to fetch market data. Please check your tickers and try again.")
        return
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Optimization", "📈 Backtesting", "📋 Analysis", "💾 Export"])
    
    with tab1:
        st.markdown('<div class="sub-header">Portfolio Optimization Results</div>', unsafe_allow_html=True)
        
        # Run optimization
        if strategy == "Black-Litterman":
            optimizer = BlackLittermanOptimizer(returns, risk_aversion=risk_aversion)
            
            # Prepare views for optimization
            views_array = None
            confidences_array = None
            if ai_views:
                views_dict = {view['ticker']: view['expected_excess_return'] for view in ai_views}
                confidences_dict = {view['ticker']: view['confidence'] for view in ai_views}
                views_array = [views_dict.get(ticker, 0) for ticker in tickers]
                confidences_array = [confidences_dict.get(ticker, 0.1) for ticker in tickers]
            
            weights = optimizer.optimize(views_array, confidences_array)
        else:
            optimizer = RiskParityOptimizer(returns)
            weights = optimizer.optimize()
        
        # Create portfolio summary
        portfolio_data = []
        for i, ticker in enumerate(tickers):
            if weights[i] > 0.001:  # Only show meaningful allocations
                shares = int((budget * weights[i]) / current_prices[ticker])
                value = shares * current_prices[ticker]
                portfolio_data.append({
                    'Ticker': ticker,
                    'Weight (%)': f"{weights[i]*100:.2f}%",
                    'Shares': shares,
                    'Price': f"${current_prices[ticker]:.2f}",
                    'Value': f"${value:,.2f}",
                    'Currency': currencies[ticker]
                })
        
        # Display portfolio table
        portfolio_df = pd.DataFrame(portfolio_data)
        st.dataframe(portfolio_df, use_container_width=True)
        
        # Portfolio visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(portfolio_df, values=[float(w.strip('%')) for w in portfolio_df['Weight (%)']],
                           names='Ticker', title='Portfolio Allocation')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(portfolio_df, x='Ticker', y=[float(w.strip('%')) for w in portfolio_df['Weight (%)']],
                           title='Portfolio Weights')
            fig_bar.update_layout(yaxis_title='Weight (%)')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Store optimized weights in session state
        st.session_state['optimized_weights'] = weights
        st.session_state['portfolio_tickers'] = tickers
        st.session_state['portfolio_budget'] = budget
    
    with tab2:
        st.markdown('<div class="sub-header">Backtesting Results</div>', unsafe_allow_html=True)
        
        if 'optimized_weights' in st.session_state:
            # Backtest parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                backtest_period = st.selectbox("Backtest Period", ["6mo", "1y", "2y", "3y"], index=1)
            with col2:
                rebalancing = st.selectbox("Rebalancing", ["Monthly", "Quarterly"], index=0)
            with col3:
                benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM"], index=0)
            
            if st.button("🚀 Run Backtest"):
                with st.spinner("Running backtest..."):
                    # Calculate backtest dates
                    end_date = datetime.now()
                    if backtest_period == "6mo":
                        start_date = end_date - timedelta(days=180)
                    elif backtest_period == "1y":
                        start_date = end_date - timedelta(days=365)
                    elif backtest_period == "2y":
                        start_date = end_date - timedelta(days=730)
                    else:
                        start_date = end_date - timedelta(days=1095)
                    
                    # Run backtest
                    backtest_engine = BacktestEngine(
                        st.session_state['portfolio_tickers'],
                        start_date,
                        end_date,
                        st.session_state['portfolio_budget']
                    )
                    
                    rebal_freq = 'M' if rebalancing == 'Monthly' else 'Q'
                    performance = backtest_engine.run_backtest(
                        st.session_state['optimized_weights'],
                        rebal_freq
                    )
                    
                    if performance is not None and not performance.empty:
                        # Fetch benchmark data with error handling
                        try:
                            benchmark_data = yf.download(
                                benchmark, 
                                start=start_date, 
                                end=end_date,
                                progress=False
                            )
                            
                            if not benchmark_data.empty:
                                if 'Adj Close' in benchmark_data.columns:
                                    benchmark_prices = benchmark_data['Adj Close']
                                elif 'Close' in benchmark_data.columns:
                                    benchmark_prices = benchmark_data['Close']
                                else:
                                    benchmark_prices = benchmark_data.iloc[:, 0]  # Take first column
                                
                                benchmark_returns = benchmark_prices.pct_change().dropna()
                            else:
                                st.warning(f"Could not fetch {benchmark} data. Continuing without benchmark comparison.")
                                benchmark_returns = None
                        except Exception as e:
                            st.warning(f"Error fetching benchmark data: {str(e)}. Continuing without benchmark comparison.")
                            benchmark_returns = None
                            benchmark_prices = None
                        
                        # Calculate metrics
                        portfolio_returns = performance['Returns'].dropna()
                        
                        if len(portfolio_returns) > 0:
                            metrics = calculate_portfolio_metrics(portfolio_returns, benchmark_returns)
                            
                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Return", f"{metrics['Total Return']:.2%}")
                            with col2:
                                st.metric("Annualized Return", f"{metrics['Annualized Return']:.2%}")
                            with col3:
                                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                            with col4:
                                st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                            
                            # Performance chart
                            fig = go.Figure()
                            
                            # Portfolio performance
                            fig.add_trace(go.Scatter(
                                x=performance['Date'],
                                y=performance['Portfolio_Value'],
                                mode='lines',
                                name='Portfolio',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Benchmark performance (if available)
                            if 'benchmark_prices' in locals() and benchmark_prices is not None:
                                try:
                                    benchmark_normalized = (benchmark_prices / benchmark_prices.iloc[0]) * st.session_state['portfolio_budget']
                                    fig.add_trace(go.Scatter(
                                        x=benchmark_normalized.index,
                                        y=benchmark_normalized.values,
                                        mode='lines',
                                        name=f'{benchmark} Benchmark',
                                        line=dict(color='red', width=2, dash='dash')
                                    ))
                                except:
                                    pass
                            
                            fig.update_layout(
                                title='Portfolio vs Benchmark Performance',
                                xaxis_title='Date',
                                yaxis_title='Portfolio Value ($)',
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Store backtest results
                            st.session_state['backtest_results'] = {
                                'performance': performance,
                                'metrics': metrics,
                                'benchmark_data': benchmark_prices if 'benchmark_prices' in locals() else None
                            }
                        else:
                            st.error("No valid portfolio returns calculated for the selected period.")
                    else:
                        st.error("Failed to run backtest. This could be due to:")
                        st.error("• Insufficient historical data for the selected period")
                        st.error("• Invalid ticker symbols")
                        st.error("• Network connectivity issues")
                        st.error("Please try a different time period or check your tickers.")
        else:
            st.info("Please run portfolio optimization first to enable backtesting.")
    
    with tab3:
        st.markdown('<div class="sub-header">Portfolio Analysis</div>', unsafe_allow_html=True)
        
        # Risk analysis
        if len(returns) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Risk Metrics")
                
                # Calculate portfolio risk metrics
                if 'optimized_weights' in st.session_state:
                    weights = st.session_state['optimized_weights']
                    portfolio_return = np.dot(weights, returns.mean()) * 252
                    portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns.cov() * 252, weights)))
                    
                    st.metric("Expected Annual Return", f"{portfolio_return:.2%}")
                    st.metric("Expected Annual Volatility", f"{portfolio_vol:.2%}")
                    st.metric("Risk-Return Ratio", f"{portfolio_return/portfolio_vol:.2f}")
                
                # Individual stock volatilities
                stock_vols = returns.std() * np.sqrt(252)
                st.subheader("Individual Stock Volatilities")
                vol_df = pd.DataFrame({
                    'Ticker': stock_vols.index,
                    'Annual Volatility': [f"{vol:.2%}" for vol in stock_vols.values]
                })
                st.dataframe(vol_df, use_container_width=True)
            
            with col2:
                st.subheader("📈 Correlation Matrix")
                
                # Correlation heatmap
                corr_matrix = returns.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, ax=ax, fmt='.2f')
                ax.set_title('Asset Correlation Matrix')
                st.pyplot(fig)
                plt.close()
        
        # AI Views summary
        if ai_views:
            st.subheader("🤖 AI-Generated Views Summary")
            views_df = pd.DataFrame(ai_views)
            views_df['Expected Return'] = views_df['expected_excess_return'].apply(lambda x: f"{x:.2%}")
            views_df['Confidence'] = views_df['confidence'].apply(lambda x: f"{x:.1%}")
            st.dataframe(views_df[['ticker', 'Expected Return', 'Confidence']], use_container_width=True)
    
    with tab4:
        st.markdown('<div class="sub-header">Export & Save Options</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Export Portfolio")
            
            if 'optimized_weights' in st.session_state and len(portfolio_data) > 0:
                # CSV export
                csv_buffer = io.StringIO()
                portfolio_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Portfolio CSV",
                    data=csv_data,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Configuration export
                config = {
                    "tickers": tickers,
                    "budget": budget,
                    "currency": currency,
                    "risk_aversion": risk_aversion,
                    "strategy": strategy,
                    "period": period,
                    "weights": st.session_state['optimized_weights'].tolist(),
                    "timestamp": datetime.now().isoformat()
                }
                
                config_json = json.dumps(config, indent=2)
                st.download_button(
                    label="⚙️ Download Configuration JSON",
                    data=config_json,
                    file_name=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("Run optimization first to enable portfolio export.")
        
        with col2:
            st.subheader("📁 Load Configuration")
            
            uploaded_config = st.file_uploader("Upload Configuration JSON", type=['json'])
            if uploaded_config:
                try:
                    config = json.load(uploaded_config)
                    st.success("Configuration loaded successfully!")
                    st.json(config)
                    
                    if st.button("Apply Configuration"):
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading configuration: {str(e)}")
        
        # Backtest results export
        if 'backtest_results' in st.session_state:
            st.subheader("📈 Export Backtest Results")
            
            backtest_data = st.session_state['backtest_results']['performance']
            backtest_csv = backtest_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Backtest Results CSV",
                data=backtest_csv,
                file_name=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()