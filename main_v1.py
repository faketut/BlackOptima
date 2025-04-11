import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime

# Constants
START_DATE = '2023-01-01'
END_DATE = '2023-10-31'
INITIAL_INVESTMENT = 750000  # CAD
MAX_STOCKS = 22
MIN_MONTHLY_VOLUME = 150000
MIN_TRADING_DAYS = 18

class PortfolioOptimizer:
    def __init__(self):
        self.tickers = self._load_and_filter_tickers()
        self.benchmark = '^GSPC'
        self.risk_free_rate = 0.04  # 4% annual risk-free rate
        
    def _load_and_filter_tickers(self):
        """Load and preprocess tickers with enhanced filtering"""
        tickers = pd.read_csv('Tickers_Example.csv', names=['Tickers'])['Tickers'].tolist()
        return list(set(t for t in tickers if t not in yf.shared._ERRORS.keys()))
    
    def _calculate_metrics(self, ticker):
        """Enhanced metric calculation with EWMA volatility"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=START_DATE, end=END_DATE)
            
            # Volume and liquidity checks
            monthly_volume = hist['Volume'].resample('ME').mean()
            if any(monthly_volume < MIN_MONTHLY_VOLUME):
                return None
                
            # Calculate returns with EWMA volatility
            returns = hist['Close'].pct_change().dropna()
            ewma_vol = returns.ewm(span=21).std().iloc[-1]  # 1-month lookback
            
            return {
                'Ticker': ticker,
                'Return': returns.mean(),
                'Volatility': ewma_vol,
                'Currency': stock.fast_info.get('currency', '')
            }
        except Exception:
            return None

    def optimize_portfolio(self):
        """Black-Litterman portfolio optimization"""
        # Calculate metrics for all valid tickers
        metrics = [m for m in (self._calculate_metrics(t) for t in self.tickers) if m]
        df = pd.DataFrame(metrics)
        
        # Calculate covariance matrix with Ledoit-Wolf shrinkage
        returns_matrix = self._get_returns_matrix(df['Ticker'])
        cov_matrix = self._ledoit_wolf_shrinkage(returns_matrix)
        
        # Black-Litterman optimization
        optimal_weights = self._black_litterman_optimization(
            df, # Pass the dataframe containing returns and tickers
            cov_matrix, # Pass the covariance matrix (DataFrame or numpy array)
            risk_aversion=2.5
        )

        # Build final portfolio
        # Ensure optimal_weights has the correct length
        if len(optimal_weights) == len(df):
            df['Weight'] = optimal_weights
        else:
            print(f"Warning: Weight array length ({len(optimal_weights)}) doesn't match DataFrame length ({len(df)}). Assigning NaN.")
            # Handle mismatch - perhaps assign NaN or raise error
            df['Weight'] = np.nan # Or handle as appropriate
        return self._build_final_portfolio(df)

    def _get_returns_matrix(self, tickers):
        """Create returns matrix with forward-filling"""
        data = {}
        for t in tickers:
            hist = yf.Ticker(t).history(start=START_DATE, end=END_DATE)
            data[t] = hist['Close'].pct_change().dropna()
        return pd.DataFrame(data)

    def _ledoit_wolf_shrinkage(self, returns):
        """Improved covariance estimation"""
        emp_cov = returns.cov()
        n = len(returns)
        p = len(returns.columns)
        
        # Shrinkage target (constant correlation)
        var = np.diag(emp_cov)
        std = np.sqrt(var)
        outer_std = np.outer(std, std)
        corr = emp_cov / outer_std
        mean_corr = (corr.values.sum() - p) / (p * (p - 1))
        target = mean_corr * outer_std
        
        # Shrinkage intensity
        delta = np.sum((returns - returns.mean()).values** 4) / (n * p)
        pi_hat = np.sum((emp_cov - target).values**2)
        shrinkage = min(1, max(0, (pi_hat - delta) / pi_hat))
        
        return (1 - shrinkage) * emp_cov + shrinkage * target

    def _black_litterman_optimization(self, df_metrics, cov_matrix_input, risk_aversion=2.5):
        """Black-Litterman model implementation"""
        mean_returns = df_metrics['Return'].values
        tickers = df_metrics['Ticker'].tolist()
        num_assets = len(tickers)

        # Ensure cov_matrix is a numpy array and has the correct shape
        if isinstance(cov_matrix_input, pd.DataFrame):
            # Reindex cov_matrix to match the order of tickers in df_metrics
            cov_matrix = cov_matrix_input.reindex(index=tickers, columns=tickers).values
        elif isinstance(cov_matrix_input, np.ndarray):
            if cov_matrix_input.shape == (num_assets, num_assets):
                cov_matrix = cov_matrix_input
            else:
                raise ValueError(f"Covariance matrix shape mismatch: expected ({num_assets}, {num_assets}), got {cov_matrix_input.shape}")
        else:
            raise TypeError("cov_matrix_input must be a pandas DataFrame or numpy array")


        # Get daily returns matrix for momentum calculation
        # Ensure tickers passed to _get_returns_matrix are only those present in cov_matrix columns
        # Fetch returns for all tickers first
        all_daily_returns = self._get_returns_matrix(self.tickers) # Use all original tickers to fetch data
        valid_tickers = [t for t in tickers if t in all_daily_returns.columns] # Filter based on fetched data and df_metrics

        if len(valid_tickers) != num_assets:
             print(f"Warning: Tickers differ between metrics and daily returns. Using {len(valid_tickers)} consistent tickers.")
             # Filter df_metrics and cov_matrix to only include valid_tickers
             df_metrics = df_metrics[df_metrics['Ticker'].isin(valid_tickers)].reset_index(drop=True)
             mean_returns = df_metrics['Return'].values
             tickers = df_metrics['Ticker'].tolist()
             num_assets = len(tickers)
             cov_matrix = pd.DataFrame(cov_matrix, index=tickers, columns=tickers).reindex(index=tickers, columns=tickers).values # Re-filter numpy cov_matrix

        daily_returns_matrix = all_daily_returns[tickers] # Select columns in the correct order


        # Market equilibrium returns (CAPM)
        tau = 0.05  # Confidence in views (scalar uncertainty of the prior)
        market_weights = np.ones(num_assets) / num_assets
        equilibrium_returns = risk_aversion * cov_matrix.dot(market_weights)

        # Investor views (momentum-based) - Use daily returns matrix
        if len(daily_returns_matrix) >= 60:
            momentum = daily_returns_matrix[-60:].mean() # Series indexed by ticker
        elif len(daily_returns_matrix) > 0:
            print("Warning: Less than 60 days of data for momentum calculation. Using all available data.")
            momentum = daily_returns_matrix.mean() # Series indexed by ticker
        else:
             print("Warning: No daily return data available for momentum. Setting momentum to zero.")
             momentum = pd.Series(0.0, index=tickers) # Series of zeros, indexed like daily_returns_matrix columns

        # Align momentum Series with the order of tickers in df_metrics (should already match)
        momentum_values = momentum.values # Get numpy array in correct order

        # Define views (Q) and confidence in views (Omega)
        # Simple momentum view: positive momentum -> positive view, negative -> negative
        views_Q = np.where(momentum_values > 0, 0.02, -0.02) # Q vector (expected excess returns for views)

        # Confidence based on absolute momentum, ensuring it's 1D and non-zero
        # Confidence represents the uncertainty of the views (diagonal of Omega matrix)
        # Lower confidence value means higher uncertainty in the view
        # Let's define Omega's diagonal as inversely proportional to momentum strength (higher momentum = lower uncertainty)
        confidence_strength = np.abs(momentum_values) * 10
        # Avoid division by zero and ensure a minimum level of uncertainty
        # view_uncertainty_diag holds the diagonal elements of Omega
        view_uncertainty_diag = 1 / np.maximum(confidence_strength, 1e-6) # Smaller value = more certain view (lower variance)

        # Combine equilibrium and views using the Black-Litterman formula
        posterior_returns = self._combine_views(
            equilibrium_returns,
            views_Q, # Q vector
            cov_matrix,
            tau, # Scalar uncertainty of prior
            view_uncertainty_diag # Diagonal elements of Omega matrix (1D array)
        )

        # Mean-variance optimization using posterior returns
        def objective(weights):
            port_return = np.sum(weights * posterior_returns)
            port_vol_sq = weights.T @ cov_matrix @ weights
            # Add small epsilon to prevent sqrt(0) or negative values if cov_matrix is not positive semi-definite
            port_vol = np.sqrt(max(port_vol_sq, 1e-12))
            # Avoid division by zero in Sharpe-like calculation if port_vol is effectively zero
            if port_vol < 1e-9:
                return -port_return # If no volatility, just maximize return (or minimize negative return)
            # Maximize utility function U = R - 0.5 * A * V
            return - (port_return - 0.5 * risk_aversion * port_vol**2)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}) # Weights sum to 1
        bounds = [(0, 0.1) for _ in range(num_assets)] # Max 10% weight per asset
        initial_weights = market_weights # Start optimization from market weights

        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False} # Adjust optimizer options if needed
        )

        if not result.success:
            print(f"Optimization failed: {result.message}. Returning market weights.")
            # Handle failure: return market weights, zero weights, or raise an error
            return market_weights

        # Ensure weights sum to 1 (due to potential numerical inaccuracies)
        final_weights = result.x / np.sum(result.x)
        return final_weights

    def _combine_views(self, equilibrium_returns, views_Q, cov_matrix, tau, view_uncertainty_diag):
        """Combine market equilibrium and investor views using Black-Litterman formula."""
        num_assets = len(equilibrium_returns)
        num_views = len(views_Q)

        # P: Linking matrix (k x n), where k=num_views, n=num_assets
        # Assuming each view corresponds directly to one asset in order
        if num_views != num_assets:
             # This case indicates a potential logic error upstream in view generation
             raise ValueError(f"View vector Q length ({num_views}) mismatch with assets ({num_assets}).")

        P = np.eye(num_assets) # Each view corresponds to one asset

        # Omega: Diagonal covariance matrix of view errors (k x k = n x n)
        # view_uncertainty_diag contains the diagonal elements (variances of view errors)
        if view_uncertainty_diag.ndim != 1 or len(view_uncertainty_diag) != num_views:
            raise ValueError(f"view_uncertainty_diag must be a 1D array of length {num_views}")
        # Ensure no zero uncertainty (infinite confidence) which breaks inversion
        omega_diag_safe = np.maximum(view_uncertainty_diag, 1e-9) # Use a small epsilon
        omega = np.diag(omega_diag_safe)
        try:
            omega_inv = np.linalg.inv(omega)
        except np.linalg.LinAlgError:
             print("Error: Omega matrix is singular. Check view uncertainties.")
             # Handle error: Increase minimum uncertainty
             omega_diag_safe = np.maximum(view_uncertainty_diag, 1e-6) # Try larger epsilon
             omega = np.diag(omega_diag_safe)
             omega_inv = np.linalg.inv(omega) # Retry inversion


        # Prior covariance matrix scaled by tau (uncertainty in prior)
        tau_cov = tau * cov_matrix
        try:
            # Adding regularization for numerical stability
            tau_cov_inv = np.linalg.inv(tau_cov + np.eye(num_assets) * 1e-10)
        except np.linalg.LinAlgError:
            print("Error: Tau * Covariance matrix is singular, even with regularization.")
            # Handle error: maybe return equilibrium returns or raise a specific exception
            raise np.linalg.LinAlgError("Could not invert tau * C matrix.")


        # Black-Litterman Master Formula for Posterior Returns (E[R])
        # E[R] = [(tau * C)^-1 + P^T * Omega^-1 * P]^-1 * [(tau * C)^-1 * Pi + P^T * Omega^-1 * Q]
        # Part 1: Inverse of posterior covariance matrix term
        posterior_cov_inv_term = tau_cov_inv + P.T @ omega_inv @ P
        try:
            # Add regularization for numerical stability
            posterior_cov_term = np.linalg.inv(posterior_cov_inv_term + np.eye(num_assets) * 1e-10)
        except np.linalg.LinAlgError:
             print("Error: Posterior covariance matrix term is singular, even with regularization.")
             # Handle error
             raise np.linalg.LinAlgError("Could not invert posterior covariance matrix term.")


        # Part 2: Weighted sum of prior and views term
        weighted_sum_term = tau_cov_inv @ equilibrium_returns + P.T @ omega_inv @ views_Q

        # Posterior expected returns
        posterior_returns = posterior_cov_term @ weighted_sum_term

        return posterior_returns

    def _build_final_portfolio(self, df):
        """Construct final portfolio with transaction costs"""
        df = df.nlargest(MAX_STOCKS, 'Weight')
        df['Weight'] = df['Weight'] / df['Weight'].sum()  # Normalize
        
        # Calculate shares and values
        for i, row in df.iterrows():
            price = yf.Ticker(row['Ticker']).history(start=START_DATE, end=END_DATE)['Close'].iloc[-1]
            if row['Currency'] == 'USD':
                price *= yf.Ticker('CADUSD=X').history(start=START_DATE, end=END_DATE)['Close'].iloc[-1]
            df.loc[i, 'Price'] = price
            df.loc[i, 'Shares'] = (INITIAL_INVESTMENT * row['Weight']) / price
        
        return df[['Ticker', 'Price', 'Currency', 'Shares', 'Weight']]

# Execute optimization
optimizer = PortfolioOptimizer()
portfolio = optimizer.optimize_portfolio()
portfolio.to_csv('optimized_portfolio.csv', index=False)

# Performance visualization
# Get benchmark returns
benchmark_returns = yf.Ticker('^GSPC').history(start=START_DATE, end=END_DATE)['Close'].pct_change().dropna()

# Calculate portfolio returns
# Get the tickers from the final optimized portfolio
portfolio_tickers = portfolio['Ticker'].tolist()
# Get the daily returns for these specific tickers
portfolio_asset_returns = optimizer._get_returns_matrix(portfolio_tickers)

# Align weights with the returns matrix columns (important if order changed)
portfolio_weights = portfolio.set_index('Ticker').loc[portfolio_asset_returns.columns, 'Weight'].values

# Calculate daily portfolio returns: dot product of daily asset returns and weights
# Ensure returns matrix and weights align in dates/assets
portfolio_returns = portfolio_asset_returns.dot(portfolio_weights)

# Align dates with benchmark for plotting
portfolio_returns = portfolio_returns.reindex(benchmark_returns.index).fillna(0)


plt.figure(figsize=(12, 6))
# Plot cumulative returns
plt.plot(portfolio_returns.cumsum(), label='Optimized Portfolio')
plt.plot(benchmark_returns.cumsum(), label='S&P 500 Benchmark')
plt.title('Optimized Portfolio vs. S&P 500 Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.show()
