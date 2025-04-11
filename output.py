from IPython.display import display, Math, Latex

import pandas as pd
import numpy as np
import numpy_financial as npf
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Date
start_date='2023-01-01'
end_date='2023-10-31'

# Read the CSV file
tickers_sample = pd.read_csv('Tickers_Example.csv', names=['Tickers'])
tickers_sample.head()
# Convert to list
temp = tickers_sample['Tickers'].to_numpy().tolist()

# Remove invalid or duplicated tickers
invalid_data = list(yf.shared._ERRORS.keys())
temp = [elt for elt in temp if elt not in invalid_data]
temp = list(set(temp))

# Create dataframe to store valid ticker information 
df_valid_tickers = pd.DataFrame(columns=['Ticker', 'Currency', 'ClosingPrice'])




# Filter ticker
for ticker in temp:
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        
        # Criteria: Check for valid currency (USD or CAD)
        # Check if 'currency' information exists for the stock
        currency = info.get('currency')
        if not currency or currency not in ['USD', 'CAD']:
            continue  # Skip tickers with missing or invalid currency
        
        # Fetch historical data
        hist = stock.history(start=start_date, end=end_date)
        
        # Criteria: Check for average monthly volume
        monthly_volume = hist['Volume'].resample('M').mean()
        if any(monthly_volume < 150000):
            continue  # Skip tickers with average monthly volume < 150000
        
        # Criteria: Check for trading days
        trading_days = hist.resample('M').size()
        if any(trading_days < 18):
            continue  # Skip tickers with < 18 trading days a month
        
        # get the monthly closing price
        closing_price = hist['Close']
        new_row = pd.DataFrame({'Ticker': [ticker], 'Currency': [currency], 'ClosingPrice': [closing_price]})
        df_valid_tickers = pd.concat([df_valid_tickers, new_row], ignore_index=True)        
        

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Initialize empty lists for monthly returns and standard deviations
monthly_returns = []
std_dev = []

# Calculate monthly returns and standard deviations for each ticker
for ticker in df_valid_tickers['Ticker']:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        monthly_close = hist['Close'].resample('M').ffill()
        
        # Calculate monthly returns
        monthly_return = abs(1 - monthly_close.pct_change()).mean()
        monthly_returns.append(monthly_return)
        
        # Calculate standard deviation of monthly returns
        monthly_std = monthly_close.pct_change().std()
        std_dev.append(monthly_std)
        
    except Exception as e:
        print(f"Error calculating metrics for {ticker}: {e}")

# Add calculated metrics to the dataframe
df_valid_tickers['MonthlyReturn'] = monthly_returns
df_valid_tickers['StdDev'] = std_dev

# Create series for order based on metrics
order_by_return = df_valid_tickers['MonthlyReturn'].rank(ascending=True)
order_by_std_dev = df_valid_tickers['StdDev'].rank(ascending=True)

# Combine the orders to get the combined order position
combined_order = order_by_return + order_by_std_dev

# Add combined order position to dataframe
df_valid_tickers['CombinedOrder'] = combined_order

# Sort tickers based on combined order position
sorted_tickers = df_valid_tickers.sort_values(by='CombinedOrder')['Ticker']



# Create series for order based on metrics
order_by_return = df_valid_tickers['MonthlyReturn'].sort_values().rank(ascending=True)
order_by_std_dev = df_valid_tickers['StdDev'].sort_values().rank(ascending=True)

# Combine the orders to get the combined order position
combined_order = order_by_return + order_by_std_dev

# Add combined order position to dataframe
df_valid_tickers['CombinedOrder'] = combined_order

# Sort tickers based on combined order position
sorted_tickers = df_valid_tickers.sort_values(by='CombinedOrder')['Ticker']

# Define the benchmark ticker symbol
benchmark_symbol = '^GSPC'

# Function to fetch historical data for a ticker and the benchmark
def get_stock_data(ticker):
    try:
        # Fetch historical data for the given ticker
        stock = yf.Ticker(ticker)
        
        # Get the historical closing prices and calculate percentage changes
        stock_hist = stock.history(start=start_date, end=end_date)['Close'].pct_change().dropna()
        
        # Fetch historical data for the benchmark (^GSPC) and calculate percentage changes
        benchmark = yf.Ticker(benchmark_symbol)
        benchmark_hist = benchmark.history(start=start_date, end=end_date)['Close'].pct_change().dropna()
        
        return stock_hist, benchmark_hist
    except Exception as e:
        # Handle any errors that occur during data fetching
        print(f"Error fetching data for {ticker}: {e}")
        return None, None

# Initialize a 'Beta' column in df_valid_tickers with None values
df_valid_tickers['Beta'] = None 

# Calculate Beta for each ticker against the benchmark
for i, ticker in enumerate(df_valid_tickers['Ticker']):
    # Fetch stock and benchmark data
    stock_data, benchmark_data = get_stock_data(ticker)
    
    # Proceed if data is available for both stock and benchmark
    if stock_data is not None and benchmark_data is not None:
        # Calculate covariance and variance
        covariance = stock_data.cov(benchmark_data)
        variance = benchmark_data.var()
        
        # Calculate Beta and assign it to the 'Beta' column in the DataFrame
        beta = covariance / variance
        df_valid_tickers.loc[i, 'Beta'] = beta



# Calculate the 'CombinedOrder' by adding 'CombinedOrder' column values 
# with the absolute rank of 'Beta' values in ascending order
# The 'CombinedOrder' column will be updated by combining the current values 
# with the rank of absolute 'Beta' values, sorting them in ascending order
df_valid_tickers['CombinedOrder'] = df_valid_tickers['CombinedOrder'] + abs(df_valid_tickers['Beta']).rank(ascending=True)


# Find tickers with highest, lowest, and closest-to-zero Beta
highest_beta_ticker = df_valid_tickers[df_valid_tickers['Beta'] == df_valid_tickers['Beta'].max()]['Ticker'].values[0]
lowest_beta_ticker = df_valid_tickers[df_valid_tickers['Beta'] == df_valid_tickers['Beta'].min()]['Ticker'].values[0]

# Find ticker closest to zero Beta (abs(Beta) closest to 0)
closest_to_zero_ticker = df_valid_tickers.iloc[(df_valid_tickers['Beta'] - 0).abs().argsort()[:1]]['Ticker'].values[0]

# Plotting monthly returns for identified tickers
plt.figure(figsize=(18, 5))
lowest_beta_ticker_l = yf.Ticker(lowest_beta_ticker)
lowest_beta_ticker_hist = lowest_beta_ticker_l.history(start=start_date, end=end_date)['Close'].resample('M').ffill().pct_change().dropna()
lowest_beta_ticker_hist.plot(label='Lowest Beta', color='green')

highest_beta_ticker_l = yf.Ticker(highest_beta_ticker)
highest_beta_ticker_hist = highest_beta_ticker_l.history(start=start_date, end=end_date)['Close'].resample('M').ffill().pct_change().dropna()
highest_beta_ticker_hist.plot(label='Highest Beta', color='blue')

close_0_beta_ticker_l = yf.Ticker(closest_to_zero_ticker)
close_0_beta_ticker_hist = close_0_beta_ticker_l.history(start=start_date, end=end_date)['Close'].resample('M').ffill().pct_change().dropna()
close_0_beta_ticker_hist.plot(label='Closet to Zero Beta', color='black')

benchmark = yf.Ticker(benchmark_symbol)
benchmark_hist = benchmark.history(start=start_date, end=end_date)['Close'].resample('M').ffill().pct_change().dropna()
benchmark_hist.plot(label='Benchmark: S&P 500', color='red')
# Add labels, legend, title, etc. as needed
plt.ylim(-.2, .2) 
plt.legend()
plt.title('Monthly Returns for High, Low, and Closest-to-Zero Beta Tickers')
plt.xlabel('Date')
plt.ylabel('Monthly Return')
plt.show()


# Create an empty DataFrame with specified columns
Portfolio_Final = pd.DataFrame({
    'Ticker': [],
    'Price': [],
    'Currency': [],
    'Shares': [],
    'Value': [],
    'Weight': []
})
initial_investment = 750000 #CAD

# Function to convert stock prices from USD to CAD
def convert_prices_to_cad(close_price_usd):
    # Fetch the USD to CAD exchange rate using Yahoo Finance
    usd_to_cad_rate = yf.Ticker('CADUSD=X').history(start=start_date, end=end_date)["Close"].tz_localize(None)  
    if close_price_usd is not None and usd_to_cad_rate is not None:
        close_price_cad = close_price_usd * usd_to_cad_rate
    return close_price_cad.iloc[-1]

# Determine the maximum number of stocks allowed
num_stock = min(22, len(df_valid_tickers.index))  # Maximum number of stocks allowed

# Order the tickers based on their combined order
df_ordered = df_valid_tickers.sort_values(by='CombinedOrder', ascending=True)

# Calculate the weight for each stock in the portfolio
weight = (1 / num_stock) 

# Initialize investment, flat fee, and calculate initial ticker value
investment = initial_investment
flat_fee = 4.95  # CAD
ticker_value = (initial_investment - num_stock * flat_fee) * weight
total_weight = 1 
# Iterate through each ticker in the ordered dataframe up to the maximum number of stocks
for ticker in df_ordered['Ticker'][:num_stock]:
    if total_weight <=0:
        break

    ticker_data = yf.Ticker(ticker)
    ticker_info = ticker_data.fast_info
    ticker_currency = ticker_info.get('currency')

    # Check if the stock is denominated in USD
    if ticker_currency == 'USD':
        # Call the convert_prices_to_cad function to convert the price to CAD
        close_price_cad = convert_prices_to_cad(ticker_data.history(start=start_date, end=end_date)["Close"].iloc[-1])
    else:
        close_price_cad = ticker_data.history(start=start_date, end=end_date)["Close"].iloc[-1]
    
    # Calculate the number of shares for the current stock
    ticker_shares = ticker_value / close_price_cad

    # Create a new DataFrame for the current stock and append it to the Portfolio_Final DataFrame
    df_new_stock = pd.DataFrame({
        'Ticker': [ticker],
        'Price': [close_price_cad],
        'Currency': ['CAD'],
        'Shares': [ticker_shares],
        'Value': [ticker_value],
        'Weight': [weight]
    })
    total_weight -= weight
    Portfolio_Final = pd.concat([Portfolio_Final, df_new_stock], ignore_index=True)

print ("The total value of the portfolio is: " + str(Portfolio_Final['Value'].sum()))
print ("The total weight of the portfolio is: " + str(Portfolio_Final['Weight'].sum()))


Stocks_Final = pd.DataFrame() # outputs the final dataframe with the recommended stocks
Stocks_Final['Ticker'] = Portfolio_Final['Ticker'] # creates a columns with the tickers for each stock in the portfolio
Stocks_Final['Shares'] = Portfolio_Final['Shares'] # creates a column with the shares for each stock in the portfolio
# Reset the index starting from 1
Portfolio_Final.reset_index(drop=True, inplace=True)
Portfolio_Final.index = Portfolio_Final.index + 1
Stocks_Final.to_csv('Stocks_Group_12.csv') # exports final portfolio to a csv file

