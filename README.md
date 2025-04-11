# BlackOptima (main_v1.py)

### Project Overview
This project implements a quantitative portfolio optimization strategy using the Black-Litterman model. It aims to construct an optimized portfolio based on historical stock data, market equilibrium, and momentum-based investor views.

|Ticker	|Price	|Currency|	Shares|	Weight|
| --- | --- | --- |--- |--- |
|AMZN|	183.99745687193536|	USD|	185.2792406507923|	0.04545454545454545|
|BMY|	66.18053321351317|	USD|	515.1198915385618|	0.04545454545454545|
|T.TO|	20.221773147583008|	CAD	|1685.8516235003742|	0.04545454545454545|
|RY.TO|	105.10741424560547|	CAD	|324.34352358101535|	0.04545454545454545|
|C|	51.017373495814354|	USD	|668.2215636543114|	0.04545454545454545|
- Table from optimized_portfolio.csv.

### Core Functionality:

1.  **Data Loading & Filtering:**
    *   Loads an initial list of stock tickers from `Tickers_Example.csv`.
    *   Filters tickers to include only those available via `yfinance` and meeting minimum liquidity requirements (average monthly volume >= 150,000 based on data from `2023-01-01` to `2023-10-31`).

2.  **Metric Calculation:**
    *   Fetches historical stock data for the specified period (`2023-01-01` to `2023-10-31`).
    *   Calculates average returns and EWMA volatility for each eligible stock.

3.  **Covariance Estimation:**
    *   Computes the covariance matrix of asset returns using the Ledoit-Wolf shrinkage method for improved stability.

4.  **Black-Litterman Model:**
    *   Estimates market equilibrium returns (implied by CAPM).
    *   Generates simple investor views based on 3-month price momentum.
    *   Combines equilibrium returns and views to produce posterior expected returns.

5.  **Portfolio Optimization:**
    *   Performs mean-variance optimization using the posterior returns and shrunk covariance matrix.
    *   Aims to maximize a utility function based on a specified risk aversion level (currently 2.5).
    *   Constraints include weights summing to 1 and a maximum allocation of 10% per stock.

6.  **Portfolio Construction:**
    *   Selects up to 22 stocks (`MAX_STOCKS = 22`) with the highest optimal weights.
    *   Normalizes the weights of the selected stocks.
    *   Calculates the number of shares for each stock based on an initial investment budget of CAD 750,000 (`INITIAL_INVESTMENT = 750000`) and recent prices (end of the analysis period). Currency conversion (USD to CAD) is handled where necessary.

### Input Files:
*   `Tickers_Example.csv`: A file containing a list of potential stock tickers (one per line).

### Output Deliverables:
*   `optimized_portfolio.csv`: A CSV file containing the final optimized portfolio, including:
    *   Ticker: Stock symbol.
    *   Price: Last closing price used for calculation (in CAD).
    *   Currency: Original currency of the stock.
    *   Shares: Calculated number of shares to purchase.
    *   Weight: Optimized weight of the stock in the final portfolio.
*   **Performance Plot:** A matplotlib chart comparing the cumulative returns of the optimized portfolio against the S&P 500 benchmark (`^GSPC`) over the analysis period. This plot is displayed when the script runs.

### Dependencies:
*   pandas
*   numpy
*   yfinance
*   matplotlib
*   scipy
