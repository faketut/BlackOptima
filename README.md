# Portfolio Robo-Advising Project

## Project Overview

The Portfolio Robo-Advising Project involves teams selecting the "safest" portfolio to compete for prizes. Teams will create portfolios dynamically while aiming for specific performance criteria.

### Goals
- **Safest Portfolio:** It aims for an ending value closest to zero compared to the initial $750,000 CAD portfolio value.

## Task Details
- **Library Usage:** numpy, pandas, matplotlib, Yahoo Finance

### Key Project Criteria
- Dynamic creation of a portfolio without prior knowledge of stock tickers.
- Read stock tickers from "Tickers.csv" located in the same directory as the code.
- Include only US and Canadian listed stocks.
- Ensure selected stocks meet minimum average monthly volume criteria.
- Construct portfolios consisting of 10 to 22 stocks, each with specified weightings.
- Utilize the entire $750,000 CAD budget, accounting for stock purchase fees.
- Set portfolio values using November 25, 2023, closing prices.

### Output Expectations
- Generate a "Portfolio_Final" DataFrame showcasing stock details and portfolio value.
- Create a "Stocks_Final" DataFrame with stock tickers and shares, saved as "Stocks_Group_XX.csv".

## Additional Insights
- **Dividends:** Not considered for this assignment.


