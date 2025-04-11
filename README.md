### Project Overview
The project is designed to dynamically select a portfolio of stocks, optimized for safety based on certain criteria, and manage a specified budget for stock purchasing. The bot operates within the constraints of the US and Canadian stock markets, with an emphasis on fulfilling specific trading requirements.

#### Portfolio Generation:
- The bot must autonomously generate a portfolio without predefined stock tickers.
- Stock tickers will be sourced from the "Tickers.csv" file, which resides in the same directory as the executable code.
- The stock selection process will be restricted to securities listed on US and Canadian exchanges.
Stocks selected must meet a minimum average monthly trading volume threshold.
#### Portfolio Construction:
- The portfolio will consist of 10 to 22 stocks, with predefined weightings for each stock.
- The bot must utilize the full CAD 750,000 budget for stock purchases, accounting for transaction fees.
Portfolio stock valuations will be based on the closing prices from November 25, 2023.

### Output Deliverables
A Portfolio_Final DataFrame will be generated, containing detailed information about the selected stocks and their respective portfolio values.
A Stocks_Final DataFrame will be created, listing stock tickers along with the number of shares, and saved as "Stocks_Group_XX.csv".
