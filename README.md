To do:

1. Exploratory Analysis of Technical Indicators:
 - Choose the indicators you want to use (e.g., SMA, EMA, etc.).
 - Calculate these indicators for your dataset (e.g., using Python and libraries like pandas and numpy).
 - Plot the indicators over the price data to visualize how they relate to price movements.
 - Justify the choice of specific values (e.g., 10, 20, 50, 100) for the indicators by observing historical price bounces around these levels. You might look for instances where the price seems to have reversed its direction around these specific points.
 - Calculating Returns and Binary Trends:

2. Forecast the price for a given horizon (e.g., 5 days in advance).
 - Calculate the difference between the current day's closing price and the predicted price.
 - Transform this difference into a binary signal (0 for price decrease, 1 for price increase).
 - Create a DataFrame with two columns: "Closing Price" and "Binary Trend".
 - Implementing the Trading Strategy:

3. Based on the binary signals, implement the trading strategy:
 - If the binary trend is 1, buy.
 - If the next cell is 1, do nothing.
 - If the next cell is 0, sell.
 - Calculating Returns and Metrics:

4. Calculate returns based on the trading strategy.
- Compute the Asset Expected Mean Return, Gain-Loss Ratio, Sharpe Ratio, and CAGR:
- Asset Expected Mean Return: This is the average return of your asset over a specified period.
- Gain-Loss Ratio: It measures the ratio of gains to losses. It's calculated by dividing the average gain by the average loss.
- Sharpe Ratio: It measures the risk-adjusted return of an investment. It's calculated by dividing the excess return of the investment by its standard deviation of returns.
- Compound Annual Growth Rate (CAGR): It represents the mean annual growth rate of an investment over a specified time period.


Which library to choose for RL: 
https://neptune.ai/blog/the-best-tools-for-reinforcement-learning-in-python

I would prefer to go with RL_Coach, as there is Quantile Regression DQN. Suggestions?
