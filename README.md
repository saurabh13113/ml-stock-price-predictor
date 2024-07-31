# ml-stock-price-predictor-💸

## Summer Project that uses SciKit Learn and pandas to analyze S&amp;P 500 data and predict tomorrow's stock price using Random Forest

This repository contains a Python script that predicts the next day's stock price movement for the S&P 500 Index using a Random Forest Classifier. The model is trained on historical stock data and is enhanced with feature engineering to improve its predictive accuracy.

## Overview

The script is structured as follows:

### Data Collection and Cleaning

- Historical data for the S&P 500 Index is retrieved using the `yfinance` library. The script fetches the maximum available period of data.
- Irrelevant columns, such as `Dividends` and `Stock Splits`, are removed to focus solely on price data.
- A new column, `Tomorrow`, is created to hold the closing price for the next trading day. This is used to set up the target variable.

### Target Variable Setup

- The target variable, `Target`, is a binary indicator where:
  - `1` indicates that tomorrow's closing price is expected to be higher than today's.
  - `0` indicates that tomorrow's closing price is expected to be lower or the same.
- This binary classification setup allows the model to predict whether the stock price will increase or decrease.

![image](https://github.com/user-attachments/assets/916822dc-f26a-4bc3-9bc1-36f40f61b9a1)

### Initial Model Training

- A `Random Forest Classifier` is used for the initial model. The following parameters are used:
  - `n_estimators=100`: The number of trees in the forest.
  - `min_samples_split=100`: The minimum number of samples required to split an internal node.
  - `random_state=1`: Ensures the reproducibility of results.
- The dataset is split into a training set (all but the last 100 days) and a testing set (the last 100 days).
- Basic predictors such as `Close`, `Volume`, `Open`, `High`, and `Low` prices are used to train the model.

### Feature Engineering

- To enhance the model, additional features are introduced:
  - Rolling averages are calculated over different time horizons (2 days, 5 days, 60 days, etc.).
  - *Trend* features are created by summing the target values over these horizons to capture market momentum.
- These new features help the model to better capture both short-term and long-term trends in the stock market.

### Improved Model

- The enhanced model uses a more complex `Random Forest Classifier` with:
  - `n_estimators=200`: A larger number of trees.
  - `min_samples_split=50`: A smaller sample split to capture finer details in the data.
- The prediction function is improved to predict the probability of a stock price increase. A threshold of 0.6 is applied:
  - If the probability is ≥ 0.6, the model predicts a price increase (`1`).
  - If the probability is < 0.6, the model predicts a price decrease (`0`).

![image](https://github.com/user-attachments/assets/33963838-be03-4410-a215-f464d698779d)

### Backtesting

- The `backtest()` function simulates how the model would have performed historically by:
  - Training the model on a portion of the data.
  - Testing it on the next segment, iterating through the entire dataset.
- This method provides a realistic assessment of the model's accuracy and robustness over time.

### Final Prediction

- The script concludes by printing whether the model predicts that the stock price will increase or decrease the next day, based on the most recent available data.

## Conclusion

This script demonstrates how to use a `Random Forest Classifier` to predict stock price movements. The model is enhanced by feature engineering, and its performance is evaluated through backtesting. While the model makes predictions based on historical data, it should be noted that real-world market behavior is influenced by a myriad of factors not captured in this model.

**Note:** This model is a simplification and should be used with caution. Incorporating additional data sources, such as macroeconomic indicators or sentiment analysis, could further improve the model's performance. 
*Please also note that both jupyter and python files have been provided* as jupyter makes it much easier for visualizing data, simple code is in the python file with comments as needed.
