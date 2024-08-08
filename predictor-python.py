import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

# Step 1: Import S&P 500 Index Stock data using Yahoo Finance
# This retrieves the historical data for the S&P 500 Index, denoted by the ticker "^GSPC".
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")  # Download the maximum available historical data

# Step 2: Clean the stock data to begin training our model
# Remove irrelevant columns 'Dividends' and 'Stock Splits' as they do not contribute to price prediction.
del sp500["Dividends"]
del sp500["Stock Splits"]

# Create a new column "Tomorrow" that contains the closing price of the next trading day.
# This will be used to set up the target variable for our machine learning model.
sp500["Tomorrow"] = sp500["Close"].shift(-1)

# Step 3: Set up Machine Learning Target
# The "Target" column is 1 if tomorrow's closing price is higher than today's, otherwise 0.
# This transforms the problem into a binary classification task.
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Step 4: Filter data to include only records from 1990 onwards.
# This focuses the model on more recent market data, which is generally more relevant.
sp500 = sp500.loc["1990-01-01":].copy()

# Step 5: Initial model training using RandomForestClassifier
# The RandomForestClassifier is chosen for its robustness and ability to handle both regression and classification tasks.
# - n_estimators=100: Number of trees in the forest.
# - min_samples_split=100: Minimum number of samples required to split an internal node.
# - random_state=1: Ensures reproducibility of results.
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# Split the data into training and testing sets.
# The last 100 rows are used for testing, and the rest for training.
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Define the predictors (features) for the model.
# These are basic stock data points like Close, Volume, Open, High, and Low prices.
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Train the model using the training dataset.
model.fit(train[predictors], train["Target"])

# Predict the target variable on the test dataset.
# The result is a series of predictions (0 or 1) indicating price decrease or increase.
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

# Evaluate the precision of the model
# Precision score is the ratio of correctly predicted positive observations to the total predicted positives.
precision = precision_score(test["Target"], preds)

# Visualization of Predicted Target vs Actual Target
# Combined DataFrame to compare actual target values and model predictions.
combined = pd.concat([test["Target"], preds], axis=1)

# Step 6: Feature Engineering for Improved Model
# Introduce new predictors by calculating rolling averages and trends over different time horizons.
# These features help capture short-term and long-term market trends.
horizons = [2, 5, 60, 250, 1000]  # Time horizons in days
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()  # Calculate rolling averages over the specified horizon

    # Ratio of the current close price to the rolling average close price.
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    # Trend feature: Sum of target values over the past `horizon` days.
    trend = f"Trend_{horizon}"
    sp500[trend] = sp500.shift(1).rolling(horizon).sum()["Target"]

    # Add new predictors to the list.
    new_predictors += [ratio_column, trend]

# Remove rows with NaN values, which may have been introduced by rolling operations.
sp500 = sp500.dropna()

# Step 7: Train an improved model with new features
# A more complex RandomForestClassifier with more estimators and a smaller minimum sample split.
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Function to predict future stock price movements using the trained model.
# Instead of just predicting 0 or 1, it predicts the probability of an increase and sets a threshold.
def predict(train, test, predictors, model):
    # Fit the model on the training data
    model.fit(train[predictors], train["Target"])

    # Predict probabilities for the test data
    preds = model.predict_proba(test[predictors])[:, 1]  # Probability of the positive class (price increase)

    # Convert probabilities into binary predictions with a threshold of 0.6
    preds[preds >= .6] = 1  # High confidence in price increase
    preds[preds < .6] = 0   # Low confidence or confidence in price decrease

    # Convert predictions to a pandas Series with the same index as the test set
    preds = pd.Series(preds, index=test.index, name="Predictions")

    # Combine the actual target values with the predictions
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Function to backtest the model over a specified period.
# It simulates how the model would perform by repeatedly training on historical data and testing on the next period.
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    # Loop through the data in increments of `step`
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()  # Use data from the start up to the current step for training
        test = data.iloc[i:(i + step)].copy()  # Use the next `step` rows for testing

        # Predict and store the results
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    # Concatenate all predictions into a single DataFrame
    return pd.concat(all_predictions)

# Precision testing the new improved model
# Run the backtest over the entire dataset and evaluate the precision of predictions.
predictions = backtest(sp500, model, new_predictors)

# Print the combined DataFrame with actual and predicted values
# print(predictions)

# Print today's stock price
today_price = sp500.iloc[-1]["Close"]
print(f"Today's stock price: {today_price:.2f} USD")

# Print statement to indicate whether tomorrow's stock price will increase or decrease
# Based on the last prediction, determine the expected movement of the stock price.
if predictions.iloc[-1]["Predictions"] == 1:
    print("Tomorrow's predicted stock price will **increase**.")
else:
    print("Tomorrow's predicted stock price will **decrease**.")
