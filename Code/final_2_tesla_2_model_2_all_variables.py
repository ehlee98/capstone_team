import pickle

import pandas as pd
from tensorflow.python.ops.gen_math_ops import select_v2
ticker='tsla'
variable_method='all_variables'

tsla_data = pd.read_csv("tsla_prepared_data.csv").dropna()
tsla_data_full = tsla_data[['Adj_Closed_price',
        'ma_10', 'ma_30', 'ema_10', 'volatility_10',
       'rsi_14', 'interest_rate_mom', 'unemployment_rate_mom',
       'gdp_growth_mom', 'cpi_mom', 'industrial_production_mom',
       'vehicle_sales_mom', 'personal_consumption_mom',
       'retail_sales_auto_mom', 'crude_oil_prices_mom', 'sp500_mom',
       'daily_sentiment_score']]
tsla_data_price_alone = tsla_data[['Adj_Closed_price']]

#Feature Selection
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(tsla_data_full)

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # Sequence of features
        y.append(data[i + sequence_length, 0])  # Target: closed_price
    return np.array(X), np.array(y)

# Number of folds
num_folds = 5
sequence_length=30
fold_size = (len(scaled_data)-sequence_length) // (num_folds+1)
# Prepare to save predictions and actual values for all folds
all_predictions = []
all_actuals = []
all_dates = []

# 5-Fold Blocked Cross-Validation with Predictions and Visualization
for fold in range(num_folds):
    test_start = (fold +1)* fold_size
    if fold<num_folds-1:
        test_end = (fold + 2) * fold_size + sequence_length
    else:
        test_end=len(scaled_data)

    # Split into training and test data for the current fold
    train_data = scaled_data[:test_start]
    test_data = scaled_data[test_start:test_end]
    test_dates = tsla_data['Date'].iloc[test_start:test_end].iloc[sequence_length:].values  # Dates for test data

    # Create sequences
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    print(fold)
    print(test_start)
    print(test_end)
    print(X_train.shape)

    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(units=50),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    model_filename = f"tesla_lstm_model_fold_{fold + 1}.keras"
    # Predict on the test set
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(
        np.concatenate([predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]
    y_test = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]

    # Save predictions and actual values for this fold
    all_predictions.extend(predictions)
    all_actuals.extend(y_test)
    all_dates.extend(test_dates)

    # Print fold MSE
    mse = mean_squared_error(y_test, predictions)
    print(f"Fold {fold + 1} MSE: {mse:.4f}")

results_df = pd.DataFrame({'Date': all_dates, 'Actual': all_actuals, 'Predicted': all_predictions})

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(results_df['Date'], results_df['Actual'], label='Actual', alpha=0.7)
plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Closed Price')
plt.title('Predicted vs Actual Closed Prices')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=True)

# Baseline performance
baseline_mse = mean_squared_error(y_test, model.predict(X_test))
# Permutation Feature Importance
feature_names = tsla_data_full.columns[1:]  # Exclude 'Adj_Closed_price'
importance_scores = {}
for i in range(X_test.shape[2]):  # Loop over each feature
    # Make a copy of the test set and shuffle only the i-th feature
    x_test_permuted = X_test.copy()
    np.random.shuffle(x_test_permuted[:, :, i])  # Shuffle each feature column
    # Calculate the error after shuffling
    permuted_mse = mean_squared_error(y_test, model.predict(x_test_permuted))
    importance_scores[feature_names[i-1]] = permuted_mse - baseline_mse  # Calculate score difference
# Display feature importance scores
importance_scores_sorted = {k: v for k, v in sorted(importance_scores.items(), key=lambda item: item[1], reverse=True)}
print("Permutation Feature Importance (Higher is more important):")
for feature, score in importance_scores_sorted.items():
    print(f"{feature}: {score:.4f}")

output_importance_scores_sorted = pd.DataFrame(list(importance_scores_sorted.items()), columns=['Variable', 'Importance'])
output_importance_scores_sorted['ticker']=ticker
output_filename = f"{ticker}_importance_scores.csv"
output_importance_scores_sorted.to_csv(output_filename, index=False)

select_top_features=list(importance_scores_sorted.keys())[:3]
select_top_features.insert(0, 'Adj_Closed_price')
select_top_features.insert(1, 'daily_sentiment_score')
tsla_data_selected=tsla_data[select_top_features]


#LSTM
model_name='lstm'

# Scale the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(tsla_data_selected)


def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # Sequence of features
        y.append(data[i + sequence_length, 0])  # Target: closed_price
    return np.array(X), np.array(y)

# Number of folds
num_folds = 5
sequence_length=30
fold_size = (len(scaled_data)-sequence_length) // (num_folds+1)
# Prepare to save predictions and actual values for all folds
all_predictions = []
all_actuals = []
all_dates = []

# 5-Fold Blocked Cross-Validation with Predictions and Visualization
for fold in range(num_folds):
    test_start = (fold +1)* fold_size
    if fold<num_folds-1:
        test_end = (fold + 2) * fold_size + sequence_length
    else:
        test_end=len(scaled_data)

    # Split into training and test data for the current fold
    train_data = scaled_data[:test_start]
    test_data = scaled_data[test_start:test_end]
    test_dates = tsla_data['Date'].iloc[test_start:test_end].iloc[sequence_length:].values  # Dates for test data

    # Create sequences
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    print(fold)
    print(test_start)
    print(test_end)
    print(X_train.shape)

    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(units=50),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    model_filename = f"{ticker}_{variable_method}_{model_name}_model_fold_{fold + 1}.keras"
    model.save(model_filename)
    print("Model saved successfully.")
    # Predict on the test set
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(
        np.concatenate([predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]
    y_test = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]

    # Save predictions and actual values for this fold
    all_predictions.extend(predictions)
    all_actuals.extend(y_test)
    all_dates.extend(test_dates)

    # Print fold MSE
    mse = mean_squared_error(y_test, predictions)
    print(f"Fold {fold + 1} MSE: {mse:.4f}")

results_df = pd.DataFrame({'Date': all_dates, 'Actual': all_actuals, 'Predicted': all_predictions})
results_df.to_csv(f'{ticker}_{variable_method}_{model_name}_predictions_vs_actuals.csv', index=False)


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(results_df['Date'], results_df['Actual'], label='Actual', alpha=0.7)
plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Closed Price')
plt.title('Predicted vs Actual Closed Prices')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=True)



from keras.models import load_model

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
all_metrics = []

for fold in range(num_folds):
    print(f"Evaluating Fold {fold + 1}...")
    test_start = (fold + 1) * fold_size
    test_end = (fold + 2) * fold_size + sequence_length

    test_data = scaled_data[test_start:test_end]
    X_test, y_test = create_sequences(test_data, sequence_length)
    model_filename = f"{ticker}_{variable_method}_{model_name}_model_fold_{fold + 1}.keras"
    model = load_model(model_filename)
    print("Model loaded successfully.")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(
        np.concatenate([predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]
    y_test = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Directional accuracy metrics
    actual_direction = np.sign(np.diff(y_test))  # 1 for increase, -1 for decrease
    predicted_direction = np.sign(np.diff(predictions))  # 1 for increase, -1 for decrease
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

    # Store metrics for the fold
    fold_metrics = {
        "Fold": fold + 1,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "Directional Accuracy (%)": directional_accuracy
    }
    all_metrics.append(fold_metrics)
    for metric, value in fold_metrics.items():
        if metric != "Fold":
            print(f"{metric}: {value:.4f}")

avg_metrics = {metric: np.mean([fold[metric] for fold in all_metrics]) for metric in all_metrics[0] if metric != "Fold"}

print("\nAverage Metrics Across All Folds:")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")

output_1_metrics = pd.DataFrame([avg_metrics])
output_1_metrics['ticker'] = ticker
output_1_metrics['variable_method'] = variable_method
output_1_metrics['model_name'] = model_name
output_filename = f"{ticker}_{variable_method}_{model_name}_metrics_1.csv"
output_1_metrics.to_csv(output_filename, index=False)

df = pd.merge(results_df[['Date','Predicted']], tsla_data[['Date', 'Next_day_Adj_Open_price']], on='Date', how='left')
df=df.rename(columns={'Next_day_Adj_Open_price': 'Actual'}).dropna()

# Add columns for percentage change in predicted and actual prices
df['predict_pct_change'] = df['Predicted'].pct_change() * 100  # Percent change in predicted price
df['actual_pct_change'] = df['Actual'].pct_change() * 100  # Percent change in actual price

quantiles = df['predict_pct_change'].quantile([0.1,0.25, 0.5, 0.75,0.9])
print(quantiles)
# Initialize variables


def trading_strategy(ticker,variable_method,model_name,df,long_threshold=0.1,short_threshold=-0.1,strategy='long_only'):
    capital = 100  # Start with $100
    position = 0  # Initially, no stock is held
    last_buy_price = None  # To track the price at which the stock was last bought
    buy_count = 0
    short_count = 0
    capital_history = [capital]  # To track capital over time
    for i in range(1, len(df)):
    # Buy Strategy
        if strategy in ['long_only','both'] and df.loc[i, 'predict_pct_change'] > long_threshold:
            if position == 0:  # If no stock is held, buy
                position = capital / df.loc[i, 'Actual']  # Number of shares bought
                last_buy_price = df.loc[i, 'Actual']
                capital = 0  # All money is invested
                buy_count += 1  # Increment the buy transaction count
            elif position < 0:  # If currently short, close the short position and then buy
                capital += abs(position) * df.loc[i, 'Actual']  # Close short at current price
                position = capital / df.loc[i, 'Actual']
                last_buy_price = df.loc[i, 'Actual']
                capital = 0
                buy_count += 1

        # Short-Sell Strategy
        elif strategy in ['both'] and df.loc[i, 'predict_pct_change'] < short_threshold:
            if position == 0:  # If no stock is held, short-sell
                position = -capital / df.loc[i, 'Actual']  # Negative for short
                last_buy_price = df.loc[i, 'Actual']
                capital = 0  # All money is tied in the short
                short_count += 1  # Increment the short transaction count
            elif position > 0:  # If currently long, close the long position and then short
                capital = position * df.loc[i, 'Actual']  # Sell all shares
                position = -capital / df.loc[i, 'Actual']
                last_buy_price = df.loc[i, 'Actual']
                capital = 0
                short_count += 1

        # Exit positions if neither condition is met (predicted price change not above 2% or below -2%)
        elif position > 0:  # If holding a long position, sell
            capital = position * df.loc[i, 'Actual']
            position = 0  # Reset position
        elif position < 0:  # If holding a short position, close it
            capital += abs(position) * df.loc[i, 'Actual']
            position = 0  # Reset position

    # Track capital history
        capital_history.append(capital if position == 0 else capital + position * df.loc[i, 'Actual'])

    # At the end, if there is still a position, close it at the last day's actual price
    if position > 0:  # If long
        capital = position * df.loc[len(df) - 1, 'Actual']
    elif position < 0:  # If short
        capital += abs(position) * df.loc[len(df) - 1, 'Actual']


    # Track final capital
    capital_history.append(capital)

    # Convert capital history to a pandas Series
    capital_history = pd.Series(capital_history)

    file_name = f"capital_history_{strategy}.csv"
    capital_history.to_csv(file_name, index=False)
    print(f"Capital history saved to {file_name}")

    # Print the final capital, buy transaction count, and short transaction count
    print(f"Final Capital: ${capital:.2f}")
    print(f"Total Buy Transactions: {buy_count}")
    print(f"Total Short Transactions: {short_count}")


    # Risk Metrics

    # 1. Volatility of Returns
    daily_returns = capital_history.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized
    print(f"Volatility (Annualized): {volatility:.4f}")

    # 4. Value at Risk (VaR)
    confidence_level = 0.95
    VaR = np.percentile(daily_returns, (1 - confidence_level) * 100)
    print(f"Value at Risk (VaR) at {confidence_level * 100:.0f}%: {VaR:.4%}")

    metrics = {
        'Final Capital ($)': [capital],
        'Total Buy Transactions': [buy_count],
        'Total Short Transactions': [short_count],
        'Volatility (Annualized)': [volatility],
        f'Value at Risk (VaR) at {confidence_level * 100:.0f}%': [VaR],
        'ticker': [ticker],
        'variable_method': [variable_method],
        'model_name': [model_name],
        'trading_strategy':[strategy],
    }
    output_2_metrics = pd.DataFrame(metrics)
    output_filename = f"{ticker}_{variable_method}_{model_name}_metrics_2.csv"
    output_2_metrics.to_csv(output_filename, index=False)

    return capital_history

trading_strategy(ticker,variable_method,model_name,df,long_threshold=quantiles[0.75],short_threshold=-0.01,strategy='long_only')

trading_strategy(ticker,variable_method,model_name,df,long_threshold=quantiles[0.75],short_threshold=quantiles[0.1],strategy='both')





# LSTM with XGBoost

model_name='lstm_xgboost'
# Number of folds
from xgboost import XGBRegressor
num_folds = 5
sequence_length=30
fold_size = (len(scaled_data)-sequence_length) // (num_folds+1)
# Prepare to save predictions and actual values for all folds
all_predictions = []
all_actuals = []
all_dates = []

# 5-Fold Blocked Cross-Validation with Predictions and Visualization
for fold in range(num_folds):
    test_start = (fold +1)* fold_size
    if fold<num_folds-1:
        test_end = (fold + 2) * fold_size + sequence_length
    else:
        test_end=len(scaled_data)

    # Split into training and test data for the current fold
    train_data = scaled_data[:test_start]
    test_data = scaled_data[test_start:test_end]
    test_dates = tsla_data['Date'].iloc[test_start:test_end].iloc[sequence_length:].values  # Dates for test data

    # Create sequences
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    print(fold)
    print(test_start)
    print(test_end)
    print(X_train.shape)

    model_filename = f"{ticker}_{variable_method}_lstm_model_fold_{fold + 1}.keras"
    lstm_model = load_model(model_filename)

    # Extract features from LSTM
    lstm_features_train = lstm_model.predict(X_train)
    lstm_features_test = lstm_model.predict(X_test)
    combined_train_features = np.hstack((X_train.reshape(X_train.shape[0], -1), lstm_features_train))
    combined_test_features = np.hstack((X_test.reshape(X_test.shape[0], -1), lstm_features_test))

    # Train XGBoost on LSTM features
    xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=50)
    xgb_model.fit(combined_train_features, y_train)
    model_filename = f"{ticker}_{variable_method}_{model_name}_model_fold_{fold + 1}.json"
    xgb_model.save_model(model_filename)
    print("Model saved successfully.")

    # Predict with XGBoost
    xgb_predictions = xgb_model.predict(combined_test_features)

    # Inverse transform predictions and actual values
    xgb_predictions = scaler.inverse_transform(
        np.concatenate([xgb_predictions.reshape(-1, 1), np.zeros((len(xgb_predictions), scaled_data.shape[1] - 1))], axis=1))[:, 0]
    y_test = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_data.shape[1] - 1))], axis=1))[:, 0]

    # Store predictions and actuals for this fold
    all_predictions.extend(xgb_predictions)
    all_actuals.extend(y_test)
    all_dates.extend(test_dates)
    mse = mean_squared_error(y_test, xgb_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, xgb_predictions)
    r2 = r2_score(y_test, xgb_predictions)

    # Directional accuracy metrics
    actual_direction = np.sign(np.diff(y_test))  # 1 for increase, -1 for decrease
    predicted_direction = np.sign(np.diff(xgb_predictions))  # 1 for increase, -1 for decrease
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

    # Store metrics for the fold
    fold_metrics = {
        "Fold": fold + 1,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "Directional Accuracy (%)": directional_accuracy
    }
    all_metrics.append(fold_metrics)
    for metric, value in fold_metrics.items():
        if metric != "Fold":
            print(f"{metric}: {value:.4f}")

avg_metrics = {metric: np.mean([fold[metric] for fold in all_metrics]) for metric in all_metrics[0] if metric != "Fold"}

print("\nAverage Metrics Across All Folds:")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")


output_1_metrics = pd.DataFrame([avg_metrics])
output_1_metrics['ticker'] = ticker
output_1_metrics['variable_method'] = variable_method
output_1_metrics['model_name'] = model_name
output_filename = f"{ticker}_{variable_method}_{model_name}_metrics_1.csv"
output_1_metrics.to_csv(output_filename, index=False)

results_df = pd.DataFrame({'Date': all_dates, 'Actual': all_actuals, 'Predicted': all_predictions})
results_df.to_csv(f'{ticker}_{variable_method}_{model_name}_predictions_vs_actuals.csv', index=False)


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(results_df['Date'], results_df['Actual'], label='Actual', alpha=0.7)
plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Closed Price')
plt.title('Predicted vs Actual Closed Prices')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=True)


df = pd.merge(results_df[['Date','Predicted']], tsla_data[['Date', 'Next_day_Adj_Open_price']], on='Date', how='left')
df=df.rename(columns={'Next_day_Adj_Open_price': 'Actual'}).dropna()


# Add columns for percentage change in predicted and actual prices
df['predict_pct_change'] = df['Predicted'].pct_change() * 100  # Percent change in predicted price
df['actual_pct_change'] = df['Actual'].pct_change() * 100  # Percent change in actual price

quantiles = df['predict_pct_change'].quantile([0.1,0.25, 0.5, 0.75,0.9])
print(quantiles)
trading_strategy(ticker,variable_method,model_name,df,long_threshold=quantiles[0.75],short_threshold=-0.01,strategy='long_only')

trading_strategy(ticker,variable_method,model_name,df,long_threshold=quantiles[0.75],short_threshold=quantiles[0.1],strategy='both')




# LSTM with Attention
model_name='lstm_attention'
# Number of folds
num_folds = 5
sequence_length=30
fold_size = (len(scaled_data)-sequence_length) // (num_folds+1)
# Prepare to save predictions and actual values for all folds
all_predictions = []
all_actuals = []
all_dates = []

# 5-Fold Blocked Cross-Validation with Predictions and Visualization
for fold in range(num_folds):
    test_start = (fold +1)* fold_size
    if fold<num_folds-1:
        test_end = (fold + 2) * fold_size + sequence_length
    else:
        test_end=len(scaled_data)

    # Split into training and test data for the current fold
    train_data = scaled_data[:test_start]
    test_data = scaled_data[test_start:test_end]
    test_dates = tsla_data['Date'].iloc[test_start:test_end].iloc[sequence_length:].values  # Dates for test data

    # Create sequences
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    print(fold)
    print(test_start)
    print(test_end)
    print(X_train.shape)

    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention
    from tensorflow.keras.models import Model

    # Define input shape
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)

    # Input layer
    inputs = Input(shape=input_shape)

    # First LSTM layer
    lstm_out_1 = LSTM(units=100, return_sequences=True)(inputs)
    dropout_1 = Dropout(0.3)(lstm_out_1)

    # Attention layer
    query = Dense(100)(dropout_1)  # Queries for attention mechanism
    key = Dense(100)(dropout_1)  # Keys for attention mechanism
    value = Dense(100)(dropout_1)  # Values for attention mechanism
    attention_out = Attention()([query, value])

    # Second LSTM layer (process attention output)
    lstm_out_2 = LSTM(units=50)(attention_out)
    dropout_2 = Dropout(0.3)(lstm_out_2)

    # Dense layers
    dense_1 = Dense(32, activation='relu')(dropout_2)
    outputs = Dense(1)(dense_1)  # Final output layer

    # Build and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    model_filename = f"{ticker}_{variable_method}_{model_name}_model_fold_{fold + 1}.keras"
    model.save(model_filename)
    print("Model saved successfully.")
    # model_filename = f"voo_PriceAlone_lstm_model_fold_{fold + 1}.keras"
    # model = load_model(model_filename)
    # Predict on the test set
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(
        np.concatenate([predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]
    y_test = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]

    # Save predictions and actual values for this fold
    all_predictions.extend(predictions)
    all_actuals.extend(y_test)
    all_dates.extend(test_dates)


results_df = pd.DataFrame({'Date': all_dates, 'Actual': all_actuals, 'Predicted': all_predictions})
results_df.to_csv(f'{ticker}_{variable_method}_{model_name}_predictions_vs_actuals.csv', index=False)


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(results_df['Date'], results_df['Actual'], label='Actual', alpha=0.7)
plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Closed Price')
plt.title('Predicted vs Actual Closed Prices')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=True)



from keras.models import load_model

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
all_metrics = []

for fold in range(num_folds):
    print(f"Evaluating Fold {fold + 1}...")
    test_start = (fold + 1) * fold_size
    test_end = (fold + 2) * fold_size + sequence_length

    test_data = scaled_data[test_start:test_end]
    X_test, y_test = create_sequences(test_data, sequence_length)
    model_filename = f"{ticker}_{variable_method}_{model_name}_model_fold_{fold + 1}.keras"
    model = load_model(model_filename)
    print("Model loaded successfully.")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(
        np.concatenate([predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]
    y_test = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Directional accuracy metrics
    actual_direction = np.sign(np.diff(y_test))  # 1 for increase, -1 for decrease
    predicted_direction = np.sign(np.diff(predictions))  # 1 for increase, -1 for decrease
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

    # Store metrics for the fold
    fold_metrics = {
        "Fold": fold + 1,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "Directional Accuracy (%)": directional_accuracy
    }
    all_metrics.append(fold_metrics)
    for metric, value in fold_metrics.items():
        if metric != "Fold":
            print(f"{metric}: {value:.4f}")

avg_metrics = {metric: np.mean([fold[metric] for fold in all_metrics]) for metric in all_metrics[0] if metric != "Fold"}

print("\nAverage Metrics Across All Folds:")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")


output_1_metrics = pd.DataFrame([avg_metrics])
output_1_metrics['ticker'] = ticker
output_1_metrics['variable_method'] = variable_method
output_1_metrics['model_name'] = model_name
output_filename = f"{ticker}_{variable_method}_{model_name}_metrics_1.csv"
output_1_metrics.to_csv(output_filename, index=False)


df = pd.merge(results_df[['Date','Predicted']], tsla_data[['Date', 'Next_day_Adj_Open_price']], on='Date', how='left')
df=df.rename(columns={'Next_day_Adj_Open_price': 'Actual'}).dropna()


# Add columns for percentage change in predicted and actual prices
df['predict_pct_change'] = df['Predicted'].pct_change() * 100  # Percent change in predicted price
df['actual_pct_change'] = df['Actual'].pct_change() * 100  # Percent change in actual price

quantiles = df['predict_pct_change'].quantile([0.1,0.25, 0.5, 0.75,0.9])
print(quantiles)
trading_strategy(ticker,variable_method,model_name,df,long_threshold=quantiles[0.75],short_threshold=-0.01,strategy='long_only')

trading_strategy(ticker,variable_method,model_name,df,long_threshold=quantiles[0.75],short_threshold=quantiles[0.1],strategy='both')



#ARIMAX

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pickle
model_name='arimax'

result = adfuller(tsla_data_selected['Adj_Closed_price'])
print(f"ADF Statistic: {result[0]}")
print(f"P-Value: {result[1]}")

# If p-value > 0.05, the series is non-stationary
if result[1] > 0.05:
    print("The series is non-stationary. Differencing is required.")
else:
    print("The series is stationary.")


dataset_diff = tsla_data_selected['Adj_Closed_price'].diff().dropna()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plot_acf(dataset_diff, ax=ax[0], title='ACF')
plot_pacf(dataset_diff, ax=ax[1], title='PACF')
plt.show(block=True)

# Fit ARIMAX model
model = ARIMA(tsla_data_selected['Adj_Closed_price'], exog=tsla_data_selected.iloc[:,1:], order=(1, 1, 1))  # Adjust order (p, d, q) as needed
model_fit = model.fit()
# Save the model
model_filename = f"{ticker}_{variable_method}_{model_name}_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model_fit, f)
print(f"ARIMA model saved successfully.")


print(f"AIC: {model_fit.aic}, BIC: {model_fit.bic}")
results_df = pd.DataFrame({
    "Date": tsla_data['Date'].iloc[1:].reset_index(drop=True),  # Exclude the first row due to differencing
    "Actual": tsla_data_selected['Adj_Closed_price'].iloc[1:].reset_index(drop=True),
    "Predicted": model_fit.fittedvalues[1:].reset_index(drop=True)
})
results_df=results_df.rename(columns={'Adj_Closed_price': 'Actual'}).dropna()
results_df.to_csv(f'{ticker}_{variable_method}_{model_name}_predictions_vs_actuals.csv', index=False)


# Assuming your DataFrame is named `results_df`
actual_values = results_df['Actual'].values
predicted_values = results_df['Predicted'].values

# Calculate regression metrics
mse = mean_squared_error(actual_values, predicted_values)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_values, predicted_values)
r2 = r2_score(actual_values, predicted_values)

# Calculate directional accuracy
actual_direction = np.sign(np.diff(actual_values))  # 1 for increase, -1 for decrease
predicted_direction = np.sign(np.diff(predicted_values))  # 1 for increase, -1 for decrease
directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

# Print metrics
metrics = {
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae,
    "R²": r2,
    "Directional Accuracy (%)": directional_accuracy
}

# Display the metrics
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")


output_1_metrics = pd.DataFrame([metrics])
output_1_metrics['ticker'] = ticker
output_1_metrics['variable_method'] = variable_method
output_1_metrics['model_name'] = model_name
output_filename = f"{ticker}_{variable_method}_{model_name}_metrics_1.csv"
output_1_metrics.to_csv(output_filename, index=False)


df = pd.merge(results_df[['Date','Predicted']], tsla_data[['Date', 'Next_day_Adj_Open_price']], on='Date', how='left')
df=df.rename(columns={'Next_day_Adj_Open_price': 'Actual'}).dropna()


# Add columns for percentage change in predicted and actual prices
df['predict_pct_change'] = df['Predicted'].pct_change() * 100  # Percent change in predicted price
df['actual_pct_change'] = df['Actual'].pct_change() * 100  # Percent change in actual price

quantiles = df['predict_pct_change'].quantile([0.1,0.25, 0.5, 0.75,0.9])
print(quantiles)
trading_strategy(ticker,variable_method,model_name,df,long_threshold=quantiles[0.75],short_threshold=-0.01,strategy='long_only')

trading_strategy(ticker,variable_method,model_name,df,long_threshold=quantiles[0.75],short_threshold=quantiles[0.1],strategy='both')
