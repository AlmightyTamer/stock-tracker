"""
stock_lstm_predictor.py
Educational example: fetch stock prices, train LSTM, predict next days.

Usage:
  python stock_lstm_predictor.py
  (change ticker, lookback, epochs below or call functions from another script)
"""

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import datetime
import os
import random

# ---------------------------
# Reproducibility (best-effort)
# ---------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ---------------------------
# Helpers: fetch data
# ---------------------------
def fetch_data(ticker='AAPL', period='3y', interval='1d'):
    """
    Fetch historical OHLCV data from yfinance.
    Returns pandas DataFrame with DatetimeIndex.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data fetched for ticker {ticker}")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

# ---------------------------
# Preprocessing: build sequences
# ---------------------------
def create_sequences(values, lookback=60):
    """
    Turn a 1D array into (X, y) sequences for supervised learning.
    X shape: (samples, lookback, 1)
    y shape: (samples, )
    """
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i])
        y.append(values[i])
    X = np.array(X)
    y = np.array(y)
    # reshape X to (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

# ---------------------------
# Build model
# ---------------------------
def build_lstm_model(input_shape, units=64, dropout=0.2):
    model = Sequential([
        LSTM(units, input_shape=input_shape, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------------------
# Train and evaluate pipeline
# ---------------------------
def train_and_evaluate(ticker='AAPL',
                       period='3y',
                       lookback=60,
                       test_fraction=0.2,
                       epochs=20,
                       batch_size=32,
                       verbose=1):
    df = fetch_data(ticker=ticker, period=period)
    closes = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(closes)

    # train/test split indices
    split_idx = int(len(scaled) * (1 - test_fraction))
    train_scaled = scaled[:split_idx]
    test_scaled = scaled[split_idx - lookback:]  # include overlap for sequence creation

    X_train, y_train = create_sequences(train_scaled.flatten(), lookback)
    X_test, y_test = create_sequences(test_scaled.flatten(), lookback)

    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), verbose=verbose)

    # Predict on test set
    preds_scaled = model.predict(X_test).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Metrics
    mse = mean_squared_error(actual, preds)
    mae = mean_absolute_error(actual, preds)
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label='Actual (test)', linewidth=1)
    plt.plot(preds, label='Predicted (test)', linewidth=1)
    plt.title(f'{ticker} - Actual vs Predicted (test set)')
    plt.legend()
    plt.show()

    return {
        'model': model,
        'scaler': scaler,
        'lookback': lookback,
        'history': history,
        'df': df,
        'X_test': X_test,
        'y_test': y_test,
        'preds': preds,
        'actual': actual
    }

# ---------------------------
# Predict future n days (recursive)
# ---------------------------
def predict_next_n_days(model, scaler, recent_close_array, lookback=60, n_days=5):
    """
    recent_close_array: 1D numpy array of most recent raw Close prices (not scaled),
                        must have at least `lookback` entries (ordered oldest->newest).
    Returns list of predicted prices (raw scale).
    """
    if len(recent_close_array) < lookback:
        raise ValueError("Need at least `lookback` number of recent prices")

    preds = []
    cur_sequence = recent_close_array[-lookback:].reshape(-1, 1)
    # scale current sequence
    cur_scaled = scaler.transform(cur_sequence).flatten()

    for _ in range(n_days):
        X_input = cur_scaled.reshape((1, lookback, 1))
        pred_scaled = model.predict(X_input).flatten()[0]
        pred_raw = scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0]
        preds.append(pred_raw)

        # append predicted scaled value to sequence and drop oldest to slide window
        cur_scaled = np.append(cur_scaled[1:], pred_scaled)

    return preds

# ---------------------------
# Example usage
# ---------------------------
if __name__ == '__main__':
    TICKER = 'AAPL'       # change to any ticker, e.g., 'TSLA', 'MSFT', 'AMZN'
    PERIOD = '3y'         # how much history to fetch
    LOOKBACK = 60         # number of days used for input sequence
    TEST_FRACTION = 0.2
    EPOCHS = 25
    BATCH = 32

    results = train_and_evaluate(ticker=TICKER,
                                 period=PERIOD,
                                 lookback=LOOKBACK,
                                 test_fraction=TEST_FRACTION,
                                 epochs=EPOCHS,
                                 batch_size=BATCH,
                                 verbose=1)

    # predict the next 5 days
    df = results['df']
    model = results['model']
    scaler = results['scaler']
    lookback = results['lookback']

    recent_prices = df['Close'].values  # numpy array, ordered oldest->newest
    n_days = 5
    preds = predict_next_n_days(model, scaler, recent_prices, lookback=lookback, n_days=n_days)
    today = df.index[-1].date()
    future_dates = [today + datetime.timedelta(days=i+1) for i in range(n_days)]
    print("Next-day predictions (naive calendar days):")
    for d, p in zip(future_dates, preds):
        print(f"{d}: ${p:.2f}")

    # Optional: plot last 200 closes and appended predictions
    last_n = 200
    plt.figure(figsize=(12,5))
    plt.plot(df['Close'].values[-last_n:], label='Recent Close (actual)', linewidth=1)
    extended_x = list(range(len(df['Close'].values[-last_n:]))) + list(range(len(df['Close'].values[-last_n:]), len(df['Close'].values[-last_n:]) + n_days))
    extended_y = list(df['Close'].values[-last_n:]) + preds
    plt.plot(extended_x, extended_y, '--', label='Extended (with predictions)')
    plt.legend()
    plt.title(f'{TICKER} recent closes + {n_days}-day predictions')
    plt.show()
