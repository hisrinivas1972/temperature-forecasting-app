import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import statsmodels.api as sm
from pmdarima import auto_arima
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

st.set_option('deprecation.showPyplotGlobalUse', False)

def create_features(df, target):
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['lag_1'] = df[target].shift(1)
    df['lag_2'] = df[target].shift(2)
    df['rolling_3'] = df[target].rolling(window=3).mean()
    df['rolling_7'] = df[target].rolling(window=7).mean()
    df.fillna(method='bfill', inplace=True)
    return df

def create_sequences(values, seq_len=7):
    X_seq, y_seq = [], []
    for i in range(len(values) - seq_len):
        X_seq.append(values[i:i+seq_len])
        y_seq.append(values[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

def run_ensemble_models(X_train, y_train, X_test):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    pred_gb = gb.predict(X_test)

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    pred_xgb = xgb_model.predict(X_test)

    return pred_rf, pred_gb, pred_xgb

def run_arima(train, test_len):
    model = auto_arima(train, seasonal=True, m=12, trace=False,
                       error_action='ignore', suppress_warnings=True)
    model.fit(train)
    pred = model.predict(n_periods=test_len)
    return pred

def run_prophet(train_df, periods):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast['yhat'].iloc[-periods:].values

def run_lstm(values, seq_len=7):
    X, y = create_sequences(values, seq_len)
    split_idx = int(len(X)*0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_len,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, epochs=50, batch_size=16,
              validation_split=0.1, callbacks=[early_stop], verbose=0)

    pred = model.predict(X_test).flatten()
    return pred, y_test

def main():
    st.title('🌡️ Temperature Forecasting with Multiple Models')

    uploaded_file = st.file_uploader("Upload your CSV file with temperature data", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        st.write('## Uploaded Data Sample')
        st.dataframe(df.head())

        target = 'Temp Max'
        df_feat = create_features(df.copy(), target)
        feature_cols = ['Rain', 'year', 'month', 'day', 'dayofweek', 'quarter', 'dayofyear', 'lag_1', 'lag_2', 'rolling_3', 'rolling_7']

        split_idx = int(len(df) * 0.8)
        X = df_feat[feature_cols]
        y = df_feat[target]

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        st.write('## Running Ensemble Models (Random Forest, Gradient Boosting, XGBoost)...')
        pred_rf, pred_gb, pred_xgb = run_ensemble_models(X_train, y_train, X_test)
        mae_rf = mean_absolute_error(y_test, pred_rf)
        mae_gb = mean_absolute_error(y_test, pred_gb)
        mae_xgb = mean_absolute_error(y_test, pred_xgb)

        st.write(f'Random Forest MAE: {mae_rf:.3f}')
        st.write(f'Gradient Boosting MAE: {mae_gb:.3f}')
        st.write(f'XGBoost MAE: {mae_xgb:.3f}')

        st.write('## Running ARIMA Model...')
        train_arima = df[target].iloc[:split_idx]
        test_arima_len = len(df) - split_idx
        pred_arima = run_arima(train_arima, test_arima_len)
        mae_arima = mean_absolute_error(df[target].iloc[split_idx:], pred_arima)
        st.write(f'ARIMA MAE: {mae_arima:.3f}')

        st.write('## Running Prophet Model...')
        df_prophet = df[['Date', target]].rename(columns={'Date': 'ds', target: 'y'})
        train_prophet = df_prophet.iloc[:split_idx]
        pred_prophet = run_prophet(train_prophet, test_arima_len)
        mae_prophet = mean_absolute_error(df_prophet['y'].iloc[split_idx:], pred_prophet)
        st.write(f'Prophet MAE: {mae_prophet:.3f}')

        st.write('## Running LSTM Model...')
        temp_values = df[target].values
        pred_lstm, y_lstm_test = run_lstm(temp_values)
        mae_lstm = mean_absolute_error(y_lstm_test, pred_lstm)
        st.write(f'LSTM MAE: {mae_lstm:.3f}')

        st.write('## Plotting Predictions vs Actual')

        plt.figure(figsize=(12, 8))
        plt.plot(y_test.values, label='Actual', marker='o')

        plt.plot(pred_rf, label=f'Random Forest MAE={mae_rf:.3f}', marker='x')
        plt.plot(pred_gb, label=f'Gradient Boosting MAE={mae_gb:.3f}', marker='^')
        plt.plot(pred_xgb, label=f'XGBoost MAE={mae_xgb:.3f}', marker='v')
        plt.plot(range(len(pred_arima)), pred_arima, label=f'ARIMA MAE={mae_arima:.3f}', marker='s')
        plt.plot(range(len(pred_prophet)), pred_prophet, label=f'Prophet MAE={mae_prophet:.3f}', marker='P')
        plt.plot(range(7, 7+len(pred_lstm)), pred_lstm, label=f'LSTM MAE={mae_lstm:.3f}', marker='d')

        plt.legend()
        plt.title('Temperature Forecasting Models Comparison')
        plt.xlabel('Test Samples')
        plt.ylabel('Temp Max')
        st.pyplot()
    else:
        st.info('Please upload a CSV file to get started.')

if __name__ == "__main__":
    main()
