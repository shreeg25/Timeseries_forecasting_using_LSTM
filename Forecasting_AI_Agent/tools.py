import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import os
from utils import calculate_metrics, prepare_data_for_lstm
from statsmodels.tsa.stattools import adfuller
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DataTransformer:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sarima_diff_order = 0

    def prepare_for_linear(self, series: pd.Series) -> pd.Series:
        """Transforms data to be stationary for ARIMA/SARIMA."""
        print("[Tools]: Checking stationarity using ADF Test...")
        df_copy = series.copy().dropna()
        
        result = adfuller(df_copy)
        p_value = result[1]
        
        if p_value > 0.05:
            print("[Tools]: Data is non-stationary. Applying 1st order differencing.")
            df_copy = df_copy.diff().dropna()
            self.sarima_diff_order = 1
        else:
            print("[Tools]: Data is already stationary. No differencing required.")
            
        return df_copy

def run_linear_forecast(series, order=(1, 1, 1), seasonal_order=None, horizon=30):
    """Unified engine that executes pure ARIMA or SARIMA depending on seasonal_order."""
    if seasonal_order is None or seasonal_order == (0, 0, 0, 0):
        print(f"[Tools Engine]: Running pure ARIMA model with order {order}...")
        model = SARIMAX(series, order=order, seasonal_order=(0, 0, 0, 0), enforce_stationarity=False)
    else:
        print(f"[Tools Engine]: Running SARIMA model with order {order} and seasonal order {seasonal_order}...")
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
        
    fitted_model = model.fit(disp=False)
    forecast = fitted_model.forecast(steps=horizon)
    return forecast

def run_lstm_forecast(data, horizon=30):
    print("Intialising LSTM...")
    
    import tensorflow as tf

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X_train, y_train = prepare_data_for_lstm(train_scaled, n_steps=12)
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(50, activation='relu', input_shape=(12, 1)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
    
    test_predictions = []
    curr_batch = train_scaled[-12:].reshape(1, 12, 1)
    
    for i in range(horizon):
        curr_pred = model.predict(curr_batch, verbose=0)[0]
        test_predictions.append(curr_pred[0])
        
        curr_batch = np.append(curr_batch[:, 1:, :], 
                            [[curr_pred]], 
                            axis=1)
    
    test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
    
    future_index = pd.date_range(start=last_date, periods=horizon + 1, freq='MS')[1:]
    forecast = pd.Series(test_predictions.flatten(), index=future_index)
    
    metrics = calculate_metrics(data, forecast)
    
    return {
        'model': model,
        'forecast': forecast,
        'metrics': metrics,
        'scaler': scaler
    }