import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import os
from .utils import calculate_metrics, prepare_data_for_lstm

# For Nixtla models
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS
    os.environ['NIXTLA_ID_AS_COL'] = '1'
    nixtla_available = True
except ImportError:
    nixtla_available = False
    print("Nixtla statsforecast package not available. Install with: pip install statsforecast")

# For LSTM model
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    tf_available = True
except ImportError:
    tf_available = False
    print("TensorFlow not available. Install with: pip install tensorflow")

def train_test_split_time_series(data, test_size=0.2):
    """Split time series data into train and test sets"""
    train_size = int(len(data) * (1 - test_size))
    train, test = data[:train_size], data[train_size:]
    return train, test

def train_sarima(train_data, test_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """Train a SARIMA model using statsmodels"""
    model = SARIMAX(train_data, 
                  order=order,
                  seasonal_order=seasonal_order,
                  enforce_stationarity=False,
                  enforce_invertibility=False)
    
    model_fit = model.fit(disp=False)
    
    forecast = model_fit.forecast(steps=len(test_data))
    
    metrics = calculate_metrics(test_data, forecast)
    
    return {
        'model': model_fit,
        'forecast': forecast,
        'metrics': metrics
    }

def train_autoarima(train_data, test_data, season_length=12):
    """Train an AutoARIMA model using Nixtla's statsforecast"""
    if not nixtla_available:
        raise ImportError("Nixtla statsforecast package not available")
    
    train_df = pd.DataFrame({
        'unique_id': 'series0',
        'ds': train_data.index,
        'y': train_data.values
    })
    
    models = [AutoARIMA(season_length=season_length)]
    sf = StatsForecast(models=models, freq='MS')
    sf.fit(df=train_df)
    
    h = len(test_data)
    forecasts = sf.forecast(h=h, df=train_df)
    
    forecast = pd.Series(forecasts['AutoARIMA'].values, index=test_data.index)
    
    metrics = calculate_metrics(test_data, forecast)
    
    return {
        'model': sf,
        'forecast': forecast,
        'metrics': metrics
    }

def train_autoets(train_data, test_data, season_length=12, model='ZZZ'):
    """Train an AutoETS model using Nixtla's statsforecast"""
    if not nixtla_available:
        raise ImportError("Nixtla statsforecast package not available")
    
    train_df = pd.DataFrame({
        'unique_id': 'series0',
        'ds': train_data.index,
        'y': train_data.values
    })
    
    models = [AutoETS(season_length=season_length, model=model)]
    sf = StatsForecast(models=models, freq='MS')
    sf.fit(df=train_df)
    
    h = len(test_data)
    forecasts = sf.forecast(h=h, df=train_df)
    
    forecast = pd.Series(forecasts['AutoETS'].values, index=test_data.index)
    
    metrics = calculate_metrics(test_data, forecast)
    
    return {
        'model': sf,
        'forecast': forecast,
        'metrics': metrics
    }

def train_lstm(train_data, test_data, n_steps=12, epochs=50, batch_size=32, neurons=50):
    """Train an LSTM model for time series forecasting"""
    if not tf_available:
        raise ImportError("TensorFlow not available")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    X_train, y_train = prepare_data_for_lstm(train_scaled, n_steps=n_steps)
    
    model = Sequential()
    model.add(LSTM(neurons, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    test_predictions = []
    curr_batch = train_scaled[-n_steps:].reshape(1, n_steps, 1)
    
    for i in range(len(test_data)):
        curr_pred = model.predict(curr_batch, verbose=0)[0]
        test_predictions.append(curr_pred[0])
        
        curr_batch = np.append(curr_batch[:, 1:, :], 
                              [[curr_pred]], 
                              axis=1)
    
    test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
    
    forecast = pd.Series(test_predictions.flatten(), index=test_data.index)
    
    metrics = calculate_metrics(test_data, forecast)
    
    return {
        'model': model,
        'forecast': forecast,
        'metrics': metrics,
        'scaler': scaler
    }