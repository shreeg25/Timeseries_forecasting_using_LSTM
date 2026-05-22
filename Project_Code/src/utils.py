import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_data(file_path):
    """Load time series data from a CSV file"""
    data = pd.read_csv(file_path)
    return data

def preprocess_airline_data(data):
    """Preprocess the airline passenger dataset"""
    # Rename columns for clarity if needed
    if 'International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60' in data.columns:
        data = data.rename(columns={
            'International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60': 'Passengers'
        })
    
    # Drop rows with NaN values
    data = data.dropna()
    
    # Convert Month to datetime and set as index
    data['Month'] = pd.to_datetime(data['Month'])
    data = data.set_index('Month')
    
    return data

def test_stationarity(timeseries):
    """Test stationarity using Augmented Dickey-Fuller test"""
    result = adfuller(timeseries.dropna())
    
    output = {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4]
    }
    
    return output

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }
    
    return metrics

def plot_time_series(data, title='Time Series Plot', figsize=(12, 6)):
    """Plot a time series"""
    plt.figure(figsize=figsize)
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

def plot_forecast(train, test, forecast, title='Forecast vs Actual', figsize=(12, 6)):
    """Plot actual vs forecasted values"""
    plt.figure(figsize=figsize)
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(forecast, label='Forecast')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def prepare_data_for_lstm(data, n_steps=12):
    """Prepare data for LSTM model - creates sequences"""
    X, y = [], []
    values = data.values if hasattr(data, 'values') else data
    
    for i in range(len(values) - n_steps):
        X.append(values[i:i+n_steps])
        y.append(values[i+n_steps])
    
    # Reshape input to be [samples, time steps, features]
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = np.array(y)
    
    return X, y
