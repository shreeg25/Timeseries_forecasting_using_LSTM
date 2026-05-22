import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import (
    load_data, 
    preprocess_airline_data, 
    plot_time_series, 
    plot_forecast, 
    test_stationarity
)
from src.train import (
    train_test_split_time_series,
    train_sarima,
    train_autoarima,
    train_autoets,
    train_lstm
)

def run_model(model_name, data_path='Data/international-airline-passengers.csv', test_size=0.2):
    """
    Main function to run a time series forecasting model
    
    Args:
        model_name (str): Name of the model to train
        data_path (str): Path to the data file
        test_size (float): Proportion of data to use for testing
    """
    # Load and preprocess data
    print(f"Loading data from {data_path}...")
    data = load_data(data_path)
    data = preprocess_airline_data(data)
    
    # Plot the time series data
    print("Plotting time series data...")
    plot_time_series(data['Passengers'], title='International Airline Passengers (1949-1960)')
    
    # Test stationarity
    print("Testing stationarity...")
    stationarity_results = test_stationarity(data['Passengers'])
    print(f"ADF Test Statistic: {stationarity_results['Test Statistic']}")
    print(f"p-value: {stationarity_results['p-value']}")
    print("Critical Values:")
    for key, value in stationarity_results['Critical Values'].items():
        print(f"\t{key}: {value}")
    
    # Split data into train and test sets
    print(f"Splitting data with test_size={test_size}...")
    train_data, test_data = train_test_split_time_series(data['Passengers'], test_size=test_size)
    
    # Train the specified model
    print(f"Training {model_name} model...")
    if model_name.lower() == 'sarima':
        # Using SARIMAX from statsmodels with predefined parameters
        results = train_sarima(
            train_data, 
            test_data, 
            order=(1, 1, 1), 
            seasonal_order=(1, 1, 1, 12)
        )
    elif model_name.lower() == 'autoarima':
        # Using AutoARIMA from Nixtla
        results = train_autoarima(
            train_data, 
            test_data, 
            season_length=12
        )
    elif model_name.lower() == 'autoets':
        # Using AutoETS from Nixtla
        results = train_autoets(
            train_data, 
            test_data, 
            season_length=12, 
            model='ZZZ'  # ZZZ means all components will be optimized
        )
    elif model_name.lower() == 'lstm':
        # Using LSTM from TensorFlow/Keras
        results = train_lstm(
            train_data, 
            test_data, 
            n_steps=12,  # Use 12 months for seasonal pattern
            epochs=50, 
            batch_size=32, 
            neurons=50
        )
    else:
        raise ValueError(f"Model '{model_name}' not recognized. Choose from: 'sarima', 'autoarima', 'autoets', 'lstm'")
    
    # Print model metrics
    print("\nModel Performance Metrics:")
    for metric_name, metric_value in results['metrics'].items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Plot forecast vs actual
    print("\nPlotting forecast vs actual values...")
    plot_forecast(
        train_data, 
        test_data, 
        results['forecast'], 
        title=f'{model_name.upper()} Forecast vs Actual'
    )
    
    return results

def main():
    """Parse command line arguments and run the model"""
    parser = argparse.ArgumentParser(description='Train a time series forecasting model')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['sarima', 'autoarima', 'autoets', 'lstm'],
                        help='Model to train')
    parser.add_argument('--data_path', type=str, default='Data/international-airline-passengers.csv',
                        help='Path to data file')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    
    args = parser.parse_args()
    
    run_model(args.model, args.data_path, args.test_size)

if __name__ == "__main__":
    main()
