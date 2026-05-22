# main.py
import os
import sys 

# THE NUCLEAR CP    U BYPASS: Force NumPy/OpenBLAS to skip all hardware profiling
os.environ["OPENBLAS_CORETYPE"] = "Generic"
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["PANDAS_FUTURE_INFER_STRING_DTYPE"] = "0"
import sys
sys.modules['pyarrow'] = None 

print(" [System Initializing]: Math overrides applied successfully...")
import pandas as pd
from agent import run_autonomous_forecasting_agent
from utils import load_data, preprocess_airline_data, plot_forecast

def main():
    print(" [System Initializing]: Setting up Time Series AI Agent Environment...")

    # 1. Path to your dataset 
    # (Update 'airline-passengers.csv' to your actual local file name or path if different)
    data_file_path = "airline-passengers.csv" 
    
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(
            f"Could not find '{data_file_path}' in the directory. "
            f"Please ensure your time series CSV file is placed in this root folder."
        )

    # 2. Load and Preprocess data using your robust utils.py engine
    raw_data = load_data(data_file_path)
    processed_df = preprocess_airline_data(raw_data)
    
    # Define the target column (for airline data, it's 'Passengers')
    target_column = 'Passengers'
    
    # 3. Simulate real-world qualitative business context
    # Try changing this text string later to see how the agent's routing behavior shifts!
    business_context = """
    We are looking at standard historical travel demand patterns. 
    Macroeconomic indicators are steady, and there are no unprecedented 
    market disruptions or flash promotions expected in the coming weeks.
    """

    # 4. Define the forecasting horizon (e.g., predict the next 12 steps/months)
    forecast_horizon = 12

    # 5. Fire up the Agent Loop!
    # The agent will look at the length, test for seasonality, read the context,
    # apply stationarity transformations, train, and return final inverted predictions.
    final_predictions = run_autonomous_forecasting_agent(
        df=processed_df,
        target_col=target_column,
        business_context=business_context,
        horizon=forecast_horizon
    )

    # 6. Output and Visualizations
    print("\n --- AGENT FORECAST RUN SUCCESSFUL ---")
    print("Generated Predictions:")
    print(final_predictions)

    # Split historical data into train and a placeholder 'test' window for plotting
    train_slice = processed_df[target_column].iloc[:-forecast_horizon]
    test_slice = processed_df[target_column].iloc[-forecast_horizon:]

    print("\n Visualizing the agent's selected path output...")
    plot_forecast(
        train=train_slice,
        test=test_slice,
        forecast=final_predictions,
        title=f"AI Agent Forecast Output Tracker ({final_predictions.name})"
    )

if __name__ == "__main__":
    main()