import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from statsmodels.tsa.seasonal import seasonal_decompose

# Importing components from tools.py
from tools import DataTransformer, run_linear_forecast, run_lstm_forecast

# 1. Define the structured output schema for the LLM Router
class RouterDecision(BaseModel):
    selected_path: Literal["ARIMA", "SARIMA", "LSTM"]
    reasoning: str = Field(
        description="Clear mathematical and context-driven justification for choosing this model track."
    )

# 2. Define the Hybrid Data + Context Router
def hybrid_agent_router(df, target_col, business_context):
    data_size = len(df)

    if data_size < 24:
        print("⚠️ [Agent Router]: Data too small. Routing to ARIMA.")
        return "ARIMA"

    try:
        decomposition = seasonal_decompose(df[target_col], model='additive', period=12, extrapolate_trend='freq')
        seasonal_variance = np.var(decomposition.seasonal)
        residual_variance = np.var(decomposition.resid)
        seasonality_ratio = seasonal_variance / (residual_variance + 1e-5)
    except Exception:
        seasonality_ratio = 0.0

    if data_size > 200:
        print("🧠 [Agent Router]: Large dataset. Routing to LSTM.")
        return "LSTM"
    elif seasonality_ratio > 0.2:
        print("🧠 [Agent Router]: Seasonality detected. Routing to SARIMA.")
        return "SARIMA"
    else:
        print("🧠 [Agent Router]: No strong seasonality. Routing to ARIMA.")
        return "ARIMA"

# 3. Main Orchestrator Loop
def run_autonomous_forecasting_agent(df: pd.DataFrame, target_col: str, business_context: str = "", horizon: int = 30):
    """
    The main execution loop. It takes raw data, passes it to the router,
    deploys the correct DataTransformer instance, executes the selected tool, 
    and returns the cleaned final predictions.
    """
    print("\n🚀 [Agent Loop]: Initializing AI forecasting loop...")
    
    # Run the hybrid router to get our destination path string
    chosen_path = hybrid_agent_router(df, target_col, business_context)
    print(f"🎯 [Agent Loop]: Target Execution Track Locked -> {chosen_path}")
    
    # Instantiate your data transformer tool
    transformer = DataTransformer()
    
    # ─── ROUTE 1: PURE ARIMA ───
    if chosen_path == "ARIMA":
        transformed_series = transformer.prepare_for_linear(df[target_col])
        forecast_output = run_linear_forecast(
            transformed_series, order=(1, 1, 1), seasonal_order=None, horizon=horizon
        )
        
        # If the tool applied differencing to make it stationary, inverse-transform it back to normal values
        if transformer.sarima_diff_order > 0:
            print("🔄 [Agent Loop]: Reversing differencing transformation...")
            forecast_output = df[target_col].iloc[-1] + forecast_output.cumsum()
            
        return pd.Series(forecast_output, name="ARIMA_Forecast")
        
    # ─── ROUTE 2: SARIMA ───
    elif chosen_path == "SARIMA":
        transformed_series = transformer.prepare_for_linear(df[target_col])
        # Passing a standard 12-month or 12-step seasonality order parameter
        forecast_output = run_linear_forecast(
            transformed_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), horizon=horizon
        )
        
        if transformer.sarima_diff_order > 0:
            print("🔄 [Agent Loop]: Reversing differencing transformation...")
            forecast_output = df[target_col].iloc[-1] + forecast_output.cumsum()
            
        return pd.Series(forecast_output, name="SARIMA_Forecast")
        
    # ─── ROUTE 3: LSTM DEEP LEARNING ───
    elif chosen_path == "LSTM":
        # Let your tools.py LSTM take over, which internally uses your custom utils.py prepare function
        lstm_result = run_lstm_forecast(df[target_col], horizon=horizon)
        
        # Extract the forecast series out of the returned tools dictionary
        forecast_output = lstm_result['forecast']
        print(f"✅ [Agent Loop]: LSTM sequence processed. In-sample model evaluation metrics generated.")
        
        return forecast_output