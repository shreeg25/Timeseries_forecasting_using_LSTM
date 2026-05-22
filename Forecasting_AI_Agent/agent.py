import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI
from statsmodels.tsa.seasonal import seasonal_decompose
from dotenv import load_dotenv

# Importing components from tools.py
from tools import DataTransformer, run_linear_forecast, run_lstm_forecast

load_dotenv()

# 1. Define the structured output schema for the LLM Router
class RouterDecision(BaseModel):
    selected_path: Literal["ARIMA", "SARIMA", "LSTM"]
    reasoning: str = Field(
        description="Clear mathematical and context-driven justification for choosing this model track."
    )

# 2. Define the Hybrid Data + Context Router
def hybrid_agent_router(df: pd.DataFrame, target_col: str, business_context: str) -> str:
    """
    Analyzes the structural properties of the dataframe alongside qualitative 
    business context to determine the optimal forecasting path.
    """
    # this fetches the key from .env file
    client = OpenAI() 
    data_size = len(df)
    
    # ─── PHASE 1: HARD DETERMINISTIC GATEKEEPER ───
    # If data is critically small, skip the LLM and deep learning entirely to avoid failure
    if data_size < 24:
        print("⚠️ [Agent Router]: Data footprint too small. Safe-routing to ARIMA.")
        return "ARIMA"
        
    # ─── PHASE 2: STATISTICAL METRIC EXTRACTION ───
    # We attempt a seasonal decomposition (assuming a default cycle of 12 for monthly or 7 for daily)
    # If the data is too short for a full period decomposition, we handle it gracefully
    try:
        decomposition = seasonal_decompose(df[target_col], model='additive', period=12, extrapolate_trend='freq')
        seasonal_variance = np.var(decomposition.seasonal)
        residual_variance = np.var(decomposition.resid)
        seasonality_ratio = seasonal_variance / (residual_variance + 1e-5)
        has_seasonality = "Strong/Detected" if seasonality_ratio > 0.2 else "Weak/Absent"
    except Exception:
        seasonality_ratio = 0.0
        has_seasonality = "Could not compute (Data too short/No clear frequency)"

    # Format a concise summary for the LLM brain to read
    data_profile = f"""
    - Available Data Points: {data_size} rows
    - Seasonality Metric Ratio: {seasonality_ratio:.4f} ({has_seasonality})
    """

    # ─── PHASE 3: COGNITIVE LLM ROUTING ───
    system_prompt = """
    You are the Lead Decision Engine of an Advanced Time Series AI Agent. Your job is to select the most optimized forecasting framework.
    
    ROUTING STANDARD OPERATING PROCEDURES:
    1. Select 'ARIMA' if data is limited/small (< 200 rows) AND seasonality metrics state it is Weak/Absent.
    2. Select 'SARIMA' if data is limited/small (< 200 rows) AND recurring periodic seasonal patterns are clearly detected.
    3. Select 'LSTM' if data scale is highly expansive (> 200 rows) OR if the business context outlines complex, non-linear disruption (e.g., massive marketing shifts, unpredictable macro events) where traditional linear statistical models collapse.
    """
    
    user_prompt = f"""
    Evaluate the following technical profile and business parameters to lock in the track:
    
    DATA METRICS SUMMARY:
    {data_profile}
    
    QUALITATIVE BUSINESS CONTEXT:
    {business_context}
    """
    
    # Call OpenAI using the native Structured Outputs parse interface
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=RouterDecision,
    )
    
    decision = completion.choices[0].message.parsed
    print(f"\n🧠 [Agent Decision Reasoning]: {decision.reasoning}")
    return decision.selected_path

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