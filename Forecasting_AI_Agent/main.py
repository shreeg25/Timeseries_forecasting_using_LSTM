# main.py (Temporary Diagnostic Version)
print("🏁 [Diagnostic]: Script execution started.")

print("🔍 Testing import: os, sys...")
import os
import sys

os.environ["OPENBLAS_MAIN_FREE"] = "1"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["OMP_NUM_THREADS"] = "1"

print("🔍 Testing import: pandas...")
import pandas as pd

print("🔍 Testing import: numpy...")
import numpy as np

print("🔍 Testing import: statsmodels (Deep API)...")
import statsmodels.api as sm

print("🔍 Testing import: statsmodels tools...")
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

print("🔍 Testing import: scikit-learn...")
from sklearn.preprocessing import MinMaxScaler

print("🔍 Testing import: openai...")
from openai import OpenAI

print("🔍 Testing import: pydantic...")
from pydantic import BaseModel, Field

print("🔍 Testing import: python-dotenv...")
from dotenv import load_dotenv

print("🎉 [SUCCESS]: All core libraries imported perfectly without freezing!")