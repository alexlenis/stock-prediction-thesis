import time
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, '.')

print("Loading models...")
t0 = time.time()
from src.predict_engine import (
    rf, xgb, lgbm, log_model, log_scaler,
    lstm_cls_model, lstm_reg_model, lstm_reg_scaler,
    create_features, lstm_classification, forecast_lstm
)
print(f"Model load: {time.time()-t0:.1f}s")

import yfinance as yf
import pandas as pd

print("Downloading data...")
t1 = time.time()
df = yf.download("AAPL", period="6mo", interval="1d")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()
df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
df = create_features(df)
print(f"Data download + features: {time.time()-t1:.1f}s")

print("Running classifiers...")
t2 = time.time()
from sklearn.preprocessing import MinMaxScaler
FEATURES = list(rf.feature_names_in_)
row = df.iloc[-1]
data = {}
for col in FEATURES:
    if col in df.columns:
        try: data[col] = float(row[col])
        except: data[col] = 0.0
    else:
        data[col] = 0.0
import pandas as pd2
X = pd.DataFrame([data])
rf.predict(X)
xgb.predict(X)
lgbm.predict(X)
log_model.predict(log_scaler.transform(X))
lstm_classification(df)
print(f"Classifiers: {time.time()-t2:.1f}s")

print("Running forecast...")
t3 = time.time()
forecast_lstm(df)
print(f"Forecast: {time.time()-t3:.1f}s")

print(f"\nTOTAL: {time.time()-t0:.1f}s")