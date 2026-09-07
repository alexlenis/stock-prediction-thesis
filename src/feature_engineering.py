import os
import pandas as pd
import numpy as np

# -----------------------------
# SETTINGS
# -----------------------------
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
    "NVDA", "META", "NFLX", "AMD", "INTC",
    "SPY", "QQQ"
]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# -----------------------------
# RSI FUNCTION
# -----------------------------
def compute_rsi(series, window=14):
    delta = series.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
for ticker in TICKERS:
    input_file = os.path.join(RAW_DIR, f"{ticker}.csv")

    if not os.path.exists(input_file):
        print(f"[WARNING] Missing file: {ticker}")
        continue

    print(f"Processing {ticker}...")

    df = pd.read_csv(input_file)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # -----------------------------
    # BASIC FEATURES
    # -----------------------------
    df["Return"] = df["Close"].pct_change()

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()

    df["Volatility_5"] = df["Return"].rolling(5).std()
    df["Volatility_10"] = df["Return"].rolling(10).std()

    # -----------------------------
    # MOMENTUM
    # -----------------------------
    df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

    # -----------------------------
    # LAG RETURNS
    # -----------------------------
    df["Return_lag_1"] = df["Return"].shift(1)
    df["Return_lag_2"] = df["Return"].shift(2)
    df["Return_lag_3"] = df["Return"].shift(3)

    # -----------------------------
    # RSI
    # -----------------------------
    df["RSI_14"] = compute_rsi(df["Close"], 14)

    # -----------------------------
    # PRICE RELATIONS
    # -----------------------------
    df["High_Low_Spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["Open_Close_Change"] = (df["Close"] - df["Open"]) / df["Open"]

    # -----------------------------
    # TARGET
    # -----------------------------
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # -----------------------------
    # CLEAN
    # -----------------------------
    df = df.dropna().reset_index(drop=True)

    output_file = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    df.to_csv(output_file, index=False)

    print(f"[OK] {ticker} -> {len(df)} rows")

print("\nFeature engineering completed.")