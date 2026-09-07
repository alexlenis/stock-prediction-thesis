import os
import yfinance as yf
import pandas as pd

# -----------------------------
# SETTINGS
# -----------------------------
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"

# Project root -> one level above /src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(RAW_DIR, exist_ok=True)

# -----------------------------
# DOWNLOAD DATA
# -----------------------------
for ticker in TICKERS:
    print(f"Downloading data for {ticker}...")

    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False)

    if df.empty:
        print(f"[WARNING] No data found for {ticker}")
        continue

    # Αν το yfinance επιστρέψει MultiIndex columns, τις απλοποιούμε
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    output_file = os.path.join(RAW_DIR, f"{ticker}.csv")
    df.to_csv(output_file, index=False)

    print(f"[OK] Saved {ticker} -> {output_file} ({len(df)} rows)")

print("\nDownload completed.")