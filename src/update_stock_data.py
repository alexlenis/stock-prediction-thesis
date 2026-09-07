import os
import yfinance as yf

# ==============================
# SETTINGS
# ==============================

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
    "NVDA", "META", "NFLX", "AMD", "INTC",
    "SPY", "QQQ"
]

START_DATE = "2019-01-01"
END_DATE   = "2026-12-31"  # yfinance caps to today automatically

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(RAW_DIR, exist_ok=True)

# ==============================
# DOWNLOAD LOOP
# ==============================

for ticker in TICKERS:
    print(f"[DOWNLOAD] {ticker}")

    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=True
        )

        if df.empty:
            print(f"[WARNING] No data for {ticker}")
            continue

        df.reset_index(inplace=True)

        file_path = os.path.join(RAW_DIR, f"{ticker}.csv")
        df.to_csv(file_path, index=False)

        print(f"[OK] Saved {ticker} ({len(df)} rows)")

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")

print("\n[DONE] All data downloaded.")