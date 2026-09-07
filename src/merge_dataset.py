import os
import pandas as pd

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
SENTIMENT_PATH = os.path.join(BASE_DIR, "data", "processed", "sentiment_daily.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
    "NVDA", "META", "NFLX", "AMD", "INTC",
    "SPY", "QQQ"
]

# ==============================
# LOAD SENTIMENT
# ==============================
sent_df = pd.read_csv(SENTIMENT_PATH)
sent_df["date"] = pd.to_datetime(sent_df["date"])
sent_df = sent_df.dropna(subset=["date"])

# ==============================
# INDICATOR FUNCTIONS
# ==============================
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = (ema_fast - ema_slow) / series  # normalised by price
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def compute_bb_percent(series, window=20):
    sma  = series.rolling(window).mean()
    std  = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb_range = upper - lower
    return (series - lower) / bb_range.replace(0, float("nan"))

# ==============================
# TARGET FUNCTION
# ==============================
def classify_return(x):
    if x > 0.02:
        return 1
    elif x < -0.02:
        return -1
    else:
        return 0

# ==============================
# PROCESS
# ==============================
all_data = []

for ticker in TICKERS:
    print(f"[PROCESS] {ticker}")

    file_path = os.path.join(RAW_DATA_PATH, f"{ticker}.csv")
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)

    # ==============================
    # FIX NUMERIC
    # ==============================
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ==============================
    # FIX DATE
    # ==============================
    if "Date" not in df.columns:
        print(f"[ERROR] No Date column in {ticker}")
        continue

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df.rename(columns={"Date": "date"}, inplace=True)
    df = df.sort_values("date")

    # ==============================
    # SENTIMENT
    # ==============================
    ticker_sent = sent_df[sent_df["ticker"] == ticker]

    if ticker_sent.empty:
        print(f"[WARNING] No sentiment for {ticker} — proceeding with zero-filled sentiment")

    ticker_sent = ticker_sent.sort_values("date")

    # ==============================
    # 🔥 FIX: NORMAL MERGE (ΟΧΙ ASOF)
    # ==============================
    merged = pd.merge(
        df,
        ticker_sent,
        on="date",
        how="left"
    )

    # ==============================
    # FILL SENTIMENT
    # ==============================
    merged["sent_mean"]       = merged["sent_mean"].fillna(0)
    merged["sent_std"]        = merged["sent_std"].fillna(0)
    merged["sent_count"]      = merged["sent_count"].fillna(0)
    merged["avg_reliability"] = merged["avg_reliability"].fillna(0) if "avg_reliability" in merged.columns else 0.0

    # ==============================
    # FEATURES
    # ==============================
    merged["Return"] = merged["Close"].pct_change()
    merged["MA_5"] = merged["Close"].rolling(5).mean()
    merged["MA_10"] = merged["Close"].rolling(10).mean()
    merged["Volatility_5"] = merged["Return"].rolling(5).std()
    merged["Momentum_5"] = merged["Close"] - merged["Close"].shift(5)

    merged["Return_lag_1"] = merged["Return"].shift(1)
    merged["Return_lag_2"] = merged["Return"].shift(2)

    merged["RSI_14"] = compute_rsi(merged["Close"])

    macd_line, macd_signal = compute_macd(merged["Close"])
    merged["MACD"]        = macd_line
    merged["MACD_signal"] = macd_signal
    merged["BB_percent"]  = compute_bb_percent(merged["Close"])
    merged["Volume_Ratio"] = merged["Volume"] / merged["Volume"].rolling(20).mean()

    # Normalised intraday range (replaces raw High/Low for logistic regression)
    merged["High_ratio"] = merged["High"] / merged["Close"] - 1
    merged["Low_ratio"]  = 1 - merged["Low"] / merged["Close"]

    # ==============================
    # SENTIMENT FEATURES
    # ==============================
    merged["sentiment_lag_1"] = merged["sent_mean"].shift(1)
    merged["sentiment_lag_2"] = merged["sent_mean"].shift(2)

    merged["sentiment_3d"] = merged["sent_mean"].rolling(3).mean()
    merged["sentiment_7d"] = merged["sent_mean"].rolling(7).mean()

    merged["sentiment_momentum"] = merged["sent_mean"] - merged["sent_mean"].shift(3)

    # ==============================
    # TARGET
    # ==============================
    merged["future_return"] = merged["Close"].shift(-5) / merged["Close"] - 1
    merged["signal"] = merged["future_return"].apply(classify_return)

    merged["ticker"] = ticker

    # ==============================
    # CLEAN — drop the indicator warm-up period instead of zero-filling it.
    # BB_percent & Volume_Ratio need 20 prior rows, so the first ~20 rows per
    # ticker would otherwise carry artificial zeros that corrupt training.
    # ==============================
    price_feats = [
        "future_return", "Return", "MA_5", "MA_10", "Volatility_5",
        "Momentum_5", "Return_lag_1", "Return_lag_2", "RSI_14",
        "MACD", "MACD_signal", "BB_percent", "Volume_Ratio",
        "High_ratio", "Low_ratio",
    ]
    merged = merged.dropna(subset=price_feats)
    # Sentiment columns are legitimately 0 on days with no news.
    merged = merged.fillna(0)

    # ==============================
    # PER-TICKER CHRONOLOGICAL SPLIT (first 80% by date = train, last 20% = test)
    # Marking the split in the dataset guarantees every model AND the backtest
    # use the IDENTICAL hold-out, and that hold-out is genuinely out-of-sample
    # for each ticker (no cross-ticker leakage, no in-sample evaluation).
    # ==============================
    merged = merged.sort_values("date").reset_index(drop=True)
    n   = len(merged)
    cut = int(n * 0.8)
    merged["is_train"] = [True] * cut + [False] * (n - cut)

    print(f"[INFO] {ticker}: {merged.shape}  (train={cut}, test={n - cut})")

    all_data.append(merged)

# ==============================
# FINAL
# ==============================
if not all_data:
    print("[ERROR] No data to concatenate!")
    exit()

final_df = pd.concat(all_data)

print(f"[INFO] Final dataset: {final_df.shape}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"[DONE] Saved to {OUTPUT_PATH}")