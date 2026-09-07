import os
import time
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ==============================
# LOAD MODELS
# ==============================
rf         = joblib.load(os.path.join(MODELS_DIR, "rf_final.pkl"))
xgb        = joblib.load(os.path.join(MODELS_DIR, "xgboost_final.pkl"))
lgbm       = joblib.load(os.path.join(MODELS_DIR, "lightgbm_final.pkl"))
log_model  = joblib.load(os.path.join(MODELS_DIR, "logistic_final.pkl"))
log_scaler = joblib.load(os.path.join(MODELS_DIR, "logistic_scaler.pkl"))
lstm_cls_model  = load_model(os.path.join(MODELS_DIR, "lstm_final.keras"))
lstm_cls_scaler = joblib.load(os.path.join(MODELS_DIR, "lstm_scaler.pkl"))

# Regression LSTM for price forecasting (loaded lazily — only if model file exists)
_lstm_reg_model  = None
_lstm_reg_feat   = None
_lstm_reg_target = None

def _load_reg_model():
    global _lstm_reg_model, _lstm_reg_feat, _lstm_reg_target
    if _lstm_reg_model is not None:
        return True
    reg_path = os.path.join(MODELS_DIR, "lstm_regressor.keras")
    if not os.path.exists(reg_path):
        return False
    _lstm_reg_model  = load_model(reg_path)
    _lstm_reg_feat   = joblib.load(os.path.join(MODELS_DIR, "lstm_reg_feat_scaler.pkl"))
    _lstm_reg_target = joblib.load(os.path.join(MODELS_DIR, "lstm_reg_target_scaler.pkl"))
    return True

# Multi-step regression LSTM — predicts 10 individual daily returns
_lstm_ms_model  = None
_lstm_ms_feat   = None
_lstm_ms_target = None
N_FORECAST_MS   = 10

def _load_ms_model():
    global _lstm_ms_model, _lstm_ms_feat, _lstm_ms_target
    if _lstm_ms_model is not None:
        return True
    ms_path = os.path.join(MODELS_DIR, "lstm_multistep.keras")
    if not os.path.exists(ms_path):
        return False
    _lstm_ms_model  = load_model(ms_path)
    _lstm_ms_feat   = joblib.load(os.path.join(MODELS_DIR, "lstm_ms_feat_scaler.pkl"))
    _lstm_ms_target = joblib.load(os.path.join(MODELS_DIR, "lstm_ms_target_scaler.pkl"))
    return True

print("[INFO] All models loaded")

# 28 features — MUST match the column order produced by merge_dataset.py /
# ml_utils.feature_columns() exactly, because the LSTM scalers are fit in that
# order. Any reordering here silently mis-aligns columns at prediction time.
LSTM_FEATURES = [
    "Close", "High", "Low", "Open", "Volume",
    "sent_mean", "sent_std", "sent_count", "avg_reliability",
    "Return", "MA_5", "MA_10", "Volatility_5", "Momentum_5",
    "Return_lag_1", "Return_lag_2", "RSI_14",
    "MACD", "MACD_signal", "BB_percent", "Volume_Ratio",
    "High_ratio", "Low_ratio",
    "sentiment_lag_1", "sentiment_lag_2",
    "sentiment_3d", "sentiment_7d", "sentiment_momentum",
]

# ==============================
# DATA CACHE (5 min TTL)
# ==============================
_data_cache: dict = {}
CACHE_TTL = 300

def _last_trading_day_before_today():
    """Return the most recent weekday on or before today."""
    d = pd.Timestamp.now().normalize()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def get_cached_data(ticker: str, force_refresh: bool = False) -> pd.DataFrame:
    now = time.time()
    if not force_refresh and ticker in _data_cache:
        df, ts = _data_cache[ticker]
        if now - ts < CACHE_TTL:
            # Also check if data is stale (last row older than last trading day)
            last_date = pd.to_datetime(df["Date"].iloc[-1])
            expected  = _last_trading_day_before_today()
            if last_date >= expected:
                print(f"[CACHE] Hit for {ticker}")
                return df.copy()
            print(f"[CACHE] Data stale ({last_date.date()} < {expected.date()}) — refreshing")
    print(f"[CACHE] Downloading fresh data for {ticker}…")
    end_dt   = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start_dt = (pd.Timestamp.now() - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start_dt, end=end_dt, interval="1d", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    _data_cache[ticker] = (df.copy(), now)
    return df


def invalidate_cache(ticker: str):
    """Force the next call to re-download from yfinance."""
    _data_cache.pop(ticker, None)

# ==============================
# SENTIMENT LOADER
# ==============================
def load_latest_sentiment(ticker: str) -> dict:
    """Return the most recent daily sentiment row for ticker, or all-zeros."""
    zero = {
        "sent_mean": 0.0, "sent_std": 0.0, "sent_count": 0.0,
        "sentiment_lag_1": 0.0, "sentiment_lag_2": 0.0,
        "sentiment_3d": 0.0, "sentiment_7d": 0.0, "sentiment_momentum": 0.0,
        "avg_reliability": 0.0,
    }
    try:
        sent_path = os.path.join(BASE_DIR, "data", "processed", "sentiment_daily.csv")
        if not os.path.exists(sent_path):
            return zero
        sent_df = pd.read_csv(sent_path)
        sent_df["date"] = pd.to_datetime(sent_df["date"])
        ticker_sent = sent_df[sent_df["ticker"] == ticker].sort_values("date")
        if ticker_sent.empty:
            return zero
        row = ticker_sent.iloc[-1]
        mean = float(row.get("sent_mean", 0) or 0)
        std  = float(row.get("sent_std",  0) or 0)
        cnt  = float(row.get("sent_count", 0) or 0)
        rel  = float(row.get("avg_reliability", 0) or 0)
        return {
            "sent_mean":          mean,
            "sent_std":           std,
            "sent_count":         cnt,
            "sentiment_lag_1":    mean,
            "sentiment_lag_2":    mean,
            "sentiment_3d":       mean,
            "sentiment_7d":       mean,
            "sentiment_momentum": 0.0,
            "avg_reliability":    rel,
        }
    except Exception as e:
        print(f"[SENTIMENT] Could not load sentiment for {ticker}: {e}")
        return zero

# ==============================
# LABEL MAP
# ==============================
def label(x):
    return {0: "SELL", 1: "HOLD", 2: "BUY"}[x]

# ==============================
# FEATURE ENGINEERING
# ==============================
def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = (ema_fast - ema_slow) / series  # normalised by price
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def compute_bb_percent(series, window=20):
    sma   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (series - lower) / (upper - lower).replace(0, float("nan"))

def create_features(df):
    df = df.copy()
    df["Return"]       = df["Close"].pct_change()
    df["MA_5"]         = df["Close"].rolling(5).mean()
    df["MA_10"]        = df["Close"].rolling(10).mean()
    df["Volatility_5"] = df["Return"].rolling(5).std()
    df["Momentum_5"]   = df["Close"] - df["Close"].shift(5)
    df["Return_lag_1"] = df["Return"].shift(1)
    df["Return_lag_2"] = df["Return"].shift(2)
    df["RSI_14"]       = compute_rsi(df["Close"])
    macd, macd_sig     = compute_macd(df["Close"])
    df["MACD"]         = macd
    df["MACD_signal"]  = macd_sig
    df["BB_percent"]   = compute_bb_percent(df["Close"])
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["High_ratio"]   = df["High"] / df["Close"] - 1
    df["Low_ratio"]    = 1 - df["Low"] / df["Close"]
    return df.dropna()

def build_lstm_features(df, ticker: str = ""):
    """Build all 28 features for LSTM in correct order."""
    df = df.copy()
    df["Return"]       = df["Close"].pct_change()
    df["MA_5"]         = df["Close"].rolling(5).mean()
    df["MA_10"]        = df["Close"].rolling(10).mean()
    df["Volatility_5"] = df["Return"].rolling(5).std()
    df["Momentum_5"]   = df["Close"] - df["Close"].shift(5)
    df["Return_lag_1"] = df["Return"].shift(1)
    df["Return_lag_2"] = df["Return"].shift(2)
    df["RSI_14"]       = compute_rsi(df["Close"])
    macd, macd_sig     = compute_macd(df["Close"])
    df["MACD"]         = macd
    df["MACD_signal"]  = macd_sig
    df["BB_percent"]   = compute_bb_percent(df["Close"])
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["High_ratio"]   = df["High"] / df["Close"] - 1
    df["Low_ratio"]    = 1 - df["Low"] / df["Close"]

    # Load real sentiment from CSV; fall back to zeros if unavailable
    sent = load_latest_sentiment(ticker) if ticker else {}
    for col in ["sent_mean", "sent_std", "sent_count",
                "sentiment_lag_1", "sentiment_lag_2",
                "sentiment_3d", "sentiment_7d", "sentiment_momentum",
                "avg_reliability"]:
        if col not in df.columns:
            df[col] = sent.get(col, 0.0)

    return df.dropna()

# ==============================
# LSTM CLASSIFICATION
# ==============================
def lstm_classification(df, ticker: str = ""):
    try:
        data = build_lstm_features(df, ticker=ticker)

        # Ensure every expected feature column exists
        for col in LSTM_FEATURES:
            if col not in data.columns:
                data[col] = 0.0

        X = data[LSTM_FEATURES].values

        if len(X) < 10:
            return 1

        n_features = len(LSTM_FEATURES)
        scaled = lstm_cls_scaler.transform(X)
        seq    = scaled[-10:]  # (10, n_features)

        seq_tensor = tf.constant(
            seq.reshape(1, 10, n_features), dtype=tf.float32
        )
        pred = lstm_cls_model(seq_tensor, training=False).numpy()[0]
        if float(np.max(pred)) < 0.40:   # match ensemble operating point
            return 1  # HOLD — confidence below the 0.40 floor
        return int(np.argmax(pred))

    except Exception as e:
        print(f"[LSTM CLS ERROR] {e}")
        return 1

# ==============================
# FORECAST
# Center line: regression LSTM predicted 5-day return (falls back to signal-biased
#              drift when regressor model is not yet trained)
# Volatility:  GARCH(1,1) — heteroscedastic, better than flat EWMA
# Band width:  scaled by ensemble model disagreement
# ==============================
SEQ_LEN_REG = 10   # must match train_lstm_regressor.py

def _garch_vol_forecast(returns: np.ndarray, steps: int) -> np.ndarray:
    """Fit GARCH(1,1) on recent returns, return per-step std-dev array."""
    try:
        from arch import arch_model
        r_pct = pd.Series(returns[-252:] * 100)
        am  = arch_model(r_pct, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        fc  = res.forecast(horizon=steps, reindex=False)
        vol = np.sqrt(fc.variance.values[-1]) / 100
        vol = np.clip(vol, 1e-4, 0.025)   # tightened: max 2.5% daily vol
        return vol.astype(np.float32)
    except Exception:
        lam, var = 0.94, float(np.var(returns[-20:]))
        for r in returns[-20:]:
            var = lam * var + (1 - lam) * r ** 2
        return np.full(steps, min(np.sqrt(var), 0.025), dtype=np.float32)


def _band_scale(model_votes: list) -> float:
    """
    Disagreement-based band multiplier.
    All 5 agree → 0.70 (tight, high conviction).
    3/5 agree   → ~1.00 (normal).
    Maximum entropy → 1.40 (wide, uncertain).
    """
    counts = np.array([model_votes.count(i) for i in range(3)], dtype=float)
    total  = counts.sum()
    if total == 0:
        return 1.0
    p   = counts[counts > 0] / total
    H   = -np.sum(p * np.log(p))          # Shannon entropy
    H_max = np.log(3)                      # max entropy for 3 classes
    disagreement = H / H_max              # [0, 1]
    return float(0.70 + 0.70 * disagreement)   # [0.70, 1.40]


def _reg_lstm_center(df_feat: pd.DataFrame, ticker: str,
                     last_price: float, steps: int,
                     ensemble_signal: int = 1) -> np.ndarray:
    """
    Multi-step LSTM: predicts N_FORECAST individual daily returns.
    Each day gets its own ML-predicted return — zigzag shape comes from
    learned market patterns, not random noise.

    For days beyond N_FORECAST, smoothly fades toward the ensemble signal drift.
    Falls back to single-step regressor, then signal-biased drift if unavailable.
    """
    try:
        if not _load_ms_model():
            return _reg_lstm_center_single(df_feat, ticker, last_price, steps, ensemble_signal)

        data = build_lstm_features(df_feat, ticker=ticker)
        feat_cols = list(_lstm_ms_feat.feature_names_in_)
        for c in feat_cols:
            if c not in data.columns:
                data[c] = 0.0

        seq = data[feat_cols].values[-SEQ_LEN_REG:]
        if len(seq) < SEQ_LEN_REG:
            return None

        seq_scaled  = _lstm_ms_feat.transform(seq)
        X_in        = seq_scaled[np.newaxis, :, :]
        pred_scaled = _lstm_ms_model.predict(X_in, verbose=0)[0]   # shape (N_FORECAST,)
        pred_returns = _lstm_ms_target.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).flatten()  # individual daily returns for days 1..N_FORECAST

        # Hybrid direction: trust magnitude from LSTM, direction from ensemble
        # (per-day direction accuracy ~50-52%; ensemble classification more reliable)
        direction = {0: -1, 1: 0, 2: 1}.get(int(ensemble_signal), 0)
        # Blend: 60% ensemble direction, 40% LSTM own direction — preserves LSTM nuance
        blended = []
        for r in pred_returns:
            lstm_dir = np.sign(r) if r != 0 else 0
            effective_dir = 0.6 * direction + 0.4 * lstm_dir
            blended.append(effective_dir * abs(r))
        blended = np.array(blended, dtype=np.float32)

        # For days beyond N_FORECAST, fade using last predicted daily return
        last_rate  = float(blended[-1]) if len(blended) > 0 else 0.0
        fade_decay = np.exp(-0.1 * np.arange(steps - N_FORECAST_MS))
        extended   = last_rate * fade_decay

        all_daily = np.concatenate([blended, extended])[:steps]
        center = last_price * np.cumprod(1 + all_daily)
        return center.astype(np.float32)

    except Exception as e:
        print(f"[MS LSTM] fallback ({e})")
        return None


def _reg_lstm_center_single(df_feat, ticker, last_price, steps, ensemble_signal):
    """Single-step regressor fallback."""
    try:
        if not _load_reg_model():
            return None
        data = build_lstm_features(df_feat, ticker=ticker)
        feat_cols = list(_lstm_reg_feat.feature_names_in_)
        for c in feat_cols:
            if c not in data.columns:
                data[c] = 0.0
        seq = data[feat_cols].values[-SEQ_LEN_REG:]
        if len(seq) < SEQ_LEN_REG:
            return None
        seq_scaled  = _lstm_reg_feat.transform(seq)
        X_in        = seq_scaled[np.newaxis, :, :]
        pred_scaled = _lstm_reg_model.predict(X_in, verbose=0)[0, 0]
        pred_return = float(_lstm_reg_target.inverse_transform([[pred_scaled]])[0, 0])
        direction   = {0: -1, 1: 0, 2: 1}.get(int(ensemble_signal), 0)
        daily_rate  = direction * abs(pred_return) / 5.0
        decay       = np.exp(-0.04 * np.arange(steps))
        center      = last_price * np.cumprod(1 + daily_rate * decay)
        return center.astype(np.float32)
    except Exception as e:
        print(f"[REG LSTM single] fallback ({e})")
        return None


def forecast_lstm(df, steps=18, simulations=500,
                  ensemble_signal=1, model_votes=None, ticker=""):

    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df = df.dropna()

    returns    = df["Return"].values
    last_price = float(df["Close"].iloc[-1])

    # ── 1. Center line ──────────────────────────────────────────────
    center = _reg_lstm_center(df, ticker, last_price, steps,
                              ensemble_signal=ensemble_signal)

    if center is None:
        # Fallback: signal-biased drift (original method)
        recent_drift  = np.mean(returns[-20:])
        longrun_drift = np.mean(returns)
        base_drift    = np.clip(0.3 * recent_drift + 0.7 * longrun_drift,
                                -0.0005, 0.0005)
        signal_nudge  = {0: -0.0003, 1: 0.0, 2: 0.0003}.get(int(ensemble_signal), 0.0)
        fade          = np.linspace(1.0, 0.0, steps)
        price = last_price
        center = np.zeros(steps, dtype=np.float32)
        for s in range(steps):
            price *= (1 + base_drift + signal_nudge * fade[s])
            center[s] = price

    # ── 2. GARCH per-day volatility ──────────────────────────────────
    garch_vol = _garch_vol_forecast(returns, steps)

    # ── 3. Band scale from model disagreement ────────────────────────
    votes = model_votes or [ensemble_signal] * 5
    scale = _band_scale(votes)

    # ── 4. Confidence bands (80% CI ≈ 1.28σ, scaled by conviction) ──
    z = 1.28 * scale
    cumvol = np.sqrt(np.cumsum(garch_vol ** 2))  # cumulative uncertainty grows
    upper  = center * (1 + z * cumvol)
    lower  = center * (1 - z * cumvol)

    # ── 5. Simulate full path ensemble (500 paths) ──────────────────────────
    np.random.seed(None)
    noise      = np.random.normal(0, 1, size=(simulations, steps)).astype(np.float32)
    all_paths  = np.zeros((simulations, steps), dtype=np.float32)
    prices_all = np.full(simulations, last_price, dtype=np.float32)

    for s in range(steps):
        daily_ret  = (center[s] / (center[s - 1] if s > 0 else last_price)) - 1
        step_noise = noise[:, s] * garch_vol[s] * scale
        step_noise = np.clip(step_noise, -0.06, 0.06)
        prices_all = prices_all * (1 + daily_ret + step_noise)
        all_paths[:, s] = prices_all

    # Center = median of first 25 paths (realistic zigzag, follows LSTM trend)
    center_noisy = np.median(all_paths[:25], axis=0)

    # ── 6. Full Monte Carlo risk analytics ──────────────────────────────────
    final_prices = all_paths[:, -1]
    final_rets   = (final_prices / last_price - 1) * 100   # % returns at horizon

    var_95 = float(np.percentile(final_rets, 5))
    es_95  = float(np.mean(final_rets[final_rets <= var_95]))

    # Max drawdown (50 paths for speed)
    max_dds = []
    for path in all_paths[:50]:
        peak = last_price
        dd   = 0.0
        for p in path:
            peak = max(peak, p)
            dd   = min(dd, (p - peak) / peak)
        max_dds.append(dd * 100)
    avg_max_dd = float(np.mean(max_dds))

    # Return histogram (40 bins, clipped to ±30%)
    h_min = max(float(np.percentile(final_rets, 1)), -30.0)
    h_max = min(float(np.percentile(final_rets, 99)), 30.0)
    counts, edges = np.histogram(final_rets, bins=40, range=(h_min, h_max))
    histogram = [
        {
            "x":       round(float((edges[i] + edges[i + 1]) / 2), 3),
            "count":   int(counts[i]),
            "is_loss": float((edges[i] + edges[i + 1]) / 2) < 0,
        }
        for i in range(len(counts))
    ]

    # Probability estimates
    n = float(len(final_rets))
    probabilities = {
        "profit":  round(float(np.sum(final_rets > 0)) / n * 100, 1),
        "gain_5":  round(float(np.sum(final_rets > 5)) / n * 100, 1),
        "gain_10": round(float(np.sum(final_rets > 10)) / n * 100, 1),
        "loss_5":  round(float(np.sum(final_rets < -5)) / n * 100, 1),
        "loss_10": round(float(np.sum(final_rets < -10)) / n * 100, 1),
    }

    # Scenario price targets (actual dollar prices)
    price_targets = {
        "p5":      round(float(np.percentile(final_prices, 5)),  2),
        "p25":     round(float(np.percentile(final_prices, 25)), 2),
        "p50":     round(float(np.percentile(final_prices, 50)), 2),
        "p75":     round(float(np.percentile(final_prices, 75)), 2),
        "p95":     round(float(np.percentile(final_prices, 95)), 2),
        "current": round(float(last_price), 2),
    }

    # Distribution statistics (numpy only, no scipy needed)
    mean  = float(np.mean(final_rets))
    std   = float(np.std(final_rets))
    norm  = (final_rets - mean) / (std + 1e-8)
    skew  = round(float(np.mean(norm ** 3)), 3)
    kurt  = round(float(np.mean(norm ** 4)) - 3, 3)  # excess kurtosis
    dist_stats = {
        "mean":     round(mean, 2),
        "std":      round(std, 2),
        "skewness": skew,
        "kurtosis": kurt,
    }

    # Time-evolution fan (percentile prices at each forecast step)
    time_evolution = {
        "p5":  [round(float(np.percentile(all_paths[:, s], 5)),  2) for s in range(steps)],
        "p25": [round(float(np.percentile(all_paths[:, s], 25)), 2) for s in range(steps)],
        "p50": [round(float(np.percentile(all_paths[:, s], 50)), 2) for s in range(steps)],
        "p75": [round(float(np.percentile(all_paths[:, s], 75)), 2) for s in range(steps)],
        "p95": [round(float(np.percentile(all_paths[:, s], 95)), 2) for s in range(steps)],
    }

    risk_metrics = {
        "var_95":         round(var_95, 2),
        "es_95":          round(es_95, 2),
        "max_drawdown":   round(avg_max_dd, 2),
        "upside_p75":     round(float(np.percentile(final_rets, 75)), 2),
        "downside_p25":   round(float(np.percentile(final_rets, 25)), 2),
        "histogram":      histogram,
        "probabilities":  probabilities,
        "price_targets":  price_targets,
        "dist_stats":     dist_stats,
        "time_evolution": time_evolution,
    }

    return center_noisy, all_paths, upper, lower, risk_metrics

# ==============================
# MAIN PREDICT
# ==============================
def predict_stock(ticker):
    df = get_cached_data(ticker)
    df = create_features(df)

    # Load real sentiment and inject into the feature row
    sent = load_latest_sentiment(ticker)

    FEATURES = list(rf.feature_names_in_)
    row  = df.iloc[-1]
    data = {}
    for col in FEATURES:
        if col in sent:
            data[col] = sent[col]
        elif col in df.columns:
            try:    data[col] = float(row[col])
            except: data[col] = 0.0
        else:
            data[col] = 0.0

    X = pd.DataFrame([data])

    # Operating point for a 3-class model: 0.40 sits comfortably above the
    # 0.333 uniform-chance baseline (so a call still means "more confident than
    # random") while matching the models' actual probability distribution.
    # A 0.50 floor was mis-calibrated for 3 balanced classes — it suppressed
    # ~70% of actionable calls for ~1 accuracy point (see threshold analysis).
    CONFIDENCE_THRESHOLD = 0.40

    def predict_with_threshold(model, X_input):
        proba = model.predict_proba(X_input)[0]
        return int(np.argmax(proba)) if np.max(proba) >= CONFIDENCE_THRESHOLD else 1

    rf_pred   = predict_with_threshold(rf,        X)
    xgb_pred  = predict_with_threshold(xgb,       X)
    lgbm_pred = predict_with_threshold(lgbm,      X)
    log_pred  = predict_with_threshold(log_model, log_scaler.transform(X))
    lstm_pred = lstm_classification(df, ticker=ticker)

    votes = [rf_pred, xgb_pred, lgbm_pred, log_pred, lstm_pred]
    final = max(set(votes), key=votes.count)

    future, paths, upper, lower, risk_metrics = forecast_lstm(
        df,
        ensemble_signal=int(final),
        model_votes=votes,
        ticker=ticker,
    )

    history = df.tail(90)[["Date", "Close"]].copy()

    return {
        "rf":               label(int(rf_pred)),
        "xgb":              label(int(xgb_pred)),
        "lgbm":             label(int(lgbm_pred)),
        "logistic":         label(int(log_pred)),
        "lstm":             label(int(lstm_pred)),
        "final":            label(int(final)),
        "votes":            votes,
        "history":          history.to_dict(orient="records"),
        "forecast":         future.tolist(),
        "confidence_upper": upper.tolist(),
        "confidence_lower": lower.tolist(),
        "forecast_horizon": len(future),
        "risk_metrics":     risk_metrics,
    }