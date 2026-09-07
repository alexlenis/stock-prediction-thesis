import sys
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, '.')
from src.predict_engine import rf, xgb, lgbm, log_model, log_scaler, create_features

print("Downloading AAPL 1 year data...")
df = yf.download("AAPL", period="1y", interval="1d")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()
df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
df = create_features(df)

# ── RSI ───────────────────────────────────────────────────────────────
def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

df["RSI_14"] = compute_rsi(df["Close"])

# ── Sentiment features — fill with 0 (no live sentiment here) ────────
for col in ["sentiment_score","sent_mean","sent_std","sent_count",
            "sentiment_lag_1","sentiment_lag_2",
            "sentiment_3d","sentiment_7d","sentiment_momentum"]:
    df[col] = 0.0

df = df.dropna()

# ── Target ────────────────────────────────────────────────────────────
df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
df["signal"] = df["future_return"].apply(
    lambda x: 2 if x > 0.02 else (0 if x < -0.02 else 1)
)
df = df.dropna()

# ── Build X with exact model features ────────────────────────────────
FEATURES = list(rf.feature_names_in_)
print(f"Model expects {len(FEATURES)} features: {FEATURES}")

# Add any missing features as 0
for col in FEATURES:
    if col not in df.columns:
        df[col] = 0.0

X = df[FEATURES].fillna(0)
y = df["signal"]

split      = int(len(X) * 0.8)
X_test     = X.iloc[split:]
y_test     = y.iloc[split:]

print(f"\nTest samples: {len(X_test)}")
print("=" * 50)

models = {
    "Random Forest": rf,
    "XGBoost":       xgb,
    "LightGBM":      lgbm,
}

for name, model in models.items():
    pred = model.predict(X_test)
    acc  = accuracy_score(y_test, pred)

    buy_mask = pred == 2
    win_rate = float((y_test[buy_mask] == 2).mean()) if buy_mask.sum() > 0 else 0.0

    try:
        proba      = model.predict_proba(X_test)
        confidence = float(np.mean(np.max(proba, axis=1)))
    except:
        confidence = 0.0

    try:
        importances = model.feature_importances_
        top3_idx    = np.argsort(importances)[::-1][:3]
        top3        = [(FEATURES[i], round(float(importances[i]), 3)) for i in top3_idx]
    except:
        top3 = []

    print(f"\n{name}")
    print(f"  Accuracy:    {acc:.1%}")
    print(f"  Win Rate:    {win_rate:.1%}")
    print(f"  Confidence:  {confidence:.1%}")
    print(f"  Top factors: {top3}")

# ── Logistic ──────────────────────────────────────────────────────────
print("\nLogistic Regression")
X_test_scaled = log_scaler.transform(X_test)
pred_log      = log_model.predict(X_test_scaled)
acc_log       = accuracy_score(y_test, pred_log)
buy_mask      = pred_log == 2
win_log       = float((y_test[buy_mask] == 2).mean()) if buy_mask.sum() > 0 else 0.0
try:
    proba_log = log_model.predict_proba(X_test_scaled)
    conf_log  = float(np.mean(np.max(proba_log, axis=1)))
except:
    conf_log  = 0.0
print(f"  Accuracy:    {acc_log:.1%}")
print(f"  Win Rate:    {win_log:.1%}")
print(f"  Confidence:  {conf_log:.1%}")

# ── Last 10 predictions ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("LAST 10 PREDICTIONS vs ACTUAL (RF)")
last10_X    = X.iloc[-10:]
last10_y    = y.iloc[-10:]
last10_pred = rf.predict(last10_X)
label_map   = {0: "SELL", 1: "HOLD", 2: "BUY"}
for i, (p, a) in enumerate(zip(last10_pred, last10_y)):
    correct = "✅" if p == a else "❌"
    print(f"  Day -{10-i}: Pred={label_map[p]:4} Actual={label_map[a]:4} {correct}")