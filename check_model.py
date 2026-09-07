import sys
import numpy as np
import yfinance as yf
import pandas as pd

sys.path.insert(0, '.')
from src.predict_engine import lstm_reg_model, lstm_reg_scaler, create_features

# ── Architecture ─────────────────────────────────────────────────────
print("=" * 50)
print("MODEL SUMMARY")
print("=" * 50)
lstm_reg_model.summary()

# ── Accuracy test ─────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("ACCURACY TEST ON AAPL 1Y")
print("=" * 50)

df = yf.download('AAPL', period='1y', interval='1d')
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
df = create_features(df)
df['Return'] = df['Close'].pct_change()
df = df.dropna()

returns = df['Return'].values.reshape(-1, 1)
scaled  = lstm_reg_scaler.transform(returns)

SEQ_LEN = 20
X, y = [], []
for i in range(SEQ_LEN, len(scaled) - 1):
    X.append(scaled[i - SEQ_LEN:i])
    y.append(scaled[i])

X = np.array(X)
y = np.array(y)

pred     = lstm_reg_model.predict(X, verbose=0)
pred_r   = lstm_reg_scaler.inverse_transform(pred)
y_r      = lstm_reg_scaler.inverse_transform(y)

mae      = np.mean(np.abs(pred_r - y_r))
mean_act = np.mean(np.abs(y_r))
skill    = mae / mean_act

print(f"MAE (real returns) : {mae:.6f}")
print(f"Mean actual move   : {mean_act:.6f}")
print(f"Skill ratio        : {skill:.2f}  (1.0=random, <1.0=better than random)")