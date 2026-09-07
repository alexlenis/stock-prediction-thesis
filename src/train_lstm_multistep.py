"""
Multi-step regression LSTM: predicts the next 10 individual daily returns.

Each output neuron = one future trading day's return.
This gives the forecast a shape grounded in learned market patterns rather
than random noise, making it as close as possible to plausible future paths.

Output shape: (N_FORECAST,) — one return per future trading day.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_utils import feature_columns, build_sequences_per_ticker

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# CONFIG
# ==============================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "lstm_multistep.keras")
FEAT_SCALER = os.path.join(BASE_DIR, "models", "lstm_ms_feat_scaler.pkl")
TGT_SCALER  = os.path.join(BASE_DIR, "models", "lstm_ms_target_scaler.pkl")

SEQUENCE_LENGTH = 10   # lookback window
N_FORECAST      = 10   # predict this many trading days ahead

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

FEATURES = feature_columns(df)

# Build per-ticker multi-step targets using the Close column.
# For each row i: target[k] = Close[i+k] / Close[i] - 1, k=1..N_FORECAST
print("[INFO] Building multi-step targets...")

close = df["Close"].values
tick  = df["ticker"].values
targets = []
valid_mask = []

for i in range(len(df)):
    # Need N_FORECAST future rows, all within the same ticker
    if i + N_FORECAST >= len(df) or len(set(tick[i : i + N_FORECAST + 1])) != 1:
        targets.append(np.zeros(N_FORECAST)); valid_mask.append(False); continue

    base_price = close[i]
    if base_price == 0:
        targets.append(np.zeros(N_FORECAST)); valid_mask.append(False); continue

    targets.append([close[i + k] / base_price - 1 for k in range(1, N_FORECAST + 1)])
    valid_mask.append(True)

targets    = np.array(targets, dtype=np.float32)
valid_mask = np.array(valid_mask)

# Keep only valid rows; the filtered frame stays sorted by ticker,date so
# the shared per-ticker sequence builder can group it correctly.
dfv     = df[valid_mask].reset_index(drop=True)
X_raw_df = dfv[FEATURES]                       # DataFrame for feature_names_in_
y_raw   = targets[valid_mask]
train_mask_v = dfv["is_train"].astype(bool).values

print(f"[INFO] {len(X_raw_df)} valid rows after filtering  (train={train_mask_v.sum()}).")

# ==============================
# SCALE — fit scalers on TRAIN rows ONLY (no test-set leakage).
# Fit on a DataFrame so feature_names_in_ is recorded (predict_engine reads it).
# ==============================
feat_scaler = MinMaxScaler()
feat_scaler.fit(X_raw_df[train_mask_v])
X_scaled = feat_scaler.transform(X_raw_df)
joblib.dump(feat_scaler, FEAT_SCALER)

# All forecast-step targets share one RobustScaler, fit on train rows only
tgt_scaler = RobustScaler()
tgt_scaler.fit(y_raw[train_mask_v].reshape(-1, 1))
y_scaled = tgt_scaler.transform(y_raw.reshape(-1, 1)).reshape(y_raw.shape)
joblib.dump(tgt_scaler, TGT_SCALER)

# ==============================
# CREATE SEQUENCES — within each ticker, split by label row's is_train flag
# ==============================
X_train, y_train, X_test, y_test = build_sequences_per_ticker(
    dfv, X_scaled, y_scaled, SEQUENCE_LENGTH
)
print(f"[INFO] Sequences — train: {X_train.shape}, test: {X_test.shape}")

# ==============================
# MODEL
# ==============================
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, X_train.shape[2])),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(N_FORECAST, activation="linear"),   # one output per future day
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=["mae"],
)

model.summary()

# ==============================
# TRAIN
# ==============================
early_stop = EarlyStopping(patience=7, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=60,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1,
)

# ==============================
# EVALUATE
# ==============================
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = tgt_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).reshape(y_pred_scaled.shape)
y_true = tgt_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)

per_day_mae = [mean_absolute_error(y_true[:, k], y_pred[:, k]) for k in range(N_FORECAST)]
direction_acc = [
    np.mean(np.sign(y_pred[:, k]) == np.sign(y_true[:, k]))
    for k in range(N_FORECAST)
]

print(f"\nMulti-step LSTM Results:")
print(f"{'Day':>4}  {'MAE':>8}  {'Direction Acc':>14}")
print("-" * 32)
for k in range(N_FORECAST):
    print(f"  {k+1:>2}  {per_day_mae[k]*100:>7.2f}%  {direction_acc[k]:>14.3f}")
print(f"\nMean MAE:       {np.mean(per_day_mae)*100:.2f}%")
print(f"Mean Dir Acc:   {np.mean(direction_acc):.3f}  (random baseline = 0.50)")

# ==============================
# SAVE
# ==============================
model.save(MODEL_PATH)
print(f"\n[DONE] Model saved to {MODEL_PATH}")
