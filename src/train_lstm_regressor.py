"""
Regression LSTM: predicts the 5-day forward price return (continuous value).

This model is used as the center-line of the forecast chart.
Architecture mirrors the classification LSTM but outputs a single float
with linear activation and Huber loss (robust to financial outliers).
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_utils import feature_columns, train_test_masks, build_sequences_per_ticker

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# PATHS
# ==============================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "lstm_regressor.keras")
FEAT_SCALER = os.path.join(BASE_DIR, "models", "lstm_reg_feat_scaler.pkl")
TGT_SCALER  = os.path.join(BASE_DIR, "models", "lstm_reg_target_scaler.pkl")

SEQUENCE_LENGTH = 10

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

FEATURES = feature_columns(df)

X_df = df[FEATURES]                       # keep as DataFrame for feature_names_in_
y = df["future_return"].values            # raw 5-day return, e.g. 0.032 = +3.2%

train_mask, _ = train_test_masks(df)

# ==============================
# SCALE — fit scalers on TRAIN rows ONLY (no test-set leakage).
# Fit on a DataFrame so feature_names_in_ is recorded (predict_engine reads it).
# ==============================
feat_scaler = MinMaxScaler()
feat_scaler.fit(X_df[train_mask])
X_scaled = feat_scaler.transform(X_df)
joblib.dump(feat_scaler, FEAT_SCALER)

# Target scaled with RobustScaler (handles outlier return days well)
tgt_scaler = RobustScaler()
tgt_scaler.fit(y[train_mask].reshape(-1, 1))
y_scaled = tgt_scaler.transform(y.reshape(-1, 1)).flatten()
joblib.dump(tgt_scaler, TGT_SCALER)

# ==============================
# CREATE SEQUENCES — within each ticker, split by label row's is_train flag
# ==============================
X_train, y_train, X_test, y_test = build_sequences_per_ticker(
    df, X_scaled, y_scaled, SEQUENCE_LENGTH
)

# ==============================
# MODEL
# ==============================
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, X_train.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="linear"),   # single float output
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.Huber(delta=1.0),  # robust to large return outliers
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
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)

# ==============================
# EVALUATE
# ==============================
y_pred_scaled = model.predict(X_test, verbose=0).flatten()
y_pred = tgt_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Direction accuracy: did the model get the sign right?
direction_acc = np.mean(np.sign(y_pred) == np.sign(y_true))

print(f"\nRegression LSTM Results")
print(f"  MAE:               {mae:.4f}  ({mae*100:.2f}% avg error on 5-day return)")
print(f"  RMSE:              {rmse:.4f}")
print(f"  Direction accuracy: {direction_acc:.3f}  (random baseline = 0.50)")

# ==============================
# SAVE
# ==============================
model.save(MODEL_PATH)
print(f"\n[DONE] Model saved to {MODEL_PATH}")
print(f"[DONE] Scalers saved to models/")
