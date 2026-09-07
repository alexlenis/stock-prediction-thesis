import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_utils import feature_columns, train_test_masks, build_sequences_per_ticker

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_final.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "lstm_scaler.pkl")

SEQUENCE_LENGTH = 10

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

FEATURES = feature_columns(df)
y = df["signal"].replace({-1: 0, 0: 1, 1: 2}).values

# ==============================
# SCALE — fit on TRAIN rows ONLY (no test-set leakage), then transform all
# ==============================
train_mask, _ = train_test_masks(df)

# Fit on a DataFrame (not .values) so the scaler records feature_names_in_,
# keeping the train/predict column order self-documenting and aligned.
scaler = MinMaxScaler()
scaler.fit(df.loc[train_mask, FEATURES])
X_scaled = scaler.transform(df[FEATURES])

joblib.dump(scaler, SCALER_PATH)

# ==============================
# CREATE SEQUENCES — built WITHIN each ticker (never crossing boundaries),
# assigned to train/test by the label row's chronological is_train flag.
# ==============================
X_train, y_train, X_test, y_test = build_sequences_per_ticker(
    df, X_scaled, y, SEQUENCE_LENGTH
)
y_train = y_train.astype(int)
y_test  = y_test.astype(int)

print(f"[INFO] Sequences — train: {X_train.shape}, test: {X_test.shape}")

# ==============================
# 🔥 CLASS WEIGHTS (FIX)
# ==============================
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)

class_weights = dict(zip(classes, weights))

print("Class weights:", class_weights)

# ==============================
# MODEL
# ==============================
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# TRAIN (BALANCED 🔥)
# ==============================
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=32,
    class_weight=class_weights,   # 🔥 ΤΟ FIX
    callbacks=[early_stop],
    verbose=1
)

# ==============================
# PREDICT
# ==============================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ==============================
# EVALUATE
# ==============================
print("\n🔥 Balanced LSTM Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# ==============================
# SAVE
# ==============================
model.save(MODEL_PATH)

print(f"[DONE] Model saved to {MODEL_PATH}")