import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_final.keras")

print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH)
print("Model loaded OK")

print("\nDownloading AAPL data...")
df = yf.download("AAPL", period="6mo", interval="1d")
if hasattr(df.columns, "levels"):
    df.columns = df.columns.get_level_values(0)

FEAT = ["Close", "High", "Low", "Open", "Volume"]
data = df[FEAT].dropna().values
print("Data shape:", data.shape)

sc     = MinMaxScaler()
scaled = sc.fit_transform(data)
print("Scaled shape:", scaled.shape)
print("Last row scaled:", scaled[-1])

print("\n--- Current prediction ---")
seq = tf.constant(scaled[-10:].reshape(1, 10, 5), dtype=tf.float32)
out = model(seq, training=False).numpy()[0]
print("Softmax output:", out)
print("Predicted class:", np.argmax(out), "→", {0:"SELL",1:"HOLD",2:"BUY"}[np.argmax(out)])
print("Confidence:", round(float(np.max(out)) * 100, 1), "%")

print("\n--- Last 10 rolling predictions ---")
n = len(scaled)
label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
for i in range(10, 0, -1):
    end_idx = n - i
    if end_idx < 10:
        print(f"  i={i}: skipped (end_idx={end_idx} < 10)")
        continue
    s   = tf.constant(scaled[end_idx-10:end_idx].reshape(1,10,5), dtype=tf.float32)
    p   = model(s, training=False).numpy()[0]
    cls = int(np.argmax(p))
    conf= round(float(np.max(p))*100, 1)
    print(f"  i={i}: end_idx={end_idx} → {label_map[cls]} ({conf}%)")

print("\nDone.")