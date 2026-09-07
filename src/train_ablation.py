"""
Ablation study: does NLP sentiment actually improve prediction?

This isolates the contribution of the sentiment/NLP features — the central
claim of the thesis. It trains the SAME XGBoost model twice on the SAME
chronological split, once WITH the 9 sentiment features and once WITHOUT,
and reports the out-of-sample difference.

Methodology that makes the comparison fair and honest:
  * Restricted to 2024-01-01 onward, the period where sentiment data exists
    (before 2024 every sentiment feature is 0, so including it would dilute
    the effect and understate sentiment's true value).
  * Per-ticker chronological 80/20 split WITHIN this window (no leakage,
    every ticker present in both train and test).
  * Identical hyperparameters, class balancing, and features otherwise.

Outputs: results/ablation_study.csv  +  results/figures/ablation_study.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_utils import feature_columns, SENTIMENT_FEATURES

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

SENTIMENT_START = "2024-01-01"   # sentiment data begins here

# ==============================
# LOAD + RESTRICT TO SENTIMENT-ACTIVE PERIOD
# ==============================
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= SENTIMENT_START].sort_values(["ticker", "date"]).reset_index(drop=True)

print(f"[INFO] Ablation window: {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"[INFO] Rows: {len(df)}  |  rows with real sentiment: "
      f"{(df['sent_mean'] != 0).sum()} ({100*(df['sent_mean']!=0).mean():.1f}%)")

ALL_FEATURES  = feature_columns(df)
TECH_FEATURES = [f for f in ALL_FEATURES if f not in SENTIMENT_FEATURES]

y = df["signal"].replace({-1: 0, 0: 1, 1: 2})

# ==============================
# PER-TICKER CHRONOLOGICAL 80/20 SPLIT WITHIN THIS WINDOW
# ==============================
train_idx, test_idx = [], []
for _, grp in df.groupby("ticker", sort=False):
    cut = int(len(grp) * 0.8)
    train_idx.extend(grp.index[:cut])
    test_idx.extend(grp.index[cut:])
train_idx, test_idx = np.array(train_idx), np.array(test_idx)
print(f"[INFO] Train rows: {len(train_idx)}  |  Test rows: {len(test_idx)}")


def train_and_eval(features, label):
    X = df[features]
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    sample_weights = compute_sample_weight("balanced", y_train)

    model = XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob", num_class=3, random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    pred = model.predict(X_test)
    acc  = accuracy_score(y_test, pred)
    f1m  = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"  {label:24s}: accuracy={acc*100:5.2f}%   macro-F1={f1m:.4f}   "
          f"({len(features)} features)")
    return acc, f1m


print("\n=== ABLATION: sentiment contribution (XGBoost, out-of-sample) ===")
acc_with,  f1_with  = train_and_eval(ALL_FEATURES,  "WITH sentiment")
acc_tech,  f1_tech  = train_and_eval(TECH_FEATURES, "WITHOUT sentiment")

acc_gain = (acc_with - acc_tech) * 100
f1_gain  = (f1_with  - f1_tech)
rel_gain = (acc_gain / (acc_tech * 100)) * 100 if acc_tech else 0.0

print("\n--- RESULT ---")
print(f"  Accuracy gain from sentiment: {acc_gain:+.2f} pts  ({rel_gain:+.1f}% relative)")
print(f"  Macro-F1 gain from sentiment: {f1_gain:+.4f}")
print(f"  Verdict: sentiment "
      f"{'IMPROVES' if acc_gain > 0 else 'does NOT improve'} out-of-sample accuracy.")

# ==============================
# SAVE RESULTS
# ==============================
res = pd.DataFrame([
    {"model": "WITH sentiment",    "n_features": len(ALL_FEATURES),
     "accuracy_pct": round(acc_with*100, 2),  "macro_f1": round(f1_with, 4)},
    {"model": "WITHOUT sentiment", "n_features": len(TECH_FEATURES),
     "accuracy_pct": round(acc_tech*100, 2),  "macro_f1": round(f1_tech, 4)},
])
res.to_csv(os.path.join(RESULTS_DIR, "ablation_study.csv"), index=False)

# Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
labels = ["WITH\nsentiment", "WITHOUT\nsentiment"]
colors = ["#22c55e", "#94a3b8"]
ax1.bar(labels, [acc_with*100, acc_tech*100], color=colors)
ax1.set_title("Out-of-sample Accuracy (%)"); ax1.set_ylim(0, max(acc_with, acc_tech)*100 + 8)
for i, v in enumerate([acc_with*100, acc_tech*100]):
    ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
ax2.bar(labels, [f1_with, f1_tech], color=colors)
ax2.set_title("Macro-F1"); ax2.set_ylim(0, max(f1_with, f1_tech) + 0.1)
for i, v in enumerate([f1_with, f1_tech]):
    ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")
fig.suptitle(f"Sentiment Ablation ({df['date'].min().date()} to {df['date'].max().date()})",
             fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ablation_study.png"), dpi=120)
plt.close()

print(f"\n[DONE] Saved results/ablation_study.csv and results/figures/ablation_study.png")
