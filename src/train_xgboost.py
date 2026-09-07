import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_utils import feature_columns, train_test_masks

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)

# ==============================
# FEATURES
# ==============================
FEATURES = feature_columns(df)

X = df[FEATURES]
y = df["signal"]

# ==============================
# LABEL FIX
# ==============================
y = y.replace({-1: 0, 0: 1, 1: 2})

# ==============================
# SPLIT — per-ticker chronological hold-out stored in the dataset.
# Genuinely out-of-sample for every ticker; identical across all models.
# ==============================
train_mask, test_mask = train_test_masks(df)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# ==============================
# 🔥 CLASS WEIGHTS
# ==============================
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)

class_weights = dict(zip(classes, weights))

sample_weights = y_train.map(class_weights)

# Sentiment-availability boost: rows that have real sentiment data get 1.5x weight
sent_boost     = np.where(X_train["sent_mean"].values != 0, 1.5, 1.0)
sample_weights = sample_weights.values * sent_boost

# ==============================
# MODEL
# ==============================
model = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    num_class=3,
    random_state=42
)

# 🔥 ΕΔΩ ΕΙΝΑΙ ΤΟ FIX
model.fit(X_train, y_train, sample_weight=sample_weights)

# ==============================
# PREDICTIONS
# ==============================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# ==============================
# EVALUATION
# ==============================
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n🔥 Balanced XGBoost Results")
print("-" * 50)
print("Accuracy:", accuracy)
print(report)

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_final.pkl"))

# ==============================
# CONFUSION MATRIX
# ==============================
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Balanced XGBoost Confusion Matrix")
plt.savefig(os.path.join(FIGURES_DIR, "xgb_confusion_matrix.png"))
plt.close()

# ==============================
# FEATURE IMPORTANCE
# ==============================
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "xgb_feature_importance.png"))
plt.close()

print("\n📊 Top Features:")
print(importance_df.head(10))

# ==============================
# DECISION ENGINE
# ==============================
def decision(prob):
    if prob[2] > 0.6:
        return "BUY"
    elif prob[0] > 0.6:
        return "SELL"
    else:
        return "HOLD"

decisions = [decision(p) for p in y_proba]

# ==============================
# RESULTS
# ==============================
results_df = pd.DataFrame({
    "Prediction": y_pred,
    "Prob_SELL": y_proba[:, 0],
    "Prob_HOLD": y_proba[:, 1],
    "Prob_BUY": y_proba[:, 2],
    "Decision": decisions
})

print("\n📌 Sample Predictions:")
print(results_df.head(10))

results_path = os.path.join(BASE_DIR, "results", "xgb_predictions.csv")
results_df.to_csv(results_path, index=False)

print(f"\n[DONE] Results saved to {results_path}")