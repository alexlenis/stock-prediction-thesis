import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_utils import feature_columns, train_test_masks

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_sample_weight

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
y = df["signal"].replace({-1: 0, 0: 1, 1: 2})

# ==============================
# SPLIT — per-ticker chronological hold-out stored in the dataset.
# Genuinely out-of-sample for every ticker; identical across all models.
# ==============================
train_mask, test_mask = train_test_masks(df)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# ==============================
# SAMPLE WEIGHTS
# class-balance weight × sentiment-availability boost (1.5x for rows with real sentiment)
# ==============================
base_weights  = compute_sample_weight("balanced", y_train)
sent_boost    = np.where(X_train["sent_mean"].values != 0, 1.5, 1.0)
sample_weights = base_weights * sent_boost

# ==============================
# MODEL
# ==============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# ==============================
# PREDICT
# ==============================
y_pred = model.predict(X_test)

# ==============================
# EVALUATION
# ==============================
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n🔥 Balanced Random Forest Results")
print("-" * 50)
print("Accuracy:", accuracy)
print(report)

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, os.path.join(MODELS_DIR, "rf_final.pkl"))

# ==============================
# CONFUSION MATRIX
# ==============================
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Balanced Random Forest Confusion Matrix")
plt.savefig(os.path.join(FIGURES_DIR, "rf_confusion_matrix.png"))
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
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "rf_feature_importance.png"))
plt.close()

print("\nTop Features:")
print(importance_df.head(10))