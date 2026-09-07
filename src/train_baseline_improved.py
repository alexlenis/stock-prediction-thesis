import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_utils import feature_columns, train_test_masks

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
# SCALING
# ==============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# SAMPLE WEIGHTS
# ==============================
base_weights   = compute_sample_weight("balanced", y_train)
sent_boost     = np.where(X_train["sent_mean"].values != 0, 1.5, 1.0)
sample_weights = base_weights * sent_boost

# ==============================
# MODEL
# ==============================
model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# ==============================
# PREDICT
# ==============================
y_pred = model.predict(X_test_scaled)

# ==============================
# EVALUATION
# ==============================
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n🔥 Balanced Logistic Regression Results")
print("-" * 50)
print("Accuracy:", accuracy)
print(report)

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, os.path.join(MODELS_DIR, "logistic_final.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "logistic_scaler.pkl"))

# ==============================
# CONFUSION MATRIX
# ==============================
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Balanced Logistic Regression Confusion Matrix")
plt.savefig(os.path.join(FIGURES_DIR, "logistic_confusion_matrix.png"))
plt.close()

# ==============================
# COEFFICIENT IMPORTANCE 🔥
# ==============================
coef_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": abs(model.coef_[0])
}).sort_values(by="Importance", ascending=False)

print("\nTop Features:")
print(coef_df.head(10))