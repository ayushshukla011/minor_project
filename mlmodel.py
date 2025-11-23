import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)

import matplotlib.pyplot as plt

# ================== CONFIG ==================
import argparse, sys

parser = argparse.ArgumentParser(description="Train noise + synthetic vibration ML models")
parser.add_argument("--data-path", "-d", default="urban_noise_levels.csv",
                    help="Path to input CSV file (default: %(default)s)")
parser.add_argument("--out-dir", "-o", default="ml_with_vibration_models",
                    help="Directory where models/artifacts will be saved (default: %(default)s)")
parser.add_argument("--random-state", type=int, default=42,
                    help="Random seed for reproducibility")
args = parser.parse_args()

DATA_PATH = Path(args.data_path).expanduser()
if not DATA_PATH.exists() or DATA_PATH.is_dir():
    print(f"ERROR: data file does not exist or is a directory: {DATA_PATH}", file=sys.stderr)
    sys.exit(1)
# use resolved absolute path for clarity
DATA_PATH = str(DATA_PATH.resolve())

OUT_DIR = Path(args.out_dir).expanduser().resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = args.random_state
NOISE_THRESHOLDS = (60, 75)  # Low <60, Medium 60–75, High >75
# ===========================================

# 1. Load dataset
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Basic cleaning
df.columns = [c.strip() for c in df.columns]

if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

# Fill numeric missing values with column median
num_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
num_cols = [c for c in num_cols if c not in ("id", "sensor_id")]
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Ensure decibel_level is float
df["decibel_level"] = df["decibel_level"].astype(float)

# ====================================================
# 3. Generate synthetic vibration feature(s)
# ====================================================
# Idea:
#   vibration_level is correlated with:
#       - decibel_level (more noise -> more vibration)
#       - traffic_density
#       - heavy_vehicle_count
#   plus some random noise (to look realistic)
#   Units: synthetic "mm/s" (you can mention this in report)

# Normalize helpers (safe for constant columns)
def safe_minmax(x):
    x = np.asarray(x)
    if np.ptp(x) == 0:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

noise_norm = safe_minmax(df["decibel_level"])

traffic = df["traffic_density"] if "traffic_density" in df.columns else 0
heavy = df["heavy_vehicle_count"] if "heavy_vehicle_count" in df.columns else 0

traffic_norm = safe_minmax(traffic) if not np.isscalar(traffic) else 0
heavy_norm = safe_minmax(heavy) if not np.isscalar(heavy) else 0

rng = np.random.default_rng(RANDOM_STATE)

# Weighted sum of components
noise_component   = 2.0 * noise_norm
traffic_component = 1.5 * traffic_norm
heavy_component   = 2.5 * heavy_norm

random_component  = rng.normal(loc=0.0, scale=0.3, size=len(df))

vibration_level = 0.5 + noise_component + traffic_component + heavy_component + random_component
vibration_level = np.clip(vibration_level, 0.1, None)  # avoid negative

df["vibration_level"] = vibration_level

# Optional: categorical vibration risk for analysis (not used as target here)
# thresholds: <2 low, 2-4 medium, >4 high
df["vibration_risk"] = pd.cut(
    df["vibration_level"],
    bins=[-1, 2, 4, 100],
    labels=["Low", "Medium", "High"]
).astype(str)

print("\nSynthetic vibration_level stats:")
print(df["vibration_level"].describe())
print("\nVibration risk distribution:")
print(df["vibration_risk"].value_counts())

# ====================================================
# 4. Noise risk label (classification target)
# ====================================================
low_th, high_th = NOISE_THRESHOLDS
df["risk_level"] = pd.cut(
    df["decibel_level"],
    bins=[-1, low_th, high_th, 300],
    labels=["Low", "Medium", "High"]
).astype(str)

print("\nNoise risk distribution:")
print(df["risk_level"].value_counts())

# ====================================================
# 5. Time & other feature engineering
# ====================================================
if "datetime" in df.columns:
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)

# cyclical hour features (important for time-of-day periodicity)
if "hour" in df.columns:
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# sensor-level aggregation (how many readings per sensor)
if "sensor_id" in df.columns:
    df["sensor_id"] = df["sensor_id"].astype(str)
    df["sensor_record_count"] = df.groupby("sensor_id")["decibel_level"].transform("count")

# ====================================================
# 6. Feature selection (X) and targets (y)
# ====================================================

candidate_features = [
    "latitude", "longitude",
    "hour", "hour_sin", "hour_cos", "day_of_week", "is_weekend",
    "temperature_c", "humidity_%", "wind_speed_kmh",
    "traffic_density", "vehicle_count", "heavy_vehicle_count",
    "honking_events", "public_event", "holiday",
    "school_zone", "noise_complaints",
    "sensor_record_count",
    "vibration_level"    # <-- synthetic vibration included in the model
]

feature_cols = [c for c in candidate_features if c in df.columns]
print("\nUsing feature columns:")
print(feature_cols)

# Drop rows where targets are missing
df = df.dropna(subset=["decibel_level", "risk_level", "vibration_level"])

X      = df[feature_cols].copy()
y_class = df["risk_level"].copy()      # classification target
y_reg   = df["decibel_level"].copy()   # regression target

# ====================================================
# 7. Train-test split
# ====================================================
X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X, y_class, y_reg,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_class
)

numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ],
    remainder="drop"
)

# ====================================================
# 8. Classification model (RandomForestClassifier)
# ====================================================
clf_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
])

clf_param_dist = {
    "clf__n_estimators": [100, 200, 400],
    "clf__max_depth": [None, 10, 20, 30],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
}

print("\nTuning classifier (RandomizedSearchCV)...")
clf_search = RandomizedSearchCV(
    clf_pipe,
    clf_param_dist,
    n_iter=12,
    cv=3,
    scoring="f1_macro",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)
clf_search.fit(X_train, y_train_class)
best_clf = clf_search.best_estimator_
print("\nBest classifier params:")
print(clf_search.best_params_)

# Evaluation
y_pred = best_clf.predict(X_test)
acc = accuracy_score(y_test_class, y_pred)
f1  = f1_score(y_test_class, y_pred, average="macro")

print("\n=== Classification Performance (Noise Risk) ===")
print("Accuracy:", acc)
print("F1 (macro):", f1)
print("\nClassification report:\n", classification_report(y_test_class, y_pred))

cm = confusion_matrix(y_test_class, y_pred, labels=["Low", "Medium", "High"])
print("Confusion matrix (Low, Medium, High):\n", cm)

# Save classifier
clf_path = OUT_DIR / "noise_vibration_classifier.pkl"
with open(clf_path, "wb") as f:
    pickle.dump(best_clf, f)
print("\nSaved classifier model to:", clf_path)

# ====================================================
# 9. Regression model (RandomForestRegressor)
# ====================================================
reg_pipe = Pipeline([
    ("pre", preprocessor),
    ("reg", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
])

reg_param_dist = {
    "reg__n_estimators": [100, 200, 400],
    "reg__max_depth": [None, 10, 20],
    "reg__min_samples_split": [2, 5, 10],
}

print("\nTuning regressor (RandomizedSearchCV)...")
reg_search = RandomizedSearchCV(
    reg_pipe,
    reg_param_dist,
    n_iter=8,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)
reg_search.fit(X_train, y_train_reg)
best_reg = reg_search.best_estimator_

print("\nBest regressor params:")
print(reg_search.best_params_)

y_pred_reg = best_reg.predict(X_test)
# Compute RMSE in a backward-compatible way (some sklearn versions do not accept `squared` kwarg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2   = r2_score(y_test_reg, y_pred_reg)

print("\n=== Regression Performance (Decibel) ===")
print("RMSE:", rmse)
print("R²:", r2)

# Save regressor
reg_path = OUT_DIR / "noise_vibration_regressor.pkl"
with open(reg_path, "wb") as f:
    pickle.dump(best_reg, f)
print("Saved regressor model to:", reg_path)

# ====================================================
# 10. Feature importance (from classifier)
# ====================================================
clf_model = best_clf.named_steps["clf"]
feature_names = numeric_features
importances = clf_model.feature_importances_

fi = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop feature importances (classifier):")
print(fi.head(15))

# Optional: visualize decibel distribution
plt.figure()
plt.hist(df["decibel_level"].dropna(), bins=40)
plt.title("Distribution of decibel_level")
plt.xlabel("Decibel (dB)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

print("\nArtifacts saved in:", OUT_DIR)
