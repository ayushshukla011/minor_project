import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# ---------------- CONFIG ----------------
DATA_PATH = "urban_noise_levels.csv"      # path to kaggle dataset
OUT_DIR = Path("monument_impact_models")  # where to save model
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
# ----------------------------------------


# ---------- 1. Load & basic clean ----------
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
print("Shape:", df.shape)

# ensure numeric
df["decibel_level"] = pd.to_numeric(df["decibel_level"], errors="coerce")

# parse datetime if present
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# fill numeric missing values
num_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
num_cols = [c for c in num_cols if c not in ("id", "sensor_id")]
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ---------- 2. Synthetic vibration (if you don’t already have real vibration) ----------
def safe_minmax(x):
    x = np.asarray(x)
    if np.ptp(x) == 0:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

if "vibration_level" not in df.columns:
    noise_norm = safe_minmax(df["decibel_level"])
    traffic = df["traffic_density"] if "traffic_density" in df.columns else np.zeros(len(df))
    heavy   = df["heavy_vehicle_count"] if "heavy_vehicle_count" in df.columns else np.zeros(len(df))

    traffic_norm = safe_minmax(traffic)
    heavy_norm   = safe_minmax(heavy)

    rng = np.random.default_rng(RANDOM_STATE)
    random_comp = rng.normal(0, 0.3, size=len(df))

    vibration_level = 0.5 + 2.0 * noise_norm + 1.5 * traffic_norm + 2.5 * heavy_norm + random_comp
    vibration_level = np.clip(vibration_level, 0.1, None)

    df["vibration_level"] = vibration_level
    print("Synthetic vibration_level generated.")

# ---------- 3. Add synthetic monument metadata (for training) ----------
# In real deployment, you will provide real material + age per monument.

materials = ["sandstone", "marble", "granite", "brick", "concrete"]

rng = np.random.default_rng(RANDOM_STATE)
df["material"] = rng.choice(materials, size=len(df), p=[0.25, 0.15, 0.20, 0.25, 0.15])
# ages between 50 and 400 years
df["age_years"] = rng.integers(50, 401, size=len(df))

# ---------- 4. Define rule-based impact function (for generating labels) ----------

def material_factor(material: str) -> float:
    m = material.lower()
    highly_sensitive = ["sandstone", "marble", "limestone", "old brick", "terracotta"]
    medium = ["stone", "brick", "wood"]
    low = ["granite", "concrete", "reinforced concrete", "steel", "metal"]

    if any(k in m for k in highly_sensitive):
        return 1.3
    if any(k in m for k in medium):
        return 1.1
    if any(k in m for k in low):
        return 0.9
    return 1.0

def age_factor(age_years: int) -> float:
    if age_years >= 300:
        return 1.3
    elif age_years >= 100:
        return 1.15
    elif age_years >= 50:
        return 1.0
    else:
        return 0.9

def noise_factor(decibel: float) -> float:
    if decibel < 60:
        return 0.5
    elif decibel < 75:
        return 1.0
    elif decibel <= 85:
        return 1.3
    else:
        return 1.6

def vibration_factor(vib: float) -> float:
    if vib < 1.5:
        return 0.7
    elif vib < 3.0:
        return 1.0
    elif vib < 5.0:
        return 1.3
    else:
        return 1.6

def score_to_impact_label(score: float) -> str:
    if score < 0.9:
        return "Low impact"
    elif score < 1.2:
        return "Moderate impact"
    elif score < 1.5:
        return "High impact"
    else:
        return "Severe impact"

def compute_impact_row(row) -> str:
    m_fac = material_factor(row["material"])
    a_fac = age_factor(row["age_years"])
    n_fac = noise_factor(row["decibel_level"])
    v_fac = vibration_factor(row["vibration_level"])

    impact_score = 0.4 * n_fac + 0.4 * v_fac + 0.2 * (m_fac * a_fac)
    return score_to_impact_label(impact_score)

# compute labels
df["impact_level"] = df.apply(compute_impact_row, axis=1)
print("\nImpact level distribution:")
print(df["impact_level"].value_counts())

# ---------- 5. Build feature matrix X and target y ----------
feature_cols = [
    "decibel_level",
    "vibration_level",
    "temperature_c",
    "humidity_%",
    "wind_speed_kmh",
    "traffic_density",
    "vehicle_count",
    "heavy_vehicle_count",
    "honking_events",
    "public_event",
    "holiday",
    "school_zone",
    "noise_complaints",
    "hour",
    "day_of_week",
    "is_weekend",
    "material",
    "age_years",
]

# keep only columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]
print("\nUsing features:", feature_cols)

df = df.dropna(subset=["impact_level"] + feature_cols)

X = df[feature_cols].copy()
y = df["impact_level"].copy()

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# numeric vs categorical
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ---------- 6. Preprocessor ----------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ---------- 7. Classifier + hyperparameter search ----------
clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", clf)
])

param_dist = {
    "clf__n_estimators": [200, 400, 800],
    "clf__max_depth": [None, 10, 20, 40],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv,
    scoring="f1_macro",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

print("\nStarting RandomizedSearchCV for impact classifier...")
search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Best params:", search.best_params_)

# ---------- 8. Evaluation ----------
y_pred = best_model.predict(X_test)

print("\n=== Impact Classification Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 (macro):", f1_score(y_test, y_pred, average="macro"))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred,
                                              labels=["Low impact","Moderate impact","High impact","Severe impact"]))

# ---------- 9. Save model ----------
model_path = OUT_DIR / "monument_impact_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "model": best_model,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_cols": feature_cols,
    }, f)

print("\nSaved monument impact model to:", model_path)
