import pickle
import pandas as pd

# Load model
bundle = pickle.load(open("monument_impact_models/monument_impact_model.pkl", "rb"))
model = bundle["model"]
feature_cols = bundle["feature_cols"]

# Example: one reading for a specific monument
sample = {
    # monument metadata (real values in deployment)
    "material": "sandstone",
    "age_years": 350,

    # from sensors / environment in real-time
    "decibel_level": 82.0,
    "vibration_level": 3.8,
    "temperature_c": 32.0,
    "humidity_%": 50.0,
    "wind_speed_kmh": 5.0,
    "traffic_density": 80,
    "vehicle_count": 120,
    "heavy_vehicle_count": 10,
    "honking_events": 4,
    "public_event": 0,
    "holiday": 0,
    "school_zone": 0,
    "noise_complaints": 2,
    "hour": 14,
    "day_of_week": 2,
    "is_weekend": 0,
}

# Ensure we only use columns the model expects
row = {col: sample.get(col, 0) for col in feature_cols}
X_new = pd.DataFrame([row])

predicted_impact = model.predict(X_new)[0]
print("Predicted impact level on monument:", predicted_impact)
