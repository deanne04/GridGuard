
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "gridguard_isolation_forest.joblib"
SAMPLE_PATH = (
    ROOT / "results" / "anomaly_detection"
    / "training_sample_anomalies.csv"
)

CLIENT = "MT_362"
TARGET = "2013-08-17 16:00:00"

bundle = joblib.load(MODEL_PATH)
sample = pd.read_csv(SAMPLE_PATH)

row = sample[
    (sample["client_id"].astype(str) == CLIENT)
    & (sample["hour_timestamp"].astype(str) == TARGET)
]

if row.empty:
    print("Training example not found.")
    raise SystemExit(1)

row = row.iloc[0]
features = bundle["feature_columns"]

print("\n=== SAVED TRAINING FEATURES ===")
for name in features:
    print(f"{name}: {row[name]}")

X = np.array([[float(row[name]) for name in features]], dtype=float)
scaled = bundle["scaler"].transform(X)

prediction = int(bundle["model"].predict(scaled)[0])
score = float(-bundle["model"].score_samples(scaled)[0])

print("\n=== MODEL RECHECK ===")
print("Prediction:", "Potential Anomaly" if prediction == -1 else "Normal")
print(f"Calculated anomaly score: {score:.4f}")
print(f"CSV anomaly score: {float(row['anomaly_score']):.4f}")
print(f"CSV potential_anomaly: {int(row['potential_anomaly'])}")

print("\nDiagnostic complete. No model files were changed.")
