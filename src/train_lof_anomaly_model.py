
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "results" / "anomaly_detection" / "training_sample_anomalies.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results" / "anomaly_detection" / "lof_training"

FEATURES = [
    "average_kw",
    "within_hour_std_kw",
    "previous_hour_kw",
    "previous_24h_kw",
    "previous_168h_kw",
    "rolling_mean_24h",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42


def main():
    start = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("GridGuard - LOF Anomaly Model Training")
    print("=" * 45)
    print("Loading the existing historical feature sample...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required file not found: {DATA_PATH}")

    # Read only required columns; do not load the 51.9M raw readings.
    df = pd.read_csv(DATA_PATH, usecols=FEATURES)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < 1000:
        raise ValueError("Not enough valid rows to train LOF.")

    sample_size = min(SAMPLE_SIZE, len(df))
    df = df.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    X = df[FEATURES].astype(np.float32)

    # Reserve a separate set for a basic unseen-data scoring check.
    split = int(len(X) * 0.8)
    X_train = X.iloc[:split]
    X_check = X.iloc[split:]

    print(f"Rows available: {len(df):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Unseen check rows: {len(X_check):,}")
    print("Scaling features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_check_scaled = scaler.transform(X_check)

    print("Training LOF. This may take some time...")
    model_start = time.time()

    lof = LocalOutlierFactor(
        n_neighbors=35,
        contamination=0.02,
        novelty=True,
        n_jobs=2,
    )
    lof.fit(X_train_scaled)

    train_seconds = time.time() - model_start

    print("Scoring unseen check rows...")
    predictions = lof.predict(X_check_scaled)
    scores = lof.decision_function(X_check_scaled)
    anomaly_mask = predictions == -1

    output = X_check.copy()
    output["lof_prediction"] = predictions
    output["lof_score"] = scores
    output["potential_anomaly"] = anomaly_mask
    output.to_csv(RESULT_DIR / "lof_unseen_sample_predictions.csv", index=False)

    bundle = {
        "model": lof,
        "scaler": scaler,
        "feature_columns": FEATURES,
        "contamination": 0.02,
        "n_neighbors": 35,
        "novelty": True,
        "training_rows": int(len(X_train)),
        "sample_rows": int(len(df)),
        "note": "Potential anomalies only; not confirmed faults.",
    }

    model_path = MODEL_DIR / "gridguard_lof_anomaly.joblib"
    joblib.dump(bundle, model_path)

    summary = {
        "algorithm": "Local Outlier Factor",
        "sample_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "unseen_check_rows": int(len(X_check)),
        "potential_anomalies_in_check": int(anomaly_mask.sum()),
        "potential_anomaly_rate_percent": round(float(anomaly_mask.mean() * 100), 3),
        "training_seconds": round(train_seconds, 2),
        "total_seconds": round(time.time() - start, 2),
        "model_path": str(model_path),
        "interpretation": (
            "Unseen-data anomaly flags are potential anomalies, "
            "not verified real-world faults."
        ),
    }

    with open(RESULT_DIR / "lof_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nLOF training completed.")
    print(json.dumps(summary, indent=2))
    print("\nIsolation Forest has not been changed.")


if __name__ == "__main__":
    main()

