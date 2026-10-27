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
RESULT_DIR = ROOT / "results" / "anomaly_detection" / "lof_traceable_training"

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
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("GridGuard - Traceable LOF Training")
    print("=" * 45)

    columns = ["client_id", "hour_timestamp"] + FEATURES
    df = pd.read_csv(DATA_PATH, usecols=columns)

    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=FEATURES).copy()

    sample_size = min(SAMPLE_SIZE, len(df))
    df = df.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # Keep identifiers separate; use the same float32 feature format
    # as the original LOF script.
    identifiers = df[["client_id", "hour_timestamp"]].copy()
    X = df[FEATURES].astype(np.float32)

    split = int(len(X) * 0.8)
    X_train = X.iloc[:split]
    X_check = X.iloc[split:]
    identifiers_check = identifiers.iloc[split:].reset_index(drop=True)

    print(f"Sample rows: {len(df):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Unseen check rows: {len(X_check):,}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_check_scaled = scaler.transform(X_check)

    print("Training LOF...")
    model_start = time.time()

    lof = LocalOutlierFactor(
        n_neighbors=35,
        contamination=0.02,
        novelty=True,
        n_jobs=2,
    )
    lof.fit(X_train_scaled)
    train_seconds = time.time() - model_start

    print("Scoring held-out rows...")
    predictions = lof.predict(X_check_scaled)
    scores = lof.decision_function(X_check_scaled)

    output = identifiers_check.copy()
    output[FEATURES] = X_check.reset_index(drop=True)
    output["lof_prediction"] = predictions
    output["lof_score"] = scores
    output["potential_anomaly"] = predictions == -1

    output_path = RESULT_DIR / "lof_unseen_sample_predictions.csv"
    output.to_csv(output_path, index=False)

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

    model_path = RESULT_DIR / "gridguard_lof_anomaly_traceable.joblib"
    joblib.dump(bundle, model_path)

    summary = {
        "algorithm": "Local Outlier Factor",
        "sample_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "unseen_check_rows": int(len(X_check)),
        "potential_anomalies_in_check": int((predictions == -1).sum()),
        "potential_anomaly_rate_percent": round(
            float((predictions == -1).mean() * 100), 3
        ),
        "training_seconds": round(train_seconds, 2),
        "total_seconds": round(time.time() - start, 2),
        "prediction_file": str(output_path),
        "model_path": str(model_path),
        "interpretation": "Potential anomalies, not verified real-world faults.",
    }

    with open(RESULT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraceable LOF training completed.")
    print(json.dumps(summary, indent=2))
    print("\nOriginal LOF model and Isolation Forest model were not overwritten.")


if __name__ == "__main__":
    main()
