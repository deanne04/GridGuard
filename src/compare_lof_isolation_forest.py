
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parent.parent

LOF_PATH = ROOT / "models" / "gridguard_lof_anomaly.joblib"
IF_PATH = ROOT / "models" / "gridguard_isolation_forest.joblib"
DATA_PATH = (
    ROOT / "results" / "anomaly_detection"
    / "lof_training" / "lof_unseen_sample_predictions.csv"
)
OUT_DIR = ROOT / "results" / "anomaly_detection" / "lof_vs_isolation_forest"

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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("GridGuard: LOF vs Isolation Forest")
    print("=" * 42)

    for path in [LOF_PATH, IF_PATH, DATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading saved models and unseen records...")
    lof_bundle = joblib.load(LOF_PATH)
    if_bundle = joblib.load(IF_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    X = X.loc[valid].reset_index(drop=True)

    # Use each model's own fitted scaler.
    lof_X = lof_bundle["scaler"].transform(X)
    if_X = if_bundle["scaler"].transform(X)

    print(f"Records evaluated: {len(X):,}")

    results = {}
    predictions = df.copy()

    # LOF scores are higher for more normal observations.
    start = time.time()
    lof_pred = lof_bundle["model"].predict(lof_X)
    lof_score = lof_bundle["model"].decision_function(lof_X)
    results["LOF"] = {
        "potential_anomalies": int((lof_pred == -1).sum()),
        "anomaly_rate_percent": round(float((lof_pred == -1).mean() * 100), 3),
        "scoring_seconds": round(time.time() - start, 3),
    }
    predictions["lof_prediction"] = lof_pred
    predictions["lof_score"] = lof_score

    # Isolation Forest uses -1 for anomaly and 1 for normal.
    start = time.time()
    if_pred = if_bundle["model"].predict(if_X)
    if_score = if_bundle["model"].decision_function(if_X)
    results["Isolation Forest"] = {
        "potential_anomalies": int((if_pred == -1).sum()),
        "anomaly_rate_percent": round(float((if_pred == -1).mean() * 100), 3),
        "scoring_seconds": round(time.time() - start, 3),
    }
    predictions["if_prediction"] = if_pred
    predictions["if_score"] = if_score

    lof_flags = lof_pred == -1
    if_flags = if_pred == -1
    intersection = int(np.logical_and(lof_flags, if_flags).sum())
    union = int(np.logical_or(lof_flags, if_flags).sum())

    results["agreement"] = {
        "both_flagged": intersection,
        "lof_only": int(np.logical_and(lof_flags, ~if_flags).sum()),
        "isolation_forest_only": int(np.logical_and(if_flags, ~lof_flags).sum()),
        "both_normal": int(np.logical_and(~lof_flags, ~if_flags).sum()),
        "jaccard_similarity_percent": round(
            100 * intersection / union, 2
        ) if union else 100.0,
    }

    predictions.to_csv(OUT_DIR / "model_predictions.csv", index=False)

    with open(OUT_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nComparison results:")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to: {OUT_DIR}")
    print("\nNote: agreement and anomaly counts do not measure accuracy.")
    print("No confirmed real-world anomaly labels are available.")


if __name__ == "__main__":
    main()
