
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


# ============================================================
# GRIDGUARD - ANOMALY DETECTION MODEL COMPARISON
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "results" / "anomaly_detection"
    / "training_sample_anomalies.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "anomaly_detection"
    / "model_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 10000
SVM_MAX_ROWS = 3000
CONTAMINATION = 0.02

FEATURE_COLUMNS = [
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
    print("=" * 65)
    print("GRIDGUARD - ANOMALY DETECTION MODEL COMPARISON")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # 1. Load existing training sample
    # --------------------------------------------------------
    print("\n1. Loading existing anomaly training sample...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows available: {len(df):,}")

    missing = [
        column for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )

    # Remove invalid numeric values.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    df = df.dropna(subset=FEATURE_COLUMNS).copy()

    # Use the same reproducible rows for Isolation Forest and LOF.
    if len(df) > SAMPLE_SIZE:
        df = df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )

    df = df.reset_index(drop=True)

    X = df[FEATURE_COLUMNS].astype(float)

    print(f"Rows selected: {len(X):,}")
    print(f"Features used: {len(FEATURE_COLUMNS)}")

    # --------------------------------------------------------
    # 2. Scale features
    # --------------------------------------------------------
    print("\n2. Scaling features...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Results are stored separately from the original model.
    results = []

    predictions = pd.DataFrame({
        "client_id": df["client_id"].to_numpy(),
        "hour_timestamp": df["hour_timestamp"].to_numpy(),
    })

    # --------------------------------------------------------
    # Helper function for model evaluation
    # --------------------------------------------------------
    def evaluate_model(name, model, data):
        print(f"\nTraining/evaluating {name}...")

        start_time = time.perf_counter()

        predicted_labels = model.fit_predict(data)

        elapsed = time.perf_counter() - start_time

        # sklearn uses -1 for anomaly and 1 for normal.
        anomaly_flags = (predicted_labels == -1).astype(int)

        anomaly_count = int(anomaly_flags.sum())
        anomaly_rate = float(anomaly_flags.mean() * 100)

        results.append({
            "model": name,
            "records_evaluated": int(len(data)),
            "potential_anomalies": anomaly_count,
            "normal_records": int(len(data) - anomaly_count),
            "anomaly_rate_percent": round(anomaly_rate, 3),
            "runtime_seconds": round(elapsed, 3),
            "contamination_setting": CONTAMINATION,
        })

        print(f"Records evaluated: {len(data):,}")
        print(f"Potential anomalies: {anomaly_count:,}")
        print(f"Anomaly rate: {anomaly_rate:.2f}%")
        print(f"Runtime: {elapsed:.2f} seconds")

        return anomaly_flags

    # --------------------------------------------------------
    # 3. Isolation Forest
    # --------------------------------------------------------
    print("\n3. Running Isolation Forest...")

    isolation_forest = IsolationForest(
        n_estimators=150,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    isolation_flags = evaluate_model(
        "Isolation Forest",
        isolation_forest,
        X_scaled,
    )

    predictions["isolation_forest_anomaly"] = isolation_flags

    # --------------------------------------------------------
    # 4. Local Outlier Factor
    # --------------------------------------------------------
    print("\n4. Running Local Outlier Factor...")

    lof = LocalOutlierFactor(
        n_neighbors=35,
        contamination=CONTAMINATION,
        n_jobs=2,
    )

    lof_flags = evaluate_model(
        "Local Outlier Factor",
        lof,
        X_scaled,
    )

    predictions["lof_anomaly"] = lof_flags

    # --------------------------------------------------------
    # 5. One-Class SVM
    # --------------------------------------------------------
    print("\n5. Running One-Class SVM...")

    # SVM can be slow, so select a reproducible subset.
    rng = np.random.default_rng(RANDOM_STATE)

    if len(X_scaled) > SVM_MAX_ROWS:
        svm_indices = np.sort(
            rng.choice(
                len(X_scaled),
                size=SVM_MAX_ROWS,
                replace=False,
            )
        )
    else:
        svm_indices = np.arange(len(X_scaled))

    X_svm = X_scaled[svm_indices]

    print(f"One-Class SVM sample size: {len(X_svm):,}")

    svm = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=CONTAMINATION,
    )

    svm_flags = evaluate_model(
        "One-Class SVM",
        svm,
        X_svm,
    )

    # IMPORTANT:
    # Only place predictions at the matching sampled row positions.
    # Other rows remain blank because SVM did not evaluate them.
    svm_column = np.full(len(df), np.nan)
    svm_column[svm_indices] = svm_flags

    predictions["one_class_svm_anomaly"] = svm_column

    # --------------------------------------------------------
    # 6. Compare agreement between models
    # --------------------------------------------------------
    predictions["all_three_flagged"] = (
        (predictions["isolation_forest_anomaly"] == 1)
        & (predictions["lof_anomaly"] == 1)
        & (predictions["one_class_svm_anomaly"] == 1)
    ).astype(int)

    # --------------------------------------------------------
    # 7. Save comparison results
    # --------------------------------------------------------
    print("\n6. Saving results...")

    results_df = pd.DataFrame(results)

    comparison_file = (
        OUTPUT_DIR / "anomaly_model_comparison.csv"
    )

    predictions_file = (
        OUTPUT_DIR / "anomaly_model_predictions.csv"
    )

    summary_file = (
        OUTPUT_DIR / "anomaly_model_comparison_summary.json"
    )

    results_df.to_csv(comparison_file, index=False)
    predictions.to_csv(predictions_file, index=False)

    summary = {
        "project": "GridGuard",
        "purpose": "Unsupervised anomaly detection comparison",
        "input_file": str(INPUT_FILE),
        "rows_used_for_isolation_forest_and_lof": int(len(df)),
        "rows_used_for_one_class_svm": int(len(X_svm)),
        "feature_columns": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "contamination_setting": CONTAMINATION,
        "results": results,
        "interpretation": (
            "Anomaly rates and runtime do not establish detection accuracy. "
            "The dataset has no verified anomaly labels. One-Class SVM "
            "was evaluated on a smaller subset, so its runtime and anomaly "
            "count should not be directly compared with full-sample totals. "
            "A separate validation experiment is needed to identify the "
            "most useful model."
        ),
    }

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # --------------------------------------------------------
    # 8. Display final report
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("ANOMALY MODEL COMPARISON RESULTS")
    print("=" * 65)

    print(
        results_df.to_string(index=False)
    )

    print("\nModel agreement:")
    print(
        "Records flagged by both Isolation Forest and LOF:",
        int(
            (
                (predictions["isolation_forest_anomaly"] == 1)
                & (predictions["lof_anomaly"] == 1)
            ).sum()
        ),
    )

    print(
        "Records flagged by all three models:",
        int(predictions["all_three_flagged"].sum()),
    )

    print("\nFiles saved successfully:")
    print(comparison_file)
    print(predictions_file)
    print(summary_file)

    print("\nOriginal Isolation Forest model was not modified.")
    print("COMPARISON COMPLETED.")


if __name__ == "__main__":
    main()
