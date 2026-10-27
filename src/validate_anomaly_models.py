
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# GRIDGUARD - MULTI-SCENARIO ANOMALY VALIDATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "results" / "anomaly_detection"
    / "training_sample_anomalies.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "anomaly_detection"
    / "multi_scenario_validation"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

BASE_SEED = 42
REFERENCE_ROWS = 10000
ANOMALY_RATE = 0.02
SEEDS = [42, 123, 2026]

# Feature indexes
CURRENT = 0
VARIABILITY = 1
PREVIOUS_HOUR = 2
PREVIOUS_DAY = 3
PREVIOUS_WEEK = 4
ROLLING_MEAN = 5


def make_model(name, seed):
    """Create a fresh model for every independent experiment."""
    if name == "Isolation Forest":
        return IsolationForest(
            n_estimators=150,
            contamination=ANOMALY_RATE,
            random_state=seed,
            n_jobs=2,
        )

    if name == "Local Outlier Factor":
        return LocalOutlierFactor(
            n_neighbors=35,
            contamination=ANOMALY_RATE,
            novelty=True,
            n_jobs=2,
        )

    if name == "One-Class SVM":
        return OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=ANOMALY_RATE,
        )

    raise ValueError(f"Unknown model: {name}")


def inject_anomalies(normal_test, scenario, seed):
    """
    Create artificial test anomalies.

    These transformations are controlled test scenarios, not
    confirmed real-world electricity faults.
    """
    rng = np.random.default_rng(seed)

    X_test = normal_test.copy()
    y_test = np.zeros(len(X_test), dtype=int)

    count = max(1, int(len(X_test) * ANOMALY_RATE))
    selected = rng.choice(len(X_test), size=count, replace=False)

    for idx in selected:
        # Use historical values to define the approximate baseline.
        history = np.abs(
            X_test[idx, [PREVIOUS_HOUR, PREVIOUS_DAY,
                         PREVIOUS_WEEK, ROLLING_MEAN]]
        )
        baseline = max(float(np.median(history)), 1.0)
        current = max(abs(float(X_test[idx, CURRENT])), 1.0)

        if scenario == "Sudden spike":
            # Unusually high current reading.
            X_test[idx, CURRENT] = max(current, baseline) * 8.0

        elif scenario == "Sudden drop":
            # Very low current reading relative to its baseline.
            X_test[idx, CURRENT] = 0.0

        elif scenario == "Unusual variability":
            # Unusually large within-hour variability.
            X_test[idx, VARIABILITY] = max(
                abs(float(X_test[idx, VARIABILITY])) * 10.0,
                baseline * 5.0,
                10.0,
            )

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        y_test[idx] = 1

    return X_test, y_test


def run_experiment(X_train, X_normal_test, scenario, seed, model_name):
    """Train and evaluate one model on one scenario and seed."""

    X_test, y_test = inject_anomalies(
        X_normal_test, scenario, seed
    )

    model = make_model(model_name, seed)

    start = time.perf_counter()
    model.fit(X_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    labels = model.predict(X_test)
    predict_seconds = time.perf_counter() - start

    y_pred = (labels == -1).astype(int)

    tp = int(((y_test == 1) & (y_pred == 1)).sum())
    fp = int(((y_test == 0) & (y_pred == 1)).sum())
    fn = int(((y_test == 1) & (y_pred == 0)).sum())
    tn = int(((y_test == 0) & (y_pred == 0)).sum())

    return {
        "seed": seed,
        "scenario": scenario,
        "model": model_name,
        "test_records": int(len(y_test)),
        "known_synthetic_anomalies": int(y_test.sum()),
        "detected_anomalies": tp,
        "false_positives": fp,
        "missed_anomalies": fn,
        "true_negatives": tn,
        "precision": round(
            float(precision_score(y_test, y_pred, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_test, y_pred, zero_division=0)), 4
        ),
        "f1_score": round(
            float(f1_score(y_test, y_pred, zero_division=0)), 4
        ),
        "fit_seconds": round(fit_seconds, 4),
        "predict_seconds": round(predict_seconds, 4),
        "total_seconds": round(fit_seconds + predict_seconds, 4),
    }


def main():
    print("=" * 72)
    print("GRIDGUARD - MULTI-SCENARIO ANOMALY VALIDATION")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print("\n1. Loading historical reference sample...")
    df = pd.read_csv(INPUT_FILE)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=FEATURES)

    if len(df) < REFERENCE_ROWS:
        raise ValueError(
            f"Need {REFERENCE_ROWS} usable rows; found {len(df)}"
        )

    df = df.sample(
        n=REFERENCE_ROWS,
        random_state=BASE_SEED,
    ).reset_index(drop=True)

    X = df[FEATURES].astype(float).to_numpy()

    print(f"Reference rows: {len(X):,}")
    print(f"Features: {len(FEATURES)}")
    print(f"Random seeds: {SEEDS}")

    all_results = []

    scenarios = [
        "Sudden spike",
        "Sudden drop",
        "Unusual variability",
    ]

    model_names = [
        "Isolation Forest",
        "Local Outlier Factor",
        "One-Class SVM",
    ]

    # Each seed uses a different reproducible train/test split.
    for seed in SEEDS:
        print(f"\n{'=' * 72}")
        print(f"EXPERIMENT SEED: {seed}")
        print("=" * 72)

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(X))

        split = int(len(indices) * 0.70)
        train_indices = indices[:split]
        test_indices = indices[split:]

        X_train_raw = X[train_indices]
        X_test_normal_raw = X[test_indices]

        # Fit scaler on training data only.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test_normal = scaler.transform(X_test_normal_raw)

        for scenario in scenarios:
            print(f"\n--- Scenario: {scenario} ---")

            for model_name in model_names:
                result = run_experiment(
                    X_train=X_train,
                    X_normal_test=X_test_normal,
                    scenario=scenario,
                    seed=seed,
                    model_name=model_name,
                )

                all_results.append(result)

                print(
                    f"{model_name}: "
                    f"Precision={result['precision']:.3f}, "
                    f"Recall={result['recall']:.3f}, "
                    f"F1={result['f1_score']:.3f}, "
                    f"Time={result['total_seconds']:.3f}s"
                )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------
    results_df = pd.DataFrame(all_results)

    detailed_file = OUTPUT_DIR / "detailed_validation_results.csv"
    summary_file = OUTPUT_DIR / "model_validation_summary.csv"
    json_file = OUTPUT_DIR / "model_validation_summary.json"

    results_df.to_csv(detailed_file, index=False)

    summary_df = (
        results_df.groupby(["scenario", "model"], as_index=False)
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1_score", "mean"),
            mean_runtime_seconds=("total_seconds", "mean"),
            f1_std_dev=("f1_score", "std"),
        )
    )

    summary_df["f1_std_dev"] = summary_df["f1_std_dev"].fillna(0)

    for column in [
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_runtime_seconds",
        "f1_std_dev",
    ]:
        summary_df[column] = summary_df[column].round(4)

    summary_df.to_csv(summary_file, index=False)

    # Average across scenarios; this ranking is specific to our tests.
    overall = (
        summary_df.groupby("model", as_index=False)
        .agg(
            average_f1=("mean_f1", "mean"),
            average_recall=("mean_recall", "mean"),
            average_precision=("mean_precision", "mean"),
            average_runtime_seconds=("mean_runtime_seconds", "mean"),
            f1_variability=("f1_std_dev", "mean"),
        )
        .sort_values(
            ["average_f1", "average_recall", "average_runtime_seconds"],
            ascending=[False, False, True],
        )
    )

    overall_file = OUTPUT_DIR / "overall_model_ranking.csv"
    overall.to_csv(overall_file, index=False)

    summary_json = {
        "project": "GridGuard",
        "experiment": "Multi-scenario synthetic anomaly validation",
        "reference_rows": REFERENCE_ROWS,
        "seeds": SEEDS,
        "scenarios": scenarios,
        "models": model_names,
        "features": FEATURES,
        "ranking_rule": (
            "Average F1 across scenarios, then recall, then runtime. "
            "This ranks models only for the defined synthetic experiments."
        ),
        "limitations": [
            "Injected anomalies are artificial and simplified.",
            "Synthetic metrics are not real-world anomaly accuracy.",
            "The source dataset has no verified anomaly labels.",
            "The results depend on the anomaly injection patterns.",
        ],
        "overall_ranking": overall.to_dict(orient="records"),
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4)

    print("\n" + "=" * 72)
    print("AVERAGE RESULTS BY SCENARIO")
    print("=" * 72)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 72)
    print("OVERALL RANKING ON THESE SYNTHETIC TESTS")
    print("=" * 72)
    print(overall.to_string(index=False))

    print("\nSaved files:")
    print(detailed_file)
    print(summary_file)
    print(overall_file)
    print(json_file)

    print(
        "\nNote: these experiments compare response to artificial patterns. "
        "They cannot prove which model is best for real-world faults."
    )
    print("\nVALIDATION COMPLETED.")


if __name__ == "__main__":
    main()
