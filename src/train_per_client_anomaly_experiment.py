
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "anomaly_detection"
    / "training_sample_anomalies.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "gridguard_per_client_experimental.joblib"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "results"
    / "anomaly_detection"
    / "per_client_experimental_results.csv"
)

MIN_TRAIN_ROWS = 200
RANDOM_SEED = 42

LOG_COLUMNS = [
    "average_kw",
    "within_hour_std_kw",
    "previous_hour_kw",
    "previous_24h_kw",
    "previous_168h_kw",
    "rolling_mean_24h",
]

RELATIVE_PAIRS = {
    "relative_to_previous_hour": (
        "average_kw",
        "previous_hour_kw",
    ),
    "relative_to_previous_24h": (
        "average_kw",
        "previous_24h_kw",
    ),
    "relative_to_previous_168h": (
        "average_kw",
        "previous_168h_kw",
    ),
    "relative_to_rolling_mean": (
        "average_kw",
        "rolling_mean_24h",
    ),
}

CALENDAR_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

RELATIVE_COLUMNS = list(RELATIVE_PAIRS.keys())

LOG_FEATURE_COLUMNS = [
    f"log_{column}" for column in LOG_COLUMNS
]

FEATURE_COLUMNS = (
    RELATIVE_COLUMNS
    + LOG_FEATURE_COLUMNS
    + CALENDAR_COLUMNS
)


def add_experimental_features(data):
    data = data.copy()

    for feature_name, (value_col, baseline_col) in (
        RELATIVE_PAIRS.items()
    ):
        baseline = data[baseline_col].astype(float)

        # Avoid unstable relative changes when the baseline
        # is zero or extremely close to zero.
        valid_baseline = baseline.abs() >= 1.0

        relative = (
            data[value_col].astype(float) - baseline
        ) / baseline.abs().clip(lower=1.0)

        relative = relative.where(valid_baseline)
        data[feature_name] = relative.clip(-3.0, 3.0)

    for column in LOG_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce")
        data[f"log_{column}"] = np.log1p(
            values.clip(lower=0)
        )

    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    return data


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print("Loading historical anomaly sample...")
    data = pd.read_csv(INPUT_FILE)

    required_columns = [
        "client_id",
        "hour_timestamp",
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

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns are missing: {missing}"
        )

    data["hour_timestamp"] = pd.to_datetime(
        data["hour_timestamp"], errors="coerce"
    )

    data = data.dropna(
        subset=["client_id", "hour_timestamp"]
    ).copy()

    data = data.sort_values(
        ["hour_timestamp", "client_id"]
    ).reset_index(drop=True)

    data = add_experimental_features(data)

    # Use a chronological holdout so each client's later
    # records are evaluated against earlier records.
    cutoff = data["hour_timestamp"].quantile(0.80)

    train_data = data[
        data["hour_timestamp"] < cutoff
    ].copy()

    evaluation_data = data[
        data["hour_timestamp"] >= cutoff
    ].copy()

    print()
    print("PER-CLIENT ANOMALY EXPERIMENT")
    print(f"Rows available: {len(data):,}")
    print(f"Clients available: {data['client_id'].nunique():,}")
    print(f"Training rows before cutoff: {len(train_data):,}")
    print(f"Evaluation rows after cutoff: {len(evaluation_data):,}")
    print(f"Chronological cutoff: {cutoff}")

    model_bundle = {
        "experiment_name": "per_client_isolation_forest",
        "feature_columns": FEATURE_COLUMNS,
        "cutoff": str(cutoff),
        "minimum_training_rows": MIN_TRAIN_ROWS,
        "client_models": {},
    }

    all_results = []
    skipped_clients = []

    for client_id, client_evaluation in evaluation_data.groupby(
        "client_id", sort=True
    ):
        client_training = train_data[
            train_data["client_id"] == client_id
        ]

        if len(client_training) < MIN_TRAIN_ROWS:
            skipped_clients.append(
                (client_id, len(client_training), "too few training rows")
            )
            continue

        X_train = client_training[FEATURE_COLUMNS].astype(float)
        X_eval = client_evaluation[FEATURE_COLUMNS].astype(float)

        # Fit preprocessing only on this client's earlier data.
        imputer = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )

        X_train_imputed = imputer.fit_transform(X_train)
        X_eval_imputed = imputer.transform(X_eval)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_eval_scaled = scaler.transform(X_eval_imputed)

        model = IsolationForest(
            n_estimators=150,
            contamination=0.02,
            random_state=RANDOM_SEED,
            n_jobs=2,
        )

        model.fit(X_train_scaled)

        predictions = model.predict(X_eval_scaled)
        scores = -model.score_samples(X_eval_scaled)

        client_model = {
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
            "training_rows": len(client_training),
            "evaluation_rows": len(client_evaluation),
        }

        model_bundle["client_models"][str(client_id)] = client_model

        result = client_evaluation[
            [
                "client_id",
                "hour_timestamp",
                "average_kw",
            ]
        ].copy()

        result["potential_anomaly"] = (
            predictions == -1
        ).astype(int)

        result["anomaly_score"] = scores
        result["training_rows_for_client"] = len(client_training)
        result["evaluation_rows_for_client"] = len(client_evaluation)

        all_results.append(result)

    if not all_results:
        raise RuntimeError(
            "No clients had enough training rows. "
            "No experiment results were produced."
        )

    results = pd.concat(all_results, ignore_index=True)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_bundle, MODEL_FILE)
    results.to_csv(RESULT_FILE, index=False)

    client_summary = (
        results.groupby("client_id")
        .agg(
            evaluation_rows=("potential_anomaly", "size"),
            flagged_count=("potential_anomaly", "sum"),
            mean_kw=("average_kw", "mean"),
            mean_anomaly_score=("anomaly_score", "mean"),
        )
        .reset_index()
    )

    client_summary["flagged_rate_pct"] = (
        100
        * client_summary["flagged_count"]
        / client_summary["evaluation_rows"]
    )

    client_summary = client_summary.sort_values(
        "flagged_rate_pct", ascending=False
    )

    flagged_total = int(results["potential_anomaly"].sum())
    total_evaluated = len(results)

    print()
    print("RESULTS")
    print(f"Clients successfully modelled: {len(model_bundle['client_models']):,}")
    print(f"Clients skipped: {len(skipped_clients):,}")
    print(f"Evaluation rows scored: {total_evaluated:,}")
    print(f"Potential anomalies: {flagged_total:,}")
    print(
        "Overall evaluation flag rate: "
        f"{100 * flagged_total / total_evaluated:.2f}%"
    )

    print()
    print("Clients with the highest evaluation flag rates:")
    print(
        client_summary.head(15).to_string(
            index=False,
            formatters={
                "mean_kw": "{:.2f}".format,
                "mean_anomaly_score": "{:.4f}".format,
                "flagged_rate_pct": "{:.2f}%".format,
            },
        )
    )

    if skipped_clients:
        print()
        print("First skipped clients:")
        for item in skipped_clients[:10]:
            print(item)

    print()
    print("Saved experimental model:")
    print(MODEL_FILE)
    print("Saved evaluation results:")
    print(RESULT_FILE)
    print()
    print(
        "IMPORTANT: These are unsupervised model flags, not "
        "confirmed faults. This experiment does not replace "
        "the production model or dashboard."
    )


if __name__ == "__main__":
    main()
