from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results" / "anomaly_detection"
    / "training_sample_anomalies.csv"
)
MODEL_OUTPUT = (
    ROOT / "models" / "gridguard_corrected_experimental.joblib"
)
RESULT_OUTPUT = (
    ROOT / "results" / "anomaly_detection"
    / "corrected_experimental_results.csv"
)

RELATIVE_FEATURES = [
    "relative_to_previous_hour",
    "relative_to_previous_24h",
    "relative_to_previous_168h",
    "relative_to_rolling_mean",
]

LOG_SOURCE_FEATURES = [
    "average_kw",
    "within_hour_std_kw",
    "previous_hour_kw",
    "previous_24h_kw",
    "previous_168h_kw",
    "rolling_mean_24h",
]

CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

FEATURES = (
    RELATIVE_FEATURES
    + ["log_" + c for c in LOG_SOURCE_FEATURES]
    + CALENDAR_FEATURES
)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT}")

    df = pd.read_csv(INPUT)
    df["hour_timestamp"] = pd.to_datetime(
        df["hour_timestamp"], errors="coerce"
    )

    required = [
        "client_id", "hour_timestamp", "average_kw",
        "within_hour_std_kw", "previous_hour_kw",
        "previous_24h_kw", "previous_168h_kw",
        "rolling_mean_24h", "hour", "day_of_week",
        "month", "is_weekend",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    df = df.dropna(subset=["client_id", "hour_timestamp"]).copy()
    df = df.sort_values("hour_timestamp").reset_index(drop=True)

    # Relative changes are unreliable when the baseline is near zero.
    baseline_pairs = [
        ("previous_hour_kw", RELATIVE_FEATURES[0]),
        ("previous_24h_kw", RELATIVE_FEATURES[1]),
        ("previous_168h_kw", RELATIVE_FEATURES[2]),
        ("rolling_mean_24h", RELATIVE_FEATURES[3]),
    ]

    for baseline_col, feature_col in baseline_pairs:
        baseline = pd.to_numeric(df[baseline_col], errors="coerce")
        current = pd.to_numeric(df["average_kw"], errors="coerce")

        valid = baseline.abs() >= 1.0
        relative = (current - baseline) / baseline.abs().where(valid)
        relative = relative.replace([np.inf, -np.inf], np.nan)
        df[feature_col] = relative.clip(-3, 3)

    # Log transforms reduce the effect of large differences in usage scale.
    for col in LOG_SOURCE_FEATURES:
        values = pd.to_numeric(df[col], errors="coerce")
        # Power usage and standard deviations should not be negative.
        df["log_" + col] = np.log1p(values.clip(lower=0))

    # Chronological holdout: train on earlier rows, evaluate on later rows.
    cutoff = df["hour_timestamp"].quantile(0.80)
    train_mask = df["hour_timestamp"] < cutoff
    test_mask = df["hour_timestamp"] >= cutoff

    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()

    if len(train) == 0 or len(test) == 0:
        raise ValueError("Chronological train/test split is empty.")

    # Fit the imputer and scaler on training data only.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_train = imputer.fit_transform(train[FEATURES])
    x_test = imputer.transform(test[FEATURES])

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        random_state=42,
        n_jobs=2,
    )
    model.fit(x_train)

    test["anomaly_score"] = -model.score_samples(x_test)
    test["potential_anomaly"] = (
        model.predict(x_test) == -1
    ).astype(int)

    # Save to new paths; leave the existing models untouched.
    bundle = {
        "model_name": "IsolationForest - corrected experimental",
        "feature_columns": FEATURES,
        "relative_feature_baseline_threshold": 1.0,
        "relative_feature_cap": 3.0,
        "imputer": imputer,
        "scaler": scaler,
        "predictor": model,
        "training_cutoff": str(cutoff),
        "note": (
            "Experimental model. Unlabelled anomalies are model flags, "
            "not confirmed real-world anomalies."
        ),
    }

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RESULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, MODEL_OUTPUT)
    test.to_csv(RESULT_OUTPUT, index=False)

    print("CORRECTED ANOMALY EXPERIMENT")
    print("=" * 45)
    print("Input rows:", len(df))
    print("Clients:", df["client_id"].nunique())
    print("Training rows:", len(train))
    print("Evaluation rows:", len(test))
    print("Chronological cutoff:", cutoff)
    print(
        "Evaluation anomaly rate:",
        round(test["potential_anomaly"].mean() * 100, 2), "%"
    )
    print("Potential anomalies:", int(test["potential_anomaly"].sum()))

    summary = test.groupby("client_id").agg(
        rows=("potential_anomaly", "size"),
        mean_kw=("average_kw", "mean"),
        flagged_rate=("potential_anomaly", "mean"),
    )
    summary["flagged_rate"] *= 100

    print("\nHighest client anomaly rates:")
    print(
        summary.sort_values("flagged_rate", ascending=False)
        .head(15).round(2).to_string()
    )

    print("\nLowest client anomaly rates:")
    print(
        summary.sort_values("flagged_rate", ascending=True)
        .head(10).round(2).to_string()
    )

    print("\nNew experimental model:", MODEL_OUTPUT)
    print("New evaluation results:", RESULT_OUTPUT)
    print("Existing production model and UI were not changed.")


if __name__ == "__main__":
    main()
