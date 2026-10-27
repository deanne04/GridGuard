
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "results" / "anomaly_detection"
    / "training_sample_anomalies.csv"
)

MODEL_PATH = (
    ROOT / "models"
    / "gridguard_client_relative_experimental.joblib"
)

RESULT_PATH = (
    ROOT / "results" / "anomaly_detection"
    / "client_relative_experimental_results.csv"
)


def main():
    print("GridGuard: Client-Relative Anomaly Experiment")
    print("=" * 52)

    df = pd.read_csv(DATA_PATH, parse_dates=["hour_timestamp"])

    required = [
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

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=required).copy()
    df = df.sort_values("hour_timestamp").reset_index(drop=True)

    # Build relative features using historical values only.
    # The small denominator floor prevents division by zero.
    floor = 1.0

    df["relative_to_previous_hour"] = (
        (df["average_kw"] - df["previous_hour_kw"])
        / df["previous_hour_kw"].abs().clip(lower=floor)
    )

    df["relative_to_previous_24h"] = (
        (df["average_kw"] - df["previous_24h_kw"])
        / df["previous_24h_kw"].abs().clip(lower=floor)
    )

    df["relative_to_previous_168h"] = (
        (df["average_kw"] - df["previous_168h_kw"])
        / df["previous_168h_kw"].abs().clip(lower=floor)
    )

    df["relative_to_rolling_mean"] = (
        (df["average_kw"] - df["rolling_mean_24h"])
        / df["rolling_mean_24h"].abs().clip(lower=floor)
    )

    # Log transforms reduce the effect of very different usage scales.
    for col in [
        "average_kw",
        "within_hour_std_kw",
        "previous_hour_kw",
        "previous_24h_kw",
        "previous_168h_kw",
        "rolling_mean_24h",
    ]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    feature_columns = [
        "relative_to_previous_hour",
        "relative_to_previous_24h",
        "relative_to_previous_168h",
        "relative_to_rolling_mean",
        "log_average_kw",
        "log_within_hour_std_kw",
        "log_previous_hour_kw",
        "log_previous_24h_kw",
        "log_previous_168h_kw",
        "log_rolling_mean_24h",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    ]

    df[feature_columns] = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    df = df.dropna(subset=feature_columns).copy()

    # Chronological split: earlier rows train; later rows evaluate.
    cutoff = df["hour_timestamp"].quantile(0.80)

    train = df[df["hour_timestamp"] < cutoff].copy()
    test = df[df["hour_timestamp"] >= cutoff].copy()

    if len(train) < 1000 or len(test) < 100:
        raise ValueError("Not enough data for the chronological split.")

    print(f"Rows available: {len(df):,}")
    print(f"Clients: {df['client_id'].nunique():,}")
    print(f"Training rows: {len(train):,}")
    print(f"Evaluation rows: {len(test):,}")
    print(f"Chronological cutoff: {cutoff}")

    # Fit transformations on training data only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_columns])
    X_test = scaler.transform(test[feature_columns])

    model = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train)

    test["anomaly_score"] = -model.score_samples(X_test)
    test["potential_anomaly"] = (
        model.predict(X_test) == -1
    ).astype(int)

    # Save only experimental artifacts.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "model_name": "ClientRelativeIsolationForest",
            "contamination": 0.02,
            "training_cutoff": str(cutoff),
            "note": (
                "Experimental client-relative features. "
                "Flags are not confirmed faults or theft."
            ),
        },
        MODEL_PATH,
    )

    test.to_csv(RESULT_PATH, index=False)

    print("\nEvaluation results")
    print("-" * 52)
    print(f"Potential anomalies: {test['potential_anomaly'].sum():,}")
    print(
        "Evaluation anomaly rate: "
        f"{test['potential_anomaly'].mean() * 100:.2f}%"
    )

    summary = test.groupby("client_id").agg(
        rows=("potential_anomaly", "size"),
        mean_kw=("average_kw", "mean"),
        flagged_rate=("potential_anomaly", "mean"),
    )
    summary["flagged_rate"] *= 100

    print("\nHighest client anomaly rates:")
    print(
        summary.sort_values("flagged_rate", ascending=False)
        .head(10).round(2).to_string()
    )

    print("\nLowest client anomaly rates:")
    print(
        summary.sort_values("flagged_rate")
        .head(10).round(2).to_string()
    )

    print("\nExperimental model:", MODEL_PATH)
    print("Evaluation results:", RESULT_PATH)
    print("Existing models were not overwritten.")


if __name__ == "__main__":
    main()
