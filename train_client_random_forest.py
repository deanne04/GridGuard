
from pathlib import Path
import json
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "client_hourly_by_client"
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results" / "client_random_forest"

TEST_FRACTION = 0.20
MAX_TRAIN_ROWS_PER_CLIENT = 1000
TEST_EVERY_N_HOURS = 24
RANDOM_STATE = 42


# --------------------------------------------------
# LOAD ALL CLIENT DATA
# --------------------------------------------------

def load_all_clients():
    folders = sorted(DATA_ROOT.glob("client_id=MT_*"))
    frames = []

    print(f"Client folders found: {len(folders)}")

    for number, folder in enumerate(folders, start=1):
        match = re.fullmatch(r"client_id=(MT_\d+)", folder.name)

        if not match:
            continue

        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            continue

        pieces = []

        for file in csv_files:
            df = pd.read_csv(
                file,
                usecols=["timestamp", "average_kw"]
            )
            pieces.append(df)

        df = pd.concat(pieces, ignore_index=True)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )
        df["average_kw"] = pd.to_numeric(
            df["average_kw"], errors="coerce"
        )

        df = df.dropna(
            subset=["timestamp", "average_kw"]
        )
        df = df.sort_values("timestamp")
        df = df.drop_duplicates(
            subset=["timestamp"], keep="last"
        )

        # Resample to hourly readings.
        df = (
            df.set_index("timestamp")
            .resample("h")["average_kw"]
            .mean()
            .to_frame()
        )

        # Fill only small internal gaps, not long gaps.
        df["average_kw"] = df["average_kw"].interpolate(
            method="time",
            limit=2,
            limit_area="inside"
        )

        df = df.reset_index()
        df["client_id"] = match.group(1)

        frames.append(df)

        if number % 50 == 0:
            print(
                f"Loaded {number}/{len(folders)} client folders..."
            )

    if not frames:
        raise RuntimeError(
            f"No client CSV files found in {DATA_ROOT}"
        )

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------
# CREATE MACHINE-LEARNING FEATURES
# --------------------------------------------------

def make_features(data):
    data = data.sort_values(
        ["client_id", "timestamp"]
    ).copy()

    grouped = data.groupby(
        "client_id", sort=False
    )["average_kw"]

    # Previous electricity readings.
    data["lag_1h"] = grouped.shift(1)
    data["lag_24h"] = grouped.shift(24)

    # Historical rolling averages.
    data["rolling_mean_6h"] = grouped.transform(
        lambda s: s.shift(1).rolling(
            6, min_periods=6
        ).mean()
    )

    data["rolling_mean_24h"] = grouped.transform(
        lambda s: s.shift(1).rolling(
            24, min_periods=24
        ).mean()
    )

    # Calendar features.
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Keep the original ID for reporting only.
    # Do NOT include this text column in model features.
    data["client_id_original"] = data["client_id"]

    # Convert client IDs into numeric one-hot columns.
    data = pd.get_dummies(
        data,
        columns=["client_id"],
        dtype=np.uint8
    )

    # IMPORTANT:
    # Only encoded client columns starting with client_id_MT_
    # are used as client identity features.
    client_columns = sorted(
        [
            column
            for column in data.columns
            if re.fullmatch(
                r"client_id_MT_\d+",
                column
            )
        ]
    )

    feature_columns = [
        "lag_1h",
        "lag_24h",
        "rolling_mean_6h",
        "rolling_mean_24h",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    ] + client_columns

    # The original text ID is intentionally NOT included here.
    data = data.dropna(
        subset=feature_columns + ["average_kw"]
    )

    return data, feature_columns


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    return {
        "mae_kw": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse_kw": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "r2": float(
            r2_score(actual, predicted)
        ),
    }


# --------------------------------------------------
# MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():
    started = time.time()

    MODEL_DIR.mkdir(
        parents=True, exist_ok=True
    )
    RESULT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    print("Loading all client histories...")

    raw = load_all_clients()

    client_count = raw["client_id"].nunique()
    raw_row_count = len(raw)

    print(f"Clients loaded: {client_count}")
    print(f"Raw readings: {raw_row_count:,}")

    if client_count != 370:
        print(
            "WARNING: Expected 370 clients, "
            f"but found {client_count}."
        )

    print("Creating lag and calendar features...")

    data, feature_columns = make_features(raw)

    del raw

    print(f"Usable feature rows: {len(data):,}")
    print("Preparing chronological train/test split...")

    # Split each client by timestamp order.
    # The newest 20% of each client's usable rows
    # are reserved for evaluation.
    data["client_time_rank"] = (
        data.groupby("client_id_original")["timestamp"]
        .rank(method="first", pct=True)
    )

    train = data[
        data["client_time_rank"] <= (1 - TEST_FRACTION)
    ].copy()

    test = data[
        data["client_time_rank"] > (1 - TEST_FRACTION)
    ].copy()

    del data

    if train.empty or test.empty:
        raise RuntimeError(
            "The train/test split produced an empty dataset."
        )

    # Sample up to 1,000 training rows per client.
    # This reduces memory use while representing all clients.
    print("Sampling training rows from each client...")

    sampled_train = (
        train.groupby(
            "client_id_original",
            group_keys=False
        )
        .apply(
            lambda group: group.sample(
                n=min(
                    MAX_TRAIN_ROWS_PER_CLIENT,
                    len(group)
                ),
                random_state=RANDOM_STATE
            ),
            include_groups=True
        )
        .reset_index(drop=True)
    )

    del train

    # Use one evaluation reading every 24 hours
    # per client to keep evaluation memory manageable.
    print("Sampling the held-out evaluation period...")

    test = test.sort_values(
        ["client_id_original", "timestamp"]
    ).copy()

    test["test_row_number"] = (
        test.groupby("client_id_original").cumcount()
    )

    sampled_test = test[
        test["test_row_number"] % TEST_EVERY_N_HOURS == 0
    ].copy()

    del test

    if sampled_test.empty:
        raise RuntimeError(
            "No evaluation rows remain after sampling."
        )

    print(
        f"Training sample rows: {len(sampled_train):,}"
    )
    print(
        f"Evaluation sample rows: {len(sampled_test):,}"
    )
    print(f"Features used: {len(feature_columns)}")
    print("Training shared Random Forest...")

    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    # Only numeric lag, calendar, and one-hot features
    # are passed to the model.
    model.fit(
        sampled_train[feature_columns],
        sampled_train["average_kw"]
    )

    print("Generating predictions...")

    rf_predictions = np.maximum(
        0,
        model.predict(
            sampled_test[feature_columns]
        )
    )

    # Simple comparison: predict the next value using
    # the previous hour's observed value.
    baseline_predictions = (
        sampled_test["lag_1h"].to_numpy()
    )

    actual = sampled_test["average_kw"].to_numpy()

    rf_metrics = calculate_metrics(
        actual, rf_predictions
    )

    baseline_metrics = calculate_metrics(
        actual, baseline_predictions
    )

    # Store the exact model feature order for future inference.
    model_path = (
        MODEL_DIR
        / "gridguard_shared_client_random_forest.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
        },
        model_path
    )

    # Save evaluation predictions.
    output = sampled_test[
        [
            "timestamp",
            "average_kw",
            "lag_1h",
            "client_id_original",
        ]
    ].copy()

    output = output.rename(
        columns={
            "average_kw": "actual_kw",
            "lag_1h": "baseline_predicted_kw",
            "client_id_original": "client_id",
        }
    )

    output["rf_predicted_kw"] = rf_predictions

    output.to_csv(
        RESULT_DIR / "test_predictions.csv",
        index=False
    )

    # Save experiment metrics and configuration.
    metrics = {
        "model_name": "Shared Random Forest",
        "clients_used": int(client_count),
        "raw_readings_loaded": int(raw_row_count),
        "usable_feature_rows": int(
            len(sampled_train) + len(sampled_test)
        ),
        "training_rows": int(len(sampled_train)),
        "evaluation_rows": int(len(sampled_test)),
        "training_rows_per_client_cap": (
            MAX_TRAIN_ROWS_PER_CLIENT
        ),
        "evaluation_sampling": (
            f"Every {TEST_EVERY_N_HOURS}th row "
            "in each client's held-out period"
        ),
        "split_method": (
            "Chronological per client; latest 20% "
            "held out before sampling"
        ),
        "random_forest": rf_metrics,
        "previous_hour_baseline": baseline_metrics,
        "feature_columns": feature_columns,
        "elapsed_seconds": round(
            time.time() - started, 2
        ),
    }

    metrics_path = RESULT_DIR / "metrics.json"

    with metrics_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\n========== EVALUATION RESULTS ==========")
    print(f"Clients used: {client_count}")
    print(f"Training rows: {len(sampled_train):,}")
    print(f"Evaluation rows: {len(sampled_test):,}")

    print("\nRandom Forest:")
    print(json.dumps(rf_metrics, indent=2))

    print("\nPrevious-hour baseline:")
    print(json.dumps(baseline_metrics, indent=2))

    print(
        "\nElapsed seconds:",
        metrics["elapsed_seconds"]
    )

    print(f"\nModel saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(
        "Predictions saved:",
        RESULT_DIR / "test_predictions.csv"
    )


if __name__ == "__main__":
    main()
