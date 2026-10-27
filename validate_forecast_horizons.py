from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# GridGuard: Optimized Recursive 1-24 Hour Forecast Validation
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "client_hourly_by_client"
MODEL_PATH = PROJECT_ROOT / "models" / "gridguard_shared_client_random_forest.joblib"
OUTPUT_DIR = PROJECT_ROOT / "results" / "forecast_horizon_validation"

CHECKPOINT_PATH = OUTPUT_DIR / "horizon_predictions_checkpoint.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "horizon_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "horizon_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "validation_summary.json"
COMPARISON_PATH = OUTPUT_DIR / "horizon_baseline_comparison.csv"

MAX_HORIZON = 24
TEST_FRACTION = 0.20
MAX_TEST_POINTS = 30
ORIGIN_GAP_HOURS = 24
MIN_HISTORY_HOURS = 100

warnings.filterwarnings(
    "ignore",
    message=".*feature names.*",
    category=UserWarning,
)


def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise ValueError("Expected a model bundle dictionary.")

    if "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Model bundle must contain model and feature_columns.")

    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])

    if not feature_columns:
        raise ValueError("The saved model has no feature columns.")

    print(f"Model loaded: {MODEL_PATH.name}")
    print(f"Feature columns: {len(feature_columns)}")

    return model, feature_columns


def client_id_from_folder(folder_name):
    prefix = "client_id="
    if folder_name.startswith(prefix):
        return folder_name[len(prefix):]
    return folder_name


def load_client_series():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Client data folder not found: {DATA_ROOT}")

    folders = sorted(
        path for path in DATA_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("client_id=")
    )

    print(f"Client folders found: {len(folders)}")
    clients = {}

    for position, folder in enumerate(folders, start=1):
        client_id = client_id_from_folder(folder.name)
        csv_files = sorted(folder.glob("*.csv"))

        if not csv_files:
            print(f"Skipping {client_id}: no CSV file.")
            continue

        frames = []

        for csv_path in csv_files:
            try:
                frames.append(
                    pd.read_csv(
                        csv_path,
                        usecols=["timestamp", "average_kw"],
                    )
                )
            except (ValueError, OSError) as exc:
                print(f"Skipping {csv_path.name}: {exc}")

        if not frames:
            print(f"Skipping {client_id}: no readable data.")
            continue

        data = pd.concat(frames, ignore_index=True)
        data["timestamp"] = pd.to_datetime(
            data["timestamp"], errors="coerce"
        )
        data["average_kw"] = pd.to_numeric(
            data["average_kw"], errors="coerce"
        )

        data = data.dropna(subset=["timestamp", "average_kw"])
        data = data.sort_values("timestamp")
        data = data.drop_duplicates(
            subset=["timestamp"], keep="last"
        )

        if data.empty:
            continue

        series = (
            data.set_index("timestamp")["average_kw"]
            .sort_index()
            .resample("h")
            .mean()
        )

        # Fill only small internal gaps, matching the project pipeline.
        series = series.interpolate(
            method="time",
            limit=2,
            limit_area="inside",
        ).dropna()

        if len(series) < MIN_HISTORY_HOURS:
            print(f"Skipping {client_id}: insufficient history.")
            continue

        clients[client_id] = series.astype(float)

        if position % 25 == 0 or position == len(folders):
            print(
                f"Loaded {position}/{len(folders)} folders; "
                f"usable clients: {len(clients)}"
            )

    print(f"Clients loaded: {len(clients)}")
    return clients


def prepare_client_features(client_id, feature_columns):
    return {
        column: int(column == f"client_id_{client_id}")
        for column in feature_columns
        if column.startswith("client_id_MT_")
    }


def build_feature_values(
    history, target_time, feature_columns, client_features
):
    if len(history) < 24:
        raise ValueError("At least 24 historical hourly values are required.")

    target_time = pd.Timestamp(target_time)
    day_of_week = target_time.day_of_week

    values_by_name = {
        "lag_1h": float(history[-1]),
        "lag_24h": float(history[-24]),
        "rolling_mean_6h": float(np.mean(history[-6:])),
        "rolling_mean_24h": float(np.mean(history[-24:])),
        "hour": target_time.hour,
        "day_of_week": day_of_week,
        "month": target_time.month,
        "is_weekend": int(day_of_week >= 5),
    }

    return [
        values_by_name.get(column, client_features.get(column, 0))
        for column in feature_columns
    ]


def get_test_origins(series):
    number_of_rows = len(series)
    split_index = int(number_of_rows * (1 - TEST_FRACTION))

    first_origin = max(24, split_index)
    last_origin_exclusive = number_of_rows - MAX_HORIZON + 1

    if last_origin_exclusive <= first_origin:
        return np.array([], dtype=int)

    possible_origins = np.arange(
        first_origin,
        last_origin_exclusive,
        ORIGIN_GAP_HOURS,
        dtype=int,
    )

    if len(possible_origins) > MAX_TEST_POINTS:
        selected_positions = np.linspace(
            0,
            len(possible_origins) - 1,
            num=MAX_TEST_POINTS,
            dtype=int,
        )
        possible_origins = possible_origins[selected_positions]

    return possible_origins


def is_client_complete(frame, client_id, expected_origins):
    if frame.empty or expected_origins == 0:
        return False

    client_frame = frame[
        frame["client_id"].astype(str) == client_id
    ]

    expected_rows = expected_origins * MAX_HORIZON

    if len(client_frame) != expected_rows:
        return False

    origin_counts = client_frame.groupby(
        "origin_timestamp"
    )["horizon_hours"].agg(
        ["count", "nunique", "min", "max"]
    )

    if len(origin_counts) != expected_origins:
        return False

    return bool(
        (origin_counts["count"] == MAX_HORIZON).all()
        and (origin_counts["nunique"] == MAX_HORIZON).all()
        and (origin_counts["min"] == 1).all()
        and (origin_counts["max"] == MAX_HORIZON).all()
    )


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "mae_kw": None,
            "rmse_kw": None,
            "r2": None,
            "samples": 0,
        }

    r2 = None
    if len(actual) >= 2 and np.var(actual) > 0:
        r2 = float(r2_score(actual, predicted))

    return {
        "mae_kw": float(mean_absolute_error(actual, predicted)),
        "rmse_kw": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": r2,
        "samples": int(len(actual)),
    }


def save_checkpoint(frame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(CHECKPOINT_PATH)


def build_metrics_table(predictions):
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []

    for horizon in range(1, MAX_HORIZON + 1):
        subset = predictions[
            predictions["horizon_hours"] == horizon
        ]
        actual = subset["actual_kw"].to_numpy(dtype=float)

        for model_name, prediction_column in [
            ("Random Forest", "rf_predicted_kw"),
            ("Persistence baseline", "baseline_predicted_kw"),
        ]:
            rows.append({
                "horizon_hours": horizon,
                "model": model_name,
                **calculate_metrics(
                    actual,
                    subset[prediction_column].to_numpy(dtype=float),
                ),
            })

    metrics_table = pd.DataFrame(rows)

    rf = metrics_table[
        metrics_table["model"] == "Random Forest"
    ].set_index("horizon_hours")

    baseline = metrics_table[
        metrics_table["model"] == "Persistence baseline"
    ].set_index("horizon_hours")

    comparison_rows = []

    for horizon in range(1, MAX_HORIZON + 1):
        rf_mae = rf.loc[horizon, "mae_kw"]
        base_mae = baseline.loc[horizon, "mae_kw"]

        improvement = None
        if pd.notna(base_mae) and base_mae > 0 and pd.notna(rf_mae):
            improvement = float(
                (base_mae - rf_mae) / base_mae * 100
            )

        comparison_rows.append({
            "horizon_hours": horizon,
            "rf_mae_kw": rf_mae,
            "baseline_mae_kw": base_mae,
            "mae_improvement_percent": improvement,
        })

    return metrics_table, pd.DataFrame(comparison_rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("GRIDGUARD: OPTIMIZED 1-24 HOUR FORECAST VALIDATION")
    print("=" * 65)
    print("The saved model and website will not be modified.")
    print("Predictions are batched by horizon and resume from checkpoint.")
    print(f"Output folder: {OUTPUT_DIR}\n")

    model, feature_columns = load_model_bundle()
    clients = load_client_series()

    if not clients:
        raise RuntimeError("No usable client histories were loaded.")

    columns = [
        "client_id",
        "origin_timestamp",
        "target_timestamp",
        "horizon_hours",
        "actual_kw",
        "rf_predicted_kw",
        "baseline_predicted_kw",
    ]

    if CHECKPOINT_PATH.exists():
        checkpoint = pd.read_csv(CHECKPOINT_PATH)

        if not set(columns).issubset(checkpoint.columns):
            raise ValueError(
                "The existing checkpoint has unexpected columns. "
                "It has not been modified."
            )

        checkpoint["client_id"] = checkpoint["client_id"].astype(str)
        print(f"Existing checkpoint rows: {len(checkpoint):,}")
    else:
        checkpoint = pd.DataFrame(columns=columns)
        print("No checkpoint found; starting a new validation.")

    completed_ids = set()

    for client_id, series in clients.items():
        origins = get_test_origins(series)

        if is_client_complete(checkpoint, client_id, len(origins)):
            completed_ids.add(client_id)

    print(f"Clients already complete in checkpoint: {len(completed_ids)}")
    print(f"Clients remaining: {len(clients) - len(completed_ids)}\n")

    # Keep only fully completed clients. Partial clients will be recalculated.
    if not checkpoint.empty:
        checkpoint = checkpoint[
            checkpoint["client_id"].isin(completed_ids)
        ].copy()

    for client_number, (client_id, series) in enumerate(
        clients.items(), start=1
    ):
        if client_id in completed_ids:
            print(
                f"[{client_number}/{len(clients)}] "
                f"{client_id}: already complete; skipping"
            )
            continue

        values = series.to_numpy(dtype=float)
        timestamps = series.index
        origins = get_test_origins(series)

        if len(origins) == 0:
            print(
                f"[{client_number}/{len(clients)}] "
                f"{client_id}: insufficient test history; skipping"
            )
            continue

        client_features = prepare_client_features(
            client_id, feature_columns
        )

        # One simulated history for each forecast origin.
        histories = [
            list(map(float, values[:origin]))
            for origin in origins
        ]

        origin_times = [
            pd.Timestamp(timestamps[origin - 1])
            for origin in origins
        ]

        forecast_by_origin = [[] for _ in origins]

        # Each model call predicts the current horizon for all origins.
        for step in range(1, MAX_HORIZON + 1):
            feature_rows = []

            for origin_index, origin_time in enumerate(origin_times):
                target_time = origin_time + pd.Timedelta(hours=step)

                feature_rows.append(
                    build_feature_values(
                        histories[origin_index],
                        target_time,
                        feature_columns,
                        client_features,
                    )
                )

            batch = pd.DataFrame(
                feature_rows,
                columns=feature_columns,
            )

            batch_predictions = np.asarray(
                model.predict(batch),
                dtype=float,
            )

            batch_predictions = np.maximum(batch_predictions, 0.0)

            for origin_index, prediction in enumerate(batch_predictions):
                forecast_by_origin[origin_index].append(float(prediction))
                histories[origin_index].append(float(prediction))

            # Save partial client progress every six horizons.
            # On restart, incomplete client rows are discarded and rerun.
            if step % 6 == 0:
                partial_rows = []

                for origin_index, origin in enumerate(origins):
                    origin_time = origin_times[origin_index]

                    for horizon in range(1, step + 1):
                        target_index = origin + horizon - 1

                        partial_rows.append({
                            "client_id": client_id,
                            "origin_timestamp": origin_time.isoformat(),
                            "target_timestamp": pd.Timestamp(
                                timestamps[target_index]
                            ).isoformat(),
                            "horizon_hours": horizon,
                            "actual_kw": float(values[target_index]),
                            "rf_predicted_kw": forecast_by_origin[
                                origin_index
                            ][horizon - 1],
                            "baseline_predicted_kw": float(
                                values[origin - 1]
                            ),
                        })

                save_checkpoint(
                    pd.concat(
                        [checkpoint, pd.DataFrame(partial_rows)],
                        ignore_index=True,
                    )
                )

        # Replace partial progress with the completed client results.
        client_rows = []

        for origin_index, origin in enumerate(origins):
            origin_time = origin_times[origin_index]
            baseline_prediction = float(values[origin - 1])

            for horizon in range(1, MAX_HORIZON + 1):
                target_index = origin + horizon - 1

                client_rows.append({
                    "client_id": client_id,
                    "origin_timestamp": origin_time.isoformat(),
                    "target_timestamp": pd.Timestamp(
                        timestamps[target_index]
                    ).isoformat(),
                    "horizon_hours": horizon,
                    "actual_kw": float(values[target_index]),
                    "rf_predicted_kw": forecast_by_origin[
                        origin_index
                    ][horizon - 1],
                    "baseline_predicted_kw": baseline_prediction,
                })

        checkpoint = pd.concat(
            [checkpoint, pd.DataFrame(client_rows)],
            ignore_index=True,
        )

        save_checkpoint(checkpoint)
        completed_ids.add(client_id)

        print(
            f"[{client_number}/{len(clients)}] {client_id}: completed "
            f"({len(origins)} origins, {len(client_rows)} records); "
            f"checkpoint rows: {len(checkpoint):,}"
        )

    if checkpoint.empty:
        raise RuntimeError("No forecast predictions were produced.")

    predictions = checkpoint.copy()
    predictions["horizon_hours"] = pd.to_numeric(
        predictions["horizon_hours"],
        errors="coerce",
    ).astype(int)

    predictions = predictions.sort_values(
        ["client_id", "origin_timestamp", "horizon_hours"]
    ).reset_index(drop=True)

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    metrics_table, comparison_table = build_metrics_table(predictions)
    metrics_table.to_csv(METRICS_PATH, index=False)
    comparison_table.to_csv(COMPARISON_PATH, index=False)

    summary = {
        "project": "GridGuard",
        "validation_type": "recursive multi-horizon forecast",
        "max_horizon_hours": MAX_HORIZON,
        "test_fraction": TEST_FRACTION,
        "max_test_origins_per_client": MAX_TEST_POINTS,
        "minimum_origin_gap_hours": ORIGIN_GAP_HOURS,
        "clients_loaded": len(clients),
        "clients_complete": len(completed_ids),
        "prediction_records": int(len(predictions)),
        "unique_clients_evaluated": int(
            predictions["client_id"].nunique()
        ),
        "unique_origins_evaluated": int(
            predictions[
                ["client_id", "origin_timestamp"]
            ].drop_duplicates().shape[0]
        ),
        "outputs": {
            "predictions": str(PREDICTIONS_PATH),
            "metrics": str(METRICS_PATH),
            "baseline_comparison": str(COMPARISON_PATH),
        },
        "notes": [
            "The saved model was loaded but not retrained or overwritten.",
            "Predictions are recursive; actual future readings are not fed into later forecast steps.",
            "Evaluation targets are in each client's chronological final 20% holdout.",
            "The baseline repeats the most recent observed value.",
            "Forecast origins are spaced at least 24 hourly rows apart.",
            "Incomplete client checkpoint rows are recalculated on resume.",
        ],
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETED")
    print("=" * 65)
    print(
        f"Clients with predictions: "
        f"{predictions['client_id'].nunique()}/{len(clients)}"
    )
    print(f"Prediction records: {len(predictions):,}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Metrics:     {METRICS_PATH}")
    print(f"Comparison:  {COMPARISON_PATH}")
    print(f"Summary:     {SUMMARY_PATH}\n")
    print("Metrics by horizon:")
    print(metrics_table.to_string(index=False))
    print("\nMAE comparison against persistence baseline:")
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()