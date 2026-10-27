
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "electricity_readings_parquet"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results" / "forecasting"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 68)
print("GRIDGUARD - FORECASTING ALGORITHM COMPARISON")
print("=" * 68)

spark = (
    SparkSession.builder
    .appName("GridGuardModelComparison")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

try:
    print("\n1. Reading processed Parquet data...")
    readings = spark.read.parquet(str(DATA_PATH))

    print("2. Creating hourly system-wide average demand...")
    hourly_spark = (
        readings
        .withColumn("hour_timestamp", F.date_trunc("hour", F.col("timestamp")))
        .groupBy("hour_timestamp")
        .agg(
            F.avg("reading_kw").alias("average_demand_kw"),
            F.count("reading_kw").alias("reading_count")
        )
    )

    hourly = hourly_spark.toPandas()
    hourly = hourly[hourly["reading_count"] >= 1480].copy()
    hourly["hour_timestamp"] = pd.to_datetime(hourly["hour_timestamp"])
    hourly = hourly.sort_values("hour_timestamp").reset_index(drop=True)

    # Predict the next hour from information available at the current hour.
    hourly["target_next_hour_kw"] = hourly["average_demand_kw"].shift(-1)

    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        hourly[f"lag_{lag}h"] = hourly["average_demand_kw"].shift(lag)

    for window in [6, 24, 168]:
        hourly[f"rolling_mean_{window}h"] = (
            hourly["average_demand_kw"].shift(1).rolling(window).mean()
        )

    ts = hourly["hour_timestamp"]
    hourly["hour"] = ts.dt.hour
    hourly["day_of_week"] = ts.dt.dayofweek
    hourly["month"] = ts.dt.month
    hourly["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    features = [
        "lag_1h", "lag_2h", "lag_3h", "lag_6h",
        "lag_12h", "lag_24h", "lag_48h", "lag_168h",
        "rolling_mean_6h", "rolling_mean_24h", "rolling_mean_168h",
        "hour", "day_of_week", "month", "is_weekend"
    ]

    data = hourly.dropna(subset=features + ["target_next_hour_kw"]).copy()
    split = int(len(data) * 0.8)

    train = data.iloc[:split]
    test = data.iloc[split:]

    X_train = train[features]
    y_train = train["target_next_hour_kw"]
    X_test = test[features]
    y_test = test["target_next_hour_kw"]

    print(f"   Training records: {len(train):,}")
    print(f"   Testing records:  {len(test):,}")
    print(f"   Training ends:    {train['hour_timestamp'].max()}")
    print(f"   Testing starts:   {test['hour_timestamp'].min()}")

    algorithms = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=2
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            loss="squared_error",
            random_state=42
        ),
        "Linear Regression": LinearRegression()
    }

    results = []
    fitted_models = {}

    print("\n3. Training and evaluating models...")

    for name, model in algorithms.items():
        print(f"\n   Training {name}...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        predictions = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))

        results.append({
            "model": name,
            "MAE_kW": mae,
            "RMSE_kW": rmse,
            "R2": r2,
            "training_seconds": round(train_seconds, 2)
        })
        fitted_models[name] = model

        print(f"      MAE: {mae:.4f} kW")
        print(f"      RMSE: {rmse:.4f} kW")
        print(f"      R²: {r2:.4f}")
        print(f"      Training time: {train_seconds:.2f} seconds")

    # Naive baseline: predict the next hour using the current hour's demand.
    baseline_mae = float(
        mean_absolute_error(y_test, test["lag_1h"].to_numpy())
    )
    print(f"\n   Previous-hour baseline MAE: {baseline_mae:.4f} kW")

    comparison = pd.DataFrame(results).sort_values(
        by=["MAE_kW", "RMSE_kW"], ascending=True
    ).reset_index(drop=True)

    comparison.to_csv(RESULT_DIR / "algorithm_comparison.csv", index=False)

    # Select the best model by MAE on the same chronological test set.
    winner_name = comparison.loc[0, "model"]
    winner_model = fitted_models[winner_name]

    winner_bundle = {
        "model": winner_model,
        "feature_columns": features,
        "target": "next_hour_average_system_demand_kw",
        "training_end": str(train["hour_timestamp"].max()),
        "test_start": str(test["hour_timestamp"].min()),
        "units": "kW",
        "aggregation": "hourly average across available client readings",
        "selected_by": "lowest MAE on chronological holdout test set",
        "model_name": winner_name
    }

    joblib.dump(
        winner_bundle,
        MODEL_DIR / "gridguard_best_forecast_model.joblib"
    )

    winner_predictions = test[
        ["hour_timestamp", "target_next_hour_kw"]
    ].copy()
    winner_predictions["predicted_demand_kw"] = winner_model.predict(X_test)
    winner_predictions["baseline_previous_hour_kw"] = test["lag_1h"].to_numpy()
    winner_predictions.to_csv(
        RESULT_DIR / "best_model_test_predictions.csv", index=False
    )

    summary = {
        "selected_model": winner_name,
        "selection_metric": "MAE",
        "previous_hour_baseline_mae_kw": baseline_mae,
        "test_records": int(len(test)),
        "training_records": int(len(train)),
        "training_end": str(train["hour_timestamp"].max()),
        "test_start": str(test["hour_timestamp"].min()),
        "comparison": comparison.to_dict(orient="records")
    }

    with open(
        RESULT_DIR / "algorithm_comparison_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 68)
    print("FINAL MODEL COMPARISON")
    print("=" * 68)
    print(comparison.to_string(index=False))
    print(f"\nPrevious-hour baseline MAE: {baseline_mae:.4f} kW")
    print(f"Selected model by lowest test MAE: {winner_name}")
    print(f"Saved comparison: {RESULT_DIR / 'algorithm_comparison.csv'}")
    print(f"Saved selected model: {MODEL_DIR / 'gridguard_best_forecast_model.joblib'}")
    print("\nMODEL COMPARISON COMPLETED.")

finally:
    spark.stop()
