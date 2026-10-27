
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from pyspark.sql import SparkSession, functions as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "electricity_readings_parquet"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results" / "forecasting"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("GRIDGUARD - ELECTRICITY DEMAND FORECASTING")
print("=" * 65)

spark = (
    SparkSession.builder
    .appName("GridGuardForecasting")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    print("\n1. Reading existing processed Parquet data...")
    readings = spark.read.parquet(str(DATA_PATH))

    print("2. Aggregating electricity readings into hourly demand...")

    # The source contains kW readings every 15 minutes.
    # Average client kW values across each hour to obtain system-wide
    # average demand in kW. A timestamp's 370 clients are averaged.
    hourly_spark = (
        readings
        .withColumn("hour_timestamp", F.date_trunc("hour", F.col("timestamp")))
        .groupBy("hour_timestamp")
        .agg(
            F.avg("reading_kw").alias("average_demand_kw"),
            F.count("reading_kw").alias("reading_count")
        )
        .orderBy("hour_timestamp")
    )

    hourly = hourly_spark.toPandas()
    hourly_spark.unpersist(blocking=False)

    if hourly.empty:
        raise ValueError("No hourly readings were produced. Check the Parquet data.")

    hourly["hour_timestamp"] = pd.to_datetime(hourly["hour_timestamp"])
    hourly = hourly.sort_values("hour_timestamp").reset_index(drop=True)

    # Exclude incomplete hours; a complete hour should have 4 readings
    # per client, for 370 * 4 = 1480 readings.
    hourly = hourly[hourly["reading_count"] >= 1480].copy()
    hourly = hourly.sort_values("hour_timestamp").reset_index(drop=True)

    if len(hourly) < 1000:
        raise ValueError(
            f"Only {len(hourly)} complete hourly records found; "
            "not enough for this training setup."
        )

    print(f"   Complete hourly records: {len(hourly):,}")
    print(f"   From: {hourly['hour_timestamp'].min()}")
    print(f"   To:   {hourly['hour_timestamp'].max()}")

    # Target is the next hour's average system-wide demand.
    # All lag features use only information available at the current hour.
    hourly["target_next_hour_kw"] = hourly["average_demand_kw"].shift(-1)

    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        hourly[f"lag_{lag}h"] = hourly["average_demand_kw"].shift(lag)

    hourly["rolling_mean_6h"] = (
        hourly["average_demand_kw"].shift(1).rolling(6).mean()
    )
    hourly["rolling_mean_24h"] = (
        hourly["average_demand_kw"].shift(1).rolling(24).mean()
    )
    hourly["rolling_mean_168h"] = (
        hourly["average_demand_kw"].shift(1).rolling(168).mean()
    )

    timestamp = hourly["hour_timestamp"]
    hourly["hour"] = timestamp.dt.hour
    hourly["day_of_week"] = timestamp.dt.dayofweek
    hourly["month"] = timestamp.dt.month
    hourly["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)

    feature_columns = [
        "lag_1h", "lag_2h", "lag_3h", "lag_6h",
        "lag_12h", "lag_24h", "lag_48h", "lag_168h",
        "rolling_mean_6h", "rolling_mean_24h", "rolling_mean_168h",
        "hour", "day_of_week", "month", "is_weekend"
    ]

    model_data = hourly.dropna(
        subset=feature_columns + ["target_next_hour_kw"]
    ).copy()

    split_index = int(len(model_data) * 0.8)
    train_data = model_data.iloc[:split_index].copy()
    test_data = model_data.iloc[split_index:].copy()

    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError("Training or testing data is empty.")

    X_train = train_data[feature_columns]
    y_train = train_data["target_next_hour_kw"]
    X_test = test_data[feature_columns]
    y_test = test_data["target_next_hour_kw"]

    print("\n3. Training Random Forest Regression model...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=2
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    baseline = test_data["lag_1h"].to_numpy()
    baseline_mae = float(mean_absolute_error(y_test, baseline))

    print("\n4. Evaluation on later, unseen timestamps")
    print(f"   Training records: {len(train_data):,}")
    print(f"   Testing records:  {len(test_data):,}")
    print(f"   MAE:              {mae:.4f} kW")
    print(f"   RMSE:             {rmse:.4f} kW")
    print(f"   R²:               {r2:.4f}")
    print(f"   Previous-hour baseline MAE: {baseline_mae:.4f} kW")

    model_path = MODEL_DIR / "gridguard_demand_forecast.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target": "next_hour_average_system_demand_kw",
            "training_end": str(train_data["hour_timestamp"].max()),
            "test_start": str(test_data["hour_timestamp"].min()),
            "units": "kW",
            "aggregation": "hourly average across available client readings"
        },
        model_path
    )

    metrics = {
        "model": "RandomForestRegressor",
        "target": "next-hour average system-wide demand (kW)",
        "training_records": int(len(train_data)),
        "testing_records": int(len(test_data)),
        "training_end": str(train_data["hour_timestamp"].max()),
        "test_start": str(test_data["hour_timestamp"].min()),
        "mae_kw": mae,
        "rmse_kw": rmse,
        "r2": r2,
        "previous_hour_baseline_mae_kw": baseline_mae
    }

    with open(RESULT_DIR / "forecast_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    comparison = test_data[["hour_timestamp", "target_next_hour_kw"]].copy()
    comparison["predicted_demand_kw"] = predictions
    comparison["baseline_previous_hour_kw"] = baseline
    comparison.to_csv(RESULT_DIR / "forecast_test_predictions.csv", index=False)

    print(f"\n5. Saved model:   {model_path}")
    print(f"   Saved metrics: {RESULT_DIR / 'forecast_metrics.json'}")
    print(f"   Saved test predictions: {RESULT_DIR / 'forecast_test_predictions.csv'}")
    print("\nFORECAST MODEL TRAINING COMPLETED.")

finally:
    spark.stop()
