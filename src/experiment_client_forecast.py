
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "electricity_readings_parquet"
OUTPUT_DIR = ROOT / "results" / "client_forecast_experiment"
MODEL_DIR = ROOT / "models" / "client_forecasts"

CLIENT_ID = "MT_001"
MIN_READINGS_PER_HOUR = 4

FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_6",
    "lag_12", "lag_24", "lag_48", "lag_168",
    "rolling_mean_6", "rolling_mean_24", "rolling_mean_168",
    "hour", "day_of_week", "month", "is_weekend",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("GridGuardClientForecastExperiment")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        print(f"Reading historical data for {CLIENT_ID}...")

        readings = (
            spark.read.parquet(str(DATA_PATH))
            .filter(F.col("client_id") == CLIENT_ID)
            .select("client_id", "timestamp", "reading_kw")
        )

        hourly = (
            readings
            .withColumn("hour_timestamp", F.date_trunc("hour", "timestamp"))
            .groupBy("client_id", "hour_timestamp")
            .agg(
                F.avg("reading_kw").alias("demand_kw"),
                F.count("*").alias("reading_count"),
            )
            .filter(F.col("reading_count") == MIN_READINGS_PER_HOUR)
            .orderBy("hour_timestamp")
        )

        # Only one client's hourly series is collected into Pandas.
        pdf = hourly.select("hour_timestamp", "demand_kw").toPandas()
        pdf["hour_timestamp"] = pd.to_datetime(pdf["hour_timestamp"])
        pdf = pdf.sort_values("hour_timestamp").reset_index(drop=True)

        if len(pdf) < 1000:
            raise RuntimeError(
                f"Not enough complete hourly readings for {CLIENT_ID}: {len(pdf)}"
            )

        # Construct features using this client's own historical demand.
        pdf["hour"] = pdf["hour_timestamp"].dt.hour
        pdf["day_of_week"] = pdf["hour_timestamp"].dt.dayofweek
        pdf["month"] = pdf["hour_timestamp"].dt.month
        pdf["is_weekend"] = (
            pdf["hour_of_week"] if "hour_of_week" in pdf.columns
            else (pdf["day_of_week"] >= 5).astype(int)
        )

        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            pdf[f"lag_{lag}"] = pdf["demand_kw"].shift(lag)

        for window in [6, 24, 168]:
            pdf[f"rolling_mean_{window}"] = (
                pdf["demand_kw"].shift(1).rolling(window).mean()
            )

        # Predict the next hour. Keep time order for a fair evaluation.
        pdf["target_next_hour_kw"] = pdf["demand_kw"].shift(-1)
        pdf = pdf.dropna(subset=FEATURES + ["target_next_hour_kw"]).reset_index(drop=True)

        split_index = int(len(pdf) * 0.8)
        train = pdf.iloc[:split_index].copy()
        test = pdf.iloc[split_index:].copy()

        X_train = train[FEATURES]
        y_train = train["target_next_hour_kw"]
        X_test = test[FEATURES]
        y_test = test["target_next_hour_kw"]

        print(f"Client: {CLIENT_ID}")
        print(f"Usable hourly rows: {len(pdf):,}")
        print(f"Training rows: {len(train):,}")
        print(f"Testing rows: {len(test):,}")
        print("Training Random Forest...")

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=2,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        baseline = test["demand_kw"].to_numpy()

        rf_mae = mean_absolute_error(y_test, predictions)
        baseline_mae = mean_absolute_error(y_test, baseline)

        metrics = {
            "client_id": CLIENT_ID,
            "hourly_records": int(len(pdf)),
            "training_records": int(len(train)),
            "testing_records": int(len(test)),
            "random_forest_mae_kw": float(rf_mae),
            "previous_hour_baseline_mae_kw": float(baseline_mae),
            "mae_improvement_percent": float(
                (baseline_mae - rf_mae) / baseline_mae * 100
            ) if baseline_mae else 0.0,
            "random_forest_rmse_kw": float(
                np.sqrt(mean_squared_error(y_test, predictions))
            ),
            "random_forest_r2": float(r2_score(y_test, predictions)),
            "test_start": str(test["hour_timestamp"].iloc[0]),
            "test_end": str(test["hour_timestamp"].iloc[-1]),
        }

        metrics_path = OUTPUT_DIR / f"{CLIENT_ID.lower()}_individual_metrics.json"
        predictions_path = OUTPUT_DIR / f"{CLIENT_ID.lower()}_individual_predictions.csv"
        model_path = MODEL_DIR / f"{CLIENT_ID.lower()}_individual_model.joblib"

        joblib.dump(
            {"model": model, "features": FEATURES, "client_id": CLIENT_ID},
            model_path,
        )

        output = pd.DataFrame({
            "timestamp": test["hour_timestamp"].astype(str),
            "actual_next_hour_kw": y_test.to_numpy(),
            "random_forest_prediction_kw": predictions,
            "previous_hour_baseline_kw": baseline,
        })
        output.to_csv(predictions_path, index=False)

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

        print("\nEXPERIMENT RESULTS")
        print(json.dumps(metrics, indent=2))
        print(f"\nMetrics: {metrics_path}")
        print(f"Predictions: {predictions_path}")
        print(f"Model: {model_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
