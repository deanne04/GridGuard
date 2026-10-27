
from pathlib import Path
import json

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "electricity_readings_parquet"
OUTPUT_DIR = ROOT / "results" / "client_forecast_experiment"
CLIENT_ID = "MT_001"

FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_6",
    "lag_12", "lag_24", "lag_48", "lag_168",
    "rolling_mean_6", "rolling_mean_24", "rolling_mean_168",
    "hour", "day_of_week", "month", "is_weekend",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("GridGuardCompareClientModels")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        readings = (
            spark.read.parquet(str(DATA_PATH))
            .filter(F.col("client_id") == CLIENT_ID)
            .select("timestamp", "reading_kw")
        )

        hourly = (
            readings
            .withColumn("hour_timestamp", F.date_trunc("hour", "timestamp"))
            .groupBy("hour_timestamp")
            .agg(
                F.avg("reading_kw").alias("demand_kw"),
                F.count("*").alias("reading_count"),
            )
            .filter(F.col("reading_count") == 4)
            .orderBy("hour_timestamp")
        )

        df = hourly.select("hour_timestamp", "demand_kw").toPandas()
        df["hour_timestamp"] = pd.to_datetime(df["hour_timestamp"])
        df = df.sort_values("hour_timestamp").reset_index(drop=True)

        df["hour"] = df["hour_timestamp"].dt.hour
        df["day_of_week"] = df["hour_timestamp"].dt.dayofweek
        df["month"] = df["hour_timestamp"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            df[f"lag_{lag}"] = df["demand_kw"].shift(lag)

        for window in [6, 24, 168]:
            df[f"rolling_mean_{window}"] = (
                df["demand_kw"].shift(1).rolling(window).mean()
            )

        df["target_next_hour_kw"] = df["demand_kw"].shift(-1)
        df = df.dropna(subset=FEATURES + ["target_next_hour_kw"]).reset_index(drop=True)

        # Ensure hourly rows are consecutive so lag features represent real hours.
        consecutive = df["hour_timestamp"].diff().dropna().eq(pd.Timedelta(hours=1))
        if not consecutive.all():
            print("Warning: some hourly gaps exist; lag features may cross gaps.")

        split = int(len(df) * 0.8)
        train = df.iloc[:split]
        test = df.iloc[split:]

        X_train, y_train = train[FEATURES], train["target_next_hour_kw"]
        X_test, y_test = test[FEATURES], test["target_next_hour_kw"]
        results = []

        # The previous observed hour is the baseline.
        baseline = test["demand_kw"].to_numpy()
        results.append({
            "model": "Previous-hour baseline",
            "mae_kw": mean_absolute_error(y_test, baseline),
            "rmse_kw": np.sqrt(mean_squared_error(y_test, baseline)),
            "r2": r2_score(y_test, baseline),
        })

        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=100, max_depth=16,
                min_samples_leaf=2, random_state=42, n_jobs=2,
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=100, max_depth=16,
                min_samples_leaf=2, random_state=42, n_jobs=2,
            ),
            "Histogram Gradient Boosting": HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.08,
                max_leaf_nodes=31, l2_regularization=1.0,
                random_state=42,
            ),
        }

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)

            results.append({
                "model": name,
                "mae_kw": mean_absolute_error(y_test, prediction),
                "rmse_kw": np.sqrt(mean_squared_error(y_test, prediction)),
                "r2": r2_score(y_test, prediction),
            })

        result_df = pd.DataFrame(results).sort_values("mae_kw")
        output_path = OUTPUT_DIR / "mt_001_model_comparison.csv"
        result_df.to_csv(output_path, index=False)

        print("\nMODEL COMPARISON — lower MAE is better")
        print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nSaved results to: {output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
