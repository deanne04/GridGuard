
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from sklearn.ensemble import HistGradientBoostingRegressor
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
        .appName("GridGuardSeasonalBaselineExperiment")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        print(f"Loading hourly readings for {CLIENT_ID}...")

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

        pdf = hourly.select("hour_timestamp", "demand_kw").toPandas()
        pdf["hour_timestamp"] = pd.to_datetime(pdf["hour_timestamp"])
        pdf = pdf.sort_values("hour_timestamp").drop_duplicates(
            "hour_timestamp"
        )

        # Reindex to a complete hourly timeline. Missing hours remain NaN,
        # preventing lag features from silently crossing gaps.
        pdf = pdf.set_index("hour_timestamp")
        full_index = pd.date_range(
            start=pdf.index.min().ceil("h"),
            end=pdf.index.max().floor("h"),
            freq="h",
        )
        pdf = pdf.reindex(full_index)
        pdf.index.name = "hour_timestamp"
        pdf["demand_kw"] = pd.to_numeric(pdf["demand_kw"], errors="coerce")

        # Predict demand for the next hour.
        pdf["target_next_hour_kw"] = pdf["demand_kw"].shift(-1)

        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            pdf[f"lag_{lag}"] = pdf["demand_kw"].shift(lag)

        for window in [6, 24, 168]:
            pdf[f"rolling_mean_{window}"] = (
                pdf["demand_kw"].shift(1).rolling(
                    window=window, min_periods=window
                ).mean()
            )

        pdf["hour"] = pdf.index.hour
        pdf["day_of_week"] = pdf.index.dayofweek
        pdf["month"] = pdf.index.month
        pdf["is_weekend"] = (pdf["day_of_week"] >= 5).astype(int)

        # These baselines predict target time t+1 using the corresponding
        # reading at target time minus 24 or 168 hours.
        pdf["previous_hour_baseline_kw"] = pdf["demand_kw"]
        pdf["previous_day_baseline_kw"] = pdf["demand_kw"].shift(23)
        pdf["previous_week_baseline_kw"] = pdf["demand_kw"].shift(167)

        required = FEATURES + [
            "target_next_hour_kw",
            "previous_hour_baseline_kw",
            "previous_day_baseline_kw",
            "previous_week_baseline_kw",
        ]
        valid = pdf.dropna(subset=required).copy()

        if len(valid) < 1000:
            raise RuntimeError(
                f"Not enough complete hourly rows after gap checks: {len(valid)}"
            )

        # Chronological split; never randomly shuffle future hours into training.
        split_index = int(len(valid) * 0.8)
        train = valid.iloc[:split_index].copy()
        test = valid.iloc[split_index:].copy()

        print(f"Valid complete rows: {len(valid):,}")
        print(f"Training rows: {len(train):,}")
        print(f"Testing rows: {len(test):,}")
        print("Training Histogram Gradient Boosting...")

        model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(train[FEATURES], train["target_next_hour_kw"])
        ml_predictions = model.predict(test[FEATURES])

        actual = test["target_next_hour_kw"].to_numpy()

        predictions = {
            "Previous-hour baseline": test[
                "previous_hour_baseline_kw"
            ].to_numpy(),
            "Previous-day same-hour baseline": test[
                "previous_day_baseline_kw"
            ].to_numpy(),
            "Previous-week same-hour baseline": test[
                "previous_week_baseline_kw"
            ].to_numpy(),
            "Histogram Gradient Boosting": ml_predictions,
        }

        results = []
        for name, pred in predictions.items():
            results.append({
                "model": name,
                "mae_kw": mean_absolute_error(actual, pred),
                "rmse_kw": np.sqrt(mean_squared_error(actual, pred)),
                "r2": r2_score(actual, pred),
            })

        results_df = pd.DataFrame(results).sort_values("mae_kw")
        output_path = OUTPUT_DIR / "mt_001_seasonal_baseline_comparison.csv"
        results_df.to_csv(output_path, index=False)

        print("\nSEASONAL BASELINE COMPARISON — lower MAE is better")
        print(
            results_df.to_string(
                index=False, float_format=lambda value: f"{value:.4f}"
            )
        )
        print(f"\nTest period: {test.index.min()} to {test.index.max()}")
        print(f"Saved results to: {output_path}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
