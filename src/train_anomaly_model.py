
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from pyspark.sql import SparkSession, functions as F
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "electricity_readings_parquet"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results" / "anomaly_detection"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 68)
print("GRIDGUARD - AUTOMATIC ELECTRICITY ANOMALY DETECTION")
print("=" * 68)

spark = (
    SparkSession.builder
    .appName("GridGuardAnomalyDetection")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    print("\n1. Reading existing processed Parquet data...")
    readings = spark.read.parquet(str(DATA_PATH))

    print("2. Aggregating readings by client and hour...")
    hourly_spark = (
        readings
        .withColumn("hour_timestamp", F.date_trunc("hour", F.col("timestamp")))
        .groupBy("client_id", "hour_timestamp")
        .agg(
            F.avg("reading_kw").alias("average_kw"),
            F.stddev("reading_kw").alias("within_hour_std_kw"),
            F.count("reading_kw").alias("reading_count")
        )
        .filter(F.col("reading_count") == 4)
        .withColumn("hour", F.hour("hour_timestamp"))
        .withColumn("day_of_week", F.dayofweek("hour_timestamp"))
        .withColumn("month", F.month("hour_timestamp"))
        .withColumn(
            "is_weekend",
            F.when(F.dayofweek("hour_timestamp").isin([1, 7]), 1).otherwise(0)
        )
    )

    # Use Spark to create historical comparison features per client.
    from pyspark.sql.window import Window

    w = Window.partitionBy("client_id").orderBy("hour_timestamp")

    hourly_spark = (
        hourly_spark
        .withColumn("previous_hour_kw", F.lag("average_kw", 1).over(w))
        .withColumn("previous_24h_kw", F.lag("average_kw", 24).over(w))
        .withColumn("previous_168h_kw", F.lag("average_kw", 168).over(w))
        .withColumn(
            "rolling_mean_24h",
            F.avg("average_kw").over(
                w.rowsBetween(-24, -1)
            )
        )
    )

    feature_columns = [
        "average_kw",
        "within_hour_std_kw",
        "previous_hour_kw",
        "previous_24h_kw",
        "previous_168h_kw",
        "rolling_mean_24h",
        "hour",
        "day_of_week",
        "month",
        "is_weekend"
    ]

    print("3. Collecting a manageable training sample...")
    # Deterministic sampling across clients and time.
    # This is a sample for fitting; anomaly scoring can be done on selected
    # client/date ranges in the dashboard later.
    sample_spark = (
        hourly_spark
        .dropna(subset=feature_columns)
        .sample(withReplacement=False, fraction=0.08, seed=42)
    )

    sample = sample_spark.select(
        "client_id", "hour_timestamp", *feature_columns
    ).toPandas()

    if len(sample) < 1000:
        raise ValueError(
            f"Only {len(sample)} training rows collected. "
            "Check the processed dataset and aggregation."
        )

    sample[feature_columns] = sample[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    sample = sample.dropna(subset=feature_columns).copy()

    print(f"   Training sample rows: {len(sample):,}")
    print(f"   Clients represented: {sample['client_id'].nunique()}")

    print("\n4. Fitting Isolation Forest...")
    scaler = StandardScaler()
    X = scaler.fit_transform(sample[feature_columns])

    model = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        random_state=42,
        n_jobs=2
    )
    model.fit(X)

    # Training flags are exploratory only; there are no confirmed labels.
    sample["anomaly_score"] = -model.score_samples(X)
    sample["potential_anomaly"] = (model.predict(X) == -1).astype(int)

    anomaly_count = int(sample["potential_anomaly"].sum())
    anomaly_rate = float(sample["potential_anomaly"].mean() * 100)

    print("\n5. Training-sample anomaly summary")
    print(f"   Rows scored: {len(sample):,}")
    print(f"   Potential anomalies: {anomaly_count:,}")
    print(f"   Potential anomaly rate: {anomaly_rate:.2f}%")
    print("   Note: flags are not confirmed faults or theft.")

    model_path = MODEL_DIR / "gridguard_isolation_forest.joblib"

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "client_column": "client_id",
            "timestamp_column": "hour_timestamp",
            "model_name": "IsolationForest",
            "contamination": 0.02,
            "sample_rows": int(len(sample)),
            "note": (
                "Unsupervised anomaly detection. Flags indicate unusual "
                "patterns, not confirmed faults or theft."
            )
        },
        model_path
    )

    output_path = RESULT_DIR / "training_sample_anomalies.csv"
    sample.to_csv(output_path, index=False)

    summary = {
        "model": "IsolationForest",
        "sample_rows": int(len(sample)),
        "clients_represented": int(sample["client_id"].nunique()),
        "potential_anomalies": anomaly_count,
        "potential_anomaly_rate_percent": round(anomaly_rate, 3),
        "contamination_setting": 0.02,
        "feature_columns": feature_columns,
        "interpretation": (
            "Potential anomalies require review; they are not confirmed faults."
        )
    }

    with open(
        RESULT_DIR / "anomaly_training_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"\n6. Saved model: {model_path}")
    print(f"   Saved training-sample results: {output_path}")
    print(f"   Saved summary: {RESULT_DIR / 'anomaly_training_summary.json'}")
    print("\nANOMALY MODEL TRAINING COMPLETED.")

finally:
    spark.stop()
