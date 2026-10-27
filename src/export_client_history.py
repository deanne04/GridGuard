
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "electricity_readings_parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "client_hourly_by_client"

spark = (
    SparkSession.builder
    .appName("GridGuardClientHistoryExport")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    # Read only the client IDs first.
    readings = spark.read.parquet(str(PARQUET_PATH))
    client_ids = [
        row["client_id"]
        for row in readings.select("client_id").distinct().collect()
    ]
    client_ids.sort()

    print(f"Found {len(client_ids)} clients to export.", flush=True)

    for index, client_id in enumerate(client_ids, start=1):
        print(
            f"[{index}/{len(client_ids)}] Exporting {client_id}...",
            flush=True,
        )

        client_readings = readings.filter(
            F.col("client_id") == client_id
        )

        hourly = (
            client_readings
            .withColumn(
                "hour_timestamp",
                F.date_trunc("hour", F.col("timestamp")),
            )
            .groupBy("hour_timestamp")
            .agg(
                F.count("reading_kw").alias("reading_count"),
                F.avg("reading_kw").alias("average_kw"),
            )
            .filter(F.col("reading_count") == 4)
            .select(
                F.date_format(
                    "hour_timestamp", "yyyy-MM-dd HH:mm"
                ).alias("timestamp"),
                F.round("average_kw", 4).alias("average_kw"),
            )
        )

        client_output = OUTPUT_DIR / f"client_id={client_id}"

        (
            hourly
            .coalesce(1)
            .write
            .mode("overwrite")
            .option("header", True)
            .csv(str(client_output))
        )

    print("All client histories exported successfully.", flush=True)

finally:
    spark.stop()
