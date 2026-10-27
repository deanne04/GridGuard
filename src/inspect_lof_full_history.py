from pathlib import Path
from pyspark.sql import SparkSession, functions as F

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "processed" / "electricity_readings_parquet"

spark = (
    SparkSession.builder
    .appName("GridGuardFlagHistoryCheck")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

try:
    print("Reading full processed Parquet dataset...")
    df = spark.read.parquet(str(DATA_PATH))

    targets = [
        ("MT_130", "2014-11-15 00:00:00"),
        ("MT_131", "2014-08-09 00:00:00"),
        ("MT_363", "2012-01-01 00:00:00"),
    ]

    for client, timestamp in targets:
        print("\n" + "=" * 80)
        print(f"CLIENT: {client} | FLAGGED HOUR: {timestamp}")

        target = F.to_timestamp(F.lit(timestamp))

        history = (
            df.filter(
                (F.col("client_id") == client)
                & (F.col("timestamp") >= target - F.expr("INTERVAL 3 HOURS"))
                & (F.col("timestamp") < target + F.expr("INTERVAL 4 HOURS"))
            )
            .groupBy("client_id", "timestamp")
            .agg(
                F.avg("reading_kw").alias("average_kw"),
                F.stddev_pop("reading_kw").alias("within_hour_std_kw"),
                F.count("*").alias("readings_in_hour"),
            )
            .orderBy("timestamp")
        )

        rows = history.collect()
        if not rows:
            print("No matching rows found.")
        else:
            for row in rows:
                print(row)

finally:
    spark.stop()
    print("\nRead-only history check completed.")
