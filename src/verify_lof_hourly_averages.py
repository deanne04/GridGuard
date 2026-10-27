from pathlib import Path
from pyspark.sql import SparkSession, functions as F

data_path = Path.cwd() / "data" / "processed" / "electricity_readings_parquet"

spark = (
    SparkSession.builder
    .appName("GridGuardCorrectHourlyCheck")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

try:
    df = spark.read.parquet(str(data_path))

    targets = [
        ("MT_130", "2014-11-15 00:00:00"),
        ("MT_131", "2014-08-09 00:00:00"),
        ("MT_363", "2012-01-01 00:00:00"),
    ]

    hourly = (
        df.withColumn("hour_timestamp", F.date_trunc("hour", F.col("timestamp")))
        .groupBy("client_id", "hour_timestamp")
        .agg(
            F.avg("reading_kw").alias("average_kw"),
            F.stddev_pop("reading_kw").alias("within_hour_std_kw"),
            F.count("*").alias("readings_in_hour")
        )
    )

    for client, timestamp in targets:
        target = F.to_timestamp(F.lit(timestamp))
        print("\n" + "=" * 75)
        print(f"{client} around {timestamp}")

        rows = (
            hourly.filter(
                (F.col("client_id") == client)
                & (F.col("hour_timestamp") >= target - F.expr("INTERVAL 3 HOURS"))
                & (F.col("hour_timestamp") <= target + F.expr("INTERVAL 3 HOURS"))
            )
            .orderBy("hour_timestamp")
            .collect()
        )

        for row in rows:
            print(row)

finally:
    spark.stop()
    print("\nHourly aggregation check completed.")
