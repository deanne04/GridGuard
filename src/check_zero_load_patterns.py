from pyspark.sql import SparkSession, functions as F

DATA_PATH = "data/processed/electricity_readings_parquet"

spark = (
    SparkSession.builder
    .appName("GridGuardZeroLoadCheck")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

try:
    readings = spark.read.parquet(DATA_PATH)

    hourly = (
        readings
        .withColumn("hour_timestamp", F.date_trunc("hour", F.col("timestamp")))
        .groupBy("client_id", "hour_timestamp")
        .agg(
            F.avg("reading_kw").alias("average_kw"),
            F.count("reading_kw").alias("reading_count")
        )
        .filter(F.col("reading_count") == 4)
    )

    client_summary = (
        hourly.groupBy("client_id")
        .agg(
            F.count("*").alias("complete_hours"),
            F.sum(F.when(F.col("average_kw") == 0, 1).otherwise(0)).alias("zero_hours"),
            F.min("hour_timestamp").alias("first_hour"),
            F.max("hour_timestamp").alias("last_hour")
        )
        .withColumn(
            "zero_percent",
            F.round(F.col("zero_hours") / F.col("complete_hours") * 100, 2)
        )
    )

    print("Clients with the highest percentage of zero-load hours:")
    client_summary.orderBy(F.desc("zero_percent")).show(10, False)

    print("\nExample timestamps for the client with the highest zero percentage:")
    top_client = (
        client_summary.orderBy(F.desc("zero_percent"))
        .select("client_id").first()["client_id"]
    )

    (
        hourly.filter(F.col("client_id") == top_client)
        .filter(F.col("average_kw") == 0)
        .orderBy("hour_timestamp")
        .select("client_id", "hour_timestamp", "average_kw")
        .show(20, False)
    )

finally:
    spark.stop()
