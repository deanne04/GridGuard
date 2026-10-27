from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = (
    SparkSession.builder
    .appName("GridGuardZeroRunCheck")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

try:
    readings = spark.read.parquet("data/processed/electricity_readings_parquet")

    hourly = (
        readings
        .withColumn("hour_timestamp", F.date_trunc("hour", F.col("timestamp")))
        .groupBy("client_id", "hour_timestamp")
        .agg(
            F.avg("reading_kw").alias("average_kw"),
            F.count("reading_kw").alias("reading_count")
        )
        .filter(F.col("reading_count") == 4)
        .withColumn("is_zero", (F.col("average_kw") == 0).cast("int"))
    )

    w = Window.partitionBy("client_id").orderBy("hour_timestamp")

    runs = (
        hourly
        .withColumn("previous_zero", F.lag("is_zero").over(w))
        .withColumn(
            "new_run",
            F.when(
                F.col("previous_zero").isNull() |
                (F.col("is_zero") != F.col("previous_zero")),
                1
            ).otherwise(0)
        )
        .withColumn(
            "run_id",
            F.sum("new_run").over(w.rowsBetween(
                Window.unboundedPreceding, Window.currentRow
            ))
        )
    )

    zero_runs = (
        runs.filter(F.col("is_zero") == 1)
        .groupBy("client_id", "run_id")
        .agg(
            F.count("*").alias("consecutive_zero_hours"),
            F.min("hour_timestamp").alias("start_time"),
            F.max("hour_timestamp").alias("end_time")
        )
    )

    print("Longest consecutive zero-load periods for selected clients:")
    zero_runs.filter(
        F.col("client_id").isin(["MT_178", "MT_133", "MT_181"])
    ).orderBy(
        "client_id", F.desc("consecutive_zero_hours")
    ).show(10, False)

    print("Longest zero-load periods across all clients:")
    zero_runs.orderBy(F.desc("consecutive_zero_hours")).show(10, False)

finally:
    spark.stop()
