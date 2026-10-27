from pyspark.sql import SparkSession, functions as F

DATA_PATH = "data/processed/electricity_readings_parquet"

spark = (
    SparkSession.builder
    .appName("GridGuardHourlyGapCheck")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

try:
    print("Reading processed electricity data...")
    readings = spark.read.parquet(DATA_PATH)

    print("Counting readings per client and hour...")
    hourly = (
        readings
        .withColumn("hour_timestamp", F.date_trunc("hour", F.col("timestamp")))
        .groupBy("client_id", "hour_timestamp")
        .agg(F.count("reading_kw").alias("reading_count"))
    )

    print("\nHourly record completeness:")
    hourly.groupBy("reading_count").count().orderBy("reading_count").show(20, False)

    print("\nChecking gaps between consecutive hourly records...")
    from pyspark.sql.window import Window

    w = Window.partitionBy("client_id").orderBy("hour_timestamp")

    gaps = (
        hourly
        .withColumn("previous_timestamp", F.lag("hour_timestamp").over(w))
        .withColumn(
            "gap_hours",
            (F.col("hour_timestamp").cast("long") -
             F.col("previous_timestamp").cast("long")) / 3600
        )
        .filter(
            F.col("previous_timestamp").isNotNull() &
            (F.col("gap_hours") > 1)
        )
    )

    print("Number of client-hour gaps greater than one hour:")
    print(f"{gaps.count():,}")

    print("\nExamples of gaps:")
    gaps.select(
        "client_id", "previous_timestamp", "hour_timestamp", "gap_hours"
    ).orderBy(F.desc("gap_hours")).show(10, False)

finally:
    spark.stop()
