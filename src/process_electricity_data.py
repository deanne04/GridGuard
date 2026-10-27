
import os
import sys
from pathlib import Path

# ============================================================
# GRIDGUARD: FULL ELECTRICITY DATA PROCESSING
# Uses PySpark local mode. No Hadoop cluster required.
# ============================================================

# 1. Configure Python for Spark
python_exe = sys.executable
os.environ["PYSPARK_PYTHON"] = python_exe
os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe

# 2. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_DIR / "data" / "raw" / "LD2011_2014.txt"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

PARQUET_DIR = PROCESSED_DIR / "electricity_readings_parquet"
CLIENT_SUMMARY_DIR = RESULTS_DIR / "client_summary"
TIME_SUMMARY_DIR = RESULTS_DIR / "time_summary"

# 3. Check input file
if not RAW_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

# Create output parent directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 4. Start Spark with conservative local settings
spark = (
    SparkSession.builder
    .appName("GridGuard-Full-Electricity-Processing")
    .master("local[2]")
    .config("spark.pyspark.python", python_exe)
    .config("spark.pyspark.driver.python", python_exe)
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.files.maxPartitionBytes", str(32 * 1024 * 1024))
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    print("\n" + "=" * 65)
    print("GRIDGUARD FULL DATASET PROCESSING")
    print("=" * 65)
    print("Python:", python_exe)
    print("Spark:", spark.version)
    print("Mode: local[2]")
    print("Input file:", RAW_FILE)
    print("Input size (MB):", round(RAW_FILE.stat().st_size / (1024 * 1024), 2))

    # 5. Read the source file.
    # The first column is the timestamp; other columns are clients.
    # The dataset is semicolon-separated and readings use decimal commas.
    wide_df = (
        spark.read
        .option("header", "true")
        .option("sep", ";")
        .option("encoding", "UTF-8")
        .option("inferSchema", "false")
        .option("mode", "FAILFAST")
        .csv(str(RAW_FILE))
    )

    # Remove quotation marks and whitespace from column names.
    original_columns = wide_df.columns
    cleaned_names = [
        name.strip().strip('"')
        for name in original_columns
    ]

    for old_name, new_name in zip(original_columns, cleaned_names):
        if old_name != new_name:
            wide_df = wide_df.withColumnRenamed(old_name, new_name)

    timestamp_column = cleaned_names[0]
    client_columns = cleaned_names[1:]

    print("Client columns found:", len(client_columns))
    print("Timestamp column:", timestamp_column)

    if len(client_columns) != 370:
        raise ValueError(
            f"Expected 370 client columns but found {len(client_columns)}. "
            "Check the source header before continuing."
        )

    # 6. Convert the wide table into long format entirely in Spark.
    # Stack produces one row per timestamp/client combination.
    stack_items = ", ".join(
        f"'{client}', `{client}`"
        for client in client_columns
    )

    long_df = wide_df.selectExpr(
        f"CAST(`{timestamp_column}` AS STRING) AS timestamp_text",
        f"stack({len(client_columns)}, {stack_items}) "
        "AS (client_id, reading_text)"
    )

    # 7. Parse timestamps and decimal-comma readings.
    # Do not discard zeros: they may be valid readings.
    cleaned_df = (
        long_df
        .withColumn(
            "timestamp",
            F.to_timestamp("timestamp_text", "yyyy-MM-dd HH:mm:ss")
        )
        .withColumn(
            "reading_kw",
            F.regexp_replace(
                F.trim(F.col("reading_text")),
                ",",
                "."
            ).cast("double")
        )
        .drop("timestamp_text", "reading_text")
    )

    # Remove only rows that could not be parsed.
    # Zero and negative values are retained for inspection.
    invalid_df = cleaned_df.filter(
        F.col("timestamp").isNull()
        | F.col("reading_kw").isNull()
        | F.isnan("reading_kw")
    )

    invalid_count = invalid_df.count()
    print("Unparseable readings:", invalid_count)

    valid_df = cleaned_df.filter(
        F.col("timestamp").isNotNull()
        & F.col("reading_kw").isNotNull()
        & ~F.isnan("reading_kw")
    )

    # Add reusable time features.
    valid_df = (
        valid_df
        .withColumn("date", F.to_date("timestamp"))
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("day_of_week", F.dayofweek("timestamp"))
        .withColumn("month", F.month("timestamp"))
        .withColumn(
            "is_weekend",
            F.dayofweek("timestamp").isin([1, 7])
        )
    )

    # 8. Save the processed data.
    # Partition by year and month to support later time-based queries.
    valid_df = (
        valid_df
        .withColumn("year", F.year("timestamp"))
    )

    print("\nWriting processed readings to Parquet...")
    print("Destination:", PARQUET_DIR)

    (
        valid_df
        .repartition(8, "year", "month")
        .write
        .mode("overwrite")
        .partitionBy("year", "month")
        .parquet(str(PARQUET_DIR))
    )

    # 9. Calculate client-level summaries.
    print("\nCalculating client summaries...")

    client_summary = (
        valid_df
        .groupBy("client_id")
        .agg(
            F.count("*").alias("reading_count"),
            F.avg("reading_kw").alias("average_kw"),
            F.min("reading_kw").alias("minimum_kw"),
            F.max("reading_kw").alias("maximum_kw"),
            F.stddev("reading_kw").alias("stddev_kw"),
            F.sum(
                F.when(F.col("reading_kw") == 0, 1).otherwise(0)
            ).alias("zero_reading_count")
        )
        .orderBy("client_id")
    )

    client_summary.write.mode("overwrite").option(
        "header", "true"
    ).csv(str(CLIENT_SUMMARY_DIR))

    # 10. Calculate hourly summaries across all clients.
    print("Calculating hourly summaries...")

    time_summary = (
        valid_df
        .groupBy("timestamp")
        .agg(
            F.count("*").alias("client_reading_count"),
            F.avg("reading_kw").alias("average_kw"),
            F.sum("reading_kw").alias("total_client_kw"),
            F.min("reading_kw").alias("minimum_kw"),
            F.max("reading_kw").alias("maximum_kw")
        )
        .orderBy("timestamp")
    )

    time_summary.write.mode("overwrite").option(
        "header", "true"
    ).csv(str(TIME_SUMMARY_DIR))

    # 11. Print final validation metrics.
    print("\nValidating saved output...")

    saved_df = spark.read.parquet(str(PARQUET_DIR))

    total_readings = saved_df.count()
    total_clients = saved_df.select("client_id").distinct().count()
    total_timestamps = saved_df.select("timestamp").distinct().count()

    print("\n" + "=" * 65)
    print("GRIDGUARD PROCESSING SUMMARY")
    print("=" * 65)
    print("Valid readings:", f"{total_readings:,}")
    print("Unique clients:", total_clients)
    print("Unique timestamps:", f"{total_timestamps:,}")
    print("Unparseable readings excluded:", f"{invalid_count:,}")
    print("Parquet output:", PARQUET_DIR)
    print("Client summary:", CLIENT_SUMMARY_DIR)
    print("Time summary:", TIME_SUMMARY_DIR)

    print("\nOverall reading statistics:")
    saved_df.select(
        F.avg("reading_kw").alias("average_kw"),
        F.min("reading_kw").alias("minimum_kw"),
        F.max("reading_kw").alias("maximum_kw")
    ).show(truncate=False)

    print("RESULT: GridGuard processing pipeline completed.")

finally:
    spark.stop()
    print("Spark session stopped.")
