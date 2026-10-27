
import os
import sys
from pathlib import Path

# 1. Configure Python for Spark
python_exe = sys.executable
os.environ["PYSPARK_PYTHON"] = python_exe
os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe

# 2. Locate the real dataset
project_dir = Path(__file__).resolve().parent.parent
data_file = project_dir / "data" / "raw" / "LD2011_2014.txt"

if not data_file.exists():
    raise FileNotFoundError(f"Dataset not found: {data_file}")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("GridGuard-Real-Data-Test")
    .master("local[2]")
    .config("spark.pyspark.python", python_exe)
    .config("spark.pyspark.driver.python", python_exe)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("\n========== GRIDGUARD REAL DATA TEST ==========")
    print("Dataset:", data_file)
    print("File size (MB):", round(data_file.stat().st_size / (1024 * 1024), 2))

    # Read only the first 20 data lines for this initial test.
    # The source file uses semicolons and decimal commas.
    with data_file.open("r", encoding="utf-8-sig") as file:
        header = file.readline().strip()
        lines = []

        for _ in range(20):
            line = file.readline()
            if not line:
                break
            lines.append(line.strip())

    # Parse the header and values.
    columns = [
        item.strip().strip('"')
        for item in header.split(";")
    ]

    rows = []
    for line in lines:
        values = [
            item.strip().strip('"')
            for item in line.split(";")
        ]

        if len(values) != len(columns):
            print("Skipped malformed line with", len(values), "fields")
            continue

        timestamp = values[0]
        for client_id, value in zip(columns[1:], values[1:]):
            # Decimal comma is the decimal separator in this dataset.
            reading = float(value.replace(",", "."))
            rows.append((timestamp, client_id, reading))

    print("Client columns:", len(columns) - 1)
    print("Sample timestamps:", len(lines))
    print("Sample readings prepared:", len(rows))

    # Create a Spark DataFrame from the small sample.
    df = spark.createDataFrame(
        rows,
        ["timestamp", "client_id", "reading_kw"]
    )

    print("\nSample records:")
    df.show(10, truncate=False)

    print("Total readings:", df.count())

    print("\nSummary statistics:")
    df.select(
        F.count("*").alias("reading_count"),
        F.countDistinct("client_id").alias("client_count"),
        F.avg("reading_kw").alias("average_kw"),
        F.min("reading_kw").alias("minimum_kw"),
        F.max("reading_kw").alias("maximum_kw")
    ).show(truncate=False)

    print("RESULT: Real dataset sample processed successfully.")

finally:
    spark.stop()
    print("Spark session stopped.")
