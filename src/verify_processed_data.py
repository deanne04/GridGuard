
import os
import sys
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parquet_path = os.path.join(
    project_root,
    "data",
    "processed",
    "electricity_readings_parquet"
)

spark = (
    SparkSession.builder
    .appName("GridGuard-Verify-Processed-Data")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    df = spark.read.parquet(parquet_path)

    print("\nGRIDGUARD SAVED DATA VERIFICATION")
    print("=" * 45)

    print("\nSchema:")
    df.printSchema()

    print("\nFirst 10 saved readings:")
    df.select(
        "timestamp", "client_id", "reading_kw"
    ).show(10, truncate=False)

    print("\nTotal saved readings:", df.count())
    print("Unique clients:", df.select("client_id").distinct().count())

    print("\nVERIFICATION COMPLETED")

finally:
    spark.stop()
