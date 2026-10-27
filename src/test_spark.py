
import os
import sys
from pathlib import Path

# --------------------------------------------------
# 1. Configure GridGuard's Python environment
# --------------------------------------------------

python_exe = sys.executable
project_dir = Path(__file__).resolve().parent.parent

pyspark_dir = (
    Path(python_exe).parent
    / "Lib"
    / "site-packages"
    / "pyspark"
)

os.environ["PYSPARK_PYTHON"] = python_exe
os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe

# Set SPARK_HOME only when the expected folder exists.
if pyspark_dir.is_dir():
    os.environ["SPARK_HOME"] = str(pyspark_dir)

# --------------------------------------------------
# 2. Import PySpark
# --------------------------------------------------

from pyspark.sql import SparkSession

# --------------------------------------------------
# 3. Start Spark in local mode
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("GridGuard-Spark-Test")
    .master("local[2]")
    .config("spark.pyspark.python", python_exe)
    .config("spark.pyspark.driver.python", python_exe)
    .getOrCreate()
)

try:
    # --------------------------------------------------
    # 4. Create sample electricity readings
    # --------------------------------------------------

    data = [
        ("MT_001", 10.5),
        ("MT_002", 25.0),
        ("MT_003", 18.2),
    ]

    df = spark.createDataFrame(
        data,
        ["client_id", "reading_kw"]
    )

    # --------------------------------------------------
    # 5. Display test results
    # --------------------------------------------------

    print("\n========== GRIDGUARD SPARK TEST ==========")
    print("Project directory:", project_dir)
    print("Python executable:", python_exe)
    print("PySpark version:", spark.version)
    print("Spark execution mode: Local (2 worker threads)")

    record_count = df.count()
    print("Test records:", record_count)

    print("\nSample electricity readings:")
    df.show()

    # Verify Spark can perform an aggregation.
    print("Average reading (kW):", df.agg(
        {"reading_kw": "avg"}
    ).first()[0])

    if record_count == 3:
        print("\nRESULT: GridGuard Spark test PASSED.")
    else:
        print("\nRESULT: GridGuard Spark test FAILED.")

finally:
    # --------------------------------------------------
    # 6. Stop Spark safely
    # --------------------------------------------------

    spark.stop()
    print("GridGuard Spark session stopped.")
