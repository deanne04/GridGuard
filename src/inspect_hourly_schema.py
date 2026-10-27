from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("GridGuardSchemaCheck")
    .master("local[*]")
    .getOrCreate()
)

path = (
    "data/processed/client_hourly_by_client/"
    "client_id=MT_001/*.csv"
)

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(path)
)

print("\n=== MT_001 CSV SCHEMA ===")
df.printSchema()

print("\n=== SAMPLE RECORDS ===")
df.show(5, truncate=False)

print("\n=== TOTAL ROWS FOR MT_001 ===")
print(df.count())

spark.stop()
