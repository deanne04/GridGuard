
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
file_path = project_dir / "data" / "processed" / "electricity_sample.csv"

df = pd.read_csv(file_path)

print("\n--- SAMPLE SUMMARY ---")
print("Rows:", len(df))
print("Unique clients:", df["client_id"].nunique())
print("Unique timestamps:", df["timestamp"].nunique())

print("\n--- READING STATISTICS ---")
print(df["reading"].describe())

print("\nZero readings:", (df["reading"] == 0).sum())
print("Negative readings:", (df["reading"] < 0).sum())
print("Missing readings:", df["reading"].isna().sum())

print("\n--- HIGHEST READINGS IN SAMPLE ---")
print(
    df.nlargest(10, "reading")[
        ["timestamp", "client_id", "reading"]
    ].to_string(index=False)
)

print("\n--- READINGS PER CLIENT ---")
print(df.groupby("client_id").size().head(10))
