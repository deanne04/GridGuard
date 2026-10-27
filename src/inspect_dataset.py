
import csv
from pathlib import Path
from datetime import datetime

file_path = (
    Path(__file__).resolve().parents[1]
    / "data" / "raw" / "LD2011_2014.txt"
)

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

row_count = 0
missing_readings = 0
wrong_length_rows = 0
first_timestamp = None
last_timestamp = None

with file_path.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file, delimiter=";")
    header = next(reader)

    client_count = len(header) - 1
    expected_columns = len(header)

    print("Client columns:", client_count)
    print("Expected columns per row:", expected_columns)
    print("\nScanning dataset...")

    for row in reader:
        if not row:
            continue

        row_count += 1

        if len(row) != expected_columns:
            wrong_length_rows += 1

        timestamp_text = row[0].strip()

        try:
            timestamp = datetime.strptime(
                timestamp_text, "%Y-%m-%d %H:%M:%S"
            )

            if first_timestamp is None:
                first_timestamp = timestamp

            last_timestamp = timestamp

        except ValueError:
            pass

        for value in row[1:]:
            if not value.strip():
                missing_readings += 1

print("\n--- DATASET INSPECTION REPORT ---")
print("Data rows:", row_count)
print("Client columns:", client_count)
print("First timestamp:", first_timestamp)
print("Last timestamp:", last_timestamp)
print("Blank readings:", missing_readings)
print("Rows with unexpected column count:", wrong_length_rows)
print("\nInspection completed.")
