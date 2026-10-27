
import csv
from pathlib import Path
from datetime import datetime

project_dir = Path(__file__).resolve().parents[1]
input_file = project_dir / "data" / "raw" / "LD2011_2014.txt"
output_file = project_dir / "data" / "processed" / "electricity_sample.csv"

output_file.parent.mkdir(parents=True, exist_ok=True)

with input_file.open("r", encoding="utf-8-sig", newline="") as source:
    reader = csv.reader(source, delimiter=";")
    header = next(reader)

    clients = header[1:]
    sample_rows = []

    for row in reader:
        if not row:
            continue

        timestamp = datetime.strptime(
            row[0].strip(), "%Y-%m-%d %H:%M:%S"
        )

        for client, raw_value in zip(clients, row[1:]):
            value = float(raw_value.strip().replace(",", "."))

            sample_rows.append([
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                client,
                value
            ])

        if len(sample_rows) >= 1000:
            break

with output_file.open("w", encoding="utf-8", newline="") as target:
    writer = csv.writer(target)
    writer.writerow(["timestamp", "client_id", "reading"])
    writer.writerows(sample_rows)

print("Sample created:", output_file)
print("Sample readings:", len(sample_rows))
print("\nFirst 5 sample rows:")

with output_file.open("r", encoding="utf-8", newline="") as file:
    for line in list(file)[:6]:
        print(line.rstrip())
