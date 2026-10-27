from pathlib import Path
import json
import pandas as pd

root = Path.cwd()
source = root / "results" / "anomaly_detection" / "lof_traceable_training" / "lof_unseen_sample_predictions.csv"
output_dir = root / "results" / "anomaly_detection" / "investigation_report"
output_dir.mkdir(parents=True, exist_ok=True)

if not source.exists():
    raise FileNotFoundError(
        f"Could not find: {source}\n"
        "Check the traceable LOF prediction file path before continuing."
    )

df = pd.read_csv(source)
print("Loaded traceable LOF predictions:", len(df))
print("Columns:", list(df.columns))

required = {"client_id", "hour_timestamp"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Required traceability columns are missing: {sorted(missing)}")

flag_col = next(
    (c for c in ["potential_anomaly", "lof_prediction", "prediction", "anomaly"] if c in df.columns),
    None
)
if flag_col is None:
    raise ValueError("Could not identify the LOF prediction column. Check the printed columns.")

flags = df[df[flag_col].astype(str).str.lower().isin(["1", "true", "-1"])].copy()
if flags.empty:
    raise ValueError(
        f"No flagged rows identified using '{flag_col}'. "
        "Check the prediction values before continuing."
    )

flags["hour_timestamp"] = pd.to_datetime(flags["hour_timestamp"], errors="coerce")
flags = flags.dropna(subset=["hour_timestamp"])

flags["investigation_status"] = "Potential anomaly — requires review"
flags["confirmed_fault"] = "Not established by this dataset"

flags.to_csv(output_dir / "traceable_flagged_readings.csv", index=False)

client_summary = (
    flags.groupby("client_id")
    .agg(
        flagged_hours=("hour_timestamp", "count"),
        first_flagged_hour=("hour_timestamp", "min"),
        last_flagged_hour=("hour_timestamp", "max"),
    )
    .reset_index()
    .sort_values("flagged_hours", ascending=False)
)
client_summary.to_csv(output_dir / "flagged_client_summary.csv", index=False)

summary = {
    "source_file": str(source.relative_to(root)),
    "prediction_column": flag_col,
    "rows_reviewed": int(len(df)),
    "flagged_rows_exported": int(len(flags)),
    "unique_clients_flagged": int(flags["client_id"].nunique()),
    "interpretation": "Potential anomalies only; no confirmed anomaly labels are available.",
    "output_files": [
        "traceable_flagged_readings.csv",
        "flagged_client_summary.csv",
        "investigation_summary.json",
    ],
}
with open(output_dir / "investigation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nInvestigation report created:")
print(output_dir)
print("\nFlagged rows:", len(flags))
print("Unique clients flagged:", flags["client_id"].nunique())
print("\nTop 10 clients by flagged hours:")
print(client_summary.head(10).to_string(index=False))
print("\nSaved files:")
for name in summary["output_files"]:
    print(" -", name)
