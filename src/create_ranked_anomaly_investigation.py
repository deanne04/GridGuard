from pathlib import Path
import numpy as np
import pandas as pd

root = Path.cwd()
source = root / "results" / "anomaly_detection" / "investigation_report" / "traceable_flagged_readings.csv"
output_dir = root / "results" / "anomaly_detection" / "ranked_investigation"
output_dir.mkdir(parents=True, exist_ok=True)

if not source.exists():
    raise FileNotFoundError(f"Missing input file: {source}")

df = pd.read_csv(source)
required = [
    "client_id", "hour_timestamp", "average_kw",
    "previous_hour_kw", "previous_24h_kw",
    "rolling_mean_24h", "lof_score"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

numeric_cols = [
    "average_kw", "previous_hour_kw", "previous_24h_kw",
    "previous_168h_kw", "rolling_mean_24h", "lof_score"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# LOF scores are more unusual when they are more negative.
df = df.sort_values("lof_score", ascending=True).reset_index(drop=True)
df.insert(0, "investigation_rank", np.arange(1, len(df) + 1))

def percent_change(current, reference):
    if pd.isna(current) or pd.isna(reference):
        return np.nan
    if reference == 0:
        return np.nan
    return ((current - reference) / abs(reference)) * 100

df["change_from_previous_hour_pct"] = [
    percent_change(a, b)
    for a, b in zip(df["average_kw"], df["previous_hour_kw"])
]
df["change_from_previous_24h_pct"] = [
    percent_change(a, b)
    for a, b in zip(df["average_kw"], df["previous_24h_kw"])
]

def explain(row):
    notes = []
    current = row["average_kw"]
    previous = row["previous_hour_kw"]
    prev24 = row["previous_24h_kw"]
    rolling = row["rolling_mean_24h"]

    if pd.notna(current) and pd.notna(previous):
        if previous == 0 and current > 0:
            notes.append("Load returned after a zero previous-hour reading")
        elif previous > 0 and current == 0:
            notes.append("Current load is zero after a non-zero previous hour")
        elif previous > 0 and abs(current - previous) / abs(previous) >= 0.5:
            notes.append("Load changed by at least 50% from the previous hour")

    if pd.notna(current) and pd.notna(prev24):
        if prev24 == 0 and current > 0:
            notes.append("Current load is non-zero while the 24-hour lag is zero")
        elif prev24 > 0 and abs(current - prev24) / abs(prev24) >= 0.5:
            notes.append("Load differs by at least 50% from the 24-hour lag")

    if pd.notna(current) and pd.notna(rolling) and rolling > 0:
        if abs(current - rolling) / abs(rolling) >= 0.5:
            notes.append("Load differs by at least 50% from the rolling mean")

    if not notes:
        notes.append("LOF identified a local-density outlier; inspect the history")

    return "; ".join(notes)

df["investigation_note"] = df.apply(explain, axis=1)
df["interpretation"] = "Potential anomaly only — not a confirmed fault"

# Main report: every flagged row, ranked by LOF score.
df.to_csv(output_dir / "ranked_anomaly_investigation.csv", index=False)

# Smaller report: the 25 most negative LOF scores for manual review.
df.head(25).to_csv(output_dir / "top_25_for_manual_review.csv", index=False)

summary = pd.DataFrame([{
    "flagged_records": len(df),
    "unique_clients": df["client_id"].nunique(),
    "most_negative_lof_score": df["lof_score"].min(),
    "least_negative_lof_score": df["lof_score"].max(),
    "ranking_rule": "More negative LOF score ranks higher for investigation",
    "limitation": "These are model-ranked potential anomalies, not confirmed faults."
}])
summary.to_csv(output_dir / "ranking_summary.csv", index=False)

print("Ranked investigation report created successfully.")
print("Output folder:", output_dir)
print("\nSummary:")
print(summary.to_string(index=False))
print("\nTop 10 records for review:")
cols = [
    "investigation_rank", "client_id", "hour_timestamp",
    "average_kw", "previous_hour_kw", "previous_24h_kw",
    "lof_score", "investigation_note"
]
print(df[cols].head(10).to_string(index=False))
print("\nFiles created:")
print("- ranked_anomaly_investigation.csv")
print("- top_25_for_manual_review.csv")
print("- ranking_summary.csv")
