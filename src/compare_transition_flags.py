from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "results" / "anomaly_detection" / "training_sample_anomalies.csv"
OUTPUT_PATH = ROOT / "results" / "anomaly_detection" / "transition_flag_comparison.csv"

transitions = {
    "MT_109": "2014-02-18 00:00:00",
    "MT_111": "2014-01-14 00:00:00",
    "MT_116": "2014-02-18 00:00:00",
    "MT_133": "2014-03-30 02:00:00",
    "MT_160": "2014-02-04 00:00:00",
    "MT_181": "2014-03-05 00:00:00",
    "MT_362": "2012-01-01 00:00:00",
    "MT_370": "2013-01-01 00:00:00",
}

if not SAMPLE_PATH.exists():
    raise FileNotFoundError(f"Training sample not found: {SAMPLE_PATH}")

df = pd.read_csv(SAMPLE_PATH, parse_dates=["hour_timestamp"])
df["hour_timestamp"] = pd.to_datetime(df["hour_timestamp"], errors="coerce")
df["potential_anomaly"] = pd.to_numeric(df["potential_anomaly"], errors="coerce")
df = df.dropna(subset=["client_id", "hour_timestamp", "potential_anomaly"])

rows = []

for client, transition_text in transitions.items():
    transition = pd.Timestamp(transition_text)
    client_df = df[df["client_id"] == client]

    for period, mask in [
        ("Before transition", client_df["hour_timestamp"] < transition),
        ("On/after transition", client_df["hour_timestamp"] >= transition),
    ]:
        part = client_df.loc[mask]
        count = len(part)
        flags = int(part["potential_anomaly"].sum()) if count else 0

        rows.append({
            "client_id": client,
            "transition_timestamp": transition,
            "period": period,
            "sampled_rows": count,
            "potential_anomaly_flags": flags,
            "flag_rate_percent": round(flags / count * 100, 2) if count else None,
        })

result = pd.DataFrame(rows)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)

print("\n=== PRODUCTION TRAINING-SAMPLE FLAG COMPARISON ===")
print(result.to_string(index=False))
print(f"\nSaved report: {OUTPUT_PATH}")
print(
    "\nInterpretation: these are rates in the existing randomly sampled "
    "training results, not full-hourly flag rates or confirmed faults."
)
