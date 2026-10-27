import pandas as pd
import numpy as np

path = "results/anomaly_detection/lof_traceable_training/lof_unseen_sample_predictions.csv"
df = pd.read_csv(path)

flagged = df[df["potential_anomaly"] == True].copy()

# Relative changes, protected against division by zero.
flagged["change_vs_previous_hour_pct"] = np.where(
    flagged["previous_hour_kw"] > 0,
    (flagged["average_kw"] - flagged["previous_hour_kw"]).abs()
    / flagged["previous_hour_kw"] * 100,
    np.nan
)

flagged["change_vs_previous_24h_pct"] = np.where(
    flagged["previous_24h_kw"] > 0,
    (flagged["average_kw"] - flagged["previous_24h_kw"]).abs()
    / flagged["previous_24h_kw"] * 100,
    np.nan
)

print("TOTAL FLAGGED RECORDS:", f"{len(flagged):,}")

checks = {
    "Current average load is zero": flagged["average_kw"] == 0,
    "Previous-hour load is zero": flagged["previous_hour_kw"] == 0,
    "Previous-24-hour load is zero": flagged["previous_24h_kw"] == 0,
    "Previous-168-hour load is zero": flagged["previous_168h_kw"] == 0,
    "Within-hour variability is zero": flagged["within_hour_std_kw"] == 0,
    "Load differs from previous hour by over 50%":
        flagged["change_vs_previous_hour_pct"] > 50,
    "Load differs from 24-hour lag by over 50%":
        flagged["change_vs_previous_24h_pct"] > 50,
}

print("\nFLAG CHARACTERISTICS")
for label, mask in checks.items():
    count = int(mask.sum())
    print(f"{label}: {count:,} ({count / len(flagged) * 100:.1f}%)")

print("\nMOST NEGATIVE LOF SCORES (strongest relative outlier signals)")
cols = [
    "client_id", "hour_timestamp", "average_kw",
    "previous_hour_kw", "previous_24h_kw",
    "previous_168h_kw", "rolling_mean_24h", "lof_score"
]
print(flagged.nsmallest(15, "lof_score")[cols].to_string(index=False))

print("\nSaved files and models have not been changed.")
