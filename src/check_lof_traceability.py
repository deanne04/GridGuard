import pandas as pd

source = pd.read_csv(
    "results/anomaly_detection/training_sample_anomalies.csv",
    usecols=["client_id", "hour_timestamp"]
)

lof = pd.read_csv(
    "results/anomaly_detection/lof_training/lof_unseen_sample_predictions.csv"
)

print("Original sample rows:", f"{len(source):,}")
print("LOF held-out rows:", f"{len(lof):,}")
print("\nLOF output columns:")
print(lof.columns.tolist())

print("\nDoes the LOF output retain client IDs and timestamps?")
print("client_id:", "client_id" in lof.columns)
print("hour_timestamp:", "hour_timestamp" in lof.columns)

print("\nFirst five LOF prediction rows:")
print(lof.head().to_string(index=False))
