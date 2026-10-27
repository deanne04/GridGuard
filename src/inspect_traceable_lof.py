import pandas as pd

path = "results/anomaly_detection/lof_traceable_training/lof_unseen_sample_predictions.csv"
df = pd.read_csv(path)

print("Total records:", f"{len(df):,}")
print("Client ID retained:", "client_id" in df.columns)
print("Timestamp retained:", "hour_timestamp" in df.columns)

flagged = df[df["potential_anomaly"] == True].copy()

print("\nPotential anomalies:", f"{len(flagged):,}")
print("\nFlagged records by client:")
print(flagged["client_id"].value_counts().head(10).to_string())

print("\nFirst 10 flagged records:")
columns = [
    "client_id", "hour_timestamp", "average_kw",
    "previous_hour_kw", "previous_24h_kw",
    "previous_168h_kw", "rolling_mean_24h", "lof_score"
]
print(flagged[columns].head(10).to_string(index=False))
