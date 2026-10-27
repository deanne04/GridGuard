import pandas as pd

source_path = "results/anomaly_detection/training_sample_anomalies.csv"
lof_path = "results/anomaly_detection/lof_training/lof_unseen_sample_predictions.csv"

features = [
    "average_kw",
    "within_hour_std_kw",
    "previous_hour_kw",
    "previous_24h_kw",
    "previous_168h_kw",
    "rolling_mean_24h",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

source = pd.read_csv(source_path, usecols=["client_id", "hour_timestamp", *features])
lof = pd.read_csv(lof_path, usecols=features + ["lof_prediction", "lof_score", "potential_anomaly"])

print("Checking exact feature matches...")
source_counts = source.groupby(features, dropna=False).size().reset_index(name="source_matches")
matched = lof.merge(source_counts, on=features, how="left")

print("LOF rows:", f"{len(lof):,}")
print("Rows with no exact source match:", f"{matched['source_matches'].isna().sum():,}")
print("Rows with multiple source matches:", f"{(matched['source_matches'] > 1).sum():,}")
print("Rows with exactly one source match:", f"{(matched['source_matches'] == 1).sum():,}")

print("\nExample rows with multiple source matches:")
print(matched.loc[matched["source_matches"] > 1, features + ["source_matches"]].head(5).to_string(index=False))
