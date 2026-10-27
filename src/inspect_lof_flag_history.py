import pandas as pd

source_path = "results/anomaly_detection/training_sample_anomalies.csv"
lof_path = "results/anomaly_detection/lof_traceable_training/lof_unseen_sample_predictions.csv"

source = pd.read_csv(
    source_path,
    usecols=[
        "client_id", "hour_timestamp", "average_kw",
        "previous_hour_kw", "previous_24h_kw",
        "previous_168h_kw", "rolling_mean_24h"
    ],
    parse_dates=["hour_timestamp"]
)

lof = pd.read_csv(lof_path)
lof["hour_timestamp"] = pd.to_datetime(lof["hour_timestamp"])

# Select three of the strongest LOF flags.
targets = [
    ("MT_130", "2014-11-15 00:00:00"),
    ("MT_131", "2014-08-09 00:00:00"),
    ("MT_363", "2012-01-01 00:00:00"),
]

for client, timestamp in targets:
    target_time = pd.Timestamp(timestamp)
    print("\n" + "=" * 85)
    print(f"CLIENT: {client} | FLAGGED TIME: {timestamp}")

    history = source[
        (source["client_id"] == client)
        & (source["hour_timestamp"] >= target_time - pd.Timedelta(hours=3))
        & (source["hour_timestamp"] <= target_time + pd.Timedelta(hours=3))
    ].sort_values("hour_timestamp")

    if history.empty:
        print("No matching surrounding records found in the original feature sample.")
    else:
        print(history.to_string(index=False))

    flagged = lof[
        (lof["client_id"] == client)
        & (lof["hour_timestamp"] == target_time)
    ]
    print("LOF flag exists in held-out predictions:", not flagged.empty)

print("\nNote: the source file is a sampled historical feature dataset, not the complete raw time series.")
