import pandas as pd

path = "results/anomaly_detection/training_sample_anomalies.csv"

columns = [
    "average_kw",
    "within_hour_std_kw",
    "previous_hour_kw",
    "previous_24h_kw",
    "previous_168h_kw",
    "rolling_mean_24h",
]

df = pd.read_csv(path, usecols=columns)

print(f"Sample rows: {len(df):,}")
print("\nZero counts by feature:")
for col in columns:
    zeros = (df[col] == 0).sum()
    print(f"{col:25s}: {zeros:8,} ({zeros / len(df) * 100:.3f}%)")

print("\nRows with previous-hour load = 0:")
print(df.loc[df["previous_hour_kw"] == 0, columns].head(10).to_string(index=False))

print("\nRows with rolling mean = 0:")
print(df.loc[df["rolling_mean_24h"] == 0, columns].head(10).to_string(index=False))
