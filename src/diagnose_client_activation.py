
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "anomaly_detection" / "training_sample_anomalies.csv"
OUTPUT = ROOT / "results" / "anomaly_detection" / "client_activation_diagnostic.csv"


def main():
    data = pd.read_csv(INPUT)
    data["hour_timestamp"] = pd.to_datetime(
        data["hour_timestamp"], errors="coerce"
    )
    data["average_kw"] = pd.to_numeric(
        data["average_kw"], errors="coerce"
    )

    data = data.dropna(
        subset=["client_id", "hour_timestamp", "average_kw"]
    ).sort_values(["client_id", "hour_timestamp"])

    results = []

    for client_id, group in data.groupby("client_id", sort=True):
        group = group.reset_index(drop=True)
        positive = group["average_kw"] > 0

        # Define an operational start as the first of 24 consecutive
        # sampled records with positive usage. This is a diagnostic
        # proxy, not proof of real-world activation.
        sustained_start = None

        for i in range(len(group) - 23):
            if positive.iloc[i:i + 24].all():
                sustained_start = i
                break

        first_positive = (
            group.loc[positive, "hour_timestamp"].iloc[0]
            if positive.any() else pd.NaT
        )

        if sustained_start is not None:
            activation_time = group.loc[sustained_start, "hour_timestamp"]
            rows_before_start = sustained_start
        else:
            activation_time = pd.NaT
            rows_before_start = None

        results.append({
            "client_id": client_id,
            "sampled_rows": len(group),
            "zero_usage_rows": int((group["average_kw"] == 0).sum()),
            "zero_usage_pct": 100 * (group["average_kw"] == 0).mean(),
            "first_positive_timestamp": first_positive,
            "first_sustained_positive_timestamp": activation_time,
            "sampled_rows_before_sustained_start": rows_before_start,
            "positive_usage_rows": int(positive.sum()),
            "mean_kw": group["average_kw"].mean(),
        })

    report = pd.DataFrame(results)
    report.to_csv(OUTPUT, index=False)

    print("CLIENT ACTIVATION DIAGNOSTIC")
    print(f"Clients examined: {len(report):,}")
    print(f"Clients with no positive readings: {(report['positive_usage_rows'] == 0).sum():,}")
    print(
        "Clients with a sustained positive-usage period: "
        f"{report['first_sustained_positive_timestamp'].notna().sum():,}"
    )
    print(f"Saved report: {OUTPUT}")

    print("\nSELECTED CLIENTS")
    selected = report[
        report["client_id"].isin([
            "MT_181", "MT_116", "MT_133", "MT_109",
            "MT_111", "MT_160", "MT_362", "MT_370"
        ])
    ]
    print(selected.to_string(index=False))

    print("\nCLIENTS WITH THE MOST ZERO-USAGE RECORDS")
    print(
        report.sort_values("zero_usage_pct", ascending=False)
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
