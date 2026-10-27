from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed" / "client_hourly_history_export"
    / "part-00000-d3dd5877-2043-452b-9b42-ca4ac933178b-c000.csv"
)

OUTPUT = ROOT / "results" / "anomaly_detection" / "verified_client_transitions.csv"

CLIENTS = [
    "MT_109", "MT_111", "MT_116", "MT_133",
    "MT_160", "MT_181", "MT_362", "MT_370",
]


def main():
    print("Reading original hourly history...")
    data = pd.read_csv(
        INPUT,
        usecols=["client_id", "timestamp", "average_kw"],
        dtype={"client_id": "string"},
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )
    data["average_kw"] = pd.to_numeric(
        data["average_kw"], errors="coerce"
    )

    data = data.dropna(
        subset=["client_id", "timestamp", "average_kw"]
    )
    data = data[data["client_id"].isin(CLIENTS)]
    data = data.sort_values(["client_id", "timestamp"])

    results = []

    for client_id, group in data.groupby("client_id"):
        group = group.drop_duplicates("timestamp").reset_index(drop=True)
        positive = group["average_kw"] > 0

        first_positive_index = (
            positive[positive].index[0] if positive.any() else None
        )

        first_sustained_index = None
        for i in range(len(group) - 23):
            if positive.iloc[i:i + 24].all():
                first_sustained_index = i
                break

        if first_positive_index is not None:
            first_positive_time = group.loc[
                first_positive_index, "timestamp"
            ]
        else:
            first_positive_time = pd.NaT

        if first_sustained_index is not None:
            sustained_time = group.loc[
                first_sustained_index, "timestamp"
            ]
        else:
            sustained_time = pd.NaT

        if first_sustained_index is not None:
            before = group.iloc[:first_sustained_index]
            after = group.iloc[first_sustained_index:]
        else:
            before = group.iloc[:0]
            after = group

        # Check the timestamp spacing around the first sustained period.
        if first_sustained_index is not None and first_sustained_index > 0:
            gap_before = (
                group.loc[first_sustained_index, "timestamp"]
                - group.loc[first_sustained_index - 1, "timestamp"]
            ).total_seconds() / 3600
        else:
            gap_before = None

        results.append({
            "client_id": client_id,
            "hourly_rows": len(group),
            "first_positive_timestamp": first_positive_time,
            "first_24_positive_hours_timestamp": sustained_time,
            "zero_pct_before_sustained_period": (
                100 * (before["average_kw"] == 0).mean()
                if len(before) else None
            ),
            "positive_pct_after_sustained_period": (
                100 * (after["average_kw"] > 0).mean()
                if len(after) else None
            ),
            "hours_between_previous_row_and_sustained_start": gap_before,
            "minimum_kw_after_sustained_period": (
                after["average_kw"].min() if len(after) else None
            ),
            "median_kw_after_sustained_period": (
                after["average_kw"].median() if len(after) else None
            ),
        })

    report = pd.DataFrame(results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT, index=False)

    print("\nVERIFIED CLIENT TRANSITIONS")
    print(report.to_string(index=False))
    print(f"\nSaved report: {OUTPUT}")


if __name__ == "__main__":
    main()