
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "anomaly_detection"
    / "training_sample_anomalies.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "anomaly_detection"
    / "per_client_experimental_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "anomaly_detection"

CLIENTS_TO_INSPECT = [
    "MT_181",
    "MT_116",
    "MT_133",
    "MT_109",
    "MT_111",
    "MT_160",
    "MT_362",
    "MT_370",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {PREDICTIONS_FILE}"
        )

    print("Reading existing historical sample...")
    data = pd.read_csv(INPUT_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    data["hour_timestamp"] = pd.to_datetime(
        data["hour_timestamp"], errors="coerce"
    )
    predictions["hour_timestamp"] = pd.to_datetime(
        predictions["hour_timestamp"], errors="coerce"
    )

    data["average_kw"] = pd.to_numeric(
        data["average_kw"], errors="coerce"
    )

    data = data.dropna(
        subset=["client_id", "hour_timestamp", "average_kw"]
    ).copy()

    predictions = predictions.dropna(
        subset=["client_id", "hour_timestamp"]
    ).copy()

    data["year_month"] = data["hour_timestamp"].dt.to_period("M").astype(str)
    data["year"] = data["hour_timestamp"].dt.year

    cutoff = data["hour_timestamp"].quantile(0.80)

    data["period"] = "Training period"
    data.loc[data["hour_timestamp"] >= cutoff, "period"] = (
        "Evaluation period"
    )

    print()
    print("OVERALL DATA TIMELINE")
    print(f"Rows inspected: {len(data):,}")
    print(f"Clients: {data['client_id'].nunique():,}")
    print(f"Earliest timestamp: {data['hour_timestamp'].min()}")
    print(f"Latest timestamp: {data['hour_timestamp'].max()}")
    print(f"Training/evaluation cutoff: {cutoff}")

    print()
    print("YEARLY CONSUMPTION SUMMARY")
    yearly = (
        data.groupby("year")
        .agg(
            records=("average_kw", "size"),
            clients=("client_id", "nunique"),
            mean_kw=("average_kw", "mean"),
            median_kw=("average_kw", "median"),
            min_kw=("average_kw", "min"),
            max_kw=("average_kw", "max"),
        )
        .reset_index()
    )
    print(yearly.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print()
    print("MONTHLY CONSUMPTION SUMMARY")
    monthly = (
        data.groupby("year_month")
        .agg(
            records=("average_kw", "size"),
            clients=("client_id", "nunique"),
            mean_kw=("average_kw", "mean"),
            median_kw=("average_kw", "median"),
        )
        .reset_index()
    )
    print(monthly.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    monthly.to_csv(
        OUTPUT_DIR / "consumption_shift_monthly_summary.csv",
        index=False,
    )

    print()
    print("CLIENT-LEVEL TRAINING VS EVALUATION SUMMARY")
    client_period = (
        data.groupby(["client_id", "period"])
        .agg(
            rows=("average_kw", "size"),
            mean_kw=("average_kw", "mean"),
            median_kw=("average_kw", "median"),
            std_kw=("average_kw", "std"),
            minimum_kw=("average_kw", "min"),
            maximum_kw=("average_kw", "max"),
        )
        .reset_index()
    )

    means = client_period.pivot(
        index="client_id", columns="period", values="mean_kw"
    )
    means.columns.name = None

    for column in ["Training period", "Evaluation period"]:
        if column not in means.columns:
            means[column] = float("nan")

    means["absolute_change_kw"] = (
        means["Evaluation period"] - means["Training period"]
    )

    # Percentage change is intentionally omitted when the baseline
    # is too close to zero to make the percentage meaningful.
    means["change_pct"] = (
        100 * means["absolute_change_kw"]
        / means["Training period"].abs()
    ).where(means["Training period"].abs() >= 1.0)

    client_means = means.reset_index()

    flag_summary = (
        predictions.groupby("client_id")
        .agg(
            evaluation_rows=("potential_anomaly", "size"),
            flagged_count=("potential_anomaly", "sum"),
        )
        .reset_index()
    )

    flag_summary["flag_rate_pct"] = (
        100 * flag_summary["flagged_count"]
        / flag_summary["evaluation_rows"]
    )

    client_report = client_means.merge(
        flag_summary, on="client_id", how="left"
    )

    client_report = client_report.merge(
        client_period[
            client_period["period"] == "Training period"
        ][["client_id", "rows", "median_kw", "std_kw"]].rename(
            columns={
                "rows": "training_rows",
                "median_kw": "training_median_kw",
                "std_kw": "training_std_kw",
            }
        ),
        on="client_id",
        how="left",
    )

    client_report = client_report.merge(
        client_period[
            client_period["period"] == "Evaluation period"
        ][["client_id", "rows", "median_kw", "std_kw"]].rename(
            columns={
                "rows": "evaluation_rows",
                "median_kw": "evaluation_median_kw",
                "std_kw": "evaluation_std_kw",
            }
        ),
        on="client_id",
        how="left",
    )

    client_report.to_csv(
        OUTPUT_DIR / "consumption_shift_client_summary.csv",
        index=False,
    )

    print()
    print("SELECTED CLIENTS: MONTHLY TREND")
    selected = data[data["client_id"].isin(CLIENTS_TO_INSPECT)]

    selected_monthly = (
        selected.groupby(["client_id", "year_month"])
        .agg(
            rows=("average_kw", "size"),
            mean_kw=("average_kw", "mean"),
            median_kw=("average_kw", "median"),
            max_kw=("average_kw", "max"),
        )
        .reset_index()
    )

    print(
        selected_monthly.to_string(
            index=False,
            formatters={
                "mean_kw": lambda x: f"{x:.2f}",
                "median_kw": lambda x: f"{x:.2f}",
                "max_kw": lambda x: f"{x:.2f}",
            },
        )
    )

    selected_monthly.to_csv(
        OUTPUT_DIR / "consumption_shift_selected_clients_monthly.csv",
        index=False,
    )

    print()
    print("SELECTED CLIENTS: PERIOD COMPARISON")
    selected_report = client_report[
        client_report["client_id"].isin(CLIENTS_TO_INSPECT)
    ].sort_values("flag_rate_pct", ascending=False)

    display_columns = [
        "client_id",
        "Training period",
        "Evaluation period",
        "absolute_change_kw",
        "change_pct",
        "training_median_kw",
        "evaluation_median_kw",
        "training_std_kw",
        "evaluation_std_kw",
        "flagged_count",
        "flag_rate_pct",
    ]

    print(
        selected_report[display_columns].to_string(
            index=False,
            formatters={
                "Training period": lambda x: (
                    f"{x:.2f}" if pd.notna(x) else "N/A"
                ),
                "Evaluation period": lambda x: (
                    f"{x:.2f}" if pd.notna(x) else "N/A"
                ),
                "absolute_change_kw": lambda x: (
                    f"{x:.2f}" if pd.notna(x) else "N/A"
                ),
                "change_pct": lambda x: (
                    f"{x:.2f}%" if pd.notna(x) else "N/A"
                ),
                "training_median_kw": lambda x: (
                    f"{x:.2f}" if pd.notna(x) else "N/A"
                ),
                "evaluation_median_kw": lambda x: (
                    f"{x:.2f}" if pd.notna(x) else "N/A"
                ),
                "training_std_kw": lambda x: (
                    f"{x:.2f}" if pd.notna(x) else "N/A"
                ),
                "evaluation_std_kw": lambda x: (
                    f"{x:.2f}" if pd.notna(x) else "N/A"
                ),
                "flag_rate_pct": lambda x: (
                    f"{x:.2f}%" if pd.notna(x) else "N/A"
                ),
            },
        )
    )

    print()
    print("FILES SAVED")
    print(OUTPUT_DIR / "consumption_shift_monthly_summary.csv")
    print(OUTPUT_DIR / "consumption_shift_client_summary.csv")
    print(
        OUTPUT_DIR
        / "consumption_shift_selected_clients_monthly.csv"
    )

    print()
    print(
        "Interpretation caution: this uses an 8% historical sample, "
        "not the complete raw dataset. A period-level change alone "
        "does not establish a data error or an electrical fault."
    )


if __name__ == "__main__":
    main()
