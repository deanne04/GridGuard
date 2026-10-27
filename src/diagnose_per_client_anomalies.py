
from pathlib import Path

import numpy as np
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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "anomaly_detection"
    / "per_client_diagnostic_report.csv"
)

HIGH_FLAG_RATE_PCT = 20.0


def main():
    for file_path in [INPUT_FILE, PREDICTIONS_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    print("Loading historical data and experimental predictions...")

    data = pd.read_csv(INPUT_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    data["hour_timestamp"] = pd.to_datetime(
        data["hour_timestamp"], errors="coerce"
    )
    predictions["hour_timestamp"] = pd.to_datetime(
        predictions["hour_timestamp"], errors="coerce"
    )

    data = data.dropna(
        subset=["client_id", "hour_timestamp"]
    ).copy()

    predictions = predictions.dropna(
        subset=["client_id", "hour_timestamp"]
    ).copy()

    cutoff = data["hour_timestamp"].quantile(0.80)

    training = data[
        data["hour_timestamp"] < cutoff
    ].copy()

    evaluation = data[
        data["hour_timestamp"] >= cutoff
    ].copy()

    def usage_summary(frame, period_name):
        summary = (
            frame.groupby("client_id")["average_kw"]
            .agg(
                record_count="count",
                mean_kw="mean",
                median_kw="median",
                std_kw="std",
                minimum_kw="min",
                maximum_kw="max",
            )
            .reset_index()
        )

        summary = summary.rename(
            columns={
                "record_count": f"{period_name}_rows",
                "mean_kw": f"{period_name}_mean_kw",
                "median_kw": f"{period_name}_median_kw",
                "std_kw": f"{period_name}_std_kw",
                "minimum_kw": f"{period_name}_min_kw",
                "maximum_kw": f"{period_name}_max_kw",
            }
        )

        return summary

    training_summary = usage_summary(training, "training")
    evaluation_summary = usage_summary(evaluation, "evaluation")

    prediction_summary = (
        predictions.groupby("client_id")
        .agg(
            scored_rows=("potential_anomaly", "size"),
            flagged_count=("potential_anomaly", "sum"),
            mean_anomaly_score=("anomaly_score", "mean"),
        )
        .reset_index()
    )

    report = (
        training_summary
        .merge(evaluation_summary, on="client_id", how="outer")
        .merge(prediction_summary, on="client_id", how="left")
    )

    report["flagged_count"] = report["flagged_count"].fillna(0)
    report["scored_rows"] = report["scored_rows"].fillna(0)

    report["flagged_rate_pct"] = np.where(
        report["scored_rows"] > 0,
        100 * report["flagged_count"] / report["scored_rows"],
        np.nan,
    )

    # Compare the average consumption level between periods.
    report["mean_change_kw"] = (
        report["evaluation_mean_kw"] - report["training_mean_kw"]
    )

    report["mean_change_pct"] = np.where(
        report["training_mean_kw"].abs() >= 1.0,
        100 * report["mean_change_kw"]
        / report["training_mean_kw"].abs(),
        np.nan,
    )

    # Compare variability, where both periods have usable values.
    report["std_change_kw"] = (
        report["evaluation_std_kw"] - report["training_std_kw"]
    )

    report["std_change_pct"] = np.where(
        report["training_std_kw"].abs() >= 1.0,
        100 * report["std_change_kw"]
        / report["training_std_kw"].abs(),
        np.nan,
    )

    report["median_change_kw"] = (
        report["evaluation_median_kw"] - report["training_median_kw"]
    )

    report["evaluation_range_kw"] = (
        report["evaluation_max_kw"] - report["evaluation_min_kw"]
    )

    report["diagnostic_note"] = np.select(
        [
            report["flagged_rate_pct"] >= 80,
            report["flagged_rate_pct"] >= HIGH_FLAG_RATE_PCT,
            report["flagged_rate_pct"] < 2,
        ],
        [
            "Very high flag rate: investigate first",
            "Elevated flag rate: review usage and model behavior",
            "Low flag rate: check whether unusual events may be missed",
        ],
        default="No immediate flag-rate warning",
    )

    report = report.sort_values(
        ["flagged_rate_pct", "flagged_count"],
        ascending=[False, False],
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_FILE, index=False)

    print()
    print("CLIENT USAGE DIAGNOSTIC")
    print(f"Chronological cutoff: {cutoff}")
    print(f"Training records: {len(training):,}")
    print(f"Evaluation records: {len(evaluation):,}")
    print(f"Clients in report: {len(report):,}")
    print(f"Diagnostic report saved to:\n{OUTPUT_FILE}")

    columns_to_show = [
        "client_id",
        "training_rows",
        "evaluation_rows",
        "training_mean_kw",
        "evaluation_mean_kw",
        "mean_change_pct",
        "training_std_kw",
        "evaluation_std_kw",
        "flagged_count",
        "flagged_rate_pct",
        "diagnostic_note",
    ]

    print()
    print("HIGHEST FLAG-RATE CLIENTS")
    print(
        report[columns_to_show]
        .head(15)
        .to_string(
            index=False,
            formatters={
                "training_mean_kw": lambda x: f"{x:.2f}",
                "evaluation_mean_kw": lambda x: f"{x:.2f}",
                "mean_change_pct": lambda x: (
                    f"{x:.2f}%" if pd.notna(x) else "N/A"
                ),
                "training_std_kw": lambda x: (
                    f"{x:.2f}" if pd.notna(x) else "N/A"
                ),
                "evaluation_std_kw": lambda x: (
                    f"{x:.2f}" if pd.notna(x) else "N/A"
                ),
                "flagged_rate_pct": lambda x: (
                    f"{x:.2f}%" if pd.notna(x) else "N/A"
                ),
            },
        )
    )

    print()
    print("CLIENTS WITH THE LARGEST ABSOLUTE MEAN-USAGE CHANGES")
    largest_changes = report.assign(
        absolute_mean_change_pct=report["mean_change_pct"].abs()
    ).sort_values(
        "absolute_mean_change_pct", ascending=False
    )

    change_columns = [
        "client_id",
        "training_mean_kw",
        "evaluation_mean_kw",
        "mean_change_pct",
        "flagged_rate_pct",
    ]

    print(
        largest_changes[change_columns]
        .head(10)
        .to_string(index=False)
    )

    print()
    print(
        "Note: Usage changes and anomaly flags are diagnostic clues, "
        "not proof of faults. A large percentage change can be misleading "
        "when training-period usage is close to zero."
    )


if __name__ == "__main__":
    main()
