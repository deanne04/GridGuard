
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT / "results" / "anomaly_detection"
    / "lof_vs_isolation_forest" / "model_predictions.csv"
)
OUTPUT_DIR = (
    ROOT / "results" / "anomaly_detection"
    / "flagged_reading_investigation"
)

FEATURES = [
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


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing comparison results: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["lof_flagged"] = df["lof_prediction"] == -1
    df["if_flagged"] = df["if_prediction"] == -1

    # Lower decision_function scores indicate more unusual records.
    df["lof_score_rank"] = df["lof_score"].rank(pct=True)
    df["if_score_rank"] = df["if_score"].rank(pct=True)

    both = df["lof_flagged"] & df["if_flagged"]
    lof_only = df["lof_flagged"] & ~df["if_flagged"]
    if_only = df["if_flagged"] & ~df["lof_flagged"]
    neither = ~df["lof_flagged"] & ~df["if_flagged"]

    groups = {
        "both_models_flagged": both,
        "lof_only": lof_only,
        "isolation_forest_only": if_only,
        "both_models_normal": neither,
    }

    summary = []
    for name, mask in groups.items():
        summary.append({
            "category": name,
            "record_count": int(mask.sum()),
            "percentage_of_records": round(float(mask.mean() * 100), 3),
        })

    pd.DataFrame(summary).to_csv(
        OUTPUT_DIR / "flagging_group_summary.csv", index=False
    )

    # Save the most unusual examples within each flagged group.
    selections = {
        "both_models_flagged": (
            both,
            (df["lof_score_rank"] + df["if_score_rank"]) / 2,
        ),
        "lof_only": (lof_only, df["lof_score"]),
        "isolation_forest_only": (if_only, df["if_score"]),
    }

    for name, (mask, ranking_score) in selections.items():
        examples = df.loc[mask].copy()
        examples["investigation_rank_score"] = ranking_score.loc[mask]

        # Smaller scores are more unusual within the selected group.
        examples = examples.sort_values("investigation_rank_score")
        columns = FEATURES + [
            "lof_prediction",
            "lof_score",
            "if_prediction",
            "if_score",
            "investigation_rank_score",
        ]
        examples[columns].head(20).to_csv(
            OUTPUT_DIR / f"{name}_top20.csv", index=False
        )

    print("Flagged-reading investigation report created.")
    print("\nGroup summary:")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nOutput folder: {OUTPUT_DIR}")
    print("\nReminder: flagged records are potential anomalies, not confirmed faults.")


if __name__ == "__main__":
    main()
