import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_DIR = PROJECT_ROOT / "results" / "forecasting"
SUMMARY_DIR = PROJECT_ROOT / "results" / "time_summary"

def load_demand_rows(limit: int = 120) -> list[dict]:
    paths = sorted(SUMMARY_DIR.glob("part-*.csv"))

    if not paths:
        return []

    rows = []

    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows.extend(csv.DictReader(file))

    rows.sort(key=lambda row: row["timestamp"])

    if len(rows) > limit:
        step = max(1, len(rows) // limit)
        rows = rows[::step][:limit]

    return [
        {
            "timestamp": row["timestamp"][:16],
            "average_kw": round(float(row["average_kw"]), 2),
            "total_client_kw": round(float(row["total_client_kw"]), 2),
        }
        for row in rows
    ]





def load_forecast_rows(limit: int = 120) -> list[dict]:
    path = FORECAST_DIR / "forecast_test_predictions.csv"

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    # Keep a representative, chronological sample for the chart.
    if len(rows) > limit:
        step = max(1, len(rows) // limit)
        rows = rows[::step][:limit]

    return [
        {
            "timestamp": row["hour_timestamp"],
            "actual": round(float(row["target_next_hour_kw"]), 2),
            "predicted": round(float(row["predicted_demand_kw"]), 2),
            "baseline": round(float(row["baseline_previous_hour_kw"]), 2),
        }
        for row in rows
    ]



import re


CLIENT_HISTORY_DIR = (
    PROJECT_ROOT / "data" / "processed" / "client_hourly_by_client"
)


def list_available_clients() -> list[str]:
    """Return valid client IDs that have exported hourly data."""
    if not CLIENT_HISTORY_DIR.exists():
        return []

    client_ids = []

    for folder in CLIENT_HISTORY_DIR.iterdir():
        if not folder.is_dir():
            continue

        match = re.fullmatch(r"client_id=(MT_\d+)", folder.name)
        if match and list(folder.glob("*.csv")):
            client_ids.append(match.group(1))

    return sorted(client_ids)


def load_client_hourly_history(client_id: str) -> list[dict]:
    """Load and chronologically sort hourly readings for one client."""
    if not re.fullmatch(r"MT_\d+", client_id):
        return []

    client_dir = CLIENT_HISTORY_DIR / f"client_id={client_id}"

    if not client_dir.is_dir():
        return []

    rows = []

    for path in client_dir.glob("*.csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                try:
                    rows.append({
                        "timestamp": row["timestamp"],
                        "average_kw": float(row["average_kw"]),
                    })
                except (KeyError, ValueError):
                    continue

    rows.sort(key=lambda row: row["timestamp"])
    return rows

