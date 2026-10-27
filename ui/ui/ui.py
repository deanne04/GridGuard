
import csv
import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import asyncio
import reflex as rx

from ui.demand_data import (
    load_demand_rows,
    load_forecast_rows,
    list_available_clients,
    load_client_hourly_history,
)
from ui.forecast_service import forecast_future



PROJECT_ROOT = Path(__file__).resolve().parents[2]

HORIZON_COMPARISON_PATH = (
    PROJECT_ROOT
    / "results"
    / "forecast_horizon_validation"
    / "horizon_baseline_comparison.csv"
)


def load_horizon_comparison():
    """Load saved Random Forest vs baseline horizon-validation results."""
    if not HORIZON_COMPARISON_PATH.exists():
        return []

    try:
        with HORIZON_COMPARISON_PATH.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            rows = list(csv.DictReader(file))

        results = []
        for row in rows:
            results.append({
                "horizon_hours": int(row["horizon_hours"]),
                "rf_mae_kw": float(row["rf_mae_kw"]),
                "baseline_mae_kw": float(row["baseline_mae_kw"]),
                "mae_improvement_percent": float(
                    row["mae_improvement_percent"]
                ),
            })

        return sorted(
            results,
            key=lambda row: row["horizon_hours"],
        )
    except (OSError, ValueError, KeyError):
        return []


AVAILABLE_CLIENTS = list_available_clients()

def load_client_forecast_data():
    """Load historical client metrics, all lookup rows, and chart samples."""
    experiment_dir = (
        PROJECT_ROOT / "results" / "client_forecast_experiment"
    )
    metrics_path = experiment_dir / "mt_001_metrics.csv"
    predictions_path = experiment_dir / "mt_001_test_predictions.csv"

    metrics = {}
    all_rows = []
    chart_rows = []

    try:
        with metrics_path.open("r", newline="", encoding="utf-8") as file:
            metrics = next(csv.DictReader(file), {})

        with predictions_path.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            source_rows = list(csv.DictReader(file))

        for row in source_rows:
            all_rows.append(
                {
                    "timestamp": row["timestamp"][:16],
                    "actual": float(row["actual_demand_kw"]),
                    "baseline": float(row["baseline_prediction_kw"]),
                    "predicted": float(
                        row["random_forest_prediction_kw"]
                    ),
                }
            )

        # Keep the overview chart readable.
        step = max(1, len(all_rows) // 120)
        chart_rows = all_rows[::step]

    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f"Could not load client forecast data: {error}")

    return metrics, chart_rows, all_rows


CLIENT_FORECAST_METRICS, CLIENT_FORECAST_ROWS, CLIENT_FORECAST_ALL_ROWS = (
    load_client_forecast_data()
)



class GridGuardState(rx.State):
    active_page: str = "Overview"


    forecast_date: str = "2026-10-25"
    forecast_hour: str = "09"
    forecast_horizon: str = "24"
    forecast_message: str = ""
    forecast_success: bool = False
    forecast_predicted_kw: str = ""
    forecast_current_kw: str = ""
    forecast_selected_hour: str = ""
    forecast_target_hour: str = ""
    forecast_average_kw: str = ""
    forecast_peak_kw: str = ""
    forecast_minimum_kw: str = ""
    forecast_chart_data: list[dict[str, str | float]] = []

    
        # Past electricity usage lookup
    past_date: str = "2014-03-16"
    past_hour: str = "05"
    past_message: str = ""
    past_success: bool = False
    past_actual_kw: str = ""
    past_ai_kw: str = ""
    past_baseline_kw: str = ""
    past_ai_difference_kw: str = ""
    past_baseline_difference_kw: str = ""
    past_chart_data: list[dict[str, str | float]] = []
        # Per-client anomaly checker
    anomaly_client_id: str = "MT_001"
    anomaly_date: str = "2014-12-30"
    anomaly_hour: str = "12"
    anomaly_average_kw: str = ""
    anomaly_std_kw: str = ""
    anomaly_message: str = ""
    anomaly_result: str = ""
    anomaly_score: str = ""
    anomaly_success: bool = False
        # Selected client history
    selected_client: str = "MT_001"
    client_history_data: list[dict[str, str | float]] = []
    client_history_message: str = "Choose a client and load its history."
    client_latest_timestamp: str = ""
    client_latest_usage_kw: str = ""
    client_next_hour_kw: str = ""
        # Client-specific past usage
    client_past_message: str = ""
    client_past_success: bool = False
    client_past_results_visible: bool = True
    client_past_actual_kw: str = ""
    client_past_previous_hour_kw: str = ""
    client_past_previous_day_kw: str = ""
    client_past_chart_data: list[dict[str, str | float]] = []

    # Client-specific future forecast
    client_forecast_date: str = "2015-01-01"
    client_forecast_hour: str = "00"
    client_forecast_horizon: str = "24"
    client_forecast_message: str = ""
    client_forecast_success: bool = False
    client_forecast_results_visible: bool = True
    client_forecast_average_kw: str = ""
    client_forecast_peak_kw: str = ""
    client_forecast_minimum_kw: str = ""
    client_forecast_chart_data: list[dict[str, str | float]] = []
    def set_anomaly_client(self, value: str):
        if value in AVAILABLE_CLIENTS:
            self.anomaly_client_id = value
            self.anomaly_message = ""
            self.anomaly_result = ""
            self.anomaly_score = ""
            self.anomaly_success = False
    def set_anomaly_date(self, value: str):
        self.anomaly_date = value
        self.anomaly_message = ""
        self.anomaly_result = ""
        self.anomaly_score = ""
        self.anomaly_success = False

    def set_anomaly_hour(self, value: str):
        self.anomaly_hour = value
        self.anomaly_message = ""
        self.anomaly_result = ""
        self.anomaly_score = ""
        self.anomaly_success = False

    def set_anomaly_average(self, value: str):
        self.anomaly_average_kw = value
        self.anomaly_message = ""
        self.anomaly_result = ""
        self.anomaly_score = ""
        self.anomaly_success = False

    def set_anomaly_std(self, value: str):
        self.anomaly_std_kw = value
        self.anomaly_message = ""
        self.anomaly_result = ""
        self.anomaly_score = ""
        self.anomaly_success = False

    def check_client_anomaly(self):
        import math

        self.anomaly_message = ""
        self.anomaly_result = ""
        self.anomaly_score = ""
        self.anomaly_success = False

        try:
            target = datetime.strptime(
                f"{self.anomaly_date} {self.anomaly_hour}",
                "%Y-%m-%d %H",
            )
            average_kw = float(self.anomaly_average_kw)
            std_kw = float(self.anomaly_std_kw)

            if not math.isfinite(average_kw) or average_kw < 0:
                raise ValueError("Average kW must be a valid non-negative number.")
            if not math.isfinite(std_kw) or std_kw < 0:
                raise ValueError(
                    "Within-hour standard deviation must be non-negative."
                )
        except (TypeError, ValueError) as error:
            self.anomaly_message = f"Check your inputs: {error}"
            return

        try:
            rows = load_client_hourly_history(self.anomaly_client_id)
            if not rows:
                self.anomaly_message = (
                    "No historical readings were found for this client."
                )
                return

            # Use only records strictly before the selected hour.
            earlier = [
                row for row in rows
                if datetime.strptime(
                    row["timestamp"][:16], "%Y-%m-%d %H:%M"
                ) < target
            ]

            if len(earlier) < 168:
                self.anomaly_message = (
                    "At least 168 earlier hourly readings are required "
                    "to calculate the model's historical features."
                )
                return

            previous_hour = float(earlier[-1]["average_kw"])
            previous_24h = float(earlier[-24]["average_kw"])
            previous_168h = float(earlier[-168]["average_kw"])
            rolling_mean = float(np.mean([
                float(row["average_kw"]) for row in earlier[-24:]
            ]))

            # Spark dayofweek convention: Sunday=1, Saturday=7.
            spark_day = ((target.weekday() + 1) % 7) + 1
            is_weekend = int(spark_day in (1, 7))

            feature_values = {
                "average_kw": average_kw,
                "within_hour_std_kw": std_kw,
                "previous_hour_kw": previous_hour,
                "previous_24h_kw": previous_24h,
                "previous_168h_kw": previous_168h,
                "rolling_mean_24h": rolling_mean,
                "hour": target.hour,
                "day_of_week": spark_day,
                "month": target.month,
                "is_weekend": is_weekend,
            }

            model_path = (
                PROJECT_ROOT / "models" / "gridguard_isolation_forest.joblib"
            )
            if not model_path.exists():
                self.anomaly_message = "The saved anomaly model was not found."
                return

            bundle = joblib.load(model_path)
            features = bundle["feature_columns"]
            X = np.array(
                [[feature_values[name] for name in features]],
                dtype=float,
            )

            if not np.isfinite(X).all():
                self.anomaly_message = (
                    "The calculated model features contain invalid values."
                )
                return

            scaled = bundle["scaler"].transform(X)
            prediction = int(bundle["model"].predict(scaled)[0])
            score = float(-bundle["model"].score_samples(scaled)[0])

            self.anomaly_result = (
                "Potential Anomaly" if prediction == -1 else "Normal"
            )
            self.anomaly_score = f"{score:.4f}"
            self.anomaly_success = True
            self.anomaly_message = (
                f"Checked {self.anomaly_client_id} at "
                f"{target:%Y-%m-%d %H}:00 using earlier client history."
            )

        except Exception as error:
            self.anomaly_message = (
                "The anomaly check could not be completed. "
                f"Details: {error}"
            )
    def set_selected_client(self, value: str):
        if value in AVAILABLE_CLIENTS:
            self.selected_client = value
            self.client_history_data = []
            self.client_history_message = (
                f"Selected {value}. Click Load Client History to continue."
            )
            self.client_latest_timestamp = ""
            self.client_latest_usage_kw = ""
            self.client_next_hour_kw = ""

    def load_selected_client_history(self):
        self.client_history_data = []
        self.client_latest_timestamp = ""
        self.client_latest_usage_kw = ""
        self.client_next_hour_kw = ""

        rows = load_client_hourly_history(self.selected_client)

        if not rows:
            self.client_history_message = (
                "No historical readings were found for this client."
            )
            return

        latest = rows[-1]
        step = max(1, len(rows) // 120)
        chart_rows = rows[::step]

        if chart_rows[-1]["timestamp"] != latest["timestamp"]:
            chart_rows.append(latest)

        self.client_history_data = [
            {
                "timestamp": row["timestamp"],
                "average_kw": round(row["average_kw"], 4),
            }
            for row in chart_rows
        ]

        self.client_latest_timestamp = latest["timestamp"]
        self.client_latest_usage_kw = f"{latest['average_kw']:.4f}"
        self.client_next_hour_kw = f"{latest['average_kw']:.4f}"

        self.client_history_message = (
            f"Loaded {len(rows):,} hourly records for "
            f"{self.selected_client}."
        )
    def set_past_date(self, value: str):
        self.past_date = value

    def set_past_hour(self, value: str):
        self.past_hour = value
    def check_past_usage(self):
        """Find a recorded hour and compare its electricity estimates."""
        self.past_success = False
        self.past_message = ""
        self.past_actual_kw = ""
        self.past_ai_kw = ""
        self.past_baseline_kw = ""
        self.past_ai_difference_kw = ""
        self.past_baseline_difference_kw = ""
        self.past_chart_data = []

        selected_timestamp = f"{self.past_date} {self.past_hour}:00"

        matching_index = next(
            (
                index
                for index, row in enumerate(CLIENT_FORECAST_ALL_ROWS)
                if row["timestamp"] == selected_timestamp
            ),
            None,
        )

        if matching_index is None:
            self.past_message = (
                "No saved record was found for that date and hour. "
                "Please choose an hour within the available historical period."
            )
            return

        row = CLIENT_FORECAST_ALL_ROWS[matching_index]

        actual = row["actual"]
        ai_estimate = row["predicted"]
        baseline_estimate = row["baseline"]

        self.past_actual_kw = f"{actual:.3f}"
        self.past_ai_kw = f"{ai_estimate:.3f}"
        self.past_baseline_kw = f"{baseline_estimate:.3f}"
        self.past_ai_difference_kw = f"{abs(actual - ai_estimate):.3f}"
        self.past_baseline_difference_kw = (
            f"{abs(actual - baseline_estimate):.3f}"
        )

        start = max(0, matching_index - 12)
        end = min(len(CLIENT_FORECAST_ALL_ROWS), matching_index + 13)

        self.past_chart_data = CLIENT_FORECAST_ALL_ROWS[start:end]
        self.past_success = True
        self.past_message = (
            f"Historical results found for {selected_timestamp}."
        )

    def check_selected_client_past_usage(self):
        """Look up historical readings for the selected client."""
        self.client_past_success = False
        self.client_past_message = ""
        self.client_past_actual_kw = ""
        self.client_past_previous_hour_kw = ""
        self.client_past_previous_day_kw = ""
        self.client_past_chart_data = []

        rows = load_client_hourly_history(self.selected_client)
        selected_timestamp = f"{self.past_date} {self.past_hour}:00"

        index_by_timestamp = {
            row["timestamp"]: index
            for index, row in enumerate(rows)
        }
        index = index_by_timestamp.get(selected_timestamp)

        if index is None:
            self.client_past_message = (
                f"No hourly reading was found for {self.selected_client} "
                f"at {selected_timestamp}. Choose a date within the "
                "client's available historical period."
            )
            return
        actual = float(rows[index]["average_kw"])
        self.client_past_actual_kw = f"{actual:.4f}"

        self.client_past_previous_hour_kw = (
            f'{float(rows[index - 1]["average_kw"]):.4f}'
            if index >= 1 else "N/A"
        )
        self.client_past_previous_day_kw = (
            f'{float(rows[index - 24]["average_kw"]):.4f}'
            if index >= 24 else "N/A"
        )

        start = max(0, index - 12)
        end = min(len(rows), index + 13)

        self.client_past_chart_data = [
            {
                "timestamp": row["timestamp"],
                "actual": round(float(row["average_kw"]), 4),
            }
            for row in rows[start:end]
        ]

        self.client_past_success = True
        self.client_past_message = (
            f"Historical reading found for {self.selected_client} "
            f"at {selected_timestamp}."
        )
    def toggle_client_past_results(self):
        self.client_past_results_visible = (
            not self.client_past_results_visible
        )
    def toggle_client_forecast_results(self):
        self.client_forecast_results_visible = (
            not self.client_forecast_results_visible
        )
    def set_client_forecast_date(self, value: str):
        self.client_forecast_date = value
        self.reset_client_forecast()

    def set_client_forecast_hour(self, value: str):
        self.client_forecast_hour = value
        self.reset_client_forecast()

    def set_client_forecast_horizon(self, value: str):
        self.client_forecast_horizon = value
        self.reset_client_forecast()

    def reset_client_forecast(self):
        self.client_forecast_message = ""
        self.client_forecast_success = False
        self.client_forecast_average_kw = ""
        self.client_forecast_peak_kw = ""
        self.client_forecast_minimum_kw = ""
        self.client_forecast_chart_data = []

    def run_selected_client_forecast(self):
        """Create a client-specific forecast from its recent daily pattern."""
        import pandas as pd

        self.reset_client_forecast()

        try:
            horizon = int(self.client_forecast_horizon)
            start = pd.Timestamp(
                f"{self.client_forecast_date} "
                f"{int(self.client_forecast_hour):02d}:00"
            )
        except (TypeError, ValueError):
            self.client_forecast_message = (
                "Choose a valid forecast date and hour."
            )
            return

        if not 1 <= horizon <= 24:
            self.client_forecast_message = (
                "Choose a forecast horizon between 1 and 24 hours."
            )
            return

        rows = load_client_hourly_history(self.selected_client)

        if len(rows) < 24:
            self.client_forecast_message = (
                "At least 24 hourly readings are required for this client."
            )
            return

        last_timestamp = pd.Timestamp(rows[-1]["timestamp"])

        if start <= last_timestamp:
            self.client_forecast_message = (
                f"Choose a start time after the historical data ends "
                f"({last_timestamp:%Y-%m-%d %H:%M})."
            )
            return

        # Use the latest 24 actual hours as a repeating daily reference.
        daily_pattern = [
            float(row["average_kw"])
            for row in rows[-24:]
        ]

        predictions = []

        for step in range(horizon):
            target = start + pd.Timedelta(hours=step)
            predicted = max(0.0, daily_pattern[target.hour])

            predictions.append({
                "timestamp": target.strftime("%Y-%m-%d %H:%M"),
                "predicted_kw": round(predicted, 4),
            })

        values = [row["predicted_kw"] for row in predictions]

        self.client_forecast_chart_data = predictions
        self.client_forecast_average_kw = f"{sum(values) / len(values):.4f}"
        self.client_forecast_peak_kw = f"{max(values):.4f}"
        self.client_forecast_minimum_kw = f"{min(values):.4f}"
        self.client_forecast_success = True
        self.client_forecast_message = (
            f"Historical-pattern forecast generated for {self.selected_client}. "
            "This is an estimate based on past usage, not live monitoring "
            "or a client-specific trained AI model."
        )
    def select_page(self, page: str):
        self.active_page = page

    def set_forecast_date(self, value: str):
        self.forecast_date = value
        self._reset_forecast()

    def set_forecast_hour(self, value: str):
        self.forecast_hour = value
        self._reset_forecast()

    def set_forecast_horizon(self, value: str):
        self.forecast_horizon = value
        self._reset_forecast()

    def _reset_forecast(self):
        self.forecast_message = ""
        self.forecast_success = False
        self.forecast_predicted_kw = ""
        self.forecast_current_kw = ""
        self.forecast_selected_hour = ""
        self.forecast_target_hour = ""
        self.forecast_average_kw = ""
        self.forecast_peak_kw = ""
        self.forecast_minimum_kw = ""
        self.forecast_chart_data = []

    def run_forecast(self):
        try:
            result = forecast_future(
                self.forecast_date,
                self.forecast_hour,
                int(self.forecast_horizon),
            )

            self.forecast_success = bool(result.get("ok", False))
            self.forecast_message = str(
                result.get("message", "Forecast completed.")
            )

            if not self.forecast_success:
                self._reset_forecast()
                self.forecast_message = str(
                    result.get("message", "Forecast could not be generated.")
                )
                return

            rows = result.get("forecast", [])
            self.forecast_chart_data = [
                {
                    "timestamp": row["timestamp"],
                    "predicted_kw": float(row["predicted_kw"]),
                }
                for row in rows
            ]
            self.forecast_predicted_kw = f'{rows[0]["predicted_kw"]:.2f}'
            self.forecast_selected_hour = result["start_hour"]
            self.forecast_target_hour = result["end_hour"]
            self.forecast_average_kw = f'{result["average_kw"]:.2f}'
            self.forecast_peak_kw = f'{result["peak_kw"]:.2f}'
            self.forecast_minimum_kw = f'{result["minimum_kw"]:.2f}'

        except Exception as exc:
            self._reset_forecast()
            self.forecast_message = (
                f"Forecast could not be generated: {exc}"
            )


NAV_ITEMS = [
    ("Overview", "layout-dashboard"),
    ("Client Forecast", "chart-no-axes-combined"),
    ("Demand Analysis", "chart-no-axes-combined"),
    ("Anomaly Detection", "shield-alert"),
    ("Model Comparison", "chart-no-axes-combined"),
    ("Data Explorer", "database"),
]

def nav_item(label: str, icon_name: str) -> rx.Component:
    return rx.button(
        rx.hstack(
            rx.icon(tag=icon_name, size=18),
            rx.text(label, size="2"),
            width="100%",
            align="center",
            spacing="3",
        ),
        on_click=lambda: GridGuardState.select_page(label),
        variant="ghost",
        color=rx.cond(
            GridGuardState.active_page == label,
            "#EAF4FF",
            "#91A4BC",
        ),
        background=rx.cond(
            GridGuardState.active_page == label,
            "#17365C",
            "transparent",
        ),
        width="100%",
        justify="start",
        padding="12px",
        border_radius="10px",
        cursor="pointer",
    )


def metric_card(
    label: str,
    value: str,
    detail: str,
    icon_name: str,
) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.vstack(
                rx.text(label, color="#91A4BC", size="2"),
                rx.heading(value, size="7", color="#F2F7FF"),
                rx.text(detail, color="#91A4BC", size="1"),
                align="start",
                spacing="2",
            ),
            rx.spacer(),
            rx.box(
                rx.icon(tag=icon_name, size=23, color="#55B8FF"),
                background="#17365C",
                padding="12px",
                border_radius="12px",
            ),
            align="start",
            width="100%",
        ),
        background="#111F32",
        border="1px solid #24374F",
        border_radius="14px",
        padding="20px",
        width="100%",
    )


def result_metric(
    label: str,
    value: str,
    note: str = "",
) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(label, color="#91A4BC", size="2"),
            rx.heading(value, size="7", color="#F2F7FF"),
            rx.text(note, color="#91A4BC", size="1")
            if note
            else rx.fragment(),
            align="start",
            spacing="2",
        ),
        background="#111F32",
        border="1px solid #24374F",
        border_radius="14px",
        padding="18px",
        width="100%",
    )


def section_card(title: str, description: str, content) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.heading(title, size="5", color="#F2F7FF"),
            rx.text(description, color="#91A4BC", size="2"),
            content,
            align="start",
            spacing="3",
            width="100%",
        ),
        background="#111F32",
        border="1px solid #24374F",
        border_radius="14px",
        padding="20px",
        width="100%",
    )


def historical_notice() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.icon(tag="info", color="#F4B860", size=20),
            rx.vstack(
                rx.text(
                    "Charts and model metrics use previously recorded "
                    "electricity readings and saved evaluation results.",
                    color="#C8B994",
                    size="1",
                ),
                align="start",
                spacing="1",
            ),
            align="center",
            spacing="3",
            width="100%",
        ),
        background="#302819",
        border="1px solid #6B512A",
        border_radius="10px",
        padding="14px",
        width="100%",
    )


def back_to_overview_button() -> rx.Component:
    return rx.button(
        rx.icon(tag="arrow-left", size=16),
        "Back to Overview",
        on_click=lambda: GridGuardState.select_page("Overview"),
        variant="outline",
        color="#DCE8F7",
        border="1px solid #304963",
        background="#13243A",
        padding="10px 16px",
        border_radius="10px",
    )


def quick_access_button(
    title: str,
    description: str,
    icon_name: str,
    destination: str,
    icon_color: str = "#55B8FF",
) -> rx.Component:
    return rx.button(
        rx.hstack(
            rx.box(
                rx.icon(tag=icon_name, size=23, color=icon_color),
                background="#17365C",
                padding="12px",
                border_radius="12px",
            ),
            rx.vstack(
                rx.text(title, color="#F2F7FF", weight="bold", size="2"),
                rx.text(description, color="#91A4BC", size="1"),
                align="start",
                spacing="1",
            ),
            rx.spacer(),
            rx.icon(tag="arrow-up-right", color="#91A4BC", size=18),
            align="center",
            spacing="3",
            width="100%",
        ),
        on_click=lambda: GridGuardState.select_page(destination),
        variant="outline",
        color="#F2F7FF",
        border="1px solid #304963",
        background="#13243A",
        padding="16px",
        height="auto",
        width="100%",
        justify="start",
    )


def sidebar() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.box(
                    rx.icon(tag="zap", size=25, color="#FFFFFF"),
                    background="#1976D2",
                    padding="10px",
                    border_radius="12px",
                ),
                rx.vstack(
                    rx.heading("GridGuard", size="5", color="#F2F7FF"),
                    rx.text(
                        "ELECTRICITY INTELLIGENCE",
                        color="#91A4BC",
                        size="1",
                        weight="bold",
                    ),
                    align="start",
                    spacing="1",
                ),
                align="center",
                spacing="3",
                width="100%",
            ),
            rx.divider(color="#24374F"),
            rx.text("WORKSPACE", color="#657C98", size="1", weight="bold"),
            *[nav_item(label, icon_name) for label, icon_name in NAV_ITEMS],
            rx.spacer(),
            rx.box(
                rx.hstack(
                    rx.icon(tag="database", size=17, color="#55B8FF"),
                    rx.text(
                        "Historical data mode",
                        color="#C5D6E9",
                        size="2",
                    ),
                    align="center",
                    spacing="2",
                ),
                rx.text(
                    "PySpark local processing",
                    color="#91A4BC",
                    size="1",
                ),
                background="#111F32",
                border="1px solid #24374F",
                border_radius="12px",
                padding="13px",
                width="100%",
            ),
            align="start",
            spacing="4",
            height="100%",
            width="100%",
        ),
        background="#0B1422",
        border_right="1px solid #24374F",
        padding="20px",
        width="260px",
        min_width="260px",
        min_height="100vh",
    )


# OVERVIEW

def overview_page() -> rx.Component:
    forecast_rows = load_forecast_rows()

    return rx.vstack(
        rx.image(
            src="/gridguard_hero.png",
            alt="Electricity grid at dusk",
            width="100%",
            height="230px",
            object_fit="cover",
            border_radius="16px",
        ),
        rx.text(
            "GRIDGUARD / CONTROL ROOM",
            color="#55B8FF",
            size="1",
            weight="bold",
        ),
        rx.heading("Electricity Intelligence", size="8", color="#F2F7FF"),
        rx.text(
            "Understand electricity consumption, compare forecasts and "
            "investigate potentially unusual usage patterns.",
            color="#91A4BC",
            size="3",
        ),
        rx.grid(
            metric_card(
                "Electricity clients", "370",
                "Recorded client series", "users",
            ),
            metric_card(
                "Historical readings", "51.89M",
                "Validated readings", "database",
            ),
            metric_card(
                "Recording interval", "15 min",
                "Original data frequency", "clock",
            ),
            metric_card(
                "Forecast model", "Random Forest",
                "Next-hour average demand", "brain-circuit",
            ),
            columns=rx.breakpoints(initial="1", sm="2", lg="4"),
            spacing="4",
            width="100%",
        ),
        section_card(
            "Actual vs Predicted Demand",
            "Random Forest next-hour predictions compared with actual demand "
            "and a previous-hour baseline.",
            rx.cond(
                len(forecast_rows) > 0,
                rx.recharts.line_chart(
                    rx.recharts.line(
                        data_key="actual",
                        name="Actual demand",
                        stroke="#2DD4BF",
                        type="monotone",
                        dot=False,
                    ),
                    rx.recharts.line(
                        data_key="predicted",
                        name="Random Forest prediction",
                        stroke="#55B8FF",
                        type="monotone",
                        dot=False,
                    ),
                    rx.recharts.line(
                        data_key="baseline",
                        name="Previous-hour baseline",
                        stroke="#F4B860",
                        type="monotone",
                        stroke_dasharray="5 4",
                        dot=False,
                    ),
                    rx.recharts.x_axis(data_key="timestamp", hide=True),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(
                        stroke="#24374F",
                        stroke_dasharray="3 3",
                    ),
                    rx.recharts.tooltip(),
                    rx.recharts.legend(),
                    data=forecast_rows,
                    width="100%",
                    height=340,
                ),
                rx.text(
                    "No saved forecast test predictions were found.",
                    color="#F4B860",
                ),
            ),
        ),
        rx.text(
            "Historical electricity data (2011–2014). This chart uses saved "
            "test predictions and is not a live forecast.",
            color="#91A4BC",
            size="1",
        ),
        section_card(
            "Forecast model status",
            "Saved model evaluation results from the historical test period.",
            rx.vstack(
                rx.hstack(
                    rx.icon(tag="check-circle", color="#55B8FF", size=19),
                    rx.text(
                        "Random Forest evaluation results are available",
                        color="#DCE8F7",
                        size="2",
                    ),
                    align="center",
                    spacing="2",
                ),
                rx.text(
                    "MAE: 15.18 kW  •  RMSE: 22.32 kW  •  R²: 0.9902",
                    color="#DCE8F7",
                    size="2",
                ),
                align="start",
                spacing="3",
                width="100%",
            ),
        ),
        section_card(
            "Quick Access",
            "Open a workspace using the buttons below or the sidebar.",
            rx.grid(
                quick_access_button(
                    "Demand Analysis", "View demand and forecast comparisons",
                    "chart-no-axes-combined", "Demand Analysis",
                ),
                quick_access_button(
                    "Anomaly Detection", "Inspect potential unusual readings",
                    "shield-alert", "Anomaly Detection", "#F4B860",
                ),
                quick_access_button(
                    "ML Performance", "Review and run model forecasts",
                    "brain-circuit", "ML Performance",
                ),
                columns=rx.breakpoints(initial="1", sm="2"),
                spacing="3",
                width="100%",
            ),
        ),
        align="start",
        spacing="5",
        width="100%",
    )


# HISTORICAL REPLAY

def demand_analysis_page() -> rx.Component:
    demand_rows = load_demand_rows()
    forecast_rows = load_forecast_rows()

    return rx.vstack(
        back_to_overview_button(),
        rx.text(
            "GRIDGUARD / HISTORICAL ANALYTICS",
            color="#55B8FF",
            size="1",
            weight="bold",
        ),
        rx.heading("Demand Analysis", size="8", color="#F2F7FF"),
        rx.text(
            "Explore recorded electricity consumption and historical forecasts.",
            color="#91A4BC",
            size="3",
        ),
        historical_notice(),
        rx.grid(
            metric_card(
                "Demand chart points", str(len(demand_rows)),
                "Sampled across the timeline", "chart-no-axes-combined",
            ),
            metric_card(
                "Forecast comparison points", str(len(forecast_rows)),
                "Saved test predictions", "brain-circuit",
            ),
            columns=rx.breakpoints(initial="1", sm="2"),
            spacing="4",
            width="100%",
        ),
        section_card(
            "Historical Average Demand",
            "Average kW across client readings. Points are sampled across "
            "the timeline and are not consecutive readings.",
            rx.cond(
                len(demand_rows) > 0,
                rx.recharts.line_chart(
                    rx.recharts.line(
                        data_key="average_kw",
                        name="Average demand (kW)",
                        stroke="#55B8FF",
                        type="monotone",
                        dot=False,
                    ),
                    rx.recharts.x_axis(data_key="timestamp", hide=True),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(
                        stroke="#24374F",
                        stroke_dasharray="3 3",
                    ),
                    rx.recharts.tooltip(),
                    data=demand_rows,
                    width="100%",
                    height=300,
                ),
                rx.text(
                    "No historical demand summary data was found.",
                    color="#F4B860",
                ),
            ),
        ),
        section_card(
            "Actual vs Predicted Demand",
            "Saved historical test results: actual demand, Random Forest "
            "prediction and previous-hour baseline.",
            rx.cond(
                len(forecast_rows) > 0,
                rx.recharts.line_chart(
                    rx.recharts.line(
                        data_key="actual",
                        name="Actual demand",
                        stroke="#F4B860",
                        type="monotone",
                        dot=False,
                    ),
                    rx.recharts.line(
                        data_key="predicted",
                        name="Random Forest prediction",
                        stroke="#55B8FF",
                        type="monotone",
                        dot=False,
                    ),
                    rx.recharts.line(
                        data_key="baseline",
                        name="Previous-hour baseline",
                        stroke="#B7A0FF",
                        type="monotone",
                        dot=False,
                    ),
                    rx.recharts.x_axis(data_key="timestamp", hide=True),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(
                        stroke="#24374F",
                        stroke_dasharray="3 3",
                    ),
                    rx.recharts.tooltip(),
                    rx.recharts.legend(),
                    data=forecast_rows,
                    width="100%",
                    height=340,
                ),
                rx.text(
                    "No forecast test predictions were found.",
                    color="#F4B860",
                ),
            ),
        ),
        rx.text(
            "Sources: saved Spark summaries and forecast test predictions. "
            "These are historical results, not live forecasts.",
            color="#91A4BC",
            size="1",
        ),
        align="start",
        spacing="5",
        width="100%",
    )


# SAVED METRICS AND ANOMALY OUTPUTS

def load_ml_metrics() -> dict:
    path = PROJECT_ROOT / "results" / "forecasting" / "forecast_metrics.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def load_anomaly_summary() -> dict:
    path = (
        PROJECT_ROOT
        / "results"
        / "anomaly_detection"
        / "investigation_report"
        / "investigation_summary.json"
    )
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def load_flagged_preview(limit: int = 6) -> list[dict]:
    path = (
        PROJECT_ROOT
        / "results"
        / "anomaly_detection"
        / "investigation_report"
        / "traceable_flagged_readings.csv"
    )
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))[:limit]
    except (OSError, csv.Error):
        return []


# ANOMALY DETECTION

def anomaly_detection_page() -> rx.Component:
    summary = load_anomaly_summary()
    flagged_rows = load_flagged_preview()

    reviewed = int(summary.get("rows_reviewed", 0))
    flagged = int(summary.get("flagged_rows_exported", 0))
    clients = int(summary.get("unique_clients_flagged", 0))

    preview_components = []

    if flagged_rows:
        columns = list(flagged_rows[0].keys())[:4]
        preview_components.append(
            rx.text(
                "  |  ".join(str(column) for column in columns),
                color="#55B8FF",
                weight="bold",
                size="1",
            )
        )
        for row in flagged_rows:
            preview_components.append(
                rx.text(
                    "  |  ".join(
                        str(row.get(column, ""))[:35]
                        for column in columns
                    ),
                    color="#C8D5E5",
                    size="1",
                )
            )
    else:
        preview_components.append(
            rx.text(
                "No flagged-reading preview was found. Check the saved CSV.",
                color="#F4B860",
                size="2",
            )
        )

    return rx.vstack(
        back_to_overview_button(),
        rx.text(
            "GRIDGUARD / ANOMALY INVESTIGATION",
            color="#55B8FF",
            size="1",
            weight="bold",
        ),
        rx.heading("Anomaly Detection", size="8", color="#F2F7FF"),
        rx.text(
            "Review unusual electricity readings flagged for further investigation.",
            color="#91A4BC",
            size="3",
        ),
        historical_notice(),

        rx.box(
            rx.vstack(
                rx.heading(
                    "Check an Individual Client",
                    size="5",
                    color="#F2F7FF",
                ),
                rx.text(
                    "Enter an hourly electricity reading and compare it "
                    "with the selected client's earlier consumption history.",
                    color="#91A4BC",
                    size="2",
                ),
                rx.text("Client ID", color="#C8D5E5", size="2"),
                rx.select(
                    AVAILABLE_CLIENTS,
                    value=GridGuardState.anomaly_client_id,
                    on_change=GridGuardState.set_anomaly_client,
                    width="100%",
                ),
                rx.text("Date", color="#C8D5E5", size="2"),
                rx.input(
                    value=GridGuardState.anomaly_date,
                    on_change=GridGuardState.set_anomaly_date,
                    type="date",
                    width="100%",
                ),
                rx.text("Hour (0–23)", color="#C8D5E5", size="2"),
                rx.select(
                    [str(hour) for hour in range(24)],
                    value=GridGuardState.anomaly_hour,
                    on_change=GridGuardState.set_anomaly_hour,
                    width="100%",
                ),
                rx.text("Average electricity usage (kW)", color="#C8D5E5", size="2"),
                rx.input(
                    value=GridGuardState.anomaly_average_kw,
                    on_change=GridGuardState.set_anomaly_average,
                    placeholder="e.g. 2.45",
                    type="number",
                    min="0",
                    width="100%",
                ),
                rx.text("Within-hour standard deviation (kW)", color="#C8D5E5", size="2"),
                rx.input(
                    value=GridGuardState.anomaly_std_kw,
                    on_change=GridGuardState.set_anomaly_std,
                    placeholder="e.g. 0.18",
                    type="number",
                    min="0",
                    width="100%",
                ),
                rx.button(
                    "Check Reading",
                    on_click=GridGuardState.check_client_anomaly,
                    width="100%",
                    background="#1687F8",
                    color="#FFFFFF",
                ),
                rx.cond(
                    GridGuardState.anomaly_message != "",
                    rx.text(
                        GridGuardState.anomaly_message,
                        color="#91A4BC",
                        size="2",
                    ),
                    rx.fragment(),
                ),
                rx.cond(
                    GridGuardState.anomaly_success,
                    rx.vstack(
                        rx.heading(
                            GridGuardState.anomaly_result,
                            size="6",
                            color="#F4B860",
                        ),
                        rx.text(
                            "Anomaly score: "
                            + GridGuardState.anomaly_score,
                            color="#C8D5E5",
                            size="2",
                        ),
                        align="start",
                        width="100%",
                    ),
                    rx.fragment(),
                ),
                rx.text(
                    "This checks a historical reading using the saved "
                    "Isolation Forest model. A potential anomaly is a "
                    "review flag, not proof of a fault.",
                    color="#F4B860",
                    size="2",
                ),
                align="start",
                spacing="3",
                width="100%",
            ),
            background="#111F32",
            border="1px solid #24374F",
            border_radius="14px",
            padding="20px",
            width="100%",
        ),

        rx.grid(
            result_metric("Readings reviewed", f"{reviewed:,}", "LOF unseen sample"),
            result_metric(
                "Potential anomalies", f"{flagged:,}",
                "Flagged readings exported",
            ),
            result_metric(
                "Clients flagged", f"{clients:,}",
                "Unique clients represented",
            ),
            columns=rx.breakpoints(initial="1", sm="3"),
            spacing="3",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.heading("Investigation preview", size="5", color="#F2F7FF"),
                rx.text(
                    "A sample from the saved flagged-readings file.",
                    color="#91A4BC",
                    size="2",
                ),
                *preview_components,
                rx.text(
                    "These are potential anomalies only. A flag does not "
                    "prove a fault or equipment failure.",
                    color="#F4B860",
                    size="2",
                ),
                align="start",
                spacing="2",
                width="100%",
            ),
            background="#111F32",
            border="1px solid #24374F",
            border_radius="14px",
            padding="20px",
            width="100%",
        ),
        rx.text(
            "Source: results/anomaly_detection/investigation_report/",
            color="#91A4BC",
            size="1",
        ),
        align="start",
        spacing="4",
        width="100%",
    )


# FUTURE DEMAND FORECAST


def forecast_panel() -> rx.Component:
    hour_options = [f"{hour:02d}" for hour in range(24)]

    return section_card(
        "Predict Future Electricity Use",
        "Choose when you want the estimate to start and how many hours "
        "you want to look ahead.",
        rx.vstack(
            rx.text(
                "Choose a start date",
                color="#C8D5E5",
                size="2",
                weight="bold",
            ),
            rx.input(
                type="date",
                value=GridGuardState.forecast_date,
                on_change=GridGuardState.set_forecast_date,
                width="100%",
            ),
            rx.text(
                "Choose a start time",
                color="#C8D5E5",
                size="2",
                weight="bold",
            ),
            rx.select(
                hour_options,
                value=GridGuardState.forecast_hour,
                on_change=GridGuardState.set_forecast_hour,
                placeholder="Choose a starting hour",
                width="100%",
            ),
            rx.text(
                "How far ahead would you like to look?",
                color="#C8D5E5",
                size="2",
                weight="bold",
            ),
            rx.select(
                [str(hour) for hour in range(1, 25)],
                value=GridGuardState.forecast_horizon,
                on_change=GridGuardState.set_forecast_horizon,
                width="100%",
            ),
            rx.text(
                "Choose between 1 and 24 hours.",
                color="#91A4BC",
                size="1",
            ),
            rx.button(
                rx.icon(tag="brain-circuit", size=17),
                "Predict Electricity Use",
                on_click=GridGuardState.run_forecast,
                background="#1976D2",
                color="#FFFFFF",
                width="100%",
                padding="12px",
            ),
            rx.cond(
                GridGuardState.forecast_message != "",
                rx.box(
                    rx.vstack(
                        rx.text(
                            GridGuardState.forecast_message,
                            color="#C8D5E5",
                            size="2",
                        ),
                        rx.cond(
                            GridGuardState.forecast_success,
                            rx.vstack(
                                rx.divider(color="#304963"),
                                rx.text(
                                    "Your selected forecast period",
                                    color="#91A4BC",
                                    size="2",
                                ),
                                rx.text(
                                    GridGuardState.forecast_selected_hour
                                    + " to "
                                    + GridGuardState.forecast_target_hour,
                                    color="#F2F7FF",
                                    size="3",
                                    weight="bold",
                                ),
                                rx.text(
                                    "Estimated power use at the starting time",
                                    color="#91A4BC",
                                    size="2",
                                ),
                                rx.heading(
                                    GridGuardState.forecast_predicted_kw + " kW",
                                    color="#55B8FF",
                                    size="7",
                                ),
                                rx.grid(
                                    rx.box(
                                        rx.text(
                                            "Average expected power",
                                            color="#91A4BC",
                                            size="1",
                                        ),
                                        rx.heading(
                                            GridGuardState.forecast_average_kw + " kW",
                                            size="4",
                                            color="#F2F7FF",
                                        ),
                                        padding="12px",
                                        background="#0D1B2A",
                                        border_radius="10px",
                                    ),
                                    rx.box(
                                        rx.text(
                                            "Highest expected power",
                                            color="#91A4BC",
                                            size="1",
                                        ),
                                        rx.heading(
                                            GridGuardState.forecast_peak_kw + " kW",
                                            size="4",
                                            color="#55B8FF",
                                        ),
                                        padding="12px",
                                        background="#0D1B2A",
                                        border_radius="10px",
                                    ),
                                    rx.box(
                                        rx.text(
                                            "Lowest expected power",
                                            color="#91A4BC",
                                            size="1",
                                        ),
                                        rx.heading(
                                            GridGuardState.forecast_minimum_kw + " kW",
                                            size="4",
                                            color="#F2F7FF",
                                        ),
                                        padding="12px",
                                        background="#0D1B2A",
                                        border_radius="10px",
                                    ),
                                    columns=rx.breakpoints(initial="1", sm="3"),
                                    spacing="3",
                                    width="100%",
                                ),
                                rx.heading(
                                    "How expected power use may change",
                                    size="4",
                                    color="#F2F7FF",
                                ),
                                rx.recharts.line_chart(
                                    rx.recharts.line(
                                        data_key="predicted_kw",
                                        name="Expected power use",
                                        stroke="#55B8FF",
                                        type="monotone",
                                        dot=False,
                                    ),
                                    rx.recharts.x_axis(
                                        data_key="timestamp",
                                        hide=True,
                                    ),
                                    rx.recharts.y_axis(
                                        label={
                                            "value": "Power (kW)",
                                            "angle": -90,
                                            "position": "insideLeft",
                                        },
                                    ),
                                    rx.recharts.cartesian_grid(
                                        stroke_dasharray="3 3",
                                    ),
                                    rx.recharts.tooltip(),
                                    rx.recharts.legend(),
                                    data=GridGuardState.forecast_chart_data,
                                    width="100%",
                                    height=300,
                                ),
                                align="start",
                                spacing="3",
                                width="100%",
                            ),
                            rx.fragment(),
                        ),
                        align="start",
                        spacing="3",
                        width="100%",
                    ),
                    background="#13243A",
                    border="1px solid #304963",
                    border_radius="12px",
                    padding="16px",
                    width="100%",
                ),
                rx.fragment(),
            ),
            rx.text(
                "Please note: this estimate is based on historical electricity "
                "data from 2011–2014. It is not a live reading, and estimates "
                "for dates far beyond that period may be less reliable.",
                color="#91A4BC",
                size="1",
            ),
            align="start",
            spacing="3",
            width="100%",
        ),
    )



# ML PERFORMANCE


# ML PERFORMANCE

def ml_performance_page() -> rx.Component:
    metrics = load_ml_metrics()

    if not metrics:
        return rx.vstack(
            back_to_overview_button(),
            rx.heading("ML Performance", size="8", color="#F2F7FF"),
            rx.text(
                "The saved forecast_metrics.json file was not found. "
                "Check results/forecasting/.",
                color="#F4B860",
            ),
            align="start",
            spacing="4",
            width="100%",
        )

    mae = float(metrics.get("mae_kw", 0))
    rmse = float(metrics.get("rmse_kw", 0))
    r2 = float(metrics.get("r2", 0))
    baseline = float(metrics.get("previous_hour_baseline_mae_kw", 0))
    improvement = (
        (baseline - mae) / baseline * 100
        if baseline > 0
        else 0
    )

    return rx.vstack(
        back_to_overview_button(),
        rx.text(
            "GRIDGUARD / MODEL EVALUATION",
            color="#55B8FF",
            size="1",
            weight="bold",
        ),
        rx.heading("ML Performance", size="8", color="#F2F7FF"),
        rx.text(
            "Review the saved Random Forest model's performance "
            "using historical electricity data.",
            color="#91A4BC",
            size="3",
        ),
        historical_notice(),
        rx.grid(
            result_metric(
                "Mean Absolute Error (MAE)",
                f"{mae:.2f} kW",
                "Lower is better",
            ),
            result_metric(
                "Root Mean Squared Error (RMSE)",
                f"{rmse:.2f} kW",
                "Lower is better",
            ),
            result_metric(
                "R² score",
                f"{r2:.4f}",
                "Closer to 1 is better",
            ),
            columns=rx.breakpoints(initial="1", sm="3"),
            spacing="3",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.heading(
                    "Baseline comparison",
                    size="5",
                    color="#F2F7FF",
                ),
                rx.text(
                    f"Previous-hour baseline MAE: {baseline:.2f} kW",
                    color="#C8D5E5",
                    size="3",
                ),
                rx.text(
                    f"Random Forest MAE: {mae:.2f} kW",
                    color="#C8D5E5",
                    size="3",
                ),
                rx.text(
                    f"MAE reduction versus baseline: {improvement:.1f}%",
                    color="#55B8FF",
                    size="4",
                    weight="bold",
                ),
                rx.text(
                    "These figures use the saved historical test set "
                    "and do not represent live forecasting accuracy.",
                    color="#91A4BC",
                    size="2",
                ),
                align="start",
                spacing="3",
                width="100%",
            ),
            background="#111F32",
            border="1px solid #24374F",
            border_radius="14px",
            padding="20px",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.heading(
                    "Evaluation dataset",
                    size="5",
                    color="#F2F7FF",
                ),
                rx.text(
                    f"Model: {metrics.get('model', 'Not recorded')}",
                    color="#C8D5E5",
                    size="2",
                ),
                rx.text(
                    f"Target: {metrics.get('target', 'Not recorded')}",
                    color="#C8D5E5",
                    size="2",
                ),
                rx.text(
                    f"Training records: "
                    f"{int(metrics.get('training_records', 0)):,}",
                    color="#C8D5E5",
                    size="2",
                ),
                rx.text(
                    f"Testing records: "
                    f"{int(metrics.get('testing_records', 0)):,}",
                    color="#C8D5E5",
                    size="2",
                ),
                rx.text(
                    f"Training end: "
                    f"{metrics.get('training_end', 'Not recorded')}",
                    color="#91A4BC",
                    size="2",
                ),
                rx.text(
                    f"Test start: "
                    f"{metrics.get('test_start', 'Not recorded')}",
                    color="#91A4BC",
                    size="2",
                ),
                align="start",
                spacing="2",
                width="100%",
            ),
            background="#111F32",
            border="1px solid #24374F",
            border_radius="14px",
            padding="20px",
            width="100%",
        ),

        section_card(
            "Recorded vs Estimated Electricity Use — MT_001",
            "Saved historical test results for client MT_001. "
            "This client-specific evaluation is separate from "
            "the system-wide forecast and other clients' forecasts.",
            rx.recharts.line_chart(
                rx.recharts.line(
                    data_key="actual",
                    name="Recorded electricity use",
                    stroke="#2DD4BF",
                    type="monotone",
                    dot=False,
                ),
                rx.recharts.line(
                    data_key="predicted",
                    name="Trained model estimate",
                    stroke="#55B8FF",
                    type="monotone",
                    dot=False,
                ),
                rx.recharts.line(
                    data_key="baseline",
                    name="Reference baseline",
                    stroke="#F4B860",
                    type="monotone",
                    stroke_dasharray="5 4",
                    dot=False,
                ),
                rx.recharts.x_axis(
                    data_key="timestamp",
                    hide=True,
                ),
                rx.recharts.y_axis(
                    label={
                        "value": "Power (kW)",
                        "angle": -90,
                        "position": "insideLeft",
                    },
                ),
                rx.recharts.cartesian_grid(
                    stroke="#24374F",
                    stroke_dasharray="3 3",
                ),
                rx.recharts.tooltip(),
                rx.recharts.legend(),
                data=CLIENT_FORECAST_ROWS,
                width="100%",
                height=320,
            ),
        ),

        rx.text(
            "Source: results/forecasting/forecast_metrics.json",
            color="#91A4BC",
            size="1",
        ),
        align="start",
        spacing="4",
        width="100%",
    )


# MODEL COMPARISON

def model_comparison_page() -> rx.Component:
    rows = load_horizon_comparison()

    if not rows:
        return rx.vstack(
            back_to_overview_button(),
            rx.heading("Model Comparison", size="8", color="#F2F7FF"),
            rx.text(
                "Could not load the forecast horizon validation results. "
                "Check results/forecast_horizon_validation/"
                "horizon_baseline_comparison.csv.",
                color="#F4B860",
            ),
            align="start",
            spacing="4",
            width="100%",
        )

    best_rf = sum(
        1 for row in rows
        if row["rf_mae_kw"] < row["baseline_mae_kw"]
    )
    best_baseline = len(rows) - best_rf

    chart_rows = [
        {
            "horizon": f'{row["horizon_hours"]}h',
            "Random Forest": round(row["rf_mae_kw"], 2),
            "Previous-value baseline": round(row["baseline_mae_kw"], 2),
        }
        for row in rows
    ]

    return rx.vstack(
        back_to_overview_button(),
        rx.text(
            "GRIDGUARD / MODEL COMPARISON",
            color="#55B8FF",
            size="1",
            weight="bold",
        ),
        rx.heading("Model Comparison", size="8", color="#F2F7FF"),
        rx.text(
            "Compare how prediction errors change when forecasting "
            "from 1 hour ahead up to 24 hours ahead.",
            color="#91A4BC",
            size="3",
        ),
        historical_notice(),

        rx.grid(
            result_metric(
                "Forecast horizons tested",
                f"{len(rows)}",
                "1 to 24 hours ahead",
            ),
            result_metric(
                "Random Forest wins",
                f"{best_rf} of {len(rows)}",
                "Lower MAE than baseline",
            ),
            result_metric(
                "Baseline wins",
                f"{best_baseline} of {len(rows)}",
                "Lower MAE than Random Forest",
            ),
            columns=rx.breakpoints(initial="1", sm="3"),
            spacing="3",
            width="100%",
        ),

        section_card(
            "Prediction error by forecast horizon",
            "Lower MAE means predictions are closer to the recorded "
            "electricity demand, on average. Compare both models at "
            "each forecast horizon.",
            rx.recharts.line_chart(
                rx.recharts.line(
                    data_key="Random Forest",
                    name="Random Forest",
                    stroke="#55B8FF",
                    type="monotone",
                    dot=False,
                ),
                rx.recharts.line(
                    data_key="Previous-value baseline",
                    name="Previous-value baseline",
                    stroke="#F4B860",
                    type="monotone",
                    dot=False,
                ),
                rx.recharts.x_axis(data_key="horizon"),
                rx.recharts.y_axis(),
                rx.recharts.cartesian_grid(),
                rx.recharts.tooltip(),
                rx.recharts.legend(),
                data=chart_rows,
                width="100%",
                height=320,
            ),
        ),


        section_card(
            "Results for each forecast horizon",
            "Lower MAE means predictions are closer to recorded "
            "electricity demand. All 24 forecast horizons are shown.",
            rx.vstack(
                rx.grid(
                    rx.text("Horizon", color="#91A4BC", weight="bold", size="2"),
                    rx.text("Random Forest MAE", color="#91A4BC", weight="bold", size="2"),
                    rx.text("Baseline MAE", color="#91A4BC", weight="bold", size="2"),
                    rx.text("Better result", color="#91A4BC", weight="bold", size="2"),
                    columns="4",
                    spacing="2",
                    width="100%",
                    padding="10px",
                    background="#1A2A40",
                    border_radius="8px",
                ),
                *[
                    rx.grid(
                        rx.text(
                            f'{row["horizon_hours"]} hours',
                            color="#F2F7FF",
                            size="2",
                        ),
                        rx.text(
                            f'{row["rf_mae_kw"]:.2f} kW',
                            color="#55B8FF",
                            size="2",
                        ),
                        rx.text(
                            f'{row["baseline_mae_kw"]:.2f} kW',
                            color="#F4B860",
                            size="2",
                        ),
                        rx.text(
                            "Random Forest"
                            if row["rf_mae_kw"] < row["baseline_mae_kw"]
                            else "Baseline"
                            if row["rf_mae_kw"] > row["baseline_mae_kw"]
                            else "Tie",
                            color="#2DD4BF",
                            size="2",
                        ),
                        columns="4",
                        spacing="2",
                        width="100%",
                        padding="10px",
                        border_bottom="1px solid #24374F",
                    )
                    for row in rows
                ],
                align="start",
                spacing="0",
                width="100%",
            ),
        ),

        section_card(
            "How to interpret these results",
            "These results come from a separate recursive 1–24-hour "
            "validation experiment using the saved shared client model. "
            "The model was not retrained for this comparison. "
            "The previous-value baseline repeats the most recent "
            "observed demand. Random Forest has lower MAE at some "
            "horizons, but it does not win at every horizon. "
            "These historical validation results are not live "
            "forecasting guarantees.",
            rx.text(
                "Model comparison uses saved historical validation results.",
                color="#91A4BC",
                size="2",
            ),
        ),

        align="start",
        spacing="5",
        width="100%",
    )


# CLIENT FORECAST



def client_forecast_page() -> rx.Component:
    metrics = CLIENT_FORECAST_METRICS
    rows = CLIENT_FORECAST_ROWS

    if not metrics or not rows:
        return rx.vstack(
            back_to_overview_button(),
            rx.heading(
                "My Electricity Forecast",
                size="8",
                color="#F2F7FF",
            ),
            historical_notice(),
            rx.text(
                "We couldn't load the saved electricity results. "
                "Please check the client forecast data files.",
                color="#F4B860",
            ),
            align="start",
            spacing="4",
            width="100%",
        )

    rf_mae = float(metrics["random_forest_mae_kw"])
    baseline_mae = float(metrics["baseline_mae_kw"])
    improvement = float(metrics["mae_improvement_percent"])

    return rx.vstack(
        back_to_overview_button(),

        rx.text(
            "GRIDGUARD / ELECTRICITY FORECAST",
            color="#55B8FF",
            size="1",
            weight="bold",
        ),

        rx.heading(
            "My Electricity Forecast",
            size="8",
            color="#F2F7FF",
        ),

        rx.text(
            "Review past electricity use or estimate what may be "
            "used in the coming hours.",
            color="#91A4BC",
            size="3",
        ),

        historical_notice(),
        
        # SELECT A CLIENT
        section_card(
            "Explore Your Client",
            "Choose one of the 370 electricity clients to explore its "
            "historical usage and a simple next-hour estimate.",
            rx.vstack(
                rx.text(
                    "Select a client",
                    color="#C8D5E5",
                    size="2",
                ),
                rx.select(
                    AVAILABLE_CLIENTS,
                    value=GridGuardState.selected_client,
                    on_change=GridGuardState.set_selected_client,
                    width="100%",
                ),
                rx.button(
                    "Load Client History",
                    on_click=GridGuardState.load_selected_client_history,
                    background="#1687F8",
                    color="#FFFFFF",
                    width="100%",
                ),
                rx.text(
                    GridGuardState.client_history_message,
                    color="#2DD4BF",
                    size="2",
                ),
                rx.cond(
                    GridGuardState.client_history_data.length() > 0,
                    rx.vstack(
                        rx.grid(
                            metric_card(
                                "Latest recorded usage",
                                GridGuardState.client_latest_usage_kw + " kW",
                                "The client's most recent available hourly reading.",
                                "zap",
                            ),
                            metric_card(
                                "Simple next-hour estimate",
                                GridGuardState.client_next_hour_kw + " kW",
                                "Repeats the latest recorded hourly usage as a reference estimate.",
                                "gauge",
                            ),
                            metric_card(
                                "Latest record",
                                GridGuardState.client_latest_timestamp,
                                "Timestamp of the latest available historical reading.",
                                "calendar-clock",
                            ),
                            columns=rx.breakpoints(initial="1", sm="3"),
                            spacing="4",
                            width="100%",
                        ),
                        rx.recharts.line_chart(
                            rx.recharts.line(
                                data_key="average_kw",
                                name="Recorded hourly usage",
                                stroke="#2DD4BF",
                                type="monotone",
                                dot=False,
                            ),
                            rx.recharts.x_axis(
                                data_key="timestamp",
                                hide=True,
                            ),
                            rx.recharts.y_axis(
                                label={
                                    "value": "Power (kW)",
                                    "angle": -90,
                                    "position": "insideLeft",
                                },
                            ),
                            rx.recharts.cartesian_grid(
                                stroke="#24374F",
                                stroke_dasharray="3 3",
                            ),
                            rx.recharts.tooltip(),
                            rx.recharts.legend(),
                            data=GridGuardState.client_history_data,
                            width="100%",
                            height=320,
                        ),
                        align="start",
                        spacing="4",
                        width="100%",
                    ),
                ),
                rx.text(
                    "The next-hour estimate is a simple reference, not an "
                    "AI-generated forecast.",
                    color="#F4B860",
                    size="2",
                ),
                align="start",
                spacing="4",
                width="100%",
            ),
        ),

        # PAST ELECTRICITY USAGE


                # PAST ELECTRICITY USAGE
        section_card(
            "1. Check Past Electricity Use",
            "Choose a date and hour to view the selected client's "
            "recorded electricity use and compare it with recent history.",
            rx.vstack(
                rx.grid(
                    rx.vstack(
                        rx.text("Choose a date", color="#C8D5E5", size="2"),
                        rx.input(
                            type="date",
                            value=GridGuardState.past_date,
                            on_change=GridGuardState.set_past_date,
                            width="100%",
                        ),
                        align="start",
                        spacing="2",
                        width="100%",
                    ),
                    rx.vstack(
                        rx.text("Choose an hour", color="#C8D5E5", size="2"),
                        rx.select(
                            [f"{hour:02d}" for hour in range(24)],
                            value=GridGuardState.past_hour,
                            on_change=GridGuardState.set_past_hour,
                            width="100%",
                        ),
                        align="start",
                        spacing="2",
                        width="100%",
                    ),
                    columns=rx.breakpoints(initial="1", sm="2"),
                    spacing="4",
                    width="100%",
                ),
                rx.button(
                    "Check Past Usage",
                    on_click=GridGuardState.check_selected_client_past_usage,
                    background="#1687F8",
                    color="#FFFFFF",
                    width="100%",
                ),
                rx.text(
                    GridGuardState.client_past_message,
                    color=rx.cond(
                        GridGuardState.client_past_success,
                        "#2DD4BF",
                        "#F4B860",
                    ),
                    size="2",
                ),
                rx.cond(
                    GridGuardState.client_past_success,
                    rx.vstack(
                        rx.button(
                            "Hide Results",
                            on_click=GridGuardState.toggle_client_past_results,
                            background="#24374F",
                            color="#FFFFFF",
                            width="100%",
                        ),
                        rx.cond(
                            GridGuardState.client_past_results_visible,
                            rx.vstack(
                                rx.grid(
                                    metric_card(
                                        "Selected client",
                                        GridGuardState.selected_client,
                                        "Client whose historical data is being checked.",
                                        "user-round",
                                    ),
                                    metric_card(
                                        "Recorded electricity use",
                                        GridGuardState.client_past_actual_kw + " kW",
                                        "Actual historical reading for the selected hour.",
                                        "zap",
                                    ),
                                    metric_card(
                                        "Previous hour",
                                        GridGuardState.client_past_previous_hour_kw + " kW",
                                        "Recorded usage one hour earlier.",
                                        "clock",
                                    ),
                                    metric_card(
                                        "Previous day, same hour",
                                        GridGuardState.client_past_previous_day_kw + " kW",
                                        "Recorded usage 24 hours earlier.",
                                        "calendar-clock",
                                    ),
                                    columns=rx.breakpoints(initial="1", sm="2"),
                                    spacing="4",
                                    width="100%",
                                ),
                                section_card(
                                    "Around the Selected Hour",
                                    "Recorded electricity use in nearby hours. "
                                    "These are historical readings, not AI predictions.",
                                    rx.recharts.line_chart(
                                        rx.recharts.line(
                                            data_key="actual",
                                            name="Recorded usage",
                                            stroke="#2DD4BF",
                                            type="monotone",
                                            dot=False,
                                        ),
                                        rx.recharts.x_axis(
                                            data_key="timestamp",
                                            hide=True,
                                        ),
                                        rx.recharts.y_axis(
                                            label={
                                                "value": "Power (kW)",
                                                "angle": -90,
                                                "position": "insideLeft",
                                            },
                                        ),
                                        rx.recharts.cartesian_grid(
                                            stroke="#24374F",
                                            stroke_dasharray="3 3",
                                        ),
                                        rx.recharts.tooltip(),
                                        rx.recharts.legend(),
                                        data=GridGuardState.client_past_chart_data,
                                        width="100%",
                                        height=320,
                                    ),
                                ),
                                align="start",
                                spacing="4",
                                width="100%",
                            ),
                        ),
                        align="start",
                        spacing="4",
                        width="100%",
                    ),
                    rx.text(
                        "Choose a date and hour, then select Check Past Usage.",
                        color="#91A4BC",
                        size="2",
                    ),
                ),
                rx.text(
                    "Historical meter data only. No client-specific trained "
                    "AI prediction is shown in this past-usage section.",
                    color="#F4B860",
                    size="2",
                ),
                align="start",
                spacing="4",
                width="100%",
            ),
        ),

        # EXISTING MODEL ACCURACY

                # EXISTING FUTURE FORECAST FEATURE

        # SELECTED CLIENT FUTURE FORECAST
        section_card(
            "2. Predict Future Electricity Use",
            "Estimate future electricity use for the client selected above, "
            "using that client's recent historical hourly pattern.",
            rx.vstack(
                rx.grid(
                    rx.vstack(
                        rx.text(
                            "Forecast start date",
                            color="#C8D5E5",
                            size="2",
                        ),
                        rx.input(
                            type="date",
                            value=GridGuardState.client_forecast_date,
                            on_change=GridGuardState.set_client_forecast_date,
                            width="100%",
                        ),
                        align="start",
                        spacing="2",
                        width="100%",
                    ),
                    rx.vstack(
                        rx.text(
                            "Starting hour",
                            color="#C8D5E5",
                            size="2",
                        ),
                        rx.select(
                            [f"{hour:02d}" for hour in range(24)],
                            value=GridGuardState.client_forecast_hour,
                            on_change=GridGuardState.set_client_forecast_hour,
                            width="100%",
                        ),
                        align="start",
                        spacing="2",
                        width="100%",
                    ),
                    rx.vstack(
                        rx.text(
                            "Forecast length",
                            color="#C8D5E5",
                            size="2",
                        ),
                        rx.select(
                            [str(hour) for hour in range(1, 25)],
                            value=GridGuardState.client_forecast_horizon,
                            on_change=GridGuardState.set_client_forecast_horizon,
                            width="100%",
                        ),
                        align="start",
                        spacing="2",
                        width="100%",
                    ),
                    columns=rx.breakpoints(initial="1", sm="3"),
                    spacing="4",
                    width="100%",
                ),
                rx.button(
                    "Generate Client Forecast",
                    on_click=GridGuardState.run_selected_client_forecast,
                    background="#1687F8",
                    color="#FFFFFF",
                    width="100%",
                ),
                rx.text(
                    GridGuardState.client_forecast_message,
                    color=rx.cond(
                        GridGuardState.client_forecast_success,
                        "#2DD4BF",
                        "#F4B860",
                    ),
                    size="2",
                ),
                rx.cond(
                    GridGuardState.client_forecast_success,
                    rx.vstack(
                        rx.button(
                            rx.cond(
                                GridGuardState.client_forecast_results_visible,
                                "Hide Results",
                                "Show Results",
                            ),
                            on_click=GridGuardState.toggle_client_forecast_results,
                            background="#24374F",
                            color="#FFFFFF",
                            width="100%",
                        ),
                        rx.cond(
                            GridGuardState.client_forecast_results_visible,
                            rx.vstack(
                                rx.grid(
                                    metric_card(
                                        "Selected client",
                                        GridGuardState.selected_client,
                                        "Client being forecast.",
                                        "user-round",
                                    ),
                                    metric_card(
                                        "Average estimated usage",
                                        GridGuardState.client_forecast_average_kw + " kW",
                                        "Average estimated hourly usage over the selected forecast period.",
                                        "zap",
                                    ),
                                    metric_card(
                                        "Peak estimated usage",
                                        GridGuardState.client_forecast_peak_kw + " kW",
                                        "Highest estimated hourly usage in the forecast period.",
                                        "trending-up",
                                    ),
                                    metric_card(
                                        "Minimum estimated usage",
                                        GridGuardState.client_forecast_minimum_kw + " kW",
                                        "Lowest estimated hourly usage in the forecast period.",
                                        "trending-down",
                                    ),
                                    columns=rx.breakpoints(initial="1", sm="2"),
                                    spacing="4",
                                    width="100%",
                                ),
                                section_card(
                                    "Your Estimated Electricity Use",
                                    "Each point shows the estimated usage for one future hour.",
                                    rx.recharts.line_chart(
                                        rx.recharts.line(
                                            data_key="predicted_kw",
                                            name="Estimated usage",
                                            stroke="#55B8FF",
                                            type="monotone",
                                            dot=True,
                                        ),
                                        rx.recharts.x_axis(
                                            data_key="timestamp",
                                            hide=True,
                                        ),
                                        rx.recharts.y_axis(
                                            label={
                                                "value": "Power (kW)",
                                                "angle": -90,
                                                "position": "insideLeft",
                                            },
                                        ),
                                        rx.recharts.cartesian_grid(
                                            stroke="#24374F",
                                            stroke_dasharray="3 3",
                                        ),
                                        rx.recharts.tooltip(),
                                        rx.recharts.legend(),
                                        data=GridGuardState.client_forecast_chart_data,
                                        width="100%",
                                        height=320,
                                    ),
                                ),
                                align="start",
                                spacing="4",
                                width="100%",
                            ),
                        ),
                        align="start",
                        spacing="4",
                        width="100%",
                    ),
                ),
                rx.text(
                    "This is a historical-pattern baseline, not a trained "
                    "client-specific AI model or a live meter prediction. "
                    "Actual future usage may differ.",
                    color="#F4B860",
                    size="2",
                ),
                align="start",
                spacing="4",
                width="100%",
            ),
        ),

        section_card(
            "About These Results",
            "Understand the data behind the forecasts.",
            rx.vstack(
                rx.text(
                    f"Historical customer: "
                    f"{metrics.get('client_id', 'Not recorded')}",
                    color="#F2F7FF",
                    size="2",
                ),
                rx.text(
                    f"Records used to train the AI: "
                    f"{int(metrics.get('training_records', 0)):,}",
                    color="#C8D5E5",
                    size="2",
                ),
                rx.text(
                    f"Records used to test the AI: "
                    f"{int(metrics.get('testing_records', 0)):,}",
                    color="#C8D5E5",
                    size="2",
                ),
                rx.text(
                    "These historical comparisons use saved test results. "
                    "They are not live meter readings. Future forecasts "
                    "are model estimates, not guaranteed electricity use.",
                    color="#F4B860",
                    size="2",
                ),
                align="start",
                spacing="3",
                width="100%",
            ),
        ),

        rx.text(
            "Source: saved GridGuard historical client forecast test results",
            color="#91A4BC",
            size="1",
        ),

        align="start",
        spacing="5",
        width="100%",
    )


# DATA EXPLORER

def placeholder_page(
    title: str,
    description: str,
    detail: str,
) -> rx.Component:
    return rx.vstack(
        back_to_overview_button(),
        rx.text(
            "GRIDGUARD / WORKSPACE",
            color="#55B8FF",
            size="1",
            weight="bold",
        ),
        rx.heading(title, size="8", color="#F2F7FF"),
        rx.text(description, color="#91A4BC", size="3"),
        historical_notice(),
        rx.box(
            rx.icon(tag="construction", color="#F4B860", size=28),
            rx.heading(
                "Workspace ready for integration",
                size="5",
                color="#F2F7FF",
            ),
            rx.text(detail, color="#91A4BC"),
            background="#111F32",
            border="1px solid #24374F",
            border_radius="14px",
            padding="24px",
            width="100%",
        ),
        align="start",
        spacing="4",
        width="100%",
    )


# PAGE ROUTING AND APPLICATION


# PAGE ROUTING AND APPLICATION

def page_content() -> rx.Component:
    return rx.match(
        GridGuardState.active_page,
        ("Overview", overview_page()),
        ("Client Forecast", client_forecast_page()),
        ("Demand Analysis", demand_analysis_page()),
        ("Anomaly Detection", anomaly_detection_page()),
        ("ML Performance", ml_performance_page()),
        ("Model Comparison", model_comparison_page()),
        (
            "Data Explorer",
            placeholder_page(
                "Data Explorer",
                "Explore the structure and coverage of recorded electricity data.",
                "This page is not connected to the processed-data preview yet.",
            ),
        ),
        overview_page(),
    )


def index() -> rx.Component:
    return rx.box(
        rx.hstack(
            sidebar(),
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.vstack(
                            rx.text(
                                "GRIDGUARD INTELLIGENCE",
                                size="1",
                                color="#91A4BC",
                                weight="bold",
                            ),
                            rx.heading(
                                GridGuardState.active_page,
                                size="5",
                                color="#F2F7FF",
                            ),
                            align="start",
                            spacing="1",
                        ),
                        rx.spacer(),
                        rx.hstack(
                            rx.box(
                                width="8px",
                                height="8px",
                                border_radius="50%",
                                background="#55B8FF",
                            ),
                            rx.text(
                                "HISTORICAL MODE",
                                color="#91A4BC",
                                size="1",
                                weight="bold",
                            ),
                            align="center",
                            spacing="2",
                        ),
                        align="center",
                        width="100%",
                    ),
                    rx.divider(color="#24374F"),
                    page_content(),
                    align="start",
                    spacing="5",
                    width="100%",
                ),
                padding="28px",
                width="100%",
                min_height="100vh",
                background="#0D1828",
            ),
            align="start",
            spacing="0",
            width="100%",
        ),
        background="#0D1828",
        color="#F2F7FF",
        min_height="100vh",
        width="100%",
    )


app = rx.App()
app.add_page(index, route="/", title="GridGuard | Electricity Intelligence")
