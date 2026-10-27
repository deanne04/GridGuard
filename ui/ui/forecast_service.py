
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "gridguard_best_forecast_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "electricity_readings_parquet"

LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [6, 24, 168]


@lru_cache(maxsize=1)
def load_forecast_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Forecast model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_hourly_demand():
    try:
        from pyspark.sql import SparkSession, functions as F
    except ImportError as exc:
        raise RuntimeError("PySpark is required to read the processed Parquet dataset.") from exc

    spark = (
        SparkSession.builder
        .appName("GridGuardForecastUI")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    readings = spark.read.parquet(str(DATA_PATH))
    hourly_spark = (
        readings
        .withColumn("hour_timestamp", F.date_trunc("hour", F.col("timestamp")))
        .groupBy("hour_timestamp")
        .agg(
            F.avg("reading_kw").alias("average_demand_kw"),
            F.count("reading_kw").alias("reading_count"),
        )
    )

    hourly = hourly_spark.toPandas()
    if hourly.empty:
        raise ValueError("No hourly demand records were found.")

    hourly["hour_timestamp"] = pd.to_datetime(hourly["hour_timestamp"])
    hourly = hourly.sort_values("hour_timestamp").reset_index(drop=True)
    hourly = hourly[hourly["reading_count"] >= 1480].copy()
    hourly = hourly.sort_values("hour_timestamp").reset_index(drop=True)

    for lag in LAGS:
        hourly[f"lag_{lag}h"] = hourly["average_demand_kw"].shift(lag)

    for window in ROLLING_WINDOWS:
        hourly[f"rolling_mean_{window}h"] = (
            hourly["average_demand_kw"].shift(1).rolling(window).mean()
        )

    ts = hourly["hour_timestamp"]
    hourly["hour"] = ts.dt.hour
    hourly["day_of_week"] = ts.dt.dayofweek
    hourly["month"] = ts.dt.month
    hourly["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    return hourly


def available_forecast_dates():
    """Keep the existing historical-date selector working."""
    hourly = load_hourly_demand()
    return sorted(hourly["hour_timestamp"].dt.strftime("%Y-%m-%d").unique().tolist())


def forecast_future(date_text: str, hour_text: str = "00", horizon: int = 24):
    """Forecast a future period using the saved Random Forest recursively."""
    if not 1 <= int(horizon) <= 24:
        return {"ok": False, "message": "Forecast horizon must be between 1 and 24 hours."}

    try:
        start = pd.Timestamp(f"{date_text} {int(hour_text):02d}:00:00")
    except (TypeError, ValueError):
        return {"ok": False, "message": "Choose a valid date and hour."}

    if start.minute != 0 or start.second != 0:
        return {"ok": False, "message": "Choose an exact hour."}

    hourly = load_hourly_demand()
    bundle = load_forecast_model()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    latest_history = hourly.dropna(subset=["average_demand_kw"])
    if len(latest_history) < 168:
        return {"ok": False, "message": "At least 168 historical hours are required."}

    last_timestamp = latest_history["hour_timestamp"].iloc[-1]
    if start <= last_timestamp:
        return {
            "ok": False,
            "message": f"Choose a date after the historical data ends ({last_timestamp:%Y-%m-%d %H:%M}).",
        }

    # Seed the demand history with the last 168 actual hourly observations.
    # For a date years beyond the dataset, this is a demonstration forecast,
    # not a claim that these are current readings.
    history = latest_history["average_demand_kw"].tail(168).astype(float).tolist()
    predictions = []

    for step in range(int(horizon)):
        timestamp = start + pd.Timedelta(hours=step)
        features = {}

        for lag in LAGS:
            features[f"lag_{lag}h"] = history[-lag]

        for window in ROLLING_WINDOWS:
            features[f"rolling_mean_{window}h"] = sum(history[-window:]) / window

        features["hour"] = timestamp.hour
        features["day_of_week"] = timestamp.dayofweek
        features["month"] = timestamp.month
        features["is_weekend"] = int(timestamp.dayofweek >= 5)

        model_input = pd.DataFrame(
            [[features[column] for column in feature_columns]],
            columns=feature_columns,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            predicted_kw = float(model.predict(model_input)[0])

        if not pd.notna(predicted_kw):
            return {"ok": False, "message": "The model returned an invalid prediction."}

        # Demand cannot be negative.
        predicted_kw = max(0.0, predicted_kw)
        history.append(predicted_kw)

        predictions.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
            "predicted_kw": round(predicted_kw, 2),
        })

    values = [row["predicted_kw"] for row in predictions]
    return {
        "ok": True,
        "message": "Experimental future forecast generated by the saved Random Forest model.",
        "forecast": predictions,
        "start_hour": predictions[0]["timestamp"],
        "end_hour": predictions[-1]["timestamp"],
        "hours": len(predictions),
        "average_kw": round(sum(values) / len(values), 2),
        "peak_kw": round(max(values), 2),
        "minimum_kw": round(min(values), 2),
    }
