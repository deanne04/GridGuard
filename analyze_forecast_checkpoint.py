from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

root = Path(__file__).resolve().parent
path = root / "results" / "forecast_horizon_validation" / "horizon_predictions_checkpoint.csv"

df = pd.read_csv(path)
print(f"Checkpoint rows: {len(df):,}")
print(f"Clients evaluated: {df['client_id'].nunique()}")

results = []

for horizon in range(1, 25):
    part = df[df["horizon_hours"] == horizon]
    if part.empty:
        continue

    actual = part["actual_kw"].to_numpy(dtype=float)
    print(f"\nHorizon {horizon}h | samples: {len(part):,}")

    for name, col in [
        ("Random Forest", "rf_predicted_kw"),
        ("Persistence baseline", "baseline_predicted_kw"),
    ]:
        pred = part[col].to_numpy(dtype=float)
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = (
            r2_score(actual, pred)
            if len(actual) > 1 and np.var(actual) > 0
            else float("nan")
        )

        results.append({
            "horizon_hours": horizon,
            "model": name,
            "mae_kw": mae,
            "rmse_kw": rmse,
            "r2": r2,
            "samples": len(part),
        })

        print(
            f"  {name}: MAE={mae:.3f} kW | "
            f"RMSE={rmse:.3f} kW | R2={r2:.4f}"
        )

out = (
    root / "results" / "forecast_horizon_validation"
    / "checkpoint_metrics_partial.csv"
)
pd.DataFrame(results).to_csv(out, index=False)

print(f"\nPartial metrics saved to: {out}")
print("WARNING: These results cover only the clients in the checkpoint.")
