"""Step 9 - extend the hourly forecast to the 15-minute products (since Oct 2025).

Run:  .venv/bin/python scripts/09_quarter_hour.py
"""

import _bootstrap  # noqa: F401  (makes `powerfc` importable, see scripts/_bootstrap.py)
import pandas as pd

from powerfc import quarter_hour, store
from powerfc.config import ROOT


def main() -> None:
    price_15min = store.load("market_15min")["price"]
    hourly_forecast = store.load("backtest_predictions")["lightgbm"]
    result = quarter_hour.quarter_hour_forecast(price_15min, hourly_forecast)

    mae = {column: (result[column] - result["price"]).abs().mean()
           for column in ["flat", "shaped", "perfect_hour_flat", "perfect_hour_shaped"]}
    table = pd.Series(mae).rename(
        {
            "flat": "Hourly forecast repeated 4x",
            "shaped": "Hourly forecast + 28-day intra-hour pattern",
            "perfect_hour_flat": "Perfect hourly price repeated 4x (shape error only)",
            "perfect_hour_shaped": "Perfect hourly price + pattern",
        }
    )
    table.round(2).rename("MAE").to_csv(ROOT / "reports" / "quarter_hour_metrics.csv")

    first, last = result.index.min(), result.index.max()
    print(f"Quarter-hour prices, {first:%Y-%m-%d} -> {last:%Y-%m-%d}, {len(result):,} quarter-hours. MAE in EUR/MWh:")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
