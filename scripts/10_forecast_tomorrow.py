"""Step 10 - the live forecast: prices for the next delivery day.

Run before 12:00 local time, after refreshing the data:

    .venv/bin/python scripts/01_download_market.py
    .venv/bin/python scripts/02_download_weather.py
    .venv/bin/python scripts/05_build_features.py
    .venv/bin/python scripts/10_forecast_tomorrow.py

Trains on all history and forecasts the first day that has inputs but no
auction result yet. The forecast is saved, so you can score it the next day.
"""

import _bootstrap  # noqa: F401  (makes `powerfc` importable, see scripts/_bootstrap.py)
import pandas as pd

from powerfc import store
from powerfc.config import ROOT
from powerfc.features import FEATURE_SETS
from powerfc.models import LightGBMModel, RidgePerHour

FEATURES = FEATURE_SETS["strict"]


def main() -> None:
    table = store.load("features_hourly")
    history = table.dropna(subset=["price", "price_lag_7d", "residual_load_wx", "residual_delta_7d_wx"])

    # Target: days after the last known auction result that have a full set of inputs.
    last_known_day = history["delivery_day"].max()
    future = table[(table["delivery_day"] > last_known_day) & table["residual_load_wx"].notna() & table["price_lag_1d"].notna()]
    if future.empty:
        raise SystemExit(
            f"No day to forecast: inputs end with the last auction result ({last_known_day:%Y-%m-%d}). "
            "Refresh the data before gate closure (see the top of this script)."
        )
    target_day = future["delivery_day"].min()
    target = future[future["delivery_day"] == target_day]

    forecast = pd.DataFrame({"hour": target["hour"]})
    forecast["ridge"] = RidgePerHour(FEATURES).fit(history).predict(target)
    forecast["lightgbm"] = LightGBMModel(FEATURES).fit(history).predict(target)
    forecast["ensemble"] = forecast[["ridge", "lightgbm"]].mean(axis=1)
    forecast["residual_load_gw"] = target["residual_load_wx"] / 1000

    out_dir = ROOT / "reports" / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"forecast_{target_day:%Y-%m-%d}.csv"
    forecast.round(2).to_csv(out_file)

    print(f"Forecast for delivery day {target_day:%A %d %B %Y} (EUR/MWh), trained on data up to {last_known_day:%Y-%m-%d}")
    print(forecast.round(1).to_string(index=False))
    print(f"\nBase (average of all hours): {forecast['ensemble'].mean():.1f}   "
          f"Peak (08-20h): {forecast.loc[forecast['hour'].between(8, 19), 'ensemble'].mean():.1f}")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
