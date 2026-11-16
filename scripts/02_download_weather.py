"""Step 2 - download archived ICON weather forecasts from Open-Meteo.

Run:  .venv/bin/python scripts/02_download_weather.py

First run: everything since WEATHER_START. Later runs: the last few days up
to tomorrow (the day you would be forecasting).
"""

import time

import _bootstrap  # noqa: F401  (makes `powerfc` importable, see scripts/_bootstrap.py)
import pandas as pd

from powerfc import store
from powerfc.config import WEATHER_START
from powerfc.data import openmeteo

DATASET = "weather_forecast_hourly"


def main() -> None:
    # Tomorrow is the furthest day the "predicted 24h/48h before" archive can serve.
    end = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if store.exists(DATASET):
        last = store.load(DATASET).index.get_level_values("time_utc").max()
        start = (last - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        print(f"Updating {DATASET} from {start}")
    else:
        start = WEATHER_START
        print(f"First download of {DATASET} from {start}")

    # Download month by month and save after every chunk. If the connection
    # drops halfway, just run the script again: it resumes where it stopped.
    for first_day, last_day in openmeteo.chunk_periods(start, end):
        print(f"  Open-Meteo {first_day:%Y-%m-%d} -> {last_day:%Y-%m-%d} ...", end=" ", flush=True)
        new = openmeteo.download_forecasts(first_day, last_day)
        if new.empty:
            print("nothing archived for this period")
        else:
            combined = store.upsert(new, DATASET)
            print(f"{len(new):,} rows (store now {len(combined):,})")
        time.sleep(1)  # be polite to a free service

    if store.exists(DATASET):
        combined = store.load(DATASET)
        print(f"Saved {len(combined):,} rows x {combined.shape[1]} columns to {store.path(DATASET)}")


if __name__ == "__main__":
    main()
