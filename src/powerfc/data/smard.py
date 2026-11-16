"""Download market data from SMARD (Bundesnetzagentur). No token required.

SMARD republishes the data the German TSOs report to ENTSO-E, under a
CC BY 4.0 licence. It is our default source because it needs no registration.

How the SMARD chart API works
-----------------------------
Every time series has a numeric id (a "filter"). Data is served in weekly
files. You first ask for an index listing the start timestamp of every weekly
file, then download the files you need:

    index: /chart_data/{id}/{region}/index_{resolution}.json
    file:  /chart_data/{id}/{region}/{id}_{region}_{resolution}_{start}.json

Each file is a list of [timestamp_in_milliseconds_UTC, value] pairs.
Because the timestamps are UTC, clock changes need no special handling here.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from powerfc.config import BIDDING_ZONE

BASE_URL = "https://www.smard.de/app/chart_data"

# Our column name -> SMARD series id.
# "_fc" = day-ahead forecast published by the TSOs, "_actual" = metered outcome.
SERIES = {
    "price": 4169,                # day-ahead auction price, EUR/MWh
    "load_fc": 411,               # forecast grid load
    "load_actual": 410,           # realised grid load
    "wind_onshore_fc": 123,
    "wind_offshore_fc": 3791,
    "solar_fc": 125,
    "wind_onshore_actual": 4067,
    "wind_offshore_actual": 1225,
    "solar_actual": 4068,
}

# SMARD reports volumes as ENERGY per interval (MWh per quarter-hour).
# Traders think in average POWER (MW). 1 MWh delivered in 15 minutes = 4 MW.
MWH_PER_QUARTER_TO_MW = 4.0


def _get_json(url: str, retries: int = 4) -> dict:
    """GET a URL and parse JSON, retrying on temporary network errors."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # wait 1s, 2s, 4s before trying again


def download_series(series_id: int, start: pd.Timestamp, region: str = BIDDING_ZONE) -> pd.Series:
    """Download one SMARD series at quarter-hour resolution from `start` onwards.

    Returns a Series indexed by UTC timestamp (the START of each quarter-hour).
    """
    index_url = f"{BASE_URL}/{series_id}/{region}/index_quarterhour.json"
    file_starts = _get_json(index_url)["timestamps"]

    # Keep the weekly files that can contain data at or after `start`.
    # A file covers 7 days, so we also need the one that begins up to a week earlier.
    cutoff_ms = (start - pd.Timedelta(days=7)).timestamp() * 1000
    wanted = [ts for ts in file_starts if ts >= cutoff_ms]

    def fetch(file_start: int) -> list:
        url = f"{BASE_URL}/{series_id}/{region}/{series_id}_{region}_quarterhour_{file_start}.json"
        return _get_json(url)["series"]

    # Four parallel downloads: fast enough, and polite to a free public service.
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, wanted))

    rows = [row for file in files for row in file]
    if not rows:
        return pd.Series(dtype="float64")

    timestamps, values = zip(*rows)
    series = pd.Series(values, index=pd.to_datetime(timestamps, unit="ms", utc=True), dtype="float64")
    series.index.name = "time_utc"
    # SMARD pads the current week with empty values for periods not yet published.
    series = series.dropna()
    return series[series.index >= start].sort_index()


def download_market(start: pd.Timestamp) -> pd.DataFrame:
    """Download every series in SERIES and return one wide table (15-min, UTC).

    Volumes are converted to MW; the price stays in EUR/MWh.
    """
    columns = {}
    for name, series_id in SERIES.items():
        print(f"  SMARD {name:22s} (id {series_id}) ...", end=" ", flush=True)
        series = download_series(series_id, start)
        if name != "price":
            series = series * MWH_PER_QUARTER_TO_MW
        columns[name] = series
        print(f"{len(series):>7,} rows, up to {series.index.max():%Y-%m-%d %H:%M} UTC")
    return pd.DataFrame(columns)
