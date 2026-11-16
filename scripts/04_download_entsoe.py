"""OPTIONAL step - download extra datasets from ENTSO-E (needs a token in .env).

Run:  .venv/bin/python scripts/04_download_entsoe.py

Without a token the script stops immediately with instructions and changes
nothing. Each dataset is saved as soon as it is downloaded.
"""

import _bootstrap  # noqa: F401  (makes `powerfc` importable, see scripts/_bootstrap.py)
import pandas as pd

from powerfc import store
from powerfc.config import MARKET_START, MARKET_TZ
from powerfc.data import entsoe_client


def main() -> None:
    client = entsoe_client.get_client()  # exits with a helpful message if there is no token

    # ENTSO-E expects time-zone-aware timestamps.
    start = pd.Timestamp(MARKET_START, tz=MARKET_TZ)
    end = pd.Timestamp.now(tz=MARKET_TZ).normalize() + pd.Timedelta(days=2)

    # Outages are deliberately not in this list: the API cannot give a
    # point-in-time history (see the warning in powerfc/data/entsoe_client.py).
    jobs = {
        "entsoe_generation_by_fuel": entsoe_client.download_generation_by_fuel,
        "entsoe_crosscheck": entsoe_client.download_crosscheck,
        "entsoe_crossborder_flows": entsoe_client.download_crossborder_flows,
    }
    for dataset, download in jobs.items():
        print(f"Downloading {dataset} ...", flush=True)
        table = download(client, start, end)
        store.save(table, dataset)
        print(f"  saved {len(table):,} rows x {table.shape[1]} columns to {store.path(dataset)}", flush=True)


if __name__ == "__main__":
    main()
