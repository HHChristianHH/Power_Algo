"""Step 1 - download prices, load and renewable forecasts/actuals from SMARD.

Run:  .venv/bin/python scripts/01_download_market.py

The first run downloads everything since MARKET_START (a few minutes).
Later runs only fetch the last two weeks and merge them into the store, so
you can run this daily to keep the data current.
"""

import _bootstrap  # noqa: F401  (makes `powerfc` importable, see scripts/_bootstrap.py)
import pandas as pd

from powerfc import store
from powerfc.config import MARKET_START
from powerfc.data import smard

DATASET = "market_15min"


def main() -> None:
    if store.exists(DATASET):
        # Re-download the most recent 14 days: recent actuals get revised.
        last = store.load(DATASET).index.max()
        start = last - pd.Timedelta(days=14)
        print(f"Updating {DATASET} from {start:%Y-%m-%d} (store ends {last:%Y-%m-%d %H:%M} UTC)")
    else:
        start = pd.Timestamp(MARKET_START, tz="UTC")
        print(f"First download of {DATASET} from {start:%Y-%m-%d}")

    new = smard.download_market(start)
    combined = store.upsert(new, DATASET)
    print(f"Saved {len(combined):,} rows x {combined.shape[1]} columns to {store.path(DATASET)}")


if __name__ == "__main__":
    main()
