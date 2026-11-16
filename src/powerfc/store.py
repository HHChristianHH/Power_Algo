"""A tiny local data store: one Parquet file per dataset.

Parquet is a columnar file format. Compared with CSV it keeps data types
(including time zones), is much smaller, and loads in milliseconds. If you
prefer SQL, DuckDB can query these files directly:
    SELECT * FROM 'data/market_15min.parquet' LIMIT 5;
"""

import pandas as pd

from powerfc.config import DATA_DIR


def path(name: str):
    return DATA_DIR / f"{name}.parquet"


def exists(name: str) -> bool:
    return path(name).exists()


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(path(name))


def save(df: pd.DataFrame, name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path(name))


def upsert(new: pd.DataFrame, name: str) -> pd.DataFrame:
    """Merge freshly downloaded rows into the stored dataset.

    "Upsert" = update rows that already exist, insert the ones that don't.
    Rows are identified by the index (timestamp, or timestamp + location).
    Where old and new overlap, the NEW value wins unless it is missing, because
    sources revise recent data (e.g. metered generation gets corrected).
    """
    if exists(name):
        old = load(name)
        # combine_first: take `new`, and fill its gaps (missing rows, missing
        # columns, NaN cells) from `old`.
        combined = new.combine_first(old)
    else:
        combined = new
    combined = combined.sort_index()
    save(combined, name)
    return combined
