"""OPTIONAL: download extra data from the ENTSO-E Transparency Platform.

Needs a personal API token (free). Written for entsoe-py 0.8.

What ENTSO-E adds on top of SMARD
---------------------------------
* generation by fuel type (gas, lignite, hard coal, ...) -> which plant is marginal
* cross-border physical flows -> is Germany importing or exporting
* a second copy of prices and forecasts to cross-check SMARD against
* unavailability of generation units (outages) - see the warning below

WARNING about outages and look-ahead
------------------------------------
Outage messages get revised (a plant returns late, a derating changes). The
API only returns the LATEST revision of each message, stamped with the time of
that revision. Example seen in the data: an outage in March 2026 whose only
available version was written in August 2026. So the history you download
today is NOT what a trader saw back then, and filtering by time stamp would
silently drop outages that were known. A point-in-time outage feature would
need a daily snapshot collected going forward. `download_outages` is therefore
kept for exploration only and is not used by the model.

The token is read from a file called `.env` (see `.env.example`). That file is
git-ignored so the token never ends up on GitHub.
"""

import os
import time

import pandas as pd
from dotenv import load_dotenv

from powerfc.config import ROOT

ZONE = "DE_LU"  # entsoe-py's code for the German-Luxembourg bidding zone
# Bidding zones electrically connected to DE-LU
NEIGHBOURS = ["FR", "NL", "BE", "AT", "CH", "CZ", "PL", "DK_1", "DK_2", "NO_2", "SE_4"]


def get_client():
    """Create the API client, or explain what to do if no token is configured."""
    from entsoe import EntsoePandasClient  # imported here so the rest of the project works without a token

    load_dotenv(ROOT / ".env")
    token = os.getenv("ENTSOE_API_TOKEN")
    if not token:
        raise SystemExit(
            "No ENTSO-E token found. Copy .env.example to .env and paste your token "
            "after ENTSOE_API_TOKEN=. Everything else in the project runs without it."
        )
    return EntsoePandasClient(api_key=token)


def _with_retries(query, *args, retries: int = 5, **kwargs):
    """Run one API query, retrying when the ENTSO-E server times out.

    The platform regularly answers with a temporary server error on large
    requests. We also re-raise failures WITHOUT the original message, because
    that message contains the request URL - including the secret token - and
    would otherwise end up in your terminal history or a log file.
    """
    for attempt in range(retries):
        try:
            return query(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - the library raises several types
            status = getattr(getattr(error, "response", None), "status_code", None)
            if attempt == retries - 1:
                raise RuntimeError(
                    f"ENTSO-E query {query.__name__} failed after {retries} attempts "
                    f"({type(error).__name__}, HTTP status {status})"
                ) from None
            time.sleep(10 * (attempt + 1))


def _yearly_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """The API limits one request to about a year, so split long ranges."""
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.DateOffset(years=1), end)
        yield chunk_start, chunk_end
        chunk_start = chunk_end


def _to_utc(obj):
    """entsoe-py returns local (Europe/Berlin) timestamps; our store is UTC."""
    obj.index = obj.index.tz_convert("UTC")
    obj.index.name = "time_utc"
    return obj


def download_generation_by_fuel(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Actual generation per production type, in MW."""
    parts = [_with_retries(client.query_generation, ZONE, start=s, end=e) for s, e in _yearly_chunks(start, end)]
    table = _to_utc(pd.concat(parts))
    # Some fuels come as ("Fuel", "Actual Aggregated") / ("Fuel", "Actual Consumption") pairs.
    if isinstance(table.columns, pd.MultiIndex):
        table = table.xs("Actual Aggregated", axis=1, level=1)
    return table


def download_crossborder_flows(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Net physical export from DE-LU to each neighbour, in MW (negative = import)."""
    columns = {}
    for neighbour in NEIGHBOURS:
        exports = pd.concat(
            [_with_retries(client.query_crossborder_flows, ZONE, neighbour, start=s, end=e) for s, e in _yearly_chunks(start, end)]
        )
        imports = pd.concat(
            [_with_retries(client.query_crossborder_flows, neighbour, ZONE, start=s, end=e) for s, e in _yearly_chunks(start, end)]
        )
        columns[f"net_export_{neighbour}"] = exports - imports
    return _to_utc(pd.DataFrame(columns))


def download_outages(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Planned and forced outages of generation units (exploration only).

    Each row is one segment of an outage message, in its latest revision.
    NOT point-in-time - read the warning at the top of this file before
    using it for anything that feeds the backtest. Keep the range short:
    a few days already returns hundreds of rows.
    """
    table = _with_retries(client.query_unavailability_of_generation_units, ZONE, start=start, end=end)
    return table.reset_index()  # the index is the revision time; keep it as a column


def download_crosscheck(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Prices and day-ahead forecasts, to compare with what SMARD gave us."""
    prices = pd.concat([_with_retries(client.query_day_ahead_prices, ZONE, start=s, end=e) for s, e in _yearly_chunks(start, end)])
    load_fc = pd.concat([_with_retries(client.query_load_forecast, ZONE, start=s, end=e) for s, e in _yearly_chunks(start, end)])
    res_fc = pd.concat(
        [_with_retries(client.query_wind_and_solar_forecast, ZONE, start=s, end=e) for s, e in _yearly_chunks(start, end)]
    )
    table = pd.concat([prices.rename("price"), load_fc, res_fc], axis=1)
    return _to_utc(table)
