"""Central settings. Every other module imports its constants from here,
so there is exactly one place to change a date, a path or a market rule."""

from pathlib import Path

import pandas as pd

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # the project folder
DATA_DIR = ROOT / "data"                    # local Parquet store (git-ignored)

# --- Market conventions ----------------------------------------------------
# The day-ahead auction is run in local German time. We STORE everything in UTC
# (no ambiguity on clock-change days) and only convert to local time when we
# need a calendar concept such as "delivery day" or "hour of day".
MARKET_TZ = "Europe/Berlin"
BIDDING_ZONE = "DE-LU"

# Orders for delivery day D must be in by 12:00 local time on D-1.
# Anything published after this moment must not be used to forecast day D.
GATE_CLOSURE_HOUR = 12

# First delivery day cleared in 15-minute products instead of hourly ones.
# (Confirmed in the SMARD data: quarter-hour prices start to differ from here.)
QUARTER_HOUR_START = pd.Timestamp("2025-10-01", tz=MARKET_TZ)

# --- Download windows ------------------------------------------------------
# Market data: 2023 onwards gives a full year of price history before the
# weather-forecast archive begins, which the lag features need.
MARKET_START = "2023-01-01"
# Weather forecasts: Open-Meteo archives past model runs from early 2024 only.
WEATHER_START = "2024-01-01"

# --- Weather-forecast availability -----------------------------------------
# A weather model run is not usable the moment it is initialised: DWD needs a
# few hours to compute and publish it. We assume a run initialised at time t
# reaches a trader's screen at t + 4h. This is deliberately conservative.
WEATHER_PUBLICATION_DELAY = pd.Timedelta(hours=4)
