"""Download archived weather FORECASTS (not observations) from Open-Meteo.

Why forecasts and not actual weather
------------------------------------
When you bid into the day-ahead auction you do not know tomorrow's weather,
only a forecast of it. Training on observed weather would hand the model
information no trader had ("look-ahead bias") and make the backtest look far
better than live trading ever would.

Which API
---------
Open-Meteo's *Previous Runs API* stores what the weather model predicted a
fixed number of days before each hour:

    wind_speed_100m_previous_day1 = value predicted 24 hours before valid time
    wind_speed_100m_previous_day2 = value predicted 48 hours before valid time

We download both lead times here and keep them raw. Which one is allowed for
which delivery hour is decided later, in the feature step, where the
gate-closure rule lives. Model: DWD ICON (the German weather service's model).

Time-stamp conventions (important when building features)
---------------------------------------------------------
* wind speed and temperature are instantaneous values at the time stamp;
* shortwave_radiation is the AVERAGE OVER THE PRECEDING HOUR, so the value
  stamped 12:00 describes 11:00-12:00.
"""

import time

import pandas as pd
import requests

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
WEATHER_MODEL = "icon_seamless"  # DWD ICON, best available resolution per lead time

# Open-Meteo variable name -> our short column prefix
VARIABLES = {
    "wind_speed_100m": "wind100",    # m/s at 100 m, roughly turbine hub height
    "shortwave_radiation": "ghi",    # W/m2, global horizontal irradiance (drives solar output)
    "temperature_2m": "temp",        # deg C, drives heating/cooling demand
}
LEAD_DAYS = [1, 2]  # forecast made 24h and 48h before valid time

# Representative points across Germany. The weights say how much of the
# national wind / solar capacity or demand each point stands for.
# They are ROUGH, hand-set shares (each column sums to 1) meant to put wind in
# the north, solar in the south and demand in the population centres. A good
# refinement exercise is to derive them from the Marktstammdatenregister.
LOCATIONS = pd.DataFrame(
    [
        # name,                  lat,   lon,  on,   off,  solar, load
        ("north_sea_offshore",   54.4,  6.9, 0.00, 0.80, 0.00, 0.00),
        ("baltic_offshore",      54.7, 13.6, 0.00, 0.20, 0.00, 0.00),
        ("schleswig_holstein",   54.4,  9.2, 0.14, 0.00, 0.03, 0.03),
        ("lower_saxony_coast",   53.3,  7.8, 0.12, 0.00, 0.04, 0.04),
        ("lower_saxony_inland",  52.5,  9.8, 0.09, 0.00, 0.05, 0.07),
        ("mecklenburg",          53.8, 12.3, 0.06, 0.00, 0.04, 0.02),
        ("brandenburg_berlin",   52.4, 13.4, 0.14, 0.00, 0.09, 0.08),
        ("saxony_anhalt",        52.0, 11.6, 0.09, 0.00, 0.05, 0.03),
        ("nrw_ruhr",             51.5,  7.4, 0.12, 0.00, 0.11, 0.22),
        ("hesse_rhine_main",     50.1,  8.7, 0.10, 0.00, 0.09, 0.13),
        ("saxony_thuringia",     50.9, 12.2, 0.05, 0.00, 0.07, 0.07),
        ("baden_wuerttemberg",   48.7,  9.2, 0.03, 0.00, 0.13, 0.13),
        ("bavaria_north",        49.5, 11.0, 0.04, 0.00, 0.12, 0.07),
        ("bavaria_south",        48.3, 11.7, 0.02, 0.00, 0.18, 0.11),
    ],
    columns=["location", "lat", "lon", "w_wind_onshore", "w_wind_offshore", "w_solar", "w_load"],
).set_index("location")


# One request covers this many days. Measured: 30 days x 14 locations returns
# in 3-8 seconds, whereas 90 days ran into read timeouts.
CHUNK_DAYS = 30


def _request(params: dict, retries: int = 5) -> list:
    """Call the API, retrying on timeouts and on the free-tier rate limit."""
    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=params, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(5 * (attempt + 1))  # network hiccup: wait a little longer each time
            continue
        if response.status_code == 429:
            time.sleep(60)  # rate limit is per minute, so one minute is enough
            continue
        response.raise_for_status()
        payload = response.json()
        # One location returns a dict, several return a list. Normalise to a list.
        return payload if isinstance(payload, list) else [payload]
    raise RuntimeError("Open-Meteo: request still failing after several retries")


def chunk_periods(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a date range into consecutive (first_day, last_day) chunks."""
    last_day = pd.Timestamp(end)
    return [
        (first, min(first + pd.Timedelta(days=CHUNK_DAYS - 1), last_day))
        for first in pd.date_range(start, end, freq=f"{CHUNK_DAYS}D")
    ]


def download_forecasts(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download archived forecasts for all LOCATIONS for one chunk of days (inclusive).

    Returns a table indexed by (time_utc, location) with one column per
    variable and lead time, e.g. wind100_d1, wind100_d2, ghi_d1, ...
    """
    # Build the list of API columns, e.g. "wind_speed_100m_previous_day1"
    api_columns = {
        f"{variable}_previous_day{lead}": f"{short}_d{lead}"
        for variable, short in VARIABLES.items()
        for lead in LEAD_DAYS
    }
    payload = _request(
        {
            "latitude": ",".join(str(v) for v in LOCATIONS["lat"]),
            "longitude": ",".join(str(v) for v in LOCATIONS["lon"]),
            "hourly": ",".join(api_columns),
            "models": WEATHER_MODEL,
            "start_date": f"{start:%Y-%m-%d}",
            "end_date": f"{end:%Y-%m-%d}",
            "timezone": "UTC",         # keep everything in UTC, like the market data
            "wind_speed_unit": "ms",
        }
    )
    frames = []
    # The API returns locations in the order we asked for them.
    for location, result in zip(LOCATIONS.index, payload):
        frame = pd.DataFrame(result["hourly"]).rename(columns=api_columns)
        frame["time_utc"] = pd.to_datetime(frame.pop("time"), utc=True)
        frame["location"] = location
        frames.append(frame)

    table = pd.concat(frames).set_index(["time_utc", "location"]).sort_index()
    # Drop hours where nothing at all is archived (before the archive starts).
    return table.astype("float64").dropna(how="all")
