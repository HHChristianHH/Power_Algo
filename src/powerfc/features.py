"""Build the model table: one row per delivery hour.

The golden rule: every column in a row must have been knowable at gate
closure (12:00 local time on the day before delivery). Each block below says
why its columns pass that test.

The table also contains columns that are NOT allowed as inputs (actual
generation, the price itself). They are kept for analysis only. Which columns
a model may use is defined once, in FEATURE_SETS at the bottom.
"""

import holidays
import numpy as np
import pandas as pd

from powerfc import renewables, store
from powerfc.config import MARKET_TZ

# ---------------------------------------------------------------------------
# Helpers for "same hour, N days earlier"
# ---------------------------------------------------------------------------


def local_day_and_hour(index_utc: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.Index]:
    """Delivery day (local midnight, time zone dropped) and local hour 0-23."""
    local = index_utc.tz_convert(MARKET_TZ).tz_localize(None)
    return local.normalize(), pd.Index(local.hour)


def daily_matrix(series: pd.Series) -> pd.DataFrame:
    """Rearrange an hourly series into a table: one row per day, 24 hour columns.

    This is how traders look at power prices, and it makes clock changes easy:
    * 25-hour day: the repeated hour (02:00 occurs twice) is averaged;
    * 23-hour day: the hour that does not exist (02:00) is copied from 01:00.
    Every day ends up with exactly 24 values.
    """
    day, hour = local_day_and_hour(series.index)
    matrix = series.groupby([day, hour]).mean().unstack()
    matrix = matrix.reindex(pd.date_range(matrix.index.min(), matrix.index.max(), freq="D"))
    return matrix.ffill(axis=1)


def same_hour_days_ago(series: pd.Series, days: int) -> pd.Series:
    """Value of `series` at the same local hour, `days` days earlier.

    Shifting by 24 hours in UTC would be off by one hour around clock
    changes, so we look the value up in the day x hour matrix instead.
    """
    matrix = daily_matrix(series)
    shifted = matrix.shift(days, freq="D").stack()   # index: (day, hour)
    day, hour = local_day_and_hour(series.index)
    values = shifted.reindex(pd.MultiIndex.from_arrays([day, hour])).to_numpy()
    return pd.Series(values, index=series.index)


def daily_stat_days_ago(series: pd.Series, stat: str, days: int = 1) -> pd.Series:
    """A daily statistic (mean/min/max) of `series`, taken `days` days earlier.

    Computed over the hours the day really had (23, 24 or 25).
    """
    day, _ = local_day_and_hour(series.index)
    daily = series.groupby(day).agg(stat).shift(days, freq="D")
    return pd.Series(daily.reindex(day).to_numpy(), index=series.index)


# ---------------------------------------------------------------------------
# The feature table
# ---------------------------------------------------------------------------


def build_features() -> pd.DataFrame:
    # --- Raw inputs, brought to hourly resolution ---------------------------
    # Averaging the four quarter-hours gives the hourly price. Since the
    # auction went to 15-minute products (Oct 2025) this average is the hourly
    # reference price; before that the four values were identical anyway.
    market = store.load("market_15min").resample("h").mean()
    weather = store.load("weather_forecast_hourly")

    table = pd.DataFrame(index=market.index)
    table["delivery_day"], table["hour"] = local_day_and_hour(table.index)
    table["price"] = market["price"]  # the TARGET - never a feature for its own hour

    # --- Calendar -----------------------------------------------------------
    # Known years in advance, so trivially available.
    day = table["delivery_day"]
    german_holidays = holidays.Germany(years=range(day.dt.year.min(), day.dt.year.max() + 1))
    # National holidays only; 24 and 31 December behave like holidays too.
    is_holiday = day.dt.date.map(lambda d: d in german_holidays) | (
        (day.dt.month == 12) & day.dt.day.isin([24, 31])
    )
    table["weekday"] = day.dt.dayofweek                     # 0 = Monday
    table["is_weekend"] = (table["weekday"] >= 5).astype(int)
    table["is_holiday"] = is_holiday.astype(int)
    # Season as a point on a circle, so 31 Dec and 1 Jan are neighbours.
    angle = 2 * np.pi * day.dt.dayofyear / 365.25
    table["season_sin"], table["season_cos"] = np.sin(angle), np.cos(angle)

    # --- Recent prices ------------------------------------------------------
    # The auction for day D-1 cleared on D-2, so at gate closure for D all of
    # D-1's prices are public. Lags of one day or more are therefore safe.
    price = table["price"]
    table["price_lag_1d"] = same_hour_days_ago(price, 1)
    table["price_lag_2d"] = same_hour_days_ago(price, 2)
    table["price_lag_7d"] = same_hour_days_ago(price, 7)
    table["price_d1_mean"] = daily_stat_days_ago(price, "mean")
    table["price_d1_min"] = daily_stat_days_ago(price, "min")
    table["price_d1_max"] = daily_stat_days_ago(price, "max")
    # Average level of the last week: a proxy for fuel and carbon costs, which
    # move slowly and for which we have no free data.
    daily_mean = price.groupby(table["delivery_day"]).mean()
    week_mean = daily_mean.rolling(7).mean().shift(1, freq="D")  # days D-7 ... D-1
    table["price_7d_mean"] = week_mean.reindex(day).to_numpy()

    # --- Demand -------------------------------------------------------------
    # The TSOs must publish the day-ahead load forecast two hours before gate
    # closure, and we saw it online in time. Tiny gaps (an hour here and
    # there) are bridged from neighbouring hours of the same forecast.
    table["load_fc"] = market["load_fc"].interpolate(limit=3)

    # --- Wind and solar from weather forecasts (STRICT: no look-ahead) ------
    selected = renewables.select_available_forecast(weather)
    inputs = renewables.weather_inputs(selected)
    for tech in ["wind_onshore", "wind_offshore", "solar"]:
        table[f"{tech}_wx"] = renewables.walk_forward_nowcast(inputs[tech], market[f"{tech}_actual"])
    table = table.join(renewables.weighted_national(selected))

    # --- Wind and solar from the TSOs (COMPARISON ONLY) ---------------------
    # Possibly published after gate closure - see README. Used only in a run
    # that is clearly labelled as such, to measure how much it would flatter us.
    for tech in ["wind_onshore", "wind_offshore", "solar"]:
        table[f"{tech}_tso"] = market[f"{tech}_fc"].interpolate(limit=3)

    # --- Residual load and friends, once per renewable source ---------------
    for source in ["wx", "tso"]:
        renewable = table[[f"wind_onshore_{source}", f"wind_offshore_{source}", f"solar_{source}"]].sum(
            axis=1, min_count=3  # missing if any of the three is missing
        )
        residual = table["load_fc"] - renewable
        # Residual load: what conventional plants and imports must cover.
        table[f"residual_load_{source}"] = residual
        # Renewable share: above ~100 % the market has to export, store or curtail.
        table[f"renewable_share_{source}"] = renewable / table["load_fc"]
        # Change versus the same hour yesterday and last week. Read together
        # with the price lags this says: "yesterday cost X; tomorrow needs
        # 5 GW more from conventional plants".
        table[f"residual_delta_1d_{source}"] = residual - same_hour_days_ago(residual, 1)
        table[f"residual_delta_7d_{source}"] = residual - same_hour_days_ago(residual, 7)
        # Shape of the whole day: tight evenings push up every hour's price.
        table[f"residual_day_mean_{source}"] = daily_stat_days_ago(residual, "mean", days=0)
        table[f"residual_day_max_{source}"] = daily_stat_days_ago(residual, "max", days=0)
        table[f"residual_day_min_{source}"] = daily_stat_days_ago(residual, "min", days=0)

    # --- Fuel mix and trade two days ago (optional, needs ENTSO-E) ----------
    # Metered data for D-1 is incomplete at gate closure; D-2 is complete.
    if store.exists("entsoe_generation_by_fuel"):
        fuel = store.load("entsoe_generation_by_fuel").resample("h").mean()
        table["gas_gen_d2"] = daily_stat_days_ago(fuel["Fossil Gas"].reindex(table.index), "mean", days=2)
        coal = fuel["Fossil Hard coal"] + fuel["Fossil Brown coal/Lignite"]
        table["coal_gen_d2"] = daily_stat_days_ago(coal.reindex(table.index), "mean", days=2)
    if store.exists("entsoe_crossborder_flows"):
        flows = store.load("entsoe_crossborder_flows").resample("h").mean().sum(axis=1, min_count=1)
        table["net_export_d2"] = daily_stat_days_ago(flows.reindex(table.index), "mean", days=2)

    # --- Kept for error analysis only (NOT features) ------------------------
    actual_renewable = market[["wind_onshore_actual", "wind_offshore_actual", "solar_actual"]].sum(axis=1, min_count=3)
    table["wind_actual"] = market["wind_onshore_actual"] + market["wind_offshore_actual"]
    table["solar_actual"] = market["solar_actual"]
    table["residual_load_actual"] = market["load_actual"] - actual_renewable

    return table


# ---------------------------------------------------------------------------
# Which columns may a model use?
# ---------------------------------------------------------------------------

CALENDAR = ["hour", "weekday", "is_weekend", "is_holiday", "season_sin", "season_cos"]
PRICE_LAGS = [
    "price_lag_1d", "price_lag_2d", "price_lag_7d",
    "price_d1_mean", "price_d1_min", "price_d1_max", "price_7d_mean",
]


def _fundamentals(source: str) -> list[str]:
    return [
        "load_fc",
        f"wind_onshore_{source}", f"wind_offshore_{source}", f"solar_{source}",
        f"residual_load_{source}", f"renewable_share_{source}",
        f"residual_delta_1d_{source}", f"residual_delta_7d_{source}",
        f"residual_day_mean_{source}", f"residual_day_max_{source}", f"residual_day_min_{source}",
    ]


WEATHER_INDICES = ["wx_wind_onshore_cf", "wx_wind_offshore_cf", "wx_ghi", "wx_temp"]
FUEL_MIX = ["gas_gen_d2", "coal_gen_d2", "net_export_d2"]

FEATURE_SETS = {
    # Everything provably available before gate closure. The headline results use this.
    "strict": CALENDAR + PRICE_LAGS + _fundamentals("wx") + WEATHER_INDICES,
    # Same, but wind and solar taken from the TSO forecasts (possible look-ahead).
    "tso": CALENDAR + PRICE_LAGS + _fundamentals("tso") + ["wx_temp"],
    # Strict plus lagged fuel mix and trade from ENTSO-E.
    "strict_fuel": CALENDAR + PRICE_LAGS + _fundamentals("wx") + WEATHER_INDICES + FUEL_MIX,
    # No fundamentals at all: how far do calendar and price history get us?
    "price_only": CALENDAR + PRICE_LAGS,
}
