"""From hourly forecasts to 15-minute forecasts.

Since delivery day 1 October 2025 the auction clears 96 quarter-hour prices
per day instead of 24 hourly ones. Our main model forecasts the hourly
average, because that target exists for the whole history. This module adds
the missing piece: how the four quarters sit around their hour's average.

The pattern is strong and regular. When solar output is ramping UP (morning),
each quarter is cheaper than the one before; when it ramps DOWN (evening),
each is dearer. So the first quarter of 09:00 is well above the hourly
average and the last well below, and the reverse at 17:00.

Method: for each of the 96 slots in a day, take the average deviation from
the hourly price over the previous 28 days, and add it to the hourly forecast.
All of those days' prices are public at gate closure, so there is no look-ahead.
"""

import pandas as pd

from powerfc.config import MARKET_TZ, QUARTER_HOUR_START

PROFILE_DAYS = 28


def slot_keys(index_utc: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.Index]:
    """Delivery day and slot number 0-95 (hour * 4 + quarter) in local time."""
    local = index_utc.tz_convert(MARKET_TZ).tz_localize(None)
    return local.normalize(), pd.Index(local.hour * 4 + local.minute // 15)


def quarter_hour_forecast(price_15min: pd.Series, hourly_forecast: pd.Series) -> pd.DataFrame:
    """Combine an hourly forecast with the recent intra-hour price pattern.

    Returns one row per quarter-hour with the actual price and two forecasts:
    `flat` (hourly forecast repeated four times) and `shaped` (flat + pattern).
    """
    price = price_15min.dropna()
    price = price[price.index >= QUARTER_HOUR_START.tz_convert("UTC")]

    # Deviation of each quarter from the average of its hour.
    hourly_actual = price.groupby(price.index.floor("h")).transform("mean")
    deviation = price - hourly_actual

    # Table: one row per day, 96 slot columns. Then, for each day, the mean of
    # the PREVIOUS 28 days (shift(1) excludes the day itself).
    day, slot = slot_keys(price.index)
    by_day = deviation.groupby([day, slot]).mean().unstack()
    profile = by_day.rolling(PROFILE_DAYS, min_periods=PROFILE_DAYS).mean().shift(1)
    pattern = profile.stack().reindex(pd.MultiIndex.from_arrays([day, slot])).to_numpy()

    result = pd.DataFrame({"price": price, "hourly_actual": hourly_actual})
    # Repeat each hourly forecast for the four quarters of its hour.
    result["flat"] = hourly_forecast.reindex(price.index.floor("h")).to_numpy()
    result["shaped"] = result["flat"] + pattern
    # Benchmark: if the hourly forecast were perfect, how much error is pure shape?
    result["perfect_hour_flat"] = hourly_actual
    result["perfect_hour_shaped"] = hourly_actual + pattern
    return result.dropna()
