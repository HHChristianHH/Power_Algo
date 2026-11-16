"""Time helpers: delivery days, gate closure and clock-change days.

Why this file exists
--------------------
Germany switches between CET (UTC+1) and CEST (UTC+2). On the last Sunday of
March the local day has 23 hours; on the last Sunday of October it has 25.
If you store local timestamps naively, one hour goes missing in spring and one
hour appears twice in autumn. Storing UTC avoids both problems, and these
helpers translate between UTC and the local "delivery day" the market uses.
"""

import datetime as dt

import pandas as pd

from powerfc.config import GATE_CLOSURE_HOUR, MARKET_TZ


def delivery_day(index_utc: pd.DatetimeIndex) -> pd.Index:
    """Local calendar date each UTC timestamp belongs to.

    Example: 2025-06-30 22:00 UTC is 2025-07-01 00:00 in Berlin (summer time),
    so it is the first hour of delivery day 1 July.
    """
    return pd.Index(index_utc.tz_convert(MARKET_TZ).date, name="delivery_day")


def gate_closure_utc(day: dt.date) -> pd.Timestamp:
    """Moment (in UTC) the day-ahead auction for delivery day `day` closes.

    It is 12:00 local time on the previous day. We build the local wall-clock
    time first and then convert, so the result is 11:00 UTC in winter and
    10:00 UTC in summer without us hard-coding either.
    """
    previous_day = day - dt.timedelta(days=1)
    local = pd.Timestamp(previous_day).tz_localize(MARKET_TZ) + pd.Timedelta(hours=GATE_CLOSURE_HOUR)
    return local.tz_convert("UTC")


def periods_per_day(index_utc: pd.DatetimeIndex) -> pd.Series:
    """Count how many timestamps fall in each local delivery day.

    Used as a data-quality check. For hourly data a normal day has 24 rows,
    the spring clock-change day 23 and the autumn one 25 (92/96/100 for
    quarter-hours). Any other number means missing or duplicated data.
    """
    days = delivery_day(index_utc)
    return pd.Series(1, index=days).groupby(level=0).sum()


def expected_periods(day: dt.date, minutes: int = 60) -> int:
    """Number of periods of length `minutes` the local day really has.

    Returns 24 on a normal day, 23 or 25 on clock-change days (92/96/100 for
    minutes=15).
    """
    start = pd.Timestamp(day).tz_localize(MARKET_TZ)
    end = pd.Timestamp(day + dt.timedelta(days=1)).tz_localize(MARKET_TZ)
    # Subtracting two tz-aware timestamps gives true elapsed time (23, 24 or 25 h).
    return int((end - start) / pd.Timedelta(minutes=minutes))
