"""Step 3 - check that the downloaded data is complete and makes sense.

Run:  .venv/bin/python scripts/03_validate_data.py

Never model on data you have not looked at. Each check below answers one
question a careful analyst would ask before trusting the numbers.
"""

import _bootstrap  # noqa: F401  (makes `powerfc` importable, see scripts/_bootstrap.py)
import pandas as pd

from powerfc import store
from powerfc.config import MARKET_TZ, QUARTER_HOUR_START
from powerfc.timeutils import delivery_day, expected_periods


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_market(market: pd.DataFrame) -> None:
    section("1. Coverage: what do we have, from when to when, how much is missing?")
    summary = pd.DataFrame(
        {
            "first": market.apply(lambda c: c.first_valid_index()),
            "last": market.apply(lambda c: c.last_valid_index()),
            "missing_inside": market.apply(
                # gaps BETWEEN first and last value (not the unpublished tail)
                lambda c: c.loc[c.first_valid_index() : c.last_valid_index()].isna().sum()
            ),
            "min": market.min().round(1),
            "mean": market.mean().round(1),
            "max": market.max().round(1),
        }
    )
    print(summary.to_string())

    section("2. Time grid: is every quarter-hour present exactly once?")
    print("duplicated timestamps:", int(market.index.duplicated().sum()))
    full_grid = pd.date_range(market.index.min(), market.index.max(), freq="15min")
    print("timestamps missing from the 15-min grid:", len(full_grid.difference(market.index)))

    section("3. Clock changes: 92 / 96 / 100 quarter-hours per local day?")
    price = market["price"].dropna()
    counts = pd.Series(1, index=delivery_day(price.index)).groupby(level=0).sum()
    counts = counts.iloc[1:-1]  # first and last day are cut by the download window
    expected = pd.Series({day: expected_periods(day, minutes=15) for day in counts.index})
    wrong = counts[counts != expected]
    print("days by number of price periods:", counts.value_counts().sort_index().to_dict())
    print("days where the count differs from the calendar:", len(wrong))
    if len(wrong):
        print(wrong.head(10).to_string())

    section("4. The switch from hourly to 15-minute prices")
    # Within each hour: do the four quarter-hour prices differ?
    spread_in_hour = price.groupby(price.index.floor("h")).agg(lambda s: s.max() - s.min())
    switch_utc = QUARTER_HOUR_START.tz_convert("UTC")
    before, after = spread_in_hour[spread_in_hour.index < switch_utc], spread_in_hour[spread_in_hour.index >= switch_utc]
    print(f"before {QUARTER_HOUR_START:%Y-%m-%d}: hours whose quarters differ = {(before > 0).mean():.1%}")
    print(f"from   {QUARTER_HOUR_START:%Y-%m-%d}: hours whose quarters differ = {(after > 0).mean():.1%}")
    print(f"        average high-low spread inside an hour since then = {after.mean():.1f} EUR/MWh")

    section("5. Price behaviour by year (hourly averages)")
    hourly = price.resample("h").mean()
    by_year = hourly.groupby(hourly.index.tz_convert(MARKET_TZ).year)
    print(
        pd.DataFrame(
            {
                "hours": by_year.size(),
                "mean": by_year.mean().round(1),
                "min": by_year.min().round(1),
                "max": by_year.max().round(1),
                "negative_hours": by_year.apply(lambda s: int((s < 0).sum())),
                "hours_above_200": by_year.apply(lambda s: int((s > 200).sum())),
            }
        ).to_string()
    )

    section("6. Sanity: do the TSO forecasts track the actuals?")
    for name in ["load", "wind_onshore", "wind_offshore", "solar"]:
        pair = market[[f"{name}_fc", f"{name}_actual"]].dropna()
        error = pair[f"{name}_fc"] - pair[f"{name}_actual"]
        print(
            f"{name:14s} mean actual {pair[f'{name}_actual'].mean() / 1000:5.1f} GW | "
            f"forecast MAE {error.abs().mean() / 1000:4.2f} GW | bias {error.mean() / 1000:+5.2f} GW | "
            f"correlation {pair.corr().iloc[0, 1]:.3f}"
        )


def check_weather(weather: pd.DataFrame) -> None:
    section("7. Weather forecasts: where does each archived lead time begin?")
    times = weather.index.get_level_values("time_utc")
    print(f"rows {len(weather):,} | locations {weather.index.get_level_values('location').nunique()} | "
          f"{times.min():%Y-%m-%d} -> {times.max():%Y-%m-%d}")
    national = weather.groupby(level="time_utc").mean()  # simple average across locations
    summary = pd.DataFrame(
        {
            "first": national.apply(lambda c: c.first_valid_index()),
            "last": national.apply(lambda c: c.last_valid_index()),
            "missing_inside": national.apply(
                lambda c: c.loc[c.first_valid_index() : c.last_valid_index()].isna().sum()
            ),
            "min": national.min().round(1),
            "mean": national.mean().round(1),
            "max": national.max().round(1),
        }
    )
    print(summary.to_string())

    section("8. Do the weather forecasts explain the TSO renewable forecasts?")
    # If our weather data is wired up correctly, forecast wind speed should
    # correlate strongly with forecast wind power, and irradiance with solar.
    market = store.load("market_15min").resample("h").mean()
    joined = national.join(market, how="inner")
    print(f"wind speed (48h-ahead) vs TSO onshore wind forecast: correlation "
          f"{joined['wind100_d2'].corr(joined['wind_onshore_fc']):.3f}")
    # Radiation is stamped at the END of the hour it describes, the market data
    # at the START, hence the shift by one hour before comparing.
    print(f"irradiance (48h-ahead) vs TSO solar forecast:        correlation "
          f"{joined['ghi_d2'].shift(-1).corr(joined['solar_fc']):.3f}")
    # Hour by hour, load follows the working day far more than the thermometer,
    # so compare DAILY averages: colder days should show higher demand (negative sign).
    daily = joined[["temp_d2", "load_fc"]].resample("D").mean()
    print(f"daily temperature vs daily TSO load forecast:        correlation "
          f"{daily['temp_d2'].corr(daily['load_fc']):.3f}")


def main() -> None:
    check_market(store.load("market_15min"))
    check_weather(store.load("weather_forecast_hourly"))


if __name__ == "__main__":
    main()
