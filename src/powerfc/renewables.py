"""Turn weather forecasts into wind and solar generation forecasts (in MW).

Why we need this
----------------
The key price driver is residual load = demand - wind - solar. The TSOs
publish wind and solar forecasts, but we cannot prove they were online before
the auction closed (see README). So we build our own from weather forecasts,
in three steps:

1. `select_available_forecast`  pick, for every hour, the freshest weather
   forecast that already existed at gate closure;
2. `wind_capacity_factor`       convert wind speed into "share of full output"
   with a turbine power curve (output rises with the CUBE of wind speed);
3. `walk_forward_nowcast`       learn how many MW each weather point stands
   for, using only past data, and re-learn it every day.
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from powerfc.config import GATE_CLOSURE_HOUR, MARKET_TZ, WEATHER_PUBLICATION_DELAY
from powerfc.data.openmeteo import LOCATIONS

# --- Step 1: which forecast was available? ----------------------------------


def gate_closure_for(times_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Gate closure (UTC) of the auction in which each timestamp was traded.

    That is 12:00 local time on the day before the timestamp's delivery day.
    We do the arithmetic on local wall-clock time (time zone removed), so it
    stays correct on 23- and 25-hour days, then attach the time zone again.
    """
    local_wall_clock = times_utc.tz_convert(MARKET_TZ).tz_localize(None)
    delivery_midnight = local_wall_clock.normalize()
    closure = delivery_midnight - pd.Timedelta(days=1) + pd.Timedelta(hours=GATE_CLOSURE_HOUR)
    return closure.tz_localize(MARKET_TZ).tz_convert("UTC")


def select_available_forecast(weather: pd.DataFrame) -> pd.DataFrame:
    """Keep one value per variable: the freshest forecast known at gate closure.

    `weather` has columns like wind100_d1 (predicted 24h before the hour) and
    wind100_d2 (predicted 48h before). The 24h-ahead value for hour T was
    computed at T-24h and published a few hours later. It may be used only if
    that moment is before gate closure; otherwise we fall back to the 48h one.

    In practice: hours up to 08:00 local use the 24h forecast, later hours the
    48h one. A real trader would have a slightly fresher forecast for the
    afternoon, so this rule errs on the side of caution.
    """
    times = weather.index.get_level_values("time_utc")
    published_d1 = times - pd.Timedelta(hours=24) + WEATHER_PUBLICATION_DELAY
    d1_usable = np.asarray(published_d1 <= gate_closure_for(times))

    selected = pd.DataFrame(index=weather.index)
    for variable in ["wind100", "ghi", "temp"]:
        d1, d2 = weather[f"{variable}_d1"], weather[f"{variable}_d2"]
        # Use d1 where allowed AND present; in every other case use d2.
        # (We never fall back from d2 to an unusable d1: that would be look-ahead.)
        selected[variable] = d2.where(~(d1_usable & d1.notna()), d1)
    return selected


# --- Step 2: physics --------------------------------------------------------

CUT_IN, RATED, CUT_OUT = 3.0, 12.0, 25.0  # m/s, typical for a modern turbine
# A single turbine's output grows with the CUBE of wind speed. A whole region
# behaves more smoothly: its turbines see different winds at the same moment,
# so the regional curve is flatter. Comparing exponents 1.5, 2 and 3 against
# metered generation, 2 gave the lowest error (onshore MAE 2.3 GW vs 2.8 GW
# for the textbook cube), so we use a square law.
CURVE_EXPONENT = 2.0


def wind_capacity_factor(speed: pd.Series) -> pd.Series:
    """Fraction of full output a region's turbines deliver at a given wind speed.

    Below `CUT_IN` they stand still, above `RATED` they produce 100 %, in
    between output rises steeply (see CURVE_EXPONENT). Above `CUT_OUT` they
    shut down for safety.
    """
    cf = (speed**CURVE_EXPONENT - CUT_IN**CURVE_EXPONENT) / (RATED**CURVE_EXPONENT - CUT_IN**CURVE_EXPONENT)
    cf = cf.clip(lower=0.0, upper=1.0)
    # mask() replaces values only where the condition is True, so a missing
    # wind speed stays missing instead of silently becoming "zero output".
    return cf.mask(speed >= CUT_OUT, 0.0)


def weather_inputs(selected: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reshape weather into one table per technology: rows = hours, columns = locations."""
    wide = selected.unstack("location")  # columns become (variable, location)

    onshore_sites = LOCATIONS.index[LOCATIONS["w_wind_onshore"] > 0]
    offshore_sites = LOCATIONS.index[LOCATIONS["w_wind_offshore"] > 0]
    solar_sites = LOCATIONS.index[LOCATIONS["w_solar"] > 0]

    capacity_factor = wide["wind100"].apply(wind_capacity_factor)
    # Radiation is stamped at the END of the hour it describes; our market data
    # is stamped at the START. Shifting by one hour lines the two up.
    irradiance = wide["ghi"].shift(-1, freq="h")
    return {
        "wind_onshore": capacity_factor[onshore_sites],
        "wind_offshore": capacity_factor[offshore_sites],
        "solar": irradiance[solar_sites],
    }


def weighted_national(selected: pd.DataFrame) -> pd.DataFrame:
    """National weather indices: location values averaged with the capacity weights."""
    wide = selected.unstack("location")
    out = pd.DataFrame(index=wide.index)
    out["wx_wind_onshore_cf"] = wide["wind100"].apply(wind_capacity_factor).mul(LOCATIONS["w_wind_onshore"]).sum(axis=1, min_count=1)
    out["wx_wind_offshore_cf"] = wide["wind100"].apply(wind_capacity_factor).mul(LOCATIONS["w_wind_offshore"]).sum(axis=1, min_count=1)
    out["wx_ghi"] = wide["ghi"].shift(-1, freq="h").mul(LOCATIONS["w_solar"]).sum(axis=1, min_count=1)
    out["wx_temp"] = wide["temp"].mul(LOCATIONS["w_load"]).sum(axis=1, min_count=1)
    return out


# --- Step 3: learn MW per weather point, walking forward --------------------


def walk_forward_nowcast(
    inputs: pd.DataFrame,
    actual_mw: pd.Series,
    window_days: int = 60,
    information_lag_days: int = 2,
) -> pd.Series:
    """Forecast generation in MW from weather inputs, never using the future.

    For each delivery day D we fit
        actual MW ~ baseline + sum(coefficient_i * input_i)
    on the `window_days` before D and apply it to D's weather forecast.

    * Coefficients are forced to be >= 0 ("non-negative least squares"): each
      one is the installed capacity that location stands for, and capacity
      cannot be negative. This also keeps the fit stable.
    * The baseline (a constant) absorbs generation our few weather points do
      not explain. Without it the wind forecast ran about 1 GW too low.
    * The window slides, so newly built wind and solar farms are picked up
      automatically within a few weeks.
    * Training stops `information_lag_days` before D, because metered
      generation for the most recent day is not complete at gate closure.
    """
    inputs = inputs.assign(baseline=1.0)  # the constant term of the regression
    data = inputs.join(actual_mw.rename("actual"), how="left")
    local_dates = pd.Index(data.index.tz_convert(MARKET_TZ).date)
    forecast = pd.Series(np.nan, index=data.index)

    for day in pd.unique(local_dates):
        train_end = day - pd.Timedelta(days=information_lag_days)   # last day we may learn from
        train_start = train_end - pd.Timedelta(days=window_days)
        in_window = (local_dates > train_start) & (local_dates <= train_end)
        train = data[in_window].dropna()
        if len(train) < 24 * window_days * 0.8:   # not enough history yet
            continue

        coefficients, _ = nnls(train[inputs.columns].to_numpy(), train["actual"].to_numpy())

        today = data.loc[local_dates == day, inputs.columns]
        complete = today.notna().all(axis=1)
        forecast.loc[today.index[complete]] = today[complete].to_numpy() @ coefficients

    return forecast
