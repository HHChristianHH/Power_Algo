"""Error analysis: when does the model fail, and why?

An average error hides the hours that matter. A trader loses money in the
tails: the evening the price triples, or the Sunday it goes negative. So we
slice the errors by market situation ("regime") and look at each slice.

Note: regimes are defined from what ACTUALLY happened (the realised price and
wind). That is fine here because we are diagnosing past forecasts, not making
new ones.
"""

import pandas as pd

SPIKE_THRESHOLD = 200.0   # EUR/MWh; about the top 6 % of hours in the test period
LOW_WIND_MW = 3000.0      # total wind output below 3 GW; about the lowest 9 % of hours


def regime_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    """True/False per hour for each market situation (they may overlap)."""
    daytime_or_evening = data["hour"].between(7, 20)
    return {
        "Normal (0 to 200 EUR)": (data["price"] >= 0) & (data["price"] <= SPIKE_THRESHOLD),
        "Negative price": data["price"] < 0,
        f"Spike (above {SPIKE_THRESHOLD:.0f} EUR)": data["price"] > SPIKE_THRESHOLD,
        "Low wind (below 3 GW)": data["wind_actual"] < LOW_WIND_MW,
        # "Dunkelflaute": little wind AND little sun while demand is high.
        "Low wind and no sun, 07-20h": (data["wind_actual"] < 5000) & (data["solar_actual"] < 1000) & daytime_or_evening,
    }


def errors_by_regime(data: pd.DataFrame, model: str, baseline: str) -> pd.DataFrame:
    """Per regime: how often it occurs, MAE, bias, and its share of the total error.

    bias = average of (forecast - actual). Negative means we forecast too low.
    """
    error = data[model] - data["price"]
    rows = {}
    for name, mask in regime_masks(data).items():
        rows[name] = {
            "share_of_hours": mask.mean(),
            "mae_baseline": (data.loc[mask, baseline] - data.loc[mask, "price"]).abs().mean(),
            "mae_model": error[mask].abs().mean(),
            "bias_model": error[mask].mean(),
            "share_of_total_error": error[mask].abs().sum() / error.abs().sum(),
        }
    return pd.DataFrame(rows).T


def error_by_weather_miss(data: pd.DataFrame, model: str) -> pd.DataFrame:
    """Price error grouped by how wrong our residual-load forecast turned out.

    If bad weather forecasts were the main problem, price errors would grow
    steeply with the size of the residual-load miss.
    """
    miss_gw = (data["residual_load_wx"] - data["residual_load_actual"]).abs() / 1000
    buckets = pd.cut(miss_gw, [0, 1, 2, 4, 6, 100], labels=["under 1", "1 to 2", "2 to 4", "4 to 6", "over 6"])
    error = (data[model] - data["price"]).abs()
    return pd.DataFrame({"hours": error.groupby(buckets, observed=True).size(), "mae": error.groupby(buckets, observed=True).mean()})


def monthly_errors(data: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """MAE per calendar month and model, plus the average price level."""
    month = data["delivery_day"].dt.to_period("M")
    table = pd.DataFrame({m: (data[m] - data["price"]).abs().groupby(month).mean() for m in models})
    table["avg_price"] = data["price"].groupby(month).mean()
    return table


def worst_days(data: pd.DataFrame, model: str, n: int = 10) -> pd.DataFrame:
    """The days with the largest average error, with context to explain them."""
    error = (data[model] - data["price"]).abs()
    daily = pd.DataFrame(
        {
            "mae": error.groupby(data["delivery_day"]).mean(),
            "price_min": data["price"].groupby(data["delivery_day"]).min(),
            "price_mean": data["price"].groupby(data["delivery_day"]).mean(),
            "price_max": data["price"].groupby(data["delivery_day"]).max(),
            "wind_gw": data["wind_actual"].groupby(data["delivery_day"]).mean() / 1000,
        }
    )
    return daily.sort_values("mae", ascending=False).head(n)
