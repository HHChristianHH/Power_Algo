"""A simple trading use of the forecast: scheduling a battery in the day-ahead auction.

Why a battery and not "buy if the forecast is above the market price"?
----------------------------------------------------------------------
To trade a directional view you need a price to trade against (the OTC or
futures price for tomorrow), and there is no free source for that. A battery
needs no such reference. Its question is purely about the SHAPE of tomorrow's
prices: which hours will be cheapest (charge) and which dearest (discharge)?
That decision must be submitted before the auction, using a forecast, and is
then settled at the real auction prices. So the profit measures exactly what
a forecast is for.

Assumptions (all of them flatter reality a little; say so in an interview)
* price taker: our 1 MW does not move the auction price;
* every order is filled (we bid volumes without a price limit);
* no grid fees, no battery wear, no intraday or balancing revenue;
* the battery starts every day empty.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

POWER_MW = 1.0        # maximum charge or discharge rate
CAPACITY_MWH = 2.0    # energy the battery can hold ("2-hour battery")
EFFICIENCY = 0.90     # of every MWh bought, 0.9 MWh can be sold again


def best_schedule(prices: np.ndarray) -> np.ndarray:
    """Most profitable charge/discharge plan for one day of hourly prices.

    Returns the net position per hour in MW: positive = selling (discharging),
    negative = buying (charging).

    This is a "linear programme": we ask a solver for the numbers that
    maximise profit subject to physical limits.

    Unknowns     charge[h] and discharge[h] for every hour h (in MW)
    Maximise     sum over h of  price[h] * (discharge[h] - charge[h])
    Subject to   0 <= charge[h], discharge[h] <= POWER_MW
                 charge[h] + discharge[h] <= POWER_MW       (not both at once)
                 0 <= energy stored after every hour <= CAPACITY_MWH
    where        energy stored after hour h
                   = sum up to h of (EFFICIENCY * charge - discharge)
    """
    n = len(prices)
    # linprog MINIMISES, so we minimise the negative profit.
    # Variable order: [charge_0 ... charge_n-1, discharge_0 ... discharge_n-1]
    cost = np.concatenate([prices, -prices])

    running_sum = np.tril(np.ones((n, n)))                 # row h adds up hours 0..h
    stored = np.hstack([EFFICIENCY * running_sum, -running_sum])
    not_both = np.hstack([np.eye(n), np.eye(n)])

    constraints = np.vstack([stored, -stored, not_both])   # stored <= cap, -stored <= 0, sum <= power
    limits = np.concatenate([np.full(n, CAPACITY_MWH), np.zeros(n), np.full(n, POWER_MW)])

    solution = linprog(cost, A_ub=constraints, b_ub=limits, bounds=(0, POWER_MW), method="highs")
    charge, discharge = solution.x[:n], solution.x[n:]
    return discharge - charge


def simulate(predictions: pd.DataFrame, forecast_columns: list[str]) -> pd.DataFrame:
    """Daily profit (EUR) of running the battery on each forecast.

    For every day: plan with the FORECAST prices, get paid the ACTUAL prices.
    The column "perfect_foresight" plans with the actual prices - an upper
    bound nobody can reach, useful as a yardstick.
    """
    rows = {}
    for day, hours in predictions.groupby("delivery_day"):
        actual = hours["price"].to_numpy()
        profit = {"perfect_foresight": float(best_schedule(actual) @ actual)}
        for column in forecast_columns:
            position = best_schedule(hours[column].to_numpy())
            profit[column] = float(position @ actual)   # settle at real prices
        rows[day] = profit
    return pd.DataFrame(rows).T


def direction_hit_rate(predictions: pd.DataFrame, forecast_column: str) -> float:
    """How often the forecast gets the direction of the daily average price right.

    "Will tomorrow's average price be above or below today's?" This is a
    statistic, not a profit: you cannot buy tomorrow's power at today's price.
    """
    daily = predictions.groupby("delivery_day")[["price", forecast_column]].mean()
    yesterday = daily["price"].shift(1)
    predicted_up = daily[forecast_column] > yesterday
    actually_up = daily["price"] > yesterday
    valid = yesterday.notna()
    return float((predicted_up == actually_up)[valid].mean())
