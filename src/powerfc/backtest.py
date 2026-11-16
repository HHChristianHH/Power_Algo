"""Walk-forward backtest: replay history one step at a time.

Why not a random train/test split?
----------------------------------
A random split would train on Wednesday and Friday and "predict" Thursday.
Prices on neighbouring days are so similar that the model would look
brilliant while having learned nothing useful. In real life you only ever
know the past. Walk-forward validation imitates that:

    train on everything before week 1  -> forecast week 1
    train on everything before week 2  -> forecast week 2
    ...

Every forecast is made by a model that has never seen that day, or any day
after it.
"""

from typing import Callable

import numpy as np
import pandas as pd

TARGET = "price"


def walk_forward(
    table: pd.DataFrame,
    make_model: Callable,
    first_test_day: str,
    refit_every_days: int = 7,
    train_window_days: int | None = None,
    collect_importance: bool = False,
):
    """Forecast every day from `first_test_day` on, re-training as time passes.

    make_model         function returning a fresh, untrained model
    refit_every_days   how often the model is re-trained. Between re-trainings
                       it still gets fresh inputs every day (yesterday's
                       prices, tomorrow's weather forecast); only its
                       learned parameters are up to a week old.
    train_window_days  None = use all history ("expanding window");
                       a number = only the most recent N days ("rolling").

    Returns a Series of forecasts (and the average feature importance if asked).
    """
    days = table.loc[table[TARGET].notna(), "delivery_day"]
    test_days = pd.DatetimeIndex(sorted(days[days >= pd.Timestamp(first_test_day)].unique()))

    forecasts, importances = [], []
    for block_start in test_days[::refit_every_days]:
        block_end = block_start + pd.Timedelta(days=refit_every_days)

        # Training data: delivery days strictly BEFORE the first day we forecast.
        # The latest of them (block_start - 1) cleared the day before gate
        # closure, so its prices were public when we would have placed our bid.
        is_train = table["delivery_day"] < block_start
        if train_window_days is not None:
            is_train &= table["delivery_day"] >= block_start - pd.Timedelta(days=train_window_days)
        is_test = (table["delivery_day"] >= block_start) & (table["delivery_day"] < block_end)

        model = make_model().fit(table[is_train])
        test = table[is_test]
        forecasts.append(pd.Series(model.predict(test), index=test.index))
        if collect_importance:
            importances.append(model.importance())

    forecast = pd.concat(forecasts)
    if collect_importance:
        return forecast, pd.concat(importances, axis=1).mean(axis=1)
    return forecast


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    """Mean absolute error: the average miss in EUR/MWh. Easy to interpret."""
    return float((actual - forecast).abs().mean())


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    """Root mean squared error: like MAE, but big misses count much more.

    A model with a good MAE and a poor RMSE is fine on normal days and badly
    wrong on spike days - worth knowing if you carry risk.
    """
    return float(np.sqrt(((actual - forecast) ** 2).mean()))


def score(predictions: pd.DataFrame, models: list[str], baseline: str) -> pd.DataFrame:
    """MAE and RMSE per model on the hours where EVERY model has a forecast.

    `rMAE` is MAE divided by the baseline's MAE: below 1 means better than
    the baseline. It makes results comparable across calm and volatile years.
    """
    complete = predictions.dropna(subset=models + [TARGET])
    rows = {
        name: {"MAE": mae(complete[TARGET], complete[name]), "RMSE": rmse(complete[TARGET], complete[name])}
        for name in models
    }
    result = pd.DataFrame(rows).T
    result["rMAE"] = result["MAE"] / result.loc[baseline, "MAE"]
    result.attrs["hours"] = len(complete)
    return result
