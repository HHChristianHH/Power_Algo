"""Step 6 - walk-forward backtest of all models.

Run:  .venv/bin/python scripts/06_run_backtest.py     (a few minutes)

Writes
  data/backtest_predictions.parquet   actual price + one forecast column per model
  reports/metrics.csv                 MAE / RMSE table
  reports/feature_importance.csv      average LightGBM importance across re-trainings
"""

import _bootstrap  # noqa: F401  (makes `powerfc` importable, see scripts/_bootstrap.py)
import pandas as pd

from powerfc import backtest, store
from powerfc.config import ROOT
from powerfc.features import FEATURE_SETS
from powerfc.models import LightGBMModel, NaiveModel, RidgePerHour

# The strict features start in April 2024, so this leaves ~5.5 months of
# training data before the first forecast.
FIRST_TEST_DAY = "2024-10-01"
REPORTS = ROOT / "reports"


def main() -> None:
    table = store.load("features_hourly")
    # Only rows with complete "strict" inputs. LightGBM could cope with gaps,
    # but all models should be trained and judged on the same hours.
    table = table.dropna(subset=["price_lag_7d", "residual_load_wx", "residual_delta_7d_wx"])

    # name -> (function creating the model, collect feature importance?)
    experiments = {
        "naive_yesterday": (lambda: NaiveModel(days=1), False),
        "naive_last_week": (lambda: NaiveModel(days=7), False),
        "ridge": (lambda: RidgePerHour(FEATURE_SETS["strict"]), False),
        "lightgbm": (lambda: LightGBMModel(FEATURE_SETS["strict"]), True),
        # --- comparison runs, see README ---
        "lightgbm_price_only": (lambda: LightGBMModel(FEATURE_SETS["price_only"]), False),
        "lightgbm_tso_lookahead": (lambda: LightGBMModel(FEATURE_SETS["tso"]), False),
    }
    # Optional: add lagged fuel mix / trade if the ENTSO-E download was run.
    fuel_features = [f for f in FEATURE_SETS["strict_fuel"] if f in table.columns]
    if len(fuel_features) > len(FEATURE_SETS["strict"]):
        experiments["lightgbm_fuel_mix"] = (lambda: LightGBMModel(fuel_features), False)

    predictions = table[["delivery_day", "hour", "price"]].copy()
    for name, (make_model, wants_importance) in experiments.items():
        print(f"Backtesting {name} ...", flush=True)
        result = backtest.walk_forward(table, make_model, FIRST_TEST_DAY, collect_importance=wants_importance)
        if wants_importance:
            result, importance = result
            REPORTS.mkdir(exist_ok=True)
            importance.sort_values(ascending=False).rename("gain").to_csv(REPORTS / "feature_importance.csv")
        predictions[name] = result

    # A simple ensemble: the average of the linear and the tree model. Their
    # errors differ, so averaging often beats both.
    predictions["ensemble"] = predictions[["ridge", "lightgbm"]].mean(axis=1)

    predictions = predictions[predictions["delivery_day"] >= FIRST_TEST_DAY]
    store.save(predictions, "backtest_predictions")

    models = [c for c in predictions.columns if c not in ("delivery_day", "hour", "price")]
    metrics = backtest.score(predictions, models, baseline="naive_yesterday")
    REPORTS.mkdir(exist_ok=True)
    metrics.round(3).to_csv(REPORTS / "metrics.csv")

    complete = predictions.dropna(subset=models + ["price"])  # hours that have an auction result
    days = complete["delivery_day"]
    print(f"\nOut-of-sample period {days.min():%Y-%m-%d} -> {days.max():%Y-%m-%d}, "
          f"{metrics.attrs['hours']:,} hours. Errors in EUR/MWh.")
    print(metrics.round(2).to_string())

    print("\nMAE by calendar year:")
    by_year = complete.groupby(complete["delivery_day"].dt.year).apply(
        lambda g: pd.Series({m: backtest.mae(g["price"], g[m]) for m in models})
    )
    print(by_year.round(2).T.to_string())


if __name__ == "__main__":
    main()
