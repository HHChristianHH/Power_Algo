"""Step 7 - error analysis and charts.

Run:  .venv/bin/python scripts/07_analyze_errors.py

Reads the backtest results and writes tables to reports/ and charts to
reports/figures/.
"""

import _bootstrap  # noqa: F401  (makes `powerfc` importable, see scripts/_bootstrap.py)
import pandas as pd

from powerfc import analysis, plots, store
from powerfc.config import ROOT

REPORTS = ROOT / "reports"
MODEL, BASELINE = "lightgbm", "naive_yesterday"


def main() -> None:
    predictions = store.load("backtest_predictions").dropna(subset=["price", MODEL])
    context = store.load("features_hourly")[
        ["wind_actual", "solar_actual", "residual_load_actual", "residual_load_wx"]
    ]
    data = predictions.join(context)
    pd.set_option("display.width", 200)

    print("=== Errors by market situation ===")
    regimes = analysis.errors_by_regime(data, MODEL, BASELINE)
    regimes.round(3).to_csv(REPORTS / "errors_by_regime.csv")
    shown = regimes.copy()
    shown[["share_of_hours", "share_of_total_error"]] = (shown[["share_of_hours", "share_of_total_error"]] * 100).round(1)
    print(shown.round(1).to_string())

    print("\n=== Price error vs size of our residual-load forecast miss ===")
    weather_miss = analysis.error_by_weather_miss(data, MODEL)
    print(weather_miss.round(1).to_string())
    miss = data["residual_load_wx"] - data["residual_load_actual"]
    print(f"correlation between price error and residual-load error: {(data[MODEL] - data['price']).corr(miss):.2f}")

    print("\n=== Ten worst days ===")
    print(analysis.worst_days(data, MODEL).round(1).to_string())

    print("\n=== Monthly ===")
    monthly = analysis.monthly_errors(data, [BASELINE, MODEL, "lightgbm_tso_lookahead"])
    monthly.round(2).to_csv(REPORTS / "monthly_errors.csv")
    print(monthly.round(1).to_string())

    print("\nCharts:")
    plots.forecast_vs_actual(
        data,
        {
            "A summer fortnight: solar pushes midday prices to zero (August 2025)": ("2025-08-11", "2025-08-24"),
            "A low-wind week with a price spike (December 2024)": ("2024-12-09", "2024-12-15"),
        },
    )
    plots.mae_by_hour(data, [BASELINE, "ridge", MODEL])
    plots.feature_importance(pd.read_csv(REPORTS / "feature_importance.csv", index_col=0)["gain"])
    plots.error_by_regime(regimes)
    plots.monthly_error(monthly, [BASELINE, MODEL])
    plots.error_vs_weather_miss(weather_miss)


if __name__ == "__main__":
    main()
