"""Step 8 - turn the forecast into a trading decision and simulate the profit.

Run:  .venv/bin/python scripts/08_trading_signal.py

Schedules a 1 MW / 2 MWh battery in the day-ahead auction using each
forecast, settles at actual prices, and compares with perfect foresight.
"""

import _bootstrap  # noqa: F401  (makes `powerfc` importable, see scripts/_bootstrap.py)

from powerfc import plots, store, trading
from powerfc.config import ROOT

FORECASTS = ["naive_yesterday", "ridge", "lightgbm", "ensemble", "lightgbm_tso_lookahead"]


def main() -> None:
    predictions = store.load("backtest_predictions").dropna(subset=["price"] + FORECASTS)
    daily = trading.simulate(predictions, FORECASTS)
    daily.round(2).to_csv(ROOT / "reports" / "battery_daily_pnl.csv")

    years = len(daily) / 365.25
    summary = daily.sum().to_frame("total_eur")
    summary["eur_per_mw_per_year"] = summary["total_eur"] / trading.POWER_MW / years
    summary["capture_rate"] = summary["total_eur"] / summary.loc["perfect_foresight", "total_eur"]
    summary["losing_days"] = (daily < 0).sum()
    summary.round(3).to_csv(ROOT / "reports" / "battery_summary.csv")

    print(f"Battery {trading.POWER_MW:.0f} MW / {trading.CAPACITY_MWH:.0f} MWh, "
          f"{trading.EFFICIENCY:.0%} round-trip efficiency, {len(daily)} days ({years:.2f} years)")
    shown = summary.copy()
    shown["capture_rate"] = (shown["capture_rate"] * 100).round(1)
    print(shown.round(0).to_string())

    print("\nDirection of the daily average price vs the previous day (share of days called correctly):")
    for column in ["lightgbm", "ensemble"]:
        print(f"  {column:10s} {trading.direction_hit_rate(predictions, column):.1%}")

    plots.cumulative_pnl(
        daily,
        {
            "perfect_foresight": ("Perfect foresight (upper bound)", plots.MUTED),
            "lightgbm": ("LightGBM forecast", plots.BLUE),
            "naive_yesterday": ("Same hour yesterday", plots.ORANGE),
        },
    )


if __name__ == "__main__":
    main()
