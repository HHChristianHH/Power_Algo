"""Evidence for the look-ahead rule: are tomorrow's forecasts published yet?

Run:  .venv/bin/python scripts/00_check_forecast_publication.py

The model may only use data that existed before the auction closes at 12:00
local time. SMARD does not tell us WHEN a forecast was published, so the only
way to know is to look. Run this shortly before noon (e.g. 11:45) on a few
days: each run appends one line per series to data/publication_log.csv saying
whether tomorrow's values were already there.

A series that is regularly missing at 11:45 must not be used as a feature for
tomorrow's prices - no matter how much it would improve the backtest.
"""

import _bootstrap  # noqa: F401  (makes `powerfc` importable, see scripts/_bootstrap.py)
import pandas as pd

from powerfc.config import DATA_DIR, MARKET_TZ
from powerfc.data import smard
from powerfc.timeutils import gate_closure_utc

FORECAST_SERIES = ["load_fc", "wind_onshore_fc", "wind_offshore_fc", "solar_fc"]


def main() -> None:
    now = pd.Timestamp.now(tz=MARKET_TZ)
    tomorrow = (now + pd.Timedelta(days=1)).date()
    tomorrow_start = pd.Timestamp(tomorrow).tz_localize(MARKET_TZ).tz_convert("UTC")
    closes = gate_closure_utc(tomorrow).tz_convert(MARKET_TZ)

    print(f"Now: {now:%Y-%m-%d %H:%M} local. Auction for {tomorrow} closes {closes:%H:%M} local.")
    rows = []
    for name in FORECAST_SERIES:
        series = smard.download_series(smard.SERIES[name], start=tomorrow_start - pd.Timedelta(days=1))
        available = bool((series.index >= tomorrow_start).any())
        print(f"  {name:18s} tomorrow's forecast published: {'YES' if available else 'no'}")
        rows.append(
            {
                "checked_at_local": now.strftime("%Y-%m-%d %H:%M"),
                "before_gate_closure": now < closes,
                "delivery_day": tomorrow,
                "series": name,
                "published": available,
            }
        )

    log_path = DATA_DIR / "publication_log.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    print(f"Appended to {log_path}")


if __name__ == "__main__":
    main()
