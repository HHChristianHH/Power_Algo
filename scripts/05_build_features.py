"""Step 5 - build the hourly model table from the raw data.

Run:  .venv/bin/python scripts/05_build_features.py

Takes about a minute (it re-learns the weather-to-generation mapping for
every day in the history).
"""

import _bootstrap  # noqa: F401  (makes `powerfc` importable, see scripts/_bootstrap.py)

from powerfc import features, store

DATASET = "features_hourly"


def main() -> None:
    table = features.build_features()
    store.save(table, DATASET)

    strict = table.dropna(subset=features.FEATURE_SETS["strict"] + ["price"])
    print(f"Saved {len(table):,} rows x {table.shape[1]} columns to {store.path(DATASET)}")
    print(f"Rows usable by the strict model: {len(strict):,} "
          f"({strict['delivery_day'].min():%Y-%m-%d} -> {strict['delivery_day'].max():%Y-%m-%d})")

    # How good are our weather-based wind and solar forecasts?
    print("\nOur weather-based generation forecast vs the TSO forecast (MAE against metered output):")
    sample = table.dropna(subset=["wind_onshore_wx", "wind_offshore_wx", "solar_wx", "wind_actual", "solar_actual"])
    ours_wind = sample["wind_onshore_wx"] + sample["wind_offshore_wx"]
    tso_wind = sample["wind_onshore_tso"] + sample["wind_offshore_tso"]
    print(f"  wind : ours {(ours_wind - sample['wind_actual']).abs().mean() / 1000:.2f} GW | "
          f"TSO {(tso_wind - sample['wind_actual']).abs().mean() / 1000:.2f} GW")
    print(f"  solar: ours {(sample['solar_wx'] - sample['solar_actual']).abs().mean() / 1000:.2f} GW | "
          f"TSO {(sample['solar_tso'] - sample['solar_actual']).abs().mean() / 1000:.2f} GW")
    resid = sample.dropna(subset=["residual_load_wx", "residual_load_tso", "residual_load_actual"])
    print(f"  residual load: ours {(resid['residual_load_wx'] - resid['residual_load_actual']).abs().mean() / 1000:.2f} GW | "
          f"TSO {(resid['residual_load_tso'] - resid['residual_load_actual']).abs().mean() / 1000:.2f} GW")
    print(f"\nCorrelation of price with our residual load forecast: "
          f"{strict['price'].corr(strict['residual_load_wx']):.3f}")


if __name__ == "__main__":
    main()
