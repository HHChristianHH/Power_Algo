# Forecasting German day-ahead power prices: one-page summary

**What it is.** A Python model that forecasts the 24 hourly day-ahead prices of the
German–Luxembourg bidding zone for the next day, using only free data and only
information available before the auction closes at 12:00 on the previous day.

**How it was tested.** Walk-forward backtest over 777 delivery days
(1 Oct 2024 to 16 Nov 2026, 18,629 hours). The model is re-trained every week on
past data only and never sees the day it forecasts.

## Results

| Model | MAE (EUR/MWh) | RMSE | vs naive |
|---|---|---|---|
| Same hour yesterday (naive) | 30.3 | 48.2 | 100% |
| Same hour last week | 39.2 | 62.5 | 129% |
| LightGBM, calendar and price history only | 27.9 | 43.3 | 92% |
| Linear model (ridge, one per hour) | 18.8 | 30.6 | 62% |
| **LightGBM, with fundamentals** | **17.7** | **29.2** | **58%** |
| Average of linear and LightGBM | 17.1 | 28.0 | 56% |
| *LightGBM with TSO wind/solar forecasts (possible look-ahead)* | *15.6* | *26.8* | *51%* |

Average price over the period: 104 EUR/MWh.

## Five things I learned

1. **Fundamentals carry the forecast, not price history.** Price lags alone improve
   on the naive forecast by only 8%. Adding load, wind and solar forecasts takes that
   to 42%. Residual load (demand minus wind minus solar) accounts for 56% of what the
   tree model uses.

2. **The honest version costs about 2 EUR/MWh.** The TSOs' wind and solar forecasts
   are better than mine (wind error 1.4 GW vs 2.6 GW), and using them would cut MAE
   from 17.7 to 15.6. But on the day I checked, they were not yet published half an
   hour before gate closure, so I built my own from archived ICON weather forecasts
   and report the stricter number.

3. **The model fails in the tails, and mostly in one direction.** Hours above
   200 EUR/MWh are 6% of the sample but 18% of the total error, and the model
   under-forecasts them by 31 EUR on average. In low-wind, no-sun hours the error is
   61 EUR. Negative-price hours are forecast about as well as normal ones (18 vs 15).

4. **Bad weather forecasts are not the main problem.** When my residual-load
   forecast misses by over 6 GW, the price error is 20 EUR; when it misses by under
   1 GW, it is still 16 EUR. Most of the error comes from the steep end of the supply
   curve, where a small change in demand moves the price a lot and there are few past
   examples to learn from.

5. **It has no view on fuel costs.** From September 2026 the monthly average price
   rose from about 100 to 145, 169 and 247 EUR/MWh, and the model's error doubled to
   around 30 EUR. With no gas or carbon price as input (no free source exists), it can
   only pick up a new price level from lagged prices, a few days late. I have not
   investigated what caused the rise.

## Trading use: scheduling a battery

A 1 MW / 2 MWh battery is scheduled each day on the forecast and settled at actual
auction prices (price taker, 90% efficiency, no fees or wear).

| Schedule based on | Profit per MW per year | Share of perfect foresight |
|---|---|---|
| Perfect foresight (upper bound) | 104,500 EUR | 100% |
| Average of linear and LightGBM | 97,700 EUR | 94% |
| LightGBM forecast | 96,500 EUR | 92% |
| Same hour yesterday | 90,000 EUR | 86% |

The forecast is worth roughly 6,400 to 7,700 EUR per MW per year over the naive
schedule. The gap is small because a battery only needs the *ranking* of hours, and
the daily shape is stable; the naive forecast already gets it mostly right.

## 15-minute products

Since 1 October 2025 the auction clears quarter-hours. Within an hour, prices fall
through the morning solar ramp and rise through the evening one. Adding the average
pattern of the previous 28 days to the hourly forecast cuts the quarter-hour error
from 21.5 to 20.5 EUR/MWh.

## Limitations I would raise myself

- **No fuel, carbon or outage inputs.** ENTSO-E's outage feed only returns the latest
  revision of each message, so it cannot be used without look-ahead. Adding gas and
  coal generation from two days earlier improved MAE by only 0.1 EUR.
- **TSO load forecast timing is assumed, not proven.** Regulation requires it two
  hours before gate closure and I saw it online in time, but historical publication
  times are not recorded.
- **Model settings are defaults**, not tuned on the test period. One design choice
  (the wind power-curve exponent) was picked by comparing a few values on the full
  history.
- **Weather coverage is thin**: 14 points and one weather model.
- **The battery simulation ignores** market impact, fees, degradation, and the
  intraday and balancing markets.

**Next steps:** add gas and carbon prices; forecast a price distribution rather than
a point, since the errors sit in the tails; snapshot outage messages daily.
