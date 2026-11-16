"""Charts. One function per figure; each saves a PNG into reports/figures/.

Design rules followed throughout (so the charts read as one family):
* colour identifies a series and is assigned in a fixed order, never by rank;
* text is always dark grey - only the marks (lines, bars) carry colour;
* thin lines and bars, faint gridlines, no chart borders;
* one y-axis per panel - two measures get two panels, never two scales.
"""

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from powerfc.config import MARKET_TZ, ROOT

FIGURES = ROOT / "reports" / "figures"

# Colour-blind-safe palette, used in this order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

# Fixed colour per model, the same in every chart.
MODEL_STYLE = {
    "naive_yesterday": ("Same hour yesterday", ORANGE),
    "ridge": ("Linear model", AQUA),
    "lightgbm": ("LightGBM", BLUE),
    "lightgbm_tso_lookahead": ("LightGBM with TSO forecasts", YELLOW),
}

FEATURE_LABELS = {
    "price_lag_1d": "Price, same hour yesterday",
    "price_lag_2d": "Price, same hour 2 days ago",
    "price_lag_7d": "Price, same hour last week",
    "price_d1_mean": "Yesterday's average price",
    "price_d1_min": "Yesterday's lowest price",
    "price_d1_max": "Yesterday's highest price",
    "price_7d_mean": "Average price, last 7 days",
    "load_fc": "Load forecast",
    "wind_onshore_wx": "Onshore wind forecast",
    "wind_offshore_wx": "Offshore wind forecast",
    "solar_wx": "Solar forecast",
    "residual_load_wx": "Residual load forecast",
    "renewable_share_wx": "Renewable share of load",
    "residual_delta_1d_wx": "Residual load change vs yesterday",
    "residual_delta_7d_wx": "Residual load change vs last week",
    "residual_day_mean_wx": "Day's average residual load",
    "residual_day_max_wx": "Day's peak residual load",
    "residual_day_min_wx": "Day's lowest residual load",
    "wx_wind_onshore_cf": "Onshore wind index (weather)",
    "wx_wind_offshore_cf": "Offshore wind index (weather)",
    "wx_ghi": "Solar irradiance (weather)",
    "wx_temp": "Temperature",
    "hour": "Hour of day",
    "weekday": "Day of week",
    "is_weekend": "Weekend",
    "is_holiday": "Public holiday",
    "season_sin": "Season (sine)",
    "season_cos": "Season (cosine)",
}


def _style(ax, title: str, ylabel: str = "") -> None:
    """Apply the shared look to one panel."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIGURES / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved reports/figures/{name}")


def _legend(ax) -> None:
    """Legend on its own line between title and plot, so it never covers the data."""
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, ncol=4, handlelength=1.6,
              loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
    ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=11, color=INK, fontweight="bold", pad=30)


def forecast_vs_actual(data: pd.DataFrame, periods: dict[str, tuple[str, str]], model: str = "lightgbm") -> None:
    """Hourly forecast against the realised price, one panel per period."""
    fig, axes = plt.subplots(len(periods), 1, figsize=(10, 3.4 * len(periods)))
    for ax, (title, (start, end)) in zip(axes, periods.items()):
        window = data[(data["delivery_day"] >= start) & (data["delivery_day"] <= end)]
        local_time = window.index.tz_convert(MARKET_TZ)
        ax.plot(local_time, window["price"], color=INK, linewidth=1.4, label="Actual price")
        ax.plot(local_time, window[model], color=BLUE, linewidth=1.8, label="LightGBM forecast")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        _style(ax, title, "EUR/MWh")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=local_time.tz))
        mae = (window[model] - window["price"]).abs().mean()
        ax.text(0.995, 0.95, f"MAE {mae:.1f} EUR/MWh", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=MUTED)
    _legend(axes[0])
    fig.tight_layout()
    _save(fig, "forecast_vs_actual.png")


def mae_by_hour(data: pd.DataFrame, models: list[str]) -> None:
    """Average absolute error for each hour of the day."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for model in models:
        label, colour = MODEL_STYLE[model]
        by_hour = (data[model] - data["price"]).abs().groupby(data["hour"]).mean()
        ax.plot(by_hour.index, by_hour, color=colour, linewidth=2, label=label)
    _style(ax, "Forecast error by hour of day", "Mean absolute error, EUR/MWh")
    ax.set_xlabel("Delivery hour (local time)", fontsize=9, color=MUTED)
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(bottom=0)
    _legend(ax)
    _save(fig, "mae_by_hour.png")


def feature_importance(importance: pd.Series, top: int = 15) -> None:
    """Which inputs the tree model relied on most (share of total error reduction)."""
    share = (importance / importance.sum()).sort_values(ascending=False).head(top)[::-1] * 100
    labels = [FEATURE_LABELS.get(name, name) for name in share.index]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(labels, share, color=BLUE, height=0.55)
    for y, value in enumerate(share):
        ax.text(value + 0.4, y, f"{value:.1f}%", va="center", fontsize=9, color=MUTED)
    _style(ax, f"What the LightGBM model relies on (top {top} of {len(importance)} features)")
    ax.set_xlabel("Share of total gain, %, averaged over all re-trainings", fontsize=9, color=MUTED)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(axis="y", labelcolor=INK)
    _save(fig, "feature_importance.png")


def error_by_regime(table: pd.DataFrame) -> None:
    """MAE of the baseline and the model in each market situation."""
    table = table[::-1]  # first regime at the top
    positions = range(len(table))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    height = 0.32
    for offset, column, model in [(height / 2 + 0.02, "mae_baseline", "naive_yesterday"), (-height / 2 - 0.02, "mae_model", "lightgbm")]:
        label, colour = MODEL_STYLE[model]
        ax.barh([p + offset for p in positions], table[column], height=height, color=colour, label=label)
        for p, value in zip(positions, table[column]):
            ax.text(value + 0.8, p + offset, f"{value:.0f}", va="center", fontsize=9, color=MUTED)
    labels = [f"{name}\n{share:.0%} of hours" for name, share in zip(table.index, table["share_of_hours"])]
    ax.set_yticks(list(positions), labels)
    _style(ax, "Forecast error by market situation")
    ax.set_xlabel("Mean absolute error, EUR/MWh", fontsize=9, color=MUTED)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(axis="y", labelcolor=INK)
    _legend(ax)
    _save(fig, "error_by_regime.png")


def monthly_error(table: pd.DataFrame, models: list[str]) -> None:
    """MAE per month (top) and the price level (bottom) - two panels, two axes."""
    months = table.index.to_timestamp()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 2])
    for model in models:
        label, colour = MODEL_STYLE[model]
        top.plot(months, table[model], color=colour, linewidth=2, label=label)
    _style(top, "Forecast error by month", "Mean absolute error, EUR/MWh")
    top.set_ylim(bottom=0)
    _legend(top)
    bottom.bar(months, table["avg_price"], width=18, color=MUTED)
    _style(bottom, "Average day-ahead price by month", "EUR/MWh")
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.tight_layout()
    _save(fig, "monthly_error.png")


def error_vs_weather_miss(table: pd.DataFrame) -> None:
    """Does a bad weather forecast explain a bad price forecast?"""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(table.index.astype(str), table["mae"], width=0.45, color=BLUE)
    for x, (value, hours) in enumerate(zip(table["mae"], table["hours"])):
        ax.text(x, value + 0.4, f"{value:.1f}", ha="center", fontsize=9, color=INK)
        ax.text(x, 0.6, f"{hours:,} h", ha="center", fontsize=8, color=SURFACE)
    _style(ax, "Price error by size of our residual-load forecast miss", "Mean absolute error, EUR/MWh")
    ax.set_xlabel("Absolute miss in forecast residual load, GW", fontsize=9, color=MUTED)
    _save(fig, "error_vs_weather_miss.png")


def cumulative_pnl(daily_pnl: pd.DataFrame, labels: dict[str, tuple[str, str]]) -> None:
    """Running total of the simulated battery profit for each forecast."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    cumulative = daily_pnl.cumsum() / 1000
    for column, (label, colour) in labels.items():
        ax.plot(cumulative.index, cumulative[column], color=colour, linewidth=2, label=label)
        ax.annotate(f"{cumulative[column].iloc[-1]:.0f}k", (cumulative.index[-1], cumulative[column].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=MUTED)
    _style(ax, "Simulated battery profit, 1 MW / 2 MWh, day-ahead only", "Cumulative profit, thousand EUR")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    _legend(ax)
    _save(fig, "battery_pnl.png")
