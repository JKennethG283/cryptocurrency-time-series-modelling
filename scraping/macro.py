from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


START_DATE = "2020-01-01"
OUTPUT_DIR = Path(__file__).resolve().parent / "macro"

# FRED series ids:
# - FEDFUNDS: Effective Federal Funds Rate (monthly)
# - CPIAUCSL: CPI for All Urban Consumers (monthly)
# - UNRATE: Unemployment Rate (monthly)
# - DGS10: 10-Year Treasury Constant Maturity Rate (daily, weekdays)
# - DTWEXBGS: Trade Weighted U.S. Dollar Index: Broad (daily, weekdays)
# - VIXCLS: CBOE Volatility Index: VIX (daily, weekdays)
SERIES: Dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "unemployment_rate": "UNRATE",
    "us10y_treasury_yield": "DGS10",
    "usd_index_broad": "DTWEXBGS",
    "vix": "VIXCLS",
}


def fetch_fred_series(series_id: str) -> pd.Series:
    """Fetch one FRED series via public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    date_col = df.columns[0]
    value_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    series = df.set_index("date")["value"].sort_index()
    return series


def build_daily_and_hourly(series: pd.Series, start_date: str) -> tuple[pd.Series, pd.Series]:
    """Resample to daily and hourly with forward-fill."""
    today = pd.Timestamp.today().normalize()
    start = pd.Timestamp(start_date)

    daily_index = pd.date_range(start=start, end=today, freq="D")
    daily = series.reindex(daily_index).ffill()

    hourly_index = pd.date_range(
        start=start, end=today + pd.Timedelta(hours=23), freq="h"
    )
    hourly = daily.reindex(hourly_index, method="ffill")

    return daily, hourly


def save_series_csv(series: pd.Series, path: Path, value_name: str) -> None:
    out_df = series.rename(value_name).rename_axis("timestamp").reset_index()
    out_df.to_csv(path, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, fred_id in SERIES.items():
        raw = fetch_fred_series(fred_id)
        raw = raw[raw.index >= pd.Timestamp(START_DATE)]

        daily, hourly = build_daily_and_hourly(raw, START_DATE)

        daily_path = OUTPUT_DIR / f"{name}_daily.csv"
        hourly_path = OUTPUT_DIR / f"{name}_hourly.csv"

        save_series_csv(daily, daily_path, "value")
        save_series_csv(hourly, hourly_path, "value")

        print(f"Saved: {daily_path}")
        print(f"Saved: {hourly_path}")


if __name__ == "__main__":
    main()
