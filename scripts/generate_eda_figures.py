"""
Generate EDA figures for the report: price index, returns/volatility, volume.

Run from project root::

    python scripts/generate_eda_figures.py
    python scripts/generate_eda_figures.py --figures 1 3
    python scripts/generate_eda_figures.py --price-normalize each

Outputs (under ``results/figures/`` by default):

* ``eda_figure1_price_index_btc_eth_sol.png`` — daily close indexed to 100
* ``eda_figure2_returns_volatility_bitcoin.png`` — 2x2 daily + hourly ``log_ret`` / ``ret_std_20``
* ``eda_figure3_volume_btc_eth_sol.png`` — engineered ``log_volume`` and ``vol_spike_20`` (daily)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _ROOT / "data"
_DEFAULT_RESULTS = _ROOT / "results"


def _fig_dir(results_dir: Path) -> Path:
    d = results_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def figure1_price_index(
    data_dir: Path,
    results_dir: Path,
    normalize: str,
) -> Path:
    """
    Line chart of daily close prices for BTC, ETH, SOL.
    ``normalize`` is ``common`` (index 100 at first date where all three exist) or
    ``each`` (each asset indexed to 100 at its own first row).
    """
    paths = {
        "Bitcoin": data_dir / "ohlcv" / "bitcoin_ohlcv_daily.csv",
        "Ethereum": data_dir / "ohlcv" / "ethereum_ohlcv_daily.csv",
        "Solana": data_dir / "ohlcv" / "solana_ohlcv_daily.csv",
    }
    cols: list[pd.Series] = []
    labels: list[str] = []
    for label, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing OHLCV file: {p}")
        df = pd.read_csv(p, parse_dates=["timestamp_utc"])
        df = df.sort_values("timestamp_utc")
        s = df.set_index("timestamp_utc")["close"].astype(float)
        cols.append(s)
        labels.append(label)

    out = _fig_dir(results_dir) / "eda_figure1_price_index_btc_eth_sol.png"
    fig, ax = plt.subplots(figsize=(12, 5))

    if normalize == "common":
        combined = pd.concat(cols, axis=1, keys=labels, join="inner").sort_index()
        combined = combined.dropna(how="any")
        if combined.empty:
            raise ValueError("No overlapping dates across BTC, ETH, SOL daily OHLCV.")
        plot_df = (combined / combined.iloc[0]) * 100.0
        for col in plot_df.columns:
            ax.plot(plot_df.index, plot_df[col].values, label=str(col), lw=1.2)
    else:
        for lab, s in zip(labels, cols):
            indexed = (s / s.iloc[0]) * 100.0
            ax.plot(indexed.index, indexed.values, label=lab, lw=1.2)
    ax.set_title(
        "Daily close (indexed to 100"
        + (" at first common date" if normalize == "common" else " at each asset’s first date")
        + ")"
    )
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Index (= 100 at base)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def figure2_returns_volatility_bitcoin(data_dir: Path, results_dir: Path) -> Path:
    """Two rows (daily, hourly) × two columns (log_ret, ret_std_20) for Bitcoin."""
    daily_p = data_dir / "pre-processing" / "bitcoin_ohlcv_daily_ml.csv"
    hourly_p = data_dir / "pre-processing" / "bitcoin_ohlcv_hourly_ml.csv"
    for p in (daily_p, hourly_p):
        if not p.is_file():
            raise FileNotFoundError(f"Missing ML file: {p}")

    daily = pd.read_csv(daily_p, parse_dates=["timestamp"]).sort_values("timestamp")
    hourly = pd.read_csv(hourly_p, parse_dates=["timestamp"]).sort_values("timestamp")

    out = _fig_dir(results_dir) / "eda_figure2_returns_volatility_bitcoin.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")

    def _plot_row(ax0, ax1, df: pd.DataFrame, title_prefix: str) -> None:
        ax0.plot(df["timestamp"], df["log_ret"], color="#1f77b4", lw=0.6, alpha=0.9)
        ax0.set_ylabel("log_ret")
        ax0.set_title(f"{title_prefix}: log return")
        ax0.grid(True, alpha=0.25)
        ax1.plot(df["timestamp"], df["ret_std_20"], color="#d62728", lw=0.6, alpha=0.9)
        ax1.set_ylabel("ret_std_20")
        ax1.set_title(f"{title_prefix}: rolling vol (20)")
        ax1.grid(True, alpha=0.25)

    _plot_row(axes[0, 0], axes[0, 1], daily, "Bitcoin daily")
    _plot_row(axes[1, 0], axes[1, 1], hourly, "Bitcoin hourly")
    axes[1, 0].set_xlabel("Time (UTC)")
    axes[1, 1].set_xlabel("Time (UTC)")
    fig.suptitle("Returns and volatility — Bitcoin (ML features)", y=1.02)
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def figure3_volume_btc_eth_sol(data_dir: Path, results_dir: Path) -> Path:
    """Two panels: log_volume and vol_spike_20 for BTC, ETH, SOL (daily ML)."""
    assets = [
        ("Bitcoin", "bitcoin_ohlcv_daily_ml.csv"),
        ("Ethereum", "ethereum_ohlcv_daily_ml.csv"),
        ("Solana", "solana_ohlcv_daily_ml.csv"),
    ]
    plot_data: dict[str, pd.DataFrame] = {}
    for label, name in assets:
        p = data_dir / "pre-processing" / name
        if not p.is_file():
            raise FileNotFoundError(f"Missing ML file: {p}")
        df = pd.read_csv(p, parse_dates=["timestamp"])[["timestamp", "log_volume", "vol_spike_20"]]
        plot_data[label] = df.sort_values("timestamp")

    out = _fig_dir(results_dir) / "eda_figure3_volume_btc_eth_sol.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = {"Bitcoin": "#f7931a", "Ethereum": "#627eea", "Solana": "#9945ff"}
    for label, df in plot_data.items():
        c = colors.get(label, None)
        axes[0].plot(df["timestamp"], df["log_volume"], label=label, lw=0.8, alpha=0.9, color=c)
        axes[1].plot(df["timestamp"], df["vol_spike_20"], label=label, lw=0.8, alpha=0.9, color=c)

    axes[0].set_ylabel("log_volume")
    axes[0].set_title("Daily volume — log scale feature")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].set_ylabel("vol_spike_20")
    axes[1].set_title("Daily volume — spike vs 20-bar MA")
    axes[1].set_xlabel("Date (UTC)")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Volume behaviour — BTC, ETH, SOL (daily)", y=1.01)
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate EDA figures (price, returns/vol, volume).")
    p.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA, help="Project data/ folder.")
    p.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS, help="Project results/ folder.")
    p.add_argument(
        "--figures",
        nargs="*",
        type=int,
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which figures to generate (default: all).",
    )
    p.add_argument(
        "--price-normalize",
        choices=["common", "each"],
        default="common",
        help="Figure 1: index 100 at first common date across assets, or each asset’s first date.",
    )
    args = p.parse_args()

    written: list[Path] = []
    figs = set(args.figures)
    if 1 in figs:
        written.append(figure1_price_index(args.data_dir, args.results_dir, args.price_normalize))
    if 2 in figs:
        written.append(figure2_returns_volatility_bitcoin(args.data_dir, args.results_dir))
    if 3 in figs:
        written.append(figure3_volume_btc_eth_sol(args.data_dir, args.results_dir))

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
