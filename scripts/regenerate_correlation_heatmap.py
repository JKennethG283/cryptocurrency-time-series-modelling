"""
Regenerate only the correlation heatmap PNG (no model training).

Uses the same logic as ``benchmark.py`` for column selection (macro/chain prioritized).

Example::

    python scripts/regenerate_correlation_heatmap.py
    python scripts/regenerate_correlation_heatmap.py --dataset ethereum_ohlcv_hourly_merged.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Run from project root so ``import benchmark`` resolves.
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark import (  # noqa: E402
    _DEFAULT_MERGE,
    _DEFAULT_RESULTS,
    build_xy,
    correlation_heatmap,
    horizons_for_freq,
    load_dataset,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate correlation heatmap from a merged CSV.")
    p.add_argument(
        "--dataset",
        type=str,
        default="bitcoin_ohlcv_daily_merged.csv",
        help="Filename under --merge-dir (e.g. bitcoin_ohlcv_hourly_merged.csv).",
    )
    p.add_argument("--merge-dir", type=Path, default=_DEFAULT_MERGE)
    p.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS)
    args = p.parse_args()

    path = args.merge_dir / args.dataset
    if not path.is_file():
        raise FileNotFoundError(path)

    df, freq, stem = load_dataset(path)
    X, y_map = build_xy(df, freq)
    hs = horizons_for_freq(freq)
    h1 = min(hs)
    ycols = [
        y_map[("return", h1)],
        y_map[("volatility", h1)],
        y_map[("volume", h1)],
    ]
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"correlation_heatmap_{stem}.png"
    correlation_heatmap(df, list(X.columns), ycols, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
