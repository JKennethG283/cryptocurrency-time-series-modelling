"""
Fast walk-forward evaluation (Bitcoin + Ethereum, return and volatility only).

* **return** at horizon **1** (next bar) — ``target_ret_fwd_1`` for hourly and daily.
* **volatility** — ``target_vol_fwd_7`` on **daily** data (week-ahead realized vol over 7 bars) and
  ``target_vol_fwd_24`` on **hourly** data (day-ahead realized vol over 24 bars). Volatility h=1 is not
  used in this project.

For the full six-dataset grid and all target families, use ``python benchmark.py``.

Hourly defaults: sliding context **600**, evaluation on the last **1000** rows. Writes metrics under
``results/quick/`` by default so full ``benchmark`` runs in ``results/tables`` are not overwritten.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

from benchmark import EvalConfig, _DEFAULT_MERGE, run_all

_ROOT = Path(__file__).resolve().parent
_DEFAULT_RESULTS_QUICK = _ROOT / "results" / "quick"

# Interpretable horizons: week of daily returns / one day of hourly returns.
_VOL_H_DAILY = 7
_VOL_H_HOURLY = 24

_DEFAULT_BTC_ETH_DATASETS: tuple[str, ...] = (
    "bitcoin_ohlcv_hourly_merged.csv",
    "ethereum_ohlcv_hourly_merged.csv",
    "bitcoin_ohlcv_daily_merged.csv",
    "ethereum_ohlcv_daily_merged.csv",
)

_DEFAULT_TARGET_PAIRS_BY_FREQ: dict[str, tuple[tuple[str, int], ...]] = {
    "hourly": (("return", 1), ("volatility", _VOL_H_HOURLY)),
    "daily": (("return", 1), ("volatility", _VOL_H_DAILY)),
}


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(
        description="Quick walk-forward: BTC+ETH, return h=1, vol h=7 daily and h=24 hourly. "
        "See benchmark.py for full runs."
    )
    p.add_argument("--merge-dir", type=Path, default=_DEFAULT_MERGE)
    p.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_QUICK)
    p.add_argument(
        "--datasets",
        nargs="*",
        default=list(_DEFAULT_BTC_ETH_DATASETS),
        help="Merged CSV filenames (default: four BTC+ETH daily/hourly files).",
    )
    p.add_argument("--refit-every-daily", type=int, default=1)
    p.add_argument("--refit-every-hourly", type=int, default=168)
    p.add_argument("--min-train-daily", type=int, default=300)
    p.add_argument("--min-train-hourly", type=int, default=2000)
    p.add_argument("--max-rows-daily", type=int, default=None)
    p.add_argument("--max-rows-hourly", type=int, default=None)
    p.add_argument("--skip-prophet", action="store_true")
    p.add_argument("--skip-neural", action="store_true")
    p.add_argument(
        "--ts-eval-stride",
        type=int,
        default=1,
        help="Evaluate ARIMA/Prophet/VAR/NLinear every k-th origin (1=all).",
    )
    p.add_argument(
        "--use-sarimax-exog",
        action="store_true",
    )
    p.add_argument(
        "--plot-dataset",
        action="append",
        default=None,
        metavar="FILE",
        dest="plot_datasets",
        help="Only write figures for this merged filename (repeat for several).",
    )
    p.add_argument("--hourly-sliding-context", type=int, default=600)
    p.add_argument("--hourly-eval-tail", type=int, default=1000)
    p.add_argument("--daily-neural-context", type=int, default=90)
    p.add_argument(
        "--neural-max-steps",
        type=int,
        default=15,
    )
    args = p.parse_args()
    return EvalConfig(
        merge_dir=args.merge_dir,
        results_dir=args.results_dir,
        datasets=list(args.datasets),
        refit_every_daily=args.refit_every_daily,
        refit_every_hourly=args.refit_every_hourly,
        min_train_daily=args.min_train_daily,
        min_train_hourly=args.min_train_hourly,
        max_rows_daily=args.max_rows_daily,
        max_rows_hourly=args.max_rows_hourly,
        hourly_sliding_context=args.hourly_sliding_context,
        hourly_eval_tail=args.hourly_eval_tail,
        skip_prophet=args.skip_prophet,
        skip_neural=args.skip_neural,
        ts_eval_stride=args.ts_eval_stride,
        use_sarimax_exog=args.use_sarimax_exog,
        plot_datasets=args.plot_datasets,
        target_pairs=None,
        target_pairs_by_freq=_DEFAULT_TARGET_PAIRS_BY_FREQ,
        neural_max_steps=args.neural_max_steps,
        daily_neural_context=args.daily_neural_context,
    )


if __name__ == "__main__":
    t0 = time.perf_counter()
    print(f"Command: {shlex.join(sys.argv)}", flush=True)
    cfg = parse_args()
    try:
        run_all(cfg)
    finally:
        print(f"Total wall time: {time.perf_counter() - t0:.2f}s", flush=True)
