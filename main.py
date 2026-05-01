"""
End-to-end pipeline: OHLCV → ML tables → merged features → quick model run.

1. ``data_preparation.build_all_ohlcv_datasets`` — writes ``data/pre-processing/*_ml.csv``
2. ``merge.merge_all`` — writes ``data/merge/*_merged.csv``
3. ``model.py`` — walk-forward evaluation (same CLI as running ``python model.py``)

Does not run ``experiment.py``, ``benchmark.py``, or ``scripts/``.

Example::

    python main.py
    python main.py --skip-data-prep --skip-merge
    python main.py --no-model
    python main.py -- --skip-neural --skip-prophet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _run_data_prep(ohlcv_dir: Path, preproc_dir: Path) -> None:
    from data_preparation import build_all_ohlcv_datasets

    print("==> Step 1/3: data_preparation (OHLCV → * _ml.csv)", flush=True)
    t0 = time.perf_counter()
    results = build_all_ohlcv_datasets(ohlcv_dir=ohlcv_dir, output_dir=preproc_dir)
    for path, shape in results:
        print(f"     Wrote {path.name}  rows={shape[0]}  cols={shape[1]}", flush=True)
    if not results:
        raise RuntimeError(
            f"No OHLCV CSVs found in {ohlcv_dir}. Add files like bitcoin_ohlcv_daily.csv or skip step 1 if *_ml.csv already exist."
        )
    print(f"     Step 1 done in {time.perf_counter() - t0:.1f}s", flush=True)


def _run_merge(preproc_dir: Path, merge_out: Path) -> None:
    from merge import merge_all

    print("==> Step 2/3: merge (ML + macro + chain → *_merged.csv)", flush=True)
    t0 = time.perf_counter()
    paths = merge_all(preproc_dir=preproc_dir, output_dir=merge_out)
    for p in paths:
        print(f"     Wrote {p.name}", flush=True)
    if not paths:
        raise RuntimeError(
            f"No ML CSVs matching '*_ohlcv_*_ml.csv' in {preproc_dir}. Run step 1 first."
        )
    print(f"     Step 2 done in {time.perf_counter() - t0:.1f}s", flush=True)


def _run_model(model_argv: list[str]) -> int:
    print("==> Step 3/3: model.py (quick walk-forward)", flush=True)
    t0 = time.perf_counter()
    cmd = [sys.executable, str(_ROOT / "model.py"), *model_argv]
    print(f"     {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=_ROOT, check=False)
    print(f"     Step 3 done in {time.perf_counter() - t0:.1f}s (exit {r.returncode})", flush=True)
    return r.returncode


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run data_preparation → merge → model.py (one command).",
    )
    p.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip step 1 (use existing data/pre-processing/*_ml.csv).",
    )
    p.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip step 2 (use existing data/merge/*_merged.csv).",
    )
    p.add_argument(
        "--no-model",
        action="store_true",
        help="Stop after merge; do not run model.py.",
    )
    p.add_argument(
        "--ohlcv-dir",
        type=Path,
        default=_ROOT / "data" / "ohlcv",
        help="Input OHLCV folder for step 1.",
    )
    p.add_argument(
        "--preproc-dir",
        type=Path,
        default=_ROOT / "data" / "pre-processing",
        help="Output folder for *_ml.csv (step 1) and input for merge (step 2).",
    )
    p.add_argument(
        "--merge-out",
        type=Path,
        default=_ROOT / "data" / "merge",
        help="Output folder for *_merged.csv (step 2).",
    )
    args, model_argv = p.parse_known_args()

    ohlcv_dir = args.ohlcv_dir.resolve()
    preproc_dir = args.preproc_dir.resolve()
    merge_out = args.merge_out.resolve()

    if not args.skip_data_prep:
        _run_data_prep(ohlcv_dir, preproc_dir)
    else:
        print("==> Skipping data_preparation (--skip-data-prep)", flush=True)

    if not args.skip_merge:
        _run_merge(preproc_dir, merge_out)
    else:
        print("==> Skipping merge (--skip-merge)", flush=True)

    if args.no_model:
        print("==> Stopping before model (--no-model).", flush=True)
        return 0

    code = _run_model(model_argv)
    return code


if __name__ == "__main__":
    sys.exit(main())
