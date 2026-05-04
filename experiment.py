"""
Fast, small-scale experiments: feature-group ablations, context sweeps, permutation importance.

Uses the same walk-forward logic as benchmark.py (RandomForest / XGBoost only by default).
Results are indicative only—not a substitute for a full benchmark.py evaluation.

Examples::

    python experiment.py
    python experiment.py --modes ablation context importance
    python experiment.py --dataset data/merge/bitcoin_ohlcv_hourly_merged.csv --max-rows 5000
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from benchmark import (
    TARGET_PREFIX,
    EvalConfig,
    build_xy,
    horizons_for_freq,
    load_dataset,
    mae,
    rmse,
    walk_forward_sklearn,
)

_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATASET = _ROOT / "data" / "merge" / "bitcoin_ohlcv_daily_merged.csv"
_RESULTS_EXP = _ROOT / "results" / "experiments"


def _classify_features(columns: Iterable[str], strict_ohlcv: bool) -> dict[str, list[str]]:
    cols = list(columns)
    macro = [c for c in cols if c.startswith("macro_")]
    chain = [c for c in cols if c.startswith("btc_bc_") or c.startswith("eth_bc_")]
    calendar = [c for c in cols if c in ("day_of_week", "hour")]
    rest = [c for c in cols if c not in macro + chain + calendar]
    if strict_ohlcv:
        ohlcv = [c for c in rest if c not in calendar]
    else:
        ohlcv = rest + calendar
    return {"ohlcv": ohlcv, "macro": macro, "chain": chain}


def _ablation_column_sets(
    X: pd.DataFrame, strict_ohlcv: bool
) -> list[tuple[str, list[str]]]:
    g = _classify_features(X.columns, strict_ohlcv=strict_ohlcv)
    o, m, ch = g["ohlcv"], g["macro"], g["chain"]
    out: list[tuple[str, list[str]]] = [
        ("ohlcv_only", o),
        ("ohlcv_plus_macro", sorted(set(o + m))),
    ]
    if ch:
        out.append(("ohlcv_macro_chain", sorted(set(o + m + ch))))
    out.append(("all_features", list(X.columns)))
    # Dedupe consecutive duplicates (e.g. solana no chain: same as ohlcv_plus_macro)
    seen: set[frozenset[str]] = set()
    uniq: list[tuple[str, list[str]]] = []
    for name, cols in out:
        key = frozenset(cols)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, cols))
    return uniq


def _target_column(family: str, horizon: int) -> str:
    tmpl = TARGET_PREFIX[family]
    return tmpl.format(h=horizon)


def _make_model_factory(cfg: EvalConfig, kind: str) -> Callable[[], Any]:
    if kind == "rf":

        def f() -> RandomForestRegressor:
            return RandomForestRegressor(
                n_estimators=cfg.rf_n_estimators,
                n_jobs=-1,
                random_state=42,
            )

        return f

    def g() -> XGBRegressor:
        return XGBRegressor(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    return g


def _base_eval_config(
    merge_dir: Path,
    results_dir: Path,
    *,
    refit_daily: int,
    refit_hourly: int,
    min_train_daily: int,
    min_train_hourly: int,
    hourly_sliding_context: int,
    hourly_eval_tail: int,
    rf_n: int,
    xgb_n: int,
) -> EvalConfig:
    return EvalConfig(
        merge_dir=merge_dir,
        results_dir=results_dir,
        refit_every_daily=refit_daily,
        refit_every_hourly=refit_hourly,
        min_train_daily=min_train_daily,
        min_train_hourly=min_train_hourly,
        hourly_sliding_context=hourly_sliding_context,
        hourly_eval_tail=hourly_eval_tail,
        rf_n_estimators=rf_n,
        xgb_n_estimators=xgb_n,
        skip_prophet=True,
        skip_neural=True,
    )


def _run_walkforward_mae(
    X: pd.DataFrame,
    y: pd.Series,
    freq: str,
    cfg: EvalConfig,
    model_kind: str,
    horizon: int,
) -> tuple[float, float]:
    factory = _make_model_factory(cfg, model_kind)
    name = "RF" if model_kind == "rf" else "XGB"
    yt, yp, ix, _ = walk_forward_sklearn(X, y, freq, cfg, factory, name, horizon)
    m = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(m):
        return float("nan"), float("nan")
    return mae(yt[m], yp[m]), rmse(yt[m], yp[m])


def run_ablation(
    X: pd.DataFrame,
    y: pd.Series,
    freq: str,
    cfg: EvalConfig,
    model_kind: str,
    strict_ohlcv: bool,
    horizon: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, cols in _ablation_column_sets(X, strict_ohlcv):
        if not cols:
            continue
        Xs = X[[c for c in cols if c in X.columns]]
        if Xs.shape[1] == 0:
            continue
        mae_v, rmse_v = _run_walkforward_mae(Xs, y, freq, cfg, model_kind, horizon)
        rows.append(
            {
                "mode": "ablation",
                "variant": label,
                "n_features": Xs.shape[1],
                "MAE": mae_v,
                "RMSE": rmse_v,
            }
        )
    return rows


def run_context_daily(
    X: pd.DataFrame,
    y: pd.Series,
    freq: str,
    base: EvalConfig,
    model_kind: str,
    min_train_grid: list[int],
    horizon: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mtr in min_train_grid:
        cfg = EvalConfig(
            merge_dir=base.merge_dir,
            results_dir=base.results_dir,
            refit_every_daily=base.refit_every_daily,
            refit_every_hourly=base.refit_every_hourly,
            min_train_daily=mtr,
            min_train_hourly=base.min_train_hourly,
            hourly_sliding_context=base.hourly_sliding_context,
            hourly_eval_tail=base.hourly_eval_tail,
            rf_n_estimators=base.rf_n_estimators,
            xgb_n_estimators=base.xgb_n_estimators,
            skip_prophet=True,
            skip_neural=True,
        )
        mae_v, rmse_v = _run_walkforward_mae(X, y, freq, cfg, model_kind, horizon)
        rows.append(
            {
                "mode": "context_daily",
                "variant": f"min_train_daily={mtr}",
                "n_features": X.shape[1],
                "MAE": mae_v,
                "RMSE": rmse_v,
            }
        )
    return rows


def run_context_hourly(
    X: pd.DataFrame,
    y: pd.Series,
    freq: str,
    base: EvalConfig,
    model_kind: str,
    context_grid: list[int],
    hourly_tail: int,
    horizon: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for C in context_grid:
        cfg = EvalConfig(
            merge_dir=base.merge_dir,
            results_dir=base.results_dir,
            refit_every_daily=base.refit_every_daily,
            refit_every_hourly=base.refit_every_hourly,
            min_train_daily=base.min_train_daily,
            min_train_hourly=max(base.min_train_hourly, C),
            hourly_sliding_context=C,
            hourly_eval_tail=hourly_tail,
            rf_n_estimators=base.rf_n_estimators,
            xgb_n_estimators=base.xgb_n_estimators,
            skip_prophet=True,
            skip_neural=True,
        )
        mae_v, rmse_v = _run_walkforward_mae(X, y, freq, cfg, model_kind, horizon)
        rows.append(
            {
                "mode": "context_hourly",
                "variant": f"hourly_sliding_context={C},hourly_eval_tail={hourly_tail}",
                "n_features": X.shape[1],
                "MAE": mae_v,
                "RMSE": rmse_v,
            }
        )
    return rows


def run_top_corr_train(X: pd.DataFrame, y: pd.Series, train_frac: float, top_k: int) -> list[dict[str, Any]]:
    n = len(X)
    split = max(50, int(n * train_frac))
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    sub = pd.concat([X_tr, y_tr.rename("_y")], axis=1).dropna()
    if len(sub) < 30:
        return []
    cor = sub.drop(columns=["_y"]).corrwith(sub["_y"]).abs().sort_values(ascending=False)
    return [
        {
            "mode": "corr_train",
            "feature": str(idx),
            "abs_corr": float(val),
        }
        for idx, val in cor.head(top_k).items()
    ]


def run_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float,
    cfg: EvalConfig,
    model_kind: str,
    n_repeats: int,
    top_k: int,
) -> list[dict[str, Any]]:
    n = len(X)
    split = max(50, int(n * train_frac))
    if split >= n - 20:
        split = n // 2
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[split:], y.iloc[split:]

    factory = _make_model_factory(cfg, model_kind)
    model = factory()
    model.fit(X_tr.to_numpy(dtype=float), y_tr.to_numpy(dtype=float))

    pi = permutation_importance(
        model,
        X_te.to_numpy(dtype=float),
        y_te.to_numpy(dtype=float),
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )
    order = np.argsort(pi.importances_mean)[::-1][:top_k]
    return [
        {
            "mode": "permutation_importance",
            "feature": X.columns[i],
            "importance_mean": float(pi.importances_mean[i]),
            "importance_std": float(pi.importances_std[i]),
        }
        for i in order
    ]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    """Append rows to CSV with stable column union (handles mixed experiment modes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    enriched = [{**meta, **r} for r in rows]
    new_df = pd.DataFrame(enriched)
    if path.is_file():
        old = pd.read_csv(path)
        combined = pd.concat([old, new_df], ignore_index=True, sort=False)
        combined.to_csv(path, index=False)
    else:
        new_df.to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Lightweight feature/context experiments (fast).")
    p.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET, help="Merged CSV path.")
    p.add_argument("--max-rows", type=int, default=1000, help="Use last N rows after load (recent history).")
    p.add_argument("--family", choices=["return", "volatility", "volume"], default="return")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument(
        "--modes",
        default="ablation",
        help="Comma-separated: ablation, context, importance, corr (corr adds train correlations; "
        "importance includes permutation importance).",
    )
    p.add_argument("--model", choices=["rf", "xgb"], default="rf")
    p.add_argument("--rf-n-estimators", type=int, default=50)
    p.add_argument("--xgb-n-estimators", type=int, default=80)
    p.add_argument("--refit-every-daily", type=int, default=48)
    p.add_argument("--refit-every-hourly", type=int, default=48)
    p.add_argument("--min-train-daily", type=int, default=200, help="Baseline min_train for ablation.")
    p.add_argument("--min-train-hourly", type=int, default=1500)
    p.add_argument("--hourly-sliding-context", type=int, default=800)
    p.add_argument("--hourly-eval-tail", type=int, default=120)
    p.add_argument(
        "--min-train-grid",
        type=str,
        default="120,200,300",
        help="For --modes context on daily data: comma-separated min_train_daily values.",
    )
    p.add_argument(
        "--hourly-context-grid",
        type=str,
        default="400,800,1600",
        help="For --modes context on hourly data: comma-separated sliding context lengths.",
    )
    p.add_argument("--strict-ohlcv", action="store_true", help="Exclude day_of_week and hour from ohlcv_only.")
    p.add_argument("--train-frac", type=float, default=0.85, help="Train fraction for permutation importance.")
    p.add_argument("--perm-repeats", type=int, default=8)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--save-csv", action="store_true", help=f"Append run to {_RESULTS_EXP / 'runs.csv'}")
    args = p.parse_args()

    modes = {m.strip().lower() for m in args.modes.split(",") if m.strip()}

    df, freq, stem = load_dataset(args.dataset)
    if args.horizon not in horizons_for_freq(freq):
        raise SystemExit(
            f"horizon {args.horizon} not in {horizons_for_freq(freq)} for freq={freq}"
        )
    ycol = _target_column(args.family, args.horizon)
    if ycol not in df.columns:
        raise SystemExit(f"Missing column {ycol}")

    df = df.tail(args.max_rows).copy()
    X, y_map = build_xy(df, freq)
    if (args.family, args.horizon) not in y_map:
        raise SystemExit(f"Target ({args.family}, {args.horizon}) not in y_map (volatility h=1 skipped?).")
    y = df[y_map[(args.family, args.horizon)]].astype(float)

    merge_dir = args.dataset.parent
    results_dir = _ROOT / "results"
    base_cfg = _base_eval_config(
        merge_dir,
        results_dir,
        refit_daily=args.refit_every_daily,
        refit_hourly=args.refit_every_hourly,
        min_train_daily=args.min_train_daily,
        min_train_hourly=args.min_train_hourly,
        hourly_sliding_context=args.hourly_sliding_context,
        hourly_eval_tail=args.hourly_eval_tail,
        rf_n=args.rf_n_estimators,
        xgb_n=args.xgb_n_estimators,
    )

    print(f"Dataset: {args.dataset.name}  stem={stem}  freq={freq}  rows={len(df)} (tail {args.max_rows})")
    print(f"Target: {args.family} h={args.horizon}  ({ycol})  features={X.shape[1]}")
    print(f"Model: {args.model}  modes={sorted(modes)}")
    print()

    all_rows: list[dict[str, Any]] = []
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset.name,
        "stem": stem,
        "freq": freq,
        "family": args.family,
        "horizon": args.horizon,
        "model": args.model,
        "max_rows": args.max_rows,
    }

    if "ablation" in modes:
        rows = run_ablation(X, y, freq, base_cfg, args.model, args.strict_ohlcv, args.horizon)
        all_rows.extend(rows)
        print("=== Ablation (feature groups) ===")
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    if "context" in modes:
        if freq == "daily":
            grid = _parse_int_list(args.min_train_grid)
            rows = run_context_daily(X, y, freq, base_cfg, args.model, grid, args.horizon)
        else:
            grid = _parse_int_list(args.hourly_context_grid)
            rows = run_context_hourly(
                X, y, freq, base_cfg, args.model, grid, args.hourly_eval_tail, args.horizon
            )
        all_rows.extend(rows)
        print("=== Context sweep ===")
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    if "importance" in modes:
        pi_rows = run_permutation_importance(
            X,
            y,
            args.train_frac,
            base_cfg,
            args.model,
            args.perm_repeats,
            args.top_k,
        )
        all_rows.extend(pi_rows)
        print("=== Permutation importance (holdout tail) ===")
        print(pd.DataFrame(pi_rows).to_string(index=False))
        print()

    if "corr" in modes:
        corr_rows = run_top_corr_train(X, y, args.train_frac, args.top_k)
        if corr_rows:
            all_rows.extend(corr_rows)
            print("=== Top |corr| with y (train slice) ===")
            print(pd.DataFrame(corr_rows).to_string(index=False))
            print()

    if args.save_csv and all_rows:
        out = _RESULTS_EXP / "runs.csv"
        _write_csv(out, all_rows, meta)
        print(f"Saved {len(all_rows)} rows to {out} (appended with column-union if file existed)")


if __name__ == "__main__":
    main()
