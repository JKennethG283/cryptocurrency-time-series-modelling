from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import EvalConfig, build_xy, hourly_eval_t_range, mase, walk_forward_arima

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"statsmodels import failed: {exc}") from exc


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))
)


def _safe_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx: np.ndarray,
    y_full: np.ndarray,
    h: int,
) -> dict[str, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return {"MAE": float("nan"), "RMSE": float("nan"), "MASE_h": float("nan"), "n_eval": 0}
    yt = y_true[m]
    yp = y_pred[m]
    ix = idx[m]
    return {
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
        "MASE_h": mase(yt, yp, y_full, h, ix),
        "n_eval": int(len(ix)),
    }


def naive_pred(y: np.ndarray, idx: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(len(idx), np.nan, dtype=float)
    for k, t in enumerate(idx):
        j = int(t) - lag
        if j >= 0:
            out[k] = float(y[j])
    return out


def walk_forward_arima_h_embargo(
    y: np.ndarray,
    context: int,
    tail: int,
    h: int,
    order: tuple[int, int, int] = (1, 0, 1),
    eval_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Strict horizon-aware variant:
    for test origin t, train only on label indices j <= t-h
    so train labels' forward windows never overlap test window [t+1, t+h].
    """
    n = len(y)
    t_first, t_end = hourly_eval_t_range(n, context, tail)
    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    idx_list: list[int] = []

    for t in range(t_first, t_end):
        if (t - t_first) % eval_stride != 0:
            continue
        start = t - context
        train_end = t - h + 1  # slice end-exclusive; includes j=t-h
        if train_end <= start + 30:
            continue
        y_train = y[start:train_end]
        y_hat = float("nan")
        try:
            mod = ARIMA(y_train, order=order)
            res = mod.fit()
            fc = res.forecast(steps=1)
            y_hat = float(np.asarray(fc).ravel()[0])
        except Exception:
            y_hat = float("nan")
        y_true_list.append(float(y[t]))
        y_pred_list.append(y_hat)
        idx_list.append(t)
    return np.asarray(y_true_list), np.asarray(y_pred_list), np.asarray(idx_list)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate BTC hourly ARIMA volatility result.")
    p.add_argument(
        "--merge-csv",
        type=Path,
        default=Path("data/merge/bitcoin_ohlcv_hourly_merged.csv"),
    )
    p.add_argument("--context", type=int, default=600)
    p.add_argument("--tail", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--order", type=int, nargs=3, default=(1, 0, 1))
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    df = pd.read_csv(args.merge_csv)
    X, y_map = build_xy(df, "hourly")
    y_col = y_map[("volatility", args.horizon)]
    y_series = df[y_col].astype(float)
    y = y_series.to_numpy(dtype=float)

    cfg = EvalConfig(
        merge_dir=Path("data/merge"),
        results_dir=Path("results/quick"),
        datasets=[args.merge_csv.name],
        hourly_sliding_context=args.context,
        hourly_eval_tail=args.tail,
        ts_eval_stride=args.stride,
        arima_order=tuple(args.order),
    )

    # Current benchmark protocol (matches walk_forward_arima in benchmark.py)
    yt_b, yp_b, ix_b, _ = walk_forward_arima(y_series, X, "hourly", cfg, args.horizon)
    m_b = _safe_metrics(yt_b, yp_b, ix_b, y, args.horizon)

    # Same protocol but non-overlapping eval points (step = h)
    cfg_nonoverlap = replace(cfg, ts_eval_stride=args.horizon)
    yt_s, yp_s, ix_s, _ = walk_forward_arima(y_series, X, "hourly", cfg_nonoverlap, args.horizon)
    m_s = _safe_metrics(yt_s, yp_s, ix_s, y, args.horizon)

    # Strict horizon-aware embargo in training labels
    yt_e, yp_e, ix_e = walk_forward_arima_h_embargo(
        y=y,
        context=args.context,
        tail=args.tail,
        h=args.horizon,
        order=tuple(args.order),
        eval_stride=args.horizon,
    )
    m_e = _safe_metrics(yt_e, yp_e, ix_e, y, args.horizon)

    # Compare ARIMA vs SARIMAX(exog) under same sparse grid.
    cfg_sarimax = replace(cfg_nonoverlap, use_sarimax_exog=True)
    yt_x, yp_x, ix_x, _ = walk_forward_arima(y_series, X, "hourly", cfg_sarimax, args.horizon)
    m_x = _safe_metrics(yt_x, yp_x, ix_x, y, args.horizon)

    # Naive baselines on benchmark index set for fair comparison
    yp_naive_1 = naive_pred(y, ix_b, 1)
    yp_naive_h = naive_pred(y, ix_b, args.horizon)
    m_n1 = _safe_metrics(yt_b, yp_naive_1, ix_b, y, args.horizon)
    m_nh = _safe_metrics(yt_b, yp_naive_h, ix_b, y, args.horizon)

    print("=== Dataset / Target ===")
    print(f"csv={args.merge_csv}")
    print(f"target={y_col}")
    print(f"context={args.context}, tail={args.tail}, h={args.horizon}, order={tuple(args.order)}")
    print()
    print("=== Current benchmark protocol (hourly sliding + tail) ===")
    print(m_b)
    print()
    print("=== Current protocol + non-overlap eval stride=h ===")
    print(m_s)
    print()
    print("=== Strict horizon-aware embargo (train j <= t-h) + stride=h ===")
    print(m_e)
    print()
    print("=== ARIMA vs SARIMAX(exog) on same stride=h protocol ===")
    print("ARIMA(default):", m_s)
    print("SARIMAX(exog):", m_x)
    print()
    print("=== Naive baselines on current protocol eval points ===")
    print("naive lag=1:", m_n1)
    print("naive lag=h:", m_nh)


if __name__ == "__main__":
    main()
