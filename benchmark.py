"""
Multi-model walk-forward evaluation on merged OHLCV + chain + macro CSVs.

How to run
==========

1. Install dependencies (from the project folder)::

     pip install -r requirements.txt

2. Ensure merged CSVs exist under ``data/merge/`` (from ``merge.py``).

3. Run everything (all six ``*_merged.csv`` files, all model families)::

     python benchmark.py

   First test (faster; one asset; coarser time-series evaluation)::

     python benchmark.py --datasets bitcoin_ohlcv_daily_merged.csv --ts-eval-stride 5 \\
         --refit-every-daily 3 --skip-neural

   Skip Prophet or neural models if needed::

     python benchmark.py --skip-prophet
     python benchmark.py --skip-neural

   A faster default (Bitcoin + Ethereum, return + volatility only) is ``python model.py``
   (see ``model.py`` in this repo; calls the same ``run_all`` with a reduced target set).

Outputs
=======

* ``results/tables/metrics_returns.csv``
* ``results/tables/metrics_volatility.csv``
* ``results/tables/metrics_volume.csv``
* ``results/figures/`` — per dataset (see ``--plot-dataset``): correlation heatmap
  (``correlation_heatmap_<stem>.png``), **best-model** actual vs predicted
  ``best_actual_vs_predicted_<bestmodel>_<family>_h<h>_<stem>.png``, diagnostics
  ``best_diag_<family>_h<h>_<stem>.png`` (best = lowest MAE among models with finite predictions for
  that target). XGBoost feature importance ``xgb_feature_importance_<stem>.png`` when XGBoost wins
  for **return** at the first horizon. **Volatility at horizon 1 is not evaluated** (see below).

**Volatility h=1:** Not used: ``data_preparation.py`` does not emit ``target_vol_fwd_1`` (std of one
future return would be 0). This script also skips ``(volatility, 1)`` in ``build_xy``; use horizons
≥ 2 for volatility.

What “all results” means
=========================

By default the script loops **six datasets** (bitcoin/ethereum/solana × daily/hourly merged CSVs).

For each dataset and each target family (returns / volatility / volume) and each horizon, it fits:

* **RandomForest**, **XGBoost** (tabular walk-forward),
* **ARIMAX** label (implementation: ARIMA by default; optional SARIMAX with ``--use-sarimax-exog``),
* **VAR_volume** (volume targets only),
* **Prophet** (unless ``--skip-prophet``),
* **NLinear** (NeuralForecast) once per target — daily matches other models: **expanding** history and
  walk-forward from ``min_train_daily`` through the end of the series, with ``input_size`` =
  ``--daily-neural-context`` (default 90); hourly matches other hourly models: **sliding** window of
  ``--hourly-sliding-context`` (default 3000) and evaluation only on the last ``--hourly-eval-tail``
  rows (default 1000).

That is **more than six models** — six is the number of **data files**, not the number of algorithms.

Walk-forward windowing
======================

* **Daily** (*expanding*): for origin ``t``, train on rows ``0 .. t-1``, predict row ``t``.
* **Hourly** (*sliding + tail*): for origin ``t`` in the last ``hourly_eval_tail`` rows (and
  ``t >= hourly_sliding_context``), train on rows ``t - C .. t - 1`` with ``C = hourly_sliding_context``,
  predict row ``t``.

Sklearn models refit every ``refit_every_*`` but still predict every step in the evaluated range.
ARIMA / Prophet / VAR / NLinear use ``ts_eval_stride`` on the same ``t`` grid (for hourly, stride is
relative to the first origin in the tail).

How long it takes (rough)
==========================

* **Daily-only, six files, stride 1, full NLinear**: still heavy (neural refit per origin) but **much
  quicker than NHITS** on the same ``max_steps``; ARIMA refits every evaluated origin.
* **Hourly** with defaults (3000 context, 1000 tail) is far fewer origins than full-series expanding
  walk-forward; still use ``--ts-eval-stride`` (e.g. 24–168) and/or ``--skip-neural`` if needed.

NLinear uses ``h=1`` in index space (the economic horizon is already in the target column).

Prophet / NLinear troubleshooting
===================================

* Empty Prophet metrics: usually every ``predict`` failed (timezone ``ds`` was a common cause;
  code strips to naive UTC). Check **CmdStanPy** for Prophet on Windows.
* Missing NLinear rows: ``--skip-neural`` or failed ``neuralforecast``/``torch`` import.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Prophet/CmdStanPy log every Stan chain per fit — very noisy during walk-forward.
for _name in ("cmdstanpy", "prophet", "stan", "plotly"):
    logging.getLogger(_name).setLevel(logging.WARNING)

_ROOT = Path(__file__).resolve().parent
_DEFAULT_MERGE = _ROOT / "data" / "merge"
_DEFAULT_RESULTS = _ROOT / "results"

HOURLY_HORIZONS = (1, 6, 24)
DAILY_HORIZONS = (1, 3, 7)

TARGET_PREFIX = {
    "return": "target_ret_fwd_{h}",
    "volatility": "target_vol_fwd_{h}",
    "volume": "target_log_vol_fwd_{h}",
}


@dataclass
class EvalConfig:
    merge_dir: Path
    results_dir: Path
    datasets: Optional[list[str]] = None
    refit_every_daily: int = 1
    refit_every_hourly: int = 168
    min_train_daily: int = 300
    min_train_hourly: int = 2000
    max_rows_daily: Optional[int] = None
    max_rows_hourly: Optional[int] = None
    # Hourly-only: sliding train window length and evaluation on the last N rows only.
    hourly_sliding_context: int = 3000
    hourly_eval_tail: int = 1000
    arima_order: tuple[int, int, int] = (1, 0, 1)
    var_maxlags: int = 5
    top_exog_k: int = 8
    rf_n_estimators: int = 80
    xgb_n_estimators: int = 150
    neural_max_steps: int = 15
    # Single NLinear input_size for daily (expanding-window train); hourly uses ``hourly_sliding_context``.
    daily_neural_context: int = 90
    skip_prophet: bool = False
    skip_neural: bool = False
    # If None, generate figures for every dataset in the run; else only listed filenames.
    plot_datasets: Optional[list[str]] = None
    # If None, evaluate all (family, h) from ``build_xy``; else only these keys (e.g. quick runs).
    target_pairs: Optional[tuple[tuple[str, int], ...]] = None
    # Like ``target_pairs`` but keyed by ``freq`` (``"hourly"`` / ``"daily"``) when horizons differ.
    # If set, takes precedence over ``target_pairs`` for filtering.
    target_pairs_by_freq: Optional[dict[str, tuple[tuple[str, int], ...]]] = None
    # Evaluate ARIMA/Prophet/VAR/NLinear only every k-th origin (1 = every step).
    ts_eval_stride: int = 1
    use_sarimax_exog: bool = False


def hourly_eval_t_range(n: int, context: int, tail: int) -> tuple[int, int]:
    """
    Hourly walk-forward: evaluate origins ``t`` in ``[t_first, t_end)`` (``t_end`` exclusive).

    Uses the last ``tail`` rows as prediction points, but each ``t`` must satisfy ``t >= context``
    so training can use a full ``context``-row history when available.
    Returns ``(0, 0)`` if there is no valid range.
    """
    if n <= 0 or tail <= 0 or context <= 0:
        return 0, 0
    t_end = n
    tail_start = max(0, n - tail)
    t_first = max(context, tail_start)
    if t_first >= t_end:
        return 0, 0
    return t_first, t_end


def _prophet_ds(timestamps: pd.Series) -> pd.Series:
    """Prophet expects naive datetimes; UTC-aware inputs often break fit/predict."""
    s = pd.to_datetime(timestamps, utc=True)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def infer_freq_from_name(path: Path) -> str:
    s = path.stem.lower()
    if "_hourly_" in s or s.endswith("_hourly_merged"):
        return "hourly"
    if "_daily_" in s or s.endswith("_daily_merged"):
        return "daily"
    raise ValueError(f"Cannot infer freq from {path.name}")


def load_dataset(path: Path) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    freq = infer_freq_from_name(path)
    return df, freq, path.stem


def horizons_for_freq(freq: str) -> tuple[int, ...]:
    return HOURLY_HORIZONS if freq == "hourly" else DAILY_HORIZONS


def shortest_volatility_horizon(horizons: tuple[int, ...]) -> int:
    """Smallest horizon > 1 (volatility h=1 is not used in ``build_xy``)."""
    cands = [h for h in horizons if h > 1]
    if not cands:
        raise ValueError("horizons must include some h > 1 for volatility")
    return min(cands)


def build_xy(df: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, dict[tuple[str, int], str]]:
    """Return feature matrix X and mapping (family, h) -> target column name.

    All ``target_*`` columns are excluded from X so labels never leak into features.
    """
    drop_cols = {"timestamp"} | {c for c in df.columns if c.startswith("target_")}
    feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].copy()
    hs = horizons_for_freq(freq)
    y_map: dict[tuple[str, int], str] = {}
    for family, tmpl in TARGET_PREFIX.items():
        for h in hs:
            # target_vol_fwd_1 = std of one future return => always 0; skip (not a meaningful forecast).
            if family == "volatility" and h == 1:
                continue
            col = tmpl.format(h=h)
            if col not in df.columns:
                raise KeyError(f"Missing target column {col}")
            y_map[(family, h)] = col
    return X, y_map


def min_horizon_for_family(y_map: dict[tuple[str, int], str], family: str) -> int:
    return min(h for (fam, h) in y_map if fam == family)


def heatmap_target_columns(y_map: dict[tuple[str, int], str]) -> list[str]:
    """One target column per family present, using the minimum horizon for that family."""
    ycols: list[str] = []
    for fam in ("return", "volatility", "volume"):
        if not any(k[0] == fam for k in y_map):
            continue
        h0 = min_horizon_for_family(y_map, fam)
        ycols.append(y_map[(fam, h0)])
    return ycols


def should_emit_figures(cfg: EvalConfig, dataset_filename: str) -> bool:
    if cfg.plot_datasets is None:
        return True
    return dataset_filename in cfg.plot_datasets


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_full: np.ndarray,
    h: int,
    eval_indices: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    MASE vs naive forecast y_hat_t = y_{t-h} on the same target series.
    eval_indices are row positions in y_full corresponding to y_true entries.
    """
    naive = np.full(len(eval_indices), np.nan)
    for k, t in enumerate(eval_indices):
        j = int(t) - h
        if j >= 0:
            naive[k] = y_full[j]
    mask = np.isfinite(naive) & np.isfinite(y_true)
    if not np.any(mask):
        return float("nan")
    naive_mae = float(np.mean(np.abs(y_true[mask] - naive[mask])))
    if naive_mae < eps:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])) / naive_mae)


def refit_every_for_freq(freq: str, cfg: EvalConfig) -> int:
    return cfg.refit_every_hourly if freq == "hourly" else cfg.refit_every_daily


def min_train_for_freq(freq: str, cfg: EvalConfig) -> int:
    return cfg.min_train_hourly if freq == "hourly" else cfg.min_train_daily


def walk_forward_indices(n: int, min_train: int, refit_every: int) -> Iterator[tuple[int, bool]]:
    for t in range(min_train, n):
        refit = (t == min_train) or ((t - min_train) % refit_every == 0)
        yield t, refit


def select_top_exog(X: pd.DataFrame, y: pd.Series, k: int) -> list[str]:
    sub = pd.concat([X, y.rename("_y")], axis=1).dropna()
    if len(sub) < 50:
        return list(X.columns[: min(k, X.shape[1])])
    cor = sub.drop(columns=["_y"]).corrwith(sub["_y"]).abs().sort_values(ascending=False)
    return list(cor.index[:k])


def walk_forward_sklearn(
    X: pd.DataFrame,
    y: pd.Series,
    freq: str,
    cfg: EvalConfig,
    make_model: Callable[[], Any],
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Train on rows ``0..t-1`` (daily) or sliding ``t-C..t-1`` (hourly); predict row ``t``."""
    n = len(X)
    refit_every = refit_every_for_freq(freq, cfg)
    y_vals = y.to_numpy(dtype=float)
    model: Any = None
    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    idx_list: list[int] = []

    if freq == "hourly":
        C = cfg.hourly_sliding_context
        t_first, t_end = hourly_eval_t_range(n, C, cfg.hourly_eval_tail)
        if t_first >= t_end:
            return (np.asarray([]), np.asarray([]), np.asarray([]), [])
        for t in range(t_first, t_end):
            start = t - C
            do_refit = (t == t_first) or ((t - t_first) % refit_every == 0)
            if do_refit or model is None:
                model = make_model()
                model.fit(X.iloc[start:t].to_numpy(), y_vals[start:t])
            pred = model.predict(X.iloc[t : t + 1].to_numpy())
            y_hat = float(pred.ravel()[0])
            y_true_list.append(float(y_vals[t]))
            y_pred_list.append(y_hat)
            idx_list.append(t)
    else:
        min_train = min_train_for_freq(freq, cfg)
        for t, do_refit in walk_forward_indices(n, min_train, refit_every):
            if do_refit or model is None:
                model = make_model()
                model.fit(X.iloc[:t].to_numpy(), y_vals[:t])
            pred = model.predict(X.iloc[t : t + 1].to_numpy())
            y_hat = float(pred.ravel()[0])
            y_true_list.append(float(y_vals[t]))
            y_pred_list.append(y_hat)
            idx_list.append(t)
    return (
        np.asarray(y_true_list),
        np.asarray(y_pred_list),
        np.asarray(idx_list),
        [],
    )


def walk_forward_arima(
    y: pd.Series,
    X: pd.DataFrame,
    freq: str,
    cfg: EvalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """ARIMA by default; optional SARIMAX with top exog. Refits each evaluated origin."""
    n = len(y)
    yv = y.to_numpy(dtype=float)

    if freq == "hourly":
        C = cfg.hourly_sliding_context
        t_first, t_end = hourly_eval_t_range(n, C, cfg.hourly_eval_tail)
        if t_first >= t_end:
            return (np.asarray([]), np.asarray([]), np.asarray([]), [])
        ex_lo = max(0, t_first - C)
        exog_cols = select_top_exog(X.iloc[ex_lo:t_first], y.iloc[ex_lo:t_first], cfg.top_exog_k)
    else:
        min_train = min_train_for_freq(freq, cfg)
        t_first, t_end = min_train, n
        exog_cols = select_top_exog(X.iloc[:min_train], y.iloc[:min_train], cfg.top_exog_k)

    Xe = X[exog_cols]

    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    idx_list: list[int] = []

    for t in range(t_first, t_end):
        if freq == "hourly":
            if (t - t_first) % cfg.ts_eval_stride != 0:
                continue
            start = t - C
        else:
            if (t - t_first) % cfg.ts_eval_stride != 0:
                continue
            start = 0

        y_train = yv[start:t]
        y_hat = float(np.nan)
        if cfg.use_sarimax_exog:
            try:
                ex_train = Xe.iloc[start:t].astype(float)
                ex_future = Xe.iloc[t : t + 1].astype(float)
                mod = SARIMAX(
                    y_train,
                    exog=ex_train,
                    order=cfg.arima_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mod.fit(disp=False, maxiter=50)
                pred = res.get_forecast(steps=1, exog=ex_future).predicted_mean
                y_hat = float(np.asarray(pred).ravel()[0])
            except Exception:
                y_hat = float(np.nan)
        if not np.isfinite(y_hat):
            try:
                mod2 = ARIMA(y_train, order=cfg.arima_order)
                res2 = mod2.fit()
                pred2 = res2.forecast(steps=1)
                y_hat = float(np.asarray(pred2).ravel()[0])
            except Exception:
                y_hat = float(np.nan)
        y_true_list.append(float(yv[t]))
        y_pred_list.append(y_hat)
        idx_list.append(t)

    return (
        np.asarray(y_true_list),
        np.asarray(y_pred_list),
        np.asarray(idx_list),
        [],
    )


def walk_forward_var_volume(
    df: pd.DataFrame,
    y_target: pd.Series,
    h: int,
    freq: str,
    cfg: EvalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    VAR on [log_ret, ret_std_20, log_volume]. Train through row ``t``; ``h``-step
    ahead forecast for ``log_volume`` vs ``target_log_vol_fwd_h`` (volume only).
    """
    required = ["log_ret", "ret_std_20", "log_volume"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"VAR volume requires column {c}")

    endog = df[required].astype(float)
    n = len(endog)
    var_floor = cfg.var_maxlags + h + 20
    yv = y_target.to_numpy(dtype=float)
    vol_idx = required.index("log_volume")

    if freq == "hourly":
        C = cfg.hourly_sliding_context
        t_first, t_end = hourly_eval_t_range(n, C, cfg.hourly_eval_tail)
        t_first = max(t_first, var_floor)
        if t_first >= t_end:
            return (
                np.asarray([]),
                np.asarray([]),
                np.asarray([]),
                [],
            )
    else:
        t_first = max(min_train_for_freq(freq, cfg), var_floor)
        t_end = n

    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    idx_list: list[int] = []

    for t in range(t_first, t_end):
        if (t - t_first) % cfg.ts_eval_stride != 0:
            continue
        if freq == "hourly":
            start = t - C
            train = endog.iloc[start : t + 1]
        else:
            train = endog.iloc[: t + 1]
        y_hat = float(np.nan)
        try:
            model = VAR(train)
            res = model.fit(maxlags=cfg.var_maxlags, ic=None)
            lag_order = res.k_ar
            y0 = train.values[-lag_order:]
            fcst = res.forecast(y0, steps=h)
            y_hat = float(fcst[h - 1, vol_idx])
        except Exception:
            y_hat = float(np.nan)
        y_true_list.append(float(yv[t]))
        y_pred_list.append(y_hat)
        idx_list.append(t)

    return (
        np.asarray(y_true_list),
        np.asarray(y_pred_list),
        np.asarray(idx_list),
        [],
    )


def walk_forward_prophet(
    timestamps: pd.Series,
    y: pd.Series,
    freq: str,
    h: int,
    cfg: EvalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    _ = h  # horizon encoded in target column; index-space forecast is 1-step
    n = len(y)
    yv = y.to_numpy(dtype=float)
    ds = _prophet_ds(timestamps)

    if freq == "hourly":
        C = cfg.hourly_sliding_context
        t_first, t_end = hourly_eval_t_range(n, C, cfg.hourly_eval_tail)
        if t_first >= t_end:
            return (np.asarray([]), np.asarray([]), np.asarray([]), [])
    else:
        t_first = min_train_for_freq(freq, cfg)
        t_end = n

    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    idx_list: list[int] = []

    for t in range(t_first, t_end):
        if (t - t_first) % cfg.ts_eval_stride != 0:
            continue
        if cfg.skip_prophet:
            y_true_list.append(float(yv[t]))
            y_pred_list.append(float("nan"))
            idx_list.append(t)
            continue
        from prophet import Prophet

        if freq == "hourly":
            start = t - C
            hist = pd.DataFrame({"ds": ds.iloc[start:t], "y": yv[start:t]})
        else:
            hist = pd.DataFrame({"ds": ds.iloc[:t], "y": yv[:t]})
        y_hat = float(np.nan)
        if np.nanstd(hist["y"].values) < 1e-12:
            y_hat = float(yv[t - 1]) if t > 0 else float(yv[t])
        else:
            try:
                m = Prophet(
                    daily_seasonality=freq == "hourly",
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                )
                m.fit(hist)
                future = pd.DataFrame({"ds": [ds.iloc[t]]})
                fcst = m.predict(future)
                y_hat = float(fcst["yhat"].iloc[0])
            except Exception:
                y_hat = float(np.nan)
        y_true_list.append(float(yv[t]))
        y_pred_list.append(y_hat)
        idx_list.append(t)

    return (
        np.asarray(y_true_list),
        np.asarray(y_pred_list),
        np.asarray(idx_list),
        [],
    )


def walk_forward_nlinear(
    timestamps: pd.Series,
    y: pd.Series,
    h: int,
    freq: str,
    cfg: EvalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    `NLinear` (NeuralForecast): linear layer over the lag window.

    Daily: same walk-forward grid as other models — expanding history ``0..t-1``, origins ``t``
    from ``min_train_daily`` through ``n-1``; ``input_size`` = ``daily_neural_context``.
    Hourly: same as other hourly models — sliding ``hourly_sliding_context`` rows, eval only on
    the last ``hourly_eval_tail`` rows.
    """
    _ = h
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NLinear
    except Exception:
        return (np.asarray([]), np.asarray([]), np.asarray([]), [])

    if cfg.skip_neural:
        return (np.asarray([]), np.asarray([]), np.asarray([]), [])

    nf_freq = "H" if freq == "hourly" else "D"
    n = len(y)
    yv = y.to_numpy(dtype=float)
    ds = _prophet_ds(timestamps)

    if freq == "hourly":
        context = cfg.hourly_sliding_context
        t_first, t_end = hourly_eval_t_range(n, context, cfg.hourly_eval_tail)
        if t_first >= t_end:
            return (np.asarray([]), np.asarray([]), np.asarray([]), [])
    else:
        context = cfg.daily_neural_context
        t_first = min_train_for_freq(freq, cfg)
        t_end = n

    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    idx_list: list[int] = []

    for t in range(t_first, t_end):
        if (t - t_first) % cfg.ts_eval_stride != 0:
            continue
        if freq == "hourly":
            start = t - context
            train_df = pd.DataFrame(
                {
                    "unique_id": "s",
                    "ds": ds.iloc[start:t].values,
                    "y": yv[start:t],
                }
            )
        else:
            train_df = pd.DataFrame(
                {
                    "unique_id": "s",
                    "ds": ds.iloc[:t].values,
                    "y": yv[:t],
                }
            )
        y_hat = float(np.nan)
        if len(train_df) >= context + 5:
            try:
                mdl = NLinear(
                    h=1,
                    input_size=context,
                    max_steps=cfg.neural_max_steps,
                    accelerator="cpu",
                    enable_progress_bar=False,
                )
                nf = NeuralForecast(models=[mdl], freq=nf_freq)
                nf.fit(df=train_df, verbose=False)
                fut = nf.predict()
                val_cols = [c for c in fut.columns if c not in ("unique_id", "ds")]
                y_hat = float(fut[val_cols[0]].iloc[0]) if val_cols else float("nan")
            except Exception:
                y_hat = float(np.nan)
        y_true_list.append(float(yv[t]))
        y_pred_list.append(y_hat)
        idx_list.append(t)

    return (
        np.asarray(y_true_list),
        np.asarray(y_pred_list),
        np.asarray(idx_list),
        [],
    )


def select_best_plot_candidate(
    candidates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Optional[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``(model_name, y_true, y_pred, idx)`` for the candidate with lowest MAE (finite preds only)."""
    best_name: Optional[str] = None
    best_m = float("inf")
    best_pack: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    for name, (yt, yp, ix) in candidates.items():
        msk = np.isfinite(yt) & np.isfinite(yp)
        if not np.any(msk):
            continue
        err = mae(yt[msk], yp[msk])
        if err < best_m:
            best_m = err
            best_name = name
            best_pack = (yt, yp, ix)
    if best_name is None or best_pack is None:
        return None
    yt, yp, ix = best_pack
    return (best_name, yt, yp, ix)


def compute_metrics_row(
    model: str,
    dataset: str,
    family: str,
    horizon: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx: np.ndarray,
    y_full: np.ndarray,
    context: str = "",
) -> dict[str, Any]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return {
            "Model": model,
            "Dataset": dataset,
            "Target": family,
            "Horizon": horizon,
            "Context": context,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MASE": np.nan,
        }
    yt = y_true[m]
    yp = y_pred[m]
    ix = idx[m]
    return {
        "Model": model,
        "Dataset": dataset,
        "Target": family,
        "Horizon": horizon,
        "Context": context,
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
        "MASE": mase(yt, yp, y_full, horizon, ix),
    }


def plot_actual_vs_predicted_timeseries(
    ts: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx: np.ndarray,
    title: str,
    out_path: Path,
    y_label: str = "Target",
) -> None:
    """Line chart of actual vs walk-forward predictions over ``timestamp``."""
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return
    yt, yp, ix = y_true[m], y_pred[m], idx[m]
    times = pd.to_datetime(ts.iloc[ix], utc=True)
    plt.figure(figsize=(12, 4.5))
    plt.plot(times, yt, label="Actual", color="#1f77b4", lw=1.2, alpha=0.95)
    plt.plot(times, yp, label="Predicted", color="#ff7f0e", lw=1.2, alpha=0.9)
    plt.legend(loc="best")
    plt.title(title)
    plt.xlabel("Time (UTC)")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_actual_vs_predicted_stock(
    ts: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx: np.ndarray,
    title: str,
    out_path: Path,
    y_label: str = "Target",
) -> None:
    """Time-series line chart (stock-style): actual vs walk-forward predictions with light error shading."""
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return
    yt, yp, ix = y_true[m], y_pred[m], idx[m]
    times = pd.to_datetime(ts.iloc[ix], utc=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, yt, label="Actual", color="#1b5e20", lw=1.15, zorder=2)
    ax.plot(times, yp, label="Predicted", color="#b71c1c", lw=1.1, alpha=0.92, zorder=2)
    try:
        ax.fill_between(times, yt, yp, color="#5c6bc0", alpha=0.12, zorder=1)
    except (TypeError, ValueError):
        pass
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.legend(loc="best", framealpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.28, linestyle="-", linewidth=0.5)
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")
    fig.autofmt_xdate()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_diagnostics(
    ts: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return
    yt, yp, ix = y_true[m], y_pred[m], idx[m]
    times = ts.iloc[ix].values
    err = yt - yp

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax = axes[0, 0]
    ax.scatter(yt, yp, s=8, alpha=0.5)
    lims = [np.nanmin(yt), np.nanmax(yt)]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Prediction vs actual")

    axes[0, 1].plot(times, err, lw=0.8)
    axes[0, 1].set_title("Error over time")
    axes[0, 1].set_xlabel("Time")

    sns.histplot(err, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Residual distribution")

    axes[1, 1].axis("off")
    axes[1, 1].text(0, 0.5, title, fontsize=10, wrap=True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def correlation_heatmap(df: pd.DataFrame, X_cols: list[str], y_cols: list[str], out_path: Path) -> None:
    """
    Pairwise correlation heatmap. ``X_cols`` order is usually OHLCV-engineered columns first,
    then blockchain/macro at the end of merged files — so we **prioritize** ``macro_*`` and
    ``*_bc_*`` columns instead of only the first N names.
    """
    avail = [c for c in X_cols if c in df.columns]
    macro_chain = [c for c in avail if c.startswith("macro_") or "_bc_" in c]
    rest = [c for c in avail if c not in macro_chain]
    max_features = 56
    budget = max(0, max_features - len(macro_chain))
    cols = macro_chain + rest[:budget]
    cols = cols + [c for c in y_cols if c in df.columns]
    sub_df = df[cols]
    std = sub_df.std(numeric_only=True)
    cols_ok = std[std > 1e-12].index.tolist()
    if len(cols_ok) < 2:
        return
    sub = sub_df[cols_ok].corr(numeric_only=True)
    plt.figure(figsize=(14, 12))
    sns.heatmap(sub, cmap="vlag", center=0, linewidths=0.2)
    plt.title("Correlation: OHLCV features + macro/chain (prioritized) + targets (sample)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def run_all(cfg: EvalConfig) -> None:
    tables_ret: list[dict[str, Any]] = []
    tables_vol: list[dict[str, Any]] = []
    tables_volume: list[dict[str, Any]] = []

    fig_dir = cfg.results_dir / "figures"
    tab_dir = cfg.results_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(cfg.merge_dir.glob("*_merged.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_merged.csv in {cfg.merge_dir}")
    if cfg.datasets:
        allow = set(cfg.datasets)
        paths = [p for p in paths if p.name in allow]
        if not paths:
            raise FileNotFoundError(f"No matching files for datasets={cfg.datasets}")

    print(
        "Starting evaluation — progress prints below (nothing until the end means a long step is running).",
        flush=True,
    )
    print(
        f"  merge_dir={cfg.merge_dir}  |  files={len(paths)}  |  ts_eval_stride={cfg.ts_eval_stride}  |  "
        f"skip_prophet={cfg.skip_prophet}  skip_neural={cfg.skip_neural}",
        flush=True,
    )
    print(
        f"  hourly: sliding_context={cfg.hourly_sliding_context}  eval_tail={cfg.hourly_eval_tail}  |  "
        f"daily_neural_context={cfg.daily_neural_context}",
        flush=True,
    )

    plot_heatmap_stems: set[str] = set()
    xgb_importance_plotted_stems: set[str] = set()

    for path in paths:
        df, freq, stem = load_dataset(path)
        if freq == "hourly" and cfg.max_rows_hourly is not None:
            df = df.iloc[: cfg.max_rows_hourly].copy()
        if freq == "daily" and cfg.max_rows_daily is not None:
            df = df.iloc[: cfg.max_rows_daily].copy()
        X, y_map = build_xy(df, freq)
        if cfg.target_pairs_by_freq is not None:
            if freq not in cfg.target_pairs_by_freq:
                raise KeyError(
                    f"target_pairs_by_freq has no key {freq!r}. Keys: {list(cfg.target_pairs_by_freq)}"
                )
            want = list(cfg.target_pairs_by_freq[freq])
        elif cfg.target_pairs is not None:
            want = list(cfg.target_pairs)
        else:
            want = None
        if want is not None:
            missing = [k for k in want if k not in y_map]
            if missing:
                raise KeyError(
                    f"target selection not in dataset y_map: {missing}. Available: {sorted(y_map)}"
                )
            y_map = {k: y_map[k] for k in want}
        ts = df["timestamp"]
        y_full_by_col: dict[str, np.ndarray] = {y_map[k]: df[y_map[k]].to_numpy(float) for k in y_map}

        print(
            f"\n>>> Dataset: {path.name}  ({freq}, n={len(df)} rows, {X.shape[1]} features)",
            flush=True,
        )
        if freq == "hourly":
            ht0, ht1 = hourly_eval_t_range(len(df), cfg.hourly_sliding_context, cfg.hourly_eval_tail)
            if ht0 < ht1:
                print(f"    hourly walk-forward origins: t in [{ht0}, {ht1})  ({ht1 - ht0} steps)", flush=True)
            else:
                print(
                    "    hourly walk-forward: no valid origins (increase rows or lower "
                    "--hourly-sliding-context / --hourly-eval-tail).",
                    flush=True,
                )

        for (family, h), ycol in y_map.items():
            y = df[ycol].astype(float)
            y_full = y_full_by_col[ycol]
            plot_candidates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

            print(f"    target={family} horizon={h} …", flush=True)

            yt, yp, ix, _ = walk_forward_sklearn(
                X,
                y,
                freq,
                cfg,
                lambda: RandomForestRegressor(
                    n_estimators=cfg.rf_n_estimators,
                    n_jobs=-1,
                    random_state=42,
                ),
                "RF",
            )
            if should_emit_figures(cfg, path.name):
                plot_candidates["RandomForest"] = (yt, yp, ix)
            row = compute_metrics_row("RandomForest", stem, family, h, yt, yp, ix, y_full)
            (tables_ret if family == "return" else tables_vol if family == "volatility" else tables_volume).append(
                row
            )

            yt, yp, ix, _ = walk_forward_sklearn(
                X,
                y,
                freq,
                cfg,
                lambda: XGBRegressor(
                    n_estimators=cfg.xgb_n_estimators,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                ),
                "XGB",
            )
            if should_emit_figures(cfg, path.name):
                plot_candidates["XGBoost"] = (yt, yp, ix)
            row = compute_metrics_row("XGBoost", stem, family, h, yt, yp, ix, y_full)
            (tables_ret if family == "return" else tables_vol if family == "volatility" else tables_volume).append(
                row
            )

            yt, yp, ix, _ = walk_forward_arima(y, X, freq, cfg)
            if should_emit_figures(cfg, path.name):
                plot_candidates["ARIMAX"] = (yt, yp, ix)
            row = compute_metrics_row("ARIMAX", stem, family, h, yt, yp, ix, y_full)
            (tables_ret if family == "return" else tables_vol if family == "volatility" else tables_volume).append(
                row
            )

            if family == "volume":
                yt, yp, ix, _ = walk_forward_var_volume(df, y, h, freq, cfg)
                if len(yt) > 0:
                    if should_emit_figures(cfg, path.name):
                        plot_candidates["VAR_volume"] = (yt, yp, ix)
                    row = compute_metrics_row("VAR_volume", stem, family, h, yt, yp, ix, y_full)
                    tables_volume.append(row)

            if not cfg.skip_prophet:
                yt, yp, ix, _ = walk_forward_prophet(ts, y, freq, h, cfg)
                if should_emit_figures(cfg, path.name):
                    plot_candidates["Prophet"] = (yt, yp, ix)
                row = compute_metrics_row("Prophet", stem, family, h, yt, yp, ix, y_full)
                (tables_ret if family == "return" else tables_vol if family == "volatility" else tables_volume).append(
                    row
                )

            if not cfg.skip_neural:
                nlin_ctx = cfg.hourly_sliding_context if freq == "hourly" else cfg.daily_neural_context
                yt, yp, ix, _ = walk_forward_nlinear(ts, y, h, freq, cfg)
                if len(yt) > 0:
                    if should_emit_figures(cfg, path.name):
                        plot_candidates["NLinear"] = (yt, yp, ix)
                    row = compute_metrics_row(
                        "NLinear",
                        stem,
                        family,
                        h,
                        yt,
                        yp,
                        ix,
                        y_full,
                        context=str(nlin_ctx),
                    )
                    (tables_ret if family == "return" else tables_vol if family == "volatility" else tables_volume).append(
                        row
                    )

            if should_emit_figures(cfg, path.name):
                try:
                    if stem not in plot_heatmap_stems:
                        ycols_plot = heatmap_target_columns(y_map)
                        correlation_heatmap(
                            df,
                            list(X.columns),
                            ycols_plot,
                            fig_dir / f"correlation_heatmap_{stem}.png",
                        )
                        plot_heatmap_stems.add(stem)

                    picked = select_best_plot_candidate(plot_candidates)
                    if picked is not None:
                        best_name, yt_b, yp_b, ix_b = picked
                        safe_best = best_name.lower().replace(" ", "_")
                        plot_actual_vs_predicted_stock(
                            ts,
                            yt_b,
                            yp_b,
                            ix_b,
                            f"Best model ({best_name}) — {stem} — {family} horizon {h}",
                            fig_dir / f"best_actual_vs_predicted_{safe_best}_{family}_h{h}_{stem}.png",
                            y_label=ycol,
                        )
                        plot_diagnostics(
                            ts,
                            yt_b,
                            yp_b,
                            ix_b,
                            f"Best: {best_name} — {stem} — {family} h={h}",
                            fig_dir / f"best_diag_{family}_h{h}_{stem}.png",
                        )

                    if (
                        stem not in xgb_importance_plotted_stems
                        and family == "return"
                        and h == min_horizon_for_family(y_map, "return")
                    ):
                        xgb_pick = select_best_plot_candidate(plot_candidates)
                        if xgb_pick is not None and xgb_pick[0] == "XGBoost":
                            m_xgb = XGBRegressor(
                                n_estimators=cfg.xgb_n_estimators,
                                max_depth=6,
                                learning_rate=0.05,
                                random_state=42,
                                n_jobs=-1,
                                verbosity=0,
                            )
                            min_tr = min_train_for_freq(freq, cfg)
                            hr0 = min_horizon_for_family(y_map, "return")
                            y_ret = df[y_map[("return", hr0)]].astype(float)
                            m_xgb.fit(X.iloc[:min_tr].to_numpy(), y_ret.iloc[:min_tr].to_numpy())
                            imp = m_xgb.feature_importances_
                            imp_df = pd.DataFrame({"feature": X.columns, "importance": imp}).sort_values(
                                "importance", ascending=False
                            ).head(25)
                            plt.figure(figsize=(8, 7))
                            sns.barplot(data=imp_df, y="feature", x="importance")
                            plt.title(
                                "XGBoost feature importance (best model for return h=1; fit on initial train window)"
                            )
                            plt.tight_layout()
                            plt.savefig(fig_dir / f"xgb_feature_importance_{stem}.png", dpi=120)
                            plt.close()
                            xgb_importance_plotted_stems.add(stem)
                except Exception:
                    pass

            print(f"    done: {family} horizon={h}", flush=True)

    pd.DataFrame(tables_ret).to_csv(tab_dir / "metrics_returns.csv", index=False)
    pd.DataFrame(tables_vol).to_csv(tab_dir / "metrics_volatility.csv", index=False)
    pd.DataFrame(tables_volume).to_csv(tab_dir / "metrics_volume.csv", index=False)
    print(
        "Wrote tables:",
        tab_dir / "metrics_returns.csv",
        tab_dir / "metrics_volatility.csv",
        tab_dir / "metrics_volume.csv",
    )
    print("Figures (if generated):", fig_dir)


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Walk-forward model evaluation on merged CSVs.")
    p.add_argument("--merge-dir", type=Path, default=_DEFAULT_MERGE)
    p.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS)
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of merged CSV filenames (e.g. bitcoin_ohlcv_daily_merged.csv). Default: all.",
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
        help="Evaluate ARIMA/Prophet/VAR/NLinear every k-th origin (1=all). Hourly stride is vs first tail origin. "
        "Sklearn predicts every step in the evaluated range (daily: from min_train; hourly: last tail rows).",
    )
    p.add_argument(
        "--neural-max-steps",
        type=int,
        default=15,
        help="Training steps for NLinear (NeuralForecast) per refit; lower is faster.",
    )
    p.add_argument(
        "--use-sarimax-exog",
        action="store_true",
        help="Try SARIMAX with top correlated exog before ARIMA (slower).",
    )
    p.add_argument(
        "--plot-dataset",
        action="append",
        default=None,
        metavar="FILE",
        dest="plot_datasets",
        help="Only write figures for this merged CSV filename (repeat option for several). "
        "Default: write figures for every dataset in the run.",
    )
    p.add_argument(
        "--hourly-sliding-context",
        type=int,
        default=3000,
        help="Hourly: training rows ending at t-1 (sliding window length).",
    )
    p.add_argument(
        "--hourly-eval-tail",
        type=int,
        default=1000,
        help="Hourly: only evaluate walk-forward on the last this many rows.",
    )
    p.add_argument(
        "--daily-neural-context",
        type=int,
        default=90,
        help="Daily NLinear input_size (expanding-window training).",
    )
    args = p.parse_args()
    return EvalConfig(
        merge_dir=args.merge_dir,
        results_dir=args.results_dir,
        datasets=args.datasets,
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
        target_pairs_by_freq=None,
        neural_max_steps=args.neural_max_steps,
        daily_neural_context=args.daily_neural_context,
    )


if __name__ == "__main__":
    _t0 = time.perf_counter()
    print(f"Command: {shlex.join(sys.argv)}", flush=True)
    _cfg = parse_args()
    try:
        run_all(_cfg)
    finally:
        _elapsed = time.perf_counter() - _t0
        print(f"Total wall time: {_elapsed:.2f}s ({_elapsed / 60.0:.2f} min)", flush=True)
