"""
OHLCV → machine-learning dataset preparation.

Transforms sorted crypto OHLCV data into vectorized features and multi-horizon
targets without lookahead. Assumes timestamps are UTC and rows are unique in time.

Typical warm-up: roughly max(50 EMA, 14 RSI/ATR, 20 Bollinger/volume MA, five return lags)
plus target horizon at the tail; final ``dropna`` removes incomplete rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# Small floor for log(volume) and ratios to avoid log(0).
_EPS: float = 1e-12

# Log-return lags: keep short memory only (lags 6–10 overlap heavily with rolling stats).
_N_RET_LAGS: int = 5

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "pre-processing"

_DEFAULT_OHLCV_DIR = Path(__file__).resolve().parent / "data" / "ohlcv"

_HOURLY_HORIZONS: tuple[int, ...] = (1, 3, 6, 12, 24)
_DAILY_HORIZONS: tuple[int, ...] = (1, 3, 7, 14)


def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with a monotonic DatetimeIndex named ``timestamp``.

    Accepts ``timestamp`` or ``timestamp_utc`` as a column, or an existing
    datetime-like index. Validates required OHLCV columns.
    """
    required = {"open", "high", "low", "close", "volume"}
    out = df.copy()

    if isinstance(out.index, pd.DatetimeIndex):
        idx = out.index
        if not idx.name:
            out.index.name = "timestamp"
    else:
        ts_col = None
        for name in ("timestamp", "timestamp_utc"):
            if name in out.columns:
                ts_col = name
                break
        if ts_col is None:
            raise ValueError(
                "DataFrame must have a DatetimeIndex or a column named "
                "'timestamp' or 'timestamp_utc'."
            )
        out[ts_col] = pd.to_datetime(out[ts_col], utc=True)
        out = out.set_index(ts_col).sort_index()
        out.index.name = "timestamp"

    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    out = out.sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    dup = out.index.duplicated()
    if dup.any():
        out = out[~dup]

    return out


def _forward_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling std over the next ``window`` observations (no lookahead in features).

    At index ``t`` uses values ``series[t], series[t+1], ..., series[t+window-1]``.
    """
    rev = series.iloc[::-1]
    # ddof=0 so horizon h=1 is the degenerate std of one future return (0), not NaN.
    rolled = rev.rolling(window=window, min_periods=window).std(ddof=0)
    return pd.Series(rolled.to_numpy()[::-1], index=series.index, name=series.name)


def _horizons_for_freq(freq: str) -> tuple[int, ...]:
    if freq == "hourly":
        return _HOURLY_HORIZONS
    if freq == "daily":
        return _DAILY_HORIZONS
    raise ValueError("freq must be 'hourly' or 'daily'.")


def _infer_freq_from_ohlcv_path(path: Path) -> str:
    """Return ``'hourly'`` or ``'daily'`` from names like ``bitcoin_ohlcv_daily.csv``."""
    stem = path.stem.lower()
    if stem.endswith("_hourly"):
        return "hourly"
    if stem.endswith("_daily"):
        return "daily"
    raise ValueError(
        f"Cannot infer freq from {path.name!r}; expected stem ending with '_hourly' or '_daily'."
    )


def compute_features(df: pd.DataFrame, freq: str = "hourly") -> pd.DataFrame:
    """
    Build feature columns from normalized OHLCV (past and current data only).

    Parameters
    ----------
    df
        Output of ``_normalize_input`` with OHLCV columns.
    freq
        ``'hourly'`` adds ``hour``; ``'daily'`` omits it (constant intraday clock).

    Returns
    -------
    pd.DataFrame
        Feature columns aligned to ``df`` index. Max lookback about 50 bars
        (EMA 50) plus indicator windows; ``_N_RET_LAGS`` return lags (default 5).
    """
    if freq not in ("hourly", "daily"):
        raise ValueError("freq must be 'hourly' or 'daily'.")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    feats: dict[str, pd.Series] = {}

    log_ret = np.log(close / close.shift(1))
    feats["log_ret"] = log_ret
    for k in range(1, _N_RET_LAGS + 1):
        feats[f"ret_lag_{k}"] = log_ret.shift(k)

    for span in (10, 20, 50):
        ema = close.ewm(span=span, adjust=False).mean()
        feats[f"ema_{span}"] = ema
        feats[f"ema_dist_{span}"] = (close - ema) / ema

    rsi = RSIIndicator(close=close, window=14).rsi()
    feats["rsi_14"] = rsi

    for n in (5, 10):
        feats[f"roc_{n}"] = close / close.shift(n) - 1.0

    for n in (10, 20):
        feats[f"momentum_{n}"] = close - close.shift(n)

    for w in (10, 20):
        feats[f"ret_std_{w}"] = log_ret.rolling(window=w, min_periods=w).std()

    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    feats["atr_14"] = atr.average_true_range()

    feats["hl_range"] = (high - low) / close

    bb = BollingerBands(close=close, window=20, window_dev=2)
    mid = bb.bollinger_mavg()
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    feats["bb_width"] = (upper - lower) / mid.replace(0, np.nan)

    vol_safe = volume.clip(lower=_EPS)
    log_volume = np.log(vol_safe)
    feats["log_volume"] = log_volume

    for w in (10, 20):
        ma = volume.rolling(window=w, min_periods=w).mean()
        feats[f"vol_ma_{w}"] = ma
        feats[f"vol_spike_{w}"] = volume / ma.replace(0, np.nan)

    # Log-volume scales with typical volume dynamics; raw V * return is dominated by scale.
    feats["ret_x_log_vol"] = log_ret * log_volume

    for w in (10, 20):
        feats[f"ret_mean_{w}"] = log_ret.rolling(window=w, min_periods=w).mean()

    idx = df.index
    feats["day_of_week"] = pd.Series(idx.dayofweek, index=idx, dtype="float64")
    if freq == "hourly":
        feats["hour"] = pd.Series(idx.hour, index=idx, dtype="float64")

    ret_std_10 = feats["ret_std_10"]
    for lag in (1, 2, 3):
        feats[f"rsi_lag_{lag}"] = rsi.shift(lag)
        feats[f"ret_std_10_lag_{lag}"] = ret_std_10.shift(lag)
        feats[f"log_volume_lag_{lag}"] = log_volume.shift(lag)

    return pd.DataFrame(feats, index=df.index)


def compute_targets(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """
    Multi-horizon targets using only future information (labels at time t).

    For each horizon ``h``:

    - ``target_ret_fwd_{h}``: log(close_{t+h} / close_t)
    - ``target_vol_fwd_{h}`` (for ``h > 1`` only): std of one-period log returns from t+1 through t+h.
      **Omitted for h=1** — std of a single future return is always 0 (not informative).
    - ``target_log_vol_fwd_{h}``: log(volume_{t+h})
    - ``target_vol_ratio_fwd_{h}``: volume_{t+h} / 20-bar volume MA at t+h
    """
    close = df["close"]
    volume = df["volume"]
    log_ret = np.log(close / close.shift(1))

    vol_ma20 = volume.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / vol_ma20.replace(0, np.nan)
    log_vol = np.log(volume.clip(lower=_EPS))

    r_lead = log_ret.shift(-1)
    tgt: dict[str, pd.Series] = {}

    for h in horizons:
        tgt[f"target_ret_fwd_{h}"] = np.log(close.shift(-h) / close)
        if h > 1:
            tgt[f"target_vol_fwd_{h}"] = _forward_rolling_std(r_lead, h)
        tgt[f"target_log_vol_fwd_{h}"] = log_vol.shift(-h)
        tgt[f"target_vol_ratio_fwd_{h}"] = vol_ratio.shift(-h)

    return pd.DataFrame(tgt, index=df.index)


def build_dataset(
    df: pd.DataFrame,
    freq: str = "hourly",
    output_path: Optional[str | Path] = None,
    output_stem: Optional[str] = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    End-to-end dataset: normalize, features, targets, align, drop NaNs, save CSV.

    Parameters
    ----------
    df
        Raw OHLCV DataFrame or Series-compatible structure.
    freq
        ``'hourly'`` uses horizons (1,3,6,12,24); ``'daily'`` uses (1,3,7,14).
    output_path
        Full file path for CSV. If None, uses ``data/pre-processing/`` under this repo
        and ``{stem}_{freq}_ml.csv`` (see ``_DEFAULT_OUTPUT_DIR``).
    output_stem
        Base name when ``output_path`` is None; default ``'dataset'``.
    save_csv
        If False, only return the DataFrame without writing.

    Returns
    -------
    pd.DataFrame
        Combined features and targets, no NaNs, DatetimeIndex ``timestamp``.
    """
    normalized = _normalize_input(df)
    horizons = _horizons_for_freq(freq)

    features = compute_features(normalized, freq=freq)
    targets = compute_targets(normalized, horizons)

    out = pd.concat([features, targets], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    if save_csv:
        path = Path(output_path) if output_path is not None else None
        if path is None:
            stem = output_stem or "dataset"
            _DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            path = _DEFAULT_OUTPUT_DIR / f"{stem}_{freq}_ml.csv"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        out.reset_index().to_csv(path, index=False)

    return out


def build_all_ohlcv_datasets(
    ohlcv_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> list[tuple[Path, tuple[int, int]]]:
    """
    Build and save one ML-ready CSV per ``*.csv`` in ``ohlcv_dir``.

    Each output file is named ``{source_stem}_ml.csv`` (e.g. ``bitcoin_ohlcv_daily_ml.csv``)
    under ``output_dir`` (default: ``_DEFAULT_OUTPUT_DIR``). Frequency is inferred from
    the filename (``*_daily.csv`` vs ``*_hourly.csv``).

    Parameters
    ----------
    ohlcv_dir
        Folder containing OHLCV CSVs. Default: ``data/ohlcv`` next to this module.
    output_dir
        Destination folder. Default: ``_DEFAULT_OUTPUT_DIR``.

    Returns
    -------
    list[tuple[Path, tuple[int, int]]]
        Written paths and ``(n_rows, n_columns)`` for each dataset.
    """
    src = Path(ohlcv_dir) if ohlcv_dir is not None else _DEFAULT_OHLCV_DIR
    dst = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    dst.mkdir(parents=True, exist_ok=True)

    results: list[tuple[Path, tuple[int, int]]] = []
    for csv_path in sorted(src.glob("*.csv")):
        freq = _infer_freq_from_ohlcv_path(csv_path)
        raw = pd.read_csv(csv_path)
        out_path = dst / f"{csv_path.stem}_ml.csv"
        built = build_dataset(raw, freq=freq, output_path=out_path, save_csv=True)
        results.append((out_path, built.shape))
    return results


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent
    _sample = _root / "data" / "ohlcv" / "bitcoin_ohlcv_daily.csv"
    if _sample.is_file():
        _raw = pd.read_csv(_sample)
        _built = build_dataset(_raw, freq="daily", output_stem="bitcoin_smoke")
        assert _built.isna().sum().sum() == 0
        assert _built.index.is_monotonic_increasing
        _c0 = _raw.sort_values(
            "timestamp_utc" if "timestamp_utc" in _raw.columns else "timestamp"
        ).reset_index(drop=True)
        _ts = pd.to_datetime(_c0["timestamp_utc"], utc=True)
        _i = _built.index[0]
        _row = _c0.loc[_ts == _i].index[0]
        _close_i = float(_c0.loc[_row, "close"])
        _close_next = float(_c0.loc[_row + 1, "close"])
        _expected = np.log(_close_next / _close_i)
        _got = _built.loc[_i, "target_ret_fwd_1"]
        assert np.isclose(_got, _expected, rtol=1e-9), (_got, _expected)
        print(f"Smoke OK: shape={_built.shape}, index[0]={_built.index[0]}")

    _written = build_all_ohlcv_datasets()
    for _path, _shape in _written:
        print(f"Wrote {_path.name}: rows={_shape[0]}, cols={_shape[1]}")
