"""
Merge pre-processed OHLCV ML CSVs with on-chain metrics and macro time series.

For each asset (bitcoin, ethereum, solana) and frequency (daily, hourly), loads the
corresponding ``*_ml.csv``, optionally merges blockchain data (BTC/ETH only),
merges all macro files for that frequency, **forward-fills** macro gaps (as-of: last
known value carried forward; no **backward** fill, which would use future data),
and writes ``*_merged.csv`` under ``data/merge/``.

Merge keys are UTC timestamps normalized to midnight (daily) or hour start (hourly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

_ROOT = Path(__file__).resolve().parent
_PREPROC = _ROOT / "data" / "pre-processing"
_BLOCKCHAIN = _ROOT / "data" / "block chain"
_MACRO = _ROOT / "data" / "macro"
_MERGE_OUT = _ROOT / "data" / "merge"

_MERGE_TS = "_merge_ts"

# asset -> (daily_csv, hourly_csv) relative to block chain folder; None = skip chain
_BLOCKCHAIN_FILES: dict[str, tuple[Optional[str], Optional[str]]] = {
    "bitcoin": ("btc_blockchain_d.csv", "btc_blockchain_h.csv"),
    "ethereum": ("eth_blockchain_d.csv", "eth_blockchain_h.csv"),
    "solana": (None, None),
}

_ML_GLOB = "*_ohlcv_*_ml.csv"


def _normalize_merge_key(series: pd.Series, freq: str) -> pd.Series:
    dt = pd.to_datetime(series, utc=True)
    if freq == "daily":
        return dt.dt.normalize()
    return dt.dt.floor("h")


def _time_col_blockchain(df: pd.DataFrame) -> str:
    if "day" in df.columns:
        return "day"
    if "hour" in df.columns:
        return "hour"
    raise ValueError(f"Blockchain CSV must have 'day' or 'hour' column; got {list(df.columns)}")


def _prefix_blockchain_columns(df: pd.DataFrame, prefix: str, time_col: str) -> pd.DataFrame:
    out = df.copy()
    skip = {time_col, _MERGE_TS}
    rename = {c: f"{prefix}{c}" for c in out.columns if c not in skip}
    return out.rename(columns=rename)


def _drop_uninformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove numeric columns that are all-NaN or constant (zero variance).
    They carry no signal for supervised models and often come from bad macro joins.
    """
    drop: list[str] = []
    for c in df.columns:
        if c == "timestamp":
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if s.isna().all():
            drop.append(c)
            continue
        if s.notna().any() and s.nunique(dropna=True) <= 1:
            drop.append(c)
    if not drop:
        return df
    return df.drop(columns=drop)


def _macro_stem_to_column(stem: str, freq: str) -> str:
    suffix = "_daily" if freq == "daily" else "_hourly"
    base = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    return f"macro_{base}"


def _load_macro_panel(freq: str) -> pd.DataFrame:
    """One row per merge timestamp, one column per macro series (``macro_*``)."""
    suffix = "_daily.csv" if freq == "daily" else "_hourly.csv"
    paths = sorted(_MACRO.glob(f"*{suffix}"))
    if not paths:
        raise FileNotFoundError(f"No macro files matching *{suffix} in {_MACRO}")

    panel: Optional[pd.DataFrame] = None
    for path in paths:
        stem = path.stem
        col = _macro_stem_to_column(stem, freq)
        m = pd.read_csv(path)
        if not {"timestamp", "value"}.issubset(m.columns):
            raise ValueError(f"{path.name} must have columns timestamp, value")
        m[_MERGE_TS] = _normalize_merge_key(m["timestamp"], freq)
        m = m[[_MERGE_TS, "value"]].rename(columns={"value": col})
        if panel is None:
            panel = m
        else:
            panel = panel.merge(m, on=_MERGE_TS, how="outer")
    assert panel is not None
    panel = panel.sort_values(_MERGE_TS).reset_index(drop=True)
    macro_cols = [c for c in panel.columns if c != _MERGE_TS]
    # Causal: only past/known values propagate forward, never from the future.
    panel[macro_cols] = panel[macro_cols].ffill()
    return panel


def _infer_asset_and_freq(ml_name: str) -> tuple[str, str]:
    """e.g. bitcoin_ohlcv_daily_ml.csv -> ('bitcoin', 'daily')."""
    stem = Path(ml_name).stem
    if not stem.endswith("_ml"):
        raise ValueError(f"Unexpected ML filename: {ml_name}")
    body = stem[: -len("_ml")]
    if body.endswith("_daily"):
        return body[: -len("_daily")].replace("_ohlcv", ""), "daily"
    if body.endswith("_hourly"):
        return body[: -len("_hourly")].replace("_ohlcv", ""), "hourly"
    raise ValueError(f"Cannot infer freq from {ml_name}")


def _blockchain_prefix(asset: str) -> str:
    if asset == "bitcoin":
        return "btc_bc_"
    if asset == "ethereum":
        return "eth_bc_"
    raise ValueError(asset)


def merge_one_ml_csv(ml_path: Path, output_dir: Path) -> Path:
    """
    Merge a single ``*_ml.csv`` with blockchain (if configured) and macro panel.

    Returns path to the written CSV.
    """
    asset, freq = _infer_asset_and_freq(ml_path.name)
    base = pd.read_csv(ml_path)
    if "timestamp" not in base.columns:
        raise ValueError(f"{ml_path} must contain a timestamp column")

    base[_MERGE_TS] = _normalize_merge_key(base["timestamp"], freq)
    out = base

    bc_tuple = _BLOCKCHAIN_FILES[asset]
    bc_file = bc_tuple[0] if freq == "daily" else bc_tuple[1]
    if bc_file is not None:
        bc_path = _BLOCKCHAIN / bc_file
        if not bc_path.is_file():
            raise FileNotFoundError(f"Missing blockchain file: {bc_path}")
        bc = pd.read_csv(bc_path)
        tcol = _time_col_blockchain(bc)
        bc[_MERGE_TS] = _normalize_merge_key(bc[tcol], freq)
        bc = _prefix_blockchain_columns(bc, _blockchain_prefix(asset), tcol)
        bc = bc.drop(columns=[tcol])
        out = out.merge(bc, on=_MERGE_TS, how="inner")

    macro = _load_macro_panel(freq)
    out = out.merge(macro, on=_MERGE_TS, how="left")

    macro_cols = [c for c in macro.columns if c != _MERGE_TS]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp", ascending=True).reset_index(drop=True)
    out[macro_cols] = out[macro_cols].ffill()

    out = out.drop(columns=[_MERGE_TS])
    ncols_before = out.shape[1]
    out = _drop_uninformative_columns(out)
    dropped = ncols_before - out.shape[1]
    if dropped:
        print(f"  Dropped {dropped} all-NaN or constant column(s) → {ml_path.name}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ml_path.stem.replace('_ml', '')}_merged.csv"
    out.to_csv(out_path, index=False)
    return out_path


def merge_all(
    preproc_dir: Path = _PREPROC,
    output_dir: Path = _MERGE_OUT,
) -> list[Path]:
    """Merge every ``*_ohlcv_*_ml.csv`` in ``preproc_dir``; return written paths."""
    paths = sorted(preproc_dir.glob(_ML_GLOB))
    if not paths:
        raise FileNotFoundError(f"No ML CSVs matching {_ML_GLOB} in {preproc_dir}")
    written: list[Path] = []
    for p in paths:
        written.append(merge_one_ml_csv(p, output_dir))
    return written


if __name__ == "__main__":
    done = merge_all()
    for p in done:
        df = pd.read_csv(p, nrows=0)
        print(f"{p.name}: columns={len(df.columns)}")
