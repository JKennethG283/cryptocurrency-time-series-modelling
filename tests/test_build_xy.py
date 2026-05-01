"""Train/test split: labels must not appear in feature matrix X."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark import (
    DAILY_HORIZONS,
    HOURLY_HORIZONS,
    TARGET_PREFIX,
    build_xy,
    horizons_for_freq,
)


def _synthetic_merged_frame(freq: str, n: int = 64) -> pd.DataFrame:
    """
    Build a minimal valid frame for ``build_xy`` for the given ``freq``,
    with numeric features and all target columns required by ``benchmark.horizons_for_freq``.
    """
    hs = horizons_for_freq(freq)
    if freq == "hourly":
        idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    else:
        idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "feature_a": np.random.default_rng(42).normal(size=n),
            "feature_b": np.random.default_rng(43).normal(size=n),
        }
    )
    for family, tmpl in TARGET_PREFIX.items():
        for h in hs:
            if family == "volatility" and h == 1:
                continue
            col = tmpl.format(h=h)
            df[col] = np.random.default_rng(44).normal(size=n)
    return df


@pytest.mark.parametrize("freq", ["hourly", "daily"])
def test_build_xy_excludes_all_target_prefix_columns(freq: str) -> None:
    df = _synthetic_merged_frame(freq, n=80)
    X, y_map = build_xy(df, freq)
    assert not any(c.startswith("target_") for c in X.columns)
    assert "timestamp" not in X.columns
    for col in y_map.values():
        assert col in df.columns
    assert set(y_map.values()).isdisjoint(X.columns)


def test_horizon_sets_match_data_preparation_overlap() -> None:
    """Benchmark grid must stay in sync with preprocessed files (see data_preparation horizons)."""
    for h in HOURLY_HORIZONS:
        assert 1 <= h <= 24
    for h in DAILY_HORIZONS:
        assert 1 <= h <= 30
