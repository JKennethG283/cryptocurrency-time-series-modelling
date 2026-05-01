"""Data fusion and repo layout invariants."""

from __future__ import annotations

from pathlib import Path

import data_preparation
import merge


def test_data_preparation_default_output_is_repo_local() -> None:
    out = data_preparation._DEFAULT_OUTPUT_DIR
    assert out.name == "pre-processing"
    assert out.parent.name == "data"
    mod = Path(data_preparation.__file__).resolve().parent
    assert out == mod / "data" / "pre-processing"


def test_merge_does_not_backward_fill_macro() -> None:
    """``bfill`` on macro/merged rows would inject future information into past rows."""
    src = Path(merge.__file__).read_text(encoding="utf-8")
    assert "bfill" not in src
