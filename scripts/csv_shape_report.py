"""Print row/column counts for CSVs under data/. Run: python scripts/csv_shape_report.py"""
from __future__ import annotations

import csv
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"


def main() -> None:
    rows: list[tuple[str, int, int]] = []
    for p in sorted(_DATA.rglob("*.csv")):
        rel = p.relative_to(_ROOT)
        try:
            with open(p, encoding="utf-8", errors="replace", newline="") as f:
                r = csv.reader(f)
                header = next(r)
                ncols = len(header)
                nrows = sum(1 for _ in r)
        except Exception as e:
            print(f"{rel}: ERROR {e}")
            continue
        rows.append((str(rel), nrows, ncols))

    w = max(len(t[0]) for t in rows) if rows else 10
    print(f"{'path':<{w}}  {'rows':>8}  {'cols':>5}")
    print("-" * (w + 20))
    for path, nrows, ncols in rows:
        print(f"{path:<{w}}  {nrows:>8}  {ncols:>5}")
    print(f"\nTotal CSV files: {len(rows)}")


if __name__ == "__main__":
    main()
