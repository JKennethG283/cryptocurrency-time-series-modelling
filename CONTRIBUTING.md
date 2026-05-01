# Contributing

- **Style:** Keep changes focused; follow existing patterns in `data_preparation.py`, `merge.py`, and `benchmark.py`.
- **Tests:** Add or update tests in `tests/` and run:
  ```bash
  pip install -r requirements.txt -r requirements-dev.txt
  python -m pytest tests/ -v
  ```
- **Time series:** New features must use only information available at the bar you predict from; labels stay in `target_*` columns and are excluded in `build_xy` from `X`.
- **Data fusion:** Do not reintroduce `bfill` on forward-looking panels (macro) — use `ffill` only after sorting by time (see `merge.py`).

Pull requests: describe what changed, why, and any trade-offs. Link related issues or capstone report sections if applicable.
