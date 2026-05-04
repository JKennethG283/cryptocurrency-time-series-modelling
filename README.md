# UTS AI Capstone — Crypto forecasting with OHLCV, macro, and on-chain data

Walk-forward experiments on **Bitcoin, Ethereum, and Solana** using **OHLCV-derived features**, **macroeconomic** series, and **on-chain** metrics (where available). The project builds tabular ML datasets, merges external panels with **causal** imputation, and evaluates several model families with time-series–aware validation: features at time *t* do not include the supervised targets or other future data (see [tests](tests/) and `benchmark.build_xy`).

**License:** [MIT](LICENSE). Course project: **UTS AI Capstone** (see [Acknowledgement](#acknowledgement)).

## What’s in this repo

| Component | Description |
|------------|-------------|
| [data_preparation.py](data_preparation.py) | OHLCV → features + multi-horizon targets → `data/pre-processing/*_ml.csv` |
| [merge.py](merge.py) | ML tables + macro + on-chain → `data/merge/*_merged.csv` (macro: `ffill` only, sorted by time) |
| [model.py](model.py) | **Default** quick run: BTC + ETH, return & volatility, `results/quick/` |
| [benchmark.py](benchmark.py) | **Full** grid: all merged datasets and target families (optional **GARCH** / **LSTM** via flags) |
| [experiment.py](experiment.py) | Ablations, context sweeps, permutation importance (sklearn) |
| [main.py](main.py) | One command: pre-process → merge → `model.py` (see [Run everything](#run-everything)) |
| [docs/features_targets_explanation.md](docs/features_targets_explanation.md) | Column and target definitions |

## Repository layout

| Path | Purpose |
|------|---------|
| `data/ohlcv/` | Raw OHLCV CSVs (e.g. from Binance; sample [scraping/ohlcv.py](scraping/ohlcv.py)) |
| `data/pre-processing/` | ML-ready tables (`*_ml.csv`) — default output of `data_preparation.py` |
| `data/macro/`, `data/block chain/` | Macro and on-chain inputs for merging |
| `data/merge/` | Modelling tables (`*_merged.csv`) |
| `results/tables/`, `results/figures/` | Full `benchmark.py` outputs (default) |
| `results/quick/tables/`, `results/quick/figures/` | Default `model.py` outputs |
| `results/experiments/` | `experiment.py` outputs |
| `scripts/` | EDA, correlation heatmaps, CSV shape reports |
| `scraping/` | Sample scripts for OHLCV, macro, and blockchain data |
| `tests/` | Pytest: label leakage, merge invariants, paths |

## Requirements

- **Python 3.10+** (3.11 recommended for local dev and [CI](.github/workflows/ci.yml).)

```bash
pip install -r requirements.txt
# optional: run tests
pip install -r requirements-dev.txt
```

**Windows note:** Prophet may need [CmdStanPy](https://mc-stan.org/cmdstanpy/). If Prophet or neural models fail, use `--skip-prophet` / `--skip-neural` on `model.py` or `benchmark.py`.

## Evaluation approach (for reports and interviews)

- **Metrics** written by `benchmark` / `model`: **MAE**, **RMSE**, **MASE** (MASE vs a naive h-step target lag benchmark). You compare models on the **same** walk-forward grid and the **same** targets — not a trading PnL claim.
- **Walk-forward:** Daily: expanding history. Hourly: sliding context + last *N* rows for eval (see `benchmark.py` docstring).
- **ARIMA / Prophet / sklearn protocol:** `ARIMA` is pure univariate ARIMA by default; `SARIMAX` is only used when `--use-sarimax-exog` is passed. ARIMA-family, Prophet, **RandomForest**, and **XGBoost** all use the **same** horizon-aware embargo on training **labels** (through `t-h`) for multi-horizon forward-window targets like `target_vol_fwd_24`.
- **GARCH / LSTM:** `model.py` runs **GARCH(1,1)** on `log_ret` for volatility targets (one-step conditional σ vs `target_vol_fwd_h`, comparable scale to per-period vol — not `σ√h`) and a small **LSTM** on lagged top exogenous features (use `--skip-garch` / `--skip-lstm` to omit). Full `benchmark.py` skips them by default for runtime; pass `--enable-garch` and/or `--enable-lstm` to include. Requires `arch` (listed in `requirements.txt`) and `torch`.
- **Slower baselines (ARIMA, Prophet, VAR, NLinear):** use `--ts-eval-stride k` to evaluate only every *k*-th origin (e.g. `24` on hourly) to cut runtime. Tree models (RF, XGB) still predict on every step in their evaluated range, with `refit_every_*` controlling refit frequency.
- **Limitations:** (1) Macro series use **as-of forward-fill** on the price bar timeline — that is not the same as “value known to the market at *t*” if releases lag; treat macro as a coarse covariate. (2) Report clearly that results are **off-sample walk-forward** under a fixed protocol, not live trading results.

## Data pipeline (end-to-end)

1. **OHLCV** — Place sorted CSVs under `data/ohlcv/` (open/high/low/close/volume + UTC time; see `data_preparation.py`).
2. **Pre-process** — `python data_preparation.py` writes `data/pre-processing/*_ml.csv` (repo-default output path). Or call `build_all_ohlcv_datasets(output_dir=...)` from Python.
3. **Merge** — `python merge.py` → `data/merge/*_merged.csv`.
4. **Evaluate** — `python model.py` (fast default) or `python benchmark.py` (full study).

## Run everything

```bash
python main.py
```

- `--skip-data-prep` / `--skip-merge` if artifacts already exist.
- `--no-model` to only build data.
- Extra arguments after `--` are passed to `model.py`, e.g. `python main.py -- --skip-neural --skip-prophet`.

## Running the models

**Default (faster):** Bitcoin + Ethereum, return **h=1**, volatility **h=7** (daily) / **h=24** (hourly); hourly context 600, last 1000 hourly rows; outputs under `results/quick/`.

```bash
python model.py
```

**Full evaluation** (all six merged datasets, all target families; default horizons are daily: `1,3,7`, hourly: `1,6,24`; can be slow):

```bash
python benchmark.py
```

**Benchmark smoke** (one dataset, stride, skip neural):

```bash
python benchmark.py --datasets bitcoin_ohlcv_daily_merged.csv --ts-eval-stride 5 --refit-every-daily 3 --skip-neural
```

**Skips** (supported by both `model.py` and `benchmark.py` where applicable):

```bash
python model.py --skip-prophet
python benchmark.py --skip-neural
```

**Quick experiments** (ablations / context / importance):

```bash
python experiment.py
python experiment.py --modes ablation context importance
python experiment.py --dataset data/merge/bitcoin_ohlcv_hourly_merged.csv --max-rows 5000
```

## Outputs

- **Full benchmark:** `results/tables/metrics_*.csv`; figures under `results/figures/` when plotting succeeds.
- **Default model:** `results/quick/tables/metrics_*.csv`; figures under `results/quick/figures/` when plotting succeeds.

After `python main.py`, the minimum expected artifacts are the three quick metrics tables:
`metrics_returns.csv`, `metrics_volatility.csv`, `metrics_volume.csv` in `results/quick/tables/`.

Details: walk-forward design, horizon lists, and volatility h=1 caveat are in the [benchmark.py](benchmark.py) module docstring.

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Continuous integration: [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Scripts (utilities)

- `scripts/generate_eda_figures.py` — EDA from merged data  
- `scripts/regenerate_correlation_heatmap.py` — heatmaps (uses `benchmark` column logic)  
- `scripts/csv_shape_report.py` — CSV shape / column counts  

## Acknowledgement

Course project for the **UTS AI Capstone** subject.
