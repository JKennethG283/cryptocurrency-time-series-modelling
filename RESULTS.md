# Quick Results Summary

This document summarizes the outputs in `results/quick`.

## Source Artifacts

- Metrics tables:
  - `results/quick/tables/metrics_returns.csv`
  - `results/quick/tables/metrics_volatility.csv`
- Diagnostic and prediction plots:
  - `results/quick/figures/`

## Return Forecasting (`h=1`)

Best model in each dataset based on lowest `MAE`:

| Dataset | Best Model | MAE | RMSE | MASE |
|---|---|---:|---:|---:|
| `bitcoin_ohlcv_daily_merged` | ARIMA | 0.021406 | 0.031024 | 0.665459 |
| `bitcoin_ohlcv_hourly_merged` | ARIMA | 0.003616 | 0.005193 | 0.688511 |
| `ethereum_ohlcv_daily_merged` | ARIMA | 0.028920 | 0.041645 | 0.675517 |
| `ethereum_ohlcv_hourly_merged` | ARIMA | 0.004538 | 0.006702 | 0.678101 |

### Observation

- ARIMA is consistently the best performer for one-step return prediction across all quick datasets.

## Volatility Forecasting (`h=7` daily, `h=24` hourly)

Best model in each dataset based on lowest `MAE`:

| Dataset | Horizon | Best Model | MAE | RMSE | MASE |
|---|---:|---|---:|---:|---:|
| `bitcoin_ohlcv_daily_merged` | 7 | XGBoost | 0.010099 | 0.013210 | 0.934988 |
| `bitcoin_ohlcv_hourly_merged` | 24 | XGBoost | 0.001465 | 0.001912 | 0.929318 |
| `ethereum_ohlcv_daily_merged` | 7 | XGBoost | 0.013753 | 0.018636 | 0.941086 |
| `ethereum_ohlcv_hourly_merged` | 24 | XGBoost | 0.001816 | 0.002330 | 0.806425 |

### Observation

- XGBoost is consistently the strongest model for quick volatility forecasting across all tested datasets.

## Figures Included

`results/quick/figures/` includes:

- `best_actual_vs_predicted_*` plots for selected best runs.
- `best_diag_*` diagnostic plots for those runs.

Use these plots with the metrics tables to inspect forecast tracking quality and residual behavior.
