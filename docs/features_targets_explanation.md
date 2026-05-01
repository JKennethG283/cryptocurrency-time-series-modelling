# OHLCV ML dataset: features and targets

This document describes columns used for modelling:

1. **Pre-processed OHLCV ML tables** — produced by [`data_preparation.py`](../data_preparation.py) under `data/pre-processing/` (technical features and targets below).
2. **Merged modelling tables** — produced by [`merge.py`](../merge.py) under `data/merge/`: same OHLCV ML columns **plus** optional **on-chain** and **macro** columns joined on UTC time (see [Blockchain features](#blockchain-features-merged-tables-only) and your macro CSVs).

**Notation:** At time \(t\), \(O, H, L, C, V\) denote open, high, low, close, and volume. Unless noted, rolling windows use **only past and current** bars (no lookahead in features). Targets use **future** bars relative to \(t\).

**Implementation details:** Features use `pandas`/`numpy`; RSI, ATR, and Bollinger bands use the **`ta`** library. Log-volume uses \(\varepsilon = 10^{-12}\) to avoid \(\log(0)\).

---

## Output CSV structure

- First column: **`timestamp`** (UTC).
- Remaining columns: engineered **features**, then **targets**.
- Rows with any missing or non-finite value after construction are **dropped**, so the saved file has **no NaNs**.

**Daily** files omit **`hour`** and use target horizons **1, 3, 7, 14** (bars = days).

**Hourly** files include **`hour`** (UTC) and use target horizons **1, 3, 6, 12, 24** (bars = hours).

---

## Features

### Returns

| Column | Formula / meaning |
|--------|-------------------|
| `log_ret` | \(\log(C_t / C_{t-1})\) |
| `ret_lag_k` for \(k = 1,\ldots,5\) | `log_ret` shifted by \(k\) bars: at \(t\) equals \(\log(C_{t-k}/C_{t-k-1})\). Longer lags overlap rolling features. |

### Trend (EMA)

| Column | Formula / meaning |
|--------|-------------------|
| `ema_W` for \(W \in \{10,20,50\}\) | Exponential moving average of **close** with `span=W`, `adjust=False` (`ewm`). |
| `ema_dist_W` | \((C_t - \text{ema}_W) / \text{ema}_W\) |

### Momentum

| Column | Formula / meaning |
|--------|-------------------|
| `rsi_14` | RSI(14) on **close** (`ta.momentum.RSIIndicator`). |
| `roc_n` for \(n \in \{5,10\}\) | \(C_t / C_{t-n} - 1\) (not multiplied by 100). |
| `momentum_n` for \(n \in \{10,20\}\) | \(C_t - C_{t-n}\) |

### Volatility and range

| Column | Formula / meaning |
|--------|-------------------|
| `ret_std_W` for \(W \in \{10,20\}\) | Sample standard deviation (`ddof=1`) of `log_ret` over the last **W** bars ending at \(t\). |
| `atr_14` | Average True Range, window 14 (`ta.volatility.AverageTrueRange`). |
| `hl_range` | \((H_t - L_t) / C_t\) |
| `bb_width` | Bollinger Bands on **close**, window **20**, **2** standard deviations: \((\text{upper} - \text{lower}) / \text{middle}\) (`ta.volatility.BollingerBands`). |

### Volume

| Column | Formula / meaning |
|--------|-------------------|
| `log_volume` | \(\log(\max(V_t, \varepsilon))\). |
| `vol_ma_W` for \(W \in \{10,20\}\) | Simple moving average of **volume** over **W** bars. |
| `vol_spike_W` | \(V_t / \text{vol\_ma\_W}\) (guards against zero MA). |
| `ret_x_log_vol` | `log_ret` \(\times\) `log_volume` (return–liquidity interaction on comparable scales). |

### Rolling mean of returns

| Column | Formula / meaning |
|--------|-------------------|
| `ret_mean_W` for \(W \in \{10,20\}\) | Mean of `log_ret` over the last **W** bars ending at \(t\). |

### Calendar

| Column | Formula / meaning |
|--------|-------------------|
| `day_of_week` | Integer **0** (Monday) through **6** (Sunday) from the timestamp. |
| `hour` | **Hourly datasets only:** hour of day in **UTC**, **0–23**. |

### Lagged key features (lags 1–3)

At row \(t\), each column is the underlying series value from **\(t - k\)** bars ago.

| Column | Underlying series |
|--------|-------------------|
| `rsi_lag_k` | `rsi_14` |
| `ret_std_10_lag_k` | `ret_std_10` |
| `log_volume_lag_k` | `log_volume` |

Lagged **returns** for lags 1–3 are already provided as `ret_lag_1`, `ret_lag_2`, `ret_lag_3`.

---

## Blockchain features (merged tables only)

These columns appear **only** in `data/merge/*_merged.csv`, not in `data/pre-processing/*_ml.csv`. They come from `data/block chain/*.csv`, built via [`scraping/block_chain.py`](../scraping/block_chain.py) (Dune Analytics queries) and merged in [`merge.py`](../merge.py) on a normalised UTC key (`day` or `hour` aligned to the OHLCV bar). Non-time columns are **prefixed** so asset-specific names stay unique.

**Coverage**

| Asset | Merged file | Blockchain columns |
|-------|-------------|-------------------|
| Bitcoin | `bitcoin_ohlcv_*_merged.csv` | `btc_bc_*` |
| Ethereum | `ethereum_ohlcv_*_merged.csv` | `eth_bc_*` |
| Solana | `solana_ohlcv_*_merged.csv` | **None** — no chain file is configured in `merge.py` for Solana. |

**Join behaviour:** For BTC and ETH, `merge.py` uses an **inner** join between the ML table and the chain table, so only timestamps present in **both** survive. That can reduce row counts versus `*_ml.csv`.

### Macro features (merged tables only)

Macro columns are merged with a **left join** onto the ML timeline and then `ffill` is applied.
Because the pipeline is causal (no `bfill`), the first rows can remain missing if macro files start later than OHLCV. This is expected and avoids injecting future macro values into earlier timestamps.

### Bitcoin (`btc_bc_*`)

Source daily/hourly files: `btc_blockchain_d.csv`, `btc_blockchain_h.csv`. Original columns (before prefix) are aggregated per day or per hour from Dune (`btc_native`-style metrics).

| Column in `*_merged.csv` | Meaning |
|---------------------------|---------|
| `btc_bc_num_tx` | Count of transactions in the bar (hour or day). |
| `btc_bc_total_btc` | Sum of native BTC amounts over transactions in the bar (as in the chain extract). |
| `btc_bc_btc_whale_tx_count` | Count of “whale” transactions in the bar — in the scraper, transactions with BTC amount **greater than 100 BTC** (`BTC_WHALE_THRESHOLD_BTC` in `block_chain.py`). |
| `btc_bc_avg_btc` | Average BTC size per transaction in the bar (definition follows the Dune aggregation). |
| `btc_bc_btc_tx_growth` | Period-over-period growth in activity for the series (e.g. vs previous hour or day — computed when building the chain CSV). First bar is typically **0**. |

### Ethereum (`eth_bc_*`)

Source daily/hourly files: `eth_blockchain_d.csv`, `eth_blockchain_h.csv`. Transfer/value metrics use token **USD** amounts from Dune’s `tokens.transfers` style queries.

| Column in `*_merged.csv` | Meaning |
|---------------------------|---------|
| `eth_bc_num_tx` | Count of transfer rows counted in the bar. |
| `eth_bc_total_usd` | Sum of **USD** notional (`amount_usd`) over transfers in the bar. |
| `eth_bc_whale_tx_count` | Count of transfers with USD amount **greater than \$100,000** (`WHALE_USD_THRESHOLD` in `block_chain.py`). |
| `eth_bc_avg_usd` | Average USD size per transfer in the bar. |
| `eth_bc_eth_tx_growth` | Period-over-period growth for the activity series (e.g. vs previous hour or day). First bar is typically **0**. |

**Note:** Exact Dune SQL and filters may evolve in `block_chain.py`; treat these columns as **on-chain activity proxies** coarsely aligned to price bars, not as exchange OHLCV volume.

---

## Targets

Targets are **labels** for supervised learning at row \(t\). They may use information at **\(t+h\)** or over **\((t+1,\ldots,t+h)\)**; they are **not** used as inputs when training on row \(t\).

### Horizons

| Data frequency | Horizons \(h\) (number of bars ahead) |
|----------------|----------------------------------------|
| Daily | 1, 3, 7, 14 |
| Hourly | 1, 3, 6, 12, 24 |

For each horizon \(h\), **`target_ret_fwd_h`**, **`target_log_vol_fwd_h`**, and **`target_vol_ratio_fwd_h`** are always created. **`target_vol_fwd_h`** is created only for **`h > 1`**: a one-bar forward window has no meaningful standard deviation (it would be identically zero), so that column is omitted.

### Per-horizon columns

| Column | Definition |
|--------|------------|
| `target_ret_fwd_h` | \(\log(C_{t+h} / C_t)\) — cumulative log return from \(t\) to the close **\(h\)** bars later. |
| `target_vol_fwd_h` (if \(h>1\)) | **Forward** variability of one-step log returns over the **next** \(h\) bars: population standard deviation (`ddof=0`) of `log_ret.shift(-1)` over that forward window (first term at \(t\) is \(\log(C_{t+1}/C_t)\)). |
| `target_log_vol_fwd_h` | \(\log(\max(V_{t+h}, \varepsilon))\). |
| `target_vol_ratio_fwd_h` | \(V_{t+h} / \text{SMA}_{20}(V)\) **computed at time \(t+h\)** (20-bar simple moving average of volume including bar \(t+h\)). |

---

## Validation notes

- The first usable row appears only after sufficient **history** for the longest lookbacks (roughly EMA 50, 20-period bands/volume, five return lags, etc.).
- The **last** usable rows require **\(h_{\max}\)** future bars for every target horizon; those incomplete tail rows are removed by `dropna`.
- Spot-check: `target_ret_fwd_1` at \(t\) should equal \(\log(C_{t+1}/C_t)\) when aligned with the raw OHLCV series sorted by time.

---

## Related files

- OHLCV feature pipeline: [`data_preparation.py`](../data_preparation.py) → `data/pre-processing/*_ml.csv`
- Raw OHLCV: `data/ohlcv/*_ohlcv_*.csv`
- On-chain extracts: [`scraping/block_chain.py`](../scraping/block_chain.py) → `data/block chain/*.csv`
- Merge (OHLCV ML + chain + macro): [`merge.py`](../merge.py) → `data/merge/*_merged.csv` (drops numeric columns that are all-NaN or constant after the join).
