import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from dune_client.client import DuneClient


START_DATE = "2020-01-01"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "blockchain"
WHALE_USD_THRESHOLD = 100000
BTC_WHALE_THRESHOLD_BTC = 100.0

ASSETS: Dict[str, Dict[str, str]] = {
    "eth": {
        "name": "Ethereum",
        "dune_chain": "ethereum",
        "hourly_file": "eth_blockchain_h.csv",
        "daily_file": "eth_blockchain_d.csv",
        "growth_col": "eth_tx_growth",
        "total_col": "total_usd",
        "avg_col": "avg_usd",
        "whale_col": "whale_tx_count",
        "query_type": "usd_transfers",
    },
    "btc": {
        "name": "Bitcoin",
        "dune_chain": "bitcoin",
        "hourly_file": "btc_blockchain_h.csv",
        "daily_file": "btc_blockchain_d.csv",
        "growth_col": "btc_tx_growth",
        "total_col": "total_btc",
        "avg_col": "avg_btc",
        "whale_col": "btc_whale_tx_count",
        "query_type": "btc_native",
    },
}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def get_dune_client() -> DuneClient:
    api_key = os.getenv("DUNE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing DUNE_API_KEY environment variable. "
            "Create a Dune API key and set it before running."
        )
    return DuneClient(api_key=api_key)


def build_usd_transfer_query(dune_chain: str, whale_usd_threshold: int, start_date: str) -> str:
    return f"""
WITH tx_hourly AS (
  SELECT
    date_trunc('hour', block_time) AS hour,
    COUNT(*) AS num_tx
  FROM {dune_chain}.transactions
  WHERE block_date >= DATE '{start_date}'
    AND block_time >= TIMESTAMP '{start_date}'
  GROUP BY 1
),
transfers_hourly AS (
  SELECT
    date_trunc('hour', block_time) AS hour,
    COALESCE(SUM(amount_usd), 0) AS total_usd,
    COALESCE(AVG(amount_usd), 0) AS avg_usd,
    COUNT(*) FILTER (WHERE amount_usd > {whale_usd_threshold}) AS whale_tx_count
  FROM tokens.transfers
  WHERE blockchain = '{dune_chain}'
    AND block_date >= DATE '{start_date}'
    AND block_time >= TIMESTAMP '{start_date}'
  GROUP BY 1
)
SELECT
  COALESCE(t.hour, x.hour) AS hour,
  COALESCE(t.num_tx, 0) AS num_tx,
  COALESCE(x.total_usd, 0) AS total_usd,
  COALESCE(x.avg_usd, 0) AS avg_usd,
  COALESCE(x.whale_tx_count, 0) AS whale_tx_count
FROM tx_hourly t
FULL OUTER JOIN transfers_hourly x ON t.hour = x.hour
ORDER BY hour
"""


def build_btc_native_query(start_date: str, btc_whale_threshold_btc: float) -> str:
    return f"""
WITH tx_level AS (
  SELECT
    id AS tx_id,
    date_trunc('hour', block_time) AS hour,
    COALESCE(SUM(output_value), 0) AS tx_output_native
  FROM bitcoin.transactions
  WHERE block_date >= DATE '{start_date}'
    AND block_time >= TIMESTAMP '{start_date}'
  GROUP BY id, date_trunc('hour', block_time)
)
SELECT
  hour,
  COUNT(*) AS num_tx,
  COALESCE(SUM(tx_output_native), 0) AS total_btc,
  COALESCE(AVG(tx_output_native), 0) AS avg_btc,
  COUNT(*) FILTER (WHERE tx_output_native > {btc_whale_threshold_btc}) AS btc_whale_tx_count
FROM tx_level
GROUP BY 1
ORDER BY hour
"""


def run_query(dune: DuneClient, sql: str, label: str) -> pd.DataFrame:
    logging.info("Running query for %s", label)
    try:
        result = dune.run_sql(query_sql=sql, ping_frequency=2, name=f"{label}_hourly_export")
    except Exception as exc:
        raise RuntimeError(
            f"Dune query failed for {label}. Verify DUNE_API_KEY, table names, and API access. "
            f"Underlying error: {exc}"
        ) from exc
    if result is None or result.result is None:
        raise RuntimeError(f"Dune returned no result object for {label}.")
    records = result.result.rows
    df = pd.DataFrame(records)
    logging.info("%s rows returned: %d", label, len(df))
    return df


def clean_hourly_dataframe(df: pd.DataFrame, growth_col: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["hour"] = pd.to_datetime(out["hour"], utc=True, errors="coerce")
    out = out.dropna(subset=["hour"]).sort_values("hour").drop_duplicates(subset=["hour"], keep="last")

    for col in out.columns:
        if col != "hour":
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out[growth_col] = out["num_tx"].pct_change().replace([float("inf"), float("-inf")], 0).fillna(0)
    out = out.fillna(0)
    return out


def hourly_to_daily(df: pd.DataFrame, growth_col: str, total_col: str, avg_col: str, whale_col: str) -> pd.DataFrame:
    if df.empty:
        return df

    x = df.copy()
    x["hour"] = pd.to_datetime(x["hour"], utc=True, errors="coerce")
    x = x.dropna(subset=["hour"]).set_index("hour")

    daily = (
        x.resample("D")
        .agg(
            {
                "num_tx": "sum",
                total_col: "sum",
                whale_col: "sum",
            }
        )
        .reset_index()
        .rename(columns={"hour": "day"})
    )
    daily[avg_col] = daily[total_col] / daily["num_tx"].replace(0, pd.NA)
    daily[avg_col] = daily[avg_col].fillna(0)
    daily[growth_col] = daily["num_tx"].pct_change().replace([float("inf"), float("-inf")], 0).fillna(0)
    daily = daily.sort_values("day").drop_duplicates(subset=["day"], keep="last").fillna(0)
    return daily


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Saved %s", output_path)


def print_summary(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title} first 5 rows:")
    print(df.head(5))
    print(f"{title} shape: {df.shape}")


def export_all(
    output_dir: Path,
    whale_usd_threshold: int,
    include_daily: bool,
    start_date: str,
    btc_whale_threshold_btc: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dune = get_dune_client()

    hourly_results: Dict[str, pd.DataFrame] = {}
    for key, cfg in ASSETS.items():
        if cfg["query_type"] == "btc_native":
            sql = build_btc_native_query(start_date, btc_whale_threshold_btc)
        else:
            sql = build_usd_transfer_query(cfg["dune_chain"], whale_usd_threshold, start_date)

        raw = run_query(dune, sql, cfg["name"])
        cleaned = clean_hourly_dataframe(raw, cfg["growth_col"])
        hourly_results[key] = cleaned
        save_csv(cleaned, output_dir / cfg["hourly_file"])

    if include_daily:
        for key, cfg in ASSETS.items():
            daily_df = hourly_to_daily(
                hourly_results[key],
                cfg["growth_col"],
                cfg["total_col"],
                cfg["avg_col"],
                cfg["whale_col"],
            )
            save_csv(daily_df, output_dir / cfg["daily_file"])

    return hourly_results["eth"], hourly_results["btc"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export hourly on-chain metrics for ETH/BTC via Dune API.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="CSV output directory.")
    parser.add_argument("--start-date", default=START_DATE, help="Inclusive UTC start date in YYYY-MM-DD format.")
    parser.add_argument(
        "--whale-usd-threshold",
        type=int,
        default=WHALE_USD_THRESHOLD,
        help="USD threshold used to count whale transactions.",
    )
    parser.add_argument(
        "--btc-whale-threshold-btc",
        type=float,
        default=BTC_WHALE_THRESHOLD_BTC,
        help="BTC native whale threshold used in bitcoin.transactions output_value.",
    )
    parser.add_argument("--include-daily", action="store_true", help="Also generate daily CSV files from hourly output.")
    return parser


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        eth_df, btc_df = export_all(
            output_dir=output_dir,
            whale_usd_threshold=args.whale_usd_threshold,
            include_daily=args.include_daily,
            start_date=args.start_date,
            btc_whale_threshold_btc=args.btc_whale_threshold_btc,
        )
        print_summary("Ethereum", eth_df)
        print_summary("Bitcoin", btc_df)
        logging.info("Completed successfully.")
        return 0
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
