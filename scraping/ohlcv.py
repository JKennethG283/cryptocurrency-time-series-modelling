import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen


BASE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1h"
LIMIT = 1000
RETRIES = 5
RETRY_SLEEP_SECONDS = 1.5

SYMBOLS: Dict[str, str] = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana": "SOLUSDT",
}

REQUIRED_COLUMNS = ["timestamp_utc", "open", "high", "low", "close", "volume"]


def utc_iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ms_from_utc_iso(ts: str) -> int:
    return int(datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() * 1000)


def call_binance_klines(symbol: str, start_time_ms: Optional[int], limit: int = LIMIT) -> List[list]:
    params = {"symbol": symbol, "interval": INTERVAL, "limit": limit}
    if start_time_ms is not None:
        params["startTime"] = start_time_ms
    query = urlencode(params)
    url = f"{BASE_URL}?{query}"

    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urlopen(url, timeout=30) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload)
        except Exception as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)
    raise RuntimeError(f"Binance request failed for {symbol}: {last_error}")


def get_earliest_open_time_ms(symbol: str) -> int:
    candles = call_binance_klines(symbol=symbol, start_time_ms=0, limit=1)
    if not candles:
        raise RuntimeError(f"No candles returned for {symbol}")
    return int(candles[0][0])


def normalize_candles(candles: List[list]) -> List[dict]:
    rows = []
    for k in candles:
        rows.append(
            {
                "timestamp_utc": utc_iso_from_ms(int(k[0])),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )
    return rows


def fetch_hourly_history(symbol: str, start_time_ms: Optional[int] = None) -> List[dict]:
    if start_time_ms is None:
        start_time_ms = get_earliest_open_time_ms(symbol)

    all_rows: List[dict] = []
    next_start = start_time_ms
    while True:
        candles = call_binance_klines(symbol=symbol, start_time_ms=next_start, limit=LIMIT)
        if not candles:
            break
        rows = normalize_candles(candles)
        all_rows.extend(rows)

        last_open_time = int(candles[-1][0])
        next_start = last_open_time + (60 * 60 * 1000)

        if len(candles) < LIMIT:
            break
        time.sleep(0.1)
    return all_rows


def ensure_data_folder(base_dir: Path) -> Path:
    data_dir = base_dir / "ohlcv"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def hourly_file_path(data_dir: Path, asset_name: str) -> Path:
    return data_dir / f"{asset_name}_ohlcv_hourly.csv"


def daily_file_path(data_dir: Path, asset_name: str) -> Path:
    return data_dir / f"{asset_name}_ohlcv_daily.csv"


def read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp_utc": row["timestamp_utc"],
                    "open": f"{float(row['open']):.10f}".rstrip("0").rstrip("."),
                    "high": f"{float(row['high']):.10f}".rstrip("0").rstrip("."),
                    "low": f"{float(row['low']):.10f}".rstrip("0").rstrip("."),
                    "close": f"{float(row['close']):.10f}".rstrip("0").rstrip("."),
                    "volume": f"{float(row['volume']):.10f}".rstrip("0").rstrip("."),
                }
            )


def deduplicate_and_sort(rows: List[dict]) -> List[dict]:
    by_ts = {}
    for row in rows:
        by_ts[row["timestamp_utc"]] = row
    ordered_keys = sorted(by_ts.keys())
    return [by_ts[k] for k in ordered_keys]


def validate_columns(rows: List[dict], source_name: str) -> None:
    if not rows:
        return
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")


def log_range(label: str, rows: List[dict]) -> None:
    if not rows:
        print(f"{label}: 0 rows")
        return
    print(f"{label}: {len(rows)} rows, {rows[0]['timestamp_utc']} -> {rows[-1]['timestamp_utc']}")


def init_full_history(base_dir: Path) -> None:
    data_dir = ensure_data_folder(base_dir)
    for asset_name, symbol in SYMBOLS.items():
        print(f"Downloading full hourly history for {asset_name} ({symbol})...")
        rows = fetch_hourly_history(symbol=symbol, start_time_ms=None)
        rows = deduplicate_and_sort(rows)
        validate_columns(rows, f"{asset_name} hourly rows")
        out_path = hourly_file_path(data_dir, asset_name)
        write_csv_rows(out_path, rows)
        log_range(f"Wrote {out_path.name}", rows)


def update_hourly_history(base_dir: Path) -> None:
    data_dir = ensure_data_folder(base_dir)
    for asset_name, symbol in SYMBOLS.items():
        path = hourly_file_path(data_dir, asset_name)
        existing_rows = read_csv_rows(path)
        validate_columns(existing_rows, f"{asset_name} existing hourly CSV")

        if not existing_rows:
            print(f"{path.name} missing or empty. Building full history...")
            new_rows = fetch_hourly_history(symbol=symbol, start_time_ms=None)
            merged = deduplicate_and_sort(new_rows)
            write_csv_rows(path, merged)
            log_range(f"Updated {path.name}", merged)
            continue

        last_ts = existing_rows[-1]["timestamp_utc"]
        next_start_ms = ms_from_utc_iso(last_ts) + (60 * 60 * 1000)
        print(f"Updating {asset_name} from {utc_iso_from_ms(next_start_ms)}...")
        fresh_rows = fetch_hourly_history(symbol=symbol, start_time_ms=next_start_ms)
        merged = deduplicate_and_sort(existing_rows + fresh_rows)
        write_csv_rows(path, merged)
        log_range(f"Updated {path.name}", merged)


def convert_hourly_to_daily(base_dir: Path) -> None:
    data_dir = ensure_data_folder(base_dir)
    for asset_name in SYMBOLS.keys():
        src = hourly_file_path(data_dir, asset_name)
        rows = read_csv_rows(src)
        validate_columns(rows, f"{asset_name} hourly CSV")
        if not rows:
            print(f"Skipping {asset_name}: no hourly data found at {src}")
            continue

        groups: Dict[str, List[dict]] = {}
        for row in deduplicate_and_sort(rows):
            day = row["timestamp_utc"][:10]
            groups.setdefault(day, []).append(row)

        daily_rows: List[dict] = []
        for day in sorted(groups.keys()):
            g = groups[day]
            daily_rows.append(
                {
                    "timestamp_utc": f"{day}T00:00:00Z",
                    "open": float(g[0]["open"]),
                    "high": max(float(x["high"]) for x in g),
                    "low": min(float(x["low"]) for x in g),
                    "close": float(g[-1]["close"]),
                    "volume": sum(float(x["volume"]) for x in g),
                }
            )

        out = daily_file_path(data_dir, asset_name)
        write_csv_rows(out, daily_rows)
        log_range(f"Wrote {out.name}", daily_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and maintain BTC/ETH/SOL OHLCV (open, high, low, close, volume) hourly and daily CSV data from Binance (UTC)."
    )
    parser.add_argument("--init", action="store_true", help="Download full hourly history from earliest Binance data.")
    parser.add_argument("--update", action="store_true", help="Append latest hourly candles to existing hourly CSV files.")
    parser.add_argument("--to-daily", action="store_true", help="Convert hourly CSV files to daily OHLCV CSV files.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Directory that will contain the 'ohlcv' subfolder (default: ../data relative to this script).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_dir = Path(args.output_dir).expanduser().resolve()

    if not any([args.init, args.update, args.to_daily]):
        parser.print_help()
        return

    if args.init:
        init_full_history(base_dir)
    if args.update:
        update_hourly_history(base_dir)
    if args.to_daily:
        convert_hourly_to_daily(base_dir)


if __name__ == "__main__":
    main()
