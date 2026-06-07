import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import RAW_DATA_DIR, SYMBOLS
from symbols import parse_symbol_args, resolve_symbols
from utils import ensure_dirs, find_mt5_symbol_candidates, setup_logger, symbol_to_filename

LOGGER = setup_logger("download_mt5_data")


def import_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 package is not installed. Run: pip install -r requirements.txt") from exc
    return mt5


def mt5_timeframe(mt5, timeframe: str):
    mapping = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[timeframe]


def connect_mt5():
    mt5 = import_mt5()
    if not mt5.initialize():
        code, message = mt5.last_error()
        raise RuntimeError(f"MT5 initialize failed: {code} {message}")
    return mt5


def ensure_symbol_visible(mt5, logical_symbol: str, mt5_symbol: str) -> None:
    info = mt5.symbol_info(mt5_symbol)
    if info is None:
        candidates = find_mt5_symbol_candidates(mt5, mt5_symbol)
        raise RuntimeError(f"Symbol {mt5_symbol} for {logical_symbol} not found. Similar symbols: {candidates}")
    if not info.visible and not mt5.symbol_select(mt5_symbol, True):
        raise RuntimeError(f"Symbol {mt5_symbol} exists but cannot be selected/visible in MT5")


def validate_rates(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    if df.empty:
        raise RuntimeError(f"{symbol}: downloaded data is empty")
    if df["time"].duplicated().any():
        raise RuntimeError(f"{symbol}: duplicate time rows found")
    if df.isna().any().any():
        raise RuntimeError(f"{symbol}: missing values found")
    if "spread" not in df.columns:
        raise RuntimeError(f"{symbol}: spread column is missing")
    expected_minutes = 5 if timeframe == "M5" else None
    if expected_minutes:
        diffs = df["time"].diff().dropna().dt.total_seconds() / 60
        large_gaps = diffs[diffs > expected_minutes * 3]
        if not large_gaps.empty:
            LOGGER.warning("%s has %s large candle gaps. This can be normal around market close.", symbol, len(large_gaps))


def download_symbol(mt5, symbol: str) -> pd.DataFrame:
    ensure_dirs([symbol])
    cfg = SYMBOLS[symbol]
    mt5_symbol = cfg["mt5_symbol"]
    ensure_symbol_visible(mt5, symbol, mt5_symbol)
    info = mt5.symbol_info(mt5_symbol)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(cfg["data_months"]) * 31)
    rates = mt5.copy_rates_range(mt5_symbol, mt5_timeframe(mt5, cfg["timeframe"]), start, end)
    if rates is None:
        code, message = mt5.last_error()
        raise RuntimeError(f"{symbol}: copy_rates_range failed: {code} {message}")
    df = pd.DataFrame(rates)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
        df["point"] = getattr(info, "point", 0.0) or 0.0
        df["tick_size"] = getattr(info, "trade_tick_size", 0.0) or 0.0
        df["tick_value"] = getattr(info, "trade_tick_value", 0.0) or 0.0
    validate_rates(df, symbol, cfg["timeframe"])
    filename = symbol_to_filename(symbol)
    output_path = RAW_DATA_DIR / symbol / f"{filename}_m5_raw.csv"
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved %s rows for %s to %s", len(df), symbol, output_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download M5 OHLCV data from MetaTrader 5.")
    parse_symbol_args(parser)
    args = parser.parse_args()
    mt5 = connect_mt5()
    try:
        for symbol in resolve_symbols(args):
            download_symbol(mt5, symbol)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
