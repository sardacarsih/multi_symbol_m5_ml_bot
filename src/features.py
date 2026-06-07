import argparse

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SYMBOLS
from symbols import parse_symbol_args, resolve_symbols
from utils import ensure_dirs, setup_logger, symbol_to_filename

LOGGER = setup_logger("features")


def rolling_percentile(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(5, window // 2)
    return series.rolling(window, min_periods=min_periods).rank(pct=True)


def add_directional_movement_features(result: pd.DataFrame) -> None:
    high_diff = result["high"].diff()
    low_diff = -result["low"].diff()
    plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0), index=result.index)
    minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0), index=result.index)
    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr_14 = true_range.rolling(14, min_periods=14).sum().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(14, min_periods=14).sum() / tr_14
    minus_di = 100 * minus_dm.rolling(14, min_periods=14).sum() / tr_14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    result["plus_di_14"] = plus_di
    result["minus_di_14"] = minus_di
    result["di_spread_14"] = (plus_di - minus_di) / 100.0
    result["adx_14"] = dx.rolling(14, min_periods=14).mean()
    result["trend_direction_strength"] = result["di_spread_14"] * (result["adx_14"] / 100.0)


def add_higher_timeframe_context(result: pd.DataFrame, timeframe: str, prefix: str) -> None:
    source = result[["time", "open", "high", "low", "close", "tick_volume"]].copy()
    htf = (
        source.set_index("time")
        .resample(timeframe, label="right", closed="right")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            tick_volume=("tick_volume", "sum"),
        )
        .dropna()
    )
    if htf.empty:
        for name in ["return_1", "close_to_ema20", "ema20_slope", "range_position_20", "atr_ratio"]:
            result[f"{prefix}_{name}"] = np.nan
        return

    htf["return_1"] = htf["close"].pct_change()
    htf["ema20"] = htf["close"].ewm(span=20, adjust=False, min_periods=5).mean()
    htf["close_to_ema20"] = htf["close"] / htf["ema20"].replace(0, np.nan) - 1
    htf["ema20_slope"] = htf["ema20"].diff(3) / htf["ema20"].shift(3).replace(0, np.nan)
    htf_high_20 = htf["high"].rolling(20, min_periods=5).max()
    htf_low_20 = htf["low"].rolling(20, min_periods=5).min()
    htf["range_position_20"] = (htf["close"] - htf_low_20) / (htf_high_20 - htf_low_20).replace(0, np.nan)
    htf["atr_like"] = (htf["high"] - htf["low"]).rolling(14, min_periods=5).mean()
    htf["atr_ratio"] = htf["atr_like"] / htf["close"].replace(0, np.nan)

    feature_columns = ["return_1", "close_to_ema20", "ema20_slope", "range_position_20", "atr_ratio"]
    completed_htf = htf[feature_columns].shift(1).add_prefix(f"{prefix}_").reset_index()
    merged = pd.merge_asof(
        result[["time"]].sort_values("time"),
        completed_htf.sort_values("time"),
        on="time",
        direction="backward",
    )
    for column in completed_htf.columns:
        if column != "time":
            result[column] = merged[column].to_numpy()


def add_features(df: pd.DataFrame, symbol: str, include_symbol_features: bool = True) -> pd.DataFrame:
    result = df.copy()
    result["time"] = pd.to_datetime(result["time"])
    result = result.sort_values("time").drop_duplicates("time")

    for column in ["open", "high", "low", "close"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "spread" not in result.columns:
        result["spread"] = 0.0
    result["spread"] = pd.to_numeric(result["spread"], errors="coerce").fillna(0.0)
    if "point" in result.columns and result["point"].fillna(0).gt(0).any():
        result["spread_points"] = result["spread"]
        result["spread"] = result["spread"] * pd.to_numeric(result["point"], errors="coerce").fillna(0.0)
    else:
        result["spread_points"] = result["spread"]
    if "tick_volume" not in result.columns:
        result["tick_volume"] = 0.0
    result["tick_volume"] = pd.to_numeric(result["tick_volume"], errors="coerce").fillna(0.0)

    result["return_1"] = result["close"].pct_change(1)
    result["return_3"] = result["close"].pct_change(3)
    result["return_6"] = result["close"].pct_change(6)
    result["return_12"] = result["close"].pct_change(12)

    atr = AverageTrueRange(result["high"], result["low"], result["close"], window=14)
    result["atr_14"] = atr.average_true_range()
    result["atr_ratio"] = result["atr_14"] / result["close"]
    result["candle_range"] = result["high"] - result["low"]
    result["rolling_std_12"] = result["return_1"].rolling(12).std()
    result["rolling_std_24"] = result["return_1"].rolling(24).std()

    for window in [20, 50, 100]:
        result[f"ema_{window}"] = EMAIndicator(result["close"], window=window).ema_indicator()
        result[f"close_to_ema{window}"] = result["close"] / result[f"ema_{window}"] - 1
    result["ema20_slope"] = result["ema_20"].diff(3) / result["ema_20"].shift(3)
    result["ema50_slope"] = result["ema_50"].diff(3) / result["ema_50"].shift(3)
    result["ema20_slope_atr"] = result["ema_20"].diff(3) / result["atr_14"].replace(0, np.nan)
    result["ema50_slope_atr"] = result["ema_50"].diff(3) / result["atr_14"].replace(0, np.nan)
    result["ema_alignment_score"] = (
        (result["close"] > result["ema_20"]).astype(int)
        + (result["ema_20"] > result["ema_50"]).astype(int)
        + (result["ema_50"] > result["ema_100"]).astype(int)
        - (result["close"] < result["ema_20"]).astype(int)
        - (result["ema_20"] < result["ema_50"]).astype(int)
        - (result["ema_50"] < result["ema_100"]).astype(int)
    ) / 3.0
    add_directional_movement_features(result)

    result["rsi_14"] = RSIIndicator(result["close"], window=14).rsi()
    macd = MACD(result["close"])
    result["macd"] = macd.macd()
    result["macd_signal"] = macd.macd_signal()
    result["macd_hist"] = macd.macd_diff()

    result["body_size"] = (result["close"] - result["open"]).abs()
    result["upper_wick"] = result["high"] - result[["open", "close"]].max(axis=1)
    result["lower_wick"] = result[["open", "close"]].min(axis=1) - result["low"]
    safe_range = result["candle_range"].replace(0, np.nan)
    result["body_to_range"] = result["body_size"] / safe_range
    result["upper_wick_to_range"] = result["upper_wick"] / safe_range
    result["lower_wick_to_range"] = result["lower_wick"] / safe_range

    for window in [12, 24]:
        result[f"highest_high_{window}"] = result["high"].rolling(window).max().shift(1)
        result[f"lowest_low_{window}"] = result["low"].rolling(window).min().shift(1)
        result[f"breakout_high_{window}"] = (result["close"] > result[f"highest_high_{window}"]).astype(int)
        result[f"breakout_low_{window}"] = (result["close"] < result[f"lowest_low_{window}"]).astype(int)

    result["hour"] = result["time"].dt.hour
    result["day_of_week"] = result["time"].dt.dayofweek
    result["is_asia_session"] = result["hour"].between(0, 7).astype(int)
    result["is_london_session"] = result["hour"].between(7, 15).astype(int)
    result["is_newyork_session"] = result["hour"].between(13, 21).astype(int)
    result["is_london_newyork_overlap"] = result["hour"].between(13, 15).astype(int)

    result["spread_to_atr"] = result["spread"] / result["atr_14"].replace(0, np.nan)

    # --- Multi-Timeframe Volatility Regime ---
    atr_5 = AverageTrueRange(result["high"], result["low"], result["close"], window=5).average_true_range()
    atr_28 = AverageTrueRange(result["high"], result["low"], result["close"], window=28).average_true_range()
    result["atr_5"] = atr_5
    result["atr_28"] = atr_28
    result["atr_5_to_14"] = atr_5 / result["atr_14"].replace(0, np.nan)
    result["atr_14_to_28"] = result["atr_14"] / atr_28.replace(0, np.nan)
    result["rolling_std_6"] = result["return_1"].rolling(6).std()
    result["vol_regime"] = result["rolling_std_6"] / result["rolling_std_24"].replace(0, np.nan)
    result["vol_expansion"] = (atr_5 > result["atr_14"]).astype(int)
    result["atr_percentile_50"] = (result["atr_14"] > result["atr_14"].rolling(50).median()).astype(int)
    result["atr_rank_100"] = rolling_percentile(result["atr_14"], 100)
    result["atr_rank_288"] = rolling_percentile(result["atr_14"], 288)
    result["realized_vol_rank_100"] = rolling_percentile(result["rolling_std_24"], 100)
    result["vol_compression"] = (result["atr_rank_100"] < 0.25).astype(int)
    result["vol_breakout_regime"] = ((result["atr_5_to_14"] > 1.1) & (result["atr_rank_100"] > 0.65)).astype(int)

    # --- Price Structure / Key Levels ---
    for window in [48, 96]:
        result[f"highest_high_{window}"] = result["high"].rolling(window).max().shift(1)
        result[f"lowest_low_{window}"] = result["low"].rolling(window).min().shift(1)
        result[f"breakout_high_{window}"] = (result["close"] > result[f"highest_high_{window}"]).astype(int)
        result[f"breakout_low_{window}"] = (result["close"] < result[f"lowest_low_{window}"]).astype(int)
    for w in [12, 24, 48, 96]:
        hh = result[f"highest_high_{w}"] if f"highest_high_{w}" in result.columns else result["high"].rolling(w).max().shift(1)
        ll = result[f"lowest_low_{w}"] if f"lowest_low_{w}" in result.columns else result["low"].rolling(w).min().shift(1)
        safe_range_w = (hh - ll).replace(0, np.nan)
        result[f"close_position_in_range_{w}"] = (result["close"] - ll) / safe_range_w
        result[f"dist_from_high_{w}_atr"] = (result["close"] - hh) / result["atr_14"].replace(0, np.nan)
        result[f"dist_from_low_{w}_atr"] = (result["close"] - ll) / result["atr_14"].replace(0, np.nan)
        result[f"range_width_{w}_atr"] = (hh - ll) / result["atr_14"].replace(0, np.nan)
    result = result.copy()
    result["dist_from_high_24"] = result["dist_from_high_24_atr"]
    result["dist_from_low_24"] = result["dist_from_low_24_atr"]
    result["breakout_high_retest_24"] = (
        (result["breakout_high_24"].rolling(3, min_periods=1).max().shift(1) > 0)
        & (result["dist_from_high_24_atr"].abs() < 0.35)
    ).astype(int)
    result["breakout_low_retest_24"] = (
        (result["breakout_low_24"].rolling(3, min_periods=1).max().shift(1) > 0)
        & (result["dist_from_low_24_atr"].abs() < 0.35)
    ).astype(int)
    result["trend_range_regime"] = (
        ((result["adx_14"] > 25) & (result["range_width_48_atr"] > result["range_width_48_atr"].rolling(100, min_periods=20).median())).astype(int)
        - ((result["adx_14"] < 18) & (result["range_width_48_atr"] < result["range_width_48_atr"].rolling(100, min_periods=20).median())).astype(int)
    )
    result = result.copy()

    # --- Momentum Divergence ---
    result["rsi_7"] = RSIIndicator(result["close"], window=7).rsi()
    result["rsi_21"] = RSIIndicator(result["close"], window=21).rsi()
    result["rsi_change_3"] = result["rsi_14"].diff(3)
    result["rsi_change_6"] = result["rsi_14"].diff(6)
    result["price_change_6"] = result["close"].diff(6)
    rsi_dir = (result["rsi_change_6"] > 0).astype(int)
    price_dir = (result["price_change_6"] > 0).astype(int)
    result["rsi_price_divergence"] = (rsi_dir != price_dir).astype(int)
    result["macd_hist_change_3"] = result["macd_hist"].diff(3)
    result["macd_hist_acceleration"] = result["macd_hist"].diff(1).diff(1)
    ema100_slope = result["ema_100"].diff(3) / result["ema_100"].shift(3).replace(0, np.nan)
    result["ema100_slope"] = ema100_slope
    result["ema_slope_alignment"] = (
        (result["ema20_slope"] > 0) & (result["ema50_slope"] > 0) & (ema100_slope > 0)
    ).astype(int) - (
        (result["ema20_slope"] < 0) & (result["ema50_slope"] < 0) & (ema100_slope < 0)
    ).astype(int)

    # --- Volume Profile ---
    result["volume_sma_12"] = result["tick_volume"].rolling(12).mean()
    result["volume_sma_24"] = result["tick_volume"].rolling(24).mean()
    vol_sma = result["volume_sma_12"].replace(0, np.nan)
    result["volume_ratio_12"] = (result["tick_volume"] / vol_sma).fillna(0.0)
    result["volume_change_3"] = result["tick_volume"].pct_change(3).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["volume_rank_100"] = rolling_percentile(result["tick_volume"], 100)
    result["volume_rank_288"] = rolling_percentile(result["tick_volume"], 288)
    result["volume_price_trend"] = (
        (result["tick_volume"] > vol_sma) & (result["return_1"].abs() > result["return_1"].rolling(12).std())
    ).astype(int)

    # --- Candle Patterns ---
    result["doji"] = (result["body_to_range"] < 0.1).astype(int)
    result["hammer"] = (
        (result["lower_wick"] > 2 * result["body_size"]) &
        (result["upper_wick"] < result["body_size"])
    ).astype(int)
    result["shooting_star"] = (
        (result["upper_wick"] > 2 * result["body_size"]) &
        (result["lower_wick"] < result["body_size"])
    ).astype(int)
    result["engulfing"] = (
        (result["body_size"] > result["body_size"].shift(1) * 1.5) &
        (result["candle_range"] > result["candle_range"].shift(1))
    ).astype(int)
    result["consecutive_green"] = (
        (result["close"] > result["open"]) &
        (result["close"].shift(1) > result["open"].shift(1)) &
        (result["close"].shift(2) > result["open"].shift(2))
    ).astype(int)
    result["consecutive_red"] = (
        (result["close"] < result["open"]) &
        (result["close"].shift(1) < result["open"].shift(1)) &
        (result["close"].shift(2) < result["open"].shift(2))
    ).astype(int)
    result = result.copy()

    # --- Mean Reversion ---
    sma20 = result["close"].rolling(20).mean()
    std20 = result["close"].rolling(20).std()
    result["bb_position"] = (result["close"] - sma20) / (2 * std20).replace(0, np.nan)
    session_date = result["time"].dt.date
    vwap_numerator = (result["close"] * result["tick_volume"]).groupby(session_date).cumsum()
    vwap_denominator = result["tick_volume"].groupby(session_date).cumsum()
    result["dist_from_vwap"] = (
        (result["close"] - vwap_numerator / vwap_denominator.replace(0, np.nan)) / result["atr_14"].replace(0, np.nan)
    ).fillna(0.0)

    spread_mean_100 = result["spread_to_atr"].rolling(100, min_periods=20).mean()
    spread_std_100 = result["spread_to_atr"].rolling(100, min_periods=20).std()
    result["spread_rank_100"] = rolling_percentile(result["spread_to_atr"], 100)
    result["spread_zscore_100"] = ((result["spread_to_atr"] - spread_mean_100) / spread_std_100.replace(0, np.nan)).fillna(0.0)
    result["abnormal_spread"] = ((result["spread_rank_100"] > 0.90) | (result["spread_zscore_100"] > 2.0)).astype(int)

    add_higher_timeframe_context(result, "15min", "m15")
    add_higher_timeframe_context(result, "1h", "h1")

    if include_symbol_features:
        symbol_names = list(SYMBOLS.keys())
        result["symbol_name"] = symbol
        result["symbol_id"] = symbol_names.index(symbol) if symbol in symbol_names else -1
        for name in symbol_names:
            result[f"symbol_{name}"] = int(name == symbol)

    result = result.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return result


def process_symbol(symbol: str) -> pd.DataFrame:
    ensure_dirs([symbol])
    filename = symbol_to_filename(symbol)
    input_path = RAW_DATA_DIR / symbol / f"{filename}_m5_raw.csv"
    output_path = PROCESSED_DATA_DIR / symbol / f"{filename}_m5_features.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Raw data not found: {input_path}")
    df = pd.read_csv(input_path)
    features = add_features(df, symbol)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    LOGGER.info("Saved %s rows to %s", len(features), output_path)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate closed-candle technical features.")
    parse_symbol_args(parser)
    args = parser.parse_args()
    for symbol in resolve_symbols(args):
        process_symbol(symbol)


if __name__ == "__main__":
    main()
