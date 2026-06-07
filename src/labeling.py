import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR, REPORTS_DIR, SYMBOLS
from symbols import parse_symbol_args, resolve_symbols
from utils import ensure_dirs, setup_logger, symbol_to_filename

LOGGER = setup_logger("labeling")

LABEL_NAMES = {0: "NO_TRADE", 1: "BUY", 2: "SELL"}


def atr_barrier_labels(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    cfg = SYMBOLS[symbol]
    lookahead = int(cfg["lookahead_candles"])
    tp_mult = float(cfg["label_tp_atr_mult"])
    sl_mult = float(cfg["label_sl_atr_mult"])
    result = df.copy().reset_index(drop=True)
    labels = np.zeros(len(result), dtype=int)

    for idx in range(len(result) - lookahead):
        close = result.at[idx, "close"]
        atr = result.at[idx, "atr_14"]
        buy_tp = close + tp_mult * atr
        buy_sl = close - sl_mult * atr
        sell_tp = close - tp_mult * atr
        sell_sl = close + sl_mult * atr

        label = 0
        for future_idx in range(idx + 1, idx + lookahead + 1):
            high = result.at[future_idx, "high"]
            low = result.at[future_idx, "low"]
            buy_hit_sl = low <= buy_sl
            buy_hit_tp = high >= buy_tp
            sell_hit_sl = high >= sell_sl
            sell_hit_tp = low <= sell_tp

            if buy_hit_tp and not buy_hit_sl:
                label = 1
                break
            if sell_hit_tp and not sell_hit_sl:
                label = 2
                break
            if buy_hit_sl or sell_hit_sl:
                label = 0
                break
        labels[idx] = label

    result["label"] = labels
    return result.iloc[: len(result) - lookahead].copy()


def barrier_ambiguity_flags(df: pd.DataFrame, symbol: str) -> pd.Series:
    required = {"close", "atr_14", "high", "low"}
    if not required.issubset(df.columns):
        return pd.Series(np.zeros(len(df), dtype=bool), name="ambiguous_barrier")
    cfg = SYMBOLS[symbol]
    lookahead = int(cfg["lookahead_candles"])
    tp_mult = float(cfg["label_tp_atr_mult"])
    sl_mult = float(cfg["label_sl_atr_mult"])
    result = df.reset_index(drop=True)
    ambiguous = np.zeros(len(result), dtype=bool)

    for idx in range(len(result) - lookahead):
        close = result.at[idx, "close"]
        atr = result.at[idx, "atr_14"]
        buy_tp = close + tp_mult * atr
        buy_sl = close - sl_mult * atr
        sell_tp = close - tp_mult * atr
        sell_sl = close + sl_mult * atr

        for future_idx in range(idx + 1, idx + lookahead + 1):
            high = result.at[future_idx, "high"]
            low = result.at[future_idx, "low"]
            buy_hit_sl = low <= buy_sl
            buy_hit_tp = high >= buy_tp
            sell_hit_sl = high >= sell_sl
            sell_hit_tp = low <= sell_tp
            if (buy_hit_tp and buy_hit_sl) or (sell_hit_tp and sell_hit_sl) or (buy_hit_tp and sell_hit_tp):
                ambiguous[idx] = True
                break
            if buy_hit_tp or sell_hit_tp or buy_hit_sl or sell_hit_sl:
                break

    return pd.Series(ambiguous[: len(result) - lookahead], name="ambiguous_barrier")


def label_diagnostics(labeled: pd.DataFrame, source_df: pd.DataFrame, symbol: str) -> dict:
    rows = len(labeled)
    counts = labeled["label"].value_counts().sort_index()
    label_counts = {LABEL_NAMES.get(int(label), str(label)): int(count) for label, count in counts.items()}
    label_pct = {name: (count / rows if rows else 0.0) for name, count in label_counts.items()}
    buy_count = int(counts.get(1, 0))
    sell_count = int(counts.get(2, 0))
    ambiguous = barrier_ambiguity_flags(source_df, symbol)

    monthly = pd.DataFrame()
    if "time" in labeled.columns and not labeled.empty:
        monthly_df = labeled[["time", "label"]].copy()
        monthly_df["time"] = pd.to_datetime(monthly_df["time"])
        monthly = (
            monthly_df.set_index("time")["label"]
            .groupby(pd.Grouper(freq="ME"))
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            .rename(columns=LABEL_NAMES)
        )

    regime_distribution = {}
    regime_col = SYMBOLS[symbol].get("regime_filter_col")
    if regime_col and regime_col in labeled.columns:
        regime_distribution = (
            labeled.groupby(regime_col)["label"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            .rename(columns=LABEL_NAMES)
            .to_dict(orient="index")
        )

    return {
        "rows": rows,
        "label_counts": label_counts,
        "label_pct": label_pct,
        "buy_sell_imbalance_ratio": float(buy_count / sell_count) if sell_count else None,
        "ambiguous_barrier_count": int(ambiguous.sum()),
        "ambiguous_barrier_pct": float(ambiguous.mean()) if len(ambiguous) else 0.0,
        "monthly_distribution": monthly.round(6).reset_index().astype({"time": str}).to_dict(orient="records") if not monthly.empty else [],
        "regime_distribution": regime_distribution,
    }


def write_label_diagnostics(labeled: pd.DataFrame, source_df: pd.DataFrame, symbol: str, report_dir: Path | None = None) -> dict:
    diagnostics = label_diagnostics(labeled, source_df, symbol)
    output_dir = report_dir or (REPORTS_DIR / symbol)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "rows": diagnostics["rows"],
        "buy_sell_imbalance_ratio": diagnostics["buy_sell_imbalance_ratio"],
        "ambiguous_barrier_count": diagnostics["ambiguous_barrier_count"],
        "ambiguous_barrier_pct": diagnostics["ambiguous_barrier_pct"],
        **{f"count_{key.lower()}": value for key, value in diagnostics["label_counts"].items()},
        **{f"pct_{key.lower()}": value for key, value in diagnostics["label_pct"].items()},
    }]).to_csv(output_dir / "label_diagnostics_summary.csv", index=False)
    pd.DataFrame(diagnostics["monthly_distribution"]).to_csv(output_dir / "label_diagnostics_monthly.csv", index=False)
    return diagnostics


def process_symbol(symbol: str) -> pd.DataFrame:
    ensure_dirs([symbol])
    filename = symbol_to_filename(symbol)
    input_path = PROCESSED_DATA_DIR / symbol / f"{filename}_m5_features.csv"
    output_path = PROCESSED_DATA_DIR / symbol / f"{filename}_m5_labeled.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Feature data not found: {input_path}")
    df = pd.read_csv(input_path)
    labeled = atr_barrier_labels(df, symbol)
    labeled.to_csv(output_path, index=False)
    write_label_diagnostics(labeled, df, symbol)
    distribution = labeled["label"].value_counts(normalize=False).sort_index().to_dict()
    LOGGER.info("%s label distribution: %s", symbol, distribution)
    LOGGER.info("Saved %s rows to %s", len(labeled), output_path)
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ATR barrier labels.")
    parse_symbol_args(parser)
    args = parser.parse_args()
    for symbol in resolve_symbols(args):
        process_symbol(symbol)


if __name__ == "__main__":
    main()
