import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR, REPORTS_DIR, SYMBOLS
from symbols import parse_symbol_args, resolve_symbols
from utils import ensure_dirs, setup_logger, symbol_to_filename

LOGGER = setup_logger("labeling")

LABEL_NAMES = {0: "NO_TRADE", 1: "BUY", 2: "SELL"}


def _barrier_outcome_frame(
    df: pd.DataFrame,
    symbol: str,
    lookahead: int | None = None,
    tp_mult: float | None = None,
    sl_mult: float | None = None,
    spread_adjusted: bool = False,
) -> pd.DataFrame:
    cfg = SYMBOLS[symbol]
    lookahead = int(lookahead if lookahead is not None else cfg["lookahead_candles"])
    tp_mult = float(tp_mult if tp_mult is not None else cfg["label_tp_atr_mult"])
    sl_mult = float(sl_mult if sl_mult is not None else cfg["label_sl_atr_mult"])
    result = df.copy().reset_index(drop=True)
    rows = []

    for idx in range(len(result) - lookahead):
        close = float(result.at[idx, "close"])
        atr = float(result.at[idx, "atr_14"])
        spread = float(result.at[idx, "spread"]) if spread_adjusted and "spread" in result.columns else 0.0
        buy_entry = close + spread
        sell_entry = close
        buy_tp = buy_entry + tp_mult * atr
        buy_sl = buy_entry - sl_mult * atr
        sell_tp = sell_entry - tp_mult * atr
        sell_sl = sell_entry + sl_mult * atr

        label = 0
        first_side = "NO_TRADE"
        first_reason = "TIME"
        time_to_event = lookahead
        ambiguous = False
        buy_result = "TIME"
        sell_result = "TIME"

        for future_idx in range(idx + 1, idx + lookahead + 1):
            high = float(result.at[future_idx, "high"])
            low = float(result.at[future_idx, "low"])
            buy_hit_sl = low <= buy_sl
            buy_hit_tp = high >= buy_tp
            sell_hit_sl = high >= sell_sl
            sell_hit_tp = low <= sell_tp

            if buy_hit_tp and buy_hit_sl:
                ambiguous = True
            if sell_hit_tp and sell_hit_sl:
                ambiguous = True
            if buy_hit_tp and sell_hit_tp:
                ambiguous = True

            if buy_hit_sl:
                buy_result = "SL"
            elif buy_hit_tp:
                buy_result = "TP"
            if sell_hit_sl:
                sell_result = "SL"
            elif sell_hit_tp:
                sell_result = "TP"

            if buy_hit_tp and not buy_hit_sl:
                label = 1
                first_side = "BUY"
                first_reason = "TP"
                time_to_event = future_idx - idx
                break
            if sell_hit_tp and not sell_hit_sl:
                label = 2
                first_side = "SELL"
                first_reason = "TP"
                time_to_event = future_idx - idx
                break
            if buy_hit_sl or sell_hit_sl:
                first_reason = "SL"
                time_to_event = future_idx - idx
                break

        future_close = float(result.at[idx + lookahead, "close"])
        rows.append(
            {
                "time": result.at[idx, "time"] if "time" in result.columns else idx,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "first_side": first_side,
                "first_reason": first_reason,
                "time_to_event": time_to_event,
                "ambiguous_barrier": ambiguous,
                "buy_result": buy_result,
                "sell_result": sell_result,
                "forward_return": future_close / close - 1 if close else 0.0,
                "forward_return_atr": (future_close - close) / atr if atr else 0.0,
                "spread_adjusted": spread_adjusted,
                "lookahead": lookahead,
            }
        )
    return pd.DataFrame(rows)


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

    execution_outcomes = _barrier_outcome_frame(source_df, symbol, spread_adjusted=True)
    execution_outcomes = execution_outcomes.iloc[:rows].reset_index(drop=True)
    label_values = labeled["label"].reset_index(drop=True).iloc[: len(execution_outcomes)]
    execution_agreement = float((label_values == execution_outcomes["label"]).mean()) if len(execution_outcomes) else 0.0

    forward_distribution = {}
    if len(labeled):
        forward_distribution = (
            labeled.assign(label_name=labeled["label"].map(LABEL_NAMES))
            .groupby("label_name")["return_1" if "return_1" in labeled.columns else "label"]
            .count()
            .to_dict()
        )

    side_expectancy = {}
    if len(execution_outcomes):
        for label_value, side in [(1, "BUY"), (2, "SELL")]:
            side_rows = execution_outcomes[label_values == label_value]
            if side_rows.empty:
                side_expectancy[side] = {"rows": 0, "tp_rate": 0.0, "avg_time_to_event": 0.0, "avg_forward_return_atr": 0.0}
            else:
                side_expectancy[side] = {
                    "rows": int(len(side_rows)),
                    "tp_rate": float((side_rows["first_reason"] == "TP").mean()),
                    "avg_time_to_event": float(side_rows["time_to_event"].mean()),
                    "avg_forward_return_atr": float(side_rows["forward_return_atr"].mean()),
                }

    entropy = 0.0
    if rows:
        probabilities = np.array(list(label_pct.values()), dtype=float)
        probabilities = probabilities[probabilities > 0]
        entropy = float(-(probabilities * np.log(probabilities)).sum())

    return {
        "rows": rows,
        "label_counts": label_counts,
        "label_pct": label_pct,
        "buy_sell_imbalance_ratio": float(buy_count / sell_count) if sell_count else None,
        "ambiguous_barrier_count": int(ambiguous.sum()),
        "ambiguous_barrier_pct": float(ambiguous.mean()) if len(ambiguous) else 0.0,
        "label_entropy": entropy,
        "execution_adjusted_label_agreement": execution_agreement,
        "execution_adjusted_ambiguous_pct": float(execution_outcomes["ambiguous_barrier"].mean()) if len(execution_outcomes) else 0.0,
        "side_expectancy": side_expectancy,
        "forward_distribution": forward_distribution,
        "monthly_distribution": monthly.round(6).reset_index().astype({"time": str}).to_dict(orient="records") if not monthly.empty else [],
        "regime_distribution": regime_distribution,
    }


def horizon_label_audit(df: pd.DataFrame, symbol: str, horizons: tuple[int, ...] = (3, 6, 9, 12)) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        outcomes = _barrier_outcome_frame(df, symbol, lookahead=horizon, spread_adjusted=True)
        if outcomes.empty:
            rows.append(
                {
                    "lookahead": horizon,
                    "rows": 0,
                    "pct_no_trade": 0.0,
                    "pct_buy": 0.0,
                    "pct_sell": 0.0,
                    "ambiguous_barrier_pct": 0.0,
                    "label_entropy": 0.0,
                    "avg_time_to_event": 0.0,
                }
            )
            continue
        distribution = outcomes["label"].value_counts(normalize=True).to_dict()
        probabilities = np.array(list(distribution.values()), dtype=float)
        rows.append(
            {
                "lookahead": horizon,
                "rows": int(len(outcomes)),
                "pct_no_trade": float(distribution.get(0, 0.0)),
                "pct_buy": float(distribution.get(1, 0.0)),
                "pct_sell": float(distribution.get(2, 0.0)),
                "ambiguous_barrier_pct": float(outcomes["ambiguous_barrier"].mean()),
                "label_entropy": float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) else 0.0,
                "avg_time_to_event": float(outcomes["time_to_event"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_label_diagnostics(labeled: pd.DataFrame, source_df: pd.DataFrame, symbol: str, report_dir: Path | None = None) -> dict:
    diagnostics = label_diagnostics(labeled, source_df, symbol)
    output_dir = report_dir or (REPORTS_DIR / symbol)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "rows": diagnostics["rows"],
        "buy_sell_imbalance_ratio": diagnostics["buy_sell_imbalance_ratio"],
        "ambiguous_barrier_count": diagnostics["ambiguous_barrier_count"],
        "ambiguous_barrier_pct": diagnostics["ambiguous_barrier_pct"],
        "label_entropy": diagnostics["label_entropy"],
        "execution_adjusted_label_agreement": diagnostics["execution_adjusted_label_agreement"],
        "execution_adjusted_ambiguous_pct": diagnostics["execution_adjusted_ambiguous_pct"],
        **{f"count_{key.lower()}": value for key, value in diagnostics["label_counts"].items()},
        **{f"pct_{key.lower()}": value for key, value in diagnostics["label_pct"].items()},
    }]).to_csv(output_dir / "label_diagnostics_summary.csv", index=False)
    pd.DataFrame(diagnostics["monthly_distribution"]).to_csv(output_dir / "label_diagnostics_monthly.csv", index=False)
    pd.DataFrame(diagnostics["side_expectancy"]).T.reset_index(names="side").to_csv(output_dir / "label_side_expectancy.csv", index=False)
    horizon_label_audit(source_df, symbol).to_csv(output_dir / "label_horizon_audit.csv", index=False)
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
