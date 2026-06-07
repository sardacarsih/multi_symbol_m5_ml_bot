import argparse
from pathlib import Path

import pandas as pd

from config import LOGS_DIR, REPORTS_DIR


def session_from_hour(hour: int) -> str:
    if 13 <= hour <= 15:
        return "london_newyork_overlap"
    if 7 <= hour <= 15:
        return "london"
    if 13 <= hour <= 21:
        return "newyork"
    if 0 <= hour <= 7:
        return "asia"
    return "other"


def summarize_live_signals(input_path, output_path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Live signals log not found: {input_path}")
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Live signals log is empty: {input_path}")

    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    df["session"] = df["hour"].apply(session_from_hour)
    df["max_trade_prob"] = df[["prob_buy", "prob_sell"]].max(axis=1)

    rows = []
    for symbol, group in df.groupby("symbol"):
        signal_rows = group[group["signal"] != "NO_TRADE"]
        rows.append(
            {
                "symbol": symbol,
                "rows": int(len(group)),
                "signals": int(len(signal_rows)),
                "buy_signals": int((group["signal"] == "BUY").sum()),
                "sell_signals": int((group["signal"] == "SELL").sum()),
                "no_trade_rows": int((group["signal"] == "NO_TRADE").sum()),
                "avg_prob_buy": float(group["prob_buy"].mean()),
                "avg_prob_sell": float(group["prob_sell"].mean()),
                "avg_max_trade_prob": float(group["max_trade_prob"].mean()),
                "max_trade_prob": float(group["max_trade_prob"].max()),
                "avg_spread_to_atr": float(group["spread_to_atr"].mean()),
                "max_spread_to_atr": float(group["spread_to_atr"].max()),
                "top_reason": group["reason"].value_counts().idxmax(),
                "top_session": group["session"].value_counts().idxmax(),
            }
        )

    summary = pd.DataFrame(rows).sort_values("symbol")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    reason_summary = df.groupby(["symbol", "reason"]).size().reset_index(name="rows")
    reason_summary.to_csv(output_path.with_name("live_signal_reason_summary.csv"), index=False)
    session_summary = df.groupby(["symbol", "session", "signal"]).size().reset_index(name="rows")
    session_summary.to_csv(output_path.with_name("live_signal_session_summary.csv"), index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze logs/live_signals.csv.")
    parser.add_argument("--input", default=str(LOGS_DIR / "live_signals.csv"))
    parser.add_argument("--output", default=str(REPORTS_DIR / "live_signal_summary.csv"))
    args = parser.parse_args()
    summary = summarize_live_signals(Path(args.input), Path(args.output))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
