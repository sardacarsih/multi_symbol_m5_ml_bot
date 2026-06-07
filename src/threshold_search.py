import argparse
from itertools import product

import numpy as np
import pandas as pd

from config import (
    MIN_THRESHOLD_COMBINED_PRECISION,
    MIN_THRESHOLD_PROFIT_FACTOR,
    MIN_THRESHOLD_SIGNALS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SYMBOLS,
)
from backtest_diagnostics import build_threshold_fingerprint
from symbols import parse_symbol_args, resolve_symbols
from strategy_filters import session_allowed, simulate_threshold_trades, trade_metrics
from train import chronological_split
from utils import ensure_dirs, load_json, load_model, save_json, setup_logger, symbol_to_filename

LOGGER = setup_logger("threshold_search")


def decide_signal(prob_buy: float, prob_sell: float, buy_threshold: float, sell_threshold: float) -> int:
    if prob_buy >= buy_threshold and prob_buy > prob_sell:
        return 1
    if prob_sell >= sell_threshold and prob_sell > prob_buy:
        return 2
    return 0


def load_symbol_artifacts(symbol: str):
    filename = symbol_to_filename(symbol)
    df_path = PROCESSED_DATA_DIR / symbol / f"{filename}_m5_labeled.csv"
    model_path = MODELS_DIR / symbol / f"{filename}_m5_xgboost.joblib"
    cols_path = MODELS_DIR / symbol / "feature_columns.json"
    if not df_path.exists() or not model_path.exists() or not cols_path.exists():
        raise FileNotFoundError(f"Missing data/model artifacts for {symbol}")
    df = pd.read_csv(df_path)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True), load_model(model_path), load_json(cols_path), model_path, df_path


def expected_value(row: pd.Series, cfg: dict) -> float:
    if row["combined_precision"] == 0 or row["total_signals"] == 0:
        return 0.0
    avg_reward = float(cfg["live_tp_atr_mult"])
    avg_risk = float(cfg["live_sl_atr_mult"])
    return row["combined_precision"] * avg_reward - (1 - row["combined_precision"]) * avg_risk


def search_symbol(symbol: str) -> pd.DataFrame:
    ensure_dirs([symbol])
    df, model, columns, model_path, df_path = load_symbol_artifacts(symbol)
    _, _, test_df = chronological_split(df)
    test_df = test_df.reset_index(drop=True)
    proba = model.predict_proba(test_df[columns])
    y_true = test_df["label"].to_numpy()
    rows = []
    days = max((test_df["time"].max() - test_df["time"].min()).days, 1)
    session_mask = np.array([session_allowed(row, SYMBOLS[symbol]) for _, row in test_df.iterrows()])
    min_signals = int(SYMBOLS[symbol].get("min_threshold_signals", MIN_THRESHOLD_SIGNALS))
    threshold_min = float(SYMBOLS[symbol].get("threshold_search_min", 0.55))
    threshold_max = float(SYMBOLS[symbol].get("threshold_search_max", 0.80))
    threshold_step = float(SYMBOLS[symbol].get("threshold_search_step", 0.01))
    threshold_values = np.round(np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step), 2)

    for buy_threshold, sell_threshold in product(threshold_values, repeat=2):
        signals = np.array([decide_signal(p[1], p[2], buy_threshold, sell_threshold) for p in proba])
        signals = np.where(session_mask, signals, 0)
        mask = signals > 0
        buy_mask = signals == 1
        sell_mask = signals == 2
        total = int(mask.sum())
        buy_precision = float((y_true[buy_mask] == 1).mean()) if buy_mask.any() else 0.0
        sell_precision = float((y_true[sell_mask] == 2).mean()) if sell_mask.any() else 0.0
        combined_precision = float(((y_true[mask] == signals[mask]).mean())) if mask.any() else 0.0
        should_simulate = total >= min_signals
        trades = (
            simulate_threshold_trades(test_df, proba, SYMBOLS[symbol], buy_threshold, sell_threshold, decide_signal)
            if should_simulate
            else pd.DataFrame()
        )
        rows.append(
            {
                "buy_threshold": round(float(buy_threshold), 2),
                "sell_threshold": round(float(sell_threshold), 2),
                "total_signals": total,
                "buy_precision": buy_precision,
                "sell_precision": sell_precision,
                "combined_precision": combined_precision,
                "trade_frequency": total / len(signals) if len(signals) else 0.0,
                "average_confidence": float(np.max(proba[mask][:, 1:3], axis=1).mean()) if mask.any() else 0.0,
                "estimated_trades_per_day": total / days,
                "backtest_evaluated": should_simulate,
                **trade_metrics(trades),
            }
        )

    result = pd.DataFrame(rows)
    result["expected_value"] = result.apply(lambda row: expected_value(row, SYMBOLS[symbol]), axis=1)
    result["eligible"] = (
        (result["backtest_trades"] >= min_signals)
        & (result["backtest_profit_factor"] >= MIN_THRESHOLD_PROFIT_FACTOR)
        & (result["backtest_net_profit"] > 0)
    )
    result = result.sort_values(
        [
            "eligible",
            "backtest_evaluated",
            "backtest_net_profit",
            "backtest_profit_factor",
            "backtest_max_drawdown",
            "combined_precision",
            "backtest_trades",
        ],
        ascending=[False, False, False, False, False, False, False],
    )
    report_path = REPORTS_DIR / symbol / "threshold_search.csv"
    result.to_csv(report_path, index=False)
    best = result.iloc[0].to_dict()
    best["artifact_fingerprint"] = build_threshold_fingerprint(model_path, df_path, columns, test_df)
    save_json(best, MODELS_DIR / symbol / "best_threshold.json")
    if not bool(best["eligible"]):
        LOGGER.warning(
            "%s no threshold met min_trades=%s min_pf=%.2f positive_net; saved best fallback",
            symbol,
            min_signals,
            MIN_THRESHOLD_PROFIT_FACTOR,
        )
    LOGGER.info(
        "Best %s threshold: buy %.2f sell %.2f trades %s net %.4f pf %.3f precision %.3f eligible %s",
        symbol,
        best["buy_threshold"],
        best["sell_threshold"],
        int(best["backtest_trades"]),
        best["backtest_net_profit"],
        best["backtest_profit_factor"],
        best["combined_precision"],
        bool(best["eligible"]),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Search confidence thresholds.")
    parse_symbol_args(parser)
    args = parser.parse_args()
    for symbol in resolve_symbols(args):
        search_symbol(symbol)


if __name__ == "__main__":
    main()
