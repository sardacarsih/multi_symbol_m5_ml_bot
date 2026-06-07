import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from backtest_diagnostics import (
    build_threshold_fingerprint,
    signal_diagnostics,
    split_diagnostics,
    threshold_fingerprint_status,
)
from config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, SYMBOLS
from symbols import parse_symbol_args, resolve_symbols
from strategy_filters import simulate_threshold_trades
from threshold_search import decide_signal
from train import chronological_split
from utils import ensure_dirs, load_json, load_model, setup_logger, symbol_to_filename

LOGGER = setup_logger("backtest")
TRADE_LOG_COLUMNS = [
    "symbol",
    "entry_time",
    "exit_time",
    "side",
    "entry",
    "exit",
    "sl",
    "tp",
    "exit_reason",
    "pnl",
    "holding_candles",
    "prob_buy",
    "prob_sell",
    "spread_to_atr",
]


def metrics_from_trades(symbol: str, trades: pd.DataFrame, equity: pd.Series) -> dict:
    if trades.empty:
        return {
            "symbol": symbol,
            "total_trades": 0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_profit": 0.0,
            "max_drawdown": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "expected_value": 0.0,
            "consecutive_losses": 0,
            "average_holding_candles": 0.0,
            "trades_per_day": 0.0,
        }
    pnl = trades["pnl"]
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl < 0].sum()
    drawdown = equity - equity.cummax()
    loss_streak = 0
    max_loss_streak = 0
    for value in pnl:
        loss_streak = loss_streak + 1 if value <= 0 else 0
        max_loss_streak = max(max_loss_streak, loss_streak)
    days = max((pd.to_datetime(trades["entry_time"]).max() - pd.to_datetime(trades["entry_time"]).min()).days, 1)
    return {
        "symbol": symbol,
        "total_trades": int(len(trades)),
        "winrate": float((pnl > 0).mean()),
        "profit_factor": float(gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf") if gross_profit > 0 else 0.0,
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "net_profit": float(pnl.sum()),
        "max_drawdown": float(drawdown.min()),
        "average_win": float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0,
        "average_loss": float(pnl[pnl <= 0].mean()) if (pnl <= 0).any() else 0.0,
        "expected_value": float(pnl.mean()),
        "consecutive_losses": int(max_loss_streak),
        "average_holding_candles": float(trades["holding_candles"].mean()),
        "trades_per_day": float(len(trades) / days),
    }


def save_equity_curve(equity: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    equity.reset_index(drop=True).plot()
    plt.title("Equity Curve")
    plt.xlabel("Trade")
    plt.ylabel("Net Profit")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def backtest_symbol(symbol: str, slippage: float = 0.0) -> tuple[dict, pd.DataFrame, pd.Series]:
    ensure_dirs([symbol])
    cfg = SYMBOLS[symbol]
    filename = symbol_to_filename(symbol)
    df_path = PROCESSED_DATA_DIR / symbol / f"{filename}_m5_labeled.csv"
    model_path = MODELS_DIR / symbol / f"{filename}_m5_xgboost.joblib"
    columns_path = MODELS_DIR / symbol / "feature_columns.json"
    threshold_path = MODELS_DIR / symbol / "best_threshold.json"
    df = pd.read_csv(df_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    train_df, val_df, test_df = chronological_split(df)
    test_df = test_df.reset_index(drop=True)
    model = load_model(model_path)
    columns = load_json(columns_path)
    thresholds = load_json(threshold_path, default={}) or {}
    buy_threshold = float(thresholds.get("buy_threshold", cfg["buy_threshold"]))
    sell_threshold = float(thresholds.get("sell_threshold", cfg["sell_threshold"]))
    threshold_source = "best_threshold.json" if threshold_path.exists() else "config_defaults"
    current_fingerprint = build_threshold_fingerprint(model_path, df_path, columns, test_df)
    fingerprint_status = threshold_fingerprint_status(thresholds, current_fingerprint)
    if threshold_source == "best_threshold.json" and fingerprint_status != "valid":
        LOGGER.warning(
            "%s threshold artifact is %s for current model/data. Run: python src/threshold_search.py --symbols %s",
            symbol,
            fingerprint_status,
            symbol,
        )
    proba = model.predict_proba(test_df[columns])

    trades_df = simulate_threshold_trades(test_df, proba, cfg, buy_threshold, sell_threshold, decide_signal, slippage)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    else:
        trades_df.insert(0, "symbol", symbol)
        trades_df = trades_df[TRADE_LOG_COLUMNS]
    equity = trades_df["pnl"].cumsum() if not trades_df.empty else pd.Series(dtype=float)
    report_dir = REPORTS_DIR / symbol
    trades_df.to_csv(report_dir / "trade_log.csv", index=False)
    metrics = metrics_from_trades(symbol, trades_df, equity)
    diagnostics = {
        **split_diagnostics(df, train_df, val_df, test_df),
        "threshold_source": threshold_source,
        "threshold_fingerprint_status": fingerprint_status,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        **signal_diagnostics(test_df, proba, cfg, buy_threshold, sell_threshold, decide_signal, len(trades_df)),
    }
    pd.DataFrame([{**metrics, **diagnostics}]).to_csv(report_dir / "backtest_result.csv", index=False)
    save_equity_curve(equity, report_dir / "equity_curve.png")
    LOGGER.info(
        "%s backtest trades: %s net: %.4f test_rows: %s signals raw/session/spread: %s/%s/%s threshold: %.2f/%.2f (%s, %s)",
        symbol,
        metrics["total_trades"],
        metrics["net_profit"],
        diagnostics["test_rows"],
        diagnostics["signals_raw"],
        diagnostics["signals_after_session"],
        diagnostics["signals_after_spread"],
        buy_threshold,
        sell_threshold,
        threshold_source,
        fingerprint_status,
    )
    return metrics, trades_df, equity


def portfolio_summary(results: list[tuple[dict, pd.DataFrame, pd.Series]]) -> None:
    rows = [item[0] for item in results]
    all_trades = pd.concat([item[1] for item in results if not item[1].empty], ignore_index=True) if any(not item[1].empty for item in results) else pd.DataFrame()
    gross_profit = all_trades.loc[all_trades["pnl"] > 0, "pnl"].sum() if not all_trades.empty else 0.0
    gross_loss = all_trades.loc[all_trades["pnl"] < 0, "pnl"].sum() if not all_trades.empty else 0.0
    curves = {row["symbol"]: equity.reset_index(drop=True) for row, _, equity in results if not equity.empty}
    corr = pd.DataFrame(curves).corr().to_json() if len(curves) > 1 else "{}"
    exposures = all_trades["symbol"].value_counts(normalize=True).to_dict() if not all_trades.empty else {}
    total_equity = all_trades.sort_values("exit_time")["pnl"].cumsum() if not all_trades.empty else pd.Series(dtype=float)
    drawdown = total_equity - total_equity.cummax() if not total_equity.empty else pd.Series([0.0])
    summary = {
        "total_trades": int(sum(row["total_trades"] for row in rows)),
        "total_net_profit": float(sum(row["net_profit"] for row in rows)),
        "combined_profit_factor": float(gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf") if gross_profit > 0 else 0.0,
        "combined_max_drawdown": float(drawdown.min()),
        "equity_curve_correlation": corr,
        "exposure_per_symbol": exposures,
    }
    pd.DataFrame([summary]).to_csv(REPORTS_DIR / "portfolio_backtest_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model threshold backtests.")
    parse_symbol_args(parser)
    parser.add_argument("--slippage", type=float, default=0.0)
    args = parser.parse_args()
    symbols = resolve_symbols(args)
    results = [backtest_symbol(symbol, args.slippage) for symbol in symbols]
    portfolio_summary(results)


if __name__ == "__main__":
    main()
