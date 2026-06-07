import pandas as pd

from config import COOLDOWN_CANDLES
from risk import calculate_atr_sl_tp, check_spread_filter


def session_allowed(row: pd.Series, cfg: dict) -> bool:
    sessions = cfg.get("allowed_entry_sessions", [])
    if not sessions:
        return True
    return any(int(row.get(session, 0)) == 1 for session in sessions)


def simulate_exit(df: pd.DataFrame, start_idx: int, side: str, entry: float, sl: float, tp: float, lookahead: int) -> tuple[int, float, str]:
    end_idx = min(start_idx + lookahead, len(df) - 1)
    for idx in range(start_idx + 1, end_idx + 1):
        high = df.at[idx, "high"]
        low = df.at[idx, "low"]
        if side == "BUY":
            if low <= sl:
                return idx, sl, "SL"
            if high >= tp:
                return idx, tp, "TP"
        else:
            if high >= sl:
                return idx, sl, "SL"
            if low <= tp:
                return idx, tp, "TP"
    return end_idx, df.at[end_idx, "close"], "TIME"


def trade_pnl(side: str, entry: float, exit_price: float, lot: float) -> float:
    multiplier = 1 if side == "BUY" else -1
    return (exit_price - entry) * multiplier * lot


def trade_pnl_money(side: str, entry: float, exit_price: float, lot: float, tick_size: float, tick_value: float) -> float | None:
    if tick_size <= 0 or tick_value <= 0:
        return None
    multiplier = 1 if side == "BUY" else -1
    price_diff = (exit_price - entry) * multiplier
    return price_diff / tick_size * tick_value * lot


def broker_specs(row: pd.Series, cfg: dict) -> tuple[float, float, bool]:
    tick_size = float(row.get("tick_size", cfg.get("tick_size", row.get("point", 0.0))) or 0.0)
    tick_value = float(row.get("tick_value", cfg.get("tick_value", 0.0)) or 0.0)
    return tick_size, tick_value, tick_size > 0 and tick_value > 0


def pnl_fields(side: str, entry: float, exit_price: float, lot: float, tick_size: float, tick_value: float, risk_distance: float) -> dict:
    pnl_price_lot = trade_pnl(side, entry, exit_price, lot)
    pnl_money = trade_pnl_money(side, entry, exit_price, lot, tick_size, tick_value)
    risk_money = (risk_distance / tick_size * tick_value * lot) if tick_size > 0 and tick_value > 0 and risk_distance > 0 else None
    return {
        "pnl": pnl_price_lot,
        "pnl_price_lot": pnl_price_lot,
        "pnl_money": pnl_money,
        "risk_money": risk_money,
        "pnl_per_risk": (pnl_money / risk_money) if pnl_money is not None and risk_money and risk_money > 0 else 0.0,
        "lot": lot,
        "tick_size": tick_size,
        "tick_value": tick_value,
        "pnl_money_available": bool(pnl_money is not None),
    }


def metric_pnl_series(trades: pd.DataFrame) -> pd.Series:
    if "pnl_money" not in trades.columns:
        return trades["pnl"]
    pnl_money = pd.to_numeric(trades["pnl_money"], errors="coerce")
    if pnl_money.notna().any():
        return pnl_money.fillna(pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0))
    return trades["pnl"]


def simulate_threshold_trades(
    df: pd.DataFrame,
    probabilities,
    cfg: dict,
    buy_threshold: float,
    sell_threshold: float,
    decide_signal,
    slippage: float = 0.0,
) -> pd.DataFrame:
    trades = []
    cooldown_until = -1
    idx = 0
    while idx < len(df) - 1:
        if idx <= cooldown_until:
            idx += 1
            continue
        row = df.iloc[idx]
        if not session_allowed(row, cfg):
            idx += 1
            continue
        if not check_spread_filter(float(row["spread_to_atr"]), float(cfg["max_spread_to_atr"])):
            idx += 1
            continue
        signal = decide_signal(float(probabilities[idx][1]), float(probabilities[idx][2]), buy_threshold, sell_threshold)
        if signal == 0:
            idx += 1
            continue

        side = "BUY" if signal == 1 else "SELL"
        spread = float(row.get("spread", 0.0))
        entry = float(row["close"]) + spread + slippage if side == "BUY" else float(row["close"]) - slippage
        levels = calculate_atr_sl_tp(entry, float(row["atr_14"]), side, cfg["live_sl_atr_mult"], cfg["live_tp_atr_mult"])
        exit_idx, exit_price, exit_reason = simulate_exit(df, idx, side, entry, levels.sl, levels.tp, int(cfg["lookahead_candles"]))
        lot = float(cfg["default_lot"])
        tick_size, tick_value, _ = broker_specs(row, cfg)
        risk_distance = abs(entry - levels.sl)
        trades.append(
            {
                "entry_index": idx,
                "exit_index": exit_idx,
                "entry_time": row["time"],
                "exit_time": df.at[exit_idx, "time"],
                "side": side,
                "entry": entry,
                "exit": exit_price,
                "sl": levels.sl,
                "tp": levels.tp,
                "exit_reason": exit_reason,
                **pnl_fields(side, entry, exit_price, lot, tick_size, tick_value, risk_distance),
                "holding_candles": exit_idx - idx,
                "prob_buy": float(probabilities[idx][1]),
                "prob_sell": float(probabilities[idx][2]),
                "spread_to_atr": float(row["spread_to_atr"]),
            }
        )
        cooldown_until = exit_idx + COOLDOWN_CANDLES
        idx = exit_idx + 1

    return pd.DataFrame(trades)


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "backtest_trades": 0,
            "backtest_winrate": 0.0,
            "backtest_profit_factor": 0.0,
            "backtest_net_profit": 0.0,
            "backtest_net_profit_price_lot": 0.0,
            "backtest_net_profit_money": 0.0,
            "backtest_avg_pnl_per_trade": 0.0,
            "backtest_max_drawdown": 0.0,
            "backtest_expected_value": 0.0,
            "backtest_pnl_money_available": False,
        }
    pnl = metric_pnl_series(trades)
    pnl_price_lot = pd.to_numeric(trades.get("pnl_price_lot", trades["pnl"]), errors="coerce").fillna(0.0)
    pnl_money_available = bool("pnl_money" in trades.columns and pd.to_numeric(trades["pnl_money"], errors="coerce").notna().any())
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl < 0].sum()
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "backtest_trades": int(len(trades)),
        "backtest_winrate": float((pnl > 0).mean()),
        "backtest_profit_factor": float(gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf") if gross_profit > 0 else 0.0,
        "backtest_net_profit": float(pnl.sum()),
        "backtest_net_profit_price_lot": float(pnl_price_lot.sum()),
        "backtest_net_profit_money": float(pnl.sum()) if pnl_money_available else 0.0,
        "backtest_avg_pnl_per_trade": float(pnl.mean()),
        "backtest_max_drawdown": float(drawdown.min()),
        "backtest_expected_value": float(pnl.mean()),
        "backtest_pnl_money_available": pnl_money_available,
    }
