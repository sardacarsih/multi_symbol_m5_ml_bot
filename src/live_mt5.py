import argparse
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import (
    COOLDOWN_CANDLES,
    DEVIATION,
    LOGS_DIR,
    MAGIC_NUMBER,
    MAX_TOTAL_RISK,
    MAX_DAILY_LOSS,
    MAX_DAILY_TRADES,
    MAX_CONSECUTIVE_LOSSES,
    MAX_OPEN_POSITIONS_PER_SYMBOL,
    MAX_TOTAL_OPEN_POSITIONS,
    MODELS_DIR,
    REQUIRE_ELIGIBLE_THRESHOLD_FOR_LIVE,
    SYMBOLS,
    USE_RISK_PERCENT,
)
from download_mt5_data import connect_mt5, ensure_symbol_visible, mt5_timeframe
from features import add_features
from risk import (
    calculate_atr_sl_tp,
    calculate_lot_by_risk,
    check_margin,
    check_open_positions,
    check_spread_filter,
    check_stop_level,
    normalize_lot,
)
from symbols import parse_symbol_args, resolve_symbols, validate_symbol
from strategy_filters import session_allowed
from threshold_search import decide_signal
from utils import append_csv_row, get_closed_candles_only, load_json, load_model, setup_logger, symbol_to_filename
from production import canonical_meta_failures
from live_dashboard import (
    LiveSnapshot,
    MarketSnapshot,
    SignalSnapshot,
    account_snapshot,
    positions_snapshot,
    trend_snapshot,
)

LOGGER = setup_logger("live_mt5")
LAST_CLOSE_INDEX: dict[str, int] = {}
LIVE_STATE_PATH = LOGS_DIR / "live_state.json"


def fetch_recent_candles(mt5, symbol: str, count: int = 300) -> pd.DataFrame:
    cfg = SYMBOLS[symbol]
    rates = mt5.copy_rates_from_pos(cfg["mt5_symbol"], mt5_timeframe(mt5, cfg["timeframe"]), 0, count)
    if rates is None:
        code, message = mt5.last_error()
        raise RuntimeError(f"{symbol}: copy_rates_from_pos failed: {code} {message}")
    df = pd.DataFrame(rates)
    if df.empty:
        raise RuntimeError(f"{symbol}: no recent candles")
    df["time"] = pd.to_datetime(df["time"], unit="s")
    info = mt5.symbol_info(cfg["mt5_symbol"])
    df["point"] = getattr(info, "point", 0.0) or 0.0
    return get_closed_candles_only(df)


def open_positions(mt5, mt5_symbol: str) -> tuple[int, int]:
    positions = mt5.positions_get()
    if positions is None:
        return 0, 0
    bot_positions = [pos for pos in positions if getattr(pos, "magic", None) == MAGIC_NUMBER]
    symbol_positions = [pos for pos in bot_positions if getattr(pos, "symbol", None) == mt5_symbol]
    return len(symbol_positions), len(bot_positions)


def latest_closed_trade_candle_index(mt5, mt5_symbol: str, candles: pd.DataFrame) -> int | None:
    if candles.empty:
        return None
    start = pd.to_datetime(candles["time"]).min().to_pydatetime()
    end = datetime.now() + timedelta(days=1)
    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return None
    exit_entry = getattr(mt5, "DEAL_ENTRY_OUT", 1)
    close_times = []
    for deal in deals:
        if getattr(deal, "symbol", None) != mt5_symbol:
            continue
        if getattr(deal, "magic", None) != MAGIC_NUMBER:
            continue
        if getattr(deal, "entry", None) != exit_entry:
            continue
        close_times.append(pd.to_datetime(getattr(deal, "time"), unit="s"))
    if not close_times:
        return None
    times = pd.to_datetime(candles["time"]).reset_index(drop=True)
    last_close = max(close_times)
    index = int(times.searchsorted(last_close, side="right") - 1)
    return max(index, 0)


def open_bot_risk_amount(mt5) -> float:
    positions = mt5.positions_get()
    if positions is None:
        return 0.0
    total = 0.0
    for position in positions:
        if getattr(position, "magic", None) != MAGIC_NUMBER:
            continue
        sl = getattr(position, "sl", 0.0) or 0.0
        price_open = getattr(position, "price_open", 0.0) or 0.0
        if sl <= 0 or price_open <= 0:
            continue
        info = mt5.symbol_info(getattr(position, "symbol", ""))
        if info is None:
            continue
        tick_value = getattr(info, "trade_tick_value", 0.0) or 0.0
        tick_size = getattr(info, "trade_tick_size", 0.0) or 0.0
        if tick_value <= 0 or tick_size <= 0:
            continue
        distance = abs(price_open - sl)
        total += distance / tick_size * tick_value * getattr(position, "volume", 0.0)
    return total


def load_live_state() -> dict:
    if not LIVE_STATE_PATH.exists():
        return {"last_orders": {}, "daily": {}}
    try:
        with LIVE_STATE_PATH.open("r", encoding="utf-8") as file:
            state = json.load(file)
        state.setdefault("last_orders", {})
        state.setdefault("daily", {})
        return state
    except Exception:
        LOGGER.exception("Failed to load live state; fail-safe state initialized")
        return {"last_orders": {}, "daily": {}, "state_load_failed": True}


def save_live_state(state: dict) -> None:
    LIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LIVE_STATE_PATH.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


def order_dedupe_key(symbol: str, side: str, candle_time, model_version: str) -> str:
    return f"{symbol}|{side}|{pd.Timestamp(candle_time).isoformat()}|{model_version}|{MAGIC_NUMBER}"


def daily_state_for_symbol(state: dict, symbol: str) -> dict:
    today = datetime.now().date().isoformat()
    key = f"{today}:{symbol}"
    daily = state.setdefault("daily", {})
    daily.setdefault(key, {"trades": 0, "realized_pnl": 0.0, "consecutive_losses": 0})
    return daily[key]


def live_state_guard(state: dict, symbol: str) -> tuple[bool, str]:
    if state.get("state_load_failed"):
        return False, "STATE_LOAD_FAILED"
    daily = daily_state_for_symbol(state, symbol)
    if int(daily.get("trades", 0)) >= MAX_DAILY_TRADES:
        return False, "MAX_DAILY_TRADES"
    if float(daily.get("realized_pnl", 0.0)) <= -abs(float(MAX_DAILY_LOSS)):
        return False, "MAX_DAILY_LOSS"
    if int(daily.get("consecutive_losses", 0)) >= MAX_CONSECUTIVE_LOSSES:
        return False, "MAX_CONSECUTIVE_LOSSES"
    return True, "OK"


def record_order_state(state: dict, symbol: str, side: str, candle_time, model_version: str) -> None:
    key = order_dedupe_key(symbol, side, candle_time, model_version)
    state.setdefault("last_orders", {})[key] = datetime.now().isoformat(timespec="seconds")
    daily_state_for_symbol(state, symbol)["trades"] += 1
    save_live_state(state)


def new_trade_risk_amount(info, lot: float, sl_distance: float) -> float:
    tick_value = getattr(info, "trade_tick_value", 0.0) or 0.0
    tick_size = getattr(info, "trade_tick_size", 0.0) or 0.0
    if tick_value <= 0 or tick_size <= 0:
        return 0.0
    return sl_distance / tick_size * tick_value * lot


def load_artifacts(symbol: str):
    symbol_model_dir = MODELS_DIR / symbol
    live_meta = load_json(symbol_model_dir / "live_model_meta.json", default={}) or {}
    if live_meta.get("source") == "walk_forward":
        cycle = live_meta.get("cycle")
        if not cycle:
            raise RuntimeError(f"{symbol}: live_model_meta.json missing walk-forward cycle")
        artifact_subdir = live_meta.get("artifact_subdir")
        if not artifact_subdir:
            raise RuntimeError(f"{symbol}: canonical live_model_meta.json missing artifact_subdir")
        artifact_dir = symbol_model_dir / str(artifact_subdir) / str(cycle)
        failures = canonical_meta_failures(live_meta, artifact_dir)
        if failures:
            raise RuntimeError(f"{symbol}: canonical live metadata failed validation: {'; '.join(failures)}")
        model, columns, thresholds = load_artifact_dir(artifact_dir)
        if live_meta.get("allowed_sides"):
            thresholds["allowed_sides"] = live_meta["allowed_sides"]
        thresholds["_live_meta"] = live_meta
        return model, columns, thresholds
    if live_meta.get("source") == "deployment":
        deployment = live_meta.get("deployment")
        if not deployment:
            raise RuntimeError(f"{symbol}: live_model_meta.json missing deployment")
        artifact_subdir = live_meta.get("artifact_subdir") or f"deployments/{deployment}"
        artifact_dir = symbol_model_dir / str(artifact_subdir)
        failures = canonical_meta_failures(live_meta, artifact_dir)
        if failures:
            raise RuntimeError(f"{symbol}: canonical live metadata failed validation: {'; '.join(failures)}")
        model, columns, thresholds = load_artifact_dir(artifact_dir)
        if live_meta.get("allowed_sides"):
            thresholds["allowed_sides"] = live_meta["allowed_sides"]
        thresholds["_live_meta"] = live_meta
        return model, columns, thresholds

    raise RuntimeError(f"{symbol}: canonical live_model_meta.json missing; retrain the symbol before live evaluation")


def load_artifact_dir(artifact_dir):
    if not artifact_dir.exists():
        raise RuntimeError(f"Separate-side artifact directory not found: {artifact_dir}")
    columns = load_json(artifact_dir / "feature_columns.json")
    thresholds = load_json(artifact_dir / "best_threshold.json", default={}) or {}
    required = [
        "buy_model.joblib",
        "sell_model.joblib",
        "buy_threshold.json",
        "sell_threshold.json",
        "side_model_meta.json",
        "feature_columns.json",
        "best_threshold.json",
    ]
    missing = [name for name in required if not (artifact_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Separate-side artifacts incomplete in {artifact_dir}: missing {', '.join(missing)}; retrain the symbol")
    model = {
        "side_training_mode": "separate",
        "models": {
            "BUY": load_model(artifact_dir / "buy_model.joblib"),
            "SELL": load_model(artifact_dir / "sell_model.joblib"),
        },
    }
    thresholds["side_training_mode"] = "separate"
    thresholds.setdefault("buy_threshold_detail", load_json(artifact_dir / "buy_threshold.json", default={}) or {})
    thresholds.setdefault("sell_threshold_detail", load_json(artifact_dir / "sell_threshold.json", default={}) or {})
    return model, columns, thresholds


def predict_proba_model(model, rows: pd.DataFrame) -> np.ndarray:
    if not isinstance(model, dict) or model.get("side_training_mode") != "separate":
        raise RuntimeError("Live inference requires separate BUY/SELL side models")
    buy_model = model["models"]["BUY"]
    sell_model = model["models"]["SELL"]
    prob_buy = predict_binary_positive_proba(buy_model, rows)
    prob_sell = predict_binary_positive_proba(sell_model, rows)
    prob_no_trade = np.maximum(0.0, 1.0 - np.maximum(prob_buy, prob_sell))
    return np.column_stack([prob_no_trade, prob_buy, prob_sell])


def predict_binary_positive_proba(model, rows: pd.DataFrame) -> np.ndarray:
    if isinstance(model, list):
        if not model:
            raise RuntimeError("Live side model ensemble is empty")
        return np.mean([member.predict_proba(rows)[:, 1] for member in model], axis=0)
    return model.predict_proba(rows)[:, 1]


def decide_live_signal(prob_buy: float, prob_sell: float, thresholds: dict, cfg: dict) -> int:
    no_trade_zone = float(thresholds.get("no_trade_zone", 0.0))
    if abs(prob_buy - prob_sell) < no_trade_zone:
        return 0
    buy_threshold = float(thresholds.get("buy_threshold", cfg["buy_threshold"]))
    sell_threshold = float(thresholds.get("sell_threshold", cfg["sell_threshold"]))
    signal = decide_signal(prob_buy, prob_sell, buy_threshold, sell_threshold)
    allowed_sides = thresholds.get("allowed_sides")
    if allowed_sides and signal == 1 and "BUY" not in allowed_sides:
        return 0
    if allowed_sides and signal == 2 and "SELL" not in allowed_sides:
        return 0
    return signal


def order_type(mt5, signal: int):
    return mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL


def calculate_lot(mt5, cfg: dict, info, signal: int, entry: float, sl_distance: float) -> float:
    min_lot = getattr(info, "volume_min", cfg["default_lot"])
    max_lot = getattr(info, "volume_max", cfg["default_lot"])
    lot_step = getattr(info, "volume_step", 0.01)
    lot = cfg["default_lot"]
    if USE_RISK_PERCENT:
        account = mt5.account_info()
        tick_value = getattr(info, "trade_tick_value", 0.0)
        tick_size = getattr(info, "trade_tick_size", 0.0)
        if account is not None:
            lot = calculate_lot_by_risk(account.balance, cfg["risk_per_trade"], sl_distance, tick_value, tick_size)
    return normalize_lot(lot, min_lot, max_lot, lot_step)


def send_order(mt5, symbol: str, signal: int, lot: float, entry: float, sl: float, tp: float):
    cfg = SYMBOLS[symbol]
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": cfg["mt5_symbol"],
        "volume": lot,
        "type": order_type(mt5, signal),
        "price": entry,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": f"{symbol}_M5_ML_BOT",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)


def validate_trade_account(mt5, allow_real: bool = False) -> None:
    account = mt5.account_info()
    if account is None:
        raise RuntimeError("MT5 account_info unavailable; cannot validate trade account mode")
    trade_mode = getattr(account, "trade_mode", None)
    demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
    if trade_mode != demo_mode and not allow_real:
        raise RuntimeError(
            "Refusing to trade on non-demo account. Use demo account or pass --allow-real-account explicitly."
        )
    if not getattr(account, "trade_allowed", False):
        raise RuntimeError("MT5 account trade_allowed is False")
    if not getattr(account, "trade_expert", False):
        raise RuntimeError("MT5 account trade_expert is False; enable algo trading/expert advisors")


def evaluate_symbol(mt5, symbol: str, trade: bool) -> LiveSnapshot:
    cfg = SYMBOLS[symbol]
    mt5_symbol = cfg["mt5_symbol"]
    ensure_symbol_visible(mt5, symbol, mt5_symbol)
    info = mt5.symbol_info(mt5_symbol)
    tick = mt5.symbol_info_tick(mt5_symbol)
    model, columns, thresholds = load_artifacts(symbol)
    candles = fetch_recent_candles(mt5, symbol)
    features = add_features(candles, symbol)
    if features.empty:
        raise RuntimeError(f"{symbol}: not enough candles to generate features")
    latest = features.iloc[-1]
    feature_time = latest.get("time", candles["time"].iloc[-1] if "time" in candles.columns and not candles.empty else datetime.now())
    live_meta = thresholds.get("_live_meta", {})
    model_version = str(live_meta.get("deployment") or live_meta.get("cycle") or "unknown")
    missing = [col for col in columns if col not in features.columns]
    if missing:
        raise RuntimeError(f"{symbol}: live features missing training columns: {missing}")
    proba = predict_proba_model(model, features[columns].tail(1))[0]
    buy_threshold = float(thresholds.get("buy_threshold", cfg["buy_threshold"]))
    sell_threshold = float(thresholds.get("sell_threshold", cfg["sell_threshold"]))
    signal = decide_live_signal(float(proba[1]), float(proba[2]), thresholds, cfg)
    side = "BUY" if signal == 1 else "SELL" if signal == 2 else "NO_TRADE"
    open_symbol, open_total = open_positions(mt5, mt5_symbol)
    reason = "NO_TRADE"

    spread = float(latest.get("spread", 0.0))
    spread_to_atr = float(latest["spread_to_atr"])
    can_trade = signal > 0
    if can_trade and not bool(cfg.get("enabled_for_live", False)):
        can_trade, reason = False, "SYMBOL_LIVE_DISABLED"
    if can_trade and live_meta.get("gate_status") != "passed":
        can_trade, reason = False, "CANONICAL_GATE_NOT_PASSED"
    if can_trade and info is None:
        can_trade, reason = False, "NO_SYMBOL_INFO"
    if can_trade and any((getattr(info, name, 0.0) or 0.0) <= 0 for name in ["point", "trade_tick_size", "trade_tick_value"]):
        can_trade, reason = False, "BROKER_SPECS_MISSING"
    if can_trade and not check_spread_filter(spread_to_atr, cfg["max_spread_to_atr"]):
        can_trade, reason = False, "SPREAD_FILTER"
    if can_trade and not session_allowed(latest, cfg):
        can_trade, reason = False, "SESSION_FILTER"
    regime_col = cfg.get("regime_filter_col")
    if can_trade and regime_col and int(latest.get(regime_col, 0)) == 0:
        can_trade, reason = False, "REGIME_FILTER"
    if can_trade and REQUIRE_ELIGIBLE_THRESHOLD_FOR_LIVE and thresholds.get("eligible") is False:
        can_trade, reason = False, "THRESHOLD_NOT_ELIGIBLE"
    if can_trade and not check_open_positions(open_symbol, open_total, MAX_OPEN_POSITIONS_PER_SYMBOL, MAX_TOTAL_OPEN_POSITIONS):
        can_trade, reason = False, "POSITION_LIMIT"
    current_index = len(candles)
    if can_trade and symbol in LAST_CLOSE_INDEX and current_index - LAST_CLOSE_INDEX[symbol] < COOLDOWN_CANDLES:
        can_trade, reason = False, "COOLDOWN"
    if can_trade:
        last_closed_index = latest_closed_trade_candle_index(mt5, mt5_symbol, candles)
        if last_closed_index is not None and current_index - last_closed_index < COOLDOWN_CANDLES:
            can_trade, reason = False, "COOLDOWN_CLOSED_TRADE"
    if can_trade and tick is None:
        can_trade, reason = False, "NO_TICK"
    state = load_live_state()
    if can_trade:
        ok, state_reason = live_state_guard(state, symbol)
        if not ok:
            can_trade, reason = False, state_reason
    if can_trade:
        dedupe_key = order_dedupe_key(symbol, side, feature_time, model_version)
        if dedupe_key in state.get("last_orders", {}):
            can_trade, reason = False, "DUPLICATE_ORDER_GUARD"
    if can_trade:
        reason = "DRY_RUN" if not trade else "READY"

    price = float(tick.ask if signal == 1 else tick.bid) if tick is not None and signal else float(latest["close"])
    levels = calculate_atr_sl_tp(price, float(latest["atr_14"]), side if side != "NO_TRADE" else "BUY", cfg["live_sl_atr_mult"], cfg["live_tp_atr_mult"])
    lot = calculate_lot(mt5, cfg, info, signal, price, levels.sl_distance) if signal else 0.0
    if can_trade and USE_RISK_PERCENT:
        account = mt5.account_info()
        total_risk = open_bot_risk_amount(mt5) + new_trade_risk_amount(info, lot, levels.sl_distance)
        if account is None or total_risk > account.balance * MAX_TOTAL_RISK:
            can_trade, reason = False, "TOTAL_RISK_LIMIT"

    daily = daily_state_for_symbol(state, symbol)
    account = account_snapshot(mt5)
    account.daily_pnl = float(daily.get("realized_pnl", 0.0))
    spread_points = latest.get("spread_points", None)
    if spread_points is None:
        point = getattr(info, "point", 0.0) if info is not None else 0.0
        spread_points = spread / point if point else None
    confidence = float(proba[signal]) if signal > 0 else float(max(proba[1], proba[2]))
    snapshot = LiveSnapshot(
        symbol=symbol,
        mt5_symbol=mt5_symbol,
        mode="LIVE" if trade else "DRY-RUN",
        strategy="ML",
        htf="ON" if any(name.startswith(("m15_", "h1_")) for name in latest.index) else "OFF",
        connected=True,
        heartbeat="OK",
        trade_allowed="YES" if can_trade else "NO",
        updated=datetime.now().strftime("%H:%M:%S"),
        open_positions_symbol=open_symbol,
        open_positions_total=open_total,
        account=account,
        market=MarketSnapshot(
            bid=float(getattr(tick, "bid", 0.0)) if tick is not None else None,
            ask=float(getattr(tick, "ask", 0.0)) if tick is not None else None,
            spread=spread,
            spread_points=float(spread_points) if spread_points is not None else None,
            atr=float(latest["atr_14"]),
            last_candle_time=str(feature_time),
        ),
        signal=SignalSnapshot(
            side=side,
            confidence=confidence,
            prob_buy=float(proba[1]),
            prob_sell=float(proba[2]),
            reason=reason,
            feature_time=str(feature_time),
            timeframe=cfg["timeframe"],
            lot=lot,
            sl=levels.sl,
            tp=levels.tp,
            model_version=model_version,
            gate_status=live_meta.get("gate_status", "missing"),
        ),
        positions=positions_snapshot(mt5, mt5_symbol, MAGIC_NUMBER),
        trend=trend_snapshot(latest),
    )

    append_csv_row(
        LOGS_DIR / "live_signals.csv",
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "symbol": symbol,
            "mt5_symbol": mt5_symbol,
            "prob_no_trade": float(proba[0]),
            "prob_buy": float(proba[1]),
            "prob_sell": float(proba[2]),
            "signal": side,
            "threshold_buy": buy_threshold,
            "threshold_sell": sell_threshold,
            "atr": float(latest["atr_14"]),
            "spread": spread,
            "spread_to_atr": spread_to_atr,
            "open_positions_symbol": open_symbol,
            "open_positions_total": open_total,
            "reason": reason,
            "model_version": model_version,
            "gate_status": live_meta.get("gate_status", "missing"),
            "allowed_sides": "|".join(thresholds.get("allowed_sides", [])),
            "feature_time": str(feature_time),
        },
    )

    if not can_trade or not trade:
        LOGGER.info("%s signal=%s reason=%s", symbol, side, reason)
        return snapshot
    if not check_stop_level(info, price, levels.sl, levels.tp):
        LOGGER.info("%s skipped: STOP_LEVEL", symbol)
        snapshot.signal.reason = "STOP_LEVEL"
        snapshot.trade_allowed = "NO"
        return snapshot
    margin_ok, _ = check_margin(mt5, order_type(mt5, signal), mt5_symbol, lot, price)
    if not margin_ok:
        LOGGER.info("%s skipped: MARGIN", symbol)
        snapshot.signal.reason = "MARGIN"
        snapshot.trade_allowed = "NO"
        return snapshot
    result = send_order(mt5, symbol, signal, lot, price, levels.sl, levels.tp)
    LAST_CLOSE_INDEX[symbol] = current_index
    record_order_state(state, symbol, side, feature_time, model_version)
    append_csv_row(
        LOGS_DIR / "live_orders.csv",
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "symbol": symbol,
            "mt5_symbol": mt5_symbol,
            "order_type": side,
            "lot": lot,
            "price": price,
            "sl": levels.sl,
            "tp": levels.tp,
            "retcode": getattr(result, "retcode", None),
            "comment": getattr(result, "comment", None),
            "ticket": getattr(result, "order", None),
        },
    )
    return snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MT5 multi-symbol live bot. Default is dry-run.")
    parse_symbol_args(parser)
    parser.add_argument("--trade", action="store_true", help="Send real market orders. Without this flag, only signals are logged.")
    parser.add_argument("--allow-real-account", action="store_true", help="Allow trading on a non-demo account. Strongly discouraged.")
    parser.add_argument("--once", action="store_true", help="Run one evaluation pass and exit.")
    parser.add_argument("--sleep-seconds", type=int, default=60, help="Delay between live evaluation loops.")
    parser.add_argument("--dashboard", action="store_true", help="Render a Rich live dashboard while evaluating symbols.")
    parser.add_argument("--dashboard-symbol", help="Initial symbol focus for --dashboard. Must be in the selected symbol set.")
    parser.add_argument("--dashboard-log-limit", type=int, default=100, help="Maximum in-memory dashboard events per panel.")
    return parser


def run_dashboard_loop(mt5, selected_symbols: list[str], trade: bool, once: bool, sleep_seconds: int, focus_symbol: str, log_limit: int) -> None:
    from rich.live import Live
    from live_dashboard import RichDashboard

    dashboard = RichDashboard(focus_symbol=focus_symbol, symbols=selected_symbols, log_limit=max(log_limit, 1))
    with Live(dashboard.render(), refresh_per_second=4, screen=True) as live:
        while True:
            for symbol in selected_symbols:
                try:
                    dashboard.set_snapshot(evaluate_symbol(mt5, symbol, trade))
                except Exception as exc:
                    LOGGER.exception("%s live evaluation failed: %s", symbol, exc)
                    dashboard.add_error(symbol, str(exc))
                live.update(dashboard.render())
            if once:
                break
            time.sleep(max(sleep_seconds, 1))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    mt5 = connect_mt5()
    try:
        if args.trade:
            validate_trade_account(mt5, allow_real=args.allow_real_account)
        selected_symbols = resolve_symbols(args)
        if args.dashboard_symbol:
            focus_symbol = validate_symbol(args.dashboard_symbol)
            if focus_symbol not in selected_symbols:
                parser.error("--dashboard-symbol must be included in --symbol/--symbols/--all selection")
        else:
            focus_symbol = selected_symbols[0]
        if args.dashboard:
            run_dashboard_loop(
                mt5,
                selected_symbols,
                args.trade,
                args.once,
                args.sleep_seconds,
                focus_symbol,
                args.dashboard_log_limit,
            )
            return
        while True:
            for symbol in selected_symbols:
                try:
                    evaluate_symbol(mt5, symbol, args.trade)
                except Exception as exc:
                    LOGGER.exception("%s live evaluation failed: %s", symbol, exc)
            if args.once:
                break
            time.sleep(max(args.sleep_seconds, 1))
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
