from dataclasses import dataclass


@dataclass(frozen=True)
class TradeLevels:
    sl: float
    tp: float
    sl_distance: float
    tp_distance: float


def normalize_lot(lot: float, min_lot: float, max_lot: float, lot_step: float) -> float:
    if lot_step <= 0:
        return max(min_lot, min(max_lot, lot))
    steps = int((lot - min_lot) / lot_step)
    normalized = min_lot + max(0, steps) * lot_step
    return round(max(min_lot, min(max_lot, normalized)), 8)


def calculate_lot_by_risk(account_balance: float, risk_percent: float, sl_distance: float, tick_value: float, tick_size: float) -> float:
    if sl_distance <= 0 or tick_value <= 0 or tick_size <= 0:
        return 0.0
    risk_amount = account_balance * risk_percent
    value_per_lot = sl_distance / tick_size * tick_value
    return risk_amount / value_per_lot if value_per_lot else 0.0


def check_margin(mt5, order_type: int, symbol: str, lot: float, price: float) -> tuple[bool, float | None]:
    margin = mt5.order_calc_margin(order_type, symbol, lot, price)
    if margin is None:
        return False, None
    account = mt5.account_info()
    if account is None:
        return False, margin
    return account.margin_free >= margin, margin


def check_stop_level(symbol_info, price: float, sl: float, tp: float) -> bool:
    point = getattr(symbol_info, "point", 0) or 0
    stop_level = getattr(symbol_info, "trade_stops_level", 0) or 0
    minimum_distance = stop_level * point
    if minimum_distance <= 0:
        return True
    return abs(price - sl) >= minimum_distance and abs(price - tp) >= minimum_distance


def check_spread_filter(spread_to_atr: float, max_spread_to_atr: float) -> bool:
    return spread_to_atr <= max_spread_to_atr


def check_open_positions(open_positions_symbol: int, open_positions_total: int, max_symbol: int, max_total: int) -> bool:
    return open_positions_symbol < max_symbol and open_positions_total < max_total


def check_cooldown(last_close_index: int | None, current_index: int, cooldown_candles: int) -> bool:
    if last_close_index is None:
        return True
    return current_index - last_close_index >= cooldown_candles


def calculate_atr_sl_tp(entry: float, atr: float, side: str, sl_atr_mult: float, tp_atr_mult: float) -> TradeLevels:
    sl_distance = atr * sl_atr_mult
    tp_distance = atr * tp_atr_mult
    if side.upper() == "BUY":
        return TradeLevels(sl=entry - sl_distance, tp=entry + tp_distance, sl_distance=sl_distance, tp_distance=tp_distance)
    if side.upper() == "SELL":
        return TradeLevels(sl=entry + sl_distance, tp=entry - tp_distance, sl_distance=sl_distance, tp_distance=tp_distance)
    raise ValueError("side must be BUY or SELL")
