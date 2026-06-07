import numpy as np
import pandas as pd
from itertools import product
from config import MIN_THRESHOLD_PROFIT_FACTOR, MIN_THRESHOLD_SIGNALS
from strategy_filters import broker_specs, pnl_fields, session_allowed, trade_metrics
from utils import setup_logger

LOGGER = setup_logger("walk_forward_threshold")

ADAPTIVE_THRESHOLD_QUANTILES = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
ADAPTIVE_THRESHOLD_FLOOR = 0.25
FIXED_GRID_MIN_THRESHOLD = 0.55
MIN_STABLE_THRESHOLD_TRADES = 10


def cfg_for_session_mode(cfg: dict, session_mode: str) -> dict:
    if session_mode != "all_sessions":
        return cfg
    relaxed = cfg.copy()
    relaxed["allowed_entry_sessions"] = []
    return relaxed


def target_trade_metrics(metrics: dict, cfg: dict, days: int) -> dict:
    target_min = float(cfg.get("target_min_trades_per_day", 0.0))
    target_max = float(cfg.get("target_max_trades_per_day", float("inf")))
    min_pf = float(cfg.get("min_target_profit_factor", MIN_THRESHOLD_PROFIT_FACTOR))
    require_positive_net = bool(cfg.get("require_positive_net_for_target", True))
    trades_per_day = float(metrics.get("backtest_trades", 0) / max(days, 1))
    below_target = target_min > 0.0 and trades_per_day < target_min
    above_target = target_max != float("inf") and trades_per_day > target_max
    meets_frequency = not below_target and not above_target
    meets_quality = (
        float(metrics.get("backtest_profit_factor", 0.0)) >= min_pf
        and (not require_positive_net or float(metrics.get("backtest_net_profit", 0.0)) > 0.0)
    )
    if below_target:
        gap = target_min - trades_per_day
    elif above_target:
        gap = trades_per_day - target_max
    else:
        gap = 0.0
    return {
        "backtest_trades_per_day": trades_per_day,
        "target_min_trades_per_day": target_min,
        "target_max_trades_per_day": target_max,
        "meets_trade_frequency_target": bool(meets_frequency),
        "below_trade_target": bool(below_target),
        "above_trade_target": bool(above_target),
        "meets_quality_gate": bool(meets_quality),
        "target_feasible": bool(meets_frequency and meets_quality),
        "trade_frequency_gap": float(gap),
    }

def decide_signal_wf(prob_buy: float, prob_sell: float, buy_threshold: float, sell_threshold: float, no_trade_zone: float) -> int:
    if abs(prob_buy - prob_sell) < no_trade_zone:
        return 0
    if prob_buy >= buy_threshold and prob_buy > prob_sell:
        return 1
    if prob_sell >= sell_threshold and prob_sell > prob_buy:
        return 2
    return 0

def simulate_wf_trades(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    cfg: dict,
    buy_threshold: float,
    sell_threshold: float,
    no_trade_zone: float,
    cooldown_candles: int = 3,
    slippage: float = 0.0,
) -> pd.DataFrame:
    trades = []
    cooldown_until = -1
    idx = 0
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    # Convert columns to lists or arrays for faster indexing
    times = df["time"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr_14"].values
    spreads = df.get("spread", pd.Series(0.0, index=df.index)).values
    spread_to_atrs = df["spread_to_atr"].values

    # Pre-calculate session permissions
    session_allowed_arr = np.array([session_allowed(row, cfg) for _, row in df.iterrows()])
    max_spread_to_atr = float(cfg["max_spread_to_atr"])
    spread_allowed_arr = spread_to_atrs <= max_spread_to_atr

    # Regime Filter Support
    regime_col = cfg.get("regime_filter_col")
    if regime_col and regime_col in df.columns:
        regime_allowed_arr = df[regime_col].values != 0
    else:
        regime_allowed_arr = np.ones(n, dtype=bool)

    prob_buys = probabilities[:, 1]
    prob_sells = probabilities[:, 2]
    diffs = np.abs(prob_buys - prob_sells)

    while idx < n - 1:
        if idx <= cooldown_until:
            idx += 1
            continue
        
        if not session_allowed_arr[idx] or not spread_allowed_arr[idx] or not regime_allowed_arr[idx]:
            idx += 1
            continue

        p_buy = prob_buys[idx]
        p_sell = prob_sells[idx]
        diff = diffs[idx]

        if diff < no_trade_zone:
            idx += 1
            continue

        if p_buy >= buy_threshold and p_buy > p_sell:
            signal = 1
        elif p_sell >= sell_threshold and p_sell > p_buy:
            signal = 2
        else:
            signal = 0

        if signal == 0:
            idx += 1
            continue

        side = "BUY" if signal == 1 else "SELL"
        spread = float(spreads[idx])
        entry = float(closes[idx]) + spread + slippage if side == "BUY" else float(closes[idx]) - slippage
        
        atr = float(atrs[idx])
        sl_mult = float(cfg["live_sl_atr_mult"])
        tp_mult = float(cfg["live_tp_atr_mult"])

        if side == "BUY":
            sl = entry - sl_mult * atr
            tp = entry + tp_mult * atr
        else:
            sl = entry + sl_mult * atr
            tp = entry - tp_mult * atr

        lookahead = int(cfg["lookahead_candles"])
        end_idx = min(idx + lookahead, n - 1)
        exit_idx = end_idx
        exit_price = closes[end_idx]
        exit_reason = "TIME"

        for check_idx in range(idx + 1, end_idx + 1):
            h = highs[check_idx]
            l = lows[check_idx]
            if side == "BUY":
                if l <= sl:
                    exit_idx = check_idx
                    exit_price = sl
                    exit_reason = "SL"
                    break
                if h >= tp:
                    exit_idx = check_idx
                    exit_price = tp
                    exit_reason = "TP"
                    break
            else:
                if h >= sl:
                    exit_idx = check_idx
                    exit_price = sl
                    exit_reason = "SL"
                    break
                if l <= tp:
                    exit_idx = check_idx
                    exit_price = tp
                    exit_reason = "TP"
                    break

        row = df.iloc[idx]
        lot = float(cfg["default_lot"])
        tick_size, tick_value, _ = broker_specs(row, cfg)
        risk_distance = abs(entry - sl)

        trades.append({
            "entry_index": idx,
            "exit_index": exit_idx,
            "entry_time": times[idx],
            "exit_time": times[exit_idx],
            "side": side,
            "entry": entry,
            "exit": exit_price,
            "sl": sl,
            "tp": tp,
            "exit_reason": exit_reason,
            **pnl_fields(side, entry, exit_price, lot, tick_size, tick_value, risk_distance),
            "holding_candles": exit_idx - idx,
            "prob_buy": float(p_buy),
            "prob_sell": float(p_sell),
            "spread_to_atr": float(spread_to_atrs[idx]),
        })
        cooldown_until = exit_idx + cooldown_candles
        idx = exit_idx + 1

    return pd.DataFrame(trades)


def simulate_side_wf_trades(
    df: pd.DataFrame,
    side_probabilities: np.ndarray,
    cfg: dict,
    side: str,
    threshold: float,
    cooldown_candles: int = 3,
    slippage: float = 0.0,
) -> pd.DataFrame:
    trades = []
    cooldown_until = -1
    idx = 0
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    times = df["time"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr_14"].values
    spreads = df.get("spread", pd.Series(0.0, index=df.index)).values
    spread_to_atrs = df["spread_to_atr"].values

    session_allowed_arr = np.array([session_allowed(row, cfg) for _, row in df.iterrows()])
    spread_allowed_arr = spread_to_atrs <= float(cfg["max_spread_to_atr"])
    regime_col = cfg.get("regime_filter_col")
    if regime_col and regime_col in df.columns:
        regime_allowed_arr = df[regime_col].values != 0
    else:
        regime_allowed_arr = np.ones(n, dtype=bool)

    side = side.upper()
    while idx < n - 1:
        if idx <= cooldown_until:
            idx += 1
            continue
        if not session_allowed_arr[idx] or not spread_allowed_arr[idx] or not regime_allowed_arr[idx]:
            idx += 1
            continue
        confidence = float(side_probabilities[idx])
        if confidence < threshold:
            idx += 1
            continue

        spread = float(spreads[idx])
        entry = float(closes[idx]) + spread + slippage if side == "BUY" else float(closes[idx]) - slippage
        atr = float(atrs[idx])
        sl_mult = float(cfg["live_sl_atr_mult"])
        tp_mult = float(cfg["live_tp_atr_mult"])
        if side == "BUY":
            sl = entry - sl_mult * atr
            tp = entry + tp_mult * atr
        else:
            sl = entry + sl_mult * atr
            tp = entry - tp_mult * atr

        lookahead = int(cfg["lookahead_candles"])
        end_idx = min(idx + lookahead, n - 1)
        exit_idx = end_idx
        exit_price = closes[end_idx]
        exit_reason = "TIME"
        for check_idx in range(idx + 1, end_idx + 1):
            h = highs[check_idx]
            l = lows[check_idx]
            if side == "BUY":
                if l <= sl:
                    exit_idx = check_idx
                    exit_price = sl
                    exit_reason = "SL"
                    break
                if h >= tp:
                    exit_idx = check_idx
                    exit_price = tp
                    exit_reason = "TP"
                    break
            else:
                if h >= sl:
                    exit_idx = check_idx
                    exit_price = sl
                    exit_reason = "SL"
                    break
                if l <= tp:
                    exit_idx = check_idx
                    exit_price = tp
                    exit_reason = "TP"
                    break

        row = df.iloc[idx]
        lot = float(cfg["default_lot"])
        tick_size, tick_value, _ = broker_specs(row, cfg)
        risk_distance = abs(entry - sl)
        trades.append({
            "entry_index": idx,
            "exit_index": exit_idx,
            "entry_time": times[idx],
            "exit_time": times[exit_idx],
            "side": side,
            "entry": entry,
            "exit": exit_price,
            "sl": sl,
            "tp": tp,
            "exit_reason": exit_reason,
            **pnl_fields(side, entry, exit_price, lot, tick_size, tick_value, risk_distance),
            "holding_candles": exit_idx - idx,
            "prob_buy": confidence if side == "BUY" else 0.0,
            "prob_sell": confidence if side == "SELL" else 0.0,
            "spread_to_atr": float(spread_to_atrs[idx]),
        })
        cooldown_until = exit_idx + cooldown_candles
        idx = exit_idx + 1

    return pd.DataFrame(trades)

def expected_value_wf(row: pd.Series, cfg: dict) -> float:
    if row["combined_precision"] == 0 or row["total_signals"] == 0:
        return 0.0
    avg_reward = float(cfg["live_tp_atr_mult"])
    avg_risk = float(cfg["live_sl_atr_mult"])
    return row["combined_precision"] * avg_reward - (1 - row["combined_precision"]) * avg_risk

def adaptive_threshold_grid(probabilities: np.ndarray, side_mask: np.ndarray) -> np.ndarray:
    values = probabilities[side_mask]
    if len(values) == 0:
        return np.array([FIXED_GRID_MIN_THRESHOLD])

    quantiles = np.quantile(values, ADAPTIVE_THRESHOLD_QUANTILES)
    thresholds = np.maximum(quantiles, ADAPTIVE_THRESHOLD_FLOOR)
    thresholds = np.append(thresholds, FIXED_GRID_MIN_THRESHOLD)
    thresholds = thresholds[(thresholds > 0.0) & (thresholds <= 0.80)]
    return np.array(sorted({round(float(value), 2) for value in thresholds}))

def threshold_diagnostics(prob_buys: np.ndarray, prob_sells: np.ndarray, valid_mask: np.ndarray) -> dict:
    fixed_buy = (prob_buys >= FIXED_GRID_MIN_THRESHOLD) & (prob_buys > prob_sells) & valid_mask
    fixed_sell = (prob_sells >= FIXED_GRID_MIN_THRESHOLD) & (prob_sells > prob_buys) & valid_mask
    return {
        "max_prob_buy": float(prob_buys[valid_mask].max()) if valid_mask.any() else 0.0,
        "max_prob_sell": float(prob_sells[valid_mask].max()) if valid_mask.any() else 0.0,
        "valid_rows": int(valid_mask.sum()),
        "signals_at_055": int(fixed_buy.sum() + fixed_sell.sum()),
    }

def fallback_reason(diagnostics: dict, min_signals: int) -> str:
    if diagnostics["valid_rows"] < min_signals:
        return "too_few_valid_rows"
    if diagnostics["signals_at_055"] == 0:
        return "probabilities_below_fixed_grid"
    return "failed_profitability_gate"

def evaluate_threshold_candidate(
    val_df: pd.DataFrame,
    proba: np.ndarray,
    cfg: dict,
    y_true: np.ndarray,
    valid_mask: np.ndarray,
    prob_buys: np.ndarray,
    prob_sells: np.ndarray,
    diffs: np.ndarray,
    buy_thresh: float,
    sell_thresh: float,
    nt_zone: float,
    days: int,
    diagnostics: dict,
    session_mode: str = "configured_sessions",
    reason: str = "",
) -> dict:
    buy_sig = (prob_buys >= buy_thresh) & (prob_buys > prob_sells) & (diffs >= nt_zone) & valid_mask
    sell_sig = (prob_sells >= sell_thresh) & (prob_sells > prob_buys) & (diffs >= nt_zone) & valid_mask

    total = int(buy_sig.sum() + sell_sig.sum())
    signals = np.zeros(len(val_df), dtype=int)
    signals[buy_sig] = 1
    signals[sell_sig] = 2
    mask = signals > 0

    buy_precision = float((y_true[buy_sig] == 1).mean()) if buy_sig.any() else 0.0
    sell_precision = float((y_true[sell_sig] == 2).mean()) if sell_sig.any() else 0.0
    combined_precision = float((y_true[mask] == signals[mask]).mean()) if mask.any() else 0.0
    candidate_cfg = cfg_for_session_mode(cfg, session_mode)
    trades = simulate_wf_trades(val_df, proba, candidate_cfg, buy_thresh, sell_thresh, nt_zone)
    metrics = trade_metrics(trades)
    target_metrics = target_trade_metrics(metrics, cfg, days)

    return {
        "session_mode": session_mode,
        "buy_threshold": round(float(buy_thresh), 2),
        "sell_threshold": round(float(sell_thresh), 2),
        "no_trade_zone": round(float(nt_zone), 2),
        "total_signals": total,
        "buy_precision": buy_precision,
        "sell_precision": sell_precision,
        "combined_precision": combined_precision,
        "trade_frequency": total / len(proba) if len(proba) else 0.0,
        "average_confidence": float(np.max(proba[mask][:, 1:3], axis=1).mean()) if mask.any() else 0.0,
        "estimated_trades_per_day": total / days,
        "backtest_evaluated": True,
        "fallback_reason": reason,
        **diagnostics,
        **metrics,
        **target_metrics,
    }


def threshold_quality_columns(result_df: pd.DataFrame, min_signals: int) -> pd.DataFrame:
    """Add conservative ranking columns for threshold selection."""
    ranked = result_df.copy()
    if "expected_value" not in ranked.columns:
        ranked["expected_value"] = 0.0
    ranked["pf_dd_ratio"] = ranked.apply(
        lambda row: float(row["backtest_profit_factor"] / (abs(row["backtest_max_drawdown"]) + 0.05))
        if row["backtest_profit_factor"] > 0 else 0.0,
        axis=1,
    )
    stable_trade_floor = max(MIN_STABLE_THRESHOLD_TRADES, int(min_signals))
    ranked["stable_trade_count"] = ranked["backtest_trades"] >= stable_trade_floor
    ranked["low_trade_penalty"] = ranked["backtest_trades"].apply(
        lambda trades: max(0.0, (stable_trade_floor - float(trades)) / stable_trade_floor)
    )
    ranked["drawdown_penalty"] = ranked["backtest_max_drawdown"].abs()
    ranked["threshold_quality_score"] = (
        ranked["pf_dd_ratio"].clip(upper=10.0)
        + ranked["backtest_expected_value"].clip(lower=-10.0, upper=10.0)
        + ranked["backtest_net_profit"].clip(lower=-1000.0, upper=1000.0) / 1000.0
        - ranked["low_trade_penalty"]
        - ranked["drawdown_penalty"].clip(upper=1000.0) / 1000.0
    )
    return ranked

def optimize_thresholds_wf(val_df: pd.DataFrame, proba: np.ndarray, cfg: dict) -> tuple[dict, pd.DataFrame]:
    y_true = val_df["label"].to_numpy()
    min_signals = int(cfg.get("min_threshold_signals", MIN_THRESHOLD_SIGNALS))
    days = max((val_df["time"].max() - val_df["time"].min()).days, 1)

    # Pre-calculate masks
    configured_session_mask = np.array([session_allowed(row, cfg) for _, row in val_df.iterrows()])
    max_spread_to_atr = float(cfg["max_spread_to_atr"])
    spread_mask = val_df["spread_to_atr"].to_numpy() <= max_spread_to_atr
    
    # Regime Filter Support
    regime_col = cfg.get("regime_filter_col")
    if regime_col and regime_col in val_df.columns:
        regime_mask = val_df[regime_col].to_numpy() != 0
    else:
        regime_mask = np.ones(len(val_df), dtype=bool)

    prob_buys = proba[:, 1]
    prob_sells = proba[:, 2]
    diffs = np.abs(prob_buys - prob_sells)

    rows = []
    no_trade_zones = np.arange(0.00, 0.11, 0.02)
    session_modes = ["configured_sessions"]
    if bool(cfg.get("allow_session_relax_for_target", False)):
        session_modes.append("all_sessions")

    for session_mode in session_modes:
        session_mask = np.ones(len(val_df), dtype=bool) if session_mode == "all_sessions" else configured_session_mask
        valid_mask = session_mask & spread_mask & regime_mask
        diagnostics = threshold_diagnostics(prob_buys, prob_sells, valid_mask)
        reason = fallback_reason(diagnostics, min_signals)
        buy_thresholds = adaptive_threshold_grid(prob_buys, (prob_buys > prob_sells) & valid_mask)
        sell_thresholds = adaptive_threshold_grid(prob_sells, (prob_sells > prob_buys) & valid_mask)

        LOGGER.info(
            "Adaptive threshold grid (%s): buy=%s sell=%s valid_rows=%s max_buy=%.4f max_sell=%.4f signals_at_055=%s",
            session_mode,
            buy_thresholds.tolist(),
            sell_thresholds.tolist(),
            diagnostics["valid_rows"],
            diagnostics["max_prob_buy"],
            diagnostics["max_prob_sell"],
            diagnostics["signals_at_055"],
        )

        for buy_thresh, sell_thresh, nt_zone in product(buy_thresholds, sell_thresholds, no_trade_zones):
            # Quick signal pre-filtering
            buy_sig = (prob_buys >= buy_thresh) & (prob_buys > prob_sells) & (diffs >= nt_zone) & valid_mask
            sell_sig = (prob_sells >= sell_thresh) & (prob_sells > prob_buys) & (diffs >= nt_zone) & valid_mask
            
            total = int(buy_sig.sum() + sell_sig.sum())
            if total < min_signals:
                continue

            # Determine precision of signals
            signals = np.zeros(len(val_df), dtype=int)
            signals[buy_sig] = 1
            signals[sell_sig] = 2
            mask = signals > 0

            buy_precision = float((y_true[buy_sig] == 1).mean()) if buy_sig.any() else 0.0
            sell_precision = float((y_true[sell_sig] == 2).mean()) if sell_sig.any() else 0.0
            combined_precision = float((y_true[mask] == signals[mask]).mean()) if mask.any() else 0.0

            # Pre-filter by combined precision to optimize search speed
            if combined_precision < 0.35:
                continue

            rows.append(
                evaluate_threshold_candidate(
                    val_df,
                    proba,
                    cfg,
                    y_true,
                    valid_mask,
                    prob_buys,
                    prob_sells,
                    diffs,
                    buy_thresh,
                    sell_thresh,
                    nt_zone,
                    days,
                    diagnostics,
                    session_mode,
                )
            )

    # If no combination had enough signals, run a fallback with less strict constraint or return a default
    if not rows:
        LOGGER.warning(
            "No threshold combination met min_signals=%s. Retrying with reduced signal requirement. reason=%s",
            min_signals,
            reason,
        )
        fallback_min = max(5, min_signals // 4)
        for session_mode in session_modes:
            session_mask = np.ones(len(val_df), dtype=bool) if session_mode == "all_sessions" else configured_session_mask
            valid_mask = session_mask & spread_mask & regime_mask
            diagnostics = threshold_diagnostics(prob_buys, prob_sells, valid_mask)
            reason = fallback_reason(diagnostics, min_signals)
            buy_thresholds = adaptive_threshold_grid(prob_buys, (prob_buys > prob_sells) & valid_mask)
            sell_thresholds = adaptive_threshold_grid(prob_sells, (prob_sells > prob_buys) & valid_mask)
            for buy_thresh, sell_thresh, nt_zone in product(buy_thresholds, sell_thresholds, no_trade_zones):
                buy_sig = (prob_buys >= buy_thresh) & (prob_buys > prob_sells) & (diffs >= nt_zone) & valid_mask
                sell_sig = (prob_sells >= sell_thresh) & (prob_sells > prob_buys) & (diffs >= nt_zone) & valid_mask
                total = int(buy_sig.sum() + sell_sig.sum())
                if total >= fallback_min:
                    rows.append(
                        evaluate_threshold_candidate(
                            val_df,
                            proba,
                            cfg,
                            y_true,
                            valid_mask,
                            prob_buys,
                            prob_sells,
                            diffs,
                            buy_thresh,
                            sell_thresh,
                            nt_zone,
                            days,
                            diagnostics,
                            session_mode,
                            reason,
                        )
                    )

    if not rows:
        # Absolute fallback if still nothing
        valid_mask = configured_session_mask & spread_mask & regime_mask
        diagnostics = threshold_diagnostics(prob_buys, prob_sells, valid_mask)
        reason = fallback_reason(diagnostics, min_signals)
        metrics = {
            "backtest_trades": 0,
            "backtest_profit_factor": 0.0,
            "backtest_net_profit": 0.0,
            "backtest_net_profit_price_lot": 0.0,
            "backtest_net_profit_money": 0.0,
            "backtest_avg_pnl_per_trade": 0.0,
            "backtest_pnl_money_available": False,
        }
        LOGGER.warning("Absolute fallback threshold configuration used. reason=%s", reason)
        rows.append({
            "session_mode": "configured_sessions",
            "buy_threshold": 0.65,
            "sell_threshold": 0.65,
            "no_trade_zone": 0.02,
            "total_signals": 0,
            "buy_precision": 0.0,
            "sell_precision": 0.0,
            "combined_precision": 0.0,
            "trade_frequency": 0.0,
            "average_confidence": 0.0,
            "estimated_trades_per_day": 0.0,
            "backtest_evaluated": False,
            "fallback_reason": reason,
            **diagnostics,
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
            **target_trade_metrics(metrics, cfg, days),
        })

    result_df = pd.DataFrame(rows)
    result_df["expected_value"] = result_df.apply(lambda row: expected_value_wf(row, cfg), axis=1)
    result_df = threshold_quality_columns(result_df, min_signals)

    # Eligibility criteria
    result_df["eligible"] = (
        (result_df["backtest_trades"] >= min_signals)
        & (result_df["backtest_profit_factor"] >= MIN_THRESHOLD_PROFIT_FACTOR)
        & (result_df["backtest_net_profit"] > 0)
    )

    # Prefer quality first; trade frequency is only a tie-breaker after profitable edge.
    result_df = result_df.sort_values(
        [
            "meets_quality_gate",
            "eligible",
            "backtest_evaluated",
            "stable_trade_count",
            "threshold_quality_score",
            "pf_dd_ratio",
            "backtest_expected_value",
            "backtest_max_drawdown",
            "backtest_profit_factor",
            "backtest_net_profit",
            "target_feasible",
            "trade_frequency_gap",
            "combined_precision",
        ],
        ascending=[False, False, False, False, False, False, False, True, False, False, False, True, False],
    ).reset_index(drop=True)

    best_thresholds = result_df.iloc[0].to_dict()
    return best_thresholds, result_df


def optimize_side_threshold_wf(val_df: pd.DataFrame, positive_proba: np.ndarray, cfg: dict, side: str) -> tuple[dict, pd.DataFrame]:
    side = side.upper()
    label_value = 1 if side == "BUY" else 2
    y_true = (val_df["label"].to_numpy() == label_value).astype(int)
    min_signals = int(cfg.get("min_threshold_signals", MIN_THRESHOLD_SIGNALS))
    days = max((val_df["time"].max() - val_df["time"].min()).days, 1)

    configured_session_mask = np.array([session_allowed(row, cfg) for _, row in val_df.iterrows()])
    spread_mask = val_df["spread_to_atr"].to_numpy() <= float(cfg["max_spread_to_atr"])
    regime_col = cfg.get("regime_filter_col")
    if regime_col and regime_col in val_df.columns:
        regime_mask = val_df[regime_col].to_numpy() != 0
    else:
        regime_mask = np.ones(len(val_df), dtype=bool)
    valid_mask = configured_session_mask & spread_mask & regime_mask

    thresholds = adaptive_threshold_grid(positive_proba, valid_mask)
    rows = []
    diagnostics = {
        f"max_prob_{side.lower()}": float(positive_proba[valid_mask].max()) if valid_mask.any() else 0.0,
        "valid_rows": int(valid_mask.sum()),
        "signals_at_055": int(((positive_proba >= FIXED_GRID_MIN_THRESHOLD) & valid_mask).sum()),
    }
    reason = fallback_reason(
        {
            "valid_rows": diagnostics["valid_rows"],
            "signals_at_055": diagnostics["signals_at_055"],
        },
        min_signals,
    )

    for threshold in thresholds:
        signal_mask = (positive_proba >= threshold) & valid_mask
        total = int(signal_mask.sum())
        if total < min_signals:
            continue
        precision = float(y_true[signal_mask].mean()) if signal_mask.any() else 0.0
        if precision < 0.35:
            continue
        trades = simulate_side_wf_trades(val_df, positive_proba, cfg, side, float(threshold))
        metrics = trade_metrics(trades)
        target_metrics = target_trade_metrics(metrics, cfg, days)
        rows.append({
            "side": side,
            "threshold": round(float(threshold), 2),
            "total_signals": total,
            "precision": precision,
            "trade_frequency": total / len(positive_proba) if len(positive_proba) else 0.0,
            "average_confidence": float(positive_proba[signal_mask].mean()) if signal_mask.any() else 0.0,
            "estimated_trades_per_day": total / days,
            "backtest_evaluated": True,
            "fallback_reason": "",
            **diagnostics,
            **metrics,
            **target_metrics,
        })

    if not rows:
        fallback_min = max(5, min_signals // 4)
        for threshold in thresholds:
            signal_mask = (positive_proba >= threshold) & valid_mask
            total = int(signal_mask.sum())
            if total < fallback_min:
                continue
            trades = simulate_side_wf_trades(val_df, positive_proba, cfg, side, float(threshold))
            metrics = trade_metrics(trades)
            rows.append({
                "side": side,
                "threshold": round(float(threshold), 2),
                "total_signals": total,
                "precision": float(y_true[signal_mask].mean()) if signal_mask.any() else 0.0,
                "trade_frequency": total / len(positive_proba) if len(positive_proba) else 0.0,
                "average_confidence": float(positive_proba[signal_mask].mean()) if signal_mask.any() else 0.0,
                "estimated_trades_per_day": total / days,
                "backtest_evaluated": True,
                "fallback_reason": reason,
                **diagnostics,
                **metrics,
                **target_trade_metrics(metrics, cfg, days),
            })

    if not rows:
        metrics = trade_metrics(pd.DataFrame())
        rows.append({
            "side": side,
            "threshold": 0.65,
            "total_signals": 0,
            "precision": 0.0,
            "trade_frequency": 0.0,
            "average_confidence": 0.0,
            "estimated_trades_per_day": 0.0,
            "backtest_evaluated": False,
            "fallback_reason": reason,
            **diagnostics,
            **metrics,
            **target_trade_metrics(metrics, cfg, days),
        })

    result_df = pd.DataFrame(rows)
    result_df = threshold_quality_columns(result_df.rename(columns={"precision": "combined_precision"}), min_signals)
    result_df = result_df.rename(columns={"combined_precision": "precision"})
    result_df["eligible"] = (
        (result_df["backtest_trades"] >= min_signals)
        & (result_df["backtest_profit_factor"] >= MIN_THRESHOLD_PROFIT_FACTOR)
        & (result_df["backtest_net_profit"] > 0)
    )
    result_df = result_df.sort_values(
        [
            "eligible",
            "backtest_evaluated",
            "stable_trade_count",
            "threshold_quality_score",
            "pf_dd_ratio",
            "backtest_expected_value",
            "backtest_max_drawdown",
            "backtest_profit_factor",
            "backtest_net_profit",
            "precision",
        ],
        ascending=[False, False, False, False, False, False, True, False, False, False],
    ).reset_index(drop=True)
    return result_df.iloc[0].to_dict(), result_df
