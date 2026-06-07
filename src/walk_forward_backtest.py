import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from walk_forward_threshold import simulate_side_wf_trades, simulate_wf_trades
from strategy_filters import metric_pnl_series
from utils import setup_logger

LOGGER = setup_logger("walk_forward_backtest")


def cfg_for_threshold_session_mode(cfg: dict, thresholds: dict) -> dict:
    if thresholds.get("session_mode") != "all_sessions":
        return cfg
    relaxed = cfg.copy()
    relaxed["allowed_entry_sessions"] = []
    return relaxed

def calculate_sharpe_ratio(trades_df: pd.DataFrame, start_date, end_date) -> float:
    if trades_df.empty:
        return 0.0
    # Create daily index
    dates = pd.date_range(start=pd.to_datetime(start_date).normalize(), end=pd.to_datetime(end_date).normalize(), freq="D")
    daily_pnl = pd.Series(0.0, index=dates)
    
    # Sum PnL by exit date normalize
    exit_dates = pd.to_datetime(trades_df["exit_time"]).dt.normalize()
    pnl_by_date = trades_df.groupby(exit_dates)["pnl"].sum()
    
    for dt, val in pnl_by_date.items():
        if dt in daily_pnl.index:
            daily_pnl[dt] = val
            
    mean_pnl = daily_pnl.mean()
    std_pnl = daily_pnl.std()
    
    if std_pnl > 1e-9:
        return float(np.sqrt(252) * mean_pnl / std_pnl)
    return 0.0

def calculate_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    cum_max = equity.cummax()
    drawdown = equity - cum_max
    return float(drawdown.min())

def calculate_consecutive_losses(trades_df: pd.DataFrame) -> int:
    if trades_df.empty:
        return 0
    pnl = trades_df["pnl"].values
    loss_streak = 0
    max_loss_streak = 0
    for val in pnl:
        if val <= 0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0
    return max_loss_streak

def compute_oos_metrics(trades_df: pd.DataFrame, start_date, end_date) -> dict:
    if trades_df.empty:
        return {
            "total_trades": 0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "net_profit": 0.0,
            "net_profit_price_lot": 0.0,
            "net_profit_money": 0.0,
            "pnl_money_available": False,
            "avg_pnl_per_trade": 0.0,
            "avg_pnl_per_trade_money": 0.0,
            "avg_pnl_per_trade_price_lot": 0.0,
            "avg_pnl_per_risk": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "consecutive_losses": 0,
            "average_holding_candles": 0.0,
            "trades_per_day": 0.0,
            "buy_trades": 0,
            "sell_trades": 0,
            "tp_exits": 0,
            "sl_exits": 0,
            "time_exits": 0,
        }
        
    pnl = metric_pnl_series(trades_df)
    pnl_price_lot = pd.to_numeric(trades_df.get("pnl_price_lot", trades_df["pnl"]), errors="coerce").fillna(0.0)
    pnl_money_raw = pd.to_numeric(trades_df["pnl_money"], errors="coerce") if "pnl_money" in trades_df.columns else pd.Series(dtype=float)
    pnl_money_available = bool(not pnl_money_raw.empty and pnl_money_raw.notna().any())
    equity = pnl.cumsum()
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl < 0].sum()
    days = max((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days, 1)
    
    buy_trades = int((trades_df["side"] == "BUY").sum())
    sell_trades = int((trades_df["side"] == "SELL").sum())
    
    tp_exits = int((trades_df["exit_reason"] == "TP").sum())
    sl_exits = int((trades_df["exit_reason"] == "SL").sum())
    time_exits = int((trades_df["exit_reason"] == "TIME").sum())
    
    return {
        "total_trades": int(len(trades_df)),
        "winrate": float((pnl > 0).mean()),
        "profit_factor": float(gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf") if gross_profit > 0 else 0.0,
        "net_profit": float(pnl.sum()),
        "net_profit_price_lot": float(pnl_price_lot.sum()),
        "net_profit_money": float(pnl.sum()) if pnl_money_available else 0.0,
        "pnl_money_available": pnl_money_available,
        "avg_pnl_per_trade": float(pnl.mean()),
        "avg_pnl_per_trade_money": float(pnl.mean()) if pnl_money_available else 0.0,
        "avg_pnl_per_trade_price_lot": float(pnl_price_lot.mean()),
        "avg_pnl_per_risk": float(pd.to_numeric(trades_df.get("pnl_per_risk", pd.Series(0.0, index=trades_df.index)), errors="coerce").fillna(0.0).mean()),
        "max_drawdown": calculate_max_drawdown(equity),
        "sharpe_ratio": calculate_sharpe_ratio(trades_df, start_date, end_date),
        "average_win": float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0,
        "average_loss": float(pnl[pnl <= 0].mean()) if (pnl <= 0).any() else 0.0,
        "consecutive_losses": calculate_consecutive_losses(trades_df),
        "average_holding_candles": float(trades_df["holding_candles"].mean()),
        "trades_per_day": float(len(trades_df) / days),
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "tp_exits": tp_exits,
        "sl_exits": sl_exits,
        "time_exits": time_exits,
    }

def run_oos_backtest(
    model,
    oos_df: pd.DataFrame,
    columns: list[str],
    cfg: dict,
    thresholds: dict,
    wf_cfg: dict,
) -> tuple[dict, pd.DataFrame]:
    if oos_df.empty:
        return compute_oos_metrics(pd.DataFrame(), None, None), pd.DataFrame()

    if isinstance(model, dict) and model.get("side_training_mode") == "separate":
        point = float(oos_df["point"].iloc[0]) if "point" in oos_df.columns else 0.01
        slippage_points = float(wf_cfg.get("slippage_points", 0.0))
        slippage_price = slippage_points * point
        side_frames = []
        side_models = model.get("models", {})
        for side in ["BUY", "SELL"]:
            if side not in side_models:
                continue
            side_threshold = thresholds.get(f"{side.lower()}_threshold_detail", {})
            if not isinstance(side_threshold, dict):
                side_threshold = {"threshold": side_threshold}
            threshold = float(side_threshold.get("threshold", cfg[f"{side.lower()}_threshold"]))
            side_model = side_models[side]
            if isinstance(side_model, list):
                positive_proba = np.mean([m.predict_proba(oos_df[columns])[:, 1] for m in side_model], axis=0)
            else:
                positive_proba = side_model.predict_proba(oos_df[columns])[:, 1]
            side_trades = simulate_side_wf_trades(
                oos_df,
                positive_proba,
                cfg_for_threshold_session_mode(cfg, side_threshold),
                side,
                threshold,
                slippage=slippage_price,
            )
            if not side_trades.empty:
                side_frames.append(side_trades)

        trades_df = pd.concat(side_frames, ignore_index=True) if side_frames else pd.DataFrame()
        if not trades_df.empty:
            trades_df = trades_df.sort_values(["entry_time", "side"]).reset_index(drop=True)
        trades_df = apply_commission(trades_df, cfg, wf_cfg)
        start_date = oos_df["time"].min()
        end_date = oos_df["time"].max()
        return compute_oos_metrics(trades_df, start_date, end_date), trades_df

    if isinstance(model, list):
        proba = np.mean([m.predict_proba(oos_df[columns]) for m in model], axis=0)
    else:
        proba = model.predict_proba(oos_df[columns])
        
    buy_thresh = thresholds.get("buy_threshold", cfg["buy_threshold"])
    sell_thresh = thresholds.get("sell_threshold", cfg["sell_threshold"])
    nt_zone = thresholds.get("no_trade_zone", 0.0)
    simulation_cfg = cfg_for_threshold_session_mode(cfg, thresholds)
    
    # Calculate slippage in price units
    point = float(oos_df["point"].iloc[0]) if "point" in oos_df.columns else 0.01
    slippage_points = float(wf_cfg.get("slippage_points", 0.0))
    slippage_price = slippage_points * point
    
    # Simulate trades
    trades_df = simulate_wf_trades(
        oos_df,
        proba,
        simulation_cfg,
        buy_thresh,
        sell_thresh,
        nt_zone,
        slippage=slippage_price
    )
    
    trades_df = apply_commission(trades_df, cfg, wf_cfg)
        
    start_date = oos_df["time"].min()
    end_date = oos_df["time"].max()
    
    metrics = compute_oos_metrics(trades_df, start_date, end_date)
    return metrics, trades_df


def apply_commission(trades_df: pd.DataFrame, cfg: dict, wf_cfg: dict) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df

    lot = float(cfg.get("default_lot", 0.01))
    commission_per_lot = float(wf_cfg.get("commission_per_lot", 0.0))
    commission = lot * commission_per_lot
    
    if commission > 0.0:
        trades_df["commission"] = commission
        if "pnl_money" in trades_df.columns and pd.to_numeric(trades_df["pnl_money"], errors="coerce").notna().any():
            trades_df["pnl_money"] = pd.to_numeric(trades_df["pnl_money"], errors="coerce") - commission
            risk_money = pd.to_numeric(trades_df.get("risk_money", pd.Series(0.0, index=trades_df.index)), errors="coerce")
            trades_df["pnl_per_risk"] = np.where(risk_money > 0, trades_df["pnl_money"] / risk_money, 0.0)
        else:
            trades_df["pnl"] = trades_df["pnl"] - commission
            if "pnl_price_lot" in trades_df.columns:
                trades_df["pnl_price_lot"] = trades_df["pnl_price_lot"] - commission
    else:
        trades_df["commission"] = 0.0
    return trades_df

def save_equity_curve_wf(equity: pd.Series, path: Path, title: str = "OOS Equity Curve") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    if not equity.empty:
        equity.reset_index(drop=True).plot(grid=True, color="#00C0A3", linewidth=2)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Trade Number", fontsize=10)
    plt.ylabel("Net Profit", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
