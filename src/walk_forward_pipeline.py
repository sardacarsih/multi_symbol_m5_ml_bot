import argparse
import json
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

# Import existing modules
from config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SYMBOLS,
    WALK_FORWARD_CONFIG,
)
from download_mt5_data import download_symbol, connect_mt5
from features import add_features
from labeling import atr_barrier_labels, write_label_diagnostics
from train import feature_columns, sample_weights, fit_model, build_model
from utils import ensure_dirs, save_json, save_model, setup_logger, symbol_to_filename
from walk_forward_threshold import optimize_side_threshold_wf, optimize_thresholds_wf
from walk_forward_backtest import run_oos_backtest, save_equity_curve_wf, compute_oos_metrics
from strategy_filters import metric_pnl_series
from calibration_utils import calibrate_prefit_classifier
from production import (
    STRICT_MAX_DRAWDOWN_ABS,
    STRICT_MIN_PROFIT_FACTOR,
    STRICT_MIN_SIDE_TRADES,
    artifact_hashes,
    data_quality_report,
    strict_side_gate_failures,
)

# Try importing optuna
try:
    import optuna
except ImportError:
    optuna = None

import shutil
import joblib

LOGGER = setup_logger("walk_forward_pipeline")

MIN_WALK_FORWARD_LIVE_TRADES = 30
MIN_WALK_FORWARD_LIVE_PROFIT_FACTOR = STRICT_MIN_PROFIT_FACTOR
MAX_WALK_FORWARD_SIDE_CONCENTRATION = 0.90
MIN_WALK_FORWARD_LIVE_CYCLE_TRADES = 10
SIDES = ("BUY", "SELL")


def uses_separate_side_models(cfg: dict) -> bool:
    return cfg.get("side_training_mode", "separate") == "separate"


def binary_target(labels: pd.Series, side: str) -> pd.Series:
    label_value = 1 if side.upper() == "BUY" else 2
    return (labels == label_value).astype(int)


def binary_sample_weights(y: pd.Series) -> pd.Series | None:
    counts = y.value_counts().to_dict()
    if len(counts) < 2:
        return None
    total = len(y)
    classes = 2
    weights = {label: total / (classes * count) for label, count in counts.items() if count > 0}
    return y.map(weights).astype(float)


def build_binary_model(params: dict | None = None, seed: int = 20260605) -> XGBClassifier:
    if params:
        model_params = params.copy()
        model_params["objective"] = "binary:logistic"
        model_params["eval_metric"] = "logloss"
        model_params["random_state"] = seed
        model_params["n_jobs"] = -1
        model_params.pop("num_class", None)
        return XGBClassifier(**model_params)
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=4,
        learning_rate=0.03,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=seed,
        n_jobs=-1,
    )


def fit_calibrated_binary_ensemble(
    X_train,
    y_train: pd.Series,
    X_val,
    y_val: pd.Series,
    best_params: dict | None = None,
) -> list:
    if y_train.nunique() < 2:
        raise ValueError("binary side model requires both positive and negative training labels")
    seeds = [20260605, 12345, 98765]
    weights = binary_sample_weights(y_train)
    ensemble = []
    for seed in seeds:
        estimator = build_binary_model(best_params, seed)
        try:
            estimator.fit(
                X_train,
                y_train,
                sample_weight=weights,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=50,
            )
        except Exception:
            estimator.fit(X_train, y_train, sample_weight=weights)
        if y_val.nunique() < 2:
            LOGGER.warning("Skipping binary calibration because validation target has one class: %s", y_val.value_counts().to_dict())
            ensemble.append(estimator)
        else:
            ensemble.append(calibrate_prefit_classifier(estimator, X_val, y_val))
    return ensemble


def side_positive_proba(model, rows: pd.DataFrame) -> np.ndarray:
    if isinstance(model, list):
        return np.mean([member.predict_proba(rows)[:, 1] for member in model], axis=0)
    return model.predict_proba(rows)[:, 1]


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS.ss."""
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    remaining_seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:05.2f}"


def walk_forward_output_mode(n_trials: int) -> str:
    """Return a stable output mode name for a walk-forward run."""
    if n_trials <= 0:
        return "non_tuned"
    return f"tuned_{n_trials}"


def walk_forward_output_paths(symbol: str, n_trials: int) -> dict[str, Path | str]:
    mode = walk_forward_output_mode(n_trials)
    folder_name = f"walk_forward_{mode}"
    return {
        "mode": mode,
        "model_dir": MODELS_DIR / symbol / folder_name,
        "report_dir": REPORTS_DIR / symbol / folder_name,
        "live_meta_path": MODELS_DIR / symbol / "live_model_meta.json",
    }


def is_cycle_complete(cycle_model_dir: Path, cycle_report_dir: Path) -> bool:
    """Check if a walk-forward cycle has all required output artifacts."""
    common_model_files = [
        "best_threshold.json",
        "feature_columns.json",
        "cycle_meta.json",
        "buy_model.joblib",
        "sell_model.joblib",
        "buy_threshold.json",
        "sell_threshold.json",
        "side_model_meta.json",
    ]
    required_report_files = ["oos_metrics.json", "oos_backtest.csv"]
    for f in common_model_files:
        if not (cycle_model_dir / f).exists():
            return False
    for f in required_report_files:
        if not (cycle_report_dir / f).exists():
            return False
    return True


def normalize_trade_datetimes(trades: pd.DataFrame) -> pd.DataFrame:
    """Return trades with comparable datetime columns for aggregation/reporting."""
    if trades.empty:
        return trades.copy()

    normalized = trades.copy()
    required_columns = ["entry_time", "exit_time"]
    if not all(column in normalized.columns for column in required_columns):
        return normalized.iloc[0:0].copy()

    for column in required_columns:
        normalized[column] = pd.to_datetime(normalized[column], errors="coerce")

    return normalized.dropna(subset=required_columns).reset_index(drop=True)


def load_completed_cycle(cycle_model_dir: Path, cycle_report_dir: Path, split: dict) -> tuple:
    """Load artifacts from a previously completed cycle for aggregation."""
    with open(cycle_model_dir / "best_threshold.json", "r") as f:
        thresholds = json.load(f)
    with open(cycle_report_dir / "oos_metrics.json", "r") as f:
        oos_metrics = json.load(f)
    oos_trades_path = cycle_report_dir / "oos_backtest.csv"
    oos_trades = pd.read_csv(oos_trades_path)
    oos_trades = normalize_trade_datetimes(oos_trades)
    
    summary_row = {
        "cycle": split["cycle"],
        "train_start": split["train_start"],
        "train_end": split["train_end"],
        "val_start": split["val_start"],
        "val_end": split["val_end"],
        "oos_start": split["oos_start"],
        "oos_end": split["oos_end"],
        "buy_threshold": thresholds.get("buy_threshold", 0),
        "sell_threshold": thresholds.get("sell_threshold", 0),
        "no_trade_zone": thresholds.get("no_trade_zone", 0),
        "session_mode": thresholds.get("session_mode", "configured_sessions"),
        "backtest_trades_per_day": thresholds.get("backtest_trades_per_day", 0.0),
        "target_min_trades_per_day": thresholds.get("target_min_trades_per_day", 0.0),
        "target_max_trades_per_day": thresholds.get("target_max_trades_per_day", 0.0),
        "target_feasible": thresholds.get("target_feasible", False),
        "trade_frequency_gap": thresholds.get("trade_frequency_gap", 0.0),
        "oos_trades": oos_metrics.get("total_trades", 0),
        "oos_net_profit": oos_metrics.get("net_profit", 0),
        "oos_net_profit_price_lot": oos_metrics.get("net_profit_price_lot", oos_metrics.get("net_profit", 0)),
        "oos_net_profit_money": oos_metrics.get("net_profit_money", 0),
        "oos_pnl_money_available": oos_metrics.get("pnl_money_available", False),
        "oos_avg_pnl_per_trade": oos_metrics.get("avg_pnl_per_trade", 0),
        "oos_avg_pnl_per_trade_money": oos_metrics.get("avg_pnl_per_trade_money", 0),
        "oos_avg_pnl_per_risk": oos_metrics.get("avg_pnl_per_risk", 0),
        "oos_profit_factor": oos_metrics.get("profit_factor", 0),
        "oos_max_drawdown": oos_metrics.get("max_drawdown", 0),
        "oos_winrate": oos_metrics.get("winrate", 0),
        "buy_trades": oos_metrics.get("buy_trades", 0),
        "sell_trades": oos_metrics.get("sell_trades", 0),
        "fallback_reason": thresholds.get("fallback_reason", ""),
        **cycle_side_metrics(oos_trades),
    }
    return summary_row, oos_trades


def aggregate_oos_metrics(all_trades: list[pd.DataFrame]) -> dict:
    if not all_trades:
        return compute_oos_metrics(pd.DataFrame(), None, None)

    trades = pd.concat([frame for frame in all_trades if not frame.empty], ignore_index=True)
    trades = normalize_trade_datetimes(trades)
    if trades.empty:
        return compute_oos_metrics(pd.DataFrame(), None, None)

    return compute_oos_metrics(trades, trades["entry_time"].min(), trades["exit_time"].max())


def aggregate_oos_metrics_for_side(all_trades: list[pd.DataFrame], side: str) -> dict:
    if not all_trades:
        return compute_oos_metrics(pd.DataFrame(), None, None)

    trades = pd.concat([frame for frame in all_trades if not frame.empty], ignore_index=True)
    trades = normalize_trade_datetimes(trades)
    trades = trades[trades["side"] == side].reset_index(drop=True)
    if trades.empty:
        return compute_oos_metrics(pd.DataFrame(), None, None)

    return compute_oos_metrics(trades, trades["entry_time"].min(), trades["exit_time"].max())


def cycle_side_metrics(oos_trades: pd.DataFrame) -> dict:
    metrics = {}
    normalized = normalize_trade_datetimes(oos_trades)
    for side in ["BUY", "SELL"]:
        side_trades = normalized[normalized["side"] == side].reset_index(drop=True) if not normalized.empty else pd.DataFrame()
        side_metrics = compute_oos_metrics(
            side_trades,
            side_trades["entry_time"].min() if not side_trades.empty else None,
            side_trades["exit_time"].max() if not side_trades.empty else None,
        )
        prefix = side.lower()
        metrics[f"{prefix}_net_profit"] = side_metrics["net_profit"]
        metrics[f"{prefix}_profit_factor"] = side_metrics["profit_factor"]
        metrics[f"{prefix}_winrate"] = side_metrics["winrate"]
    return metrics


def live_allowed_sides(
    all_trades: list[pd.DataFrame],
    min_total_trades: int = STRICT_MIN_SIDE_TRADES,
    min_profit_factor: float = MIN_WALK_FORWARD_LIVE_PROFIT_FACTOR,
) -> tuple[list[str], dict[str, dict]]:
    side_metrics = {side: aggregate_oos_metrics_for_side(all_trades, side) for side in ["BUY", "SELL"]}
    allowed = []
    for side, metrics in side_metrics.items():
        total_trades = int(metrics.get("total_trades", 0))
        net_profit = float(metrics.get("net_profit", 0.0))
        profit_factor = float(metrics.get("profit_factor", 0.0))
        ev = float(metrics.get("avg_pnl_per_trade", 0.0))
        if (
            total_trades >= min_total_trades
            and net_profit > 0.0
            and profit_factor >= min_profit_factor
            and ev > 0.0
            and bool(metrics.get("pnl_money_available", False))
        ):
            allowed.append(side)
    return allowed, side_metrics


def walk_forward_gate_failures(
    summary_rows: list[dict],
    all_trades: list[pd.DataFrame],
    expected_cycles: int,
    symbol: str | None = None,
    min_total_trades: int = MIN_WALK_FORWARD_LIVE_TRADES,
    min_profit_factor: float = MIN_WALK_FORWARD_LIVE_PROFIT_FACTOR,
    max_side_concentration: float = MAX_WALK_FORWARD_SIDE_CONCENTRATION,
) -> list[str]:
    failures = []
    completed_cycles = len(summary_rows)
    if completed_cycles < expected_cycles:
        failures.append(f"walk-forward incomplete: {completed_cycles}/{expected_cycles} cycles complete")

    metrics = aggregate_oos_metrics(all_trades)
    total_trades = int(metrics.get("total_trades", 0))
    net_profit = float(metrics.get("net_profit", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    buy_trades = int(metrics.get("buy_trades", 0))
    sell_trades = int(metrics.get("sell_trades", 0))

    if total_trades < min_total_trades:
        failures.append(f"aggregate OOS trades below minimum: {total_trades} < {min_total_trades}")
    if net_profit <= 0.0:
        failures.append(f"aggregate OOS net profit is not positive: {net_profit:.4f}")
    allowed_sides, _ = live_allowed_sides(all_trades, min_total_trades, min_profit_factor)
    if symbol == "XAUUSD" and not allowed_sides:
        failures.append("XAUUSD has no individually profitable side that passes live gates")
    if np.isnan(profit_factor) or profit_factor < min_profit_factor:
        if allowed_sides:
            LOGGER.warning(
                "Aggregate OOS profit factor %.3f is below %.3f; allowing side-filtered live selection for %s",
                profit_factor,
                min_profit_factor,
                ", ".join(allowed_sides),
            )
        else:
            failures.append(f"aggregate OOS profit factor below minimum: {profit_factor:.3f} < {min_profit_factor:.3f}")

    if total_trades > 0:
        dominant_ratio = max(buy_trades, sell_trades) / total_trades
        profitable_cycles = sum(
            1 for row in summary_rows
            if int(row.get("oos_trades", 0)) > 0 and float(row.get("oos_net_profit", 0.0)) > 0.0
        )
        if dominant_ratio > max_side_concentration and profitable_cycles < 3:
            failures.append(
                "OOS side distribution is too concentrated "
                f"({buy_trades} BUY, {sell_trades} SELL) without enough profitable cycles"
            )

    allowed_sides, side_metrics = live_allowed_sides(all_trades, min_total_trades, min_profit_factor)
    failures.extend(
        strict_side_gate_failures(
            side_metrics,
            allowed_sides,
            min_trades=min_total_trades,
            min_profit_factor=min_profit_factor,
            max_drawdown_abs=STRICT_MAX_DRAWDOWN_ABS,
        )
    )
    return failures


def row_profit_factor(row: dict, key: str = "oos_profit_factor") -> float:
    profit_factor = float(row.get(key, 0.0))
    return 999.0 if np.isfinite(profit_factor) and profit_factor == float("inf") else profit_factor


def healthy_oos_cycle(
    row: dict,
    min_trades: int = MIN_WALK_FORWARD_LIVE_CYCLE_TRADES,
    min_profit_factor: float = MIN_WALK_FORWARD_LIVE_PROFIT_FACTOR,
) -> bool:
    profit_factor = float(row.get("oos_profit_factor", 0.0))
    if not np.isfinite(profit_factor):
        profit_factor = 999.0
    return (
        int(row.get("oos_trades", 0)) >= min_trades
        and float(row.get("oos_net_profit", 0.0)) > 0.0
        and profit_factor >= min_profit_factor
    )


def healthy_side_oos_cycle(
    row: dict,
    side: str,
    min_trades: int = MIN_WALK_FORWARD_LIVE_CYCLE_TRADES,
    min_profit_factor: float = MIN_WALK_FORWARD_LIVE_PROFIT_FACTOR,
) -> bool:
    prefix = side.lower()
    profit_factor = float(row.get(f"{prefix}_profit_factor", 0.0))
    if not np.isfinite(profit_factor):
        profit_factor = 999.0
    return (
        healthy_oos_cycle(row, min_trades, min_profit_factor)
        and int(row.get(f"{prefix}_trades", 0)) > 0
        and float(row.get(f"{prefix}_net_profit", 0.0)) > 0.0
        and profit_factor >= min_profit_factor
    )


def select_live_deploy_cycle(summary_rows: list[dict], allowed_sides: list[str] | None = None) -> dict | None:
    candidates = [row for row in summary_rows if healthy_oos_cycle(row)]
    if allowed_sides:
        candidates = [
            row for row in candidates
            if any(healthy_side_oos_cycle(row, side) for side in allowed_sides)
        ]
    if not candidates:
        return None

    if allowed_sides:
        best = max(candidates, key=lambda row: int(row["cycle"]))
        selection = "latest_side_filtered_walk_forward"
    else:
        def score(row: dict) -> tuple[float, float, int]:
            profit_factor = float(row.get("oos_profit_factor", 0.0))
            if not np.isfinite(profit_factor):
                profit_factor = 999.0
            drawdown = abs(float(row.get("oos_max_drawdown", 0.0)))
            pf_dd_ratio = profit_factor / (drawdown + 0.05) if profit_factor > 0 else 0.0
            return pf_dd_ratio, float(row.get("oos_net_profit", 0.0)), int(row.get("oos_trades", 0))

        best = max(candidates, key=score)
        selection = "best_oos_pf_dd_ratio"

    profit_factor = float(best.get("oos_profit_factor", 0.0))
    if not np.isfinite(profit_factor):
        profit_factor = 999.0
    drawdown = abs(float(best.get("oos_max_drawdown", 0.0)))
    pf_dd_ratio = profit_factor / (drawdown + 0.05) if profit_factor > 0 else 0.0
    return {
        "source": "walk_forward",
        "gate_status": "passed",
        "cycle": f"cycle_{int(best['cycle']):03d}",
        "selection": selection,
        "oos_profit_factor": float(best.get("oos_profit_factor", 0.0)),
        "oos_max_drawdown": float(best.get("oos_max_drawdown", 0.0)),
        "oos_net_profit": float(best.get("oos_net_profit", 0.0)),
        "oos_trades": int(best.get("oos_trades", 0)),
        "pf_dd_ratio": float(pf_dd_ratio),
        "allowed_sides": allowed_sides or ["BUY", "SELL"],
    }


def optuna_objective(trial, X_train, y_train, X_val, y_val, w_train, val_df: pd.DataFrame | None = None, cfg: dict | None = None) -> float:
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 20260605,
        "n_jobs": -1,
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
    }

    model = XGBClassifier(**params)
    
    try:
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=50
        )
        best_iteration = getattr(model, "best_iteration", params["n_estimators"])
        trial.set_user_attr("best_n_estimators", int(best_iteration))
    except Exception:
        model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)
        trial.set_user_attr("best_n_estimators", params["n_estimators"])
        
    preds = model.predict_proba(X_val)
    loss = log_loss(y_val, preds, labels=[0, 1, 2])
    if val_df is None or cfg is None:
        return loss

    try:
        _thresholds, grid = optimize_thresholds_wf(val_df, preds, cfg)
        best = grid.iloc[0]
        pf_dd = float(best.get("pf_dd_ratio", 0.0))
        expected = float(best.get("backtest_expected_value", 0.0))
        net_profit = float(best.get("backtest_net_profit", 0.0))
        drawdown = abs(float(best.get("backtest_max_drawdown", 0.0)))
        trades = float(best.get("backtest_trades", 0.0))
        min_trades = max(float(cfg.get("min_threshold_signals", 0.0)), 10.0)
        trade_penalty = max(0.0, (min_trades - trades) / min_trades)
        validation_score = (
            min(pf_dd, 10.0)
            + max(min(expected, 10.0), -10.0)
            + max(min(net_profit / 1000.0, 1.0), -1.0)
            - min(drawdown / 1000.0, 1.0)
            - trade_penalty
        )
        return float(loss - 0.05 * validation_score)
    except Exception as exc:
        LOGGER.warning("Optuna validation-backtest scoring failed; using logloss only: %s", exc)
        return loss

def generate_walk_forward_splits(df: pd.DataFrame, cfg: dict, wf_cfg: dict) -> list[dict]:
    # Sort dataframe by time
    df = df.sort_values("time").reset_index(drop=True)
    min_time = df["time"].min()
    max_time = df["time"].max()
    
    train_len_months = wf_cfg.get("training_window_months", 12)
    val_len_months = wf_cfg.get("validation_months", 2)
    oos_len_months = wf_cfg.get("oos_months", 1)
    
    freq = wf_cfg.get("retrain_frequency", "weekly")
    expanding = wf_cfg.get("expanding_window", False)
    
    splits = []
    
    # Calculate first OOS start date
    current_oos_start = min_time + pd.DateOffset(months=train_len_months + val_len_months)
    
    if freq == "weekly":
        shift_delta = pd.DateOffset(weeks=1)
        oos_delta = pd.DateOffset(weeks=1)
    else:  # monthly
        shift_delta = pd.DateOffset(months=1)
        oos_delta = pd.DateOffset(months=oos_len_months)
        
    cycle_idx = 1
    while current_oos_start + oos_delta <= max_time:
        current_oos_end = current_oos_start + oos_delta
        current_val_end = current_oos_start
        current_val_start = current_val_end - pd.DateOffset(months=val_len_months)
        current_train_end = current_val_start
        
        if expanding:
            current_train_start = min_time
        else:
            current_train_start = current_train_end - pd.DateOffset(months=train_len_months)
            
        splits.append({
            "cycle": cycle_idx,
            "train_start": current_train_start,
            "train_end": current_train_end,
            "val_start": current_val_start,
            "val_end": current_val_end,
            "oos_start": current_oos_start,
            "oos_end": current_oos_end
        })
        
        # Shift forward
        current_oos_start += shift_delta
        cycle_idx += 1
        
    return splits

def run_tuning(
    X_train,
    y_train,
    X_val,
    y_val,
    w_train,
    n_trials: int,
    val_df: pd.DataFrame | None = None,
    cfg: dict | None = None,
) -> dict:
    if optuna is None or n_trials <= 0:
        LOGGER.info("Skipping Optuna tuning (trials <= 0 or optuna not installed). Using default parameters.")
        return {}
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    
    def progress_callback(study, trial):
        LOGGER.info("Optuna Trial %d/%d. Best LogLoss: %.6f", trial.number + 1, n_trials, study.best_value)

    study.optimize(
        lambda t: optuna_objective(t, X_train, y_train, X_val, y_val, w_train, val_df, cfg),
        n_trials=n_trials,
        callbacks=[progress_callback]
    )
    
    best_params = study.best_params
    best_n_estimators = study.best_trials[0].user_attrs.get("best_n_estimators", best_params.get("n_estimators", 500))
    best_params["n_estimators"] = int(best_n_estimators)
    
    LOGGER.info("Best validation LogLoss: %.6f", study.best_value)
    return best_params


def binary_optuna_objective(trial, X_train, y_train, X_val, y_val, w_train) -> float:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 20260605,
        "n_jobs": -1,
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
    }
    model = XGBClassifier(**params)
    try:
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=50,
        )
        best_iteration = getattr(model, "best_iteration", params["n_estimators"])
        trial.set_user_attr("best_n_estimators", int(best_iteration))
    except Exception:
        model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)
        trial.set_user_attr("best_n_estimators", params["n_estimators"])
    preds = model.predict_proba(X_val)
    return log_loss(y_val, preds, labels=[0, 1])


def run_binary_tuning(X_train, y_train, X_val, y_val, w_train, n_trials: int) -> dict:
    if optuna is None or n_trials <= 0 or y_train.nunique() < 2:
        LOGGER.info("Skipping binary Optuna tuning (trials <= 0, optuna missing, or single-class target).")
        return {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: binary_optuna_objective(t, X_train, y_train, X_val, y_val, w_train), n_trials=n_trials)
    best_params = study.best_params
    best_n_estimators = study.best_trials[0].user_attrs.get("best_n_estimators", best_params.get("n_estimators", 500))
    best_params["n_estimators"] = int(best_n_estimators)
    LOGGER.info("Best binary validation LogLoss: %.6f", study.best_value)
    return best_params

def generate_reports(
    symbol: str,
    summary_rows: list[dict],
    all_trades: list[pd.DataFrame],
    base_report_dir: Path,
    expected_cycles: int | None = None,
    gate_failures: list[str] | None = None,
) -> None:
    ensure_dirs([symbol])
    
    # 1. Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(base_report_dir / "walk_forward_summary.csv", index=False)
    
    # 2. Aggregated OOS Trades
    if all_trades:
        agg_trades = pd.concat(all_trades, ignore_index=True)
        agg_trades = normalize_trade_datetimes(agg_trades)
        agg_trades.to_csv(base_report_dir / "aggregated_oos_trades.csv", index=False)
        
        # Calculate aggregated equity curve
        agg_trades = agg_trades.sort_values("exit_time").reset_index(drop=True)
        agg_pnl = metric_pnl_series(agg_trades)
        agg_equity = agg_pnl.cumsum()
        save_equity_curve_wf(agg_equity, base_report_dir / "aggregated_oos_equity.png", title=f"Aggregated OOS Equity Curve - {symbol}")
        
        # Monthly return calculations
        agg_trades["month"] = agg_trades["exit_time"].dt.to_period("M")
        agg_trades["_metric_pnl"] = agg_pnl
        monthly_pnl = agg_trades.groupby("month")["_metric_pnl"].sum().reset_index(name="pnl")
        monthly_pnl.to_csv(base_report_dir / "monthly_returns.csv", index=False)
        agg_trades = agg_trades.drop(columns=["_metric_pnl"])
        
        overall_metrics = compute_oos_metrics(agg_trades, agg_trades["entry_time"].min(), agg_trades["exit_time"].max())
    else:
        agg_trades = pd.DataFrame()
        overall_metrics = compute_oos_metrics(pd.DataFrame(), None, None)

    expected_cycles = expected_cycles or len(summary_rows)
    gate_failures = gate_failures or []
    completed_cycles = len(summary_rows)
    allowed_sides, side_metrics = live_allowed_sides(all_trades)
    fallback_cycles = [
        int(row["cycle"]) for row in summary_rows
        if row.get("fallback_reason") == "probabilities_below_fixed_grid"
    ]
    latest_cycle = max((int(row["cycle"]) for row in summary_rows), default=0)
    latest_cycle_status = (
        "complete" if latest_cycle >= expected_cycles
        else f"waiting for cycle {latest_cycle + 1:03d}/{expected_cycles:03d}"
    )
    pf_gate_status = "PASS" if overall_metrics["profit_factor"] >= MIN_WALK_FORWARD_LIVE_PROFIT_FACTOR else (
        f"SIDE-FILTERED ({', '.join(allowed_sides)})" if allowed_sides else "FAIL"
    )
        
    # 3. Human readable markdown report
    report_path = base_report_dir / "walk_forward_report.md"
    
    report_content = f"""# Walk-Forward Validation Report: {symbol}

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Trades** | {overall_metrics['total_trades']} |
| **Win Rate** | {overall_metrics['winrate']:.2%} |
| **Profit Factor** | {overall_metrics['profit_factor']:.3f} |
| **Net Profit (metric basis)** | {overall_metrics['net_profit']:.4f} |
| **Net Profit Price-Lot** | {overall_metrics['net_profit_price_lot']:.4f} |
| **Net Profit Money** | {overall_metrics['net_profit_money']:.4f} |
| **Money PnL Available** | {'YES' if overall_metrics['pnl_money_available'] else 'NO'} |
| **Avg PnL / Trade** | {overall_metrics['avg_pnl_per_trade']:.4f} |
| **Avg PnL / Trade Money** | {overall_metrics['avg_pnl_per_trade_money']:.4f} |
| **Avg PnL / Risk** | {overall_metrics['avg_pnl_per_risk']:.4f} |
| **Max Drawdown (metric basis)** | {overall_metrics['max_drawdown']:.4f} |
| **Sharpe Ratio** | {overall_metrics['sharpe_ratio']:.3f} |
| **Avg Win** | {overall_metrics['average_win']:.4f} |
| **Avg Loss** | {overall_metrics['average_loss']:.4f} |
| **Consecutive Losses** | {overall_metrics['consecutive_losses']} |
| **Trades Per Day** | {overall_metrics['trades_per_day']:.3f} |

### Trade Distribution
- **BUY Trades**: {overall_metrics['buy_trades']}
- **SELL Trades**: {overall_metrics['sell_trades']}
- **TP Exits**: {overall_metrics['tp_exits']}
- **SL Exits**: {overall_metrics['sl_exits']}
- **TIME Exits**: {overall_metrics['time_exits']}

---

### Side-Specific Readiness

| Side | Trades | Net Profit | Net Profit Money | Avg PnL/Trade | Profit Factor | Win Rate | Live Allowed |
|------|--------|------------|------------------|---------------|---------------|----------|--------------|
| BUY | {side_metrics['BUY']['total_trades']} | {side_metrics['BUY']['net_profit']:.4f} | {side_metrics['BUY']['net_profit_money']:.4f} | {side_metrics['BUY']['avg_pnl_per_trade']:.4f} | {side_metrics['BUY']['profit_factor']:.3f} | {side_metrics['BUY']['winrate']:.2%} | {'YES' if 'BUY' in allowed_sides else 'NO'} |
| SELL | {side_metrics['SELL']['total_trades']} | {side_metrics['SELL']['net_profit']:.4f} | {side_metrics['SELL']['net_profit_money']:.4f} | {side_metrics['SELL']['avg_pnl_per_trade']:.4f} | {side_metrics['SELL']['profit_factor']:.3f} | {side_metrics['SELL']['winrate']:.2%} | {'YES' if 'SELL' in allowed_sides else 'NO'} |

---

## Walk-Forward Readiness Review

| Check | Status |
|-------|--------|
| Completed Cycles | {completed_cycles}/{expected_cycles} |
| Latest Cycle Status | {latest_cycle_status} |
| Aggregate OOS Net Profit > 0 | {'PASS' if overall_metrics['net_profit'] > 0 else 'FAIL'} |
| Aggregate OOS Profit Factor >= {MIN_WALK_FORWARD_LIVE_PROFIT_FACTOR:.2f} | {pf_gate_status} |
| Aggregate OOS Trades >= {MIN_WALK_FORWARD_LIVE_TRADES} | {'PASS' if overall_metrics['total_trades'] >= MIN_WALK_FORWARD_LIVE_TRADES else 'FAIL'} |
| Live Allowed Sides | {', '.join(allowed_sides) if allowed_sides else 'none'} |
| Adaptive Fallback Concerns | {', '.join(f'cycle_{cycle:03d}' for cycle in fallback_cycles) if fallback_cycles else 'none'} |

### Gate Result
{'READY for live model selection' if not gate_failures else 'BLOCKED from live model selection'}

{chr(10).join(f'- {failure}' for failure in gate_failures) if gate_failures else f"- Side-filtered live selection allowed for: {', '.join(allowed_sides)}."}

---

## Cycle Breakdown

| Cycle | OOS Period | Trades | Net Profit | Money | Avg/Trade | Win Rate | Profit Factor | BUY | SELL | Session Mode | Val Trades/Day | Target OK | Buy Thresh | Sell Thresh | No Trade Zone | Concern |
|-------|------------|--------|------------|-------|-----------|----------|---------------|-----|------|--------------|----------------|-----------|------------|-------------|---------------|---------|
"""
    for row in summary_rows:
        concern = row.get("fallback_reason") or ""
        session_mode = row.get("session_mode", "configured_sessions")
        target_ok = "YES" if bool(row.get("target_feasible", False)) else "NO"
        val_trades_per_day = float(row.get("backtest_trades_per_day", 0.0))
        report_content += f"| {row['cycle']} | {row['oos_start'].strftime('%Y-%m-%d')} - {row['oos_end'].strftime('%Y-%m-%d')} | {row['oos_trades']} | {row['oos_net_profit']:.4f} | {row.get('oos_net_profit_money', 0.0):.4f} | {row.get('oos_avg_pnl_per_trade', 0.0):.4f} | {row['oos_winrate']:.2%} | {row['oos_profit_factor']:.3f} | {row.get('buy_trades', 0)} | {row.get('sell_trades', 0)} | {session_mode} | {val_trades_per_day:.2f} | {target_ok} | {row['buy_threshold']:.2f} | {row['sell_threshold']:.2f} | {row['no_trade_zone']:.2f} | {concern} |\n"
        
    report_content += f"""
---

## Aggregated Equity Curve
![Aggregated OOS Equity Curve](aggregated_oos_equity.png)
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    LOGGER.info("Walk-forward reports successfully written to %s", base_report_dir)


def walk_forward_mode_label(report_dir: Path) -> str:
    name = report_dir.name
    if name.startswith("walk_forward_"):
        return name.removeprefix("walk_forward_")
    return name


def summarize_walk_forward_report_dir(report_dir: Path) -> dict | None:
    summary_path = report_dir / "walk_forward_summary.csv"
    trades_path = report_dir / "aggregated_oos_trades.csv"
    if not summary_path.exists() or not trades_path.exists():
        return None

    summary_df = pd.read_csv(summary_path)
    trades_df = pd.read_csv(trades_path)
    trades_df = normalize_trade_datetimes(trades_df)
    overall = compute_oos_metrics(
        trades_df,
        trades_df["entry_time"].min() if not trades_df.empty else None,
        trades_df["exit_time"].max() if not trades_df.empty else None,
    )
    buy_metrics = compute_oos_metrics(
        trades_df[trades_df["side"] == "BUY"].reset_index(drop=True) if not trades_df.empty else pd.DataFrame(),
        trades_df.loc[trades_df["side"] == "BUY", "entry_time"].min() if not trades_df.empty and (trades_df["side"] == "BUY").any() else None,
        trades_df.loc[trades_df["side"] == "BUY", "exit_time"].max() if not trades_df.empty and (trades_df["side"] == "BUY").any() else None,
    )
    sell_metrics = compute_oos_metrics(
        trades_df[trades_df["side"] == "SELL"].reset_index(drop=True) if not trades_df.empty else pd.DataFrame(),
        trades_df.loc[trades_df["side"] == "SELL", "entry_time"].min() if not trades_df.empty and (trades_df["side"] == "SELL").any() else None,
        trades_df.loc[trades_df["side"] == "SELL", "exit_time"].max() if not trades_df.empty and (trades_df["side"] == "SELL").any() else None,
    )
    latest_3 = summary_df.tail(3)
    return {
        "mode": walk_forward_mode_label(report_dir),
        "report_dir": report_dir,
        "cycles": len(summary_df),
        "positive_cycles": int((summary_df["oos_net_profit"] > 0).sum()) if "oos_net_profit" in summary_df else 0,
        "latest_3_net": float(latest_3["oos_net_profit"].sum()) if "oos_net_profit" in latest_3 else 0.0,
        "latest_3_trades": int(latest_3["oos_trades"].sum()) if "oos_trades" in latest_3 else 0,
        "latest_3_min_pf": float(latest_3["oos_profit_factor"].replace([np.inf, -np.inf], np.nan).min()) if "oos_profit_factor" in latest_3 else 0.0,
        "overall": overall,
        "BUY": buy_metrics,
        "SELL": sell_metrics,
    }


def generate_walk_forward_comparison_report(symbol: str) -> Path | None:
    symbol_report_dir = REPORTS_DIR / symbol
    if not symbol_report_dir.exists():
        return None

    candidate_dirs = sorted(
        path for path in symbol_report_dir.iterdir()
        if path.is_dir() and path.name.startswith("walk_forward_")
    )
    summaries = [
        summary for summary in (summarize_walk_forward_report_dir(path) for path in candidate_dirs)
        if summary is not None
    ]
    if not summaries:
        return None

    summaries = sorted(
        summaries,
        key=lambda item: (
            float(item["overall"]["profit_factor"]),
            float(item["overall"]["net_profit"]),
            -abs(float(item["overall"]["max_drawdown"])),
        ),
        reverse=True,
    )
    report_path = symbol_report_dir / "walk_forward_comparison.md"
    lines = [
        f"# Walk-Forward Comparison: {symbol}",
        "",
        "| Mode | Cycles | Trades | PF | Net | Max DD | Sharpe | Positive Cycles | Latest 3 Net | Latest 3 Trades | Latest 3 Min PF | BUY PF/Net | SELL PF/Net |",
        "|------|--------|--------|----|-----|--------|--------|-----------------|--------------|-----------------|-----------------|------------|-------------|",
    ]
    for item in summaries:
        overall = item["overall"]
        buy = item["BUY"]
        sell = item["SELL"]
        lines.append(
            f"| {item['mode']} | {item['cycles']} | {overall['total_trades']} | "
            f"{overall['profit_factor']:.3f} | {overall['net_profit']:.4f} | "
            f"{overall['max_drawdown']:.4f} | {overall['sharpe_ratio']:.3f} | "
            f"{item['positive_cycles']}/{item['cycles']} | {item['latest_3_net']:.4f} | "
            f"{item['latest_3_trades']} | {item['latest_3_min_pf']:.3f} | "
            f"{buy['profit_factor']:.3f}/{buy['net_profit']:.4f} | "
            f"{sell['profit_factor']:.3f}/{sell['net_profit']:.4f} |"
        )
    lines.extend([
        "",
        "Ranking diurutkan terutama dari aggregate PF, lalu net profit, lalu drawdown lebih rendah.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Walk-forward comparison report written to %s", report_path)
    return report_path

def run_pipeline(
    symbol: str,
    n_trials: int,
    force_download: bool,
    train_override: int | None,
    val_override: int | None,
    resume: bool = False,
) -> None:
    pipeline_start = time.perf_counter()
    ensure_dirs([symbol])
    cfg = SYMBOLS[symbol]
    output_paths = walk_forward_output_paths(symbol, n_trials)
    output_mode = str(output_paths["mode"])
    
    # 1. MT5 Data Download
    filename = symbol_to_filename(symbol)
    raw_path = RAW_DATA_DIR / symbol / f"{filename}_m5_raw.csv"
    
    if force_download or not raw_path.exists():
        LOGGER.info("Connecting to MT5 to download fresh rates for %s...", symbol)
        mt5 = connect_mt5()
        try:
            download_symbol(mt5, symbol)
        finally:
            mt5.shutdown()
            
    # Load raw data
    raw_df = pd.read_csv(raw_path)
    raw_df["time"] = pd.to_datetime(raw_df["time"])
    (REPORTS_DIR / symbol).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([data_quality_report(raw_df)]).to_csv(REPORTS_DIR / symbol / "data_quality_raw.csv", index=False)
    
    # 2. Feature Engineering
    LOGGER.info("Running Feature Engineering...")
    features_df = add_features(raw_df, symbol)
    pd.DataFrame([data_quality_report(features_df)]).to_csv(REPORTS_DIR / symbol / "data_quality_features.csv", index=False)
    features_path = PROCESSED_DATA_DIR / symbol / f"{filename}_m5_features.csv"
    features_df.to_csv(features_path, index=False)
    
    # 3. Labeling
    LOGGER.info("Running ATR Barrier Labeling...")
    labeled_df = atr_barrier_labels(features_df, symbol)
    labeled_path = PROCESSED_DATA_DIR / symbol / f"{filename}_m5_labeled.csv"
    labeled_df.to_csv(labeled_path, index=False)
    
    # 4. Walk-Forward configuration
    wf_cfg = WALK_FORWARD_CONFIG.copy()
    wf_cfg.update(cfg.get("walk_forward", {}))
    if train_override is not None and train_override > 0:
        wf_cfg["training_window_months"] = train_override
    if val_override is not None and val_override > 0:
        wf_cfg["validation_months"] = val_override
        
    LOGGER.info("Walk-Forward Configuration: %s", json.dumps(wf_cfg, indent=2))
    
    # Generate splits
    splits = generate_walk_forward_splits(labeled_df, cfg, wf_cfg)
    LOGGER.info("Generated %d Walk-Forward splits.", len(splits))
    
    if not splits:
        LOGGER.error("No splits generated! Check if data contains enough months.")
        sys.exit(1)
        
    # Setup directories
    base_model_dir = Path(output_paths["model_dir"])
    base_report_dir = Path(output_paths["report_dir"])
    base_model_dir.mkdir(parents=True, exist_ok=True)
    base_report_dir.mkdir(parents=True, exist_ok=True)
    write_label_diagnostics(labeled_df, features_df, symbol, REPORTS_DIR / symbol)
    LOGGER.info("Walk-forward mode: %s", output_mode)
    LOGGER.info("Reports: %s", base_report_dir)
    LOGGER.info("Models: %s", base_model_dir)
    
    columns = feature_columns(labeled_df)
    lookahead = int(cfg["lookahead_candles"])
    lookahead_delta = pd.Timedelta(minutes=5 * lookahead)
    
    summary_rows = []
    all_trades = []
    skipped_cycles = 0
    
    # 5. Cycle Execution Loop
    for split in splits:
        cycle = split["cycle"]
        cycle_start = time.perf_counter()
        
        cycle_model_dir = base_model_dir / f"cycle_{cycle:03d}"
        cycle_report_dir = base_report_dir / f"cycle_{cycle:03d}"
        
        # Resume: skip completed cycles, load their artifacts for aggregation
        if resume and is_cycle_complete(cycle_model_dir, cycle_report_dir):
            LOGGER.info("CYCLE %03d/%03d — ALREADY COMPLETE, loading artifacts...", cycle, len(splits))
            try:
                summary_row, oos_trades = load_completed_cycle(cycle_model_dir, cycle_report_dir, split)
                summary_rows.append(summary_row)
                if not oos_trades.empty:
                    all_trades.append(oos_trades)
                skipped_cycles += 1
                LOGGER.info("Cycle %03d skipped in %s", cycle, format_duration(time.perf_counter() - cycle_start))
                continue
            except Exception as e:
                LOGGER.warning("Failed to load completed cycle %d artifacts: %s. Re-running.", cycle, e)
        
        # Resume: clean up partially completed cycles before re-running
        if resume:
            for d in [cycle_model_dir, cycle_report_dir]:
                if d.exists():
                    shutil.rmtree(d)
                    LOGGER.info("Cleaned up partial cycle directory: %s", d)
        
        LOGGER.info("========================================")
        LOGGER.info("RUNNING WALK-FORWARD CYCLE %03d/%03d", cycle, len(splits))
        LOGGER.info("Train: %s to %s", split["train_start"], split["train_end"])
        LOGGER.info("Val:   %s to %s", split["val_start"], split["val_end"])
        LOGGER.info("OOS:   %s to %s", split["oos_start"], split["oos_end"])
        LOGGER.info("========================================")
        
        cycle_model_dir.mkdir(parents=True, exist_ok=True)
        cycle_report_dir.mkdir(parents=True, exist_ok=True)
        
        # Split data
        train_df = labeled_df[(labeled_df["time"] >= split["train_start"]) & (labeled_df["time"] < split["train_end"])]
        val_df = labeled_df[(labeled_df["time"] >= split["val_start"]) & (labeled_df["time"] < split["val_end"])]
        oos_df = labeled_df[(labeled_df["time"] >= split["oos_start"]) & (labeled_df["time"] < split["oos_end"])]
        
        if len(train_df) < 100 or len(val_df) < 20 or len(oos_df) < 5:
            LOGGER.warning("Cycle %d skipped due to insufficient rows: train=%s val=%s oos=%s", 
                           cycle, len(train_df), len(val_df), len(oos_df))
            LOGGER.info("Cycle %03d skipped in %s", cycle, format_duration(time.perf_counter() - cycle_start))
            continue
            
        # Data Leakage Prevention: Purge boundary overlapping rows
        # Training labels cannot depend on future validation/OOS data
        train_df = train_df[train_df["time"] < (split["train_end"] - lookahead_delta)]
        val_df = val_df[val_df["time"] < (split["val_end"] - lookahead_delta)]
        
        X_train = train_df[columns]
        y_train = train_df["label"]
        X_val = val_df[columns]
        y_val = val_df["label"]
        w_train = sample_weights(y_train)
        
        # Save split meta
        meta = {
            "cycle": cycle,
            "train_start": str(split["train_start"]),
            "train_end": str(split["train_end"]),
            "val_start": str(split["val_start"]),
            "val_end": str(split["val_end"]),
            "oos_start": str(split["oos_start"]),
            "oos_end": str(split["oos_end"]),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "oos_rows": len(oos_df),
        }
        save_json(meta, cycle_model_dir / "cycle_meta.json")
        
        if not uses_separate_side_models(cfg):
            raise RuntimeError(f"{symbol}: side_training_mode must be 'separate' for walk-forward training")

        LOGGER.info("Step 6/7: Training separate BUY and SELL binary models...")
        side_models = {}
        side_thresholds = {}
        side_grid_frames = []
        for side in SIDES:
            y_side_train = binary_target(y_train, side)
            y_side_val = binary_target(y_val, side)
            LOGGER.info("%s binary target distribution train=%s val=%s", side, y_side_train.value_counts().to_dict(), y_side_val.value_counts().to_dict())
            side_params = run_binary_tuning(
                X_train,
                y_side_train,
                X_val,
                y_side_val,
                binary_sample_weights(y_side_train),
                n_trials,
            )
            if side_params:
                save_json(side_params, cycle_model_dir / f"{side.lower()}_tuned_params.json")
            side_model = fit_calibrated_binary_ensemble(X_train, y_side_train, X_val, y_side_val, side_params)
            side_models[side] = side_model
            save_model(side_model, cycle_model_dir / f"{side.lower()}_model.joblib")
            positive_proba = side_positive_proba(side_model, X_val)
            side_threshold, side_grid = optimize_side_threshold_wf(val_df, positive_proba, cfg, side)
            side_thresholds[side] = side_threshold
            save_json(side_threshold, cycle_model_dir / f"{side.lower()}_threshold.json")
            side_grid.to_csv(cycle_report_dir / f"{side.lower()}_threshold_search.csv", index=False)
            side_grid_frames.append(side_grid)

        save_json(columns, cycle_model_dir / "feature_columns.json")
        allowed_sides = [
            side for side, threshold in side_thresholds.items()
            if bool(threshold.get("eligible", False))
        ]
        thresholds = {
            "side_training_mode": "separate",
            "live_side_policy": cfg.get("live_side_policy", "gated"),
            "allowed_sides": allowed_sides,
            "buy_threshold": float(side_thresholds["BUY"]["threshold"]),
            "sell_threshold": float(side_thresholds["SELL"]["threshold"]),
            "no_trade_zone": 0.02,
            "session_mode": "configured_sessions",
            "eligible": bool(allowed_sides),
            "buy_threshold_detail": side_thresholds["BUY"],
            "sell_threshold_detail": side_thresholds["SELL"],
            "fallback_reason": ";".join(
                sorted(
                    {
                        str(item.get("fallback_reason", ""))
                        for item in side_thresholds.values()
                        if item.get("fallback_reason")
                    }
                )
            ),
        }
        save_json(thresholds, cycle_model_dir / "best_threshold.json")
        save_json(
            {
                "side_training_mode": "separate",
                "model_files": {"BUY": "buy_model.joblib", "SELL": "sell_model.joblib"},
                "threshold_files": {"BUY": "buy_threshold.json", "SELL": "sell_threshold.json"},
            },
            cycle_model_dir / "side_model_meta.json",
        )
        pd.concat(side_grid_frames, ignore_index=True).to_csv(cycle_report_dir / "threshold_search.csv", index=False)
        model = {"side_training_mode": "separate", "models": side_models}
        
        # Out-of-sample Backtest
        LOGGER.info("Step 9: Out-of-sample backtest...")
        oos_metrics, oos_trades = run_oos_backtest(model, oos_df, columns, cfg, thresholds, wf_cfg)
        
        # Save OOS artifacts
        oos_trades.to_csv(cycle_report_dir / "oos_backtest.csv", index=False)
        save_json(oos_metrics, cycle_report_dir / "oos_metrics.json")
        
        if not oos_trades.empty:
            all_trades.append(oos_trades)
            oos_equity = metric_pnl_series(oos_trades).cumsum()
            save_equity_curve_wf(oos_equity, cycle_report_dir / "equity_curve.png", title=f"OOS Equity Curve - Cycle {cycle}")
            
        LOGGER.info("Cycle %d metrics: Trades=%d PnL=%.4f PF=%.3f WR=%.2f%%", 
                    cycle, oos_metrics["total_trades"], oos_metrics["net_profit"], 
                    oos_metrics["profit_factor"], oos_metrics["winrate"]*100)
        
        summary_rows.append({
            "cycle": cycle,
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "val_start": split["val_start"],
            "val_end": split["val_end"],
            "oos_start": split["oos_start"],
            "oos_end": split["oos_end"],
            "buy_threshold": thresholds["buy_threshold"],
            "sell_threshold": thresholds["sell_threshold"],
            "no_trade_zone": thresholds["no_trade_zone"],
            "session_mode": thresholds.get("session_mode", "configured_sessions"),
            "backtest_trades_per_day": thresholds.get("backtest_trades_per_day", 0.0),
            "target_min_trades_per_day": thresholds.get("target_min_trades_per_day", 0.0),
            "target_max_trades_per_day": thresholds.get("target_max_trades_per_day", 0.0),
            "target_feasible": thresholds.get("target_feasible", False),
            "trade_frequency_gap": thresholds.get("trade_frequency_gap", 0.0),
            "oos_trades": oos_metrics["total_trades"],
            "oos_net_profit": oos_metrics["net_profit"],
            "oos_net_profit_price_lot": oos_metrics["net_profit_price_lot"],
            "oos_net_profit_money": oos_metrics["net_profit_money"],
            "oos_pnl_money_available": oos_metrics["pnl_money_available"],
            "oos_avg_pnl_per_trade": oos_metrics["avg_pnl_per_trade"],
            "oos_avg_pnl_per_trade_money": oos_metrics["avg_pnl_per_trade_money"],
            "oos_avg_pnl_per_risk": oos_metrics["avg_pnl_per_risk"],
            "oos_profit_factor": oos_metrics["profit_factor"],
            "oos_max_drawdown": oos_metrics["max_drawdown"],
            "oos_winrate": oos_metrics["winrate"],
            "buy_trades": oos_metrics["buy_trades"],
            "sell_trades": oos_metrics["sell_trades"],
            "fallback_reason": thresholds.get("fallback_reason", ""),
            **cycle_side_metrics(oos_trades),
        })
        LOGGER.info("Cycle %03d completed in %s", cycle, format_duration(time.perf_counter() - cycle_start))
        
    # 6. Post-pipeline Aggregation
    LOGGER.info("Aggregation phase: generating reports and metrics...")
    if resume and skipped_cycles > 0:
        LOGGER.info("Resumed pipeline: %d cycles skipped (already complete), %d cycles newly executed.",
                    skipped_cycles, len(splits) - skipped_cycles)
    gate_failures = walk_forward_gate_failures(summary_rows, all_trades, expected_cycles=len(splits), symbol=symbol)
    generate_reports(symbol, summary_rows, all_trades, base_report_dir, expected_cycles=len(splits), gate_failures=gate_failures)
    generate_walk_forward_comparison_report(symbol)
    allowed_sides, side_metrics = live_allowed_sides(all_trades)
    live_meta = select_live_deploy_cycle(summary_rows, allowed_sides=allowed_sides) if not gate_failures else None
    if live_meta:
        live_meta["walk_forward_mode"] = output_mode
        live_meta["artifact_subdir"] = f"walk_forward_{output_mode}"
        live_meta["side_metrics"] = side_metrics
        live_meta["artifact_hashes"] = artifact_hashes(base_model_dir / live_meta["cycle"])
        save_json(live_meta, Path(output_paths["live_meta_path"]))
        LOGGER.info("Selected walk-forward live model: %s", json.dumps(live_meta, indent=2))
    elif gate_failures:
        LOGGER.warning("Walk-forward live selection blocked; mode-specific live meta unchanged: %s", gate_failures)
    else:
        LOGGER.warning("No healthy OOS cycle available; mode-specific live meta was not updated.")
    LOGGER.info("Pipeline complete in %s! All results saved to disk.", format_duration(time.perf_counter() - pipeline_start))

def main() -> None:
    parser = argparse.ArgumentParser(description="Institutional walk-forward ML evaluation pipeline.")
    parser.add_argument("--symbol", default="USTEC_X100", help="Symbol to evaluate (default: USTEC_X100)")
    parser.add_argument("--trials", type=int, default=50, help="Optuna hyperparameter tuning trials per cycle (default: 50)")
    parser.add_argument("--force-download", action="store_true", help="Force MT5 download of fresh prices")
    parser.add_argument("--train-months", type=int, default=None, help="Override training window size in months")
    parser.add_argument("--val-months", type=int, default=None, help="Override validation window size in months")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed cycle instead of starting from scratch")
    args = parser.parse_args()
    
    run_pipeline(args.symbol, args.trials, args.force_download, args.train_months, args.val_months, resume=args.resume)

if __name__ == "__main__":
    main()
