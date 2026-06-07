import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR, SYMBOLS, WALK_FORWARD_CONFIG
from download_mt5_data import connect_mt5, download_symbol
from features import add_features
from labeling import atr_barrier_labels
from train import feature_columns
from utils import ensure_dirs, load_json, load_model, save_json, save_model, setup_logger, symbol_to_filename
from production import (
    STRICT_MAX_DATA_STALENESS_DAYS,
    STRICT_MAX_DRAWDOWN_ABS,
    STRICT_MIN_PROFIT_FACTOR,
    STRICT_MIN_SIDE_TRADES,
    artifact_hashes,
    data_quality_report,
    strict_side_gate_failures,
    threshold_has_fallback,
)
from walk_forward_backtest import run_oos_backtest, save_equity_curve_wf
from walk_forward_pipeline import (
    SIDES,
    binary_sample_weights,
    binary_target,
    fit_calibrated_binary_ensemble,
    run_binary_tuning,
    side_positive_proba,
)
from walk_forward_threshold import optimize_side_threshold_wf

LOGGER = setup_logger("weekly_retrain")


DEFAULT_SYMBOL = "USTEC_X100"
DEFAULT_TRAIN_MONTHS = 8
DEFAULT_VAL_MONTHS = 1
DEFAULT_OOS_WEEKS = 1
DEFAULT_MIN_OOS_TRADES = STRICT_MIN_SIDE_TRADES
DEFAULT_MIN_OOS_PROFIT_FACTOR = STRICT_MIN_PROFIT_FACTOR
DEFAULT_MAX_OOS_DRAWDOWN = STRICT_MAX_DRAWDOWN_ABS
DEFAULT_MAX_DATA_STALENESS_DAYS = STRICT_MAX_DATA_STALENESS_DAYS
ENSEMBLE_SEEDS = [20260605, 12345, 98765]


def effective_retrain_config(
    symbol: str,
    train_months: int | None = None,
    val_months: int | None = None,
    oos_weeks: int | None = None,
) -> dict:
    cfg = SYMBOLS[symbol]
    wf_cfg = WALK_FORWARD_CONFIG.copy()
    wf_cfg.update(cfg.get("walk_forward", {}))
    return {
        "train_months": int(train_months if train_months is not None else wf_cfg.get("training_window_months", DEFAULT_TRAIN_MONTHS)),
        "val_months": int(val_months if val_months is not None else wf_cfg.get("validation_months", DEFAULT_VAL_MONTHS)),
        "oos_weeks": int(oos_weeks if oos_weeks is not None else DEFAULT_OOS_WEEKS),
        "commission_per_lot": float(wf_cfg.get("commission_per_lot", 0.0)),
        "slippage_points": float(wf_cfg.get("slippage_points", 2.0)),
    }


def resolve_retrain_symbols(args) -> list[str]:
    if getattr(args, "all", False):
        return list(SYMBOLS.keys())
    if getattr(args, "symbols", None):
        return [symbol.upper() for symbol in args.symbols]
    return [args.symbol.upper()]


def generate_latest_retrain_split(
    df: pd.DataFrame,
    train_months: int = DEFAULT_TRAIN_MONTHS,
    val_months: int = DEFAULT_VAL_MONTHS,
    oos_weeks: int = DEFAULT_OOS_WEEKS,
) -> dict:
    df = df.sort_values("time").reset_index(drop=True)
    min_time = pd.to_datetime(df["time"].min())
    max_time = pd.to_datetime(df["time"].max())
    oos_end = max_time
    oos_start = oos_end - pd.DateOffset(weeks=oos_weeks)
    val_end = oos_start
    val_start = val_end - pd.DateOffset(months=val_months)
    train_end = val_start
    train_start = train_end - pd.DateOffset(months=train_months)
    if train_start < min_time:
        raise ValueError(
            f"Not enough data for latest retrain split: need train_start={train_start}, data starts={min_time}"
        )
    return {
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "oos_start": oos_start,
        "oos_end": oos_end,
    }


def prepare_labeled_data(symbol: str, force_download: bool = False) -> pd.DataFrame:
    ensure_dirs([symbol])
    filename = symbol_to_filename(symbol)
    raw_path = RAW_DATA_DIR / symbol / f"{filename}_m5_raw.csv"
    if force_download or not raw_path.exists():
        mt5 = connect_mt5()
        try:
            download_symbol(mt5, symbol)
        finally:
            mt5.shutdown()
    raw_df = pd.read_csv(raw_path)
    raw_df["time"] = pd.to_datetime(raw_df["time"])
    report_dir = REPORTS_DIR / symbol
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([data_quality_report(raw_df)]).to_csv(report_dir / "data_quality_raw.csv", index=False)
    features_df = add_features(raw_df, symbol)
    pd.DataFrame([data_quality_report(features_df)]).to_csv(report_dir / "data_quality_features.csv", index=False)
    features_path = PROCESSED_DATA_DIR / symbol / f"{filename}_m5_features.csv"
    features_df.to_csv(features_path, index=False)
    labeled_df = atr_barrier_labels(features_df, symbol)
    labeled_path = PROCESSED_DATA_DIR / symbol / f"{filename}_m5_labeled.csv"
    labeled_df.to_csv(labeled_path, index=False)
    return labeled_df


def validate_deployment_artifacts(deployment_dir: Path, expected_columns: list[str]) -> list[str]:
    failures = []
    required = [
        "buy_model.joblib",
        "sell_model.joblib",
        "buy_threshold.json",
        "sell_threshold.json",
        "side_model_meta.json",
        "feature_columns.json",
        "best_threshold.json",
        "retrain_meta.json",
        "oos_metrics.json",
        "oos_backtest.csv",
    ]
    for name in required:
        if not (deployment_dir / name).exists():
            failures.append(f"missing {name}")
    for name in ["buy_model.joblib", "sell_model.joblib"]:
        if (deployment_dir / name).exists():
            try:
                load_model(deployment_dir / name)
            except Exception as exc:
                failures.append(f"{name} artifact cannot be loaded: {exc}")
    columns = load_json(deployment_dir / "feature_columns.json", default=None)
    if columns != expected_columns:
        failures.append("feature column metadata mismatch")
    thresholds = load_json(deployment_dir / "best_threshold.json", default={}) or {}
    if thresholds.get("eligible") is not True:
        failures.append("threshold is not eligible")
    if threshold_has_fallback(thresholds):
        failures.append("threshold uses fallback search result")
    return failures


def deployment_gate_failures(
    deployment_dir: Path,
    expected_columns: list[str],
    latest_time,
    current_time=None,
    min_oos_trades: int = DEFAULT_MIN_OOS_TRADES,
    min_profit_factor: float = DEFAULT_MIN_OOS_PROFIT_FACTOR,
    max_drawdown_abs: float = DEFAULT_MAX_OOS_DRAWDOWN,
    max_staleness_days: int = DEFAULT_MAX_DATA_STALENESS_DAYS,
) -> list[str]:
    failures = validate_deployment_artifacts(deployment_dir, expected_columns)
    metrics = load_json(deployment_dir / "oos_metrics.json", default={}) or {}
    thresholds = load_json(deployment_dir / "best_threshold.json", default={}) or {}
    allowed_sides = thresholds.get("allowed_sides", [])
    side_metrics = load_json(deployment_dir / "side_metrics.json", default={}) or {}
    if int(metrics.get("total_trades", 0)) < min_oos_trades:
        failures.append("OOS trade count below minimum")
    if float(metrics.get("net_profit", 0.0)) <= 0:
        failures.append("OOS net profit is not positive")
    if float(metrics.get("profit_factor", 0.0)) < min_profit_factor:
        failures.append("OOS profit factor below minimum")
    if abs(float(metrics.get("max_drawdown", 0.0))) > max_drawdown_abs:
        failures.append("OOS drawdown above maximum")
    failures.extend(
        strict_side_gate_failures(
            side_metrics,
            allowed_sides,
            min_trades=min_oos_trades,
            min_profit_factor=min_profit_factor,
            max_drawdown_abs=max_drawdown_abs,
        )
    )
    now = pd.Timestamp(current_time if current_time is not None else datetime.now())
    latest = pd.Timestamp(latest_time)
    if now.tzinfo is not None:
        now = now.tz_localize(None)
    if latest.tzinfo is not None:
        latest = latest.tz_localize(None)
    if now - latest > pd.Timedelta(days=max_staleness_days):
        failures.append("latest data is stale")
    return failures


def write_live_meta(
    symbol: str,
    deployment_name: str,
    split: dict,
    metrics: dict,
    thresholds: dict,
    deployment_dir: Path,
    side_metrics: dict,
    gate_failures: list[str],
) -> None:
    meta = {
        "source": "deployment",
        "gate_status": "passed" if not gate_failures else "failed",
        "gate_failures": gate_failures,
        "deployment": deployment_name,
        "artifact_subdir": f"deployments/{deployment_name}",
        "selection": "latest_weekly_retrain",
        "symbol": symbol,
        "train_start": str(split["train_start"]),
        "train_end": str(split["train_end"]),
        "val_start": str(split["val_start"]),
        "val_end": str(split["val_end"]),
        "oos_start": str(split["oos_start"]),
        "oos_end": str(split["oos_end"]),
        "oos_profit_factor": float(metrics.get("profit_factor", 0.0)),
        "oos_max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "oos_net_profit": float(metrics.get("net_profit", 0.0)),
        "oos_trades": int(metrics.get("total_trades", 0)),
        "buy_threshold": float(thresholds.get("buy_threshold", 0.0)),
        "sell_threshold": float(thresholds.get("sell_threshold", 0.0)),
        "no_trade_zone": float(thresholds.get("no_trade_zone", 0.0)),
        "allowed_sides": thresholds.get("allowed_sides", []),
        "side_metrics": side_metrics,
        "artifact_hashes": artifact_hashes(deployment_dir),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(meta, MODELS_DIR / symbol / "live_model_meta.json")


def run_weekly_retrain(
    symbol: str = DEFAULT_SYMBOL,
    n_trials: int = 50,
    train_months: int | None = None,
    val_months: int | None = None,
    oos_weeks: int | None = None,
    force_download: bool = False,
    deploy: bool = True,
    deployment_name: str | None = None,
    min_oos_trades: int = DEFAULT_MIN_OOS_TRADES,
    min_profit_factor: float = DEFAULT_MIN_OOS_PROFIT_FACTOR,
    max_drawdown_abs: float = DEFAULT_MAX_OOS_DRAWDOWN,
    max_staleness_days: int = DEFAULT_MAX_DATA_STALENESS_DAYS,
) -> dict:
    symbol = symbol.upper()
    ensure_dirs([symbol])
    cfg = SYMBOLS[symbol]
    retrain_cfg = effective_retrain_config(symbol, train_months, val_months, oos_weeks)
    labeled_df = prepare_labeled_data(symbol, force_download=force_download)
    split = generate_latest_retrain_split(
        labeled_df,
        retrain_cfg["train_months"],
        retrain_cfg["val_months"],
        retrain_cfg["oos_weeks"],
    )
    columns = feature_columns(labeled_df)
    lookahead_delta = pd.Timedelta(minutes=5 * int(cfg["lookahead_candles"]))

    train_df = labeled_df[(labeled_df["time"] >= split["train_start"]) & (labeled_df["time"] < split["train_end"])]
    val_df = labeled_df[(labeled_df["time"] >= split["val_start"]) & (labeled_df["time"] < split["val_end"])]
    oos_df = labeled_df[(labeled_df["time"] >= split["oos_start"]) & (labeled_df["time"] < split["oos_end"])]
    train_df = train_df[train_df["time"] < (split["train_end"] - lookahead_delta)]
    val_df = val_df[val_df["time"] < (split["val_end"] - lookahead_delta)]
    if len(train_df) < 100 or len(val_df) < 20 or len(oos_df) < 5:
        raise RuntimeError(
            f"Insufficient rows for weekly retrain: train={len(train_df)} val={len(val_df)} oos={len(oos_df)}"
        )

    X_train = train_df[columns]
    X_val = val_df[columns]
    LOGGER.info("Running latest weekly retrain for %s", symbol)
    LOGGER.info("Train: %s to %s", split["train_start"], split["train_end"])
    LOGGER.info("Val:   %s to %s", split["val_start"], split["val_end"])
    LOGGER.info("OOS:   %s to %s", split["oos_start"], split["oos_end"])

    side_models = {}
    side_thresholds = {}
    side_grid_frames = []
    for side in SIDES:
        y_side_train = binary_target(train_df["label"], side)
        y_side_val = binary_target(val_df["label"], side)
        best_params = run_binary_tuning(
            X_train,
            y_side_train,
            X_val,
            y_side_val,
            binary_sample_weights(y_side_train),
            n_trials,
        )
        side_model = fit_calibrated_binary_ensemble(X_train, y_side_train, X_val, y_side_val, best_params)
        side_models[side] = side_model
        positive_proba = side_positive_proba(side_model, X_val)
        side_threshold, side_grid = optimize_side_threshold_wf(val_df, positive_proba, cfg, side)
        side_thresholds[side] = side_threshold
        side_grid_frames.append(side_grid)
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
        "eligible": bool(allowed_sides),
        "buy_threshold_detail": side_thresholds["BUY"],
        "sell_threshold_detail": side_thresholds["SELL"],
    }
    threshold_grid = pd.concat(side_grid_frames, ignore_index=True)
    model = {"side_training_mode": "separate", "models": side_models}
    wf_cfg = {
        "commission_per_lot": retrain_cfg["commission_per_lot"],
        "slippage_points": retrain_cfg["slippage_points"],
    }
    oos_metrics, oos_trades = run_oos_backtest(model, oos_df, columns, cfg, thresholds, wf_cfg)
    side_metrics = {}
    for side in SIDES:
        side_trades = oos_trades[oos_trades["side"] == side].reset_index(drop=True) if not oos_trades.empty else pd.DataFrame()
        side_metrics[side] = {
            "total_trades": int(len(side_trades)),
            "net_profit": float(side_trades["pnl"].sum()) if not side_trades.empty else 0.0,
            "profit_factor": 0.0,
            "avg_pnl_per_trade": float(side_trades["pnl"].mean()) if not side_trades.empty else 0.0,
            "max_drawdown": float((side_trades["pnl"].cumsum() - side_trades["pnl"].cumsum().cummax()).min()) if not side_trades.empty else 0.0,
            "pnl_money_available": bool(not side_trades.empty and "pnl_money" in side_trades and pd.to_numeric(side_trades["pnl_money"], errors="coerce").notna().any()),
        }
        if not side_trades.empty:
            pnl = pd.to_numeric(side_trades["pnl"], errors="coerce").fillna(0.0)
            gross_profit = pnl[pnl > 0].sum()
            gross_loss = pnl[pnl < 0].sum()
            side_metrics[side]["profit_factor"] = float(gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf") if gross_profit > 0 else 0.0

    deployment_name = deployment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    deployment_dir = MODELS_DIR / symbol / "deployments" / deployment_name
    report_dir = REPORTS_DIR / symbol / "deployments" / deployment_name
    deployment_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    save_model(side_models["BUY"], deployment_dir / "buy_model.joblib")
    save_model(side_models["SELL"], deployment_dir / "sell_model.joblib")
    save_json(side_thresholds["BUY"], deployment_dir / "buy_threshold.json")
    save_json(side_thresholds["SELL"], deployment_dir / "sell_threshold.json")
    save_json(
        {
            "side_training_mode": "separate",
            "model_files": {"BUY": "buy_model.joblib", "SELL": "sell_model.joblib"},
            "threshold_files": {"BUY": "buy_threshold.json", "SELL": "sell_threshold.json"},
        },
        deployment_dir / "side_model_meta.json",
    )
    save_json(columns, deployment_dir / "feature_columns.json")
    save_json(thresholds, deployment_dir / "best_threshold.json")
    save_json(oos_metrics, deployment_dir / "oos_metrics.json")
    save_json(side_metrics, deployment_dir / "side_metrics.json")
    threshold_grid.to_csv(report_dir / "threshold_search.csv", index=False)
    oos_trades.to_csv(deployment_dir / "oos_backtest.csv", index=False)
    oos_trades.to_csv(report_dir / "oos_backtest.csv", index=False)
    if not oos_trades.empty:
        save_equity_curve_wf(oos_trades["pnl"].cumsum(), report_dir / "oos_equity_curve.png")
    retrain_meta = {
        "symbol": symbol,
        "train_months": retrain_cfg["train_months"],
        "val_months": retrain_cfg["val_months"],
        "oos_weeks": retrain_cfg["oos_weeks"],
        "n_trials": n_trials,
        "commission_per_lot": retrain_cfg["commission_per_lot"],
        "slippage_points": retrain_cfg["slippage_points"],
        "train_start": str(split["train_start"]),
        "train_end": str(split["train_end"]),
        "val_start": str(split["val_start"]),
        "val_end": str(split["val_end"]),
        "oos_start": str(split["oos_start"]),
        "oos_end": str(split["oos_end"]),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "oos_rows": len(oos_df),
        "data_start": str(labeled_df["time"].min()),
        "data_end": str(labeled_df["time"].max()),
        "deployment": deployment_name,
    }
    save_json(retrain_meta, deployment_dir / "retrain_meta.json")
    save_json(retrain_meta, report_dir / "retrain_meta.json")
    save_json(oos_metrics, report_dir / "oos_metrics.json")
    save_json(side_metrics, report_dir / "side_metrics.json")

    failures = deployment_gate_failures(
        deployment_dir,
        columns,
        labeled_df["time"].max(),
        min_oos_trades=min_oos_trades,
        min_profit_factor=min_profit_factor,
        max_drawdown_abs=max_drawdown_abs,
        max_staleness_days=max_staleness_days,
    )
    if deploy and not failures:
        write_live_meta(symbol, deployment_name, split, oos_metrics, thresholds, deployment_dir, side_metrics, failures)
        LOGGER.info("Deployment gate passed; live_model_meta.json updated to %s", deployment_name)
    elif failures:
        LOGGER.warning("Deployment gate failed; live_model_meta.json unchanged: %s", failures)
    return {
        "deployment": deployment_name,
        "deployment_dir": str(deployment_dir),
        "report_dir": str(report_dir),
        "split": split,
        "oos_metrics": oos_metrics,
        "thresholds": thresholds,
        "gate_failures": failures,
        "live_meta_updated": bool(deploy and not failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run latest-only weekly retrain and optional live deployment.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--symbols", nargs="+", help="Run weekly retrain for multiple symbols")
    parser.add_argument("--all", action="store_true", help="Run weekly retrain for all configured symbols")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--train-months", type=int, default=None, help="Override training window size in months")
    parser.add_argument("--val-months", type=int, default=None, help="Override validation window size in months")
    parser.add_argument("--oos-weeks", type=int, default=None, help="Override OOS gate window size in weeks")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-deploy", action="store_true", help="Write artifacts but do not update live_model_meta.json.")
    parser.add_argument("--min-oos-trades", type=int, default=DEFAULT_MIN_OOS_TRADES)
    parser.add_argument("--min-profit-factor", type=float, default=DEFAULT_MIN_OOS_PROFIT_FACTOR)
    parser.add_argument("--max-drawdown-abs", type=float, default=DEFAULT_MAX_OOS_DRAWDOWN)
    parser.add_argument("--max-staleness-days", type=int, default=DEFAULT_MAX_DATA_STALENESS_DAYS)
    args = parser.parse_args()
    results = []
    for symbol in resolve_retrain_symbols(args):
        result = run_weekly_retrain(
            symbol=symbol,
            n_trials=args.trials,
            train_months=args.train_months,
            val_months=args.val_months,
            oos_weeks=args.oos_weeks,
            force_download=args.force_download,
            deploy=not args.no_deploy,
            min_oos_trades=args.min_oos_trades,
            min_profit_factor=args.min_profit_factor,
            max_drawdown_abs=args.max_drawdown_abs,
            max_staleness_days=args.max_staleness_days,
        )
        results.append(result)
    if any(result["gate_failures"] for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
