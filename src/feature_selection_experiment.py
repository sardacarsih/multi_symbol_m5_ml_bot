import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from calibration_utils import calibrate_prefit_classifier
from config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, SYMBOLS, WALK_FORWARD_CONFIG
from symbols import parse_symbol_args, resolve_symbols
from train import build_model, feature_columns, sample_weights
from utils import ensure_dirs, save_json, save_model, setup_logger, symbol_to_filename
from walk_forward_backtest import compute_oos_metrics, run_oos_backtest, save_equity_curve_wf
from walk_forward_pipeline import (
    generate_walk_forward_splits,
    is_cycle_complete,
    load_completed_cycle,
    normalize_trade_datetimes,
    run_tuning,
)
from walk_forward_threshold import optimize_thresholds_wf

LOGGER = setup_logger("feature_selection_experiment")

CORE30_FEATURES = [
    "hour",
    "is_london_session",
    "is_newyork_session",
    "is_london_newyork_overlap",
    "is_london_killzone",
    "is_newyork_killzone",
    "spread_to_atr",
    "session_spread_stress",
    "atr_ratio",
    "atr_5_to_14",
    "atr_14_to_28",
    "vol_regime",
    "vol_expansion",
    "vol_compression_release",
    "rolling_std_24",
    "return_3",
    "return_6",
    "rsi_7",
    "macd_hist",
    "macd_hist_change_3",
    "ema_alignment_score",
    "trend_direction_strength",
    "close_position_in_range_24",
    "range_width_48_atr",
    "prev_day_high_dist_atr",
    "prev_day_low_dist_atr",
    "intraday_range_adr",
    "market_structure_state",
    "liquidity_sweep_high_24",
    "volume_ratio_12",
]

CORE20_FEATURES = [
    "hour",
    "is_london_session",
    "is_newyork_session",
    "is_london_newyork_overlap",
    "is_london_killzone",
    "spread_to_atr",
    "atr_ratio",
    "atr_5_to_14",
    "atr_14_to_28",
    "vol_regime",
    "vol_expansion",
    "vol_compression_release",
    "rolling_std_24",
    "return_3",
    "return_6",
    "macd_hist",
    "ema_alignment_score",
    "trend_direction_strength",
    "close_position_in_range_24",
    "volume_ratio_12",
]


def feature_set_label(feature_set: str, top_n: int | None = None) -> str:
    if feature_set == "importance_top_n":
        return f"importance_top_{top_n}"
    return feature_set


def experiment_report_dir(symbol: str, feature_set: str, top_n: int | None = None, output_dir: str | None = None) -> Path:
    if output_dir:
        return Path(output_dir)
    return REPORTS_DIR / symbol / "feature_selection" / feature_set_label(feature_set, top_n)


def experiment_model_dir(symbol: str, feature_set: str, top_n: int | None = None) -> Path:
    return MODELS_DIR / symbol / "feature_selection" / feature_set_label(feature_set, top_n)


def filter_valid_features(requested: list[str], valid_columns: list[str], df: pd.DataFrame) -> tuple[list[str], list[str]]:
    valid_set = set(valid_columns)
    selected = []
    skipped = []
    for feature in requested:
        if feature in valid_set and feature in df.columns:
            if feature not in selected:
                selected.append(feature)
        else:
            skipped.append(feature)
    return selected, skipped


def load_importance_features(symbol: str, top_n: int, valid_columns: list[str]) -> list[str]:
    path = REPORTS_DIR / symbol / "feature_importance.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature importance file not found: {path}")

    frame = pd.read_csv(path)
    if "feature" not in frame.columns or "importance" not in frame.columns:
        raise ValueError(f"Invalid feature importance file: {path}")

    valid_set = set(valid_columns)
    frame = frame[frame["feature"].isin(valid_set)].copy()
    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").fillna(0.0)
    frame = frame.sort_values("importance", ascending=False)
    return frame["feature"].head(top_n).tolist()


def select_features(
    feature_set: str,
    df: pd.DataFrame,
    valid_columns: list[str],
    symbol: str,
    top_n: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    if feature_set == "core20":
        requested = CORE20_FEATURES
    elif feature_set == "core30":
        requested = CORE30_FEATURES
    elif feature_set == "robust70":
        requested = valid_columns
    elif feature_set == "importance_top_n":
        if top_n is None or top_n <= 0:
            raise ValueError("--top-n must be a positive integer for importance_top_n")
        requested = load_importance_features(symbol, top_n, valid_columns)
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    selected, skipped = filter_valid_features(requested, valid_columns, df)
    return selected, skipped, list(requested)


def load_labeled_data(symbol: str) -> pd.DataFrame:
    filename = symbol_to_filename(symbol)
    path = PROCESSED_DATA_DIR / symbol / f"{filename}_m5_labeled.csv"
    if not path.exists():
        raise FileNotFoundError(f"Labeled data not found: {path}")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def train_calibrated_ensemble(X_train, y_train, X_val, y_val, w_train, best_params: dict, symbol: str):
    seeds = [20260605, 12345, 98765]
    ensemble = []
    for seed in seeds:
        if best_params:
            params = best_params.copy()
            params.update(
                {
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "eval_metric": "mlogloss",
                    "random_state": seed,
                    "n_jobs": -1,
                }
            )
            estimator = XGBClassifier(**params)
        else:
            estimator = build_model(symbol)
            estimator.set_params(random_state=seed)

        try:
            estimator.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=50,
            )
        except Exception:
            estimator.fit(X_train, y_train, sample_weight=w_train)

        ensemble.append(calibrate_prefit_classifier(estimator, X_val, y_val))
    return ensemble


def summary_row_from_cycle(cycle: int, split: dict, thresholds: dict, oos_metrics: dict) -> dict:
    return {
        "cycle": cycle,
        "train_start": split["train_start"],
        "train_end": split["train_end"],
        "val_start": split["val_start"],
        "val_end": split["val_end"],
        "oos_start": split["oos_start"],
        "oos_end": split["oos_end"],
        "buy_threshold": thresholds.get("buy_threshold", 0.0),
        "sell_threshold": thresholds.get("sell_threshold", 0.0),
        "no_trade_zone": thresholds.get("no_trade_zone", 0.0),
        "session_mode": thresholds.get("session_mode", "configured_sessions"),
        "backtest_trades_per_day": thresholds.get("backtest_trades_per_day", 0.0),
        "target_min_trades_per_day": thresholds.get("target_min_trades_per_day", 0.0),
        "target_max_trades_per_day": thresholds.get("target_max_trades_per_day", 0.0),
        "target_feasible": thresholds.get("target_feasible", False),
        "trade_frequency_gap": thresholds.get("trade_frequency_gap", 0.0),
        "oos_trades": oos_metrics["total_trades"],
        "oos_net_profit": oos_metrics["net_profit"],
        "oos_profit_factor": oos_metrics["profit_factor"],
        "oos_max_drawdown": oos_metrics["max_drawdown"],
        "oos_winrate": oos_metrics["winrate"],
        "buy_trades": oos_metrics["buy_trades"],
        "sell_trades": oos_metrics["sell_trades"],
        "fallback_reason": thresholds.get("fallback_reason", ""),
    }


def aggregate_trades(all_trades: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in all_trades if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    trades = pd.concat(non_empty, ignore_index=True)
    return normalize_trade_datetimes(trades).sort_values("exit_time").reset_index(drop=True)


def load_baseline_metrics(symbol: str) -> dict | None:
    metrics_path = REPORTS_DIR / symbol / "walk_forward" / "aggregated_oos_metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    trades_path = REPORTS_DIR / symbol / "walk_forward" / "aggregated_oos_trades.csv"
    if trades_path.exists():
        trades = pd.read_csv(trades_path)
        trades = normalize_trade_datetimes(trades)
        if not trades.empty:
            return compute_oos_metrics(trades, trades["entry_time"].min(), trades["exit_time"].max())
    return None


def recommendation_for_metrics(reduced: dict, baseline: dict | None, min_trades: int = 30) -> str:
    if reduced.get("total_trades", 0) < min_trades or reduced.get("net_profit", 0.0) <= 0:
        return "KEEP_BASELINE"
    if baseline is None:
        return "KEEP_BASELINE"

    reduced_pf = float(reduced.get("profit_factor", 0.0))
    baseline_pf = float(baseline.get("profit_factor", 0.0))
    reduced_dd = abs(float(reduced.get("max_drawdown", 0.0)))
    baseline_dd = abs(float(baseline.get("max_drawdown", 0.0)))
    baseline_trades = int(baseline.get("total_trades", 0))

    if (
        reduced_pf >= baseline_pf
        and reduced_dd <= baseline_dd
        and int(reduced.get("total_trades", 0)) >= min_trades
        and int(reduced.get("total_trades", 0)) >= max(1, int(baseline_trades * 0.5))
    ):
        return "PROMOTE_REDUCED_FEATURES"
    return "KEEP_BASELINE"


def metric_value(metrics: dict | None, key: str):
    if metrics is None:
        return "n/a"
    value = metrics.get(key, "n/a")
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def write_final_report(
    symbol: str,
    feature_set_name: str,
    selected_features: list[str],
    skipped_features: list[str],
    baseline_count: int,
    summary_rows: list[dict],
    aggregate_metrics: dict,
    baseline_metrics: dict | None,
    recommendation: str,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(report_dir / "feature_selection_summary.csv", index=False)
    save_json(aggregate_metrics, report_dir / "aggregated_oos_metrics.json")

    content = f"""# Feature Selection Experiment: {symbol} / {feature_set_name}

## Executive Summary

| Item | Value |
|------|-------|
| Baseline Feature Count | {baseline_count} |
| Selected Feature Count | {len(selected_features)} |
| Completed Cycles | {len(summary_rows)} |
| Recommendation | {recommendation} |

## OOS Metrics

| Metric | Baseline | Reduced |
|--------|----------|---------|
| Total Trades | {metric_value(baseline_metrics, "total_trades")} | {aggregate_metrics["total_trades"]} |
| Net Profit | {metric_value(baseline_metrics, "net_profit")} | {aggregate_metrics["net_profit"]:.4f} |
| Profit Factor | {metric_value(baseline_metrics, "profit_factor")} | {aggregate_metrics["profit_factor"]:.4f} |
| Max Drawdown | {metric_value(baseline_metrics, "max_drawdown")} | {aggregate_metrics["max_drawdown"]:.4f} |
| Winrate | {metric_value(baseline_metrics, "winrate")} | {aggregate_metrics["winrate"]:.2%} |
| BUY Trades | {metric_value(baseline_metrics, "buy_trades")} | {aggregate_metrics["buy_trades"]} |
| SELL Trades | {metric_value(baseline_metrics, "sell_trades")} | {aggregate_metrics["sell_trades"]} |

## Selected Features

{chr(10).join(f"- `{feature}`" for feature in selected_features)}

## Missing Or Skipped Features

{chr(10).join(f"- `{feature}`" for feature in skipped_features) if skipped_features else "- none"}

## Cycle Breakdown

| Cycle | OOS Period | Trades | Net Profit | Profit Factor | Winrate | BUY | SELL |
|-------|------------|--------|------------|---------------|---------|-----|------|
"""
    for row in summary_rows:
        content += (
            f"| {row['cycle']} | {row['oos_start'].strftime('%Y-%m-%d')} - "
            f"{row['oos_end'].strftime('%Y-%m-%d')} | {row['oos_trades']} | "
            f"{row['oos_net_profit']:.4f} | {row['oos_profit_factor']:.4f} | "
            f"{row['oos_winrate']:.2%} | {row['buy_trades']} | {row['sell_trades']} |\n"
        )

    with (report_dir / "feature_selection_summary.md").open("w", encoding="utf-8") as file:
        file.write(content)


def run_feature_selection_experiment(
    symbol: str,
    feature_set: str,
    top_n: int | None,
    n_trials: int,
    resume: bool,
    output_dir: str | None = None,
    max_cycles: int | None = None,
) -> dict:
    ensure_dirs([symbol])
    cfg = SYMBOLS[symbol]
    labeled_df = load_labeled_data(symbol)
    baseline_columns = feature_columns(labeled_df)
    selected_columns, skipped_features, _requested = select_features(feature_set, labeled_df, baseline_columns, symbol, top_n)
    if not selected_columns:
        raise ValueError(f"No valid features selected for {symbol} / {feature_set}")

    wf_cfg = WALK_FORWARD_CONFIG.copy()
    wf_cfg.update(cfg.get("walk_forward", {}))
    splits = generate_walk_forward_splits(labeled_df, cfg, wf_cfg)
    if max_cycles is not None:
        splits = splits[:max_cycles]
    if not splits:
        raise RuntimeError(f"No walk-forward splits generated for {symbol}")

    feature_set_name = feature_set_label(feature_set, top_n)
    base_model_dir = experiment_model_dir(symbol, feature_set, top_n)
    base_report_dir = experiment_report_dir(symbol, feature_set, top_n, output_dir)
    base_model_dir.mkdir(parents=True, exist_ok=True)
    base_report_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("%s %s: running %d cycles with %d selected features", symbol, feature_set_name, len(splits), len(selected_columns))
    lookahead_delta = pd.Timedelta(minutes=5 * int(cfg["lookahead_candles"]))
    summary_rows = []
    all_trades = []

    for split in splits:
        cycle = split["cycle"]
        cycle_model_dir = base_model_dir / f"cycle_{cycle:03d}"
        cycle_report_dir = base_report_dir / f"cycle_{cycle:03d}"

        if resume and is_cycle_complete(cycle_model_dir, cycle_report_dir):
            LOGGER.info("%s %s cycle %03d already complete; loading artifacts", symbol, feature_set_name, cycle)
            summary_row, oos_trades = load_completed_cycle(cycle_model_dir, cycle_report_dir, split)
            summary_rows.append(summary_row)
            if not oos_trades.empty:
                all_trades.append(oos_trades)
            continue

        if resume:
            for path in [cycle_model_dir, cycle_report_dir]:
                if path.exists():
                    shutil.rmtree(path)

        cycle_model_dir.mkdir(parents=True, exist_ok=True)
        cycle_report_dir.mkdir(parents=True, exist_ok=True)

        train_df = labeled_df[(labeled_df["time"] >= split["train_start"]) & (labeled_df["time"] < split["train_end"])]
        val_df = labeled_df[(labeled_df["time"] >= split["val_start"]) & (labeled_df["time"] < split["val_end"])]
        oos_df = labeled_df[(labeled_df["time"] >= split["oos_start"]) & (labeled_df["time"] < split["oos_end"])]
        if len(train_df) < 100 or len(val_df) < 20 or len(oos_df) < 5:
            LOGGER.warning("Skipping cycle %03d due to insufficient rows", cycle)
            continue

        train_df = train_df[train_df["time"] < (split["train_end"] - lookahead_delta)]
        val_df = val_df[val_df["time"] < (split["val_end"] - lookahead_delta)]
        X_train = train_df[selected_columns]
        y_train = train_df["label"]
        X_val = val_df[selected_columns]
        y_val = val_df["label"]
        w_train = sample_weights(y_train)

        save_json(
            {
                "cycle": cycle,
                "feature_set": feature_set_name,
                "feature_count": len(selected_columns),
                "train_start": str(split["train_start"]),
                "train_end": str(split["train_end"]),
                "val_start": str(split["val_start"]),
                "val_end": str(split["val_end"]),
                "oos_start": str(split["oos_start"]),
                "oos_end": str(split["oos_end"]),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "oos_rows": len(oos_df),
            },
            cycle_model_dir / "cycle_meta.json",
        )

        best_params = run_tuning(X_train, y_train, X_val, y_val, w_train, n_trials)
        if best_params:
            save_json(best_params, cycle_model_dir / "tuned_params.json")
        model = train_calibrated_ensemble(X_train, y_train, X_val, y_val, w_train, best_params, symbol)
        save_model(model, cycle_model_dir / "model.joblib")
        save_json(selected_columns, cycle_model_dir / "feature_columns.json")

        val_proba = np.mean([member.predict_proba(X_val) for member in model], axis=0)
        thresholds, grid_results = optimize_thresholds_wf(val_df, val_proba, cfg)
        save_json(thresholds, cycle_model_dir / "best_threshold.json")
        grid_results.to_csv(cycle_report_dir / "threshold_search.csv", index=False)

        oos_metrics, oos_trades = run_oos_backtest(model, oos_df, selected_columns, cfg, thresholds, wf_cfg)
        oos_trades.to_csv(cycle_report_dir / "oos_backtest.csv", index=False)
        save_json(oos_metrics, cycle_report_dir / "oos_metrics.json")
        if not oos_trades.empty:
            all_trades.append(oos_trades)
            save_equity_curve_wf(oos_trades["pnl"].cumsum(), cycle_report_dir / "equity_curve.png", title=f"{symbol} {feature_set_name} Cycle {cycle}")

        summary_rows.append(summary_row_from_cycle(cycle, split, thresholds, oos_metrics))
        LOGGER.info(
            "%s %s cycle %03d: trades=%d pnl=%.4f pf=%.3f",
            symbol,
            feature_set_name,
            cycle,
            oos_metrics["total_trades"],
            oos_metrics["net_profit"],
            oos_metrics["profit_factor"],
        )

    aggregate = aggregate_trades(all_trades)
    if aggregate.empty:
        aggregate_metrics = compute_oos_metrics(pd.DataFrame(), None, None)
    else:
        aggregate.to_csv(base_report_dir / "aggregated_oos_trades.csv", index=False)
        save_equity_curve_wf(aggregate["pnl"].cumsum(), base_report_dir / "aggregated_oos_equity.png", title=f"{symbol} {feature_set_name} Aggregated OOS")
        aggregate_metrics = compute_oos_metrics(aggregate, aggregate["entry_time"].min(), aggregate["exit_time"].max())

    baseline_metrics = load_baseline_metrics(symbol)
    recommendation = recommendation_for_metrics(aggregate_metrics, baseline_metrics)
    write_final_report(
        symbol,
        feature_set_name,
        selected_columns,
        skipped_features,
        len(baseline_columns),
        summary_rows,
        aggregate_metrics,
        baseline_metrics,
        recommendation,
        base_report_dir,
    )

    LOGGER.info("%s %s complete. Recommendation: %s", symbol, feature_set_name, recommendation)
    return {
        "symbol": symbol,
        "feature_set": feature_set_name,
        "report_dir": str(base_report_dir),
        "model_dir": str(base_model_dir),
        "metrics": aggregate_metrics,
        "recommendation": recommendation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reduced-feature walk-forward experiments.")
    parse_symbol_args(parser)
    parser.add_argument("--feature-set", choices=["core20", "core30", "robust70", "importance_top_n"], default="core20")
    parser.add_argument("--top-n", type=int, default=20, help="Number of importance-ranked features for importance_top_n")
    parser.add_argument("--trials", type=int, default=0, help="Optuna trials per cycle. 0 uses baseline parameters.")
    parser.add_argument("--resume", action="store_true", help="Load completed cycles and rerun partial cycles.")
    parser.add_argument("--output-dir", default=None, help="Override report output directory for a single-symbol run.")
    parser.add_argument("--max-cycles", type=int, default=None, help="Limit cycles for smoke tests.")
    return parser.parse_args()


def resolve_experiment_symbols(args) -> list[str]:
    if getattr(args, "symbol", None) and args.symbol.upper() == "ALL":
        return list(SYMBOLS.keys())
    return resolve_symbols(args)


def main() -> None:
    args = parse_args()
    symbols = resolve_experiment_symbols(args)
    if len(symbols) > 1 and args.output_dir:
        raise ValueError("--output-dir can only be used with a single symbol")

    results = []
    for symbol in symbols:
        results.append(
            run_feature_selection_experiment(
                symbol=symbol,
                feature_set=args.feature_set,
                top_n=args.top_n,
                n_trials=args.trials,
                resume=args.resume,
                output_dir=args.output_dir,
                max_cycles=args.max_cycles,
            )
        )

    LOGGER.info("Experiment results: %s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
