import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, SYMBOLS, WALK_FORWARD_CONFIG
from feature_selection_experiment import (
    aggregate_trades,
    select_features,
    summary_row_from_cycle,
    train_calibrated_ensemble,
)
from labeling import LABEL_NAMES, _barrier_outcome_frame
from symbols import parse_symbol_args, resolve_symbols
from train import feature_columns, sample_weights
from utils import ensure_dirs, save_json, save_model, setup_logger, symbol_to_filename
from walk_forward_backtest import compute_oos_metrics, run_oos_backtest, save_equity_curve_wf
from walk_forward_pipeline import generate_walk_forward_splits, load_completed_cycle, run_tuning
from walk_forward_threshold import optimize_thresholds_wf

LOGGER = setup_logger("label_selection_experiment")


@dataclass(frozen=True)
class LabelVariant:
    name: str
    lookahead: int
    tp_atr_mult: float
    sl_atr_mult: float
    spread_adjusted: bool = False


DEFAULT_VARIANTS = [
    LabelVariant("current", 8, 1.2, 1.2, False),
    LabelVariant("spread_current", 8, 1.2, 1.2, True),
    LabelVariant("fast", 3, 1.0, 1.0, True),
    LabelVariant("balanced", 6, 1.2, 1.0, True),
    LabelVariant("selective", 9, 1.5, 1.0, True),
    LabelVariant("trend", 12, 1.6, 1.1, True),
]


def load_feature_data(symbol: str) -> pd.DataFrame:
    filename = symbol_to_filename(symbol)
    path = PROCESSED_DATA_DIR / symbol / f"{filename}_m5_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature data not found: {path}. Run features.py first.")
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.sort_values("time").reset_index(drop=True)


def label_variant_frame(features: pd.DataFrame, symbol: str, variant: LabelVariant) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = _barrier_outcome_frame(
        features,
        symbol,
        lookahead=variant.lookahead,
        tp_mult=variant.tp_atr_mult,
        sl_mult=variant.sl_atr_mult,
        spread_adjusted=variant.spread_adjusted,
    )
    labeled = features.iloc[: len(outcomes)].copy().reset_index(drop=True)
    labeled["label"] = outcomes["label"].to_numpy(dtype=int)
    return labeled, outcomes


def label_quality_row(symbol: str, variant: LabelVariant, outcomes: pd.DataFrame) -> dict:
    if outcomes.empty:
        return {
            "symbol": symbol,
            **asdict(variant),
            "rows": 0,
            "pct_no_trade": 0.0,
            "pct_buy": 0.0,
            "pct_sell": 0.0,
            "label_entropy": 0.0,
            "ambiguous_barrier_pct": 0.0,
            "avg_time_to_event": 0.0,
            "buy_tp_rate": 0.0,
            "sell_tp_rate": 0.0,
            "buy_avg_forward_return_atr": 0.0,
            "sell_avg_forward_return_atr": 0.0,
        }

    distribution = outcomes["label"].value_counts(normalize=True).to_dict()
    probabilities = np.array(list(distribution.values()), dtype=float)
    buy_rows = outcomes[outcomes["label"] == 1]
    sell_rows = outcomes[outcomes["label"] == 2]
    return {
        "symbol": symbol,
        **asdict(variant),
        "rows": int(len(outcomes)),
        "pct_no_trade": float(distribution.get(0, 0.0)),
        "pct_buy": float(distribution.get(1, 0.0)),
        "pct_sell": float(distribution.get(2, 0.0)),
        "label_entropy": float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) else 0.0,
        "ambiguous_barrier_pct": float(outcomes["ambiguous_barrier"].mean()),
        "avg_time_to_event": float(outcomes["time_to_event"].mean()),
        "buy_tp_rate": float((buy_rows["first_reason"] == "TP").mean()) if not buy_rows.empty else 0.0,
        "sell_tp_rate": float((sell_rows["first_reason"] == "TP").mean()) if not sell_rows.empty else 0.0,
        "buy_avg_forward_return_atr": float(buy_rows["forward_return_atr"].mean()) if not buy_rows.empty else 0.0,
        "sell_avg_forward_return_atr": float(sell_rows["forward_return_atr"].mean()) if not sell_rows.empty else 0.0,
    }


def label_selection_report_dir(symbol: str, feature_set: str, output_dir: str | None = None) -> Path:
    if output_dir:
        return Path(output_dir)
    return REPORTS_DIR / symbol / "label_selection" / feature_set


def label_selection_model_dir(symbol: str, feature_set: str) -> Path:
    return MODELS_DIR / symbol / "label_selection" / feature_set


def safe_variant_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name)


def is_label_cycle_complete(cycle_model_dir: Path, cycle_report_dir: Path) -> bool:
    required_model_files = [
        "cycle_meta.json",
        "model.joblib",
        "feature_columns.json",
        "best_threshold.json",
    ]
    required_report_files = ["oos_metrics.json", "oos_backtest.csv"]
    for name in required_model_files:
        if not (cycle_model_dir / name).exists():
            return False
    for name in required_report_files:
        if not (cycle_report_dir / name).exists():
            return False
    return True


def run_variant_walk_forward(
    symbol: str,
    features: pd.DataFrame,
    variant: LabelVariant,
    feature_set: str,
    n_trials: int,
    resume: bool,
    output_dir: str | None = None,
    max_cycles: int | None = None,
) -> dict:
    cfg = SYMBOLS[symbol].copy()
    cfg["lookahead_candles"] = variant.lookahead
    labeled_df, outcomes = label_variant_frame(features, symbol, variant)
    baseline_columns = feature_columns(labeled_df)
    selected_columns, skipped_features, _requested = select_features(feature_set, labeled_df, baseline_columns, symbol)
    if not selected_columns:
        raise ValueError(f"No valid features selected for {symbol} / {variant.name} / {feature_set}")

    wf_cfg = WALK_FORWARD_CONFIG.copy()
    wf_cfg.update(cfg.get("walk_forward", {}))
    splits = generate_walk_forward_splits(labeled_df, cfg, wf_cfg)
    if max_cycles is not None:
        splits = splits[:max_cycles]
    if not splits:
        raise RuntimeError(f"No walk-forward splits generated for {symbol} / {variant.name}")

    variant_name = safe_variant_name(variant.name)
    base_model_dir = label_selection_model_dir(symbol, feature_set) / variant_name
    base_report_dir = label_selection_report_dir(symbol, feature_set, output_dir) / variant_name
    base_model_dir.mkdir(parents=True, exist_ok=True)
    base_report_dir.mkdir(parents=True, exist_ok=True)

    quality = label_quality_row(symbol, variant, outcomes)
    save_json(quality, base_report_dir / "label_quality.json")

    lookahead_delta = pd.Timedelta(minutes=5 * variant.lookahead)
    summary_rows = []
    all_trades = []
    for split in splits:
        cycle = split["cycle"]
        cycle_model_dir = base_model_dir / f"cycle_{cycle:03d}"
        cycle_report_dir = base_report_dir / f"cycle_{cycle:03d}"

        if resume and is_label_cycle_complete(cycle_model_dir, cycle_report_dir):
            LOGGER.info("%s %s %s cycle %03d already complete; loading artifacts", symbol, variant.name, feature_set, cycle)
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
            LOGGER.warning("%s %s cycle %03d skipped due to insufficient rows", symbol, variant.name, cycle)
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
                "symbol": symbol,
                "feature_set": feature_set,
                "variant": asdict(variant),
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
            save_equity_curve_wf(
                oos_trades["pnl"].cumsum(),
                cycle_report_dir / "equity_curve.png",
                title=f"{symbol} {variant.name} {feature_set} Cycle {cycle}",
            )

        summary_rows.append(summary_row_from_cycle(cycle, split, thresholds, oos_metrics))
        LOGGER.info(
            "%s %s %s cycle %03d: trades=%d pnl=%.4f pf=%.3f",
            symbol,
            variant.name,
            feature_set,
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
        save_equity_curve_wf(
            aggregate["pnl"].cumsum(),
            base_report_dir / "aggregated_oos_equity.png",
            title=f"{symbol} {variant.name} {feature_set} Aggregated OOS",
        )
        aggregate_metrics = compute_oos_metrics(aggregate, aggregate["entry_time"].min(), aggregate["exit_time"].max())

    pd.DataFrame(summary_rows).to_csv(base_report_dir / "label_selection_summary.csv", index=False)
    save_json(aggregate_metrics, base_report_dir / "aggregated_oos_metrics.json")
    save_json(selected_columns, base_report_dir / "selected_features.json")
    save_json(skipped_features, base_report_dir / "skipped_features.json")

    result = {
        "symbol": symbol,
        "feature_set": feature_set,
        "variant": asdict(variant),
        "feature_count": len(selected_columns),
        "completed_cycles": len(summary_rows),
        **quality,
        **{f"oos_{key}": value for key, value in aggregate_metrics.items()},
        "report_dir": str(base_report_dir),
        "model_dir": str(base_model_dir),
    }
    save_json(result, base_report_dir / "result.json")
    return result


def rank_results(results: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    if frame.empty:
        return frame
    frame["abs_drawdown"] = frame["oos_max_drawdown"].abs()
    frame["label_balance_penalty"] = (frame[["pct_no_trade", "pct_buy", "pct_sell"]].max(axis=1) - 0.50).clip(lower=0.0)
    frame["selection_score"] = (
        frame["oos_profit_factor"].replace(np.inf, 10.0).clip(upper=10.0)
        + frame["oos_avg_pnl_per_trade"].clip(lower=-10.0, upper=10.0)
        + frame["label_entropy"].clip(upper=1.10)
        - frame["abs_drawdown"].clip(upper=1000.0) / 1000.0
        - frame["label_balance_penalty"]
    )
    return frame.sort_values(
        ["selection_score", "oos_profit_factor", "oos_net_profit", "abs_drawdown", "oos_total_trades"],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)


def write_comparison_report(symbol: str, feature_set: str, ranked: pd.DataFrame, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(report_dir / "label_variant_comparison.csv", index=False)
    if ranked.empty:
        (report_dir / "label_variant_comparison.md").write_text("# Label Variant Comparison\n\nNo results.\n", encoding="utf-8")
        return

    rows = []
    for item in ranked.itertuples():
        rows.append(
            f"| {item.name} | {item.lookahead} | {item.tp_atr_mult:.2f} | {item.sl_atr_mult:.2f} | "
            f"{bool(item.spread_adjusted)} | {item.feature_count} | {item.oos_total_trades} | "
            f"{item.oos_net_profit:.4f} | {item.oos_profit_factor:.4f} | {item.oos_max_drawdown:.4f} | "
            f"{item.oos_buy_trades} | {item.oos_sell_trades} | {item.label_entropy:.4f} | "
            f"{item.ambiguous_barrier_pct:.4%} | {item.selection_score:.4f} |"
        )
    content = f"""# Label Variant Comparison: {symbol} / {feature_set}

## Best Candidate

`{ranked.iloc[0]["name"]}` is ranked first by OOS profit factor, net profit, drawdown, and label quality score.

## Comparison

| Variant | Horizon | TP ATR | SL ATR | Spread Adj | Features | Trades | Net Profit | PF | Max DD | BUY | SELL | Entropy | Ambiguous | Score |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}
"""
    (report_dir / "label_variant_comparison.md").write_text(content, encoding="utf-8")


def parse_variant_names(value: str) -> list[str]:
    if value.lower() == "all":
        return [variant.name for variant in DEFAULT_VARIANTS]
    return [item.strip() for item in value.split(",") if item.strip()]


def variants_by_name(names: list[str]) -> list[LabelVariant]:
    mapping = {variant.name: variant for variant in DEFAULT_VARIANTS}
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ValueError(f"Unknown label variant(s): {', '.join(missing)}. Available: {', '.join(mapping)}")
    return [mapping[name] for name in names]


def run_label_selection(
    symbol: str,
    feature_set: str,
    variants: list[LabelVariant],
    n_trials: int,
    resume: bool,
    output_dir: str | None = None,
    max_cycles: int | None = None,
) -> pd.DataFrame:
    ensure_dirs([symbol])
    features = load_feature_data(symbol)
    results = []
    for variant in variants:
        LOGGER.info("%s: running label variant %s with %s", symbol, variant.name, feature_set)
        results.append(
            run_variant_walk_forward(
                symbol=symbol,
                features=features,
                variant=variant,
                feature_set=feature_set,
                n_trials=n_trials,
                resume=resume,
                output_dir=output_dir,
                max_cycles=max_cycles,
            )
        )
    ranked = rank_results(results)
    report_dir = label_selection_report_dir(symbol, feature_set, output_dir)
    write_comparison_report(symbol, feature_set, ranked, report_dir)
    LOGGER.info("Label selection complete for %s/%s. Best=%s", symbol, feature_set, ranked.iloc[0]["name"] if not ranked.empty else "none")
    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run label-variant walk-forward experiments before feature/model tuning.")
    parse_symbol_args(parser)
    parser.add_argument("--feature-set", choices=["core20", "core30", "robust70"], default="core20")
    parser.add_argument("--variants", default="all", help="Comma-separated variants or 'all'.")
    parser.add_argument("--trials", type=int, default=0, help="Optuna trials per cycle. 0 uses baseline parameters.")
    parser.add_argument("--resume", action="store_true", help="Resume completed variant cycles.")
    parser.add_argument("--output-dir", default=None, help="Override report output dir for one symbol.")
    parser.add_argument("--max-cycles", type=int, default=None, help="Limit cycles for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = resolve_symbols(args)
    if len(symbols) > 1 and args.output_dir:
        raise ValueError("--output-dir can only be used with a single symbol")
    variants = variants_by_name(parse_variant_names(args.variants))
    for symbol in symbols:
        run_label_selection(
            symbol=symbol,
            feature_set=args.feature_set,
            variants=variants,
            n_trials=args.trials,
            resume=args.resume,
            output_dir=args.output_dir,
            max_cycles=args.max_cycles,
        )


if __name__ == "__main__":
    main()
