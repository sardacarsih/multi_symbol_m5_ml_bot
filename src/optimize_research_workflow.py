import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from config import PROJECT_ROOT, REPORTS_DIR
from symbols import configured_symbols, parse_symbol_args, resolve_symbols
from utils import load_json, save_json, setup_logger


LOGGER = setup_logger("optimize_research_workflow")

STAGE_PREPARE = "prepare"
STAGE_LABEL = "label"
STAGE_FEATURE = "feature"
STAGE_WALK_FORWARD = "walk_forward"
STAGE_RETRAIN = "retrain"
STAGE_SUMMARY = "summary"
DEFAULT_STAGES = [STAGE_PREPARE, STAGE_LABEL, STAGE_FEATURE, STAGE_WALK_FORWARD, STAGE_RETRAIN, STAGE_SUMMARY]
DEFAULT_FEATURE_SETS = ["core20", "core30", "robust70"]
DEFAULT_IMPORTANCE_TOP_N = [20, 30]


@dataclass(frozen=True)
class WorkflowCommand:
    stage: str
    argv: list[str]


def normalize_stages(stages: list[str]) -> list[str]:
    if not stages or "all" in stages:
        return list(DEFAULT_STAGES)
    normalized = []
    for stage in stages:
        normalized.append(stage.replace("-", "_"))
    return normalized


def command_text(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv)


def script_command(script_name: str, *args: str) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / "src" / script_name), *args]


def symbol_args(symbols: list[str]) -> list[str]:
    configured = configured_symbols()
    if symbols == configured:
        return ["--all"]
    if len(symbols) == 1:
        return ["--symbol", symbols[0]]
    return ["--symbols", *symbols]


def symbol_batches(symbols: list[str], per_symbol: bool) -> list[list[str]]:
    if per_symbol:
        return [[symbol] for symbol in symbols]
    return [symbols]


def importance_symbols(symbols: list[str]) -> list[str]:
    return [symbol for symbol in symbols if (REPORTS_DIR / symbol / "feature_importance.csv").exists()]


def build_workflow_commands(symbols: list[str], args: argparse.Namespace) -> list[WorkflowCommand]:
    stages = normalize_stages(args.stages)
    trials = 0 if args.smoke else args.trials
    max_cycles = "3" if args.smoke and args.max_cycles is None else args.max_cycles
    commands: list[WorkflowCommand] = []
    batches = symbol_batches(symbols, args.per_symbol)

    if STAGE_PREPARE in stages:
        if args.force_download:
            # download_mt5_data always downloads; this flag is kept for stage parity.
            LOGGER.info("--force-download requested; prepare stage will refresh MT5 data.")
        for batch in batches:
            batch_symbol_args = symbol_args(batch)
            if not args.skip_download:
                commands.append(WorkflowCommand(STAGE_PREPARE, script_command("download_mt5_data.py", *batch_symbol_args)))
            commands.extend(
                [
                    WorkflowCommand(STAGE_PREPARE, script_command("features.py", *batch_symbol_args)),
                    WorkflowCommand(STAGE_PREPARE, script_command("labeling.py", *batch_symbol_args)),
                ]
            )

    if STAGE_LABEL in stages:
        for batch in batches:
            label_args = [
                *symbol_args(batch),
                "--feature-set",
                args.label_feature_set,
                "--variants",
                args.label_variants,
                "--trials",
                str(trials),
            ]
            if args.resume:
                label_args.append("--resume")
            if max_cycles is not None:
                label_args.extend(["--max-cycles", str(max_cycles)])
            commands.append(WorkflowCommand(STAGE_LABEL, script_command("label_selection_experiment.py", *label_args)))

    if STAGE_FEATURE in stages:
        for batch in batches:
            for feature_set in args.feature_sets:
                feature_args = [*symbol_args(batch), "--feature-set", feature_set, "--trials", str(trials)]
                if args.resume:
                    feature_args.append("--resume")
                if max_cycles is not None:
                    feature_args.extend(["--max-cycles", str(max_cycles)])
                commands.append(WorkflowCommand(STAGE_FEATURE, script_command("feature_selection_experiment.py", *feature_args)))

        for symbol in importance_symbols(symbols):
            for top_n in args.importance_top_n:
                feature_args = [
                    "--symbol",
                    symbol,
                    "--feature-set",
                    "importance_top_n",
                    "--top-n",
                    str(top_n),
                    "--trials",
                    str(trials),
                ]
                if args.resume:
                    feature_args.append("--resume")
                if max_cycles is not None:
                    feature_args.extend(["--max-cycles", str(max_cycles)])
                commands.append(WorkflowCommand(STAGE_FEATURE, script_command("feature_selection_experiment.py", *feature_args)))

    if STAGE_WALK_FORWARD in stages:
        for symbol in symbols:
            wf_args = ["--symbol", symbol, "--trials", str(trials)]
            if args.resume:
                wf_args.append("--resume")
            if args.force_download:
                wf_args.append("--force-download")
            commands.append(WorkflowCommand(STAGE_WALK_FORWARD, script_command("walk_forward_pipeline.py", *wf_args)))

    if STAGE_RETRAIN in stages:
        for batch in batches:
            retrain_args = [*symbol_args(batch), "--trials", str(trials)]
            if args.force_download:
                retrain_args.append("--force-download")
            if not args.deploy:
                retrain_args.append("--no-deploy")
            commands.append(WorkflowCommand(STAGE_RETRAIN, script_command("weekly_retrain.py", *retrain_args)))

    return commands


def safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def metric_score(metrics: dict) -> float:
    profit_factor = safe_float(metrics.get("profit_factor"))
    if math.isinf(profit_factor):
        profit_factor = 10.0
    profit_factor = max(0.0, min(profit_factor, 10.0))
    net_profit = safe_float(metrics.get("net_profit"))
    drawdown = abs(safe_float(metrics.get("max_drawdown")))
    trades = max(0, int(safe_float(metrics.get("total_trades"))))
    winrate = safe_float(metrics.get("winrate"))
    trade_bonus = min(math.log1p(trades) / 5.0, 1.5)
    profit_bonus = max(min(net_profit / 10.0, 2.0), -2.0)
    drawdown_penalty = min(drawdown / 10.0, 3.0)
    return profit_factor + profit_bonus + trade_bonus + winrate - drawdown_penalty


def row_from_metrics(symbol: str, category: str, candidate: str, metrics: dict, path: Path) -> dict:
    row = {
        "symbol": symbol,
        "category": category,
        "candidate": candidate,
        "total_trades": int(safe_float(metrics.get("total_trades"))),
        "net_profit": safe_float(metrics.get("net_profit")),
        "profit_factor": safe_float(metrics.get("profit_factor")),
        "max_drawdown": safe_float(metrics.get("max_drawdown")),
        "winrate": safe_float(metrics.get("winrate")),
        "buy_trades": int(safe_float(metrics.get("buy_trades"))),
        "sell_trades": int(safe_float(metrics.get("sell_trades"))),
        "score": metric_score(metrics),
        "path": str(path),
    }
    return row


def collect_label_rows(symbol: str) -> list[dict]:
    base = REPORTS_DIR / symbol / "label_selection"
    rows = []
    if not base.exists():
        return rows
    for metrics_path in base.glob("*/*/aggregated_oos_metrics.json"):
        metrics = load_json(metrics_path, default={})
        if not metrics:
            continue
        feature_set = metrics_path.parents[1].name
        variant = metrics_path.parent.name
        rows.append(row_from_metrics(symbol, "label", f"{feature_set}/{variant}", metrics, metrics_path.parent))
    return rows


def collect_feature_rows(symbol: str) -> list[dict]:
    base = REPORTS_DIR / symbol / "feature_selection"
    rows = []
    if not base.exists():
        return rows
    for metrics_path in base.glob("*/aggregated_oos_metrics.json"):
        metrics = load_json(metrics_path, default={})
        if not metrics:
            continue
        rows.append(row_from_metrics(symbol, "feature", metrics_path.parent.name, metrics, metrics_path.parent))
    return rows


def collect_walk_forward_rows(symbol: str) -> list[dict]:
    rows = []
    for mode in ["walk_forward", "walk_forward_tuned", "walk_forward_non_tuned"]:
        base = REPORTS_DIR / symbol / mode
        metrics_path = base / "aggregated_oos_metrics.json"
        metrics = load_json(metrics_path, default={})
        if metrics:
            rows.append(row_from_metrics(symbol, "walk_forward", mode, metrics, base))
    return rows


def collect_summary_rows(symbols: list[str]) -> list[dict]:
    rows = []
    for symbol in symbols:
        rows.extend(collect_label_rows(symbol))
        rows.extend(collect_feature_rows(symbol))
        rows.extend(collect_walk_forward_rows(symbol))
    return sorted(
        rows,
        key=lambda row: (
            row["symbol"],
            row["category"],
            -row["score"],
            -row["profit_factor"],
            -row["net_profit"],
            abs(row["max_drawdown"]),
        ),
    )


def best_rows_by_group(rows: list[dict]) -> list[dict]:
    best = {}
    for row in rows:
        key = (row["symbol"], row["category"])
        if key not in best or row["score"] > best[key]["score"]:
            best[key] = row
    return [best[key] for key in sorted(best)]


def write_summary_report(symbols: list[str]) -> Path:
    rows = collect_summary_rows(symbols)
    report_dir = REPORTS_DIR / "optimization"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "optimization_summary.csv"
    json_path = report_dir / "optimization_summary.json"
    md_path = report_dir / "optimization_summary.md"

    fieldnames = [
        "symbol",
        "category",
        "candidate",
        "total_trades",
        "net_profit",
        "profit_factor",
        "max_drawdown",
        "winrate",
        "buy_trades",
        "sell_trades",
        "score",
        "path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_json({"rows": rows, "best": best_rows_by_group(rows)}, json_path)

    lines = [
        "# Optimization Summary",
        "",
        "Best candidates by symbol and category.",
        "",
        "| Symbol | Category | Candidate | Trades | Net Profit | PF | Max DD | Winrate | BUY | SELL | Score |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows_by_group(rows):
        lines.append(
            f"| {row['symbol']} | {row['category']} | {row['candidate']} | {row['total_trades']} | "
            f"{row['net_profit']:.4f} | {row['profit_factor']:.4f} | {row['max_drawdown']:.4f} | "
            f"{row['winrate']:.2%} | {row['buy_trades']} | {row['sell_trades']} | {row['score']:.4f} |"
        )
    if not rows:
        lines.append("| n/a | n/a | No completed optimization reports found | 0 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0 | 0 | 0.0000 |")
    lines.extend(
        [
            "",
            "Full ranking is written to `optimization_summary.csv` and `optimization_summary.json`.",
            "",
            "Selection rule: prefer positive net profit, profit factor above 1, controlled drawdown, enough trades, and live BUY/SELL coverage.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Optimization summary written to %s", md_path)
    return md_path


def run_commands(commands: list[WorkflowCommand], dry_run: bool) -> None:
    for command in commands:
        text = command_text(command.argv)
        if dry_run:
            print(f"[{command.stage}] {text}")
            continue
        LOGGER.info("[%s] %s", command.stage, text)
        subprocess.run(command.argv, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the labeling, feature, and walk-forward optimization workflow.")
    parse_symbol_args(parser)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", "prepare", "label", "feature", "walk_forward", "walk-forward", "retrain", "summary"],
        help="Workflow stages to run. Default: all.",
    )
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per cycle for full runs.")
    parser.add_argument("--smoke", action="store_true", help="Use --trials 0 and --max-cycles 3 for quick validation.")
    parser.add_argument("--max-cycles", type=int, default=None, help="Limit experiment cycles.")
    parser.add_argument("--resume", action="store_true", help="Resume completed experiment cycles.")
    parser.add_argument("--force-download", action="store_true", help="Force fresh MT5 download where supported.")
    parser.add_argument("--skip-download", action="store_true", help="Use cached raw data and skip download_mt5_data.py in prepare stage.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--per-symbol", action="store_true", help="Run multi-symbol stages as separate --symbol commands.")
    parser.add_argument("--label-feature-set", choices=["core20", "core30", "robust70"], default="core20")
    parser.add_argument("--label-variants", default="all", help="Comma-separated label variants or 'all'.")
    parser.add_argument("--feature-sets", nargs="+", default=DEFAULT_FEATURE_SETS)
    parser.add_argument("--importance-top-n", nargs="+", type=int, default=DEFAULT_IMPORTANCE_TOP_N)
    parser.add_argument("--deploy", action="store_true", help="Allow weekly_retrain.py to update live_model_meta.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = resolve_symbols(args)
    stages = normalize_stages(args.stages)
    if set(stages) - {STAGE_SUMMARY}:
        run_commands(build_workflow_commands(symbols, args), dry_run=args.dry_run)
    if STAGE_SUMMARY in stages and not args.dry_run:
        write_summary_report(symbols)


if __name__ == "__main__":
    main()
