import argparse
import sys
import json
import shutil
from pathlib import Path

# Add src directory to path to ensure clean imports
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, SYMBOLS
from utils import setup_logger, symbol_to_filename, ensure_dirs, load_json

# Import pipeline steps
from download_mt5_data import connect_mt5, download_symbol
from features import process_symbol as run_features
from labeling import process_symbol as run_labeling
from tune_hyperparameters import tune_symbol as run_tuning
from train import train_symbol as run_training
from threshold_search import search_symbol as run_threshold_search
from backtest import backtest_symbol as run_backtest

LOGGER = setup_logger("auto_tune_pipeline")


def run_pipeline(symbol: str, n_trials: int, force_download: bool, skip_tuning: bool) -> None:
    symbol = symbol.upper()
    filename = symbol_to_filename(symbol)
    ensure_dirs([symbol])
    
    LOGGER.info("=" * 60)
    LOGGER.info("STARTING AUTO-TUNING & COMPARISON PIPELINE FOR SYMBOL: %s", symbol)
    LOGGER.info("=" * 60)
    
    # 1. Download Data
    raw_path = RAW_DATA_DIR / symbol / f"{filename}_m5_raw.csv"
    if not raw_path.exists() or force_download:
        LOGGER.info("[PREP 1/3] Downloading raw M5 data from MT5...")
        try:
            mt5 = connect_mt5()
            try:
                download_symbol(mt5, symbol)
            finally:
                mt5.shutdown()
            LOGGER.info("[PREP 1/3] Download completed successfully.")
        except Exception as e:
            LOGGER.error("Failed to download data from MT5: %s", e)
            LOGGER.error("Please make sure MT5 terminal is open and connected.")
            sys.exit(1)
    else:
        LOGGER.info("[PREP 1/3] Local raw data exists. Skipping download (use --force-download to refresh).")
        
    # 2. Features
    LOGGER.info("[PREP 2/3] Generating technical features...")
    try:
        run_features(symbol)
        LOGGER.info("[PREP 2/3] Feature generation complete.")
    except Exception as e:
        LOGGER.error("Failed in Feature Generation step: %s", e)
        sys.exit(1)
        
    # 3. Labeling
    LOGGER.info("[PREP 3/3] Generating labels using ATR barriers...")
    try:
        run_labeling(symbol)
        LOGGER.info("[PREP 3/3] Labeling complete.")
    except Exception as e:
        LOGGER.error("Failed in Labeling step: %s", e)
        sys.exit(1)

    # Backup existing tuned params if any
    tuned_params_path = MODELS_DIR / symbol / "tuned_params.json"
    backup_params_path = MODELS_DIR / symbol / "tuned_params_backup.json"
    
    has_existing_tuned = tuned_params_path.exists()
    if has_existing_tuned:
        shutil.copy2(tuned_params_path, backup_params_path)
        LOGGER.info("Backed up existing tuned parameters to %s", backup_params_path.name)
        
    baseline_metrics = None
    baseline_thresholds = None
    baseline_params = None
    
    tuned_metrics = None
    tuned_thresholds = None
    tuned_params = None
    
    try:
        # ----------------------------------------------------
        # STAGE 1: EVALUATE BASELINE (UNTUNED) MODEL
        # ----------------------------------------------------
        LOGGER.info("\n" + "=" * 50)
        LOGGER.info("STAGE 1: EVALUATING BASELINE (UNTUNED) MODEL")
        LOGGER.info("=" * 50)
        
        # Remove tuned_params.json temporarily to force default/fallback parameters
        if tuned_params_path.exists():
            tuned_params_path.unlink()
            
        LOGGER.info("Training baseline model with default configurations...")
        run_training(symbol)
        
        LOGGER.info("Optimizing decision thresholds for baseline...")
        run_threshold_search(symbol)
        
        LOGGER.info("Running baseline backtest...")
        baseline_metrics, _, _ = run_backtest(symbol)
        
        best_threshold_path = MODELS_DIR / symbol / "best_threshold.json"
        baseline_thresholds = load_json(best_threshold_path, default={})
        
        # Get baseline parameters used
        # XAUUSD has specialized default params, other symbols use standard defaults
        if symbol == "XAUUSD":
            baseline_params = {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 800}
        else:
            baseline_params = {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 500}
            
        LOGGER.info("Baseline evaluation complete.")
        
        # ----------------------------------------------------
        # STAGE 2: RUN TUNING & EVALUATE TUNED MODEL
        # ----------------------------------------------------
        if not skip_tuning:
            LOGGER.info("\n" + "=" * 50)
            LOGGER.info("STAGE 2: OPTUNA HYPERPARAMETER TUNING & EVALUATION")
            LOGGER.info("=" * 50)
            
            LOGGER.info("Running Optuna Hyperparameter Tuning (%d trials)...", n_trials)
            run_tuning(symbol, n_trials)
            
            LOGGER.info("Training tuned model with optimized parameters...")
            run_training(symbol)
            
            LOGGER.info("Optimizing decision thresholds for tuned model...")
            run_threshold_search(symbol)
            
            LOGGER.info("Running tuned model backtest...")
            tuned_metrics, _, _ = run_backtest(symbol)
            
            tuned_thresholds = load_json(best_threshold_path, default={})
            tuned_params = load_json(tuned_params_path, default={})
            LOGGER.info("Tuned model evaluation complete.")
            
        else:
            LOGGER.info("\n[INFO] Skipping Stage 2 (Tuning and Tuned model evaluation).")
            # If skip_tuning was requested and we had existing tuned parameters, restore them
            if backup_params_path.exists():
                shutil.copy2(backup_params_path, tuned_params_path)
                LOGGER.info("Restored pre-existing tuned parameters from backup.")
                
                # Re-train and re-run threshold search to ensure reports match the restored tuned params
                LOGGER.info("Re-evaluating pre-existing tuned model...")
                run_training(symbol)
                run_threshold_search(symbol)
                tuned_metrics, _, _ = run_backtest(symbol)
                tuned_thresholds = load_json(best_threshold_path, default={})
                tuned_params = load_json(tuned_params_path, default={})
                
    finally:
        # Clean up backups
        if backup_params_path.exists():
            backup_params_path.unlink()
            LOGGER.info("Cleaned up backup parameter files.")

    # ----------------------------------------------------
    # STAGE 3: OUTPUT SIDE-BY-SIDE COMPARISON
    # ----------------------------------------------------
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info(" PERFORMANCE COMPARISON FOR %s (BEFORE VS AFTER TUNING)", symbol)
    LOGGER.info("=" * 70)
    
    # Format values for table
    def fmt_num(val, fmt="%.4f"):
        if val is None:
            return "N/A"
        return fmt % float(val)
        
    def fmt_pct(val):
        if val is None:
            return "N/A"
        return "%.2f%%" % (float(val) * 100)
        
    def fmt_bool(val):
        if val is None:
            return "N/A"
        return "TRUE" if bool(val) else "FALSE"

    b_tr = baseline_thresholds or {}
    t_tr = tuned_thresholds or {}
    
    b_p = baseline_params or {}
    t_p = tuned_params or {}
    
    b_m = baseline_metrics or {}
    t_m = tuned_metrics or {}

    rows = [
        ("Parameter: max_depth", str(b_p.get("max_depth", "N/A")), str(t_p.get("max_depth", "N/A"))),
        ("Parameter: learning_rate", fmt_num(b_p.get("learning_rate"), "%.4f"), fmt_num(t_p.get("learning_rate"), "%.4f")),
        ("Parameter: n_estimators", str(b_p.get("n_estimators", "N/A")), str(t_p.get("n_estimators", "N/A"))),
        ("-" * 30, "-" * 16, "-" * 16),
        ("Optimal Buy Threshold", fmt_num(b_tr.get("buy_threshold"), "%.2f"), fmt_num(t_tr.get("buy_threshold"), "%.2f")),
        ("Optimal Sell Threshold", fmt_num(b_tr.get("sell_threshold"), "%.2f"), fmt_num(t_tr.get("sell_threshold"), "%.2f")),
        ("-" * 30, "-" * 16, "-" * 16),
        ("Total Trades", str(int(b_m.get("total_trades", 0))), str(int(t_m.get("total_trades", 0)))),
        ("Win Rate (%)", fmt_pct(b_m.get("winrate")), fmt_pct(t_m.get("winrate"))),
        ("Net Profit (ATR units)", fmt_num(b_m.get("net_profit"), "%.4f"), fmt_num(t_m.get("net_profit"), "%.4f")),
        ("Profit Factor", fmt_num(b_m.get("profit_factor"), "%.3f"), fmt_num(t_m.get("profit_factor"), "%.3f")),
        ("Max Drawdown", fmt_num(b_m.get("max_drawdown"), "%.4f"), fmt_num(t_m.get("max_drawdown"), "%.4f")),
        ("Avg Holding (candles)", fmt_num(b_m.get("average_holding_candles"), "%.2f"), fmt_num(t_m.get("average_holding_candles"), "%.2f")),
        ("Trades Per Day", fmt_num(b_m.get("trades_per_day"), "%.2f"), fmt_num(t_m.get("trades_per_day"), "%.2f")),
        ("-" * 30, "-" * 16, "-" * 16),
        ("Eligible for Live", fmt_bool(b_tr.get("eligible")), fmt_bool(t_tr.get("eligible"))),
    ]
    
    # Print the table
    LOGGER.info("%-32s | %-16s | %-16s", "Metric / Attribute", "BEFORE TUNING", "AFTER TUNING")
    LOGGER.info("-" * 32 + "-+-" + "-" * 16 + "-+-" + "-" * 16)
    for r in rows:
        LOGGER.info("%-32s | %-16s | %-16s", r[0], r[1], r[2])
    
    LOGGER.info("=" * 70)
    LOGGER.info("Tuned parameters saved in models/ and reports/ directories.")
    LOGGER.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Hyperparameter Tuning and Model Comparison Pipeline.")
    parser.add_argument("--symbol", required=True, help="Symbol to process (e.g. GBPUSD, EURUSD, XAUUSD)")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna tuning trials (default: 50)")
    parser.add_argument("--force-download", action="store_true", help="Force MT5 download even if raw data exists")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip Optuna tuning and use existing/default params")
    args = parser.parse_args()
    
    run_pipeline(args.symbol, args.trials, args.force_download, args.skip_tuning)


if __name__ == "__main__":
    main()
