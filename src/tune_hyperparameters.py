import argparse
import sys
import json
from pathlib import Path

# Check if optuna is installed before continuing
try:
    import optuna
except ImportError:
    print("\n[ERROR] Optuna is not installed.")
    print("Please install it first by running: pip install optuna\n")
    sys.exit(1)

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

from config import MODELS_DIR
from train import load_labeled, feature_columns, chronological_split, sample_weights
from utils import ensure_dirs, setup_logger

LOGGER = setup_logger("tune_hyperparameters")

def objective(trial, X_train, y_train, X_val, y_val, w_train) -> float:
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
        # Capture early stopping iteration
        best_iteration = getattr(model, "best_iteration", params["n_estimators"])
        trial.set_user_attr("best_n_estimators", int(best_iteration))
    except Exception:
        # Fallback if early stopping fails
        model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)
        trial.set_user_attr("best_n_estimators", params["n_estimators"])
        
    preds = model.predict_proba(X_val)
    loss = log_loss(y_val, preds, labels=[0, 1, 2])
    return loss

def tune_symbol(symbol: str, n_trials: int) -> None:
    ensure_dirs([symbol])
    LOGGER.info("Starting hyperparameter tuning for %s using %d trials...", symbol, n_trials)
    
    # Load and split labeled data
    df = load_labeled(symbol)
    train_df, val_df, _ = chronological_split(df)
    columns = feature_columns(df)
    
    X_train = train_df[columns]
    y_train = train_df["label"]
    X_val = val_df[columns]
    y_val = val_df["label"]
    w_train = sample_weights(y_train)

    # Disable optuna logs to avoid clutter, showing only progress
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="minimize")
    
    # Progress feedback callback
    def progress_callback(study, trial):
        LOGGER.info("Trial %3d/%d completed. Best LogLoss so far: %.6f", trial.number + 1, n_trials, study.best_value)

    study.optimize(
        lambda t: objective(t, X_train, y_train, X_val, y_val, w_train), 
        n_trials=n_trials,
        callbacks=[progress_callback]
    )

    best_params = study.best_params
    best_n_estimators = study.best_trials[0].user_attrs.get("best_n_estimators", best_params.get("n_estimators", 500))
    
    # Update n_estimators with the actual early stopped count
    best_params["n_estimators"] = best_n_estimators
    
    output_dir = MODELS_DIR / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tuned_params.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
        
    LOGGER.info("Tuning complete for %s!", symbol)
    LOGGER.info("Best validation LogLoss: %.6f", study.best_value)
    LOGGER.info("Best parameters saved to %s", output_path)
    LOGGER.info("Tuned parameters: %s", json.dumps(best_params, indent=2))

def main() -> None:
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters using Optuna.")
    parser.add_argument("--symbol", required=True, help="Symbol to tune (e.g. XAUUSD, USTEC)")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials to search (default: 50)")
    args = parser.parse_args()
    
    tune_symbol(args.symbol, args.trials)

if __name__ == "__main__":
    main()
