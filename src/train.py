import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from config import EXCLUDE_TRAIN_COLUMNS, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, SYMBOLS, USE_CLASS_WEIGHTS
from symbols import parse_symbol_args, resolve_symbols
from utils import ensure_dirs, save_json, save_model, setup_logger, symbol_to_filename

LOGGER = setup_logger("train")

REDUNDANT_TRAIN_COLUMNS = {
    "point",
    "tick_size",
    "tick_value",
    "spread",
    "spread_points",
    "symbol_id",
    "dist_from_high_24",
    "dist_from_low_24",
    "price_change_6",
    "atr_14",
    "atr_5",
    "atr_28",
    "volume_sma_12",
    "volume_sma_24",
    "body_size",
    "upper_wick",
    "lower_wick",
    "candle_range",
    "macd",
    "macd_signal",
    "rsi_21",
}

REDUNDANT_TRAIN_PREFIXES = (
    "symbol_",
    "ema_",
    "highest_high_",
    "lowest_low_",
)

SUSPECT_SHORT_HORIZON_PATTERN_COLUMNS = {
    "doji",
    "hammer",
    "shooting_star",
    "engulfing",
    "consecutive_green",
    "consecutive_red",
    "rsi_price_divergence",
}

ROBUST_TRAIN_FEATURES = {
    "hour",
    "is_london_session",
    "is_newyork_session",
    "is_london_newyork_overlap",
    "is_london_killzone",
    "is_newyork_killzone",
    "spread_to_atr",
    "spread_rank_100",
    "session_spread_stress",
    "session_liquidity_stress",
    "abnormal_spread",
    "return_3",
    "return_6",
    "atr_ratio",
    "atr_5_to_14",
    "atr_14_to_28",
    "vol_regime",
    "vol_expansion",
    "atr_rank_100",
    "atr_rank_288",
    "realized_vol_rank_100",
    "vol_compression",
    "vol_breakout_regime",
    "vol_compression_release",
    "rolling_std_24",
    "close_to_ema20",
    "close_to_ema50",
    "ema20_slope",
    "ema20_slope_atr",
    "ema_alignment_score",
    "adx_14",
    "di_spread_14",
    "trend_direction_strength",
    "trend_range_regime",
    "rsi_7",
    "rsi_14",
    "macd_hist",
    "macd_hist_change_3",
    "close_position_in_range_24",
    "dist_from_high_24_atr",
    "dist_from_low_24_atr",
    "range_width_48_atr",
    "breakout_high_24",
    "breakout_low_24",
    "prev_day_high_dist_atr",
    "prev_day_low_dist_atr",
    "prev_week_high_dist_atr",
    "prev_week_low_dist_atr",
    "intraday_range_adr",
    "adr_14_atr",
    "adr_expansion",
    "swing_high_dist_atr",
    "swing_low_dist_atr",
    "market_structure_state",
    "liquidity_sweep_high_24",
    "liquidity_sweep_low_24",
    "failed_breakout_high_24",
    "failed_breakout_low_24",
    "volume_ratio_12",
    "volume_rank_100",
    "volume_price_trend",
    "dist_from_vwap",
    "bb_position",
    "m15_return_1",
    "m15_close_to_ema20",
    "m15_range_position_20",
    "m15_atr_ratio",
    "h1_return_1",
    "h1_close_to_ema20",
    "h1_range_position_20",
    "h1_atr_ratio",
}


def load_labeled(symbol: str) -> pd.DataFrame:
    filename = symbol_to_filename(symbol)
    path = PROCESSED_DATA_DIR / symbol / f"{filename}_m5_labeled.csv"
    if not path.exists():
        raise FileNotFoundError(f"Labeled data not found: {path}")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(EXCLUDE_TRAIN_COLUMNS)
    excluded.update(col for col in df.columns if col.startswith("future") or col.startswith("target"))
    columns = []
    for col in df.columns:
        if col in excluded or df[col].dtype == object:
            continue
        if col in REDUNDANT_TRAIN_COLUMNS or col.startswith(REDUNDANT_TRAIN_PREFIXES):
            continue
        if col in SUSPECT_SHORT_HORIZON_PATTERN_COLUMNS:
            continue
        if col not in ROBUST_TRAIN_FEATURES:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            columns.append(col)
    return columns


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    if train_end == 0 or val_end <= train_end or n - val_end == 0:
        raise ValueError("Dataset too small for 70/15/15 chronological split")
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def build_model(symbol: str | None = None) -> XGBClassifier:
    if symbol:
        tuned_path = MODELS_DIR / symbol / "tuned_params.json"
        if tuned_path.exists():
            try:
                with open(tuned_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                # Force standard structural parameters
                params["objective"] = "multi:softprob"
                params["num_class"] = 3
                params["eval_metric"] = "mlogloss"
                params["random_state"] = 20260605
                params["n_jobs"] = -1
                LOGGER.info("Using OPTUNA tuned parameters loaded from %s", tuned_path)
                return XGBClassifier(**params)
            except Exception as e:
                LOGGER.warning("Failed to load tuned params for %s: %s. Falling back to defaults.", symbol, e)

    if symbol == "XAUUSD":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            max_depth=3,
            learning_rate=0.02,
            n_estimators=800,
            subsample=0.7,
            colsample_bytree=0.6,
            min_child_weight=5,
            reg_lambda=3.0,
            reg_alpha=0.5,
            gamma=0.3,
            random_state=20260605,
            n_jobs=-1,
        )
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        max_depth=4,
        learning_rate=0.03,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=20260605,
        n_jobs=-1,
    )


def sample_weights(y: pd.Series) -> pd.Series | None:
    if not USE_CLASS_WEIGHTS:
        return None
    counts = y.value_counts().to_dict()
    total = len(y)
    classes = 3
    weights = {label: total / (classes * count) for label, count in counts.items() if count > 0}
    return y.map(weights).astype(float)


def fit_model(model: XGBClassifier, x_train, y_train, x_val, y_val) -> XGBClassifier:
    weights = sample_weights(y_train)
    try:
        model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_val, y_val)], verbose=False, early_stopping_rounds=50)
    except TypeError:
        model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_val, y_val)], verbose=False)
    return model


def write_reports(model: XGBClassifier, x_test, y_test, columns: list[str], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    preds = model.predict(x_test)
    labels = [0, 1, 2]
    report = classification_report(y_test, preds, labels=labels, target_names=["NO_TRADE", "BUY", "SELL"], zero_division=0)
    (report_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(confusion_matrix(y_test, preds, labels=labels), index=labels, columns=labels).to_csv(report_dir / "confusion_matrix.csv")
    importances = getattr(model, "feature_importances_", np.zeros(len(columns)))
    pd.DataFrame({"feature": columns, "importance": importances}).sort_values("importance", ascending=False).to_csv(
        report_dir / "feature_importance.csv", index=False
    )


def train_symbol(symbol: str) -> XGBClassifier:
    ensure_dirs([symbol])
    df = load_labeled(symbol)
    train_df, val_df, test_df = chronological_split(df)
    columns = feature_columns(df)
    model = fit_model(
        build_model(symbol),
        train_df[columns],
        train_df["label"],
        val_df[columns],
        val_df["label"],
    )

    filename = symbol_to_filename(symbol)
    model_dir = MODELS_DIR / symbol
    report_dir = REPORTS_DIR / symbol
    save_model(model, model_dir / f"{filename}_m5_xgboost.joblib")
    save_json(columns, model_dir / "feature_columns.json")
    write_reports(model, test_df[columns], test_df["label"], columns, report_dir)
    LOGGER.info("Trained %s model with %s features", symbol, len(columns))
    return model


def train_global() -> XGBClassifier:
    ensure_dirs(["global"])
    frames = [load_labeled(symbol) for symbol in SYMBOLS]
    df = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    train_df, val_df, test_df = chronological_split(df)
    columns = feature_columns(df)
    model = fit_model(build_model(), train_df[columns], train_df["label"], val_df[columns], val_df["label"])
    model_dir = MODELS_DIR / "global"
    report_dir = REPORTS_DIR / "global"
    save_model(model, model_dir / "global_m5_xgboost.joblib")
    save_json(columns, model_dir / "feature_columns.json")
    write_reports(model, test_df[columns], test_df["label"], columns, report_dir)
    LOGGER.info("Trained global model with %s rows", len(df))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost classifiers.")
    parse_symbol_args(parser)
    parser.add_argument("--global", dest="global_model", action="store_true", help="Train one global model")
    args = parser.parse_args()
    if args.global_model:
        train_global()
        return
    for symbol in resolve_symbols(args):
        train_symbol(symbol)


if __name__ == "__main__":
    main()
