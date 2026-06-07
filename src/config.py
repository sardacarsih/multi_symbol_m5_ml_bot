from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRADES_DIR = DATA_DIR / "trades"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

TIMEFRAME = "M5"
MAX_OPEN_POSITIONS_PER_SYMBOL = 1
MAX_TOTAL_OPEN_POSITIONS = 2
COOLDOWN_CANDLES = 3
MAGIC_NUMBER = 20260605
DEVIATION = 30
MODEL_TYPE = "xgboost"
TRAIN_MODE = "per_symbol"
USE_RISK_PERCENT = False
MAX_TOTAL_RISK = 0.01
MAX_DAILY_LOSS = 0.02
MAX_DAILY_TRADES = 3
MAX_CONSECUTIVE_LOSSES = 3
USE_CLASS_WEIGHTS = True
MIN_THRESHOLD_SIGNALS = 30
MIN_THRESHOLD_COMBINED_PRECISION = 0.28
MIN_THRESHOLD_PROFIT_FACTOR = 1.15
REQUIRE_ELIGIBLE_THRESHOLD_FOR_LIVE = True

class SymbolsDict(dict):
    def __contains__(self, key):
        if not isinstance(key, str):
            return False
        return super().__contains__(key.upper()) or super().__contains__(key)

    def __getitem__(self, key):
        if not isinstance(key, str):
            return super().__getitem__(key)
        key_upper = key.upper()
        if super().__contains__(key_upper):
            return super().__getitem__(key_upper)
        if super().__contains__(key):
            return super().__getitem__(key)
        raise KeyError(f"Unknown symbol '{key}'. Configure it explicitly in SYMBOLS before use.")


SYMBOLS = SymbolsDict({
    "XAUUSD": {
        "mt5_symbol": "XAUUSD",
        "timeframe": "M5",
        "data_months": 24,
        "lookahead_candles": 8,
        "label_tp_atr_mult": 1.2,
        "label_sl_atr_mult": 1.2,
        "live_sl_atr_mult": 1.0,
        "live_tp_atr_mult": 1.5,
        "buy_threshold": 0.66,
        "sell_threshold": 0.66,
        "threshold_search_min": 0.50,
        "threshold_search_max": 0.80,
        "threshold_search_step": 0.02,
        "target_min_trades_per_day": 10.0,
        "target_max_trades_per_day": 15.0,
        "min_target_profit_factor": 1.20,
        "require_positive_net_for_target": True,
        "allow_session_relax_for_target": True,
        "max_spread_to_atr": 0.15,
        "allowed_entry_sessions": ["is_london_session", "is_newyork_session", "is_london_newyork_overlap"],
        "min_threshold_signals": 20,
        "enabled_for_live": False,
        "default_lot": 0.01,
        "risk_per_trade": 0.005,
        "tick_size": 0.0,
        "tick_value": 0.0,
        "side_training_mode": "separate",
        "live_side_policy": "gated",
    },
    "USTEC": {
        "mt5_symbol": "USTEC",
        "timeframe": "M5",
        "data_months": 24,
        "lookahead_candles": 6,
        "label_tp_atr_mult": 1.3,
        "label_sl_atr_mult": 1.0,
        "live_sl_atr_mult": 1.1,
        "live_tp_atr_mult": 1.6,
        "buy_threshold": 0.66,
        "sell_threshold": 0.66,
        "max_spread_to_atr": 0.25,
        "allowed_entry_sessions": [],
        "min_threshold_signals": 30,
        "enabled_for_live": False,
        "default_lot": 0.01,
        "risk_per_trade": 0.005,
        "tick_size": 0.0,
        "tick_value": 0.0,
        "side_training_mode": "separate",
        "live_side_policy": "gated",
    },
    "USTEC_X100": {
        "mt5_symbol": "USTEC_x100",
        "timeframe": "M5",
        "data_months": 12,
        "lookahead_candles": 9,
        "label_tp_atr_mult": 1.2,
        "label_sl_atr_mult": 1.0,
        "live_sl_atr_mult": 1.1,
        "live_tp_atr_mult": 1.6,
        "buy_threshold": 0.66,
        "sell_threshold": 0.66,
        "max_spread_to_atr": 0.30,
        "allowed_entry_sessions": [],
        "min_threshold_signals": 30,
        "enabled_for_live": False,
        "default_lot": 0.01,
        "risk_per_trade": 0.005,
        "tick_size": 0.0,
        "tick_value": 0.0,
        "side_training_mode": "separate",
        "live_side_policy": "gated",
        "regime_filter_col": "vol_expansion",
        "walk_forward": {
            "training_window_months": 8,
            "validation_months": 1,
            "oos_months": 1,
            "retrain_frequency": "weekly",
            "expanding_window": False,
            "n_optuna_trials": 50,
            "commission_per_lot": 0.0,
            "slippage_points": 2.0,
        },
    },
})


EXCLUDE_TRAIN_COLUMNS = {
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "real_volume",
    "label",
}

WALK_FORWARD_CONFIG = {
    "training_window_months": 12,
    "validation_months": 2,
    "oos_months": 1,
    "retrain_frequency": "weekly",  # weekly retraining
    "expanding_window": False,     # rolling window
    "n_optuna_trials": 50,
    "commission_per_lot": 0.0,     # commission per lot
    "slippage_points": 2.0,        # slippage in points
}
