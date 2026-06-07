from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from risk import check_spread_filter
from strategy_filters import session_allowed


def file_fingerprint(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "mtime_ns": None, "size": None}
    stat = path.stat()
    return {"path": str(path), "exists": True, "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def build_threshold_fingerprint(
    model_path: Path,
    labeled_path: Path,
    columns: list[str],
    test_df: pd.DataFrame,
) -> dict:
    return {
        "model": file_fingerprint(model_path),
        "labeled_data": file_fingerprint(labeled_path),
        "feature_column_count": len(columns),
        "test_row_count": int(len(test_df)),
        "test_start": str(test_df["time"].min()) if not test_df.empty else None,
        "test_end": str(test_df["time"].max()) if not test_df.empty else None,
    }


def threshold_fingerprint_status(thresholds: dict, current_fingerprint: dict) -> str:
    stored = thresholds.get("artifact_fingerprint")
    if not stored:
        return "missing"
    return "valid" if stored == current_fingerprint else "stale"


def split_diagnostics(full_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    def part(prefix: str, frame: pd.DataFrame) -> dict:
        return {
            f"{prefix}_rows": int(len(frame)),
            f"{prefix}_start": str(frame["time"].min()) if not frame.empty else None,
            f"{prefix}_end": str(frame["time"].max()) if not frame.empty else None,
        }

    result = {}
    result.update(part("data", full_df))
    result.update(part("train", train_df))
    result.update(part("validation", val_df))
    result.update(part("test", test_df))
    return result


def signal_diagnostics(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    cfg: dict,
    buy_threshold: float,
    sell_threshold: float,
    decide_signal: Callable[[float, float, float, float], int],
    final_trades: int,
) -> dict:
    if len(df) == 0 or len(probabilities) == 0:
        return {
            "prob_buy_max": 0.0,
            "prob_sell_max": 0.0,
            "prob_buy_p95": 0.0,
            "prob_sell_p95": 0.0,
            "signals_raw": 0,
            "signals_after_session": 0,
            "signals_after_spread": 0,
            "final_trades": int(final_trades),
        }

    prob_buy = probabilities[:, 1]
    prob_sell = probabilities[:, 2]
    raw_signals = np.array(
        [decide_signal(float(buy), float(sell), buy_threshold, sell_threshold) for buy, sell in zip(prob_buy, prob_sell)]
    )
    session_mask = np.array([session_allowed(row, cfg) for _, row in df.iterrows()])
    spread_mask = np.array(
        [check_spread_filter(float(value), float(cfg["max_spread_to_atr"])) for value in df["spread_to_atr"].to_numpy()]
    )
    session_signals = np.where(session_mask, raw_signals, 0)
    spread_signals = np.where(session_mask & spread_mask, raw_signals, 0)

    return {
        "prob_buy_max": float(prob_buy.max()),
        "prob_sell_max": float(prob_sell.max()),
        "prob_buy_p95": float(np.quantile(prob_buy, 0.95)),
        "prob_sell_p95": float(np.quantile(prob_sell, 0.95)),
        "signals_raw": int((raw_signals > 0).sum()),
        "signals_after_session": int((session_signals > 0).sum()),
        "signals_after_spread": int((spread_signals > 0).sum()),
        "final_trades": int(final_trades),
    }
