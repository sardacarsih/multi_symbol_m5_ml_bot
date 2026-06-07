import json
import logging
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from config import (
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TRADES_DIR,
)


def setup_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_dirs(symbols: Iterable[str] | None = None) -> None:
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TRADES_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    for symbol in symbols or []:
        for base in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
            (base / symbol).mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_model(path: Path):
    return joblib.load(path)


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def symbol_to_filename(symbol: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in symbol.strip())
    return "_".join(part for part in cleaned.split("_") if part)


def get_closed_candles_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    if "time" in result.columns:
        result["time"] = pd.to_datetime(result["time"])
        result = result.sort_values("time")
    return result.iloc[:-1].copy() if len(result) > 1 else result.iloc[0:0].copy()


def find_mt5_symbol_candidates(mt5, requested: str, limit: int = 20) -> list[str]:
    symbols = mt5.symbols_get()
    if symbols is None:
        return []
    requested_lower = requested.lower()
    tokens = [requested_lower, requested_lower.replace("xau", "gold"), "nas", "ustec", "us100"]
    matches = []
    for item in symbols:
        name = item.name
        name_lower = name.lower()
        if any(token and token in name_lower for token in tokens) or name_lower in requested_lower:
            matches.append(name)
        if len(matches) >= limit:
            break
    return matches


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
