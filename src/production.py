import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

STRICT_MIN_SIDE_TRADES = 30
STRICT_MIN_PROFIT_FACTOR = 1.25
STRICT_MAX_DRAWDOWN_ABS = 5.0
STRICT_MAX_DATA_STALENESS_DAYS = 7

REQUIRED_ARTIFACT_FILES = [
    "buy_model.joblib",
    "sell_model.joblib",
    "buy_threshold.json",
    "sell_threshold.json",
    "side_model_meta.json",
    "feature_columns.json",
    "best_threshold.json",
]


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(artifact_dir: Path, files: Iterable[str] = REQUIRED_ARTIFACT_FILES) -> dict:
    return {name: file_sha256(artifact_dir / name) for name in files}


def validate_broker_specs_from_frame(df: pd.DataFrame) -> list[str]:
    failures = []
    for column in ["point", "tick_size", "tick_value"]:
        if column not in df.columns:
            failures.append(f"broker spec missing: {column}")
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if not values.gt(0).any():
            failures.append(f"broker spec unavailable or non-positive: {column}")
    return failures


def data_quality_report(df: pd.DataFrame, timeframe_minutes: int = 5) -> dict:
    if df.empty:
        return {
            "rows": 0,
            "duplicate_times": 0,
            "missing_values": 0,
            "large_gaps": 0,
            "max_gap_minutes": 0.0,
            "broker_spec_failures": ["data is empty"],
        }
    result = df.copy()
    result["time"] = pd.to_datetime(result["time"], errors="coerce")
    times = result["time"].dropna().sort_values()
    diffs = times.diff().dropna()
    large_gaps = diffs[diffs > pd.Timedelta(minutes=timeframe_minutes * 3)]
    return {
        "rows": int(len(result)),
        "start": str(times.min()) if not times.empty else None,
        "end": str(times.max()) if not times.empty else None,
        "duplicate_times": int(result["time"].duplicated().sum()) if "time" in result else 0,
        "missing_values": int(result.isna().sum().sum()),
        "large_gaps": int(len(large_gaps)),
        "max_gap_minutes": float(diffs.max().total_seconds() / 60) if not diffs.empty else 0.0,
        "abnormal_spread_rows": int(pd.to_numeric(result.get("abnormal_spread", pd.Series(0, index=result.index)), errors="coerce").fillna(0).sum()),
        "broker_spec_failures": validate_broker_specs_from_frame(result),
    }


def threshold_has_fallback(thresholds: dict) -> bool:
    if thresholds.get("fallback_reason"):
        return True
    for key in ["buy_threshold_detail", "sell_threshold_detail"]:
        detail = thresholds.get(key)
        if isinstance(detail, dict) and detail.get("fallback_reason"):
            return True
    return False


def side_threshold_eligible(thresholds: dict, side: str) -> bool:
    detail = thresholds.get(f"{side.lower()}_threshold_detail")
    if not isinstance(detail, dict):
        return False
    return bool(detail.get("eligible", False)) and not bool(detail.get("fallback_reason"))


def strict_side_gate_failures(
    side_metrics: dict,
    allowed_sides: list[str],
    min_trades: int = STRICT_MIN_SIDE_TRADES,
    min_profit_factor: float = STRICT_MIN_PROFIT_FACTOR,
    max_drawdown_abs: float = STRICT_MAX_DRAWDOWN_ABS,
) -> list[str]:
    failures = []
    if not allowed_sides:
        return ["no allowed sides"]
    for side in allowed_sides:
        metrics = side_metrics.get(side, {})
        trades = int(metrics.get("total_trades", 0))
        net = float(metrics.get("net_profit", 0.0))
        pf = float(metrics.get("profit_factor", 0.0))
        drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
        ev = float(metrics.get("avg_pnl_per_trade", 0.0))
        if trades < min_trades:
            failures.append(f"{side} side trades below minimum: {trades} < {min_trades}")
        if net <= 0:
            failures.append(f"{side} side net profit is not positive: {net:.4f}")
        if pf < min_profit_factor:
            failures.append(f"{side} side profit factor below minimum: {pf:.3f} < {min_profit_factor:.3f}")
        if ev <= 0:
            failures.append(f"{side} side expected value is not positive: {ev:.6f}")
        if drawdown > max_drawdown_abs:
            failures.append(f"{side} side drawdown above maximum: {drawdown:.4f} > {max_drawdown_abs:.4f}")
        if not bool(metrics.get("pnl_money_available", False)):
            failures.append(f"{side} side money PnL unavailable")
    return failures


def canonical_meta_failures(meta: dict, artifact_dir: Path) -> list[str]:
    failures = []
    if not meta:
        return ["live_model_meta.json missing or empty"]
    if meta.get("gate_status") != "passed":
        failures.append("canonical gate status is not passed")
    if meta.get("source") not in {"walk_forward", "deployment"}:
        failures.append("canonical source must be walk_forward or deployment")
    if not meta.get("artifact_subdir") and meta.get("source") == "walk_forward":
        failures.append("walk-forward canonical metadata missing artifact_subdir")
    allowed_sides = meta.get("allowed_sides") or []
    if not allowed_sides:
        failures.append("canonical metadata has no allowed sides")
    if not artifact_dir.exists():
        failures.append(f"artifact directory not found: {artifact_dir}")
        return failures
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (artifact_dir / name).exists()]
    if missing:
        failures.append(f"artifact directory incomplete: {', '.join(missing)}")
    stored_hashes = meta.get("artifact_hashes")
    if stored_hashes:
        current = artifact_hashes(artifact_dir)
        mismatched = [name for name, value in stored_hashes.items() if value != current.get(name)]
        if mismatched:
            failures.append(f"artifact hashes stale: {', '.join(sorted(mismatched))}")
    return failures
