# Multi-Symbol M5 ML Trading Bot

Production-hardened Python research and execution framework for MetaTrader 5 machine-learning signals on M5 market data.

The project currently targets:

- `XAUUSD`
- `USTEC`
- `USTEC_X100`

It supports historical data download, feature engineering, ATR-barrier labeling, side-specific ML models, walk-forward validation, threshold search, weekly retraining, dry-run live signals, and guarded MT5 order execution.

## Current Production Policy

This repository is configured to **fail closed**.

- Real order placement is disabled by default for every symbol.
- Live trading requires a canonical `models/<SYMBOL>/live_model_meta.json`.
- Live metadata must pass strict artifact, side, broker-spec, and deployment gates.
- Missing, stale, incomplete, or mismatched artifacts block trading.
- The live bot defaults to dry-run unless `--trade` is explicitly passed.
- Non-demo accounts are refused unless `--allow-real-account` is explicitly passed.

Do not trade live capital until forward/demo results match backtest assumptions over several weeks.

## Main Capabilities

- MT5 OHLCV data download per symbol.
- 80+ technical, volatility, session, range, momentum, and volume-derived features.
- ATR barrier labels for `NO_TRADE`, `BUY`, and `SELL`.
- Separate BUY and SELL binary model support.
- XGBoost models with optional Optuna tuning.
- Probability calibration and ensemble training in the walk-forward pipeline.
- Backtest-aware threshold optimization.
- Rolling walk-forward validation with purged split boundaries.
- Weekly retraining with strict deployment gates.
- MT5 live dry-run signal logging and guarded order execution.
- Persistent live state for cooldown and duplicate-order protection.
- Data-quality and artifact-integrity reporting.

## Repository Layout

```text
src/
  config.py                  Global config and symbol definitions
  download_mt5_data.py        MT5 historical data downloader
  features.py                 Feature engineering
  labeling.py                 ATR barrier labels and diagnostics
  train.py                    Baseline model training
  walk_forward_pipeline.py    Institutional walk-forward pipeline
  walk_forward_backtest.py    OOS trade simulation and metrics
  walk_forward_threshold.py   Threshold search
  weekly_retrain.py           Latest-window retraining and deployment gate
  live_mt5.py                 MT5 live dry-run / execution runner
  production.py               Production artifact and gate validation helpers
tests/
  test_core.py                Unit and regression tests
data/
models/
reports/
logs/
```

Generated data, models, reports, logs, and trade exports are ignored by Git. Placeholder `.gitkeep` files keep the directory structure.

## Requirements

- Python 3.10+
- MetaTrader 5 desktop terminal
- Logged-in MT5 account
- Broker symbols visible in Market Watch

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure Symbols

Edit `src/config.py` if your broker uses different symbol names:

```python
SYMBOLS["XAUUSD"]["mt5_symbol"] = "XAUUSDm"
SYMBOLS["USTEC"]["mt5_symbol"] = "US100.cash"
SYMBOLS["USTEC_X100"]["mt5_symbol"] = "USTEC_x100"
```

Unknown symbols are intentionally rejected. Add new symbols explicitly to `SYMBOLS` before using them.

## Baseline Research Pipeline

Run from the repository root:

```powershell
python src/download_mt5_data.py --all
python src/features.py --all
python src/labeling.py --all
python src/train.py --all
python src/threshold_search.py --all
python src/backtest.py --all
```

## Walk-Forward Validation

Run institutional walk-forward evaluation per symbol:

```powershell
python src/walk_forward_pipeline.py --symbol XAUUSD --trials 50
python src/walk_forward_pipeline.py --symbol USTEC --trials 50
python src/walk_forward_pipeline.py --symbol USTEC_X100 --trials 50
```

Use `--resume` to continue a partially completed run:

```powershell
python src/walk_forward_pipeline.py --symbol USTEC_X100 --trials 50 --resume
```

The walk-forward pipeline writes reports under:

```text
reports/<SYMBOL>/walk_forward_<MODE>/
models/<SYMBOL>/walk_forward_<MODE>/
```

Only strict gate-passed results may update canonical live metadata:

```text
models/<SYMBOL>/live_model_meta.json
```

## Weekly Retraining

Run latest-window retraining:

```powershell
python src/weekly_retrain.py --all --trials 50 --force-download
```

Dry-run retraining without updating live metadata:

```powershell
python src/weekly_retrain.py --all --trials 50 --force-download --no-deploy
```

If deployment gates fail, artifacts and reports are still written, but `live_model_meta.json` is left unchanged.

## Live Dry-Run

Evaluate latest closed candles once:

```powershell
python src/live_mt5.py --all --once
```

Run continuous dry-run:

```powershell
python src/live_mt5.py --all
```

Signals are appended to:

```text
logs/live_signals.csv
```

Analyze live signal logs:

```powershell
python src/analyze_live_signals.py
```

## Real Order Execution

Real orders require all of the following:

- `--trade` flag.
- Demo account, unless `--allow-real-account` is explicitly passed.
- Symbol config `enabled_for_live=True`.
- Canonical `live_model_meta.json` with `gate_status="passed"`.
- Complete artifact hash validation.
- Eligible threshold with no fallback.
- Allowed side present in metadata.
- Broker tick size, tick value, and point available.
- Spread, session, regime, position, risk, cooldown, duplicate-order, and daily guard checks passing.

Command:

```powershell
python src/live_mt5.py --all --trade
```

Use this only after sustained demo validation.

## Validation

Run:

```powershell
python -m compileall src tests
python -m unittest discover tests
```

The test suite covers:

- Symbol validation.
- Feature and label pipeline behavior.
- Walk-forward split and gate behavior.
- Threshold search behavior.
- Canonical artifact loading.
- Weekly retrain gate behavior.
- Live fail-safe and duplicate-order logic.

## Production Readiness Notes

The system contains production hardening controls, but profitability is not guaranteed. Before live use:

- Re-run walk-forward validation using current broker data.
- Confirm money PnL is available for each symbol.
- Include realistic commission and slippage assumptions.
- Review side-specific OOS performance.
- Run live dry-run and demo forward tests for several weeks.
- Compare actual fills, spread, slippage, and PnL with backtest assumptions.

## Risk Disclaimer

This software is for research and engineering use. Algorithmic trading involves substantial risk. Backtests and walk-forward reports can overstate live performance due to broker conditions, spread, slippage, latency, rejection, news events, regime change, and model drift. Use at your own risk.
