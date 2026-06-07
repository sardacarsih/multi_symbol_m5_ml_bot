# Laporan Progres Project `multi_symbol_m5_ml_bot`

## Ringkasan Status

Project `multi_symbol_m5_ml_bot` sudah mencapai **Round 3** — institutional-grade walk-forward ML pipeline.

### Round 3 (Current) — Walk-Forward + Ensemble + Regime Filter

Round 3 mengimplementasikan pipeline evaluasi ML institutional-style:

- **Multi-symbol walk-forward** untuk `XAUUSD`, `USTEC`, dan `USTEC_X100`.
- **Walk-Forward Validation** dengan weekly retraining:
  - `XAUUSD` dan `USTEC`: 12-month training / 2-month validation / weekly OOS windows.
  - `USTEC_X100`: 8-month training / 1-month validation / weekly OOS windows.
- **Optuna Hyperparameter Tuning**: 50 trials per walk-forward cycle.
- **Threshold Optimization berbasis PF/DD**: ranking `pf_dd_ratio = PF / (abs(DD) + 0.05)` menggantikan pure precision ranking.
- **Regime Filter**: `vol_expansion` (ATR_5 > ATR_14) sebagai filter entry — hanya trade saat volatility ekspansif.
- **Ensemble + Calibration**: 3-seed XGBClassifier ensemble dengan sigmoid `CalibratedClassifierCV`.
- **Live walk-forward deployment**: live bot bisa membaca `models/<SYMBOL>/live_model_meta.json` untuk memakai ensemble model dari `models/<SYMBOL>/walk_forward/cycle_XXX/`.
- **Weekly retraining deployment**: `weekly_retrain.py` membuat deployment terbaru dari rolling train/validation window dan memakai 1 minggu terakhir sebagai OOS gate.
- **Bug fix**: `consecutive_red` candle pattern memakai `>` bukan `<` pada candle ke-3 — sudah diperbaiki.
- **Code cleanup**: Duplicate feature computation blocks (88 baris) dihapus dari `features.py`.

### Round 2 — Backtest-Aware Threshold + Session Filter

- Threshold search sekarang backtest-aware: threshold dipilih berdasarkan hasil simulasi trade, bukan hanya label precision.
- Entry filter berbasis session sudah ditambahkan per symbol.
- Live bot sekarang bisa menolak threshold fallback yang tidak eligible untuk live order.
- Backtest dan threshold search memakai logika session/cooldown/spread yang selaras.

### Round 1 — Baseline Pipeline

- Pipeline MT5 dasar: download, feature engineering, labeling, training, backtest, live dry-run.

Live trading real belum dijalankan. Bot tetap dry-run secara default dan hanya mengirim order real jika memakai `--trade`.

## Pipeline yang Sudah Dijalankan

### Round 1-2 Commands

```powershell
python src/download_mt5_data.py --all
python src/features.py --all
python src/labeling.py --all
python src/train.py --all
python src/train.py --global
python src/walk_forward.py --all
python src/threshold_search.py --all
python src/backtest.py --all
python src/live_mt5.py --all --once
python src/analyze_live_signals.py
```

### Round 3 Commands

```powershell
# Walk-Forward Pipeline (50 Optuna trials per cycle)
python src/walk_forward_pipeline.py --symbol XAUUSD --trials 50
python src/walk_forward_pipeline.py --symbol USTEC --trials 50
python src/walk_forward_pipeline.py --symbol USTEC_X100 --trials 50
```

Gunakan `--resume` untuk melanjutkan tanpa mengulang cycle yang sudah lengkap.

### Weekly Retrain Commands

```powershell
python src/weekly_retrain.py --all --trials 50 --force-download
```

Tanpa override manual, weekly retrain memakai:

| Symbol | Train | Validation | OOS Gate |
|---|---:|---:|---:|
| XAUUSD | 12 months | 2 months | 1 week |
| USTEC | 12 months | 2 months | 1 week |
| USTEC_X100 | 8 months | 1 month | 1 week |
 
Jika OOS gate gagal, `live_model_meta.json` tidak diubah.

## Symbols

| Symbol | MT5 Symbol | Status | Walk-Forward | Enabled Live |
|---|---|---|---|---|
| XAUUSD | XAUUSD | Round 3 ready | ✅ Configured | False |
| USTEC | USTEC | Round 3 ready | ✅ Configured | True |
| USTEC_X100 | USTEC_x100 | Round 3 ready | ✅ Configured | True |

## Data MT5

Download data berhasil untuk semua symbols:

| Symbol | Raw File Size | Notes |
|---|---:|---|
| XAUUSD | 10.4 MB | 24.4 months M5 data, 144,201 raw rows |
| USTEC | 10.1 MB | 24.4 months M5 data, 143,379 raw rows |
| USTEC_X100 | 4.9 MB | 11.6 months M5 data from broker, 69,086 raw rows |

## Walk-Forward Pipeline (Round 3)

### Configuration

Global config for `XAUUSD` and `USTEC`:

```python
walk_forward = True
training_window_months = 12
validation_months = 2
oos_months = "weekly"     # 1-week OOS windows
retrain_frequency = "weekly"
expanding_window = False  # rolling window
n_optuna_trials = 50
commission_per_lot = 0.0
slippage_points = 2.0
```

`USTEC_X100` uses symbol-specific override:

```python
training_window_months = 8
validation_months = 1
oos_months = "weekly"
retrain_frequency = "weekly"
expanding_window = False
n_optuna_trials = 50
commission_per_lot = 0.0
slippage_points = 2.0
```

### Pipeline Steps per Cycle

1. ✅ Data download dari MT5 (or use cached)
2. ✅ Feature engineering (86+ features)
3. ✅ ATR barrier labeling (3-class: NO_TRADE, BUY, SELL)
4. ✅ Walk-forward split generation
5. ✅ Data leakage prevention (purge boundary rows)
6. 🔄 Optuna hyperparameter tuning (50 trials per cycle)
7. 🔲 Ensemble training (3-seed XGBClassifier)
8. 🔲 Sigmoid calibration (CalibratedClassifierCV, cv="prefit")
9. 🔲 PF/DD threshold optimization on validation data
10. 🔲 Out-of-sample backtest with spread + slippage
11. 🔲 Shift window forward → repeat
12. 🔲 Aggregated OOS report generation

### Walk-Forward Splits

| Cycle | Train Period | Val Period | OOS Period |
|---|---|---|---|
| XAUUSD | 12 months rolling | 2 months rolling | weekly, 45 cycles |
| USTEC | 12 months rolling | 2 months rolling | weekly, 45 cycles |
| USTEC_X100 | 8 months rolling | 1 month rolling | weekly, 11 cycles |

### Key Enhancements

#### 1. Threshold Optimization berbasis PF/DD

```python
pf_dd_ratio = profit_factor / (abs(max_drawdown) + 0.05)
```

Eligibility criteria:
- `backtest_trades >= min_threshold_signals`
- `backtest_profit_factor >= 1.03`
- `backtest_net_profit > 0`

Ranking: eligible terlebih dahulu, lalu `pf_dd_ratio` descending.

#### 2. Regime Filter

```python
regime_filter_col = "vol_expansion"
# vol_expansion = (ATR_5 > ATR_14).astype(int)
# Only trade when volatility is expanding
```

Filter diterapkan di:
- `simulate_wf_trades()` — trade simulation
- `optimize_thresholds_wf()` — threshold search
- `live_mt5.py` — prediksi live menghormati `no_trade_zone` dari threshold walk-forward.

#### 3. Ensemble + Calibration

```python
seeds = [20260605, 12345, 98765]
# For each seed:
#   1. Train XGBClassifier with seed-specific random_state
#   2. Wrap with CalibratedClassifierCV(method="sigmoid", cv="prefit")
#   3. Calibrate on validation data
# Final prediction = mean(ensemble probabilities)
```

#### 4. Live Model Deployment Metadata

Setelah walk-forward selesai, pipeline memilih cycle live terbaik berdasarkan `pf_dd_ratio = PF / (abs(DD) + 0.05)` dan menulis:

```text
models/<SYMBOL>/live_model_meta.json
```

Jika file ini menunjuk ke `source = "walk_forward"`, live bot memuat `model.joblib`, `feature_columns.json`, dan `best_threshold.json` dari cycle tersebut. Jika file tidak ada, live bot tetap fallback ke model baseline `models/<SYMBOL>/<symbol>_m5_xgboost.joblib`.

## Hasil Round 2 (Sebelumnya)

### Threshold Results

| Symbol | Buy Threshold | Sell Threshold | Raw Signals | Backtest Trades | Net Profit | Profit Factor | Eligible |
|---|---:|---:|---:|---:|---:|---:|---|
| XAUUSD | 0.55 | 0.55 | 25 | 7 | -0.0344 | 0.7488 | False |
| USTEC | 0.55 | 0.56 | 588 | 133 | 3.3874 | 1.1370 | True |

### Backtest Results

| Symbol | Trades | Winrate | Profit Factor | Net Profit | Max Drawdown | Avg Holding |
|---|---:|---:|---:|---:|---:|---:|
| XAUUSD | 7 | 0.4286 | 0.7488 | -0.0344 | -0.1208 | 3.86 candle |
| USTEC | 133 | 0.4436 | 1.1370 | 3.3874 | -4.5051 | 3.41 candle |

## Bug Fixes (Round 3)

### `consecutive_red` Candle Pattern Fix

```diff
 result["consecutive_red"] = (
     (result["close"] < result["open"]) &
     (result["close"].shift(1) < result["open"].shift(1)) &
-    (result["close"].shift(2) > result["open"].shift(2))  # BUG: checking green!
+    (result["close"].shift(2) < result["open"].shift(2))  # FIX: checking red
 ).astype(int)
```

### Duplicate Feature Blocks Removed

`features.py` memiliki 88 baris kode duplikat (Multi-Timeframe Volatility Regime, Price Structure, Momentum Divergence, Volume Profile, Candle Patterns, Mean Reversion) yang dihitung dua kali. Sudah dihapus.

## File Output Penting

### Round 1-2
- Raw data: `data/raw/XAUUSD/`, `data/raw/USTEC/`
- Processed data: `data/processed/XAUUSD/`, `data/processed/USTEC/`
- Model per-symbol: `models/XAUUSD/`, `models/USTEC/`
- Global model: `models/global/`
- Reports per-symbol: `reports/XAUUSD/`, `reports/USTEC/`
- Portfolio summary: `reports/portfolio_backtest_summary.csv`
- Live signal log: `logs/live_signals.csv`

### Round 3
- Raw data: `data/raw/<SYMBOL>/<symbol>_m5_raw.csv`
- Walk-forward models: `models/<SYMBOL>/walk_forward/cycle_XXX/`
  - `model.joblib` — 3-model calibrated ensemble
  - `tuned_params.json` — Optuna best hyperparameters
  - `best_threshold.json` — PF/DD optimized thresholds
  - `feature_columns.json` — feature list
  - `cycle_meta.json` — split metadata
- Walk-forward reports: `reports/<SYMBOL>/walk_forward/`
  - `walk_forward_summary.csv` — all cycles summary
  - `walk_forward_report.md` — human-readable report
  - `aggregated_oos_trades.csv` — all OOS trades concatenated
  - `aggregated_oos_equity.png` — equity curve
  - `monthly_returns.csv` — monthly PnL breakdown
- Weekly retrain deployments: `models/<SYMBOL>/deployments/<timestamp>/`
  - `model.joblib` — calibrated ensemble terbaru
  - `best_threshold.json` — threshold hasil validation terbaru
  - `oos_metrics.json` — OOS gate metrics minggu terakhir
  - `retrain_meta.json` — split dan metadata deployment

## Verifikasi Teknis

Compile check:

```powershell
python -m compileall src tests
```

Import check (semua module):

```powershell
python -c "from walk_forward_pipeline import run_pipeline; print('OK')"
```

## Risiko dan Batasan

- Walk-forward pipeline belum final sampai setiap symbol yang dipakai live selesai semua cycle dan aggregated report.
- XAUUSD diset `enabled_for_live = False` sampai ada walk-forward result yang mendukung.
- Profit factor portfolio round 2 masih tipis (1.1349).
- Backtest belum membuktikan performa live. Spread, slippage, tick value, stop level, dan session broker bisa mengubah hasil.
- Live order real belum diuji dan tidak direkomendasikan untuk akun real.
- Ensemble + calibration menambah complexity — perlu monitoring overfitting.
- Regime filter bisa mengurangi jumlah trade secara signifikan.

## Rekomendasi Langkah Berikutnya

1. Jalankan walk-forward untuk `XAUUSD`, `USTEC`, dan `USTEC_X100` dengan command Round 3 terbaru.
2. Evaluasi aggregated OOS performance dari setiap walk-forward report.
3. Jika walk-forward profitable, gunakan `live_model_meta.json` hasil pipeline untuk live dry-run.
4. Bandingkan performa walk-forward vs single-split backtest round 2.
5. Jalankan `weekly_retrain.py --all --trials 50 --force-download` setiap minggu setelah market close.
6. Pertimbangkan menambahkan symbol lain untuk diversifikasi portfolio.
7. Setelah dry-run stabil, uji `--trade` hanya di demo account.
