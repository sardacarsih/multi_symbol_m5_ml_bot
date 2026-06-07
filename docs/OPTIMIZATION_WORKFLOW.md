# Panduan Penggunaan Optimization Workflow

Dokumen ini menjelaskan cara memakai `src/optimize_research_workflow.py` untuk mencari kombinasi labeling, feature set, walk-forward, dan weekly retrain terbaik.

Workflow ini mengorkestrasi skrip yang sudah ada:

1. `download_mt5_data.py`
2. `features.py`
3. `labeling.py`
4. `label_selection_experiment.py`
5. `feature_selection_experiment.py`
6. `walk_forward_pipeline.py`
7. `weekly_retrain.py`
8. summary ranking di `reports/optimization/`

## Persiapan

Jalankan dari root repository:

```powershell
cd D:\VSCODE\bottrading\multi_symbol_m5_ml_bot
.\.venv\Scripts\Activate.ps1
```

Pastikan MetaTrader 5 sudah terbuka, akun sudah login, dan symbol broker tersedia di Market Watch jika menjalankan stage yang mengunduh data.

## Quick Smoke Test

Gunakan smoke test sebelum full run. Mode ini memakai `--trials 0` dan membatasi eksperimen ke 3 cycle.

```powershell
python src/optimize_research_workflow.py --all --per-symbol --smoke --resume --skip-download --dry-run
```

Jika daftar command sudah benar, jalankan:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --smoke --resume --skip-download
```

## Full Optimization Run

Command utama untuk menjalankan semua tahap per symbol:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --trials 50 --resume --skip-download
```

Default stage yang dijalankan:

```text
prepare -> label -> feature -> walk_forward -> retrain -> summary
```

`--per-symbol` direkomendasikan untuk run panjang karena setiap symbol dieksekusi sebagai command terpisah. Jika satu symbol gagal, symbol lain dan artifact yang sudah selesai lebih mudah dilanjutkan dengan `--resume`.

## Menjalankan Tanpa Download Data

Gunakan `--skip-download` jika raw data lokal sudah tersedia di:

```text
data/raw/<SYMBOL>/<symbol>_m5_raw.csv
```

Dengan opsi ini, stage `prepare` hanya menjalankan:

```text
features.py -> labeling.py
```

dan tidak memanggil `download_mt5_data.py`.

Command rekomendasi tanpa download:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --trials 50 --resume --skip-download
```

Untuk hanya regenerate feature dan label dari data lokal:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --stages prepare --skip-download
```

Jangan pakai `--force-download` jika tujuannya memakai data lokal/cache.

## Menjalankan Symbol Tertentu

Satu symbol:

```powershell
python src/optimize_research_workflow.py --symbol XAUUSD --trials 50 --resume --skip-download
```

Contoh lengkap untuk `XAUUSD` tanpa download data:

```powershell
# Lihat command tanpa menjalankan
python src/optimize_research_workflow.py --symbol XAUUSD --per-symbol --smoke --resume --skip-download --dry-run

# Smoke test cepat
python src/optimize_research_workflow.py --symbol XAUUSD --per-symbol --smoke --resume --skip-download

# Full optimization
python src/optimize_research_workflow.py --symbol XAUUSD --per-symbol --trials 50 --resume --skip-download

# Summary hasil XAUUSD
python src/optimize_research_workflow.py --symbol XAUUSD --stages summary
```

Beberapa symbol, dipisah per symbol:

```powershell
python src/optimize_research_workflow.py --symbols XAUUSD USTEC --per-symbol --trials 50 --resume
```

Beberapa symbol dalam satu command gabungan untuk stage yang mendukung:

```powershell
python src/optimize_research_workflow.py --symbols XAUUSD USTEC --trials 50 --resume
```

## Menjalankan Stage Tertentu

Hanya siapkan data, feature, dan label baseline:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --stages prepare --skip-download
```

Hanya cari labeling terbaik:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --stages label --trials 50 --resume
```

Hanya cari feature set terbaik:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --stages feature --trials 50 --resume
```

Hanya final walk-forward:

```powershell
python src/optimize_research_workflow.py --all --stages walk_forward --trials 50 --resume
```

Hanya weekly retrain tanpa deploy live metadata:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --stages retrain --trials 50
```

Hanya buat summary dari report yang sudah ada:

```powershell
python src/optimize_research_workflow.py --all --stages summary
```

## Deploy vs No Deploy

Secara default, stage `retrain` menjalankan:

```text
weekly_retrain.py ... --no-deploy
```

Artinya artifact retrain dibuat, tetapi `models/<SYMBOL>/live_model_meta.json` tidak diperbarui.

Untuk mengizinkan update live metadata setelah gate lolos:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --stages retrain --trials 50 --deploy
```

Gunakan `--deploy` hanya setelah hasil walk-forward dan demo/dry-run sudah layak.

## Opsi Penting

| Opsi | Fungsi |
|---|---|
| `--all` | Jalankan semua symbol di `src/config.py`. |
| `--symbol XAUUSD` | Jalankan satu symbol. |
| `--symbols XAUUSD USTEC` | Jalankan beberapa symbol. |
| `--per-symbol` | Pecah multi-symbol menjadi command `--symbol` terpisah. Direkomendasikan. |
| `--stages ...` | Pilih stage: `prepare`, `label`, `feature`, `walk_forward`, `retrain`, `summary`. |
| `--trials 50` | Jumlah Optuna trials per cycle untuk full run. |
| `--smoke` | Pakai `--trials 0` dan `--max-cycles 3`. |
| `--resume` | Lanjutkan cycle yang belum selesai dan skip artifact lengkap. |
| `--force-download` | Paksa download data MT5 baru pada stage yang mendukung. |
| `--skip-download` | Lewati `download_mt5_data.py` dan pakai raw data lokal/cache. |
| `--dry-run` | Cetak command tanpa menjalankan. |
| `--deploy` | Izinkan weekly retrain update `live_model_meta.json`. |

## Custom Label dan Feature Experiment

Label feature set default adalah `core20`.

```powershell
python src/optimize_research_workflow.py --all --per-symbol --stages label --label-feature-set core30 --trials 50 --resume
```

Variant default adalah `all`. Untuk variant tertentu:

```powershell
python src/optimize_research_workflow.py --symbol XAUUSD --stages label --label-variants balanced,trend --trials 50 --resume
```

Feature set default:

```text
core20 core30 robust70
```

Untuk membatasi feature experiment:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --stages feature --feature-sets core20 core30 --trials 50 --resume
```

Jika `reports/<SYMBOL>/feature_importance.csv` tersedia, workflow otomatis menjalankan:

```text
importance_top_n --top-n 20
importance_top_n --top-n 30
```

Ubah top-N:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --stages feature --importance-top-n 15 20 30 --trials 50 --resume
```

## Output Report

Report utama workflow:

```text
reports/optimization/optimization_summary.md
reports/optimization/optimization_summary.csv
reports/optimization/optimization_summary.json
```

Report per tahap:

```text
reports/<SYMBOL>/label_selection/<FEATURE_SET>/<VARIANT>/
reports/<SYMBOL>/feature_selection/<FEATURE_SET>/
reports/<SYMBOL>/walk_forward*/
models/<SYMBOL>/walk_forward*/
models/<SYMBOL>/deployments/
```

## Cara Membaca Summary

`optimization_summary.md` memilih kandidat terbaik per symbol dan kategori:

- `label`: kandidat labeling terbaik.
- `feature`: feature set terbaik.
- `walk_forward`: hasil walk-forward terbaik yang ditemukan.

Kolom penting:

| Kolom | Arti |
|---|---|
| `Trades` | Jumlah trade OOS. Terlalu kecil berarti hasil kurang kuat. |
| `Net Profit` | Total profit OOS. Harus positif. |
| `PF` | Profit factor. Idealnya di atas 1, lebih baik di atas target internal. |
| `Max DD` | Max drawdown. Makin kecil absolutnya makin baik. |
| `BUY` / `SELL` | Distribusi trade per side. Hindari model yang hanya hidup di satu side kecuali memang disengaja. |
| `Score` | Ranking praktis dari profit factor, net profit, trade count, winrate, dan drawdown. |

Score hanya alat bantu ranking. Keputusan akhir tetap harus melihat stabilitas per cycle, equity curve, dan realistisnya asumsi spread/slippage.

## Urutan Rekomendasi Operasional

1. Jalankan dry-run command:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --smoke --resume --skip-download --dry-run
```

2. Jalankan smoke test:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --smoke --resume --skip-download
```

3. Jalankan full label dan feature experiment:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --stages prepare label feature --trials 50 --resume --skip-download
```

4. Review:

```text
reports/optimization/optimization_summary.md
reports/<SYMBOL>/label_selection/
reports/<SYMBOL>/feature_selection/
```

5. Jalankan walk-forward final:

```powershell
python src/optimize_research_workflow.py --all --stages walk_forward summary --trials 50 --resume
```

6. Jalankan retrain tanpa deploy:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --stages retrain summary --trials 50
```

7. Deploy hanya jika gate dan review manual layak:

```powershell
python src/optimize_research_workflow.py --all --per-symbol --stages retrain --trials 50 --deploy
```

## Catatan Risiko

Workflow ini membantu memilih konfigurasi riset terbaik, bukan menjamin profit live. Sebelum trading real, lakukan live dry-run dan demo forward test beberapa minggu, lalu bandingkan hasil aktual dengan asumsi backtest, terutama spread, slippage, commission, fill, dan perubahan regime market.
