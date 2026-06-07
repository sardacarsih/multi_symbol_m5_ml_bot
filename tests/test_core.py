import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from calibration_utils import calibrate_prefit_classifier
from backtest_diagnostics import build_threshold_fingerprint, signal_diagnostics, threshold_fingerprint_status
from features import add_features
from labeling import label_diagnostics, atr_barrier_labels
from analyze_live_signals import session_from_hour
from live_mt5 import decide_live_signal, evaluate_symbol, load_artifacts, order_dedupe_key, predict_proba_model, validate_trade_account
from risk import calculate_atr_sl_tp, check_spread_filter, normalize_lot
from strategy_filters import session_allowed, simulate_exit
from train import sample_weights
from threshold_search import decide_signal
from utils import symbol_to_filename
from walk_forward_backtest import run_oos_backtest
from walk_forward_pipeline import aggregate_oos_metrics, binary_target, format_duration, generate_reports, generate_walk_forward_comparison_report, is_cycle_complete, run_pipeline, select_live_deploy_cycle, walk_forward_gate_failures, walk_forward_output_paths
from walk_forward_threshold import optimize_thresholds_wf, threshold_quality_columns
from weekly_retrain import deployment_gate_failures, effective_retrain_config, generate_latest_retrain_split, run_weekly_retrain


class DummyModel:
    def __init__(self, proba):
        self.proba = np.array(proba, dtype=float)

    def predict_proba(self, rows):
        return np.tile(self.proba, (len(rows), 1))


class CoreBehaviorTests(unittest.TestCase):
    def test_format_duration_uses_hours_minutes_seconds(self):
        self.assertEqual(format_duration(0), "00:00:00.00")
        self.assertEqual(format_duration(1.234), "00:00:01.23")
        self.assertEqual(format_duration(3661.987), "01:01:01.99")

    def test_symbols_dict_dynamic_defaults(self):
        from config import SYMBOLS
        self.assertIn("XAUUSD", SYMBOLS)
        self.assertEqual(SYMBOLS["XAUUSD"]["mt5_symbol"], "XAUUSD")

        new_symbol = "EURUSD"
        self.assertNotIn(new_symbol, SYMBOLS)
        with self.assertRaisesRegex(KeyError, "Unknown symbol"):
            _ = SYMBOLS[new_symbol]

    def test_ustec_x100_uses_robust_label_config(self):
        from config import SYMBOLS
        cfg = SYMBOLS["USTEC_X100"]
        self.assertEqual(cfg["lookahead_candles"], 9)
        self.assertEqual(cfg["label_tp_atr_mult"], 1.2)
        self.assertEqual(cfg["label_sl_atr_mult"], 1.0)
        self.assertEqual(cfg["live_tp_atr_mult"], 1.6)
        self.assertEqual(cfg["live_sl_atr_mult"], 1.1)

    def test_xauusd_threshold_search_includes_lower_probability_grid(self):
        from config import SYMBOLS
        cfg = SYMBOLS["XAUUSD"]
        values = np.round(
            np.arange(
                float(cfg["threshold_search_min"]),
                float(cfg["threshold_search_max"]) + float(cfg["threshold_search_step"]) / 2,
                float(cfg["threshold_search_step"]),
            ),
            2,
        )

        self.assertIn(0.50, values)
        self.assertIn(0.52, values)
        self.assertEqual(float(values.min()), 0.50)

    def test_symbol_to_filename(self):
        self.assertEqual(symbol_to_filename("US100.cash"), "us100_cash")
        self.assertEqual(symbol_to_filename("XAUUSDm"), "xauusdm")


    def test_decide_signal(self):
        self.assertEqual(decide_signal(0.70, 0.20, 0.66, 0.66), 1)
        self.assertEqual(decide_signal(0.20, 0.70, 0.66, 0.66), 2)
        self.assertEqual(decide_signal(0.65, 0.64, 0.66, 0.66), 0)

    def test_threshold_fingerprint_detects_stale_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model_path = base / "model.joblib"
            labeled_path = base / "labeled.csv"
            model_path.write_text("model-v1", encoding="utf-8")
            labeled_path.write_text("data-v1", encoding="utf-8")
            test_df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=3, freq="5min")})
            fingerprint = build_threshold_fingerprint(model_path, labeled_path, ["a", "b"], test_df)

            self.assertEqual(threshold_fingerprint_status({"artifact_fingerprint": fingerprint}, fingerprint), "valid")
            self.assertEqual(threshold_fingerprint_status({}, fingerprint), "missing")

            changed = dict(fingerprint)
            changed["test_row_count"] = 4
            self.assertEqual(threshold_fingerprint_status({"artifact_fingerprint": fingerprint}, changed), "stale")

    def test_signal_diagnostics_counts_threshold_and_filters(self):
        df = pd.DataFrame(
            [
                {"spread_to_atr": 0.01, "is_london_session": 1},
                {"spread_to_atr": 0.01, "is_london_session": 0},
                {"spread_to_atr": 0.30, "is_london_session": 1},
                {"spread_to_atr": 0.01, "is_london_session": 1},
            ]
        )
        probabilities = np.array(
            [
                [0.10, 0.70, 0.20],
                [0.10, 0.72, 0.20],
                [0.10, 0.20, 0.75],
                [0.80, 0.10, 0.10],
            ]
        )
        cfg = {"allowed_entry_sessions": ["is_london_session"], "max_spread_to_atr": 0.2}
        diagnostics = signal_diagnostics(df, probabilities, cfg, 0.65, 0.65, decide_signal, final_trades=1)

        self.assertEqual(diagnostics["signals_raw"], 3)
        self.assertEqual(diagnostics["signals_after_session"], 2)
        self.assertEqual(diagnostics["signals_after_spread"], 1)
        self.assertEqual(diagnostics["final_trades"], 1)
        self.assertAlmostEqual(diagnostics["prob_buy_max"], 0.72)
        self.assertAlmostEqual(diagnostics["prob_sell_max"], 0.75)

    def test_live_signal_respects_no_trade_zone(self):
        cfg = {"buy_threshold": 0.55, "sell_threshold": 0.55}
        thresholds = {"buy_threshold": 0.55, "sell_threshold": 0.55, "no_trade_zone": 0.10}
        self.assertEqual(decide_live_signal(0.60, 0.55, thresholds, cfg), 0)
        self.assertEqual(decide_live_signal(0.70, 0.55, thresholds, cfg), 1)

    def test_live_signal_respects_allowed_sides(self):
        cfg = {"buy_threshold": 0.55, "sell_threshold": 0.55}
        thresholds = {"buy_threshold": 0.55, "sell_threshold": 0.55, "allowed_sides": ["BUY"]}
        self.assertEqual(decide_live_signal(0.70, 0.20, thresholds, cfg), 1)
        self.assertEqual(decide_live_signal(0.20, 0.70, thresholds, cfg), 0)

    def test_binary_target_maps_side_vs_rest(self):
        labels = pd.Series([0, 1, 2, 1, 0, 2])
        self.assertEqual(binary_target(labels, "BUY").tolist(), [0, 1, 0, 1, 0, 0])
        self.assertEqual(binary_target(labels, "SELL").tolist(), [0, 0, 1, 0, 0, 1])

    def test_live_signal_resolves_separate_side_conflicts_with_no_trade_zone(self):
        cfg = {"buy_threshold": 0.55, "sell_threshold": 0.55}
        thresholds = {"buy_threshold": 0.55, "sell_threshold": 0.55, "no_trade_zone": 0.10, "allowed_sides": ["BUY", "SELL"]}
        self.assertEqual(decide_live_signal(0.60, 0.56, thresholds, cfg), 0)
        self.assertEqual(decide_live_signal(0.60, 0.72, thresholds, cfg), 2)

    def test_predict_proba_supports_single_model_and_ensemble(self):
        rows = pd.DataFrame({"x": [1.0, 2.0]})
        model = {
            "side_training_mode": "separate",
            "models": {
                "BUY": [DummyModel([0.2, 0.8]), DummyModel([0.4, 0.6])],
                "SELL": DummyModel([0.3, 0.7]),
            },
        }
        self.assertTrue(np.allclose(predict_proba_model(model, rows)[0], [0.3, 0.7, 0.7]))
        with self.assertRaisesRegex(RuntimeError, "requires separate BUY/SELL"):
            predict_proba_model(DummyModel([0.1, 0.8, 0.1]), rows)

    def test_prefit_calibration_helper_accepts_validation_data(self):
        X, y = make_classification(
            n_samples=90,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            n_classes=3,
            random_state=20260605,
        )
        estimator = LogisticRegression(max_iter=500).fit(X[:60], y[:60])
        calibrated = calibrate_prefit_classifier(estimator, X[60:], y[60:])
        proba = calibrated.predict_proba(X[60:65])
        self.assertEqual(proba.shape, (5, 3))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_load_artifacts_rejects_legacy_walk_forward_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            artifact_dir = base / "USTEC_X100" / "walk_forward" / "cycle_002"
            artifact_dir.mkdir(parents=True)
            joblib.dump([DummyModel([0.2, 0.7, 0.1])], artifact_dir / "model.joblib")
            (artifact_dir / "feature_columns.json").write_text(json.dumps(["atr_14"]), encoding="utf-8")
            (artifact_dir / "best_threshold.json").write_text(json.dumps({"buy_threshold": 0.55}), encoding="utf-8")
            meta_path = base / "USTEC_X100" / "live_model_meta.json"
            meta_path.write_text(
                json.dumps({
                    "source": "walk_forward",
                    "gate_status": "passed",
                    "cycle": "cycle_002",
                    "artifact_subdir": "walk_forward",
                    "allowed_sides": ["BUY"],
                }),
                encoding="utf-8",
            )

            with patch("live_mt5.MODELS_DIR", base):
                with self.assertRaisesRegex(RuntimeError, "canonical live metadata failed validation"):
                    load_artifacts("USTEC_X100")

    def test_load_artifacts_merges_walk_forward_allowed_sides(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            artifact_dir = base / "USTEC_X100" / "walk_forward" / "cycle_002"
            artifact_dir.mkdir(parents=True)
            joblib.dump([DummyModel([0.2, 0.8])], artifact_dir / "buy_model.joblib")
            joblib.dump([DummyModel([0.3, 0.7])], artifact_dir / "sell_model.joblib")
            (artifact_dir / "feature_columns.json").write_text(json.dumps(["atr_14"]), encoding="utf-8")
            (artifact_dir / "buy_threshold.json").write_text(json.dumps({"threshold": 0.55}), encoding="utf-8")
            (artifact_dir / "sell_threshold.json").write_text(json.dumps({"threshold": 0.65}), encoding="utf-8")
            (artifact_dir / "side_model_meta.json").write_text(json.dumps({"side_training_mode": "separate"}), encoding="utf-8")
            (artifact_dir / "best_threshold.json").write_text(json.dumps({"buy_threshold": 0.55, "sell_threshold": 0.65}), encoding="utf-8")
            meta_path = base / "USTEC_X100" / "live_model_meta.json"
            meta_path.write_text(
                json.dumps({
                    "source": "walk_forward",
                    "gate_status": "passed",
                    "cycle": "cycle_002",
                    "artifact_subdir": "walk_forward",
                    "allowed_sides": ["BUY"],
                }),
                encoding="utf-8",
            )

            with patch("live_mt5.MODELS_DIR", base):
                _, _, thresholds = load_artifacts("USTEC_X100")

            self.assertEqual(thresholds["allowed_sides"], ["BUY"])

    def test_load_artifacts_supports_separate_side_walk_forward_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            artifact_dir = base / "USTEC_X100" / "walk_forward" / "cycle_002"
            artifact_dir.mkdir(parents=True)
            joblib.dump([DummyModel([0.2, 0.8])], artifact_dir / "buy_model.joblib")
            joblib.dump([DummyModel([0.3, 0.7])], artifact_dir / "sell_model.joblib")
            (artifact_dir / "feature_columns.json").write_text(json.dumps(["atr_14"]), encoding="utf-8")
            (artifact_dir / "buy_threshold.json").write_text(json.dumps({"threshold": 0.60}), encoding="utf-8")
            (artifact_dir / "sell_threshold.json").write_text(json.dumps({"threshold": 0.70}), encoding="utf-8")
            (artifact_dir / "side_model_meta.json").write_text(json.dumps({"side_training_mode": "separate"}), encoding="utf-8")
            (artifact_dir / "best_threshold.json").write_text(
                json.dumps({"side_training_mode": "separate", "buy_threshold": 0.60, "sell_threshold": 0.70}),
                encoding="utf-8",
            )
            (base / "USTEC_X100" / "live_model_meta.json").write_text(
                json.dumps({
                    "source": "walk_forward",
                    "gate_status": "passed",
                    "cycle": "cycle_002",
                    "artifact_subdir": "walk_forward",
                    "allowed_sides": ["BUY"],
                }),
                encoding="utf-8",
            )

            with patch("live_mt5.MODELS_DIR", base):
                model, columns, thresholds = load_artifacts("USTEC_X100")
                proba = predict_proba_model(model, pd.DataFrame({"atr_14": [1.0]}))

            self.assertEqual(model["side_training_mode"], "separate")
            self.assertEqual(columns, ["atr_14"])
            self.assertEqual(thresholds["allowed_sides"], ["BUY"])
            self.assertTrue(np.allclose(proba[0], [0.2, 0.8, 0.7]))


    def test_load_artifacts_supports_separate_side_deployment_live_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            artifact_dir = base / "USTEC_X100" / "deployments" / "20260606_100000"
            artifact_dir.mkdir(parents=True)
            joblib.dump([DummyModel([0.2, 0.8])], artifact_dir / "buy_model.joblib")
            joblib.dump([DummyModel([0.3, 0.7])], artifact_dir / "sell_model.joblib")
            (artifact_dir / "feature_columns.json").write_text(json.dumps(["atr_14"]), encoding="utf-8")
            (artifact_dir / "buy_threshold.json").write_text(json.dumps({"threshold": 0.57}), encoding="utf-8")
            (artifact_dir / "sell_threshold.json").write_text(json.dumps({"threshold": 0.67}), encoding="utf-8")
            (artifact_dir / "side_model_meta.json").write_text(json.dumps({"side_training_mode": "separate"}), encoding="utf-8")
            (artifact_dir / "best_threshold.json").write_text(json.dumps({"buy_threshold": 0.57, "sell_threshold": 0.67}), encoding="utf-8")
            meta_path = base / "USTEC_X100" / "live_model_meta.json"
            meta_path.write_text(
                json.dumps({
                    "source": "deployment",
                    "gate_status": "passed",
                    "deployment": "20260606_100000",
                    "artifact_subdir": "deployments/20260606_100000",
                    "allowed_sides": ["BUY"],
                }),
                encoding="utf-8",
            )

            with patch("live_mt5.MODELS_DIR", base):
                model, columns, thresholds = load_artifacts("USTEC_X100")

            self.assertEqual(model["side_training_mode"], "separate")
            self.assertEqual(columns, ["atr_14"])
            self.assertEqual(thresholds["buy_threshold"], 0.57)

    def test_load_artifacts_without_live_meta_requires_retrain(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("live_mt5.MODELS_DIR", Path(tmp)):
                with self.assertRaisesRegex(RuntimeError, "retrain the symbol"):
                    load_artifacts("USTEC_X100")

    def test_sample_weights_raise_minority_classes(self):
        y = pd.Series([0, 0, 0, 0, 1, 2])
        weights = sample_weights(y)
        self.assertIsNotNone(weights)
        self.assertGreater(weights.iloc[4], weights.iloc[0])
        self.assertGreater(weights.iloc[5], weights.iloc[0])

    def test_feature_columns_excludes_redundant_single_symbol_features(self):
        from train import feature_columns
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=3, freq="5min"),
                "label": [0, 1, 2],
                "point": [0.01, 0.01, 0.01],
                "symbol_id": [2, 2, 2],
                "symbol_USTEC_X100": [1, 1, 1],
                "ema_20": [100.0, 101.0, 102.0],
                "highest_high_24": [103.0, 104.0, 105.0],
                "lowest_low_24": [99.0, 100.0, 101.0],
                "close_to_ema20": [0.01, 0.02, 0.03],
                "ema20_slope": [0.001, 0.002, 0.003],
                "dist_from_high_24": [-1.0, -0.5, -0.2],
                "atr_14_to_28": [1.1, 1.2, 1.3],
            }
        )

        columns = feature_columns(df)

        self.assertNotIn("point", columns)
        self.assertNotIn("symbol_id", columns)
        self.assertNotIn("symbol_USTEC_X100", columns)
        self.assertNotIn("ema_20", columns)
        self.assertNotIn("highest_high_24", columns)
        self.assertNotIn("lowest_low_24", columns)
        self.assertIn("close_to_ema20", columns)
        self.assertIn("ema20_slope", columns)
        self.assertIn("dist_from_high_24", columns)
        self.assertIn("atr_14_to_28", columns)

    def test_feature_selection_core_sets(self):
        from feature_selection_experiment import CORE20_FEATURES, CORE30_FEATURES, select_features

        self.assertEqual(len(CORE20_FEATURES), 20)
        self.assertEqual(len(CORE30_FEATURES), 30)
        df = pd.DataFrame({feature: [1.0, 2.0] for feature in CORE30_FEATURES})
        selected_20, skipped_20, requested_20 = select_features("core20", df, CORE30_FEATURES, "XAUUSD")
        selected_30, skipped_30, requested_30 = select_features("core30", df, CORE30_FEATURES, "XAUUSD")

        self.assertEqual(selected_20, CORE20_FEATURES)
        self.assertEqual(requested_20, CORE20_FEATURES)
        self.assertEqual(skipped_20, [])
        self.assertEqual(selected_30, CORE30_FEATURES)
        self.assertEqual(requested_30, CORE30_FEATURES)
        self.assertEqual(skipped_30, [])

    def test_filter_valid_features_records_missing_and_invalid(self):
        from feature_selection_experiment import filter_valid_features

        df = pd.DataFrame({"atr_14": [1.0], "rsi_14": [50.0], "not_trainable": [1.0]})
        selected, skipped = filter_valid_features(
            ["atr_14", "missing", "not_trainable", "rsi_14", "atr_14"],
            ["atr_14", "rsi_14"],
            df,
        )

        self.assertEqual(selected, ["atr_14", "rsi_14"])
        self.assertEqual(skipped, ["missing", "not_trainable"])

    def test_feature_selection_output_dir_naming(self):
        from feature_selection_experiment import experiment_report_dir

        self.assertEqual(
            experiment_report_dir("XAUUSD", "core20"),
            ROOT / "reports" / "XAUUSD" / "feature_selection" / "core20",
        )
        self.assertEqual(
            experiment_report_dir("XAUUSD", "importance_top_n", top_n=25),
            ROOT / "reports" / "XAUUSD" / "feature_selection" / "importance_top_25",
        )
        self.assertEqual(experiment_report_dir("XAUUSD", "core20", output_dir="custom"), Path("custom"))

    def test_feature_selection_recommendation_logic(self):
        from feature_selection_experiment import recommendation_for_metrics

        baseline = {"total_trades": 80, "net_profit": 1.0, "profit_factor": 1.2, "max_drawdown": -0.8}
        promote = {"total_trades": 60, "net_profit": 1.3, "profit_factor": 1.25, "max_drawdown": -0.7}
        low_trades = {"total_trades": 10, "net_profit": 2.0, "profit_factor": 2.0, "max_drawdown": -0.2}
        worse_drawdown = {"total_trades": 60, "net_profit": 1.3, "profit_factor": 1.25, "max_drawdown": -1.1}

        self.assertEqual(recommendation_for_metrics(promote, baseline), "PROMOTE_REDUCED_FEATURES")
        self.assertEqual(recommendation_for_metrics(promote, None), "KEEP_BASELINE")
        self.assertEqual(recommendation_for_metrics(low_trades, baseline), "KEEP_BASELINE")
        self.assertEqual(recommendation_for_metrics(worse_drawdown, baseline), "KEEP_BASELINE")

    def test_atr_levels(self):
        buy = calculate_atr_sl_tp(100.0, 2.0, "BUY", 1.2, 1.8)
        self.assertAlmostEqual(buy.sl, 97.6)
        self.assertAlmostEqual(buy.tp, 103.6)
        sell = calculate_atr_sl_tp(100.0, 2.0, "SELL", 1.0, 1.5)
        self.assertAlmostEqual(sell.sl, 102.0)
        self.assertAlmostEqual(sell.tp, 97.0)

    def test_lot_and_spread_filters(self):
        self.assertEqual(normalize_lot(0.037, 0.01, 1.0, 0.01), 0.03)
        self.assertTrue(check_spread_filter(0.19, 0.20))
        self.assertFalse(check_spread_filter(0.21, 0.20))

    def test_session_filter(self):
        row = pd.Series({"is_london_session": 0, "is_newyork_session": 1})
        self.assertTrue(session_allowed(row, {"allowed_entry_sessions": ["is_newyork_session"]}))
        self.assertFalse(session_allowed(row, {"allowed_entry_sessions": ["is_london_session"]}))
        self.assertTrue(session_allowed(row, {}))

    def test_session_from_hour(self):
        self.assertEqual(session_from_hour(3), "asia")
        self.assertEqual(session_from_hour(9), "london")
        self.assertEqual(session_from_hour(14), "london_newyork_overlap")
        self.assertEqual(session_from_hour(20), "newyork")

    def test_validate_trade_account_blocks_real(self):
        class Account:
            trade_mode = 2
            trade_allowed = True
            trade_expert = True

        class Mt5:
            ACCOUNT_TRADE_MODE_DEMO = 0

            @staticmethod
            def account_info():
                return Account()

        with self.assertRaises(RuntimeError):
            validate_trade_account(Mt5(), allow_real=False)
        validate_trade_account(Mt5(), allow_real=True)

    def test_disabled_symbol_does_not_send_order(self):
        class Info:
            volume_min = 0.01
            volume_max = 1.0
            volume_step = 0.01
            trade_tick_value = 1.0
            trade_tick_size = 0.01

        class Tick:
            ask = 101.0
            bid = 100.0

        class Mt5:
            def symbol_info(self, _symbol):
                return Info()

            def symbol_info_tick(self, _symbol):
                return Tick()

            def positions_get(self):
                return []

        features = pd.DataFrame(
            [{"close": 100.0, "atr_14": 2.0, "spread": 0.01, "spread_to_atr": 0.01}]
        )
        side_model = {
            "side_training_mode": "separate",
            "models": {
                "BUY": DummyModel([0.2, 0.8]),
                "SELL": DummyModel([0.9, 0.1]),
            },
        }

        with patch("live_mt5.ensure_symbol_visible"), \
             patch("live_mt5.load_artifacts", return_value=(side_model, ["close", "atr_14", "spread_to_atr"], {"buy_threshold": 0.55, "sell_threshold": 0.55, "eligible": True})), \
             patch("live_mt5.fetch_recent_candles", return_value=pd.DataFrame({"close": [100.0, 101.0]})), \
             patch("live_mt5.add_features", return_value=features), \
             patch("live_mt5.append_csv_row") as append_row, \
             patch("live_mt5.send_order") as send_order:
            evaluate_symbol(Mt5(), "XAUUSD", trade=True)

        self.assertFalse(send_order.called)
        self.assertEqual(append_row.call_args.args[1]["reason"], "SYMBOL_LIVE_DISABLED")

    def test_live_regime_filter_blocks_order(self):
        class Info:
            volume_min = 0.01
            volume_max = 1.0
            volume_step = 0.01
            trade_tick_value = 1.0
            trade_tick_size = 0.01
            trade_stops_level = 0
            point = 0.01

        class Tick:
            ask = 101.0
            bid = 100.0

        class Mt5:
            def symbol_info(self, _symbol):
                return Info()

            def symbol_info_tick(self, _symbol):
                return Tick()

            def positions_get(self):
                return []

            def history_deals_get(self, _start, _end):
                return []

        features = pd.DataFrame(
            [{"close": 100.0, "atr_14": 2.0, "spread": 0.01, "spread_to_atr": 0.01, "vol_expansion": 0}]
        )
        side_model = {
            "side_training_mode": "separate",
            "models": {
                "BUY": DummyModel([0.2, 0.8]),
                "SELL": DummyModel([0.9, 0.1]),
            },
        }
        from config import SYMBOLS as CONFIG_SYMBOLS
        live_symbols = {key: value.copy() for key, value in CONFIG_SYMBOLS.items()}
        live_symbols["USTEC_X100"]["enabled_for_live"] = True

        with patch("live_mt5.ensure_symbol_visible"), \
             patch("live_mt5.SYMBOLS", live_symbols), \
             patch("live_mt5.load_artifacts", return_value=(side_model, ["close", "atr_14", "spread_to_atr", "vol_expansion"], {"buy_threshold": 0.55, "sell_threshold": 0.55, "eligible": True, "_live_meta": {"gate_status": "passed"}})), \
             patch("live_mt5.fetch_recent_candles", return_value=pd.DataFrame({"close": [100.0, 101.0]})), \
             patch("live_mt5.add_features", return_value=features), \
             patch("live_mt5.append_csv_row") as append_row, \
             patch("live_mt5.send_order") as send_order:
            evaluate_symbol(Mt5(), "USTEC_X100", trade=True)

        self.assertFalse(send_order.called)
        self.assertEqual(append_row.call_args.args[1]["reason"], "REGIME_FILTER")

    def test_walk_forward_cycle_complete_requires_all_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            report_dir = Path(tmp) / "report"
            model_dir.mkdir()
            report_dir.mkdir()
            self.assertFalse(is_cycle_complete(model_dir, report_dir))
            for name in [
                "buy_model.joblib",
                "sell_model.joblib",
                "buy_threshold.json",
                "sell_threshold.json",
                "best_threshold.json",
                "feature_columns.json",
                "cycle_meta.json",
                "side_model_meta.json",
            ]:
                (model_dir / name).write_text("{}", encoding="utf-8")
            (report_dir / "oos_metrics.json").write_text("{}", encoding="utf-8")
            self.assertFalse(is_cycle_complete(model_dir, report_dir))
            (report_dir / "oos_backtest.csv").write_text("", encoding="utf-8")
            self.assertTrue(is_cycle_complete(model_dir, report_dir))

    def test_walk_forward_output_paths_are_separated_by_tuning_mode(self):
        non_tuned = walk_forward_output_paths("USTEC_X100", 0)
        tuned_50 = walk_forward_output_paths("USTEC_X100", 50)
        tuned_10 = walk_forward_output_paths("USTEC_X100", 10)

        self.assertEqual(non_tuned["mode"], "non_tuned")
        self.assertEqual(Path(non_tuned["report_dir"]).name, "walk_forward_non_tuned")
        self.assertEqual(Path(non_tuned["model_dir"]).name, "walk_forward_non_tuned")
        self.assertEqual(Path(non_tuned["live_meta_path"]).name, "live_model_meta.json")

        self.assertEqual(tuned_50["mode"], "tuned_50")
        self.assertEqual(Path(tuned_50["report_dir"]).name, "walk_forward_tuned_50")
        self.assertEqual(Path(tuned_50["model_dir"]).name, "walk_forward_tuned_50")
        self.assertEqual(Path(tuned_50["live_meta_path"]).name, "live_model_meta.json")

        self.assertEqual(tuned_10["mode"], "tuned_10")
        self.assertEqual(Path(tuned_10["report_dir"]).name, "walk_forward_tuned_10")

    def test_run_pipeline_uses_mode_specific_walk_forward_paths(self):
        raw_rows = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=3, freq="5min"),
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
            }
        )
        labeled_rows = pd.DataFrame(
            {
                "time": pd.to_datetime([]),
                "label": pd.Series(dtype=int),
            }
        )
        split = {
            "cycle": 1,
            "train_start": pd.Timestamp("2026-01-01"),
            "train_end": pd.Timestamp("2026-02-01"),
            "val_start": pd.Timestamp("2026-02-01"),
            "val_end": pd.Timestamp("2026-03-01"),
            "oos_start": pd.Timestamp("2026-03-01"),
            "oos_end": pd.Timestamp("2026-04-01"),
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            processed_dir = tmp_path / "processed"
            models_dir = tmp_path / "models"
            reports_dir = tmp_path / "reports"
            symbol_raw_dir = raw_dir / "USTEC_X100"
            symbol_processed_dir = processed_dir / "USTEC_X100"
            symbol_raw_dir.mkdir(parents=True)
            symbol_processed_dir.mkdir(parents=True)
            raw_rows.to_csv(symbol_raw_dir / "ustec_x100_m5_raw.csv", index=False)

            with patch("walk_forward_pipeline.RAW_DATA_DIR", raw_dir), \
                 patch("walk_forward_pipeline.PROCESSED_DATA_DIR", processed_dir), \
                 patch("walk_forward_pipeline.MODELS_DIR", models_dir), \
                 patch("walk_forward_pipeline.REPORTS_DIR", reports_dir), \
                 patch("walk_forward_pipeline.SYMBOLS", {"USTEC_X100": {"lookahead_candles": 3, "walk_forward": {}}}), \
                 patch("walk_forward_pipeline.add_features", return_value=raw_rows), \
                 patch("walk_forward_pipeline.atr_barrier_labels", return_value=labeled_rows), \
                 patch("walk_forward_pipeline.generate_walk_forward_splits", return_value=[split]), \
                 patch("walk_forward_pipeline.feature_columns", return_value=[]), \
                 patch("walk_forward_pipeline.generate_reports") as generate_reports_mock, \
                 patch("walk_forward_pipeline.walk_forward_gate_failures", return_value=[]), \
                 patch("walk_forward_pipeline.live_allowed_sides", return_value=([], {})), \
                 patch("walk_forward_pipeline.select_live_deploy_cycle", return_value=None), \
                 patch("walk_forward_pipeline.save_json") as save_json_mock:
                run_pipeline("USTEC_X100", 50, False, None, None)

            self.assertTrue((reports_dir / "USTEC_X100" / "walk_forward_tuned_50").exists())
            self.assertTrue((models_dir / "USTEC_X100" / "walk_forward_tuned_50").exists())
            self.assertEqual(generate_reports_mock.call_args.args[3], reports_dir / "USTEC_X100" / "walk_forward_tuned_50")
            self.assertFalse(
                any(
                    len(call.args) > 1 and Path(call.args[1]).name == "live_model_meta.json"
                    for call in save_json_mock.call_args_list
                )
            )

    def test_walk_forward_gate_blocks_partial_negative_aggregate(self):
        trades = pd.DataFrame(
            [
                {"entry_time": "2026-03-16", "exit_time": "2026-03-16", "side": "SELL", "exit_reason": "SL", "pnl": -0.5, "holding_candles": 1},
                {"entry_time": "2026-03-17", "exit_time": "2026-03-17", "side": "SELL", "exit_reason": "TP", "pnl": 0.2, "holding_candles": 1},
            ]
        )
        failures = walk_forward_gate_failures(
            [{"cycle": 1, "oos_trades": 2, "oos_net_profit": -0.3}],
            [trades],
            expected_cycles=3,
            min_total_trades=2,
            min_profit_factor=1.2,
        )

        self.assertTrue(any("walk-forward incomplete" in item for item in failures))
        self.assertTrue(any("net profit is not positive" in item for item in failures))
        self.assertTrue(any("profit factor below minimum" in item for item in failures))

    def test_walk_forward_gate_allows_complete_profitable_aggregate(self):
        rows = []
        for idx in range(12):
            rows.append(
                {
                    "entry_time": f"2026-03-{idx + 1:02d}",
                    "exit_time": f"2026-03-{idx + 1:02d}",
                    "side": "BUY" if idx % 2 == 0 else "SELL",
                    "exit_reason": "TP" if idx < 8 else "SL",
                    "pnl": 0.3 if idx < 8 else -0.1,
                    "pnl_money": 0.3 if idx < 8 else -0.1,
                    "holding_candles": 1,
                }
            )
        trades = pd.DataFrame(rows)
        summary_rows = [
            {"cycle": 1, "oos_trades": 4, "oos_net_profit": 0.8},
            {"cycle": 2, "oos_trades": 4, "oos_net_profit": 0.6},
            {"cycle": 3, "oos_trades": 4, "oos_net_profit": 0.6},
        ]

        metrics = aggregate_oos_metrics([trades])
        failures = walk_forward_gate_failures(
            summary_rows,
            [trades],
            expected_cycles=3,
            min_total_trades=5,
            min_profit_factor=1.2,
        )

        self.assertGreater(metrics["net_profit"], 0.0)
        self.assertEqual(failures, [])

    def test_walk_forward_gate_allows_side_filtered_profitable_buy(self):
        rows = []
        for idx in range(30):
            rows.append(
                {
                    "entry_time": pd.Timestamp("2026-03-01") + pd.Timedelta(days=idx),
                    "exit_time": pd.Timestamp("2026-03-01") + pd.Timedelta(days=idx, minutes=5),
                    "side": "BUY",
                    "exit_reason": "TP" if idx < 20 else "SL",
                    "pnl": 0.3 if idx < 20 else -0.1,
                    "pnl_money": 0.3 if idx < 20 else -0.1,
                    "holding_candles": 1,
                }
            )
        for idx in range(30):
            rows.append(
                {
                    "entry_time": pd.Timestamp("2026-04-01") + pd.Timedelta(days=idx),
                    "exit_time": pd.Timestamp("2026-04-01") + pd.Timedelta(days=idx, minutes=5),
                    "side": "SELL",
                    "exit_reason": "SL" if idx < 20 else "TP",
                    "pnl": -0.25 if idx < 20 else 0.1,
                    "pnl_money": -0.25 if idx < 20 else 0.1,
                    "holding_candles": 1,
                }
            )
        trades = pd.DataFrame(rows)
        summary_rows = [{"cycle": 1, "oos_trades": 60, "oos_net_profit": 1.0}]

        failures = walk_forward_gate_failures(
            summary_rows,
            [trades],
            expected_cycles=1,
            min_total_trades=30,
            min_profit_factor=1.2,
        )

        self.assertEqual(failures, [])

    def test_xauusd_live_gate_requires_individually_profitable_side(self):
        rows = []
        for idx in range(20):
            rows.append(
                {
                    "entry_time": pd.Timestamp("2026-03-01") + pd.Timedelta(days=idx),
                    "exit_time": pd.Timestamp("2026-03-01") + pd.Timedelta(days=idx, minutes=5),
                    "side": "BUY" if idx % 2 == 0 else "SELL",
                    "exit_reason": "TP" if idx < 12 else "SL",
                    "pnl": 0.2 if idx < 12 else -0.1,
                    "holding_candles": 1,
                }
            )
        trades = pd.DataFrame(rows)

        failures = walk_forward_gate_failures(
            [{"cycle": 1, "oos_trades": 20, "oos_net_profit": 1.6}],
            [trades],
            expected_cycles=1,
            symbol="XAUUSD",
            min_total_trades=30,
            min_profit_factor=1.2,
        )

        self.assertTrue(any("XAUUSD has no individually profitable side" in item for item in failures))

    def test_side_filtered_live_selection_prefers_latest_allowed_cycle(self):
        summary_rows = [
            {
                "cycle": 7,
                "oos_trades": 12,
                "oos_net_profit": 0.9,
                "oos_profit_factor": float("inf"),
                "oos_max_drawdown": 0.0,
                "buy_trades": 12,
                "sell_trades": 0,
                "buy_net_profit": 0.9,
                "buy_profit_factor": float("inf"),
            },
            {
                "cycle": 11,
                "oos_trades": 12,
                "oos_net_profit": 0.5,
                "oos_profit_factor": 1.5,
                "oos_max_drawdown": -0.4,
                "buy_trades": 12,
                "sell_trades": 0,
                "buy_net_profit": 0.5,
                "buy_profit_factor": 1.5,
            },
        ]

        live_meta = select_live_deploy_cycle(summary_rows, allowed_sides=["BUY"])

        self.assertEqual(live_meta["cycle"], "cycle_011")
        self.assertEqual(live_meta["selection"], "latest_side_filtered_walk_forward")
        self.assertEqual(live_meta["allowed_sides"], ["BUY"])

    def test_side_filtered_live_selection_rejects_negative_or_unhealthy_cycle(self):
        summary_rows = [
            {
                "cycle": 10,
                "oos_trades": 12,
                "oos_net_profit": -0.5,
                "oos_profit_factor": 0.8,
                "oos_max_drawdown": -1.0,
                "buy_trades": 12,
                "sell_trades": 0,
                "buy_net_profit": 1.0,
                "buy_profit_factor": 1.5,
            },
            {
                "cycle": 11,
                "oos_trades": 12,
                "oos_net_profit": 0.8,
                "oos_profit_factor": 1.4,
                "oos_max_drawdown": -0.4,
                "buy_trades": 12,
                "sell_trades": 0,
                "buy_net_profit": -0.2,
                "buy_profit_factor": 0.7,
            },
        ]

        self.assertIsNone(select_live_deploy_cycle(summary_rows, allowed_sides=["BUY"]))

    def test_live_selection_requires_minimum_cycle_trades(self):
        summary_rows = [
            {
                "cycle": 12,
                "oos_trades": 4,
                "oos_net_profit": 1.0,
                "oos_profit_factor": 2.0,
                "oos_max_drawdown": -0.1,
                "sell_trades": 4,
                "sell_net_profit": 1.0,
                "sell_profit_factor": 2.0,
            }
        ]

        self.assertIsNone(select_live_deploy_cycle(summary_rows, allowed_sides=["SELL"]))

    def test_aggregate_oos_metrics_handles_mixed_datetime_types(self):
        csv_loaded_trades = pd.DataFrame(
            [
                {
                    "entry_time": "2026-03-16 10:00:00",
                    "exit_time": "2026-03-16 10:05:00",
                    "side": "BUY",
                    "exit_reason": "TP",
                    "pnl": 0.4,
                    "holding_candles": 1,
                }
            ]
        )
        fresh_trades = pd.DataFrame(
            [
                {
                    "entry_time": pd.Timestamp("2026-03-17 10:00:00"),
                    "exit_time": pd.Timestamp("2026-03-17 10:05:00"),
                    "side": "SELL",
                    "exit_reason": "SL",
                    "pnl": -0.1,
                    "holding_candles": 1,
                }
            ]
        )

        metrics = aggregate_oos_metrics([csv_loaded_trades, fresh_trades])

        self.assertEqual(metrics["total_trades"], 2)
        self.assertAlmostEqual(metrics["net_profit"], 0.3)
        self.assertEqual(metrics["buy_trades"], 1)
        self.assertEqual(metrics["sell_trades"], 1)

    def test_generate_reports_handles_mixed_datetime_trade_frames(self):
        summary_rows = [
            {
                "cycle": 1,
                "oos_start": pd.Timestamp("2026-03-01"),
                "oos_end": pd.Timestamp("2026-03-31"),
                "oos_trades": 1,
                "oos_net_profit": 0.4,
                "oos_winrate": 1.0,
                "oos_profit_factor": float("inf"),
                "buy_trades": 1,
                "sell_trades": 0,
                "buy_threshold": 0.55,
                "sell_threshold": 0.55,
                "no_trade_zone": 0.0,
                "fallback_reason": "",
            },
            {
                "cycle": 2,
                "oos_start": pd.Timestamp("2026-04-01"),
                "oos_end": pd.Timestamp("2026-04-30"),
                "oos_trades": 1,
                "oos_net_profit": -0.1,
                "oos_winrate": 0.0,
                "oos_profit_factor": 0.0,
                "buy_trades": 0,
                "sell_trades": 1,
                "buy_threshold": 0.55,
                "sell_threshold": 0.55,
                "no_trade_zone": 0.0,
                "fallback_reason": "",
            },
        ]
        csv_loaded_trades = pd.DataFrame(
            [
                {
                    "entry_time": "2026-03-16 10:00:00",
                    "exit_time": "2026-03-16 10:05:00",
                    "side": "BUY",
                    "exit_reason": "TP",
                    "pnl": 0.4,
                    "holding_candles": 1,
                }
            ]
        )
        fresh_trades = pd.DataFrame(
            [
                {
                    "entry_time": pd.Timestamp("2026-04-17 10:00:00"),
                    "exit_time": pd.Timestamp("2026-04-17 10:05:00"),
                    "side": "SELL",
                    "exit_reason": "SL",
                    "pnl": -0.1,
                    "holding_candles": 1,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            generate_reports(
                "USTEC_X100",
                summary_rows,
                [csv_loaded_trades, fresh_trades],
                report_dir,
                expected_cycles=2,
                gate_failures=[],
            )

            self.assertTrue((report_dir / "walk_forward_report.md").exists())
            self.assertTrue((report_dir / "aggregated_oos_trades.csv").exists())
            self.assertTrue((report_dir / "monthly_returns.csv").exists())

    def test_generate_walk_forward_comparison_report_summarizes_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp)
            symbol_dir = reports_dir / "USTEC_X100"
            for mode, pnl_values in {
                "walk_forward_non_tuned": [0.4, -0.1, 0.3],
                "walk_forward_tuned_50": [0.2, 0.2, -0.05],
            }.items():
                mode_dir = symbol_dir / mode
                mode_dir.mkdir(parents=True)
                pd.DataFrame(
                    [
                        {"cycle": 1, "oos_trades": 1, "oos_net_profit": pnl_values[0], "oos_profit_factor": 1.5},
                        {"cycle": 2, "oos_trades": 1, "oos_net_profit": pnl_values[1], "oos_profit_factor": 0.8},
                        {"cycle": 3, "oos_trades": 1, "oos_net_profit": pnl_values[2], "oos_profit_factor": 2.0},
                    ]
                ).to_csv(mode_dir / "walk_forward_summary.csv", index=False)
                pd.DataFrame(
                    [
                        {
                            "entry_time": pd.Timestamp("2026-03-01") + pd.Timedelta(days=idx),
                            "exit_time": pd.Timestamp("2026-03-01") + pd.Timedelta(days=idx, minutes=5),
                            "side": "BUY" if idx % 2 == 0 else "SELL",
                            "exit_reason": "TP" if pnl > 0 else "SL",
                            "pnl": pnl,
                            "holding_candles": 1,
                        }
                        for idx, pnl in enumerate(pnl_values)
                    ]
                ).to_csv(mode_dir / "aggregated_oos_trades.csv", index=False)

            with patch("walk_forward_pipeline.REPORTS_DIR", reports_dir):
                report_path = generate_walk_forward_comparison_report("USTEC_X100")

            self.assertTrue(report_path.exists())
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("non_tuned", content)
            self.assertIn("tuned_50", content)
            self.assertIn("Latest 3 Net", content)

    def test_walk_forward_threshold_uses_adaptive_grid_for_low_calibrated_probabilities(self):
        rows = []
        proba = []
        labels = []
        price = 100.0
        for idx in range(90):
            is_buy = idx % 3 == 0
            is_sell = idx % 3 == 1
            label = 1 if is_buy else 2 if is_sell else 0
            labels.append(label)
            rows.append(
                {
                    "time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=5 * idx),
                    "high": price + 2.0,
                    "low": price - 2.0,
                    "close": price,
                    "atr_14": 1.0,
                    "spread": 0.0,
                    "spread_to_atr": 0.01,
                    "label": label,
                }
            )
            if is_buy:
                proba.append([0.28, 0.42, 0.30])
            elif is_sell:
                proba.append([0.30, 0.28, 0.42])
            else:
                proba.append([0.46, 0.27, 0.27])
            price += 0.1

        cfg = {
            "allowed_entry_sessions": [],
            "max_spread_to_atr": 0.2,
            "min_threshold_signals": 20,
            "live_sl_atr_mult": 1.0,
            "live_tp_atr_mult": 1.5,
            "lookahead_candles": 3,
            "default_lot": 0.01,
        }
        thresholds, grid = optimize_thresholds_wf(pd.DataFrame(rows), np.array(proba), cfg)

        self.assertTrue(bool(grid["backtest_evaluated"].any()))
        self.assertLess(thresholds["buy_threshold"], 0.55)
        self.assertLess(thresholds["sell_threshold"], 0.55)
        self.assertGreater(int(grid["total_signals"].max()), 0)
        self.assertIn("max_prob_buy", grid.columns)
        self.assertIn("signals_at_055", grid.columns)
        self.assertEqual(int(grid["signals_at_055"].iloc[0]), 0)

    def test_threshold_ranking_penalizes_low_trade_count_and_drawdown(self):
        candidates = pd.DataFrame(
            [
                {
                    "name": "too_few_trades",
                    "backtest_trades": 2,
                    "backtest_profit_factor": 2.0,
                    "backtest_net_profit": 0.3,
                    "backtest_expected_value": 0.15,
                    "backtest_max_drawdown": -1.0,
                },
                {
                    "name": "stable_low_drawdown",
                    "backtest_trades": 12,
                    "backtest_profit_factor": 1.5,
                    "backtest_net_profit": 1.0,
                    "backtest_expected_value": 0.2,
                    "backtest_max_drawdown": -0.1,
                },
                {
                    "name": "large_drawdown",
                    "backtest_trades": 12,
                    "backtest_profit_factor": 2.0,
                    "backtest_net_profit": 1.0,
                    "backtest_expected_value": 0.2,
                    "backtest_max_drawdown": -100.0,
                },
            ]
        )

        ranked = threshold_quality_columns(candidates, min_signals=10).sort_values(
            ["stable_trade_count", "threshold_quality_score"],
            ascending=[False, False],
        )

        self.assertEqual(ranked.iloc[0]["name"], "stable_low_drawdown")
        self.assertLess(
            float(ranked.loc[ranked["name"] == "too_few_trades", "threshold_quality_score"].iloc[0]),
            float(ranked.loc[ranked["name"] == "stable_low_drawdown", "threshold_quality_score"].iloc[0]),
        )
        self.assertLess(
            float(ranked.loc[ranked["name"] == "large_drawdown", "threshold_quality_score"].iloc[0]),
            float(ranked.loc[ranked["name"] == "stable_low_drawdown", "threshold_quality_score"].iloc[0]),
        )

    def test_walk_forward_threshold_prefers_target_trade_frequency_with_quality_gate(self):
        rows = []
        proba = []
        for idx in range(90):
            rows.append(
                {
                    "time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=35 * idx),
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "atr_14": 1.0,
                    "spread": 0.0,
                    "spread_to_atr": 0.01,
                    "is_london_session": 1 if idx < 20 else 0,
                    "label": 1,
                }
            )
            proba.append([0.20, 0.42, 0.38])

        cfg = {
            "allowed_entry_sessions": ["is_london_session"],
            "max_spread_to_atr": 0.2,
            "min_threshold_signals": 5,
            "live_sl_atr_mult": 1.0,
            "live_tp_atr_mult": 1.5,
            "lookahead_candles": 1,
            "default_lot": 0.01,
            "target_min_trades_per_day": 10.0,
            "target_max_trades_per_day": 15.0,
            "min_target_profit_factor": 1.03,
            "allow_session_relax_for_target": True,
        }

        def fake_simulate(_df, _proba, candidate_cfg, *_args, **_kwargs):
            trades = 24 if not candidate_cfg.get("allowed_entry_sessions") else 6
            return pd.DataFrame({"pnl": [0.1] * trades})

        with patch("walk_forward_threshold.simulate_wf_trades", side_effect=fake_simulate):
            thresholds, grid = optimize_thresholds_wf(pd.DataFrame(rows), np.array(proba), cfg)

        self.assertTrue(bool(grid["target_feasible"].any()))
        self.assertEqual(thresholds["session_mode"], "all_sessions")
        self.assertTrue(bool(thresholds["target_feasible"]))
        self.assertGreaterEqual(float(thresholds["backtest_trades_per_day"]), 10.0)
        self.assertLessEqual(float(thresholds["backtest_trades_per_day"]), 15.0)

    def test_walk_forward_threshold_rejects_target_frequency_when_quality_fails(self):
        rows = []
        proba = []
        for idx in range(90):
            rows.append(
                {
                    "time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=35 * idx),
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "atr_14": 1.0,
                    "spread": 0.0,
                    "spread_to_atr": 0.01,
                    "is_london_session": 1 if idx < 20 else 0,
                    "label": 1,
                }
            )
            proba.append([0.20, 0.42, 0.38])

        cfg = {
            "allowed_entry_sessions": ["is_london_session"],
            "max_spread_to_atr": 0.2,
            "min_threshold_signals": 5,
            "live_sl_atr_mult": 1.0,
            "live_tp_atr_mult": 1.5,
            "lookahead_candles": 1,
            "default_lot": 0.01,
            "target_min_trades_per_day": 10.0,
            "target_max_trades_per_day": 15.0,
            "min_target_profit_factor": 1.03,
            "allow_session_relax_for_target": True,
        }

        def fake_simulate(_df, _proba, candidate_cfg, *_args, **_kwargs):
            if not candidate_cfg.get("allowed_entry_sessions"):
                return pd.DataFrame({"pnl": [-0.1] * 24})
            return pd.DataFrame({"pnl": [0.1] * 6})

        with patch("walk_forward_threshold.simulate_wf_trades", side_effect=fake_simulate):
            thresholds, grid = optimize_thresholds_wf(pd.DataFrame(rows), np.array(proba), cfg)

        target_rows = grid[grid["meets_trade_frequency_target"] == True]
        self.assertTrue(bool((target_rows["meets_quality_gate"] == False).any()))
        self.assertFalse(bool(thresholds["target_feasible"]))
        self.assertTrue(bool(thresholds["meets_quality_gate"]))

    def test_oos_backtest_applies_all_sessions_threshold_mode(self):
        rows = []
        for idx in range(12):
            rows.append(
                {
                    "time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=5 * idx),
                    "high": 101.0,
                    "low": 99.5,
                    "close": 100.0,
                    "atr_14": 1.0,
                    "spread": 0.0,
                    "spread_to_atr": 0.01,
                    "is_london_session": 0,
                    "close_feature": 100.0,
                }
            )
        oos_df = pd.DataFrame(rows)
        cfg = {
            "allowed_entry_sessions": ["is_london_session"],
            "max_spread_to_atr": 0.2,
            "live_sl_atr_mult": 1.0,
            "live_tp_atr_mult": 1.5,
            "lookahead_candles": 1,
            "default_lot": 0.01,
            "buy_threshold": 0.55,
            "sell_threshold": 0.55,
        }
        model = DummyModel([0.1, 0.8, 0.1])
        configured_metrics, _ = run_oos_backtest(
            model,
            oos_df,
            ["close_feature"],
            cfg,
            {"buy_threshold": 0.55, "sell_threshold": 0.55, "no_trade_zone": 0.0},
            {"slippage_points": 0.0, "commission_per_lot": 0.0},
        )
        relaxed_metrics, _ = run_oos_backtest(
            model,
            oos_df,
            ["close_feature"],
            cfg,
            {"buy_threshold": 0.55, "sell_threshold": 0.55, "no_trade_zone": 0.0, "session_mode": "all_sessions"},
            {"slippage_points": 0.0, "commission_per_lot": 0.0},
        )

        self.assertEqual(configured_metrics["total_trades"], 0)
        self.assertGreater(relaxed_metrics["total_trades"], 0)

    def test_oos_backtest_reports_broker_money_pnl_when_specs_exist(self):
        rows = []
        for idx in range(4):
            rows.append(
                {
                    "time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=5 * idx),
                    "high": 102.0,
                    "low": 99.5,
                    "close": 100.0,
                    "atr_14": 1.0,
                    "spread": 0.0,
                    "spread_to_atr": 0.01,
                    "point": 0.01,
                    "tick_size": 0.01,
                    "tick_value": 1.0,
                    "x": 1.0,
                }
            )
        oos_df = pd.DataFrame(rows)
        cfg = {
            "allowed_entry_sessions": [],
            "max_spread_to_atr": 0.2,
            "live_sl_atr_mult": 1.0,
            "live_tp_atr_mult": 1.5,
            "lookahead_candles": 1,
            "default_lot": 0.10,
            "buy_threshold": 0.55,
            "sell_threshold": 0.55,
        }

        metrics, trades = run_oos_backtest(
            DummyModel([0.1, 0.8, 0.1]),
            oos_df,
            ["x"],
            cfg,
            {"buy_threshold": 0.55, "sell_threshold": 0.55, "no_trade_zone": 0.0},
            {"slippage_points": 0.0, "commission_per_lot": 0.0},
        )

        self.assertTrue(metrics["pnl_money_available"])
        self.assertIn("pnl_price_lot", trades.columns)
        self.assertIn("pnl_money", trades.columns)
        self.assertAlmostEqual(trades["pnl_price_lot"].iloc[0], 0.15)
        self.assertAlmostEqual(trades["pnl_money"].iloc[0], 15.0)
        self.assertAlmostEqual(metrics["net_profit_money"], metrics["net_profit"])
        self.assertAlmostEqual(metrics["avg_pnl_per_trade_money"], metrics["avg_pnl_per_trade"])

    def test_oos_backtest_keeps_price_lot_fallback_without_broker_specs(self):
        rows = []
        for idx in range(4):
            rows.append(
                {
                    "time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=5 * idx),
                    "high": 102.0,
                    "low": 99.5,
                    "close": 100.0,
                    "atr_14": 1.0,
                    "spread": 0.0,
                    "spread_to_atr": 0.01,
                    "x": 1.0,
                }
            )
        cfg = {
            "allowed_entry_sessions": [],
            "max_spread_to_atr": 0.2,
            "live_sl_atr_mult": 1.0,
            "live_tp_atr_mult": 1.5,
            "lookahead_candles": 1,
            "default_lot": 0.10,
            "buy_threshold": 0.55,
            "sell_threshold": 0.55,
        }

        metrics, trades = run_oos_backtest(
            DummyModel([0.1, 0.8, 0.1]),
            pd.DataFrame(rows),
            ["x"],
            cfg,
            {"buy_threshold": 0.55, "sell_threshold": 0.55, "no_trade_zone": 0.0},
            {"slippage_points": 0.0, "commission_per_lot": 0.0},
        )

        self.assertFalse(metrics["pnl_money_available"])
        self.assertAlmostEqual(metrics["net_profit"], metrics["net_profit_price_lot"])
        self.assertEqual(metrics["net_profit_money"], 0.0)
        self.assertTrue(trades["pnl_money"].isna().all())

    def test_latest_retrain_split_uses_latest_week(self):
        df = pd.DataFrame(
            {
                "time": pd.date_range("2025-06-16 06:30:00", "2026-06-05 18:00:00", freq="30min")
            }
        )
        split = generate_latest_retrain_split(df, train_months=8, val_months=1, oos_weeks=1)
        self.assertEqual(split["oos_end"], pd.Timestamp("2026-06-05 18:00:00"))
        self.assertEqual(split["oos_start"], pd.Timestamp("2026-05-29 18:00:00"))
        self.assertEqual(split["val_start"], pd.Timestamp("2026-04-29 18:00:00"))
        self.assertEqual(split["train_start"], pd.Timestamp("2025-08-29 18:00:00"))

    def test_effective_retrain_config_uses_symbol_windows(self):
        xau = effective_retrain_config("XAUUSD")
        ustec = effective_retrain_config("USTEC")
        x100 = effective_retrain_config("USTEC_X100")
        override = effective_retrain_config("USTEC_X100", train_months=3, val_months=2, oos_weeks=2)

        self.assertEqual((xau["train_months"], xau["val_months"], xau["oos_weeks"]), (12, 2, 1))
        self.assertEqual((ustec["train_months"], ustec["val_months"], ustec["oos_weeks"]), (12, 2, 1))
        self.assertEqual((x100["train_months"], x100["val_months"], x100["oos_weeks"]), (8, 1, 1))
        self.assertEqual((override["train_months"], override["val_months"], override["oos_weeks"]), (3, 2, 2))

    def test_deployment_gate_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            deployment_dir = Path(tmp)
            joblib.dump([DummyModel([0.2, 0.8])], deployment_dir / "buy_model.joblib")
            joblib.dump([DummyModel([0.3, 0.7])], deployment_dir / "sell_model.joblib")
            (deployment_dir / "buy_threshold.json").write_text(json.dumps({"threshold": 0.55}), encoding="utf-8")
            (deployment_dir / "sell_threshold.json").write_text(json.dumps({"threshold": 0.65}), encoding="utf-8")
            (deployment_dir / "side_model_meta.json").write_text(json.dumps({"side_training_mode": "separate"}), encoding="utf-8")
            (deployment_dir / "feature_columns.json").write_text(json.dumps(["x"]), encoding="utf-8")
            (deployment_dir / "best_threshold.json").write_text(json.dumps({"eligible": True, "allowed_sides": ["BUY"]}), encoding="utf-8")
            (deployment_dir / "side_metrics.json").write_text(
                json.dumps({
                    "BUY": {
                        "total_trades": 30,
                        "net_profit": 1.0,
                        "profit_factor": 1.3,
                        "avg_pnl_per_trade": 0.03,
                        "max_drawdown": -0.5,
                        "pnl_money_available": True,
                    }
                }),
                encoding="utf-8",
            )
            (deployment_dir / "retrain_meta.json").write_text("{}", encoding="utf-8")
            (deployment_dir / "oos_metrics.json").write_text(
                json.dumps({"total_trades": 30, "net_profit": 1.0, "profit_factor": 1.3, "max_drawdown": -0.5}),
                encoding="utf-8",
            )
            (deployment_dir / "oos_backtest.csv").write_text("", encoding="utf-8")
            self.assertEqual(
                deployment_gate_failures(
                    deployment_dir,
                    ["x"],
                    pd.Timestamp("2026-06-05"),
                    current_time=pd.Timestamp("2026-06-06"),
                ),
                [],
            )
            (deployment_dir / "best_threshold.json").write_text(json.dumps({"eligible": False}), encoding="utf-8")
            failures = deployment_gate_failures(
                deployment_dir,
                ["x"],
                pd.Timestamp("2026-06-05"),
                current_time=pd.Timestamp("2026-06-06"),
            )
            self.assertIn("threshold is not eligible", failures)

    def test_order_dedupe_key_includes_model_and_magic(self):
        key_a = order_dedupe_key("USTEC_X100", "BUY", pd.Timestamp("2026-06-01 10:00"), "cycle_001")
        key_b = order_dedupe_key("USTEC_X100", "BUY", pd.Timestamp("2026-06-01 10:00"), "cycle_002")
        self.assertNotEqual(key_a, key_b)
        self.assertIn("USTEC_X100|BUY|2026-06-01T10:00:00", key_a)

    def test_weekly_retrain_writes_artifacts_without_failed_live_meta_update(self):
        times = pd.date_range("2025-01-01", "2025-07-15", freq="h")
        labeled = pd.DataFrame({"time": times, "x": np.arange(len(times), dtype=float), "label": np.arange(len(times)) % 3})
        old_meta = {"source": "deployment", "deployment": "old"}

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "USTEC_X100").mkdir(parents=True)
            (base / "USTEC_X100" / "live_model_meta.json").write_text(json.dumps(old_meta), encoding="utf-8")
            with patch("weekly_retrain.MODELS_DIR", base), \
                 patch("weekly_retrain.REPORTS_DIR", base / "reports"), \
                 patch("weekly_retrain.prepare_labeled_data", return_value=labeled), \
                 patch("weekly_retrain.feature_columns", return_value=["x"]), \
                 patch("weekly_retrain.run_binary_tuning", return_value={}), \
                 patch("weekly_retrain.fit_calibrated_binary_ensemble", return_value=[DummyModel([0.2, 0.8])]), \
                 patch("weekly_retrain.optimize_side_threshold_wf", return_value=({"threshold": 0.55, "eligible": False}, pd.DataFrame([{"x": 1}]))), \
                 patch("weekly_retrain.run_oos_backtest", return_value=({"total_trades": 0, "net_profit": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}, pd.DataFrame())):
                result = run_weekly_retrain(
                    symbol="USTEC_X100",
                    n_trials=0,
                    train_months=3,
                    val_months=1,
                    oos_weeks=1,
                    deployment_name="test_deploy",
                    max_staleness_days=99999,
                )

            deployment_dir = base / "USTEC_X100" / "deployments" / "test_deploy"
            self.assertTrue((deployment_dir / "buy_model.joblib").exists())
            self.assertTrue((deployment_dir / "sell_model.joblib").exists())
            self.assertTrue((deployment_dir / "best_threshold.json").exists())
            self.assertTrue(result["gate_failures"])
            current_meta = json.loads((base / "USTEC_X100" / "live_model_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(current_meta, old_meta)

    def test_same_candle_sl_is_conservative(self):
        df = pd.DataFrame(
            [
                {"high": 100, "low": 100, "close": 100},
                {"high": 103, "low": 97, "close": 101},
            ]
        )
        exit_idx, exit_price, reason = simulate_exit(df, 0, "BUY", 100.0, 98.0, 102.0, 1)
        self.assertEqual(exit_idx, 1)
        self.assertEqual(exit_price, 98.0)
        self.assertEqual(reason, "SL")

    def test_feature_and_label_pipeline_sample(self):
        rows = []
        price = 100.0
        for idx in range(160):
            close = price + 0.1
            rows.append(
                {
                    "time": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5 * idx),
                    "open": price,
                    "high": close + 0.4,
                    "low": price - 0.4,
                    "close": close,
                    "tick_volume": 100 + idx,
                    "spread": 3,
                    "point": 0.01,
                    "real_volume": 0,
                }
            )
            price = close
        features = add_features(pd.DataFrame(rows), "XAUUSD")
        labeled = atr_barrier_labels(features, "XAUUSD")
        self.assertFalse(features.empty)
        self.assertIn("atr_14", features.columns)
        self.assertIn("spread_to_atr", features.columns)
        self.assertAlmostEqual(features["spread"].iloc[0], 0.03)
        self.assertAlmostEqual(features["spread_points"].iloc[0], 3.0)
        self.assertIn("label", labeled.columns)
        self.assertTrue(set(labeled["label"].unique()).issubset({0, 1, 2}))

    def test_label_diagnostics_reports_distribution_and_ambiguity(self):
        source = pd.DataFrame(
            [
                {
                    "time": pd.Timestamp("2026-01-01 00:00") + pd.Timedelta(minutes=5 * idx),
                    "open": 100.0,
                    "high": high,
                    "low": low,
                    "close": 100.0,
                    "atr_14": 1.0,
                }
                for idx, (high, low) in enumerate(
                    [
                        (100.0, 100.0),
                        (102.0, 98.0),
                        (101.0, 99.0),
                        (100.5, 99.5),
                        (100.5, 99.5),
                        (100.5, 99.5),
                        (100.5, 99.5),
                        (100.5, 99.5),
                        (100.5, 99.5),
                    ]
                )
            ]
        )
        labeled = source.iloc[:1].copy()
        labeled["label"] = [0]

        diagnostics = label_diagnostics(labeled, source, "XAUUSD")

        self.assertIn("label_counts", diagnostics)
        self.assertIn("monthly_distribution", diagnostics)
        self.assertIn("ambiguous_barrier_count", diagnostics)
        self.assertGreaterEqual(diagnostics["ambiguous_barrier_count"], 1)

    def test_enriched_features_are_numeric_and_finite(self):
        rows = []
        price = 100.0
        for idx in range(360):
            close = price + np.sin(idx / 9) * 0.4 + 0.05
            rows.append(
                {
                    "time": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5 * idx),
                    "open": price,
                    "high": max(price, close) + 0.6,
                    "low": min(price, close) - 0.6,
                    "close": close,
                    "tick_volume": 100 + (idx % 40),
                    "spread": 3 + (idx % 5),
                    "point": 0.01,
                    "real_volume": 0,
                }
            )
            price = close

        features = add_features(pd.DataFrame(rows), "USTEC")
        new_columns = [
            "adx_14",
            "trend_direction_strength",
            "ema_alignment_score",
            "atr_rank_100",
            "close_position_in_range_96",
            "dist_from_high_96_atr",
            "spread_rank_100",
            "spread_zscore_100",
            "abnormal_spread",
            "volume_rank_100",
            "m15_return_1",
            "m15_close_to_ema20",
            "h1_return_1",
            "h1_close_to_ema20",
        ]

        self.assertFalse(features.empty)
        for column in new_columns:
            self.assertIn(column, features.columns)
            self.assertTrue(pd.api.types.is_numeric_dtype(features[column]), column)
            self.assertTrue(np.isfinite(features[column].to_numpy()).all(), column)
        self.assertTrue(features["time"].is_monotonic_increasing)
        self.assertEqual(features["time"].nunique(), len(features))

    def test_higher_timeframe_features_do_not_use_current_m5_bucket(self):
        rows = []
        price = 100.0
        for idx in range(420):
            drift = 0.03 + np.sin(idx / 13) * 0.2
            close = price + drift
            rows.append(
                {
                    "time": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5 * idx),
                    "open": price,
                    "high": max(price, close) + 0.5,
                    "low": min(price, close) - 0.5,
                    "close": close,
                    "tick_volume": 100 + idx % 50,
                    "spread": 2 + idx % 4,
                    "point": 0.01,
                    "real_volume": 0,
                }
            )
            price = close

        baseline = pd.DataFrame(rows)
        changed = baseline.copy(deep=True)
        target_time = baseline.loc[300, "time"]
        changed.loc[changed["time"] == target_time, ["high", "close"]] += 100.0

        baseline_features = add_features(baseline, "USTEC")
        changed_features = add_features(changed, "USTEC")
        baseline_row = baseline_features.loc[baseline_features["time"] == target_time].iloc[0]
        changed_row = changed_features.loc[changed_features["time"] == target_time].iloc[0]

        for column in ["m15_return_1", "m15_close_to_ema20", "h1_return_1", "h1_close_to_ema20"]:
            self.assertAlmostEqual(float(baseline_row[column]), float(changed_row[column]), places=12)

    def test_features_work_without_optional_broker_columns(self):
        rows = []
        price = 100.0
        for idx in range(360):
            close = price + 0.05
            rows.append(
                {
                    "time": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=5 * idx),
                    "open": price,
                    "high": close + 0.4,
                    "low": price - 0.4,
                    "close": close,
                }
            )
            price = close

        features = add_features(pd.DataFrame(rows), "USTEC", include_symbol_features=False)

        self.assertFalse(features.empty)
        self.assertIn("spread_to_atr", features.columns)
        self.assertIn("tick_volume", features.columns)
        self.assertIn("m15_return_1", features.columns)
        self.assertTrue(np.isfinite(features.select_dtypes(include=[np.number]).to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
