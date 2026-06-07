import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import optimize_research_workflow as workflow


class OptimizeResearchWorkflowTests(unittest.TestCase):
    def test_build_smoke_commands_use_safe_resume_workflow(self):
        args = argparse.Namespace(
            stages=["label", "feature", "walk_forward", "retrain"],
            smoke=True,
            trials=50,
            max_cycles=None,
            resume=True,
            force_download=False,
            skip_download=False,
            label_feature_set="core20",
            label_variants="all",
            feature_sets=["core20", "core30"],
            importance_top_n=[20, 30],
            deploy=False,
            per_symbol=False,
        )

        commands = workflow.build_workflow_commands(["XAUUSD"], args)
        command_texts = [workflow.command_text(command.argv) for command in commands]

        self.assertTrue(any("label_selection_experiment.py" in text for text in command_texts))
        self.assertTrue(any("--trials 0" in text and "--max-cycles 3" in text for text in command_texts))
        self.assertTrue(any("feature_selection_experiment.py" in text and "--feature-set core30" in text for text in command_texts))
        self.assertTrue(any("walk_forward_pipeline.py" in text and "--symbol XAUUSD" in text for text in command_texts))
        self.assertTrue(any("weekly_retrain.py" in text and "--no-deploy" in text for text in command_texts))

    def test_per_symbol_mode_splits_multi_symbol_stage_commands(self):
        args = argparse.Namespace(
            stages=["prepare", "label", "feature", "retrain"],
            smoke=True,
            trials=50,
            max_cycles=None,
            resume=True,
            force_download=False,
            skip_download=False,
            label_feature_set="core20",
            label_variants="all",
            feature_sets=["core20"],
            importance_top_n=[20],
            deploy=False,
            per_symbol=True,
        )

        commands = workflow.build_workflow_commands(["XAUUSD", "USTEC"], args)
        command_texts = [workflow.command_text(command.argv) for command in commands]

        self.assertTrue(any("download_mt5_data.py" in text and "--symbol XAUUSD" in text for text in command_texts))
        self.assertTrue(any("download_mt5_data.py" in text and "--symbol USTEC" in text for text in command_texts))
        self.assertTrue(any("label_selection_experiment.py" in text and "--symbol XAUUSD" in text for text in command_texts))
        self.assertTrue(any("feature_selection_experiment.py" in text and "--symbol USTEC" in text for text in command_texts))
        self.assertTrue(any("weekly_retrain.py" in text and "--symbol XAUUSD" in text for text in command_texts))
        self.assertFalse(any("--symbols XAUUSD USTEC" in text for text in command_texts))

    def test_skip_download_keeps_cached_data_prepare_steps(self):
        args = argparse.Namespace(
            stages=["prepare"],
            smoke=False,
            trials=50,
            max_cycles=None,
            resume=False,
            force_download=False,
            skip_download=True,
            label_feature_set="core20",
            label_variants="all",
            feature_sets=["core20"],
            importance_top_n=[20],
            deploy=False,
            per_symbol=True,
        )

        commands = workflow.build_workflow_commands(["XAUUSD"], args)
        command_texts = [workflow.command_text(command.argv) for command in commands]

        self.assertFalse(any("download_mt5_data.py" in text for text in command_texts))
        self.assertTrue(any("features.py" in text and "--symbol XAUUSD" in text for text in command_texts))
        self.assertTrue(any("labeling.py" in text and "--symbol XAUUSD" in text for text in command_texts))

    def test_write_summary_report_selects_best_by_symbol_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_root = Path(tmp)
            label_dir = report_root / "XAUUSD" / "label_selection" / "core20" / "balanced"
            weak_label_dir = report_root / "XAUUSD" / "label_selection" / "core20" / "fast"
            feature_dir = report_root / "XAUUSD" / "feature_selection" / "core20"
            walk_dir = report_root / "XAUUSD" / "walk_forward"
            for path in [label_dir, weak_label_dir, feature_dir, walk_dir]:
                path.mkdir(parents=True)

            (label_dir / "aggregated_oos_metrics.json").write_text(
                json.dumps({"total_trades": 40, "net_profit": 5.0, "profit_factor": 1.4, "max_drawdown": -1.0, "winrate": 0.55}),
                encoding="utf-8",
            )
            (weak_label_dir / "aggregated_oos_metrics.json").write_text(
                json.dumps({"total_trades": 30, "net_profit": -1.0, "profit_factor": 0.8, "max_drawdown": -2.0, "winrate": 0.45}),
                encoding="utf-8",
            )
            (feature_dir / "aggregated_oos_metrics.json").write_text(
                json.dumps({"total_trades": 35, "net_profit": 3.0, "profit_factor": 1.2, "max_drawdown": -1.5, "winrate": 0.5}),
                encoding="utf-8",
            )
            (walk_dir / "aggregated_oos_metrics.json").write_text(
                json.dumps({"total_trades": 45, "net_profit": 6.0, "profit_factor": 1.5, "max_drawdown": -1.2, "winrate": 0.56}),
                encoding="utf-8",
            )

            with patch.object(workflow, "REPORTS_DIR", report_root):
                output = workflow.write_summary_report(["XAUUSD"])
                summary = json.loads((report_root / "optimization" / "optimization_summary.json").read_text(encoding="utf-8"))

            self.assertEqual(output, report_root / "optimization" / "optimization_summary.md")
            best = {(row["symbol"], row["category"]): row["candidate"] for row in summary["best"]}
            self.assertEqual(best[("XAUUSD", "label")], "core20/balanced")
            self.assertEqual(best[("XAUUSD", "feature")], "core20")
            self.assertEqual(best[("XAUUSD", "walk_forward")], "walk_forward")


if __name__ == "__main__":
    unittest.main()
