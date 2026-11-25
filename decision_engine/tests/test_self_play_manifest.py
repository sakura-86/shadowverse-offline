# -*- coding: utf-8 -*-
"""Self-play training run manifest tests."""

from __future__ import annotations

import unittest

from decision_engine.training.self_play_manifest import SelfPlayTrainingRunManifest


class SelfPlayManifestTest(unittest.TestCase):
    def test_manifest_validation(self) -> None:
        manifest = SelfPlayTrainingRunManifest(
            run_id="self-play-001",
            run_dir="runs/self-play-001",
            deck_a="deck_a.json",
            deck_b="deck_b.json",
            config_json="config.json",
            output_jsonl="examples.jsonl",
            report_json="report.json",
            analysis_json="analysis.json",
            analysis_md="analysis.md",
            trace_dir="traces",
            seed_start=1,
            episodes=4,
            max_actions_per_turn=12,
            first_player="alternate",
            simulations=32,
            max_depth=1,
            c_puct=1.5,
            training_examples_written=10,
            episodes_total=4,
            episodes_terminal=3,
            episodes_completed=4,
            episodes_failed=1,
            predict_error_count=0,
            policy_target_count=10,
            value_target_count=3,
            sample_summary={"self_play_report": {"schema": "SelfPlaySearchSamplingReport", "version": 1}, "analysis": {"schema": "TrainingExampleAnalysisReport", "version": 1}},
        )

        payload = manifest.to_jsonable()
        self.assertEqual(payload["schema"], "SelfPlayTrainingRunManifest")
        self.assertEqual(payload["run_id"], "self-play-001")
        self.assertEqual(payload["training_examples_written"], 10)

    def test_manifest_rejects_missing_run_id(self) -> None:
        manifest = SelfPlayTrainingRunManifest(
            run_dir="runs/self-play-missing",
            deck_a="deck_a.json",
            deck_b="deck_b.json",
            config_json="config.json",
            output_jsonl="examples.jsonl",
            report_json="report.json",
        )
        with self.assertRaisesRegex(ValueError, "run_id"):
            manifest.validate()


if __name__ == "__main__":
    unittest.main()
