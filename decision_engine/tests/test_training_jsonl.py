# -*- coding: utf-8 -*-
"""TrainingExample JSONL helper tests."""

import tempfile
import unittest
from pathlib import Path

from decision_engine.training.jsonl import load_training_examples_jsonl, write_training_examples_jsonl
from decision_engine.training.training_example import TrainingExample


class TrainingJsonlTest(unittest.TestCase):
    def test_write_and_load_training_examples_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "examples.jsonl"
            count = write_training_examples_jsonl(path, [_example("a"), _example("b")])

            loaded = load_training_examples_jsonl(path)

        self.assertEqual(count, 2)
        self.assertEqual([item["sample_id"] for item in loaded], ["a", "b"])
        self.assertEqual(loaded[0]["policy_target"], [1.0, 0.0])
        self.assertEqual(loaded[0]["generation_config"], {"kind": "policy_value_tensorization", "source": "unit_test"})

    def test_load_rejects_invalid_training_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "examples.jsonl"
            path.write_text('{"schema":"TrainingExample","version":1}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sample_id"):
                load_training_examples_jsonl(path)


def _example(sample_id: str) -> TrainingExample:
    return TrainingExample(
        sample_id=sample_id,
        source={"kind": "unit_test"},
        trust={"level": "trusted"},
        player="me",
        turn_index=1,
        state_features={"schema": "StateFeatures", "version": 1, "player": {"life": 20, "crests": [], "extra_crests": [], "boons": []}},
        legal_action_features=(
            {"schema": "ActionFeatures", "version": 1, "action_kind": "play"},
            {"schema": "ActionFeatures", "version": 1, "action_kind": "turn_end"},
        ),
        selected_action_index=0,
        policy_target=(1.0, 0.0),
        generation_config={"kind": "policy_value_tensorization", "source": "unit_test"},
    )


if __name__ == "__main__":
    unittest.main()
