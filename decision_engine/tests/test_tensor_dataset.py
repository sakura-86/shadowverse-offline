# -*- coding: utf-8 -*-
"""Tensor dataset loading tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from decision_engine.training.tensor_dataset import flatten_tensor_dataset, load_tensor_dataset_dir, summarize_tensor_dataset, validate_tensorized_example


class TensorDatasetTest(unittest.TestCase):
    def test_validate_tensorized_example_accepts_export_row(self) -> None:
        payload = _tensorized_example()
        self.assertEqual(validate_tensorized_example(payload)["schema"], "TrainingExampleTensorization")

    def test_load_and_summarize_tensor_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_jsonl(root / "train.jsonl", [_tensorized_example("train-1", 2)])
            _write_jsonl(root / "val.jsonl", [_tensorized_example("val-1", 2)])
            _write_jsonl(root / "test.jsonl", [_tensorized_example("test-1", 2)])

            dataset = load_tensor_dataset_dir(root)
            summary = summarize_tensor_dataset(dataset)

            self.assertEqual(summary["schema"], "PolicyValueTensorDataset")
            self.assertEqual(summary["total_samples"], 3)
            self.assertEqual(summary["splits"]["train"]["count"], 1)
            self.assertEqual(summary["splits"]["train"]["value_target_present_count"], 1)
            self.assertEqual(len(flatten_tensor_dataset(dataset)), 3)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _tensorized_example(sample_id: str = "sample-1", action_count: int = 2) -> dict:
    return {
        "schema": "TrainingExampleTensorization",
        "version": 1,
        "sample_id": sample_id,
        "state_vector": [1.0, 0.0, 20.0],
        "action_vectors": [[1.0] * 19 for _ in range(action_count)],
        "policy_target": [1.0 / action_count for _ in range(action_count)],
        "value_target": 1.0,
        "labels": {"selected_action_index": 0, "policy_target_present": True, "value_target_present": True, "policy_target_width": action_count},
        "feature_shapes": {"state_vector": 3, "action_vectors": [19 for _ in range(action_count)]},
        "metadata": {},
    }


if __name__ == "__main__":
    unittest.main()
