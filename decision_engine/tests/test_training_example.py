# -*- coding: utf-8 -*-
"""TrainingExample schema tests."""

import unittest

from decision_engine.training.training_example import TrainingExample, one_hot_policy_target, validate_training_example


class TrainingExampleTest(unittest.TestCase):
    def test_training_example_validates_and_serializes(self) -> None:
        example = TrainingExample(
            sample_id="sample-1",
            source={"kind": "unit_test"},
            trust={"level": "trusted"},
            player="me",
            turn_index=3,
            state_features={
                "schema": "StateFeatures",
                "version": 1,
                "player": {"life": 20, "crests": [], "extra_crests": [], "boons": []},
                "legal_summary": {"present": False},
            },
            legal_action_features=(
                {"schema": "ActionFeatures", "version": 1, "action_kind": "play"},
                {"schema": "ActionFeatures", "version": 1, "action_kind": "turn_end"},
            ),
            selected_action_index=0,
            policy_target=one_hot_policy_target(2, 0),
            value_target=1.0,
            generation_config={"kind": "policy_value_tensorization", "source": "unit_test"},
            metadata={"generation": "test"},
        )

        payload = example.to_jsonable()

        self.assertEqual(payload["schema"], "TrainingExample")
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["sample_id"], "sample-1")
        self.assertEqual(payload["source"], {"kind": "unit_test"})
        self.assertEqual(payload["trust"], {"level": "trusted"})
        self.assertEqual(payload["policy_target"], [1.0, 0.0])
        self.assertEqual(payload["selected_action_index"], 0)
        self.assertEqual(payload["value_target"], 1.0)
        self.assertEqual(payload["generation_config"], {"kind": "policy_value_tensorization", "source": "unit_test"})

    def test_training_example_rejects_selected_index_out_of_range(self) -> None:
        example = _example(selected_action_index=2)

        with self.assertRaisesRegex(ValueError, "selected_action_index out of range"):
            validate_training_example(example)

    def test_training_example_rejects_policy_length_mismatch(self) -> None:
        example = _example(policy_target=(1.0, 0.0, 0.0))

        with self.assertRaisesRegex(ValueError, "policy_target length"):
            validate_training_example(example)

    def test_training_example_rejects_policy_sum_mismatch(self) -> None:
        example = _example(policy_target=(0.6, 0.3))

        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            validate_training_example(example)

    def test_training_example_rejects_leaked_feature_keys(self) -> None:
        example = _example(state_features={"after_context": {}})

        with self.assertRaisesRegex(ValueError, "leaked key"):
            validate_training_example(example)

    def test_one_hot_policy_target_validates_bounds(self) -> None:
        self.assertEqual(one_hot_policy_target(3, 1), (0.0, 1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "out of range"):
            one_hot_policy_target(3, 3)


def _example(**overrides) -> TrainingExample:
    values = {
        "sample_id": "sample-1",
        "source": {"kind": "unit_test"},
        "trust": {"level": "trusted"},
        "player": "me",
        "turn_index": 1,
        "state_features": {"schema": "StateFeatures"},
        "legal_action_features": (
            {"schema": "ActionFeatures", "action_kind": "play"},
            {"schema": "ActionFeatures", "action_kind": "turn_end"},
        ),
        "selected_action_index": 0,
        "policy_target": (1.0, 0.0),
        "value_target": None,
        "metadata": {},
    }
    values.update(overrides)
    return TrainingExample(**values)


if __name__ == "__main__":
    unittest.main()
