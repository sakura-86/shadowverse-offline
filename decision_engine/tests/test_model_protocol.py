# -*- coding: utf-8 -*-
"""Policy/value model protocol tests."""

from __future__ import annotations

import unittest

from decision_engine.training.model_protocol import PolicyValueModelArtifact, PolicyValuePrediction


class ModelProtocolTest(unittest.TestCase):
    def test_prediction_validation(self) -> None:
        prediction = PolicyValuePrediction(policy_prior=(0.25, 0.75), value=0.5)
        prediction.validate(2)
        self.assertEqual(prediction.to_jsonable()["value"], 0.5)

    def test_prediction_validation_rejects_invalid_width(self) -> None:
        prediction = PolicyValuePrediction(policy_prior=(1.0,), value=0.0)
        with self.assertRaisesRegex(ValueError, "policy_prior width"):
            prediction.validate(2)

    def test_model_artifact_validation(self) -> None:
        artifact = PolicyValueModelArtifact(run_id="run-001", backend="placeholder", state_vector_width=64, action_vector_width=19, policy_head_width=19)
        artifact.validate()
        self.assertEqual(artifact.to_jsonable()["schema"], "PolicyValueModelArtifact")

    def test_model_artifact_rejects_stale_feature_schema(self) -> None:
        artifact = PolicyValueModelArtifact(
            run_id="run-001",
            backend="placeholder",
            state_vector_width=64,
            action_vector_width=19,
            policy_head_width=19,
            feature_schema_version=1,
        )
        with self.assertRaisesRegex(ValueError, "feature_schema_version"):
            artifact.validate()

    def test_model_artifact_rejects_missing_run_id(self) -> None:
        artifact = PolicyValueModelArtifact(backend="placeholder", state_vector_width=64, action_vector_width=19, policy_head_width=19)
        with self.assertRaisesRegex(ValueError, "run_id"):
            artifact.validate()


if __name__ == "__main__":
    unittest.main()
