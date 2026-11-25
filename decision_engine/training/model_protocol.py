# -*- coding: utf-8 -*-
"""Policy/value model and trainer protocol definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from decision_engine.training.feature_schema import TRAINING_FEATURE_SCHEMA_VERSION, to_jsonable

MODEL_ARTIFACT_SCHEMA = "PolicyValueModelArtifact"
MODEL_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class PolicyValuePrediction:
    policy_prior: tuple[float, ...]
    value: float
    policy_source: str = "model"
    value_source: str = "model"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, action_count: int) -> None:
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        if len(self.policy_prior) != action_count:
            raise ValueError("policy_prior width must match action count")
        if any(value < 0.0 for value in self.policy_prior):
            raise ValueError("policy_prior values must be non-negative")
        if self.policy_prior and abs(sum(self.policy_prior) - 1.0) > 1e-6:
            raise ValueError("policy_prior must sum to 1.0")
        if self.value < -1.0 or self.value > 1.0:
            raise ValueError("value must be in [-1.0, 1.0]")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "policy_prior": list(self.policy_prior),
            "value": self.value,
            "policy_source": self.policy_source,
            "value_source": self.value_source,
            "diagnostics": to_jsonable(self.diagnostics),
        }


@dataclass(frozen=True)
class PolicyValueModelArtifact:
    # Current dense baseline artifact.
    # Future structured-input variants should version their collection layout explicitly instead of relying on vector widths alone.
    schema: str = MODEL_ARTIFACT_SCHEMA
    version: int = MODEL_ARTIFACT_VERSION
    run_id: str = ""
    backend: str = "placeholder"
    model_kind: str = "policy_value_dual_head"
    input_tensorization_schema: str = "TrainingExampleTensorization"
    dataset_schema: str = "PolicyValueTensorDataset"
    state_vector_width: int = 0
    action_vector_width: int = 0
    policy_head_width: int = 0
    value_head_output: str = "scalar"
    feature_schema_version: int = TRAINING_FEATURE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("model artifact run_id must be non-empty")
        if self.state_vector_width <= 0:
            raise ValueError("model artifact state_vector_width must be positive")
        if self.action_vector_width <= 0:
            raise ValueError("model artifact action_vector_width must be positive")
        if self.policy_head_width <= 0:
            raise ValueError("model artifact policy_head_width must be positive")
        if self.value_head_output != "scalar":
            raise ValueError("model artifact value_head_output must be scalar")
        if self.feature_schema_version != TRAINING_FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "model artifact feature_schema_version does not match current training feature schema"
            )
        if self.schema != MODEL_ARTIFACT_SCHEMA:
            raise ValueError("model artifact schema mismatch")
        if self.version != MODEL_ARTIFACT_VERSION:
            raise ValueError("model artifact version mismatch")

    def to_jsonable(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "version": self.version,
            "run_id": self.run_id,
            "backend": self.backend,
            "model_kind": self.model_kind,
            "input_tensorization_schema": self.input_tensorization_schema,
            "dataset_schema": self.dataset_schema,
            "state_vector_width": self.state_vector_width,
            "action_vector_width": self.action_vector_width,
            "policy_head_width": self.policy_head_width,
            "value_head_output": self.value_head_output,
            "feature_schema_version": self.feature_schema_version,
            "metadata": to_jsonable(self.metadata),
        }


@runtime_checkable
class PolicyValueModel(Protocol):
    def predict(
        self,
        *,
        state_vector: Sequence[float],
        action_vectors: Sequence[Sequence[float]],
        metadata: Mapping[str, Any] | None = None,
    ) -> PolicyValuePrediction:
        ...


@runtime_checkable
class PolicyValueTrainer(Protocol):
    def fit(
        self,
        *,
        tensor_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        run_id: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        ...
