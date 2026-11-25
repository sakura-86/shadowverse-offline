# -*- coding: utf-8 -*-
"""Run manifest schema for offline policy/value training artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from decision_engine.training.feature_schema import to_jsonable
from decision_engine.training.model_protocol import MODEL_ARTIFACT_SCHEMA, MODEL_ARTIFACT_VERSION

RUN_MANIFEST_SCHEMA = "PolicyValueTrainingRunManifest"
RUN_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class PolicyValueTrainingRunManifest:
    schema: str = RUN_MANIFEST_SCHEMA
    version: int = RUN_MANIFEST_VERSION
    run_id: str = ""
    input_dir: str = ""
    output_dir: str = ""
    config_json: str = ""
    sample_manifest_json: str = ""
    train_metrics_json: str = ""
    eval_report_json: str = ""
    model_artifact_json: str = ""
    tensor_dataset_summary: Mapping[str, Any] = field(default_factory=dict)
    training_summary: Mapping[str, Any] = field(default_factory=dict)
    model_artifact_schema: str = MODEL_ARTIFACT_SCHEMA
    model_artifact_version: int = MODEL_ARTIFACT_VERSION
    non_goals: tuple[str, ...] = (
        "not_model_training",
        "not_runtime_change",
        "not_real_backend",
    )

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run manifest run_id must be non-empty")
        if not self.input_dir:
            raise ValueError("run manifest input_dir must be non-empty")
        if not self.output_dir:
            raise ValueError("run manifest output_dir must be non-empty")
        if not self.config_json:
            raise ValueError("run manifest config_json must be non-empty")
        if not self.sample_manifest_json:
            raise ValueError("run manifest sample_manifest_json must be non-empty")
        if not self.train_metrics_json:
            raise ValueError("run manifest train_metrics_json must be non-empty")
        if not self.eval_report_json:
            raise ValueError("run manifest eval_report_json must be non-empty")
        if not self.model_artifact_json:
            raise ValueError("run manifest model_artifact_json must be non-empty")
        if self.schema != RUN_MANIFEST_SCHEMA:
            raise ValueError("run manifest schema mismatch")
        if self.version != RUN_MANIFEST_VERSION:
            raise ValueError("run manifest version mismatch")

    def to_jsonable(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "version": self.version,
            "run_id": self.run_id,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "config_json": self.config_json,
            "sample_manifest_json": self.sample_manifest_json,
            "train_metrics_json": self.train_metrics_json,
            "eval_report_json": self.eval_report_json,
            "model_artifact_json": self.model_artifact_json,
            "tensor_dataset_summary": to_jsonable(self.tensor_dataset_summary),
            "training_summary": to_jsonable(self.training_summary),
            "model_artifact_schema": self.model_artifact_schema,
            "model_artifact_version": self.model_artifact_version,
            "non_goals": list(self.non_goals),
        }
