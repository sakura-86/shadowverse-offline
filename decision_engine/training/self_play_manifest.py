# -*- coding: utf-8 -*-
"""Run manifest schema for self-play generation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from decision_engine.training.feature_schema import to_jsonable

SELF_PLAY_MANIFEST_SCHEMA = "SelfPlayTrainingRunManifest"
SELF_PLAY_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class SelfPlayTrainingRunManifest:
    schema: str = SELF_PLAY_MANIFEST_SCHEMA
    version: int = SELF_PLAY_MANIFEST_VERSION
    run_id: str = ""
    run_dir: str = ""
    deck_a: str = ""
    deck_b: str = ""
    config_json: str = ""
    output_jsonl: str = ""
    report_json: str = ""
    analysis_json: str = ""
    analysis_md: str = ""
    trace_dir: str = ""
    seed_start: int = 0
    episodes: int = 0
    max_actions_per_turn: int = 0
    first_player: str = ""
    simulations: int = 0
    max_depth: int = 0
    c_puct: float = 0.0
    training_examples_written: int = 0
    episodes_total: int = 0
    episodes_terminal: int = 0
    episodes_completed: int = 0
    episodes_failed: int = 0
    predict_error_count: int = 0
    policy_target_count: int = 0
    value_target_count: int = 0
    baseline_name: str = "n1_no_nn_puct"
    baseline_version: int = 1
    search_name: str = "puct"
    neural_network_used: bool = False
    legal_action_source: str = "PlayerLocalActionEnumerator"
    predictor_source: str = "PlayerLocalPredictor"
    prior_source: str = "uniform"
    value_source: str = "terminal"
    policy_target_source: str = "visit_counts"
    requested_simulations: int = 0
    requested_max_depth: int = 1
    effective_max_depth: int = 1
    sample_summary: Mapping[str, Any] = field(default_factory=dict)
    non_goals: tuple[str, ...] = (
        "not_model_training",
        "not_runtime_change",
        "not_heuristic_fallback",
    )

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("self-play manifest run_id must be non-empty")
        if not self.run_dir:
            raise ValueError("self-play manifest run_dir must be non-empty")
        if not self.deck_a:
            raise ValueError("self-play manifest deck_a must be non-empty")
        if not self.deck_b:
            raise ValueError("self-play manifest deck_b must be non-empty")
        if not self.config_json:
            raise ValueError("self-play manifest config_json must be non-empty")
        if not self.output_jsonl:
            raise ValueError("self-play manifest output_jsonl must be non-empty")
        if not self.report_json:
            raise ValueError("self-play manifest report_json must be non-empty")
        if self.schema != SELF_PLAY_MANIFEST_SCHEMA:
            raise ValueError("self-play manifest schema mismatch")
        if self.version != SELF_PLAY_MANIFEST_VERSION:
            raise ValueError("self-play manifest version mismatch")

    def to_jsonable(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "version": self.version,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "deck_a": self.deck_a,
            "deck_b": self.deck_b,
            "config_json": self.config_json,
            "output_jsonl": self.output_jsonl,
            "report_json": self.report_json,
            "analysis_json": self.analysis_json,
            "analysis_md": self.analysis_md,
            "trace_dir": self.trace_dir,
            "seed_start": self.seed_start,
            "episodes": self.episodes,
            "max_actions_per_turn": self.max_actions_per_turn,
            "first_player": self.first_player,
            "simulations": self.simulations,
            "max_depth": self.max_depth,
            "c_puct": self.c_puct,
            "training_examples_written": self.training_examples_written,
            "episodes_total": self.episodes_total,
            "episodes_terminal": self.episodes_terminal,
            "episodes_completed": self.episodes_completed,
            "episodes_failed": self.episodes_failed,
            "predict_error_count": self.predict_error_count,
            "policy_target_count": self.policy_target_count,
            "value_target_count": self.value_target_count,
            "baseline_name": self.baseline_name,
            "baseline_version": self.baseline_version,
            "search_name": self.search_name,
            "neural_network_used": self.neural_network_used,
            "legal_action_source": self.legal_action_source,
            "predictor_source": self.predictor_source,
            "prior_source": self.prior_source,
            "value_source": self.value_source,
            "policy_target_source": self.policy_target_source,
            "requested_simulations": self.requested_simulations,
            "requested_max_depth": self.requested_max_depth,
            "effective_max_depth": self.effective_max_depth,
            "sample_summary": to_jsonable(self.sample_summary),
            "non_goals": list(self.non_goals),
        }
