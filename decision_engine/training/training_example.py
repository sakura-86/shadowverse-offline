# -*- coding: utf-8 -*-
"""Training example schema for policy/value self-play data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from decision_engine.training.feature_schema import TRAINING_EXAMPLE_SCHEMA, TRAINING_EXAMPLE_VERSION, assert_no_leaked_feature_keys, to_jsonable


@dataclass(frozen=True)
class TrainingExample:
    sample_id: str
    source: Mapping[str, Any]
    trust: Mapping[str, Any]
    player: str
    turn_index: int
    state_features: Mapping[str, Any]
    legal_action_features: tuple[Mapping[str, Any], ...]
    selected_action_index: int | None = None
    policy_target: tuple[float, ...] = ()
    value_target: float | None = None
    history_state_features: tuple[Mapping[str, Any], ...] = ()
    #: 最近若干步**已执行动作**（按时间顺序，最早在前）。与 history_state_features 不同：
    #: 那个是历史**状态**，这个是历史**动作**（谁做了什么、打给谁），供 policy 学
    #: combo / 顺序依赖。上限 4 步，与 entity 的 HISTORY_ACTION_LIMIT 对齐。
    history_actions: tuple[Mapping[str, Any], ...] = ()
    generation_config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = TRAINING_EXAMPLE_SCHEMA
    version: int = TRAINING_EXAMPLE_VERSION

    def validate(self) -> None:
        validate_training_example(self)

    def to_jsonable(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "version": self.version,
            "sample_id": self.sample_id,
            "source": to_jsonable(self.source),
            "trust": to_jsonable(self.trust),
            "player": self.player,
            "turn_index": self.turn_index,
            "state_features": to_jsonable(self.state_features),
            "legal_action_features": to_jsonable(self.legal_action_features),
            "selected_action_index": self.selected_action_index,
            "policy_target": list(self.policy_target),
            "value_target": self.value_target,
            "history_state_features": to_jsonable(self.history_state_features),
            "history_actions": to_jsonable(self.history_actions),
            "generation_config": to_jsonable(self.generation_config),
            "metadata": to_jsonable(self.metadata),
        }


def validate_training_example(example: TrainingExample | Mapping[str, Any]) -> None:
    data = _example_mapping(example)
    if data.get("schema") != TRAINING_EXAMPLE_SCHEMA:
        raise ValueError(f"training example schema must be {TRAINING_EXAMPLE_SCHEMA}")
    if data.get("version") != TRAINING_EXAMPLE_VERSION:
        raise ValueError(f"training example version must be {TRAINING_EXAMPLE_VERSION}")
    sample_id = data.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("training example sample_id must be a non-empty string")
    for key in ("source", "trust", "state_features", "generation_config", "metadata"):
        if not isinstance(data.get(key), Mapping):
            raise ValueError(f"training example {key} must be a mapping")
    legal_actions = data.get("legal_action_features")
    if not isinstance(legal_actions, tuple | list):
        raise ValueError("training example legal_action_features must be a sequence")
    legal_count = len(legal_actions)
    if legal_count <= 0:
        raise ValueError("training example requires at least one legal action")
    for index, item in enumerate(legal_actions):
        if not isinstance(item, Mapping):
            raise ValueError(f"training example legal_action_features[{index}] must be a mapping")
    selected = data.get("selected_action_index")
    if selected is not None:
        if not isinstance(selected, int):
            raise ValueError("training example selected_action_index must be int or null")
        if selected < 0 or selected >= legal_count:
            raise ValueError("training example selected_action_index out of range")
    policy = data.get("policy_target") or ()
    if not isinstance(policy, tuple | list):
        raise ValueError("training example policy_target must be a sequence")
    if policy:
        if len(policy) != legal_count:
            raise ValueError("training example policy_target length must equal legal_action_features length")
        total = 0.0
        for value in policy:
            if not isinstance(value, int | float):
                raise ValueError("training example policy_target values must be numeric")
            if value < 0.0:
                raise ValueError("training example policy_target values must be non-negative")
            total += float(value)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("training example policy_target must sum to 1.0")
    value_target = data.get("value_target")
    if value_target is not None:
        if not isinstance(value_target, int | float):
            raise ValueError("training example value_target must be numeric or null")
        if float(value_target) < -1.0 or float(value_target) > 1.0:
            raise ValueError("training example value_target must be in [-1.0, 1.0]")
    history = data.get("history_state_features") or ()
    if not isinstance(history, tuple | list):
        raise ValueError("training example history_state_features must be a sequence")
    if len(history) > 4:
        raise ValueError("training example history_state_features supports at most four states")
    for index, state in enumerate(history):
        if not isinstance(state, Mapping):
            raise ValueError(f"training example history_state_features[{index}] must be a mapping")
        assert_no_leaked_feature_keys(state, path=f"history_state_features[{index}]")
    assert_no_leaked_feature_keys(data.get("state_features"), path="state_features")
    assert_no_leaked_feature_keys(data.get("legal_action_features"), path="legal_action_features")

    # 历史动作：最多 4 步（与 entity 的 HISTORY_ACTION_LIMIT 对齐）。这里只校验形状，
    # 语义（是否含 uid / 泄漏字段）由导出端保证——动作里只允许 base_card_id / kind / is_own。
    history_actions = data.get("history_actions") or ()
    if not isinstance(history_actions, tuple | list):
        raise ValueError("training example history_actions must be a sequence")
    if len(history_actions) > 4:
        raise ValueError("training example history_actions supports at most four steps")
    for index, item in enumerate(history_actions):
        if not isinstance(item, Mapping):
            raise ValueError(f"training example history_actions[{index}] must be a mapping")


def one_hot_policy_target(action_count: int, selected_action_index: int) -> tuple[float, ...]:
    if action_count <= 0:
        raise ValueError("action_count must be positive")
    if selected_action_index < 0 or selected_action_index >= action_count:
        raise ValueError("selected_action_index out of range")
    return tuple(1.0 if index == selected_action_index else 0.0 for index in range(action_count))


def _example_mapping(example: TrainingExample | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(example, TrainingExample):
        return {
            "schema": example.schema,
            "version": example.version,
            "sample_id": example.sample_id,
            "source": example.source,
            "trust": example.trust,
            "player": example.player,
            "turn_index": example.turn_index,
            "state_features": example.state_features,
            "legal_action_features": example.legal_action_features,
            "selected_action_index": example.selected_action_index,
            "policy_target": example.policy_target,
            "value_target": example.value_target,
            "history_state_features": example.history_state_features,
            "generation_config": example.generation_config,
            "metadata": example.metadata,
        }
    return example
