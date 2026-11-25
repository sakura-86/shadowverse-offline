# -*- coding: utf-8 -*-
"""Tensorization helpers for policy/value training examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from decision_engine.training.action_effect_tags import TAGS as ACTION_EFFECT_TAGS
from decision_engine.training.feature_schema import TRAINING_FEATURE_SCHEMA_VERSION, assert_no_leaked_feature_keys, to_jsonable
from decision_engine.training.training_example import validate_training_example

TENSORIZATION_SCHEMA = "TrainingExampleTensorization"
TENSORIZATION_VERSION = 1


@dataclass(frozen=True)
class TensorizationResult:
    schema: str
    version: int
    sample_id: str
    state_vector: tuple[float, ...]
    action_vectors: tuple[tuple[float, ...], ...]
    policy_target: tuple[float, ...]
    value_target: float | None
    labels: Mapping[str, Any]
    feature_shapes: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable({
            "schema": self.schema,
            "version": self.version,
            "sample_id": self.sample_id,
            "state_vector": self.state_vector,
            "action_vectors": self.action_vectors,
            "policy_target": self.policy_target,
            "value_target": self.value_target,
            "labels": self.labels,
            "feature_shapes": self.feature_shapes,
            "metadata": self.metadata,
        })


STATE_VECTOR_FEATURE_ORDER = (
    # 全局状态 (2)
    "is_ally_turn",
    "is_turn_ended",
    # 我方资源 (8)
    "player.life",
    "player.max_life",
    "player.pp",
    "player.max_pp",
    "player.ep",
    "player.sep",
    "player.extra_pp_available",  # 是否有可用额外PP（合并原 6 维为 1 维）
    "player.turn",
    # 我方状态 / 不可从当前盘面重算的累计量 (12)
    "player.is_first_side",
    "player.hand_count",
    "player.field_count",
    "player.deck_count",
    "player.crest_count",
    "player.play_count",
    "player.graveyard_count",
    "player.rally",
    "player.evolve_count",
    "player.is_kakusei",
    "player.super_evolve_unlocked",
    "player.follower_attacked_leader_previous_turn",
    # 对方资源 (7)
    "opponent.life",
    "opponent.max_life",
    "opponent.pp",
    "opponent.max_pp",
    "opponent.ep",
    "opponent.sep",
    "opponent.hand_count",
    # 对方状态 / 不可从当前盘面重算的累计量 (10)
    "opponent.field_count",
    "opponent.deck_count",
    "opponent.crest_count",
    "opponent.play_count",
    "opponent.graveyard_count",
    "opponent.rally",
    "opponent.evolve_count",
    "opponent.is_kakusei",
    "opponent.super_evolve_unlocked",
    "opponent.follower_attacked_leader_previous_turn",
    # 场面压力 (9)
    "pressure.player_field_attack_total",
    "pressure.opponent_field_attack_total",
    "pressure.player_field_life_total",
    "pressure.opponent_field_life_total",
    "pressure.player_guard_count",
    "pressure.player_guard_life_total",
    "pressure.opponent_guard_count",
    "pressure.pp_used_ratio",
    "pressure.playable_next_turn_count",
    # 合法动作摘要 (7)
    "legal_summary.present",
    "legal_summary.can_play_count",
    "legal_summary.can_attack_leader_count",
    "legal_summary.can_attack_field_count",
    "legal_summary.can_evolve_count",
    "legal_summary.can_super_evolve_count",
    "legal_summary.can_end_turn",
)

ACTION_VECTOR_FEATURE_ORDER = (
    "action_kind",
    # PlayKind 是互斥枚举。禁用旧 play_kind 字符串哈希编码，改为显式 one-hot：
    # NORMAL / ENHANCE(爆能强化) / ACCELERATE(激奏) / CRYSTAL(结晶)。非出牌动作四位均为 0。
    "is_normal_play",
    "is_enhance",
    "is_accelerate",
    "is_crystal",
    "cost",
    "pp_before_action",
    "unused_pp_after_action",
    "uses_full_pp",
    "leaves_one_pp",
    "target_group_count",
    "target_count",
    "random_target_group_count",
    "random_target_count",
    "selected_index_count",
    "material_count",
    "has_target",
    "has_materials",
    "is_terminal_action",
    "is_resource_action",
    "is_combat_action",
    "is_evolve_action",
    "costs_extra_pp",
    # v10：动作级效果标签（29 维 multi-hot）。定义见 training/action_effect_tags.py。
    # 缺失 key 由 _scalar_feature 读成 0.0，所以 jsonl 里只需存为 1 的标签。
    *tuple(f"effect_tags.{tag}" for tag in ACTION_EFFECT_TAGS),
)

STATE_VECTOR_DIM = len(STATE_VECTOR_FEATURE_ORDER)
ACTION_VECTOR_DIM = len(ACTION_VECTOR_FEATURE_ORDER)
#: ``effect_tags.*`` 在 action scalar 向量中的下标（v10 追加的 29 维）。
#: 消融实验（``--ablate-effect-tags``）用它精确屏蔽这批输入通道。
ACTION_EFFECT_TAG_DIM_INDICES: tuple[int, ...] = tuple(
    index for index, name in enumerate(ACTION_VECTOR_FEATURE_ORDER) if name.startswith("effect_tags.")
)


def state_vector_from_state_features(state_features: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(_scalar_feature(state_features, path) for path in STATE_VECTOR_FEATURE_ORDER)


def action_vectors_from_action_features(action_features: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(_scalar_feature(action, path) for path in ACTION_VECTOR_FEATURE_ORDER) for action in action_features)


def tensorize_training_example(example: Mapping[str, Any]) -> TensorizationResult:
    validate_training_example(example)
    state_features = _mapping(example.get("state_features"))
    action_features = _list(example.get("legal_action_features"))
    state_vector = state_vector_from_state_features(state_features)
    action_vectors = action_vectors_from_action_features(action_features)
    labels = {
        "selected_action_index": example.get("selected_action_index"),
        "policy_target_present": bool(_list(example.get("policy_target"))),
        "value_target_present": example.get("value_target") is not None,
        "policy_target_width": len(_list(example.get("policy_target"))),
    }
    feature_shapes = {
        "state_vector": len(state_vector),
        "action_vectors": [len(vector) for vector in action_vectors],
    }
    result = TensorizationResult(
        schema=TENSORIZATION_SCHEMA,
        version=TENSORIZATION_VERSION,
        sample_id=str(example.get("sample_id") or ""),
        state_vector=state_vector,
        action_vectors=action_vectors,
        policy_target=tuple(float(value) for value in _list(example.get("policy_target"))),
        value_target=_float_or_none(example.get("value_target")),
        labels=labels,
        feature_shapes=feature_shapes,
        metadata=_mapping(example.get("metadata")),
    )
    _validate_tensorization_result(result)
    return result


def tensorize_training_examples(examples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [tensorize_training_example(example).to_jsonable() for example in examples]


def _validate_tensorization_result(result: TensorizationResult) -> None:
    if result.schema != TENSORIZATION_SCHEMA:
        raise ValueError("tensorization schema mismatch")
    if result.version != TENSORIZATION_VERSION:
        raise ValueError("tensorization version mismatch")
    if not result.sample_id:
        raise ValueError("tensorization sample_id must be non-empty")
    if not result.state_vector:
        raise ValueError("tensorization requires state features")
    if not result.action_vectors:
        raise ValueError("tensorization requires action features")
    if len(result.policy_target) != len(result.action_vectors):
        raise ValueError("tensorization policy_target width must match action_vectors length")
    assert_no_leaked_feature_keys(result.to_jsonable())


def _scalar_feature(mapping: Mapping[str, Any], path: str) -> float:
    value = _lookup(mapping, path)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    if value is None:
        return 0.0
    if isinstance(value, str):
        return float(_stable_string_value(value))
    return 0.0


def _lookup(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for segment in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current


def _stable_string_value(value: str) -> int:
    total = 0
    for index, char in enumerate(value):
        total += (index + 1) * ord(char)
    return total


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    raise ValueError("value_target must be numeric or null")
