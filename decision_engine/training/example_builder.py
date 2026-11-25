# -*- coding: utf-8 -*-
"""Build TrainingExample objects from snapshots and legal actions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from decision_engine.actions.model import LegalAction
from decision_engine.state.snapshot import BattleSnapshot
from decision_engine.training.action_features import legal_actions_features
from decision_engine.training.state_features import state_features
from decision_engine.training.training_example import TrainingExample, one_hot_policy_target


def build_training_example(
    *,
    snapshot: BattleSnapshot,
    legal_actions: Sequence[LegalAction],
    sample_id: str,
    source: Mapping[str, Any],
    trust: Mapping[str, Any],
    perspective: str = "me",
    selected_action: LegalAction | None = None,
    selected_action_index: int | None = None,
    policy_target: Sequence[float] | None = None,
    visit_counts: Sequence[int | float] | None = None,
    value_target: float | None = None,
    history_state_features: Sequence[Mapping[str, Any]] | None = None,
    history_actions: Sequence[Mapping[str, Any]] | None = None,
    generation_config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingExample:
    actions = tuple(legal_actions)
    resolved_index = _selected_action_index(
        actions,
        selected_action=selected_action,
        selected_action_index=selected_action_index,
    )
    resolved_policy = _policy_target(
        action_count=len(actions),
        selected_action_index=resolved_index,
        policy_target=policy_target,
        visit_counts=visit_counts,
    )
    generation_config_map = dict(generation_config or {})
    _validate_generation_config(generation_config_map)
    example = TrainingExample(
        sample_id=sample_id,
        source=dict(source),
        trust=dict(trust),
        player=perspective,
        turn_index=_turn_index(snapshot, perspective),
        state_features=state_features(snapshot, perspective=perspective),
        legal_action_features=tuple(legal_actions_features(snapshot, actions, perspective=perspective)),
        selected_action_index=resolved_index,
        policy_target=resolved_policy,
        value_target=value_target,
        history_state_features=tuple(dict(item) for item in (history_state_features or ())),
        history_actions=tuple(dict(item) for item in (history_actions or ())),
        generation_config=generation_config_map,
        metadata=dict(metadata or {}),
    )
    example.validate()
    return example


def _selected_action_index(
    legal_actions: Sequence[LegalAction],
    *,
    selected_action: LegalAction | None,
    selected_action_index: int | None,
) -> int | None:
    if selected_action is not None and selected_action_index is not None:
        if selected_action_index < 0 or selected_action_index >= len(legal_actions):
            raise ValueError("selected_action_index out of range")
        if legal_actions[selected_action_index] != selected_action:
            raise ValueError("selected_action and selected_action_index disagree")
        return selected_action_index
    if selected_action_index is not None:
        return selected_action_index
    if selected_action is None:
        return None
    for index, action in enumerate(legal_actions):
        if action == selected_action:
            return index
    raise ValueError("selected_action is not present in legal_actions")


def _policy_target(
    *,
    action_count: int,
    selected_action_index: int | None,
    policy_target: Sequence[float] | None,
    visit_counts: Sequence[int | float] | None,
) -> tuple[float, ...]:
    explicit_count = sum(item is not None for item in (policy_target, visit_counts))
    if explicit_count > 1:
        raise ValueError("policy_target and visit_counts are mutually exclusive")
    if policy_target is not None:
        return tuple(float(value) for value in policy_target)
    if visit_counts is not None:
        return _normalize_visit_counts(action_count, visit_counts)
    if selected_action_index is not None:
        return one_hot_policy_target(action_count, selected_action_index)
    return ()


def _normalize_visit_counts(action_count: int, visit_counts: Sequence[int | float]) -> tuple[float, ...]:
    if len(visit_counts) != action_count:
        raise ValueError("visit_counts length must equal legal_actions length")
    total = sum(float(value) for value in visit_counts)
    if total <= 0.0:
        raise ValueError("visit_counts must have positive total")
    if any(float(value) < 0.0 for value in visit_counts):
        raise ValueError("visit_counts must be non-negative")
    return tuple(float(value) / total for value in visit_counts)


def _turn_index(snapshot: BattleSnapshot, perspective: str) -> int:
    if perspective == "me":
        return int(snapshot.me.turn or 0)
    if perspective == "opponent":
        return int(snapshot.opponent.turn or 0)
    raise ValueError(f"unsupported training perspective: {perspective}")


def _validate_generation_config(generation_config: Mapping[str, Any]) -> None:
    if not generation_config:
        raise ValueError("generation_config must be non-empty")
    # 受支持的训练来源（封闭枚举）。新增来源必须同时：
    # 1. 在 tools/ 里有对应导出工具；2. 在 docs/CURRENT.md 或对应 plan 文档登记口径。
    # plan_turn_teacher：plan_turn 作为强教师的单步行为克隆样本，
    # 导出工具 tools/export_plan_turn_samples.py。
    if generation_config.get("kind") not in {
        "self_play_search",
        "proof_search",
        "route_sample",
        "policy_value_tensorization",
        "synthetic_heuristic",
        "pvp_real_extraction",
        "plan_turn_teacher",
        "human_pvp_replay",
        # ppo_rollout：PPO on-policy 采样轨迹（策略分布采样，非搜索、非 argmax）。
        # 导出/采集工具 tools/run_ppo_ignition.py，见 docs/plan/PPO_IGNITION_CHECK.md。
        "ppo_rollout",
    }:
        raise ValueError("generation_config kind must be a supported training source")
