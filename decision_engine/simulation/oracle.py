# -*- coding: utf-8 -*-
"""Oracle 接口骨架。

Oracle 表示未来真实执行反馈来源：调用方提交一个协议动作，Oracle 返回执行后的
BattleSnapshot。这里不连接真实 API、不编码 MessagePack，也不做本地模拟。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol, TypeAlias, runtime_checkable

from decision_engine.actions.model import LegalAction, PlannedAction
from decision_engine.state.snapshot import BattleSnapshot

OracleAction: TypeAlias = PlannedAction | LegalAction
ActionKey: TypeAlias = tuple[str, int, tuple[tuple[int, ...], ...], tuple[int, ...], str | None]
TransitionInput: TypeAlias = Mapping[ActionKey | OracleAction, BattleSnapshot] | Iterable[tuple[ActionKey | OracleAction, BattleSnapshot]]


@runtime_checkable
class Oracle(Protocol):
    """真实执行反馈接口。

    current 返回当前真实快照；apply 提交一个动作并返回执行后的真实快照。
    """

    def current(self) -> BattleSnapshot:
        """返回当前快照。"""

    def apply(self, action: OracleAction) -> BattleSnapshot:
        """提交动作，返回执行后的快照。"""


def action_key(action: OracleAction) -> ActionKey:
    """生成 MockOracle transition 使用的稳定动作 key。

    key 只使用协议动作的核心执行字段：kind/source_uid/targets/materials/play_kind。
    LegalAction.target_groups 与 PlannedAction.target_unique_id_by_skills 会归一到同一字段。
    """

    targets = getattr(action, "target_groups", None)
    if targets is None:
        targets = getattr(action, "target_unique_id_by_skills", ())

    return (
        action.kind.value,
        action.source_uid,
        tuple(tuple(group) for group in targets),
        tuple(action.material_uids),
        action.play_kind.value if action.play_kind is not None else None,
    )


class MockOracle:
    """用于测试 Planner/Loop 的可配置 Oracle。

    MockOracle 只按预配置 transition 切换快照；未匹配动作时抛出异常，避免悄悄退化为本地模拟。
    """

    def __init__(self, current_snapshot: BattleSnapshot, transitions: TransitionInput | None = None):
        self._current_snapshot = current_snapshot
        self._transitions: dict[ActionKey, BattleSnapshot] = {}
        if transitions is not None:
            self.set_transitions(transitions)

    def current(self) -> BattleSnapshot:
        """返回当前快照。"""

        return self._current_snapshot

    def apply(self, action: OracleAction) -> BattleSnapshot:
        """应用已配置 transition，并返回下一快照。"""

        key = action_key(action)
        if key not in self._transitions:
            available = ", ".join(str(item) for item in self._transitions.keys()) or "<none>"
            raise KeyError(f"MockOracle has no transition for action key {key}; available keys: {available}")

        self._current_snapshot = self._transitions[key]
        return self._current_snapshot

    def set_transitions(self, transitions: TransitionInput) -> None:
        """批量设置 transition。

        transitions 可传 dict，也可传 [(action_or_key, snapshot), ...]。
        """

        items = transitions.items() if isinstance(transitions, Mapping) else transitions
        for key_or_action, snapshot in items:
            key = key_or_action if _is_action_key(key_or_action) else action_key(key_or_action)
            self._transitions[key] = snapshot


def _is_action_key(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 5 and isinstance(value[0], str)
