# -*- coding: utf-8 -*-
"""Entity backend 的 PUCT/runtime 统一接入（方案 C 阶段 C6）.

C6.1 正确性优先：PUCT / runtime 与训练共用同一 canonical 路径

    BattleSnapshot + LegalAction[]
      -> state_features(perspective) + legal_actions_features
      -> entity_tensorize_example
      -> collate_entity_rows

因此 full path（训练）与 fast path（推理）逐 tensor parity 由构造保证（C6.4）。
任何后续 cache 优化（C6.5）只能在 parity 测试通过后引入。

C6.6 runtime loader 校验 feature schema / entity layout / vocabulary / backend /
information mode。
C6.7 只消费 LegalAction，不生成新动作；prior 只覆盖合法动作。
C6.8 DeckInstance 不可用时按 available-only 契约由上游特征层如实标记，
本模块不静默切旧模型或旧 selector。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from decision_engine.actions.model import LegalAction
from decision_engine.state.snapshot import BattleSnapshot
from decision_engine.training.action_features import legal_actions_features
from decision_engine.training.card_vocabulary import CardVocabulary
from decision_engine.training.state_features import state_features
from decision_engine.training.entity_tensorization import (
    entity_tensorize_example,
    collate_entity_rows,
)


def entity_example_from_snapshot(
    snapshot: BattleSnapshot,
    actions: Sequence[LegalAction],
    *,
    perspective: str = "me",
    vocabulary: CardVocabulary | None = None,
) -> Any:
    """canonical：snapshot + actions -> EntityTensorizationRow（无 labels）。

    与训练路径共用 state_features / legal_actions_features / entity_tensorize_example，
    保证 full/fast 语义一致。
    """
    example = {
        "state_features": state_features(snapshot, perspective=perspective),
        "legal_action_features": legal_actions_features(snapshot, actions, perspective=perspective),
        "policy_target": [0.0] * len(actions),
        "value_target": None,
    }
    return entity_tensorize_example(example, vocabulary=vocabulary)


def entity_batch_from_snapshot(
    snapshot: BattleSnapshot,
    actions: Sequence[LegalAction],
    *,
    perspective: str = "me",
    vocabulary: CardVocabulary | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """canonical：snapshot + actions -> collated entity batch（B=1）。"""
    row = entity_example_from_snapshot(snapshot, actions, perspective=perspective, vocabulary=vocabulary)
    return collate_entity_rows([row], device=device)


def entity_snapshot_fingerprint(snapshot: BattleSnapshot, perspective: str, *, vocabulary: CardVocabulary | None = None) -> tuple:
    """C6.3 state cache fingerprint：包含 deck multiset，但不含 draw order。

    通过对 canonical entity 序列（不含 action 部分）取稳定摘要实现：
    - deck token 已按 base_card_id 升序，draw order 不进入指纹；
    - 不同剩余牌组构成产生不同指纹（deck multiset 命中不同 state cache）。
    """
    row = entity_example_from_snapshot(snapshot, (), perspective=perspective, vocabulary=vocabulary)
    entity_key = tuple(
        (t.card_id_index, t.type_id, t.zone_id, t.owner_id, t.visibility_id, t.position_id,
         round(t.count, 6), round(t.count_ratio, 6))
        for t in row.entities
    )
    return (perspective, tuple(round(v, 6) for v in row.state_vector), entity_key)
