# -*- coding: utf-8 -*-
"""Canonical own remaining-deck features (方案 C 阶段 C1).

从 `DeckInstanceState` 生成确定、无顺序泄漏、perspective-aware 的
「当前 perspective 玩家自己的剩余牌组多重集合」，作为后续 Deck Entity Tokens
的权威输入契约。

设计约束（见 DECK_ENTITY_TRANSFORMER_IMPLEMENTATION_PLAN.md §4）：

- 每个唯一 `base_card_id` 聚合为一个条目，携带 `remaining_count`。
- 不逐实例展开、不输入 `draw_pile` 顺序、不输入 instance id / uid / order_hint /
  visibility / source / warnings。
- `remaining_deck_cards` 按 `base_card_id` 升序确定性排序，仅用于稳定序列化。
- 只从 `draw_pile`（尚未抽取的牌）聚合，不含已抽到手牌/场上/墓地的实例。
- 正式第一版只接受 `status == "available"`；ambiguous / invalid / None /
  count mismatch 显式失败，禁止静默改用 `deck_count` 或猜测。

本模块是纯特征契约，不执行任何规则，也不修改 DeckInstance 生命周期。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from decision_engine.state.deck_instance import DeckInstanceState

# knowledge status 常量
DECK_STATUS_AVAILABLE = "available"


class DeckFeatureError(ValueError):
    """own remaining-deck 特征无法确定性生成时的显式失败。

    禁止在特征层静默回退到 deck_count 或猜测牌组构成；调用方应显式处理
    （拒绝样本 / runtime gate 阻断），而不是吞掉错误。
    """


@dataclass(frozen=True)
class DeckCardEntry:
    base_card_id: int
    remaining_count: int


@dataclass(frozen=True)
class RemainingDeckFeatures:
    """canonical own remaining-deck 特征。"""

    deck_knowledge_status: str
    remaining_deck_count: int
    remaining_deck_cards: tuple[DeckCardEntry, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "deck_knowledge_status": self.deck_knowledge_status,
            "remaining_deck_count": self.remaining_deck_count,
            "remaining_deck_cards": [
                {"base_card_id": entry.base_card_id, "remaining_count": entry.remaining_count}
                for entry in self.remaining_deck_cards
            ],
        }


def remaining_deck_counter(deck_instances: DeckInstanceState) -> Counter[int]:
    """从 draw_pile 聚合尚未抽取的 base_card_id 多重集合。

    只接受 status == "available"，且 draw_pile 中每个 instance_id 必须能在
    instances 中找到对应实例。任何不一致都显式失败。
    """
    if deck_instances is None:  # 防御：类型上不应为 None，调用方已过滤
        raise DeckFeatureError("deck_instances is None")
    if deck_instances.status != DECK_STATUS_AVAILABLE:
        raise DeckFeatureError(
            f"deck knowledge status is not available: {deck_instances.status!r}"
        )

    by_id = deck_instances.by_instance_id()
    remaining: Counter[int] = Counter()
    for instance_id in deck_instances.draw_pile:
        instance = by_id.get(instance_id)
        if instance is None:
            raise DeckFeatureError(
                f"draw_pile references missing instance_id: {instance_id}"
            )
        base_card_id = instance.card.base_card_id
        if base_card_id is None:
            raise DeckFeatureError(
                f"deck instance {instance_id} has no base_card_id"
            )
        remaining[int(base_card_id)] += 1
    return remaining


def build_remaining_deck_features(
    deck_instances: DeckInstanceState | None,
    *,
    deck_count: int | None = None,
) -> RemainingDeckFeatures:
    """构建 canonical own remaining-deck 特征。

    - `deck_instances` 为 None 或非 available 状态：显式失败（DeckFeatureError）。
    - 若提供 `deck_count`，校验 sum(remaining_count) == deck_count；不一致显式失败。
    - remaining_deck_cards 按 base_card_id 升序排序，保证 draw_pile 排列不同、
      多重集合相同的输入产出完全相同的特征。
    """
    if deck_instances is None:
        raise DeckFeatureError("deck_instances is None")

    remaining = remaining_deck_counter(deck_instances)
    total = int(sum(remaining.values()))

    if deck_count is not None and total != int(deck_count):
        raise DeckFeatureError(
            f"remaining deck count {total} != player.deck_count {deck_count}"
        )

    entries = tuple(
        DeckCardEntry(base_card_id=base_card_id, remaining_count=count)
        for base_card_id, count in sorted(remaining.items())
    )
    return RemainingDeckFeatures(
        deck_knowledge_status=DECK_STATUS_AVAILABLE,
        remaining_deck_count=total,
        remaining_deck_cards=entries,
    )
