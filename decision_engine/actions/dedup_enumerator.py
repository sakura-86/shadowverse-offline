# -*- coding: utf-8 -*-
"""去重动作枚举器：合并等价手牌动作以提高搜索效率。

当手中有多张完全相同的卡牌（同 base_card_id、同费用、同身材、同强化费用等），
打哪张对局面没有区别。本模块在枚举后合并这些等价动作，只保留一个代表。

设计原则：
- 仅合并 play 类动作（attack/evolve 等场面动作不合并，uid 决定来源）
- 等价判断基于卡牌的完整功能指纹（排除 uid 等唯一标识）
- 不修改 PlayerLocalActionEnumerator 的枚举逻辑
"""

from __future__ import annotations

from decision_engine.actions.model import LegalAction, ActionKind
from decision_engine.actions.player_local_enumerator import PlayerLocalActionEnumerator
from decision_engine.state.snapshot import BattleSnapshot, CardView


def _card_equivalence_key(card: CardView) -> tuple:
    """提取卡牌的功能指纹，用于判断两张卡是否在搜索中完全等价。

    排除 uid、card_id（仅区分卡面风格）、style_id、attack_targets 等局面相关或纯展示字段。
    包含所有影响出牌效果的属性。
    """
    return (
        card.base_card_id,
        card.cost,
        card.atk,
        card.life,
        card.max_life,
        card.enhance_costs,
        card.accelerate_costs,
        card.crystal_costs,
        card.spell_boost_count,
        card.variable_x,
        card.enhanced_cost,
        card.can_enhance,
        card.is_changed_ability,
        card.has_spell_boost,
        card.card_type,
        tuple(sorted((k, v) for k, v in card.supplement_info.items() if k != "deck_order")) if card.supplement_info else (),
    )


def _action_dedup_key(action: LegalAction, card: CardView | None) -> tuple | None:
    """生成动作的去重 key。仅对 play 类动作生成；其他动作返回 None（不去重）。"""
    if action.kind != ActionKind.PLAY:
        return None
    if card is None:
        return None
    return (
        action.kind,
        action.play_kind,
        action.cost,
        action.target_groups,
        action.selected_indexes,
        action.material_uids,
        _card_equivalence_key(card),
    )


class DeduplicatedActionEnumerator:
    """在 PlayerLocalActionEnumerator 基础上合并等价的手牌出牌动作。

    用法：
        enumerator = DeduplicatedActionEnumerator()
        actions = enumerator.enumerate(snapshot, player="me")
        # actions 中等价的 play 动作已被合并，只保留第一个 uid 的代表
    """

    def __init__(self, base: PlayerLocalActionEnumerator | None = None):
        self.base = base or PlayerLocalActionEnumerator()

    def enumerate(self, snapshot: BattleSnapshot, *, player: str = "me") -> list[LegalAction]:
        raw_actions = self.base.enumerate(snapshot, player=player)
        if not raw_actions:
            return raw_actions

        seen_keys: set[tuple] = set()
        deduped: list[LegalAction] = []

        for action in raw_actions:
            card = snapshot.cards_by_uid.get(action.source_uid)
            key = _action_dedup_key(action, card)

            if key is None:
                # 非 play 动作，不去重
                deduped.append(action)
            elif key not in seen_keys:
                seen_keys.add(key)
                deduped.append(action)
            # else: 等价动作，跳过

        return deduped
