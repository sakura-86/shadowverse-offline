# -*- coding: utf-8 -*-
"""Mulligan 阶段手牌特征编码。

将换牌前的起手状态编码为训练特征。每张手牌编码为固定结构的 dict，
全局上下文（先后手、双方职业）作为独立字段。
"""

from __future__ import annotations

from typing import Any

from decision_engine.state.snapshot import BattleSnapshot, CardView
from decision_engine.training.feature_schema import TRAINING_FEATURE_SCHEMA_VERSION, to_jsonable

MULLIGAN_FEATURES_SCHEMA = "MulliganFeatures"
MULLIGAN_FEATURES_VERSION = 1


def mulligan_hand_card_features(card: CardView | None) -> dict[str, Any]:
    """编码单张手牌的特征。

    用于 mulligan 训练样本中 hand_card_features 的每个元素。
    """
    if card is None:
        return {"known": False}
    return {
        "known": True,
        "base_card_id": card.base_card_id,
        "card_id": card.card_id,
        "cost": card.cost,
        "atk": card.atk,
        "life": card.life,
        "card_type": card.card_type.value,
        # 关键词能力
        "has_guard": card.has_guard,
        "has_storm": card.has_storm,
        "has_rush": card.has_rush,
        "has_drain": card.has_drain,
        "has_bane": card.has_bane,
        "has_ambush": card.has_ambush,
        "has_activation": card.has_activation,
        "has_last_words": card.has_last_words,
        # 特殊机制
        "has_spell_boost": card.has_spell_boost,
        "countdown": card.countdown,
        "tribe_count": len(card.tribes),
    }


def mulligan_state_features(
    snapshot: BattleSnapshot,
    *,
    perspective: str = "me",
    my_class: str = "",
    opponent_class: str = "",
) -> dict[str, Any]:
    """编码换牌阶段的完整特征。

    包括：
    - 全局上下文（先后手、双方职业）
    - 手牌中每张牌的特征

    Args:
        snapshot: 换牌完成后的 snapshot（即 perform_mulligan 之后的状态）。
        perspective: 视角玩家，"me" 或 "opponent"。
        my_class: 视角玩家的职业名（如 "Runecraft"）。
        opponent_class: 对手的职业名。

    Returns:
        特征 dict，可直接用于 MulliganTrainingExample.state_features。
    """
    player = snapshot.me if perspective == "me" else snapshot.opponent
    is_first = player.is_first_side

    hand_cards = [
        mulligan_hand_card_features(snapshot.cards_by_uid.get(uid))
        for uid in player.hand
    ]

    features = {
        "schema": MULLIGAN_FEATURES_SCHEMA,
        "version": MULLIGAN_FEATURES_VERSION,
        "perspective": perspective,
        "is_first": is_first,
        "my_class": my_class,
        "opponent_class": opponent_class,
        "hand_count": len(player.hand),
        "deck_count": player.deck_count,
        "hand_cards": hand_cards,
    }
    return to_jsonable(features)
