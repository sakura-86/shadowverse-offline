# -*- coding: utf-8 -*-
"""BattleSnapshot -> model training state features."""

from __future__ import annotations

from typing import Any

from decision_engine.cards.spellboost import boost_cap_features
from decision_engine.state.snapshot import BattleSnapshot, CardView, ConditionProgress, ConditionState, PlayerView
from decision_engine.training.deck_entity_features import (
    DECK_STATUS_AVAILABLE,
    DeckFeatureError,
    build_remaining_deck_features,
)
from decision_engine.training.feature_schema import TRAINING_FEATURE_SCHEMA_VERSION, assert_no_leaked_feature_keys, to_jsonable
from decision_engine.training.entity_position import position_for_field, position_for_hand


def state_features(snapshot: BattleSnapshot, *, perspective: str = "me", for_inference: bool = False) -> dict[str, Any]:
    player = _player(snapshot, perspective)
    opponent = _player(snapshot, "opponent" if perspective == "me" else "me")
    # is_my_turn: 从当前 perspective 视角看，是否轮到自己行动
    is_my_turn = snapshot.is_ally_turn if perspective == "me" else not snapshot.is_ally_turn
    features = {
        "schema": "StateFeatures",
        "version": TRAINING_FEATURE_SCHEMA_VERSION,
        "perspective": perspective,
        "is_ally_turn": is_my_turn,
        "is_turn_ended": snapshot.is_turn_ended,
        "current_player": "me" if is_my_turn else "opponent",
        # own remaining-deck 特征只对当前 perspective 玩家输出（C1.3/C1.4）；
        # 对手视角下不输出对手真实剩余牌组构成。
        "player": _player_features(player, snapshot, include_remaining_deck=True, self_side=True),
        "opponent": _player_features(opponent, snapshot, include_remaining_deck=False, self_side=False),
        "pressure": _pressure_features(player, opponent, snapshot),
        "legal_summary": _legal_summary(snapshot) if perspective == "me" else _empty_legal_summary(),
    }
    if for_inference:
        # 推理路径(prior/value 张量化)只经 state_vector_from_state_features 读取标量字段,
        # 不需要 to_jsonable 的嵌套结构规整,也不需要训练期防特征泄漏断言。跳过二者以提速。
        # 已验证: state_vector_from_state_features(features) == 对 to_jsonable(features) 的结果
        # (bit-level 一致),故推理向量不受影响。
        return features
    result = to_jsonable(features)
    assert_no_leaked_feature_keys(result)
    return result


def _player(snapshot: BattleSnapshot, perspective: str) -> PlayerView:
    if perspective == "me":
        return snapshot.me
    if perspective == "opponent":
        return snapshot.opponent
    raise ValueError(f"unsupported training perspective: {perspective}")


def _player_features(
    player: PlayerView,
    snapshot: BattleSnapshot,
    *,
    include_remaining_deck: bool = False,
    self_side: bool = True,
) -> dict[str, Any]:
    hand_cards = [
        {
            **card_features(
                snapshot.cards_by_uid.get(uid),
                zone="hand",
                progress=snapshot.condition_progress_by_uid.get(uid),
            ),
            # 固定位置坐标（见 entity_position.py）：手牌按顺序，供 pointer 区分同卡多副本。
            "entity_position": position_for_hand(index),
        }
        for index, uid in enumerate(player.hand)
    ]
    field_cards = [
        {
            **card_features(
                snapshot.cards_by_uid.get(uid),
                zone="field",
                progress=snapshot.condition_progress_by_uid.get(uid),
            ),
            "entity_position": position_for_field(self_side=self_side, position=pos),
        }
        for pos, uid in enumerate(player.field, start=1)
    ]

    # 额外PP是否可用：有额外PP机制、有余量、本回合未使用
    extra_pp = player.extra_pp
    extra_pp_available = (
        extra_pp is not None
        and extra_pp.extra_pp > 0
        and not extra_pp.is_used_extra_pp_this_turn
    )

    features: dict[str, Any] = {
        "life": player.life,
        "max_life": player.max_life,
        "pp": player.pp,
        "max_pp": player.max_pp,
        "ep": player.ep,
        "sep": player.sep,
        "extra_pp_available": extra_pp_available,
        "turn": player.turn,
        "is_first_side": player.is_first_side,
        "hand_count": len(player.hand),
        "field_count": len(player.field),
        "deck_count": player.deck_count,
        "crest_count": len(player.crests),
        # 档B: EffectSpec condition 依赖的不可推导累积量(状态层已维护, condition_evaluator 直读)
        "play_count": player.play_count,
        "graveyard_count": player.graveyard_count,
        "rally": player.rally,
        "evolve_count": player.evolve_count,
        "is_kakusei": player.is_kakusei,
        "super_evolve_unlocked": bool(player.super_evolve_turn > 0 and player.turn >= player.super_evolve_turn),
        "follower_attacked_leader_previous_turn": player.follower_attacked_leader_previous_turn,
        "crests": [_crest_features(crest) for crest in player.crests],
        "hand_cards": hand_cards,
        "field_cards": field_cards,
    }

    if include_remaining_deck:
        features["remaining_deck"] = _remaining_deck_features(player)

    return features


def _remaining_deck_features(player: PlayerView) -> dict[str, Any]:
    """当前 perspective 玩家自己的剩余牌组构成（C1.3）。

    只有 deck_instances.status == "available" 时输出精确 remaining_deck_cards；
    否则如实报告 knowledge status 且不输出任何构成，不静默改用 deck_count。
    真正的数据损坏（available 但 draw_pile 引用缺失 / 计数与 deck_count 不一致）
    由 build_remaining_deck_features 抛出 DeckFeatureError，交由上层显式处理。
    """
    deck_instances = player.deck_instances
    if deck_instances is None:
        return {
            "deck_knowledge_status": "unavailable",
            "remaining_deck_count": 0,
            "remaining_deck_cards": [],
        }
    if deck_instances.status != DECK_STATUS_AVAILABLE:
        return {
            "deck_knowledge_status": deck_instances.status,
            "remaining_deck_count": 0,
            "remaining_deck_cards": [],
        }
    return build_remaining_deck_features(deck_instances, deck_count=player.deck_count).to_jsonable()


def _condition_feature_values(progress: ConditionProgress | None) -> tuple[int, int, int, int, int, int]:
    """把条件进度映射为 6 个 0/1 特征。

    前 4 个是「效果条件」三态；后 2 个是留牌潜在价值的另外两类静态来源
    （被舍弃时发动、在手牌中降费）。progress 为 None 表示没有计算过
    （对手卡、非己方视角），前 4 个输出 unknown=1 而不是 0/0/0。
    """
    if progress is None:
        return 0, 0, 0, 1, 0, 0
    return (
        int(progress.has_conditional_effect),
        int(progress.state is ConditionState.SATISFIED),
        int(progress.state is ConditionState.UNSATISFIED),
        int(progress.state is ConditionState.UNKNOWN),
        int(progress.has_discard_payoff),
        int(progress.has_hand_cost_reduction),
    )


def card_features(card: CardView | None, *, zone: str, progress: ConditionProgress | None = None) -> dict[str, Any]:
    if card is None:
        return {"known": False, "zone": zone}
    (
        has_condition,
        condition_satisfied,
        condition_unsatisfied,
        condition_unknown,
        has_discard_payoff,
        has_hand_cost_reduction,
    ) = _condition_feature_values(progress)
    boost_saturation, boost_cap_ratio, boost_cap_remaining = boost_cap_features(
        card.base_card_id, card.spell_boost_count
    )
    return {
        "known": True,
        "zone": zone,
        "card_id": card.card_id,
        "base_card_id": card.base_card_id,
        "cost": card.cost,
        "atk": card.atk,
        "life": card.life,
        "max_life": card.max_life,
        "card_type": card.card_type.value,
        "has_guard": card.has_guard,
        "has_storm": card.has_storm,
        "has_rush": card.has_rush,
        "has_drain": card.has_drain,
        "has_bane": card.has_bane,
        "has_ambush": card.has_ambush,
        "has_cant_attack": card.has_cant_attack,
        "has_cant_be_attacked": card.has_cant_be_attacked,
        "has_cant_select": card.has_cant_select,
        "has_activation": card.has_activation,
        "activation_used_this_turn": card.activation_used_this_turn,
        "can_attack": card.can_attack,
        "can_attack_leader": card.can_attack_leader,
        "can_evolve": card.can_evolve,
        "can_super_evolve": card.can_super_evolve,
        "attacks_left": card.attacks_left,
        "countdown": card.countdown,
        "stack": card.stack,
        "evolve_state": card.evolve_state,
        "has_spell_boost": card.has_spell_boost,
        "spell_boost_count": card.spell_boost_count,
        # 增幅饱和点（policy 侧）。cap_ratio=0 表示这张牌没有饱和点（线性成长/随从 buff 型），
        # saturation=1 表示已到最高阈值/0 费，此刻打出不再丢失任何可攒的进度。
        # 阈值/费用来源见 cards/spellboost.py，与 value 侧多阈值进度势同源。
        "boost_saturation": boost_saturation,
        "boost_cap_ratio": boost_cap_ratio,
        "boost_cap_remaining": boost_cap_remaining,
        "is_earth_sigil": card.is_earth_sigil,
        "tribe_count": len(card.tribes),
        "skybound_art_count": card.supplement_info.get("skybound_art_count", 0),
        "super_skybound_art_count": card.supplement_info.get("super_skybound_art_count", 0),
        # 档C: 卡级当前 X(增幅计数) 与原始(印刷)费用。original_cost 区别于被增幅/降费后的
        # 当前 cost, 覆盖一整类"原始费用>=N"条件卡(天斧系/原始费用护符等)。
        "variable_x": card.variable_x,
        "original_cost": _original_cost(card),
        # 打出的效果条件进度（本地推导，见 execution/condition_progress.py）。
        # 组合含义：has=0/unknown=0 → 已确认平卡；has=1/satisfied=1 → 条件已达成；
        # has=1/unsatisfied=1 → 有条件但未达成；unknown=1 → 未计算或不可判定。
        "has_conditional_effect": has_condition,
        "conditional_effect_satisfied": condition_satisfied,
        "conditional_effect_unsatisfied": condition_unsatisfied,
        "conditional_effect_unknown": condition_unknown,
        # 留牌潜在价值（与效果条件独立）：被舍弃时发动 / 在手牌中等待降费。
        "has_discard_payoff": has_discard_payoff,
        "has_hand_cost_reduction": has_hand_cost_reduction,
    }


def _original_cost(card: CardView) -> int:
    """卡的原始(印刷)费用, 取 resolver 印刷费用; 卡不在库时回退当前 cost。

    与 condition_evaluator 的 field_max_original_cost / field_has_cost_ge_5_follower
    取法同源(resolver.get_base_by_card_id(card_id).cost, base 为 None 时回退 card.cost),
    保证特征与判定一致。resolver 加载失败不吞异常(属环境错误应显式暴露)。
    """
    base = _card_resolver().get_base_by_card_id(card.card_id)
    return int(base.cost) if base is not None else int(card.cost)


_CARD_RESOLVER_CACHE = None


def _card_resolver():
    global _CARD_RESOLVER_CACHE
    if _CARD_RESOLVER_CACHE is None:
        from decision_engine.execution.generated_card_materializer import default_card_resolver
        _CARD_RESOLVER_CACHE = default_card_resolver()
    return _CARD_RESOLVER_CACHE


def _crest_features(crest: Any) -> dict[str, Any]:
    return {
        "card_id": getattr(crest, "card_id", 0),
        "countdown": getattr(crest, "countdown", 0),
        "faith_value": getattr(crest, "faith_value", 0),
        "variable_x": getattr(crest, "variable_x", 0),
        "style_id": getattr(crest, "style_id", 0),
        "supplement_info": getattr(crest, "supplement_info", 0),
        "buff": getattr(crest, "buff", 0),
        "is_show_skill_notice_effect": getattr(crest, "is_show_skill_notice_effect", False),
        "is_show_skill_notice_icon": getattr(crest, "is_show_skill_notice_icon", False),
        "activated_random_once_indexes": len(getattr(crest, "activated_random_once_indexes", ()) or ()),
        "current_run_in_order_count": getattr(crest, "current_run_in_order_count", 0),
    }


def _pressure_features(player: PlayerView, opponent: PlayerView, snapshot: BattleSnapshot) -> dict[str, Any]:
    player_field = [snapshot.cards_by_uid.get(uid) for uid in player.field]
    opponent_field = [snapshot.cards_by_uid.get(uid) for uid in opponent.field]
    player_hand = [snapshot.cards_by_uid.get(uid) for uid in player.hand]

    # 场面攻击力
    player_field_atk = sum(card.atk for card in player_field if card is not None)
    opponent_field_atk = sum(card.atk for card in opponent_field if card is not None)

    # 场面血量
    player_field_life = sum(card.life for card in player_field if card is not None)
    opponent_field_life = sum(card.life for card in opponent_field if card is not None)

    # 守护
    opponent_guard_count = sum(1 for card in opponent_field if card is not None and card.has_guard)
    player_guard_count = sum(1 for card in player_field if card is not None and card.has_guard)
    player_guard_life = sum(card.life for card in player_field if card is not None and card.has_guard)

    # 节奏：PP 使用率（已使用 PP / max_pp，回合结束时 pp=0 表示完美使用）
    max_pp = player.max_pp if player.max_pp > 0 else 1
    pp_used_ratio = 1.0 - (player.pp / max_pp)

    # 节奏：下回合可打出的手牌数量（cost <= max_pp + 1 的手牌数）
    next_turn_max_pp = player.max_pp + 1  # 下回合 max_pp 增加 1（上限 10）
    if next_turn_max_pp > 10:
        next_turn_max_pp = 10
    playable_next_turn = sum(
        1 for card in player_hand
        if card is not None and card.cost is not None and card.cost <= next_turn_max_pp
    )

    return {
        "player_field_attack_total": player_field_atk,
        "opponent_field_attack_total": opponent_field_atk,
        "opponent_guard_count": opponent_guard_count,
        # 新增特征
        "player_field_life_total": player_field_life,
        "opponent_field_life_total": opponent_field_life,
        "player_guard_count": player_guard_count,
        "player_guard_life_total": player_guard_life,
        "pp_used_ratio": pp_used_ratio,
        "playable_next_turn_count": playable_next_turn,
    }


def _legal_summary(snapshot: BattleSnapshot) -> dict[str, Any]:
    legal = snapshot.legal
    if legal is None:
        return {"present": False}
    return {
        "present": True,
        "can_play_count": len(legal.can_play),
        "can_attack_leader_count": len(legal.can_attack_leader),
        "can_attack_field_count": len(legal.can_attack_field),
        "can_evolve_count": len(legal.can_evolve),
        "can_super_evolve_count": len(legal.can_super_evolve),
        "can_activate_count": len(legal.can_activate),
        "can_fusion_count": len(legal.can_fusion),
        "can_end_turn": legal.can_end_turn,
    }


def _empty_legal_summary() -> dict[str, Any]:
    """对方视角下无法观察到对方的合法动作集。"""
    return {"present": False}
