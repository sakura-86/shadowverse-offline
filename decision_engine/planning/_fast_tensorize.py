# -*- coding: utf-8 -*-
"""Fast tensorization for inference (prior/value) with card-level caching.

Bypasses the full state_features → structured_tensorize → to_jsonable → _collate
pipeline by building tensors directly from BattleSnapshot + LegalAction.

Supports two-level caching:
  - Card cache: (card_uid, dynamic_state_key) → (id_index, stat_list)
  - State cache: managed by caller (search_prior.py) via snapshot fingerprint
"""

from __future__ import annotations

from typing import Sequence

import torch

from decision_engine.cards.spellboost import boost_cap_features
from decision_engine.state.snapshot import BattleSnapshot, CardView, ConditionProgress, ConditionState, PlayerView
from decision_engine.actions.model import LegalAction
from decision_engine.training.card_vocabulary import CardVocabulary
from decision_engine.training.structured_tensorization import (
    card_id_to_index,
    card_stat_vector,
    CARD_TYPE_MAP,
    MAX_HAND_CARDS,
    MAX_FIELD_CARDS,
    MAX_OPP_HAND_CARDS,
    MAX_OPP_FIELD_CARDS,
    MAX_CRESTS,
    MAX_ACTION_TARGETS,
    CARD_STAT_DIM,
    evolve_effect_flags_by_base_card,
    CREST_STAT_DIM,
)
from decision_engine.training.state_features import state_features
from decision_engine.training.action_features import action_features as extract_action_features
from decision_engine.training.tensorization import (
    ACTION_VECTOR_DIM,
    state_vector_from_state_features,
    action_vectors_from_action_features,
)

# ──────────────────────────────────────────────────────────────────────
# Card-level cache
# ──────────────────────────────────────────────────────────────────────

# Card type 数值映射统一使用 structured_tensorization.CARD_TYPE_MAP（权威 full path），
# 避免 full/fast 两条路径出现 spell/amulet 值互换等漂移（见 test_entity_feature_parity）。

# Global card encoding cache: card_state_key → (id_index, stat_list)

_CARD_RESOLVER_CACHE = None


def _fast_original_cost(card: CardView) -> int:
    """与 full tensorization 相同的原始(印刷)费用语义。"""
    global _CARD_RESOLVER_CACHE
    if _CARD_RESOLVER_CACHE is None:
        from decision_engine.execution.generated_card_materializer import default_card_resolver
        _CARD_RESOLVER_CACHE = default_card_resolver()
    base = _CARD_RESOLVER_CACHE.get_base_by_card_id(card.card_id)
    return int(base.cost) if base is not None else int(card.cost)

# Key includes dynamic fields that can change during a game.
_card_encoding_cache: dict[tuple, tuple[int, list[float]]] = {}
_CARD_CACHE_MAX_SIZE = 1024


def clear_card_cache():
    """Clear the global card encoding cache."""
    _card_encoding_cache.clear()


def _condition_stat_values(progress: ConditionProgress | None) -> tuple[float, float, float, float, float, float]:
    """与 training.state_features._condition_feature_values 相同的三态 + 潜在价值语义。"""
    if progress is None:
        return 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
    state = progress.state
    return (
        1.0 if progress.has_conditional_effect else 0.0,
        1.0 if state is ConditionState.SATISFIED else 0.0,
        1.0 if state is ConditionState.UNSATISFIED else 0.0,
        1.0 if state is ConditionState.UNKNOWN else 0.0,
        1.0 if progress.has_discard_payoff else 0.0,
        1.0 if progress.has_hand_cost_reduction else 0.0,
    )


def _card_cache_key(card: CardView, vocabulary: CardVocabulary | None = None, progress: ConditionProgress | None = None) -> tuple:
    """Build a cache key from base identity, dynamic state, and vocabulary contract."""
    return (
        id(vocabulary),
        card.base_card_id,
        card.uid,
        card.spell_boost_count,
        card.atk,
        card.life,
        card.evolve_state,
        card.attacks_left,
        card.countdown,
        card.can_attack,
        card.stack,
        card.variable_x,
        card.supplement_info.get("skybound_art_count", 0),
        card.supplement_info.get("super_skybound_art_count", 0),
        # 条件进度必须进入 key：否则条件达成后仍会命中旧编码，特征永远不更新。
        (
            progress.state.value,
            progress.satisfied,
            progress.unsatisfied,
            progress.unknown,
            progress.has_discard_payoff,
            progress.has_hand_cost_reduction,
        )
        if progress is not None
        else ("none", 0, 0, 1, False, False),
    )


def _encode_card_cached(
    card: CardView,
    *,
    vocabulary: CardVocabulary | None = None,
    progress: ConditionProgress | None = None,
) -> tuple[int, list[float]]:
    """Encode a single card with caching. Returns (id_index, stat_list)."""
    key = _card_cache_key(card, vocabulary, progress)
    result = _card_encoding_cache.get(key)
    if result is not None:
        return result

    id_idx = card_id_to_index(card.base_card_id, vocabulary=vocabulary)
    stats = [
        float(card.cost),
        float(card.atk),
        float(card.life),
        float(card.max_life),
        CARD_TYPE_MAP.get(card.card_type.value, 0.0),
        float(card.has_guard),
        float(card.has_storm),
        float(card.has_rush),
        float(card.has_drain),
        float(card.has_bane),
        float(card.has_ambush),
        float(card.has_cant_attack),
        float(card.has_cant_be_attacked),
        float(card.has_cant_select),
        float(card.has_activation),
        float(card.can_attack),
        float(card.can_attack_leader),
        float(card.can_evolve),
        float(card.attacks_left),
        float(card.countdown),
        float(card.stack),
        float(card.evolve_state),
        float(card.spell_boost_count),
        float(card.is_earth_sigil),
        float(len(card.tribes)),
        float(card.supplement_info.get("skybound_art_count", 0)),
        float(card.supplement_info.get("super_skybound_art_count", 0)),
        float(card.variable_x),
        float(_fast_original_cost(card)),
        *_condition_stat_values(progress),
        *boost_cap_features(card.base_card_id, card.spell_boost_count),
        # schema v9：进化/超进化是否触发额外效果，取值唯一入口见
        # structured_tensorization.evolve_effect_flags_by_base_card。
        *evolve_effect_flags_by_base_card(card.base_card_id),
    ]
    result = (id_idx, stats)

    # Evict if too large
    if len(_card_encoding_cache) >= _CARD_CACHE_MAX_SIZE:
        _card_encoding_cache.clear()

    _card_encoding_cache[key] = result
    return result


# ──────────────────────────────────────────────────────────────────────
# State encoding (card zones + scalar)
# ──────────────────────────────────────────────────────────────────────

def _fill_card_zone_cached(
    uids: Sequence[int],
    cards_by_uid: dict,
    max_len: int,
    ids_out: torch.Tensor,
    stats_out: torch.Tensor,
    mask_out: torch.Tensor,
    batch_idx: int = 0,
    vocabulary: CardVocabulary | None = None,
    progress_by_uid: dict[int, ConditionProgress] | None = None,
):
    """Fill pre-allocated tensors for a card zone using card cache."""
    for i in range(min(len(uids), max_len)):
        card = cards_by_uid.get(uids[i])
        if card is not None:
            id_idx, stat_list = _encode_card_cached(
                card,
                vocabulary=vocabulary,
                progress=progress_by_uid.get(uids[i]) if progress_by_uid else None,
            )
            ids_out[batch_idx, i] = id_idx
            stats_out[batch_idx, i] = torch.tensor(stat_list, dtype=torch.float32)
            mask_out[batch_idx, i] = True


def _fill_crest_zone(
    crests: list,
    max_len: int,
    ids_out: torch.Tensor,
    stats_out: torch.Tensor,
    mask_out: torch.Tensor,
    batch_idx: int = 0,
    vocabulary: CardVocabulary | None = None,
):
    """Fill pre-allocated tensors for crest zone."""
    for i in range(min(len(crests), max_len)):
        crest = crests[i]
        ids_out[batch_idx, i] = card_id_to_index(getattr(crest, "base_card_id", None) or getattr(crest, "card_id", None), vocabulary=vocabulary)
        stats_out[batch_idx, i, 0] = float(getattr(crest, "faith_value", 0))
        stats_out[batch_idx, i, 1] = float(getattr(crest, "countdown", 0))
        stats_out[batch_idx, i, 2] = float(getattr(crest, "variable_x", 0))
        mask_out[batch_idx, i] = True


def snapshot_fingerprint(snapshot: BattleSnapshot, player: str) -> tuple:
    """Compute a lightweight fingerprint for state cache keying.

    Captures zone composition + key scalar state. Two snapshots with the
    same fingerprint will produce identical state tensors.
    """
    me = snapshot.me if player == "me" else snapshot.opponent
    opp = snapshot.opponent if player == "me" else snapshot.me
    # Include card UIDs (zone composition) + dynamic scalars that affect state_vector
    return (
        tuple(me.hand),
        tuple(me.field),
        tuple(opp.hand),
        tuple(opp.field),
        me.life, me.pp, me.max_pp, me.ep, me.sep, me.turn,
        me.play_count, me.rally, me.graveyard_count,
        me.is_kakusei, me.is_evolved_this_turn,
        opp.life, opp.pp, opp.max_pp, opp.ep, opp.turn,
        opp.play_count, opp.rally, opp.graveyard_count,
        len(me.crests), len(opp.crests),
        # Include spellboost counts for hand cards (they change in-zone)
        tuple(
            getattr(snapshot.cards_by_uid.get(uid), "spell_boost_count", 0)
            for uid in me.hand
        ),
        # 条件进度影响卡张量，必须进入缓存指纹，否则条件变化后复用旧张量。
        tuple(
            (
                uid,
                progress.state.value,
                progress.satisfied,
                progress.unsatisfied,
                progress.unknown,
                progress.has_conditional_effect,
                progress.has_discard_payoff,
                progress.has_hand_cost_reduction,
            )
            for uid, progress in sorted(snapshot.condition_progress_by_uid.items())
        ),
    )


def encode_state(
    snapshot: BattleSnapshot,
    player: str,
    device: str = "cpu",
    *,
    vocabulary: CardVocabulary | None = None,
) -> dict[str, torch.Tensor]:
    """Encode state part of the model input (zones + scalar). ~14ms.

    This is the expensive part that can be cached across prior calls
    within the same search step.
    """
    me = snapshot.me if player == "me" else snapshot.opponent
    opp = snapshot.opponent if player == "me" else snapshot.me
    cards_by_uid = snapshot.cards_by_uid
    B = 1

    # State scalar
    sf = state_features(snapshot, perspective=player, for_inference=True)
    state_vector = state_vector_from_state_features(sf)
    state_scalar = torch.tensor([state_vector], dtype=torch.float32, device=device)

    # Hand
    hand_card_ids = torch.zeros(B, MAX_HAND_CARDS, dtype=torch.long, device=device)
    hand_card_stats = torch.zeros(B, MAX_HAND_CARDS, CARD_STAT_DIM, dtype=torch.float32, device=device)
    hand_card_mask = torch.zeros(B, MAX_HAND_CARDS, dtype=torch.bool, device=device)
    _fill_card_zone_cached(me.hand, cards_by_uid, MAX_HAND_CARDS, hand_card_ids, hand_card_stats, hand_card_mask, vocabulary=vocabulary, progress_by_uid=snapshot.condition_progress_by_uid)

    # Field
    field_card_ids = torch.zeros(B, MAX_FIELD_CARDS, dtype=torch.long, device=device)
    field_card_stats = torch.zeros(B, MAX_FIELD_CARDS, CARD_STAT_DIM, dtype=torch.float32, device=device)
    field_card_mask = torch.zeros(B, MAX_FIELD_CARDS, dtype=torch.bool, device=device)
    _fill_card_zone_cached(me.field, cards_by_uid, MAX_FIELD_CARDS, field_card_ids, field_card_stats, field_card_mask, vocabulary=vocabulary, progress_by_uid=snapshot.condition_progress_by_uid)

    # Opp hand
    opp_hand_card_ids = torch.zeros(B, MAX_OPP_HAND_CARDS, dtype=torch.long, device=device)
    opp_hand_card_stats = torch.zeros(B, MAX_OPP_HAND_CARDS, CARD_STAT_DIM, dtype=torch.float32, device=device)
    opp_hand_card_mask = torch.zeros(B, MAX_OPP_HAND_CARDS, dtype=torch.bool, device=device)
    _fill_card_zone_cached(opp.hand, cards_by_uid, MAX_OPP_HAND_CARDS, opp_hand_card_ids, opp_hand_card_stats, opp_hand_card_mask, vocabulary=vocabulary, progress_by_uid=snapshot.condition_progress_by_uid)

    # Opp field
    opp_field_card_ids = torch.zeros(B, MAX_OPP_FIELD_CARDS, dtype=torch.long, device=device)
    opp_field_card_stats = torch.zeros(B, MAX_OPP_FIELD_CARDS, CARD_STAT_DIM, dtype=torch.float32, device=device)
    opp_field_card_mask = torch.zeros(B, MAX_OPP_FIELD_CARDS, dtype=torch.bool, device=device)
    _fill_card_zone_cached(opp.field, cards_by_uid, MAX_OPP_FIELD_CARDS, opp_field_card_ids, opp_field_card_stats, opp_field_card_mask, vocabulary=vocabulary, progress_by_uid=snapshot.condition_progress_by_uid)

    # Crests
    crest_card_ids = torch.zeros(B, MAX_CRESTS, dtype=torch.long, device=device)
    crest_stats = torch.zeros(B, MAX_CRESTS, CREST_STAT_DIM, dtype=torch.float32, device=device)
    crest_mask = torch.zeros(B, MAX_CRESTS, dtype=torch.bool, device=device)
    _fill_crest_zone(me.crests, MAX_CRESTS, crest_card_ids, crest_stats, crest_mask, vocabulary=vocabulary)

    opp_crest_card_ids = torch.zeros(B, MAX_CRESTS, dtype=torch.long, device=device)
    opp_crest_stats = torch.zeros(B, MAX_CRESTS, CREST_STAT_DIM, dtype=torch.float32, device=device)
    opp_crest_mask = torch.zeros(B, MAX_CRESTS, dtype=torch.bool, device=device)
    _fill_crest_zone(opp.crests, MAX_CRESTS, opp_crest_card_ids, opp_crest_stats, opp_crest_mask, vocabulary=vocabulary)

    return {
        "state_scalar": state_scalar,
        "hand_card_ids": hand_card_ids,
        "hand_card_stats": hand_card_stats,
        "hand_card_mask": hand_card_mask,
        "field_card_ids": field_card_ids,
        "field_card_stats": field_card_stats,
        "field_card_mask": field_card_mask,
        "opp_hand_card_ids": opp_hand_card_ids,
        "opp_hand_card_stats": opp_hand_card_stats,
        "opp_hand_card_mask": opp_hand_card_mask,
        "opp_field_card_ids": opp_field_card_ids,
        "opp_field_card_stats": opp_field_card_stats,
        "opp_field_card_mask": opp_field_card_mask,
        "crest_card_ids": crest_card_ids,
        "crest_stats": crest_stats,
        "crest_mask": crest_mask,
        "opp_crest_card_ids": opp_crest_card_ids,
        "opp_crest_stats": opp_crest_stats,
        "opp_crest_mask": opp_crest_mask,
    }


# ──────────────────────────────────────────────────────────────────────
# Action encoding
# ──────────────────────────────────────────────────────────────────────

def encode_actions(
    snapshot: BattleSnapshot,
    actions: Sequence[LegalAction],
    player: str,
    device: str = "cpu",
    *,
    vocabulary: CardVocabulary | None = None,
) -> dict[str, torch.Tensor]:
    """Encode action part of the model input. ~5ms for 7 actions."""
    B = 1
    max_actions = len(actions)

    af_list = [extract_action_features(snapshot, action, perspective=player) for action in actions]
    action_scalar_vectors = action_vectors_from_action_features(af_list)
    action_width = len(action_scalar_vectors[0]) if action_scalar_vectors else ACTION_VECTOR_DIM

    action_scalars = torch.zeros(B, max_actions, action_width, dtype=torch.float32, device=device)
    action_mask_t = torch.zeros(B, max_actions, dtype=torch.bool, device=device)
    action_source_ids = torch.zeros(B, max_actions, dtype=torch.long, device=device)
    action_source_stats = torch.zeros(B, max_actions, CARD_STAT_DIM, dtype=torch.float32, device=device)
    action_target_ids = torch.zeros(B, max_actions, MAX_ACTION_TARGETS, dtype=torch.long, device=device)
    action_target_stats = torch.zeros(B, max_actions, MAX_ACTION_TARGETS, CARD_STAT_DIM, dtype=torch.float32, device=device)
    action_target_mask = torch.zeros(B, max_actions, MAX_ACTION_TARGETS, dtype=torch.bool, device=device)

    for i, (af, sv) in enumerate(zip(af_list, action_scalar_vectors)):
        action_scalars[0, i] = torch.tensor(sv, dtype=torch.float32)
        action_mask_t[0, i] = True

        # Source card
        source = af.get("source")
        if source and isinstance(source, dict) and source.get("known"):
            action_source_ids[0, i] = card_id_to_index(source.get("base_card_id"), vocabulary=vocabulary)
            action_source_stats[0, i] = torch.tensor(_card_stat_from_dict(source), dtype=torch.float32)

        # Target cards
        targets_raw = af.get("targets", [])
        flat_targets = []
        for tg in targets_raw:
            if isinstance(tg, list):
                for t in tg:
                    if isinstance(t, dict):
                        flat_targets.append(t)
            elif isinstance(tg, dict):
                flat_targets.append(tg)

        for j in range(min(len(flat_targets), MAX_ACTION_TARGETS)):
            t = flat_targets[j]
            action_target_ids[0, i, j] = card_id_to_index(t.get("base_card_id"), vocabulary=vocabulary)
            action_target_stats[0, i, j] = torch.tensor(_card_stat_from_dict(t), dtype=torch.float32)
            action_target_mask[0, i, j] = True

    # Dummy policy/value targets (not used in inference)
    policy = torch.zeros(B, max_actions, dtype=torch.float32, device=device)
    value = torch.zeros(B, dtype=torch.float32, device=device)
    value_mask = torch.zeros(B, dtype=torch.bool, device=device)

    return {
        "action_scalars": action_scalars,
        "action_mask": action_mask_t,
        "action_source_ids": action_source_ids,
        "action_source_stats": action_source_stats,
        "action_target_ids": action_target_ids,
        "action_target_stats": action_target_stats,
        "action_target_mask": action_target_mask,
        "policy": policy,
        "value": value,
        "value_mask": value_mask,
    }


# ──────────────────────────────────────────────────────────────────────
# Combined (backward-compatible entry point)
# ──────────────────────────────────────────────────────────────────────

def fast_tensorize_for_inference(
    snapshot: BattleSnapshot,
    actions: Sequence[LegalAction],
    *,
    player: str = "me",
    device: str = "cpu",
    vocabulary: CardVocabulary | None = None,
) -> dict[str, torch.Tensor]:
    """Build model input tensors directly from snapshot + actions.

    This is the combined entry point (state + actions). For cached usage,
    call encode_state() and encode_actions() separately.
    """
    state_tensors = encode_state(snapshot, player, device, vocabulary=vocabulary)
    action_tensors = encode_actions(snapshot, actions, player, device, vocabulary=vocabulary)
    return {**state_tensors, **action_tensors}


# turn_end dummy action feature，用于 value 评估时提供 forward 所需的最小 action 输入。
# value 输出只依赖 state（forward 中 value_head 走 encode_value_state(state_emb, batch)），
# action 部分仅为满足 forward 的 action_mask/action_emb shape 契约，取值不影响 value。
# 该 dict 与旧 search_value dummy_example 的 turn_end action feature 保持一致。
_VALUE_DUMMY_ACTION_FEATURE = {
    "action_kind": "turn_end", "play_kind": "none", "cost": 0,
    "pp_before_action": 0, "unused_pp_after_action": 0, "uses_full_pp": False,
    "leaves_one_pp": False, "source": {}, "targets": [], "target_group_count": 0,
    "target_count": 0, "random_target_group_count": 0, "random_target_count": 0,
    "selected_index_count": 0, "material_count": 0, "has_target": False,
    "has_materials": False, "is_terminal_action": True, "is_resource_action": False,
    "is_combat_action": False, "is_evolve_action": False,
    "schema": "ActionFeatures", "version": 1, "variant": None,
}


def encode_value_input(
    snapshot: BattleSnapshot,
    player: str,
    device: str = "cpu",
    *,
    vocabulary: CardVocabulary | None = None,
    state_tensors: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Build model input for value evaluation via the fast path.

    复用 encode_state（可传入缓存的 state_tensors 复用 prior 的 state cache），
    action 部分用单个 turn_end dummy（value 输出不依赖 action），完全跳过
    state_features→structured_tensorize→to_jsonable→_collate 的重管线。
    """
    if state_tensors is None:
        state_tensors = encode_state(snapshot, player, device, vocabulary=vocabulary)

    action_scalar_vectors = action_vectors_from_action_features([_VALUE_DUMMY_ACTION_FEATURE])
    action_width = len(action_scalar_vectors[0]) if action_scalar_vectors else ACTION_VECTOR_DIM
    action_scalars = torch.zeros(1, 1, action_width, dtype=torch.float32, device=device)
    action_scalars[0, 0] = torch.tensor(action_scalar_vectors[0], dtype=torch.float32)
    action_mask_t = torch.zeros(1, 1, dtype=torch.bool, device=device)
    action_mask_t[0, 0] = True
    action_tensors = {
        "action_scalars": action_scalars,
        "action_mask": action_mask_t,
        "action_source_ids": torch.zeros(1, 1, dtype=torch.long, device=device),
        "action_source_stats": torch.zeros(1, 1, CARD_STAT_DIM, dtype=torch.float32, device=device),
        "action_target_ids": torch.zeros(1, 1, MAX_ACTION_TARGETS, dtype=torch.long, device=device),
        "action_target_stats": torch.zeros(1, 1, MAX_ACTION_TARGETS, CARD_STAT_DIM, dtype=torch.float32, device=device),
        "action_target_mask": torch.zeros(1, 1, MAX_ACTION_TARGETS, dtype=torch.bool, device=device),
        "policy": torch.zeros(1, 1, dtype=torch.float32, device=device),
        "value": torch.zeros(1, dtype=torch.float32, device=device),
        "value_mask": torch.zeros(1, dtype=torch.bool, device=device),
    }
    return {**state_tensors, **action_tensors}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _card_stat_from_dict(card: dict) -> list[float]:
    """Extract CARD_STAT_DIM floats from a card features dict.

    直接复用 structured 权威实现 card_stat_vector，保证 full/fast 两条路径
    的 per-card stat 契约（含奥义值维度）不漂移。
    """
    return list(card_stat_vector(card))
