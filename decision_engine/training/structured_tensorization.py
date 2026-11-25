# -*- coding: utf-8 -*-
"""Structured tensorization for card-level features.

Phase N2: 让模型看到单卡 identity（card_id embedding + stat features + set pooling）。

Schema 设计：
- 每张卡用 (base_card_id_index, stat_features...) 表示
- base_card_id_index 是在 vocab 中的索引（0=padding, 1=unknown, 2...=actual cards）
- stat_features 是该卡当前属性的 float vector
- 各 collection (hand, field, opp_field) 用 padded sequence 表示

Card stat features (per card):
    cost, atk, life, max_life,
    card_type (one-hot: follower=1/amulet=2/spell=3/crest=4/countdown_crest=5/faith=6; countdown_amulet=2 兼容),
    has_guard, has_storm, has_rush, has_drain, has_bane, has_ambush,
    has_cant_attack, has_cant_be_attacked, has_cant_select, has_activation,
    can_attack, can_attack_leader, can_evolve,
    attacks_left, countdown, stack, evolve_state,
    spell_boost_count, is_earth_sigil, tribe_count,
    skybound_art_count, super_skybound_art_count,
    variable_x, original_cost
    => 29 dims per card

Action source/target card features: 同一 encoder，共享权重。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from decision_engine.training.card_vocabulary import CardVocabulary, build_card_vocabulary
from decision_engine.training.feature_schema import TRAINING_FEATURE_SCHEMA_VERSION, to_jsonable

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

STRUCTURED_TENSORIZATION_SCHEMA = "StructuredTrainingExampleTensorization"
STRUCTURED_TENSORIZATION_VERSION = 3

# Padding limits (max cards per zone)
MAX_HAND_CARDS = 9  # 影之诗手牌上限
MAX_FIELD_CARDS = 5  # 影之诗场面上限
MAX_OPP_HAND_CARDS = 9  # 对方手牌上限（完美信息模式）
MAX_OPP_FIELD_CARDS = 5
MAX_CRESTS = 5  # 纹章区上限
MAX_ACTION_TARGETS = 5  # 单个动作最大 target 数

# Per-card stat feature dimension (不含 card_id index)
CARD_STAT_DIM = 40

#: 增幅进度饱和度三维在 ``card_stat_vector`` 中的下标（v8 追加在末尾）。
#: 消融实验（`--ablate-boost-progress`）用它精确定位要屏蔽的输入通道；
#: 与 ``card_stat_vector`` 的追加顺序强绑定，改顺序必须同步改这里。
# 增幅进度饱和度三维（schema v8 引入），位于"倒数第五到倒数第三"位。
# 保持这种"从末尾倒数"表达，使新增特征追加在它之后时索引不漂移。
CARD_STAT_BOOST_PROGRESS_INDEX = (CARD_STAT_DIM - 5, CARD_STAT_DIM - 4, CARD_STAT_DIM - 3)
# 进化 / 超进化是否触发额外效果（schema v9 引入），最后两维。
CARD_STAT_EVOLVE_EFFECT_INDEX = (CARD_STAT_DIM - 2, CARD_STAT_DIM - 1)

# Per-crest stat feature dimension (不含 card_id index)
CREST_STAT_DIM = 3  # faith_value, countdown, variable_x

# Card type encoding (向后兼容: follower/amulet/spell 保持 1/2/3 不变)
CARD_TYPE_MAP = {
    "follower": 1.0,
    "amulet": 2.0,
    "countdown_amulet": 2.0,  # 兼容：countdown_amulet 视为 amulet
    "spell": 3.0,
    "crest": 4.0,
    "countdown_crest": 5.0,
    "faith": 6.0,
}

# ──────────────────────────────────────────────────────────────────────
# Vocab
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_VOCABULARY: CardVocabulary | None = None


def default_card_vocabulary() -> CardVocabulary:
    """Return the deterministic base-card vocabulary for standalone callers."""
    global _DEFAULT_VOCABULARY
    if _DEFAULT_VOCABULARY is None:
        db_path = Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"
        _DEFAULT_VOCABULARY = build_card_vocabulary(db_path)
    return _DEFAULT_VOCABULARY


def vocab_size(vocabulary: CardVocabulary | None = None) -> int:
    return (vocabulary or default_card_vocabulary()).vocab_size


def card_id_to_index(card_id: int | None, *, vocabulary: CardVocabulary | None = None) -> int:
    """Legacy alias: callers must pass a base_card_id when identity is known."""
    return (vocabulary or default_card_vocabulary()).index_for_base_card_id(card_id)


# ──────────────────────────────────────────────────────────────────────
# Per-card feature extraction
# ──────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _evolve_effect_table() -> dict[int, tuple[bool, bool]]:
    """base_card_id → (有【进化时】效果, 有【超进化时】效果)。

    数据来自 `card_database.json` 的 `evolve_desc` / `super_evolve_desc`：字段非空
    即表示该卡在对应进化方式下会触发额外效果。826 个随从里 225 个有
    `evolve_desc`、120 个有 `super_evolve_desc`、21 个两者都有。

    为什么需要这个特征：policy 头吃的是 `[state_emb ; action_emb]`，只能看到
    "这是一个 evolve 动作、对象是某张卡"，**看不到"进化它会不会打 3"**。
    实测（`legacy_svhook_..._0001.json` frame 41，坦忒拉&安古蕾雅局）：
    `v8_gen1` 给"进化 5/5 安的巨大英灵" prior 0.364、给"进化 4/4 玛纳利亚密友"
    只有 0.145——它只知道后者身材更小；而用人类数据训出的 policy 把这两个值
    反了过来（0.034 vs 0.160）。这一个布尔位就是那部分信息。
    """
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parent.parent / "data" / "public" / "card_database.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    cards = payload.get("cards") if isinstance(payload, dict) else payload
    out: dict[int, tuple[bool, bool]] = {}
    for card in cards or ():
        if not isinstance(card, dict):
            continue
        bid = int(card.get("base_card_id") or 0)
        if not bid:
            continue
        has_evolve = bool(str(card.get("evolve_desc") or "").strip())
        has_super = bool(str(card.get("super_evolve_desc") or "").strip())
        if has_evolve or has_super:
            out[bid] = (has_evolve, has_super)
    return out


def _evolve_effect_flags(card: Mapping[str, Any]) -> tuple[bool, bool]:
    table = _evolve_effect_table()
    if not table:
        return (False, False)
    bid = int(card.get("base_card_id") or card.get("card_id") or 0)
    if not bid:
        return (False, False)
    return table.get(bid) or table.get(bid // 10 * 10) or (False, False)


def evolve_effect_flags_by_base_card(base_card_id: int) -> tuple[float, float]:
    """供 `planning/_fast_tensorize.py` 复用：base_card_id → 两维 float 标志。

    两条卡特征路径（`structured_tensorization.card_stat_vector` 与
    `_fast_tensorize._encode_card_cached`）必须产出同一套 40 维，这个函数是
    唯一取值入口，避免两处分叉。
    """
    table = _evolve_effect_table()
    if not table or not base_card_id:
        return (0.0, 0.0)
    has_evolve, has_super = table.get(base_card_id) or table.get(base_card_id // 10 * 10) or (False, False)
    return (float(has_evolve), float(has_super))


def card_stat_vector(card: Mapping[str, Any]) -> tuple[float, ...]:
    """从 card features dict 提取 CARD_STAT_DIM 维 float vector。"""
    has_evolve_effect, has_super_evolve_effect = _evolve_effect_flags(card)
    return (
        float(card.get("cost", 0)),
        float(card.get("atk", 0)),
        float(card.get("life", 0)),
        float(card.get("max_life", 0)),
        CARD_TYPE_MAP.get(card.get("card_type", ""), 0.0),
        float(card.get("has_guard", False)),
        float(card.get("has_storm", False)),
        float(card.get("has_rush", False)),
        float(card.get("has_drain", False)),
        float(card.get("has_bane", False)),
        float(card.get("has_ambush", False)),
        float(card.get("has_cant_attack", False)),
        float(card.get("has_cant_be_attacked", False)),
        float(card.get("has_cant_select", False)),
        float(card.get("has_activation", False)),
        float(card.get("can_attack", False)),
        float(card.get("can_attack_leader", False)),
        float(card.get("can_evolve", False)),
        float(card.get("attacks_left", 0)),
        float(card.get("countdown", 0)),
        float(card.get("stack", 0)),
        float(card.get("evolve_state", 0)),
        float(card.get("spell_boost_count", 0)),
        float(card.get("is_earth_sigil", False)),
        float(card.get("tribe_count", 0)),
        float(card.get("skybound_art_count", 0)),
        float(card.get("super_skybound_art_count", 0)),
        float(card.get("variable_x", 0)),
        float(card.get("original_cost", 0)),
        float(card.get("has_conditional_effect", False)),
        float(card.get("conditional_effect_satisfied", False)),
        float(card.get("conditional_effect_unsatisfied", False)),
        float(card.get("conditional_effect_unknown", True)),
        float(card.get("has_discard_payoff", False)),
        float(card.get("has_hand_cost_reduction", False)),
        float(card.get("boost_saturation", 0.0)),
        float(card.get("boost_cap_ratio", 0.0)),
        float(card.get("boost_cap_remaining", 0.0)),
        float(has_evolve_effect),
        float(has_super_evolve_effect),
    )


def crest_stat_vector(crest: Mapping[str, Any]) -> tuple[float, ...]:
    """从 crest features dict 提取 CREST_STAT_DIM 维 float vector。"""
    return (
        float(crest.get("faith_value", 0)),
        float(crest.get("countdown", 0)),
        float(crest.get("variable_x", 0)),
    )


# ──────────────────────────────────────────────────────────────────────
# Structured tensorization result
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StructuredTensorizationResult:
    schema: str
    version: int
    sample_id: str

    # Scalar state (same as dense baseline)
    state_vector: tuple[float, ...]

    # Card sequences: list of (card_id_index, stat_vector)
    hand_card_ids: tuple[int, ...]        # [MAX_HAND_CARDS], vocab indices
    hand_card_stats: tuple[tuple[float, ...], ...]  # [MAX_HAND_CARDS, CARD_STAT_DIM]
    hand_card_mask: tuple[bool, ...]      # [MAX_HAND_CARDS], True if valid

    field_card_ids: tuple[int, ...]
    field_card_stats: tuple[tuple[float, ...], ...]
    field_card_mask: tuple[bool, ...]

    opp_field_card_ids: tuple[int, ...]
    opp_field_card_stats: tuple[tuple[float, ...], ...]
    opp_field_card_mask: tuple[bool, ...]

    # Opponent hand (完美信息模式)
    opp_hand_card_ids: tuple[int, ...]        # [MAX_OPP_HAND_CARDS]
    opp_hand_card_stats: tuple[tuple[float, ...], ...]  # [MAX_OPP_HAND_CARDS, CARD_STAT_DIM]
    opp_hand_card_mask: tuple[bool, ...]      # [MAX_OPP_HAND_CARDS]

    # Crests
    crest_card_ids: tuple[int, ...]        # [MAX_CRESTS]
    crest_stats: tuple[tuple[float, ...], ...]  # [MAX_CRESTS, CREST_STAT_DIM]
    crest_mask: tuple[bool, ...]           # [MAX_CRESTS]
    opp_crest_card_ids: tuple[int, ...]
    opp_crest_stats: tuple[tuple[float, ...], ...]
    opp_crest_mask: tuple[bool, ...]

    # Action-level features
    action_scalar_vectors: tuple[tuple[float, ...], ...]   # [N_actions, ACTION_VECTOR_DIM] same as dense
    action_source_card_ids: tuple[int, ...]                 # [N_actions], vocab indices
    action_source_card_stats: tuple[tuple[float, ...], ...]  # [N_actions, CARD_STAT_DIM]
    action_target_card_ids: tuple[tuple[int, ...], ...]     # [N_actions, MAX_ACTION_TARGETS]
    action_target_card_stats: tuple[tuple[tuple[float, ...], ...], ...]  # [N_actions, MAX_ACTION_TARGETS, CARD_STAT_DIM]
    action_target_mask: tuple[tuple[bool, ...], ...]        # [N_actions, MAX_ACTION_TARGETS]

    # Historical turn-boundary scalar states, oldest -> newest, padded to four.
    history_state_vectors: tuple[tuple[float, ...], ...]
    history_mask: tuple[bool, ...]

    # Labels
    policy_target: tuple[float, ...]
    value_target: float | None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return to_jsonable({
            "schema": self.schema,
            "version": self.version,
            "sample_id": self.sample_id,
            "state_vector": self.state_vector,
            "hand_card_ids": self.hand_card_ids,
            "hand_card_stats": self.hand_card_stats,
            "hand_card_mask": self.hand_card_mask,
            "field_card_ids": self.field_card_ids,
            "field_card_stats": self.field_card_stats,
            "field_card_mask": self.field_card_mask,
            "opp_field_card_ids": self.opp_field_card_ids,
            "opp_field_card_stats": self.opp_field_card_stats,
            "opp_field_card_mask": self.opp_field_card_mask,
            "opp_hand_card_ids": self.opp_hand_card_ids,
            "opp_hand_card_stats": self.opp_hand_card_stats,
            "opp_hand_card_mask": self.opp_hand_card_mask,
            "crest_card_ids": self.crest_card_ids,
            "crest_stats": self.crest_stats,
            "crest_mask": self.crest_mask,
            "opp_crest_card_ids": self.opp_crest_card_ids,
            "opp_crest_stats": self.opp_crest_stats,
            "opp_crest_mask": self.opp_crest_mask,
            "action_scalar_vectors": self.action_scalar_vectors,
            "action_source_card_ids": self.action_source_card_ids,
            "action_source_card_stats": self.action_source_card_stats,
            "action_target_card_ids": self.action_target_card_ids,
            "action_target_card_stats": self.action_target_card_stats,
            "action_target_mask": self.action_target_mask,
            "history_state_vectors": self.history_state_vectors,
            "history_mask": self.history_mask,
            "policy_target": self.policy_target,
            "value_target": self.value_target,
            "metadata": self.metadata,
        })


# ──────────────────────────────────────────────────────────────────────
# Tensorize
# ──────────────────────────────────────────────────────────────────────

_PAD_STAT = tuple(0.0 for _ in range(CARD_STAT_DIM))
_PAD_CREST_STAT = tuple(0.0 for _ in range(CREST_STAT_DIM))


def _pad_card_sequence(cards: list[Mapping[str, Any]], max_len: int, *, vocabulary: CardVocabulary) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    """Pad a card sequence to max_len. Returns (ids, stats, mask)."""
    ids: list[int] = []
    stats: list[tuple[float, ...]] = []
    mask: list[bool] = []

    for i in range(max_len):
        if i < len(cards):
            card = cards[i]
            base_card_id = card.get("base_card_id")
            ids.append(vocabulary.index_for_base_card_id(base_card_id))
            stats.append(card_stat_vector(card))
            mask.append(True)
        else:
            ids.append(0)
            stats.append(_PAD_STAT)
            mask.append(False)

    return tuple(ids), tuple(stats), tuple(mask)


def _pad_crest_sequence(crests: list[Mapping[str, Any]], max_len: int, *, vocabulary: CardVocabulary) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    """Pad a crest sequence to max_len. Returns (ids, stats, mask)."""
    ids: list[int] = []
    stats: list[tuple[float, ...]] = []
    mask: list[bool] = []

    for i in range(max_len):
        if i < len(crests):
            crest = crests[i]
            base_card_id = crest.get("base_card_id") or crest.get("card_id")
            ids.append(vocabulary.index_for_base_card_id(base_card_id))
            stats.append(crest_stat_vector(crest))
            mask.append(True)
        else:
            ids.append(0)
            stats.append(_PAD_CREST_STAT)
            mask.append(False)

    return tuple(ids), tuple(stats), tuple(mask)


def _extract_action_source(action: Mapping[str, Any], *, vocabulary: CardVocabulary) -> tuple[int, tuple[float, ...]]:
    """Extract a source identity from base_card_id plus dynamic card state."""
    source = action.get("source")
    if source and isinstance(source, dict) and source.get("known"):
        return vocabulary.index_for_base_card_id(source.get("base_card_id")), card_stat_vector(source)
    return 0, _PAD_STAT


def _extract_action_targets(action: Mapping[str, Any], *, vocabulary: CardVocabulary) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    """提取动作的 target cards 信息，padded to MAX_ACTION_TARGETS。"""
    targets_raw = action.get("targets", [])
    flat_targets: list[Mapping[str, Any]] = []

    for tg in targets_raw:
        if isinstance(tg, list):
            for t in tg:
                if isinstance(t, dict):
                    flat_targets.append(t)
        elif isinstance(tg, dict):
            flat_targets.append(tg)

    ids: list[int] = []
    stats: list[tuple[float, ...]] = []
    mask: list[bool] = []

    for i in range(MAX_ACTION_TARGETS):
        if i < len(flat_targets):
            t = flat_targets[i]
            ids.append(vocabulary.index_for_base_card_id(t.get("base_card_id")))
            stats.append(card_stat_vector(t))
            mask.append(True)
        else:
            ids.append(0)
            stats.append(_PAD_STAT)
            mask.append(False)

    return tuple(ids), tuple(stats), tuple(mask)


def structured_tensorize_training_example(
    example: Mapping[str, Any],
    *,
    vocabulary: CardVocabulary | None = None,
) -> StructuredTensorizationResult:
    """Tensorize a TrainingExample with stable base-card identity indices."""
    vocabulary = vocabulary or default_card_vocabulary()
    from decision_engine.training.tensorization import (
        state_vector_from_state_features,
        action_vectors_from_action_features,
    )

    state_features = example.get("state_features", {})
    action_features = example.get("legal_action_features", [])

    # Dense state vector (兼容 baseline)
    state_vector = state_vector_from_state_features(state_features)

    # Dense action vectors (兼容 baseline)
    action_scalar_vectors = action_vectors_from_action_features(action_features)

    # Structured: card sequences from state
    player = state_features.get("player", {})
    opponent = state_features.get("opponent", {})

    hand_cards = player.get("hand_cards", [])
    field_cards = player.get("field_cards", [])
    opp_hand_cards = opponent.get("hand_cards", [])
    opp_field_cards = opponent.get("field_cards", [])

    hand_ids, hand_stats, hand_mask = _pad_card_sequence(hand_cards, MAX_HAND_CARDS, vocabulary=vocabulary)
    field_ids, field_stats, field_mask = _pad_card_sequence(field_cards, MAX_FIELD_CARDS, vocabulary=vocabulary)
    opp_hand_ids, opp_hand_stats, opp_hand_mask = _pad_card_sequence(opp_hand_cards, MAX_OPP_HAND_CARDS, vocabulary=vocabulary)
    opp_field_ids, opp_field_stats, opp_field_mask = _pad_card_sequence(opp_field_cards, MAX_OPP_FIELD_CARDS, vocabulary=vocabulary)

    # Crests
    player_crests = player.get("crests", [])
    opp_crests = opponent.get("crests", [])
    crest_ids, crest_stats, crest_mask = _pad_crest_sequence(player_crests, MAX_CRESTS, vocabulary=vocabulary)
    opp_crest_ids, opp_crest_stats, opp_crest_mask = _pad_crest_sequence(opp_crests, MAX_CRESTS, vocabulary=vocabulary)

    # Structured: per-action source/target cards
    action_source_ids: list[int] = []
    action_source_stats: list[tuple[float, ...]] = []
    action_target_ids: list[tuple[int, ...]] = []
    action_target_stats: list[tuple[tuple[float, ...], ...]] = []
    action_target_masks: list[tuple[bool, ...]] = []

    for act in action_features:
        src_id, src_stat = _extract_action_source(act, vocabulary=vocabulary)
        action_source_ids.append(src_id)
        action_source_stats.append(src_stat)

        tgt_ids, tgt_stats, tgt_mask = _extract_action_targets(act, vocabulary=vocabulary)
        action_target_ids.append(tgt_ids)
        action_target_stats.append(tgt_stats)
        action_target_masks.append(tgt_mask)

    # Historical turn-boundary states use the same scalar feature schema as the current state.
    # They are padded at the beginning so valid entries stay oldest -> newest.
    history_features = list(example.get("history_state_features", ()))
    if len(history_features) > 4:
        raise ValueError("history_state_features supports at most four states")
    history_vectors_raw = [state_vector_from_state_features(item) for item in history_features]
    zero_history = tuple(0.0 for _ in state_vector)
    history_state_vectors = tuple([zero_history] * (4 - len(history_vectors_raw)) + history_vectors_raw)
    history_mask = tuple([False] * (4 - len(history_vectors_raw)) + [True] * len(history_vectors_raw))

    # Labels
    policy_target = tuple(float(v) for v in example.get("policy_target", []))
    value_target_raw = example.get("value_target")
    value_target = float(value_target_raw) if value_target_raw is not None else None

    return StructuredTensorizationResult(
        schema=STRUCTURED_TENSORIZATION_SCHEMA,
        version=STRUCTURED_TENSORIZATION_VERSION,
        sample_id=str(example.get("sample_id", "")),
        state_vector=state_vector,
        hand_card_ids=hand_ids,
        hand_card_stats=hand_stats,
        hand_card_mask=hand_mask,
        field_card_ids=field_ids,
        field_card_stats=field_stats,
        field_card_mask=field_mask,
        opp_field_card_ids=opp_field_ids,
        opp_field_card_stats=opp_field_stats,
        opp_field_card_mask=opp_field_mask,
        opp_hand_card_ids=opp_hand_ids,
        opp_hand_card_stats=opp_hand_stats,
        opp_hand_card_mask=opp_hand_mask,
        crest_card_ids=crest_ids,
        crest_stats=crest_stats,
        crest_mask=crest_mask,
        opp_crest_card_ids=opp_crest_ids,
        opp_crest_stats=opp_crest_stats,
        opp_crest_mask=opp_crest_mask,
        action_scalar_vectors=action_scalar_vectors,
        action_source_card_ids=tuple(action_source_ids),
        action_source_card_stats=tuple(action_source_stats),
        action_target_card_ids=tuple(action_target_ids),
        action_target_card_stats=tuple(action_target_stats),
        action_target_mask=tuple(action_target_masks),
        history_state_vectors=history_state_vectors,
        history_mask=history_mask,
        policy_target=policy_target,
        value_target=value_target,
        metadata=dict(example.get("metadata", {})),
    )
