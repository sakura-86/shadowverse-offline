# -*- coding: utf-8 -*-
"""Entity tensorization contract (方案 C 阶段 C2).

把 `state_features()` 产出的结构化特征转换为一条「可变长 entity 序列」，作为
Cross-Zone Entity Transformer (C3) 与 Action-to-Entity Cross-Attention (C4) 的
权威输入契约。训练、PUCT fast path 与 runtime 必须共用本契约（parity 见 C6）。

序列布局（每条样本）：

    token 0        : [STATE]  —— 始终有效，携带 scalar state 投影所需字段
    token 1..H     : self hand cards
    ...            : self field / opp hand / opp field / self crest / opp crest
    ...            : self remaining-deck unique cards（每个唯一 base_card_id 一个 token）

每个 entity token 携带：

    card_id_index          : vocabulary 索引（card 类实体）
    card_stat[CARD_STAT_DIM]: card 类实体的 stat 向量（crest 实体为 0）
    crest_stat[CREST_STAT_DIM]: crest 类实体的 stat 向量（card 实体为 0）
    type_id / zone_id / owner_id / visibility_id : 稳定枚举 ID
    position_id            : field 保留槽位位置；hand/deck/crest/state 为 0
    count                  : deck 实体的 remaining_count；其他为 1
    count_ratio            : deck 实体的 remaining_count / remaining_deck_count；其他为 0

约束：

- deck / hand token 的排列不改变 canonical 语义（deck 已按 base_card_id 升序，
  hand 不加位置）；field 通过 position_id 可区分。
- 不硬编码 unique deck type 上限；collate 时按 batch 最大实体数动态 padding。
- padding token mask=False，不得作为 attention key/value。
- 不含 uid / instance_id / draw order / after_context 等泄漏字段。
- 与结构化 stat 契约共享 CARD_STAT_DIM / CREST_STAT_DIM / CARD_TYPE_MAP，避免漂移。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from decision_engine.training.card_vocabulary import CardVocabulary
from decision_engine.training.structured_tensorization import (
    CARD_STAT_DIM,
    CARD_TYPE_MAP,
    CREST_STAT_DIM,
    MAX_ACTION_TARGETS,
    card_stat_vector,
    crest_stat_vector,
    default_card_vocabulary,
)
from decision_engine.training.tensorization import ACTION_VECTOR_DIM
from decision_engine.training.slot_spec_vocabulary import SlotSpecVocabulary, default_slot_spec_vocabulary
from decision_engine.training.entity_position import (
    ENTITY_POSITION_OPP_LEADER,
    ENTITY_POSITION_PAD,
    ENTITY_POSITION_SELF_LEADER,
    MAX_ENTITY_POSITIONS,
    PHASE_PAD,
    PHASE_SOURCE_POSITIONS,
    phase_index,
    position_for_crest,
    position_for_field,
    position_for_hand,
)

# ──────────────────────────────────────────────────────────────────────
# Schema / version
# ──────────────────────────────────────────────────────────────────────

ENTITY_TENSORIZATION_SCHEMA = "EntityTensorization"
ENTITY_TENSORIZATION_VERSION = 1
#: layout 2：新增 history-action token（最近 K 步动作作为实体序列的一部分）。
#: 旧 row 会被 assert_entity_schema 显式拒绝——这是有意的版本闸，不是兼容性问题。
ENTITY_LAYOUT_VERSION = 2

#: 最近 K 步动作进入 entity 序列。4 步覆盖"本回合内 combo 链"，
#: 跨多回合的长程依赖交给 value head，不塞进输入。
HISTORY_ACTION_LIMIT = 4

#: 单个动作的最大目标槽数。实测数据里 slot 数分布是 0/1/2/3（只有 1 张卡有 3 个），
#: 这里留一格裕量。超出的槽会被截断——截断必须是**显式**的，
#: 见 ``_extract_action_slots`` 的校验。
MAX_ACTION_SLOTS = 4

# ──────────────────────────────────────────────────────────────────────
# 稳定 ID 枚举（append-only；新增只能在末尾追加，不得重排既有值）
# 0 一律保留为 padding / none。
# ──────────────────────────────────────────────────────────────────────

# entity type
ENTITY_TYPE_PAD = 0
ENTITY_TYPE_STATE = 1
ENTITY_TYPE_CARD = 2
ENTITY_TYPE_CREST = 3
ENTITY_TYPE_DECK_CARD = 4
#: 5 = 历史动作 token（最近 K 步）。``card_id_index`` 放动作来源卡，
#: ``count`` 放动作 kind 编码，``position_id`` 放步序（1 = 最近一步）。
ENTITY_TYPE_HISTORY_ACTION = 5

# zone
ZONE_PAD = 0
ZONE_STATE = 1
ZONE_HAND = 2
ZONE_FIELD = 3
ZONE_CREST = 4
ZONE_DECK = 5
#: 6 = 历史动作序列（不是场上区域，只为让 type/zone 组合可区分）。
ZONE_HISTORY = 6

# owner / side（相对当前 perspective）
OWNER_PAD = 0
OWNER_SELF = 1
OWNER_OPPONENT = 2
OWNER_NEUTRAL = 3  # [STATE]

# visibility
VIS_PAD = 0
VIS_VISIBLE = 1
VIS_HIDDEN = 2

_PAD_CARD_STAT = tuple(0.0 for _ in range(CARD_STAT_DIM))
_PAD_CREST_STAT = tuple(0.0 for _ in range(CREST_STAT_DIM))

#: 历史动作 kind → 数值（写进 history-action token 的 ``count``）。append-only。
ACTION_KIND_IDS: dict[str, int] = {
    "turn_end": 1,
    "play": 2,
    "attack_follower": 3,
    "attack_leader": 4,
    "evolve": 5,
    "super_evolve": 6,
    "activate": 7,
    "extra_pp": 8,
    "fusion": 9,
    "turn_start": 10,
}


# ──────────────────────────────────────────────────────────────────────
# Entity token
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EntityToken:
    card_id_index: int
    card_stat: tuple[float, ...]
    crest_stat: tuple[float, ...]
    type_id: int
    zone_id: int
    owner_id: int
    visibility_id: int
    position_id: int
    count: float
    count_ratio: float
    #: 实体在**固定位置坐标系**里的坐标（见 ``entity_position.py``）。
    #: 生成式 pointer 用它把「第 P 个位置」与实体 token 关联起来。
    #: 默认 ``ENTITY_POSITION_PAD``：旧 row 没有该字段，加载后坐标为无效值，
    #: pointer 训练会据此跳过这些实体。
    entity_position: int = ENTITY_POSITION_PAD

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "card_id_index": self.card_id_index,
            "card_stat": list(self.card_stat),
            "crest_stat": list(self.crest_stat),
            "type_id": self.type_id,
            "zone_id": self.zone_id,
            "owner_id": self.owner_id,
            "visibility_id": self.visibility_id,
            "position_id": self.position_id,
            "count": self.count,
            "count_ratio": self.count_ratio,
            "entity_position": self.entity_position,
        }


@dataclass(frozen=True)
class EntityTensorizationRow:
    schema: str
    version: int
    entity_layout_version: int
    state_vector: tuple[float, ...]
    entities: tuple[EntityToken, ...]

    # action tensors（沿用结构化 action 契约，供 C4 action query 使用）
    action_scalar_vectors: tuple[tuple[float, ...], ...]
    action_source_card_ids: tuple[int, ...]
    action_source_card_stats: tuple[tuple[float, ...], ...]
    action_target_card_ids: tuple[tuple[int, ...], ...]
    action_target_card_stats: tuple[tuple[tuple[float, ...], ...], ...]
    action_target_mask: tuple[tuple[bool, ...], ...]

    #: 每个 target 属于第几个目标槽（与 ``action_target_card_ids`` 逐位对齐，-1 = pad）。
    #: ``targets`` 在 ``action_features`` 里是平铺的，这里用槽索引还原分组。
    action_target_slot_index: tuple[tuple[int, ...], ...] = ()
    #: 每个动作的槽 spec 词表下标（``SlotSpecVocabulary``），0 = pad。
    action_slot_spec_ids: tuple[tuple[int, ...], ...] = ()
    #: 槽 mask（True = 该槽存在），与 ``action_slot_spec_ids`` 对齐。
    action_slot_mask: tuple[tuple[bool, ...], ...] = ()
    #: 实体的固定位置坐标（与 ``entities`` 逐位对齐，-1 表示无坐标）。
    entity_positions: tuple[int, ...] = ()
    #: 动作 source 的位置坐标（生成式 pointer 的监督信号）。
    action_source_positions: tuple[int, ...] = ()
    #: 动作各 target 的位置坐标（与 ``action_target_card_ids`` 逐位对齐，-1 = pad）。
    action_target_positions: tuple[tuple[int, ...], ...] = ()
    #: 动作的 pointer 动作头下标（``ActionHead`` 枚举序号，-1 = 无法映射）。
    action_head_indices: tuple[int, ...] = ()
    #: 回合内阶段（``play`` / ``evolve`` / ``exchange``）。None = 未知（旧数据）。
    #: phase 是**策略所处的阶段**，不是状态属性，必须由数据生成侧记录。
    phase: str | None = None

    policy_target: tuple[float, ...] = ()
    value_target: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "entity_layout_version": self.entity_layout_version,
            "state_vector": list(self.state_vector),
            "entities": [tok.to_jsonable() for tok in self.entities],
            "action_scalar_vectors": [list(v) for v in self.action_scalar_vectors],
            "action_source_card_ids": list(self.action_source_card_ids),
            "action_source_card_stats": [list(v) for v in self.action_source_card_stats],
            "action_target_card_ids": [list(v) for v in self.action_target_card_ids],
            "action_target_card_stats": [[list(t) for t in row] for row in self.action_target_card_stats],
            "action_target_mask": [list(v) for v in self.action_target_mask],
            "action_target_slot_index": [list(v) for v in self.action_target_slot_index],
            "action_slot_spec_ids": [list(v) for v in self.action_slot_spec_ids],
            "action_slot_mask": [list(v) for v in self.action_slot_mask],
            "entity_positions": list(self.entity_positions),
            "action_source_positions": list(self.action_source_positions),
            "action_target_positions": [list(v) for v in self.action_target_positions],
            "action_head_indices": list(self.action_head_indices),
            "phase": self.phase,
            "policy_target": list(self.policy_target),
            "value_target": self.value_target,
            "metadata": dict(self.metadata),
        }


# ──────────────────────────────────────────────────────────────────────
# Feature -> entity tokens
# ──────────────────────────────────────────────────────────────────────

def _card_type_value(card_features: Mapping[str, Any]) -> float:
    return CARD_TYPE_MAP.get(card_features.get("card_type", ""), 0.0)


def _card_token(
    card_features: Mapping[str, Any],
    *,
    zone_id: int,
    owner_id: int,
    visibility_id: int,
    position_id: int,
    type_id: int,
    vocabulary: CardVocabulary,
    entity_position: int = ENTITY_POSITION_PAD,
) -> EntityToken:
    return EntityToken(
        card_id_index=vocabulary.index_for_base_card_id(card_features.get("base_card_id")),
        card_stat=card_stat_vector(card_features),
        crest_stat=_PAD_CREST_STAT,
        type_id=type_id,
        zone_id=zone_id,
        owner_id=owner_id,
        visibility_id=visibility_id,
        position_id=position_id,
        count=1.0,
        count_ratio=0.0,
        entity_position=entity_position,
    )


def _crest_token(
    crest_features: Mapping[str, Any],
    *,
    owner_id: int,
    vocabulary: CardVocabulary,
    entity_position: int = ENTITY_POSITION_PAD,
) -> EntityToken:
    base_id = crest_features.get("base_card_id")
    if base_id is None:
        base_id = crest_features.get("card_id")
    return EntityToken(
        card_id_index=vocabulary.index_for_base_card_id(base_id),
        card_stat=_PAD_CARD_STAT,
        crest_stat=crest_stat_vector(crest_features),
        type_id=ENTITY_TYPE_CREST,
        zone_id=ZONE_CREST,
        owner_id=owner_id,
        visibility_id=VIS_VISIBLE,
        position_id=0,
        count=1.0,
        count_ratio=0.0,
        entity_position=entity_position,
    )


def _deck_token(base_card_id: int, remaining_count: int, remaining_total: int, *, vocabulary: CardVocabulary) -> EntityToken:
    ratio = float(remaining_count) / float(remaining_total) if remaining_total > 0 else 0.0
    return EntityToken(
        card_id_index=vocabulary.index_for_base_card_id(base_card_id),
        card_stat=_PAD_CARD_STAT,
        crest_stat=_PAD_CREST_STAT,
        type_id=ENTITY_TYPE_DECK_CARD,
        zone_id=ZONE_DECK,
        owner_id=OWNER_SELF,
        visibility_id=VIS_HIDDEN,
        position_id=0,
        count=float(remaining_count),
        count_ratio=ratio,
    )


def _state_token() -> EntityToken:
    """[STATE] token：始终有效，scalar state 通过 state_vector 单独投影。"""
    return EntityToken(
        card_id_index=0,
        card_stat=_PAD_CARD_STAT,
        crest_stat=_PAD_CREST_STAT,
        type_id=ENTITY_TYPE_STATE,
        zone_id=ZONE_STATE,
        owner_id=OWNER_NEUTRAL,
        visibility_id=VIS_VISIBLE,
        position_id=0,
        count=1.0,
        count_ratio=0.0,
    )


def build_entity_tokens(
    state_features_payload: Mapping[str, Any],
    *,
    vocabulary: CardVocabulary,
    history_actions: Sequence[Mapping[str, Any]] = (),
) -> tuple[EntityToken, ...]:
    """从 state_features 输出构建 canonical entity 序列（含 [STATE] token）。

    ``history_actions`` 是**最近 K 步已执行动作**（按时间顺序，最早在前），
    会被追加成 history-action token（最多 ``HISTORY_ACTION_LIMIT`` 个，取最后 K 个）。

    为什么单独传而不是塞进 state_features：动作不是状态。状态特征回答"现在盘面怎样"，
    历史动作回答"我是怎么走到这里的"——后者是 policy 学 combo 依赖所必需的输入，
    但混进 state_features 会让"同一盘面"在不同到达路径下指纹不同，破坏 memo/缓存契约。

    信息边界：对手手牌内容不产出 token（见下方 opp hand 注释）；对手的**历史动作**
    是公开信息（打出的牌、攻击目标都可见），所以可以带 ``is_own=False`` 进序列。
    """
    player = state_features_payload.get("player", {})
    opponent = state_features_payload.get("opponent", {})

    tokens: list[EntityToken] = [_state_token()]

    # self hand（不加位置 embedding，但保留固定坐标供 pointer 使用）
    for index, card in enumerate(player.get("hand_cards", [])):
        tokens.append(_card_token(card, zone_id=ZONE_HAND, owner_id=OWNER_SELF,
                                  visibility_id=VIS_VISIBLE, position_id=0,
                                  type_id=ENTITY_TYPE_CARD, vocabulary=vocabulary,
                                  entity_position=position_for_hand(index)))
    # self field（保留槽位位置，position_id 从 1 起）
    for pos, card in enumerate(player.get("field_cards", []), start=1):
        tokens.append(_card_token(card, zone_id=ZONE_FIELD, owner_id=OWNER_SELF,
                                  visibility_id=VIS_VISIBLE, position_id=pos,
                                  type_id=ENTITY_TYPE_CARD, vocabulary=vocabulary,
                                  entity_position=position_for_field(self_side=True, position=pos)))
    # opp hand：**不输出 token**。影之诗的对手手牌是隐藏信息，把它作为 VIS_VISIBLE
    # entity 输入会让训练期看到实机拿不到的信息（训练/推理分布不一致，实机必崩）。
    # 手牌张数不丢：``opponent.hand_count`` 已在 state_vector 的标量列表里
    # （见 tensorization.state_scalar_paths）。已公开的对手信息（场上随从、纹章、
    # 本回合已打出张数等）继续通过 opponent.field_cards / crests / 标量特征表达。
    # opp field
    for pos, card in enumerate(opponent.get("field_cards", []), start=1):
        tokens.append(_card_token(card, zone_id=ZONE_FIELD, owner_id=OWNER_OPPONENT,
                                  visibility_id=VIS_VISIBLE, position_id=pos,
                                  type_id=ENTITY_TYPE_CARD, vocabulary=vocabulary,
                                  entity_position=position_for_field(self_side=False, position=pos)))
    # self crest
    for index, crest in enumerate(player.get("crests", [])):
        tokens.append(_crest_token(crest, owner_id=OWNER_SELF, vocabulary=vocabulary,
                                   entity_position=position_for_crest(self_side=True, index=index)))
    # opp crest
    for index, crest in enumerate(opponent.get("crests", [])):
        tokens.append(_crest_token(crest, owner_id=OWNER_OPPONENT, vocabulary=vocabulary,
                                   entity_position=position_for_crest(self_side=False, index=index)))

    # self remaining-deck unique cards（仅当 knowledge status == available）
    remaining_deck = player.get("remaining_deck")
    if remaining_deck and remaining_deck.get("deck_knowledge_status") == "available":
        total = int(remaining_deck.get("remaining_deck_count", 0))
        for entry in remaining_deck.get("remaining_deck_cards", []):
            tokens.append(_deck_token(
                int(entry["base_card_id"]), int(entry["remaining_count"]), total,
                vocabulary=vocabulary,
            ))

    # 历史动作 token（追加在序列末尾，不改变既有 token 的相对顺序）
    recent = tuple(history_actions)[-HISTORY_ACTION_LIMIT:]
    for step_index, act in enumerate(reversed(recent), start=1):
        kind = str(act.get("kind") or "")
        source_id = act.get("source_base_card_id")
        tokens.append(EntityToken(
            card_id_index=vocabulary.index_for_base_card_id(source_id) if source_id else 0,
            card_stat=_PAD_CARD_STAT,
            crest_stat=_PAD_CREST_STAT,
            type_id=ENTITY_TYPE_HISTORY_ACTION,
            zone_id=ZONE_HISTORY,
            owner_id=OWNER_SELF if act.get("is_own") else OWNER_OPPONENT,
            visibility_id=VIS_VISIBLE,
            position_id=step_index,  # 1 = 最近一步
            count=float(ACTION_KIND_IDS.get(kind, 0)),
            count_ratio=0.0,
        ))

    return tuple(tokens)


# ──────────────────────────────────────────────────────────────────────
# Example -> row
# ──────────────────────────────────────────────────────────────────────

def _extract_action_source(action: Mapping[str, Any], *, vocabulary: CardVocabulary) -> tuple[int, tuple[float, ...]]:
    source = action.get("source")
    if source and isinstance(source, dict) and source.get("known"):
        return vocabulary.index_for_base_card_id(source.get("base_card_id")), card_stat_vector(source)
    return 0, _PAD_CARD_STAT


def _extract_action_targets(
    action: Mapping[str, Any], *, vocabulary: CardVocabulary
) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...], tuple[bool, ...], tuple[int, ...]]:
    """返回 ``(ids, stats, mask, slot_index)``，四者逐位对齐。

    ``action["targets"]`` 是**平铺**的（``action_features`` 不保留 group 边界）。
    边界由 ``target_group_sizes`` 还原；旧数据没有该字段时全部记成 -1（无槽信息），
    下游据此退化为「不区分槽」，而不是按位置猜语义。
    """
    targets_raw = action.get("targets", [])
    flat: list[Mapping[str, Any]] = []
    for tg in targets_raw:
        if isinstance(tg, list):
            flat.extend(t for t in tg if isinstance(t, dict))
        elif isinstance(tg, dict):
            flat.append(tg)

    sizes_raw = action.get("target_group_sizes") or []
    sizes = [int(value) for value in sizes_raw]
    if sizes and sum(sizes) == len(flat):
        slot_of_target: list[int] = []
        for slot, size in enumerate(sizes):
            slot_of_target.extend([slot] * size)
    else:
        slot_of_target = [-1] * len(flat)

    ids: list[int] = []
    stats: list[tuple[float, ...]] = []
    mask: list[bool] = []
    slot_index: list[int] = []
    for i in range(MAX_ACTION_TARGETS):
        if i < len(flat):
            t = flat[i]
            ids.append(vocabulary.index_for_base_card_id(t.get("base_card_id")))
            stats.append(card_stat_vector(t))
            mask.append(True)
            slot_index.append(slot_of_target[i] if i < len(slot_of_target) else -1)
        else:
            ids.append(0)
            stats.append(_PAD_CARD_STAT)
            mask.append(False)
            slot_index.append(-1)
    return tuple(ids), tuple(stats), tuple(mask), tuple(slot_index)


def _extract_action_slots(
    action: Mapping[str, Any], *, vocabulary: SlotSpecVocabulary
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """返回 ``(slot_spec_ids, slot_mask)``，长度固定为 ``MAX_ACTION_SLOTS``。

    旧数据的 ``LegalAction`` 不带 ``target_group_specs``（replay 来源），这里得到
    全 pad + 全 False 的槽，下游据此走「无 slot 信息」路径。
    声明了 spec 但超过 ``MAX_ACTION_SLOTS`` 的槽直接报错——静默截断会让
    target 落在不存在的槽里。
    """
    specs = action.get("target_group_specs") or []
    sizes = action.get("target_group_sizes") or []
    ids: list[int] = []
    mask: list[bool] = []
    if specs and len(specs) == len(sizes):
        if len(specs) > MAX_ACTION_SLOTS:
            raise ValueError(f"动作目标槽数 {len(specs)} 超过 MAX_ACTION_SLOTS={MAX_ACTION_SLOTS}: spec={specs}")
        for spec in specs:
            ids.append(vocabulary.index_for_spec(str(spec)))
            mask.append(True)
    while len(ids) < MAX_ACTION_SLOTS:
        ids.append(0)
        mask.append(False)
    return tuple(ids), tuple(mask)


def _extract_action_head_index(action: Mapping[str, Any]) -> int:
    """把 ``action_kind`` 字符串映射到 pointer 动作头下标（``ActionHead`` 枚举序号）。"""
    from decision_engine.actions.model import ActionKind
    from decision_engine.training.action_pointer import ActionHead, HEAD_BY_KIND

    raw = str(action.get("action_kind") or "")
    try:
        kind = ActionKind(raw)
    except ValueError:
        return -1
    head = HEAD_BY_KIND.get(kind)
    if head is None:
        return -1
    return list(ActionHead).index(head)


def _extract_action_positions(
    action: Mapping[str, Any],
) -> tuple[int, tuple[int, ...]]:
    """返回 ``(source_position, target_positions)``，供生成式 pointer 做监督。"""
    source_position = int((action.get("source") or {}).get("entity_position", ENTITY_POSITION_PAD))
    targets_raw = action.get("targets", [])
    flat: list[Mapping[str, Any]] = []
    for tg in targets_raw:
        if isinstance(tg, list):
            flat.extend(t for t in tg if isinstance(t, dict))
        elif isinstance(tg, dict):
            flat.append(tg)
    positions: list[int] = []
    for i in range(MAX_ACTION_TARGETS):
        if i < len(flat):
            positions.append(int(flat[i].get("entity_position", ENTITY_POSITION_PAD)))
        else:
            positions.append(ENTITY_POSITION_PAD)
    return source_position, tuple(positions)


def entity_tensorize_example(
    example: Mapping[str, Any],
    *,
    vocabulary: CardVocabulary | None = None,
    slot_vocabulary: SlotSpecVocabulary | None = None,
) -> EntityTensorizationRow:
    """把带 state_features / legal_action_features 的 example 转为 entity row。"""
    vocabulary = vocabulary or default_card_vocabulary()
    slot_vocabulary = slot_vocabulary or default_slot_spec_vocabulary()
    from decision_engine.training.tensorization import (
        state_vector_from_state_features,
        action_vectors_from_action_features,
    )

    state_features_payload = example.get("state_features", {})
    action_features = example.get("legal_action_features", [])

    state_vector = state_vector_from_state_features(state_features_payload)
    entities = build_entity_tokens(
        state_features_payload,
        vocabulary=vocabulary,
        history_actions=example.get("history_actions", ()),
    )
    action_scalar_vectors = action_vectors_from_action_features(action_features)

    src_ids: list[int] = []
    src_stats: list[tuple[float, ...]] = []
    tgt_ids: list[tuple[int, ...]] = []
    tgt_stats: list[tuple[tuple[float, ...], ...]] = []
    tgt_masks: list[tuple[bool, ...]] = []
    tgt_slot_index: list[tuple[int, ...]] = []
    slot_spec_ids: list[tuple[int, ...]] = []
    slot_masks: list[tuple[bool, ...]] = []
    src_positions: list[int] = []
    tgt_positions: list[tuple[int, ...]] = []
    head_indices: list[int] = []
    for act in action_features:
        sid, sstat = _extract_action_source(act, vocabulary=vocabulary)
        src_ids.append(sid)
        src_stats.append(sstat)
        tids, tstats, tmask, tslot = _extract_action_targets(act, vocabulary=vocabulary)
        tgt_ids.append(tids)
        tgt_stats.append(tstats)
        tgt_masks.append(tmask)
        tgt_slot_index.append(tslot)
        sids, smask = _extract_action_slots(act, vocabulary=slot_vocabulary)
        slot_spec_ids.append(sids)
        slot_masks.append(smask)
        src_pos, tgt_pos = _extract_action_positions(act)
        src_positions.append(src_pos)
        tgt_positions.append(tgt_pos)
        head_indices.append(_extract_action_head_index(act))
    entity_positions = tuple(int(getattr(token, "entity_position", ENTITY_POSITION_PAD)) for token in entities)

    policy_target = tuple(float(v) for v in example.get("policy_target", []))
    value_target_raw = example.get("value_target")
    value_target = float(value_target_raw) if value_target_raw is not None else None
    # 阶段一致性校验：若该样本声明了 phase，被选中的动作（policy_target>0）的 source
    # 必须在阶段允许的位置内。否则 pointer 的 phase mask 会把它压成 -inf，
    # 联合 CE 直接发散（实测首 epoch 就到 val_policy_loss=218）。
    phase_name = example.get("phase")
    if phase_name:
        allowed = set(PHASE_SOURCE_POSITIONS.get(phase_name, ()))
        if not allowed:
            raise ValueError(f"未知阶段: {phase_name!r}")
        for index, (position, weight) in enumerate(zip(src_positions, policy_target)):
            if weight > 0 and position not in allowed:
                raise ValueError(
                    f"阶段 {phase_name} 不允许 source 位置 {position}（动作 #{index}）；"
                    f"允许 {sorted(allowed)[:8]}... —— phase mask 会把它压成 -inf"
                )

    return EntityTensorizationRow(
        schema=ENTITY_TENSORIZATION_SCHEMA,
        version=ENTITY_TENSORIZATION_VERSION,
        entity_layout_version=ENTITY_LAYOUT_VERSION,
        state_vector=state_vector,
        entities=entities,
        action_scalar_vectors=action_scalar_vectors,
        action_source_card_ids=tuple(src_ids),
        action_source_card_stats=tuple(src_stats),
        action_target_card_ids=tuple(tgt_ids),
        action_target_card_stats=tuple(tgt_stats),
        action_target_mask=tuple(tgt_masks),
        action_target_slot_index=tuple(tgt_slot_index),
        action_slot_spec_ids=tuple(slot_spec_ids),
        action_slot_mask=tuple(slot_masks),
        entity_positions=entity_positions,
        action_source_positions=tuple(src_positions),
        action_target_positions=tuple(tgt_positions),
        action_head_indices=tuple(head_indices),
        phase=example.get("phase"),
        policy_target=policy_target,
        value_target=value_target,
        metadata=dict(example.get("metadata", {})),
    )


# ──────────────────────────────────────────────────────────────────────
# Collate（动态 padding）与 schema 校验
# ──────────────────────────────────────────────────────────────────────

class EntitySchemaMismatch(ValueError):
    """entity tensor schema / layout 版本不匹配时的显式拒绝。"""


def assert_entity_schema(row: Mapping[str, Any]) -> None:
    """校验 row 的 schema / version / entity_layout_version 与当前契约一致。

    新旧 schema 混用时显式拒绝，禁止 strict=False 静默吞掉结构差异。
    """
    schema = row.get("schema")
    if schema != ENTITY_TENSORIZATION_SCHEMA:
        raise EntitySchemaMismatch(f"unexpected entity schema: {schema!r}")
    version = row.get("version")
    if version != ENTITY_TENSORIZATION_VERSION:
        raise EntitySchemaMismatch(
            f"entity tensorization version {version!r} != {ENTITY_TENSORIZATION_VERSION}"
        )
    layout = row.get("entity_layout_version")
    if layout != ENTITY_LAYOUT_VERSION:
        raise EntitySchemaMismatch(
            f"entity layout version {layout!r} != {ENTITY_LAYOUT_VERSION}"
        )


def _row_entities(row: Any) -> Sequence[Any]:
    if isinstance(row, EntityTensorizationRow):
        return row.entities
    return row["entities"]


def _entity_field(entity: Any, name: str) -> Any:
    if isinstance(entity, EntityToken):
        return getattr(entity, name)
    return entity[name]


def _entity_field_optional(entity: Any, name: str, default: Any) -> Any:
    """读可选字段：旧 row 的 dict entity 没有 ``entity_position`` 时不报错。"""
    if isinstance(entity, EntityToken):
        return getattr(entity, name, default)
    return entity.get(name, default)


def collate_entity_rows(rows: Sequence[Any], *, device: str = "cpu") -> dict[str, Any]:
    """把若干 entity row 组成 batch，按当批最大实体数 / 动作数动态 padding。

    输入可以是 `EntityTensorizationRow` 或其 `to_jsonable()` dict。
    dict 输入会先做 schema 校验，拒绝新旧 schema 混用。

    返回张量（B = batch size, E = 当批最大实体数, A = 当批最大动作数）：
        state_scalar        [B, state_width]
        entity_card_ids     [B, E]  long
        entity_card_stats   [B, E, CARD_STAT_DIM]
        entity_crest_stats  [B, E, CREST_STAT_DIM]
        entity_type_ids     [B, E]  long
        entity_zone_ids     [B, E]  long
        entity_owner_ids    [B, E]  long
        entity_visibility_ids [B, E] long
        entity_position_ids [B, E]  long
        entity_count        [B, E]  float
        entity_count_ratio  [B, E]  float
        entity_mask         [B, E]  bool  (True=有效)
        action_scalars / action_mask / action_source_* / action_target_* / policy / value / value_mask
    """
    import torch

    if not rows:
        raise ValueError("collate_entity_rows requires at least one row")

    # 校验 dict 输入的 schema
    for row in rows:
        if not isinstance(row, EntityTensorizationRow):
            assert_entity_schema(row)

    B = len(rows)

    # 先在 CPU 上填满，最后一次搬到 device。
    # 逐标量 ``torch.tensor(..., device="cuda")`` 会为每个样本的每个动作用户端发起
    # 上千次小 kernel launch + 隐式同步，实测让 GPU 利用率掉到 20% 以下、3.1M 模型
    # 在 3643 样本上要 1.3 分钟/epoch——瓶颈在 collate 而不在前向/反向。
    fill_device = "cpu"

    def _state_vector(row: Any) -> Sequence[float]:
        return row.state_vector if isinstance(row, EntityTensorizationRow) else row["state_vector"]

    def _actions(row: Any, name: str) -> Sequence[Any]:
        return getattr(row, name) if isinstance(row, EntityTensorizationRow) else row[name]

    def _actions_optional(row: Any, name: str) -> Sequence[Any]:
        """读可选的动作字段：缺失时返回空序列（旧 row 不含 slot 字段）。"""
        if isinstance(row, EntityTensorizationRow):
            return getattr(row, name, ()) or ()
        return row.get(name) or ()

    state_width = len(_state_vector(rows[0]))
    max_entities = max(len(_row_entities(row)) for row in rows)
    max_actions = max(len(_actions(row, "action_scalar_vectors")) for row in rows)
    first_actions = _actions(rows[0], "action_scalar_vectors")
    action_width = len(first_actions[0]) if first_actions else ACTION_VECTOR_DIM

    state_scalar = torch.zeros(B, state_width, dtype=torch.float32, device=fill_device)
    entity_card_ids = torch.zeros(B, max_entities, dtype=torch.long, device=fill_device)
    entity_card_stats = torch.zeros(B, max_entities, CARD_STAT_DIM, dtype=torch.float32, device=fill_device)
    entity_crest_stats = torch.zeros(B, max_entities, CREST_STAT_DIM, dtype=torch.float32, device=fill_device)
    entity_type_ids = torch.zeros(B, max_entities, dtype=torch.long, device=fill_device)
    entity_zone_ids = torch.zeros(B, max_entities, dtype=torch.long, device=fill_device)
    entity_owner_ids = torch.zeros(B, max_entities, dtype=torch.long, device=fill_device)
    entity_visibility_ids = torch.zeros(B, max_entities, dtype=torch.long, device=fill_device)
    entity_position_ids = torch.zeros(B, max_entities, dtype=torch.long, device=fill_device)
    entity_count = torch.zeros(B, max_entities, dtype=torch.float32, device=fill_device)
    entity_count_ratio = torch.zeros(B, max_entities, dtype=torch.float32, device=fill_device)
    entity_mask = torch.zeros(B, max_entities, dtype=torch.bool, device=fill_device)

    max_targets = MAX_ACTION_TARGETS
    max_slots = MAX_ACTION_SLOTS
    action_scalars = torch.zeros(B, max_actions, action_width, dtype=torch.float32, device=fill_device)
    action_mask = torch.zeros(B, max_actions, dtype=torch.bool, device=fill_device)
    action_source_ids = torch.zeros(B, max_actions, dtype=torch.long, device=fill_device)
    action_source_stats = torch.zeros(B, max_actions, CARD_STAT_DIM, dtype=torch.float32, device=fill_device)
    action_target_ids = torch.zeros(B, max_actions, max_targets, dtype=torch.long, device=fill_device)
    action_target_stats = torch.zeros(B, max_actions, max_targets, CARD_STAT_DIM, dtype=torch.float32, device=fill_device)
    action_target_mask = torch.zeros(B, max_actions, max_targets, dtype=torch.bool, device=fill_device)
    # 目标槽：-1 表示「该 target 不属于任何已知槽」（旧数据无边界信息）
    action_target_slot_index = torch.full((B, max_actions, max_targets), -1, dtype=torch.long, device=fill_device)
    action_slot_spec_ids = torch.zeros(B, max_actions, max_slots, dtype=torch.long, device=fill_device)
    action_slot_mask = torch.zeros(B, max_actions, max_slots, dtype=torch.bool, device=fill_device)
    # 生成式 pointer 的指向空间：固定位置坐标
    entity_anchor_ids = torch.full((B, max_entities), ENTITY_POSITION_PAD, dtype=torch.long, device=fill_device)
    action_source_positions = torch.full((B, max_actions), ENTITY_POSITION_PAD, dtype=torch.long, device=fill_device)
    action_target_positions = torch.full((B, max_actions, max_targets), ENTITY_POSITION_PAD, dtype=torch.long, device=fill_device)
    action_head_indices = torch.full((B, max_actions), -1, dtype=torch.long, device=fill_device)
    phase_ids = torch.full((B,), PHASE_PAD, dtype=torch.long, device=fill_device)
    position_mask = torch.zeros(B, MAX_ENTITY_POSITIONS, dtype=torch.bool, device=fill_device)

    policy = torch.zeros(B, max_actions, dtype=torch.float32, device=fill_device)
    value = torch.zeros(B, dtype=torch.float32, device=fill_device)
    value_mask = torch.zeros(B, dtype=torch.bool, device=fill_device)

    for b, row in enumerate(rows):
        sv = _state_vector(row)
        state_scalar[b, :len(sv)] = torch.tensor(sv, dtype=torch.float32, device=fill_device)

        for i, ent in enumerate(_row_entities(row)):
            entity_card_ids[b, i] = int(_entity_field(ent, "card_id_index"))
            entity_card_stats[b, i] = torch.tensor(_entity_field(ent, "card_stat"), dtype=torch.float32, device=fill_device)
            entity_crest_stats[b, i] = torch.tensor(_entity_field(ent, "crest_stat"), dtype=torch.float32, device=fill_device)
            entity_type_ids[b, i] = int(_entity_field(ent, "type_id"))
            entity_zone_ids[b, i] = int(_entity_field(ent, "zone_id"))
            entity_owner_ids[b, i] = int(_entity_field(ent, "owner_id"))
            entity_visibility_ids[b, i] = int(_entity_field(ent, "visibility_id"))
            entity_position_ids[b, i] = int(_entity_field(ent, "position_id"))
            entity_count[b, i] = float(_entity_field(ent, "count"))
            entity_count_ratio[b, i] = float(_entity_field(ent, "count_ratio"))
            entity_mask[b, i] = True
            position = int(_entity_field_optional(ent, "entity_position", ENTITY_POSITION_PAD))
            entity_anchor_ids[b, i] = position
            if 0 <= position < MAX_ENTITY_POSITIONS:
                position_mask[b, position] = True

        # 双方主战者始终可被指向（ATTACK_LEADER / selected_enemy_follower_or_leader）
        position_mask[b, ENTITY_POSITION_SELF_LEADER] = True
        position_mask[b, ENTITY_POSITION_OPP_LEADER] = True

        scalars = _actions(row, "action_scalar_vectors")
        src_ids = _actions(row, "action_source_card_ids")
        src_stats = _actions(row, "action_source_card_stats")
        tgt_ids = _actions(row, "action_target_card_ids")
        tgt_stats = _actions(row, "action_target_card_stats")
        tgt_mask = _actions(row, "action_target_mask")
        tgt_slot = _actions_optional(row, "action_target_slot_index")
        slot_ids = _actions_optional(row, "action_slot_spec_ids")
        slot_mask = _actions_optional(row, "action_slot_mask")
        src_positions = _actions_optional(row, "action_source_positions")
        tgt_positions = _actions_optional(row, "action_target_positions")
        head_indices = _actions_optional(row, "action_head_indices")
        policy_target = _actions(row, "policy_target")
        for i in range(len(scalars)):
            action_scalars[b, i] = torch.tensor(scalars[i], dtype=torch.float32, device=fill_device)
            action_mask[b, i] = True
            action_source_ids[b, i] = int(src_ids[i])
            action_source_stats[b, i] = torch.tensor(src_stats[i], dtype=torch.float32, device=fill_device)
            for j in range(min(len(tgt_ids[i]), max_targets)):
                action_target_ids[b, i, j] = int(tgt_ids[i][j])
                action_target_stats[b, i, j] = torch.tensor(tgt_stats[i][j], dtype=torch.float32, device=fill_device)
                action_target_mask[b, i, j] = bool(tgt_mask[i][j])
                if i < len(tgt_slot) and j < len(tgt_slot[i]):
                    action_target_slot_index[b, i, j] = int(tgt_slot[i][j])
            if i < len(slot_ids):
                for k in range(min(len(slot_ids[i]), max_slots)):
                    action_slot_spec_ids[b, i, k] = int(slot_ids[i][k])
                    action_slot_mask[b, i, k] = bool(slot_mask[i][k]) if i < len(slot_mask) and k < len(slot_mask[i]) else True
            if i < len(src_positions):
                action_source_positions[b, i] = int(src_positions[i])
            if i < len(tgt_positions):
                for j in range(min(len(tgt_positions[i]), max_targets)):
                    action_target_positions[b, i, j] = int(tgt_positions[i][j])
            if i < len(head_indices):
                action_head_indices[b, i] = int(head_indices[i])
            if i < len(policy_target):
                policy[b, i] = float(policy_target[i])

        vt = row.value_target if isinstance(row, EntityTensorizationRow) else row.get("value_target")
        if vt is not None:
            value[b] = float(vt)
            value_mask[b] = True
        phase_name = getattr(row, "phase", None) if isinstance(row, EntityTensorizationRow) else row.get("phase")
        phase_ids[b] = phase_index(phase_name)

    batch = {
        "state_scalar": state_scalar,
        "entity_card_ids": entity_card_ids,
        "entity_card_stats": entity_card_stats,
        "entity_crest_stats": entity_crest_stats,
        "entity_type_ids": entity_type_ids,
        "entity_zone_ids": entity_zone_ids,
        "entity_owner_ids": entity_owner_ids,
        "entity_visibility_ids": entity_visibility_ids,
        "entity_position_ids": entity_position_ids,
        "entity_count": entity_count,
        "entity_count_ratio": entity_count_ratio,
        "entity_mask": entity_mask,
        "action_scalars": action_scalars,
        "action_mask": action_mask,
        "action_source_ids": action_source_ids,
        "action_source_stats": action_source_stats,
        "action_target_ids": action_target_ids,
        "action_target_stats": action_target_stats,
        "action_target_mask": action_target_mask,
        "action_target_slot_index": action_target_slot_index,
        "action_slot_spec_ids": action_slot_spec_ids,
        "action_slot_mask": action_slot_mask,
        "entity_anchor_ids": entity_anchor_ids,
        "action_source_positions": action_source_positions,
        "action_target_positions": action_target_positions,
        "action_head_indices": action_head_indices,
        "phase_ids": phase_ids,
        "position_mask": position_mask,
        "policy": policy,
        "value": value,
        "value_mask": value_mask,
    }
    if device != fill_device:
        batch = {key: tensor.to(device, non_blocking=True) for key, tensor in batch.items()}
    return batch
