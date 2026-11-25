# -*- coding: utf-8 -*-
"""实体位置坐标系：生成式 pointer 的指向空间。

pointer 需要指向「哪个实体」，但训练数据里**没有 uid**——局内 uid 绝对值既不可泛化，
也被项目规则禁止作为模型输入。所以用**固定位置坐标**表示「哪一格」：

    0               视角方主战者（self leader）
    1               对方主战者（opp leader）
    10 + i          视角方手牌第 i 张      （i = 0..8，共 9 张）
    20 + p          视角方场地第 p 格      （p = 1..5）
    30 + p          对方场地第 p 格        （p = 1..5）
    40 + i          视角方纹章第 i 张      （i = 0..4）
    50 + i          对方纹章第 i 张        （i = 0..4）

坐标是**状态的函数**，与局内 uid 无关，可以跨对局泛化；同一张卡的两个副本落在
不同坐标上（手牌按顺序、场地按格子），所以 pointer 能区分它们。

「视角方」由 ``perspective`` 决定，语义与 ``action_features._zone_for_uid`` 一致：
``perspective="me"`` 时视角方是 ``snapshot.me``。
"""

from __future__ import annotations

from typing import Any

ENTITY_POSITION_PAD = -1
ENTITY_POSITION_SELF_LEADER = 0
ENTITY_POSITION_OPP_LEADER = 1

ENTITY_POSITION_SELF_HAND_BASE = 10
ENTITY_POSITION_SELF_FIELD_BASE = 20
ENTITY_POSITION_OPP_FIELD_BASE = 30
ENTITY_POSITION_SELF_CREST_BASE = 40
ENTITY_POSITION_OPP_CREST_BASE = 50

MAX_HAND_POSITIONS = 9
MAX_FIELD_POSITIONS = 5
MAX_CREST_POSITIONS = 5

#: pointer 输出宽度。取 64 是为后续区域留裕量，坐标本身只用到 0..54。
MAX_ENTITY_POSITIONS = 64

#: 回合内阶段，与 ``turn_planner`` 的 ``phased`` 搜索一致：play → evolve → exchange。
#: 阶段是**策略所处的状态**，不是 snapshot 的属性——同一个盘面在搜索的不同阶段
#: 会重复出现，所以必须显式记录，不能从状态推断。
PHASE_PLAY = "play"
PHASE_EVOLVE = "evolve"
PHASE_EXCHANGE = "exchange"
PHASE_NAMES = (PHASE_PLAY, PHASE_EVOLVE, PHASE_EXCHANGE)
NUM_PHASES = len(PHASE_NAMES)
PHASE_PAD = -1

#: 各阶段允许的 source 位置。
#:
#: **每个阶段都必须包含 ``ENTITY_POSITION_SELF_LEADER``**：无 source 的动作
#: （``TURN_END`` / ``EXTRA_PP``）锚到该坐标。漏掉它会让这些动作的 source logit
#: 变成 ``-inf``，联合 CE 直接发散（实测 ``pointer_phase`` 首 epoch 就到
#: ``val_policy_loss=218``、``pointer_source=inf``）。
#:
#: play 阶段同时可能用 PLAY（手牌）和 ACTIVATE（场地），所以两者都保留；
#: evolve / exchange 只动场地。
PHASE_SOURCE_POSITIONS: dict[str, tuple[int, ...]] = {
    PHASE_PLAY: (
        ENTITY_POSITION_SELF_LEADER,
        *range(ENTITY_POSITION_SELF_HAND_BASE, ENTITY_POSITION_SELF_HAND_BASE + MAX_HAND_POSITIONS),
        *range(ENTITY_POSITION_SELF_FIELD_BASE + 1, ENTITY_POSITION_SELF_FIELD_BASE + MAX_FIELD_POSITIONS + 1),
        *range(ENTITY_POSITION_SELF_CREST_BASE, ENTITY_POSITION_SELF_CREST_BASE + MAX_CREST_POSITIONS),
    ),
    PHASE_EVOLVE: (
        ENTITY_POSITION_SELF_LEADER,
        *range(ENTITY_POSITION_SELF_FIELD_BASE + 1, ENTITY_POSITION_SELF_FIELD_BASE + MAX_FIELD_POSITIONS + 1),
    ),
    PHASE_EXCHANGE: (
        ENTITY_POSITION_SELF_LEADER,
        *range(ENTITY_POSITION_SELF_FIELD_BASE + 1, ENTITY_POSITION_SELF_FIELD_BASE + MAX_FIELD_POSITIONS + 1),
    ),
}

#: 各阶段允许的 target 位置。play 阶段的目标最广（打牌可以选自己/对方的手牌、场地、
#: 纹章）；evolve 次之；exchange 只能选对方场地和主战者。
PHASE_TARGET_POSITIONS: dict[str, tuple[int, ...]] = {
    PHASE_PLAY: tuple(range(MAX_ENTITY_POSITIONS)),
    PHASE_EVOLVE: (
        ENTITY_POSITION_SELF_LEADER,
        ENTITY_POSITION_OPP_LEADER,
        *range(ENTITY_POSITION_SELF_FIELD_BASE + 1, ENTITY_POSITION_SELF_FIELD_BASE + MAX_FIELD_POSITIONS + 1),
        *range(ENTITY_POSITION_OPP_FIELD_BASE + 1, ENTITY_POSITION_OPP_FIELD_BASE + MAX_FIELD_POSITIONS + 1),
        *range(ENTITY_POSITION_SELF_CREST_BASE, ENTITY_POSITION_SELF_CREST_BASE + MAX_CREST_POSITIONS),
        *range(ENTITY_POSITION_OPP_CREST_BASE, ENTITY_POSITION_OPP_CREST_BASE + MAX_CREST_POSITIONS),
    ),
    PHASE_EXCHANGE: (
        ENTITY_POSITION_OPP_LEADER,
        *range(ENTITY_POSITION_OPP_FIELD_BASE + 1, ENTITY_POSITION_OPP_FIELD_BASE + MAX_FIELD_POSITIONS + 1),
    ),
}


def phase_index(name: str | None) -> int:
    """阶段名 → 下标；``None`` / 未知 → ``PHASE_PAD``。"""
    return PHASE_NAMES.index(name) if name in PHASE_NAMES else PHASE_PAD


def phase_position_mask(phase: str, *, max_positions: int = MAX_ENTITY_POSITIONS) -> list[bool]:
    """某阶段的 target 位置 mask（列表形式，供 tensorization 填张量）。"""
    allowed = set(PHASE_TARGET_POSITIONS.get(phase, ()))
    return [index in allowed for index in range(max_positions)]


def position_for_hand(index: int) -> int:
    if not 0 <= index < MAX_HAND_POSITIONS:
        raise ValueError(f"手牌位置越界: {index}")
    return ENTITY_POSITION_SELF_HAND_BASE + index


def position_for_field(*, self_side: bool, position: int) -> int:
    """``position`` 是 1-based 场地格号（与 ``EntityToken.position_id`` 一致）。"""
    if not 1 <= position <= MAX_FIELD_POSITIONS:
        raise ValueError(f"场地位置越界: {position}")
    base = ENTITY_POSITION_SELF_FIELD_BASE if self_side else ENTITY_POSITION_OPP_FIELD_BASE
    return base + position


def position_for_crest(*, self_side: bool, index: int) -> int:
    if not 0 <= index < MAX_CREST_POSITIONS:
        raise ValueError(f"纹章位置越界: {index}")
    base = ENTITY_POSITION_SELF_CREST_BASE if self_side else ENTITY_POSITION_OPP_CREST_BASE
    return base + index


def _views(snapshot: Any, perspective: str) -> tuple[Any, Any]:
    if perspective == "me":
        return snapshot.me, snapshot.opponent
    return snapshot.opponent, snapshot.me


def position_for_uid(snapshot: Any, uid: int, *, perspective: str = "me") -> int:
    """把 uid 反查成位置坐标；无法定位时返回 ``ENTITY_POSITION_PAD``。

    主战者有两种表示：``ATTACK_LEADER`` 用真实 ``unique_id``，
    ``selected_enemy_follower_or_leader`` 用哨兵 ``0``。两者都映射到 opp leader 坐标。
    """
    player_view, opponent_view = _views(snapshot, perspective)
    if uid == 0 or uid == getattr(opponent_view, "unique_id", None):
        return ENTITY_POSITION_OPP_LEADER
    if uid == getattr(player_view, "unique_id", None):
        return ENTITY_POSITION_SELF_LEADER
    if uid in player_view.hand:
        index = player_view.hand.index(uid)
        return position_for_hand(index) if 0 <= index < MAX_HAND_POSITIONS else ENTITY_POSITION_PAD
    if uid in player_view.field:
        index = player_view.field.index(uid)
        return position_for_field(self_side=True, position=index + 1) if index < MAX_FIELD_POSITIONS else ENTITY_POSITION_PAD
    if uid in opponent_view.field:
        index = opponent_view.field.index(uid)
        return position_for_field(self_side=False, position=index + 1) if index < MAX_FIELD_POSITIONS else ENTITY_POSITION_PAD
    for index, crest in enumerate(getattr(player_view, "crests", ()) or ()):
        if getattr(crest, "unique_id", None) == uid:
            return position_for_crest(self_side=True, index=index) if index < MAX_CREST_POSITIONS else ENTITY_POSITION_PAD
    for index, crest in enumerate(getattr(opponent_view, "crests", ()) or ()):
        if getattr(crest, "unique_id", None) == uid:
            return position_for_crest(self_side=False, index=index) if index < MAX_CREST_POSITIONS else ENTITY_POSITION_PAD
    return ENTITY_POSITION_PAD


def position_mask_from_positions(positions: Any, *, max_positions: int = MAX_ENTITY_POSITIONS) -> Any:
    """由一组位置坐标构造 ``[max_positions]`` 的可用 mask（torch 张量）。"""
    import torch

    mask = torch.zeros(max_positions, dtype=torch.bool)
    for position in positions:
        if position != ENTITY_POSITION_PAD and 0 <= position < max_positions:
            mask[position] = True
    return mask
