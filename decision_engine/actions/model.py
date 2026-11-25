# -*- coding: utf-8 -*-
"""协议风格动作模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionKind(Enum):
    """可规划动作类型。"""

    PLAY = "play"
    ATTACK_LEADER = "attack_leader"
    ATTACK_FOLLOWER = "attack_follower"
    EVOLVE = "evolve"
    SUPER_EVOLVE = "super_evolve"
    ACTIVATE = "activate"
    FUSION = "fusion"
    SPECIAL_ACTION = "special_action"
    EXTRA_PP = "extra_pp"
    TURN_END = "turn_end"
    TURN_START = "turn_start"
    #: 换牌阶段（每局第一个决策阶段）：把 ``source_uid`` 指向的手牌换掉。
    #: 换牌动作与局内动作同构，走同一套 action_features / policy head / rollout 记录，
    #: 因此不需要额外的换牌网络。见 ``decision_engine/actions/mulligan_enumerator.py``。
    MULLIGAN_CHANGE = "mulligan_change"
    #: 结束换牌：保留剩余手牌。总是可选，因此换牌阶段不会死锁。
    MULLIGAN_END = "mulligan_end"


class PlayKind(Enum):
    """出牌方式。"""

    NORMAL = "normal"
    ENHANCE = "enhance"
    ACCELERATE = "accelerate"
    CRYSTAL = "crystal"


@dataclass(frozen=True)
class LegalAction:
    """动作枚举器产出的合法动作候选。

    target_groups 对齐协议中的 targetUniqueIdBySkills，而不是压缩成单 target_uid。
    """

    kind: ActionKind
    source_uid: int
    play_kind: PlayKind | None = None
    cost: int | None = None
    target_groups: tuple[tuple[int, ...], ...] = ()
    random_target_groups: tuple[tuple[int, ...], ...] = ()
    selected_indexes: tuple[int, ...] = ()
    material_uids: tuple[int, ...] = ()
    variant: str | None = None
    legality_source: str = "heuristic"
    confidence: str = "approx"
    description: str = ""
    #: 与 ``target_groups`` **等长**的语义标识，指明每个 group 来自哪个效果选择。
    #:
    #: 回合内一个动作可能同时需要多次目标选择（一张卡同一触发事件下有多个
    #: ``selected_*`` spec），而 ``target_groups`` 只有 uid、没有语义。若下游按
    #: **位置**理解 slot，条件不满足的 spec 被跳过时后面的 group 会前移，同一位置
    #: 在不同状态下指向不同效果。pointer 动作表示（``training/action_pointer.py``）
    #: 依赖这个字段定位 slot，不依赖位置。
    #:
    #: 取值：effect-spec 路径填 ``spec.spec_id``；协议/本地 ``select_skill``、
    #: ``mode_skill``、``choice_skill`` 等非 spec 路径填对应的来源标识
    #: （``protocol_select`` / ``local_select`` / ``mode_choice`` / ``choice``）。
    target_group_specs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedAction:
    """准备编码或执行的动作。"""

    kind: ActionKind
    source_uid: int
    play_kind: PlayKind | None = None
    target_unique_id_by_skills: tuple[tuple[int, ...], ...] = ()
    selected_indexes: tuple[int, ...] = ()
    material_uids: tuple[int, ...] = ()
    description: str = ""
