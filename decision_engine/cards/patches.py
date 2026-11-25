# -*- coding: utf-8 -*-
"""卡牌补丁数据模型。

本模块只定义结构化卡牌知识，不执行效果，也不复刻旧 spell_effect DSL。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from decision_engine.cards.effect_facts import CardEffectFact


class EffectTiming(Enum):
    """效果触发时机。"""

    FANFARE = "fanfare"
    LAST_WORDS = "last_words"
    EVOLVE = "evolve"
    SUPER_EVOLVE = "super_evolve"
    ACTIVATION = "activation"
    ON_ATTACK = "on_attack"
    ON_CLASH = "on_clash"
    PASSIVE = "passive"
    FUSION = "fusion"
    SPELL = "spell"


class EffectKind(Enum):
    """结构化效果类别。

    这里只描述效果类别，具体预测由 simulation/effect_predictor 或 Oracle 负责。
    """

    DAMAGE = "damage"
    HEAL = "heal"
    BUFF = "buff"
    DESTROY = "destroy"
    SUMMON = "summon"
    DRAW = "draw"
    ADD_HAND = "add_hand"
    BOUNCE = "bounce"
    RESTORE_PP = "restore_pp"
    GRANT_KEYWORD = "grant_keyword"
    UNKNOWN = "unknown"


class TargetHint(Enum):
    """目标提示，用于动作枚举和预测器，不作为协议合法性裁判。"""

    NONE = "none"
    ENEMY_LEADER = "enemy_leader"
    ENEMY_FOLLOWER = "enemy_follower"
    ALL_ENEMIES = "all_enemies"
    ALLY_FOLLOWER = "ally_follower"
    ALL_ALLIES = "all_allies"
    SELF = "self"
    ANY_FOLLOWER = "any_follower"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EffectPatch:
    """单个结构化效果补丁。"""

    timing: EffectTiming
    kind: EffectKind
    target: TargetHint = TargetHint.UNKNOWN
    amount: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class CardPatch:
    """一张 base card 的人工/半自动补丁。"""

    base_card_id: int
    name: str = ""
    effects: tuple[EffectPatch, ...] = ()
    effect_facts: tuple[CardEffectFact, ...] = ()
    ignored_auto_fact_raw_texts: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    note: str = ""

    def effects_at(self, timing: EffectTiming) -> tuple[EffectPatch, ...]:
        return tuple(effect for effect in self.effects if effect.timing == timing)
