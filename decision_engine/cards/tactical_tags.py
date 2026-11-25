# -*- coding: utf-8 -*-
"""战术标签提取。

从卡牌效果文本中提取结构化战术标签，供 Evaluator 消费。
本模块只做文本模式匹配，不执行效果，不复刻旧 spell_effect DSL。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_KEYWORD_MAP: dict[str, str] = {
    "守护": "guard",
    "疾驰": "storm",
    "突进": "rush",
    "潜行": "ambush",
    "威慑": "intimidate",
    "虹吸": "drain",
    "屏障": "barrier",
    "毁灭": "bane",
    "灵气": "cant_select",
}

_RESOURCE_MAP: dict[str, str] = {
    "连击": "combo",
    "协作": "rally",
    "融合": "fusion",
    "觉醒": "kakusei",
    "爆能强化": "enhance",
    "土之秘术": "earth_rite",
    "奥义": "secret_art",
    "解放奥义": "release_secret_art",
    "亡者召还": "reanimate",
    "唤灵": "invoke",
    "瞬念召唤": "accelerate_summon",
    "魔力增幅时": "spellboost",
    "信仰": "faith",
    "信仰值": "faith",
}

_FREE_TEXT_RESOURCE_MAP: dict[str, str] = {
    "信仰": "faith",
    "信仰值": "faith",
}

_TIMING_MAP: dict[str, str] = {
    "入场曲": "fanfare",
    "谢幕曲": "last_words",
    "进化时": "on_evolve",
    "超进化时": "on_super_evolve",
    "攻击时": "on_attack",
    "交战时": "on_clash",
    "启动": "activation",
    "吟唱": "countdown",
    "模式": "mode",
}

_BRACKET_RE = re.compile(r"【([^】]+?)】")
_DAMAGE_RE = re.compile(r"造成(\d+)点伤害")
_HEAL_RE = re.compile(r"回复.*?(\d+)点生命")
_DRAW_RE = re.compile(r"抽取(\d+)")
_SUMMON_RE = re.compile(r"召唤")
_DESTROY_RE = re.compile(r"破坏")
_BOUNCE_RE = re.compile(r"返回手牌")
_BUFF_RE = re.compile(r"[+＋](\d+)/[+＋](\d+)")
_COST_REDUCE_RE = re.compile(r"费用[-\-](\d+)|费用变为")
_AOE_INDICATORS = ("所有随从", "随机", "所有的")
_LEADER_DAMAGE_INDICATORS = ("主战者", "对手的主战者")

_FREE_TEXT_TIMING_MAP: dict[str, str] = {
    "回合开始时": "turn_start",
    "回合结束时": "turn_end",
    "进入战场时": "on_enter",
    "被破坏时": "on_destroyed",
    "消失时": "on_vanish",
    "攻击时": "on_attack",
    "回复时": "on_heal",
    "进化时": "on_evolve",
    "使用": "on_play",
}


@dataclass(frozen=True)
class TacticalTags:
    """卡牌战术标签集合。"""

    keywords: frozenset[str] = frozenset()
    timings: frozenset[str] = frozenset()
    resources: frozenset[str] = frozenset()
    has_damage: bool = False
    has_aoe: bool = False
    has_leader_damage: bool = False
    has_heal: bool = False
    has_draw: bool = False
    has_summon: bool = False
    has_destroy: bool = False
    has_bounce: bool = False
    has_buff: bool = False
    has_cost_reduce: bool = False
    max_damage: int = 0


def extract_tactical_tags(skill_desc: str) -> TacticalTags:
    """从效果文本提取战术标签。"""
    if not skill_desc:
        return TacticalTags()

    keywords: set[str] = set()
    timings: set[str] = set()
    resources: set[str] = set()

    for match in _BRACKET_RE.finditer(skill_desc):
        raw = match.group(1).split("_")[0].split(" ")[0]
        if raw in _KEYWORD_MAP:
            keywords.add(_KEYWORD_MAP[raw])
        if raw in _TIMING_MAP:
            timings.add(_TIMING_MAP[raw])
        if raw in _RESOURCE_MAP:
            resources.add(_RESOURCE_MAP[raw])

    for phrase, tag in _FREE_TEXT_TIMING_MAP.items():
        if phrase in skill_desc:
            timings.add(tag)

    for phrase, tag in _FREE_TEXT_RESOURCE_MAP.items():
        if phrase in skill_desc:
            resources.add(tag)

    damage_values = [int(m.group(1)) for m in _DAMAGE_RE.finditer(skill_desc)]
    has_damage = bool(damage_values)
    max_damage = max(damage_values) if damage_values else 0

    has_aoe = any(indicator in skill_desc for indicator in _AOE_INDICATORS)
    has_leader_damage = has_damage and any(indicator in skill_desc for indicator in _LEADER_DAMAGE_INDICATORS)
    has_heal = bool(_HEAL_RE.search(skill_desc))
    has_draw = bool(_DRAW_RE.search(skill_desc))
    has_summon = bool(_SUMMON_RE.search(skill_desc))
    has_destroy = bool(_DESTROY_RE.search(skill_desc))
    has_bounce = bool(_BOUNCE_RE.search(skill_desc))
    has_buff = bool(_BUFF_RE.search(skill_desc))
    has_cost_reduce = bool(_COST_REDUCE_RE.search(skill_desc))

    return TacticalTags(
        keywords=frozenset(keywords),
        timings=frozenset(timings),
        resources=frozenset(resources),
        has_damage=has_damage,
        has_aoe=has_aoe,
        has_leader_damage=has_leader_damage,
        has_heal=has_heal,
        has_draw=has_draw,
        has_summon=has_summon,
        has_destroy=has_destroy,
        has_bounce=has_bounce,
        has_buff=has_buff,
        has_cost_reduce=has_cost_reduce,
        max_damage=max_damage,
    )
