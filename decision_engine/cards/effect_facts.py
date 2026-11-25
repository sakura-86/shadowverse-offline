# -*- coding: utf-8 -*-
"""卡牌效果事实提取。

CardEffectFact 只描述文本中的战术事实与复杂度信号，不执行效果。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decision_engine.state.snapshot import CardView


class EffectFactTiming(Enum):
    FANFARE = "fanfare"
    LAST_WORDS = "last_words"
    ON_EVOLVE = "on_evolve"
    ON_SUPER_EVOLVE = "on_super_evolve"
    ON_ATTACK = "on_attack"
    ON_CLASH = "on_clash"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    ACTIVATION = "activation"
    ON_PLAY = "on_play"
    SPELLBOOST = "spellboost"
    FUSION = "fusion"
    ENHANCE = "enhance"
    ACCELERATE = "accelerate"
    CRYSTAL = "crystal"
    CONTINUOUS = "continuous"
    UNKNOWN = "unknown"


class EffectFactKind(Enum):
    LEADER_DAMAGE = "leader_damage"
    FIELD_DAMAGE = "field_damage"
    DISTRIBUTED_DAMAGE = "distributed_damage"
    HEAL = "heal"
    DRAW = "draw"
    SUMMON = "summon"
    ADD_TO_HAND = "add_to_hand"
    DESTROY = "destroy"
    BANISH = "banish"
    BOUNCE = "bounce"
    TRANSFORM = "transform"
    STAT_BUFF = "stat_buff"
    STAT_DEBUFF = "stat_debuff"
    KEYWORD_GAIN = "keyword_gain"
    KEYWORD_REMOVE = "keyword_remove"
    AUTO_EVOLVE = "auto_evolve"
    COST_CHANGE = "cost_change"
    PP_CHANGE = "pp_change"
    SPELLBOOST_HAND = "spellboost_hand"
    CREST_OR_FAITH = "crest_or_faith"
    DECK_OPERATION = "deck_operation"
    DISCARD = "discard"
    GRAVEYARD = "graveyard"
    SPECIAL_RESULT = "special_result"
    REANIMATE = "reanimate"
    EARTH_SIGIL = "earth_sigil"
    ARTS_GAUGE = "arts_gauge"
    VARIABLE_DEFINE = "variable_define"
    UNKNOWN_COMPLEX = "unknown_complex"


class EffectTargeting(Enum):
    NONE = "none"
    SELF = "self"
    SELECTED = "selected"
    RANDOM = "random"
    ALL = "all"
    UNKNOWN = "unknown"


class EffectValueSource(Enum):
    NONE = "none"
    FIXED = "fixed"
    VARIABLE_X = "variable_x"
    COUNT_BASED = "count_based"
    HISTORY_BASED = "history_based"
    UNKNOWN = "unknown"


class EffectFactConfidence(Enum):
    TEXT_PATTERN = "text_pattern"
    UNSUPPORTED_COMPLEX = "unsupported_complex"


@dataclass(frozen=True)
class CardEffectFact:
    timing: EffectFactTiming
    kind: EffectFactKind
    amount: int | None = None
    count: int | None = None
    target_side: str = ""
    target_zone: str = ""
    target_type: str = ""
    targeting: EffectTargeting = EffectTargeting.NONE
    value_source: EffectValueSource = EffectValueSource.NONE
    conditions: frozenset[str] = frozenset()
    safe_to_apply: bool = False
    requires_oracle: bool = True
    confidence: EffectFactConfidence = EffectFactConfidence.TEXT_PATTERN
    raw_text: str = ""
    unsupported_reason: str = ""


@dataclass(frozen=True)
class EffectMetadata:
    hand_spellboost: int = 0
    body: tuple[int, int] | None = None
    keywords: frozenset[str] = frozenset()
    refill_to_max: bool = False
    last_words_spellboost: int = 0


_HAND_SPELLBOOST_RE = re.compile(r"(?:所有手牌|手牌中(?:的)?(?:所有)?卡牌|自己的所有手牌).*?(?:发动|进行)(\d+)?次?魔力增幅")
_PATCH_HAND_SPELLBOOST_RE = re.compile(r"hand_spellboost:(\d+)")
_BODY_RE = re.compile(r"body:(\d+)/(\d+)")
_KEYWORD_RE = re.compile(r"keyword:([a-z_]+)")
_LAST_WORDS_SPELLBOOST_RE = re.compile(r"last_words_spellboost:(\d+)")


def parse_effect_metadata(*texts: str) -> EffectMetadata:
    hand_spellboost = 0
    body: tuple[int, int] | None = None
    keywords: set[str] = set()
    refill_to_max = False
    last_words_spellboost = 0
    for text in texts:
        if not text:
            continue
        if "refill_to_max" in text:
            refill_to_max = True
        for match in _PATCH_HAND_SPELLBOOST_RE.finditer(text):
            hand_spellboost = max(hand_spellboost, int(match.group(1)))
        for match in _HAND_SPELLBOOST_RE.finditer(text):
            hand_spellboost = max(hand_spellboost, int(match.group(1) or 1))
        body_match = _BODY_RE.search(text)
        if body_match:
            body = (int(body_match.group(1)), int(body_match.group(2)))
        for match in _KEYWORD_RE.finditer(text):
            keywords.add(match.group(1))
        for match in _LAST_WORDS_SPELLBOOST_RE.finditer(text):
            last_words_spellboost = max(last_words_spellboost, int(match.group(1)))
    return EffectMetadata(
        hand_spellboost=hand_spellboost,
        body=body,
        keywords=frozenset(keywords),
        refill_to_max=refill_to_max,
        last_words_spellboost=last_words_spellboost,
    )


@dataclass(frozen=True)
class _EffectPattern:
    kind: EffectFactKind
    pattern: re.Pattern[str]


_BRACKET_TIMINGS: tuple[tuple[str, EffectFactTiming], ...] = (
    ("入场曲", EffectFactTiming.FANFARE),
    ("谢幕曲", EffectFactTiming.LAST_WORDS),
    ("超进化时", EffectFactTiming.ON_SUPER_EVOLVE),
    ("进化时", EffectFactTiming.ON_EVOLVE),
    ("攻击时", EffectFactTiming.ON_ATTACK),
    ("交战时", EffectFactTiming.ON_CLASH),
    ("启动", EffectFactTiming.ACTIVATION),
    ("魔力增幅", EffectFactTiming.SPELLBOOST),
    ("融合", EffectFactTiming.FUSION),
    ("爆能强化", EffectFactTiming.ENHANCE),
    ("激奏", EffectFactTiming.ACCELERATE),
    ("结晶", EffectFactTiming.CRYSTAL),
)

_FREE_TIMINGS: tuple[tuple[str, EffectFactTiming], ...] = (
    ("回合开始时", EffectFactTiming.TURN_START),
    ("回合结束时", EffectFactTiming.TURN_END),
    ("进入战场时", EffectFactTiming.CONTINUOUS),
    ("离开战场时", EffectFactTiming.CONTINUOUS),
    ("使用法术时", EffectFactTiming.ON_PLAY),
    ("使用法术卡时", EffectFactTiming.ON_PLAY),
    ("被破坏时", EffectFactTiming.LAST_WORDS),
    ("回复时", EffectFactTiming.CONTINUOUS),
)

_EFFECT_PATTERNS: tuple[_EffectPattern, ...] = (
    _EffectPattern(EffectFactKind.LEADER_DAMAGE, re.compile(r"主战者造成(\d+|X)点伤害|对手的主战者造成(\d+|X)点伤害|对对手的主战者造成(\d+|X)点伤害")),
    _EffectPattern(EffectFactKind.DISTRIBUTED_DAMAGE, re.compile(r"分配(\d+|X)点伤害")),
    _EffectPattern(EffectFactKind.FIELD_DAMAGE, re.compile(r"随从造成(\d+|X)点伤害|战场上的.*?造成(\d+|X)点伤害|对手的战场上的.*?造成(\d+|X)点伤害")),
    _EffectPattern(EffectFactKind.HEAL, re.compile(r"回复(?:(?!PP).)*?(\d+|X)点(?:生命值|生命)")),
    _EffectPattern(EffectFactKind.DRAW, re.compile(r"抽取(\d+|X)张|抽(\d+|X)张")),
    _EffectPattern(EffectFactKind.SUMMON, re.compile(r"召唤(\d+)?个|召唤")),
    _EffectPattern(EffectFactKind.ADD_TO_HAND, re.compile(r"加入手牌|置入手牌")),
    _EffectPattern(EffectFactKind.DESTROY, re.compile(r"破坏")),
    _EffectPattern(EffectFactKind.BANISH, re.compile(r"消灭|消失")),
    _EffectPattern(EffectFactKind.BOUNCE, re.compile(r"返回手牌|返回牌组|回到手牌")),
    _EffectPattern(EffectFactKind.TRANSFORM, re.compile(r"变身|转变为|变为")),
    _EffectPattern(EffectFactKind.STAT_BUFF, re.compile(r"\+(\d+|X)/\+(\d+|X)|\+(\d+|X)/-?(\d+|X)|获得\+|使.*\+\d+|使.*\+X")),
    _EffectPattern(EffectFactKind.STAT_DEBUFF, re.compile(r"-(\d+)/-(\d+)|使.*-\d+")),
    _EffectPattern(EffectFactKind.KEYWORD_GAIN, re.compile(r"获得【|获得『|赋予")),
    _EffectPattern(EffectFactKind.KEYWORD_REMOVE, re.compile(r"失去【|无法攻击")),
    _EffectPattern(EffectFactKind.AUTO_EVOLVE, re.compile(r"本随从进化|使.*进化|使.*超进化")),
    _EffectPattern(EffectFactKind.COST_CHANGE, re.compile(r"费用.*?变为|费用-\d+|消费-\d+|使.*费用")),
    _EffectPattern(EffectFactKind.PP_CHANGE, re.compile(r"回复.*PP|恢复.*PP|PP回复|PP恢复|增加.*PP|减少.*PP|能量点|进化点|EP")),
    _EffectPattern(EffectFactKind.SPELLBOOST_HAND, re.compile(r"(?:所有手牌|手牌中(?:的)?(?:所有)?卡牌|自己的所有手牌).*?(?:发动|进行)(\d+)?次?魔力增幅")),
    _EffectPattern(EffectFactKind.REANIMATE, re.compile(r"亡者召还_(\d+)")),
    _EffectPattern(EffectFactKind.EARTH_SIGIL, re.compile(r"土之印\+(\d+)")),
    _EffectPattern(EffectFactKind.ARTS_GAUGE, re.compile(r"奥义槽\+(\d+)")),
    _EffectPattern(EffectFactKind.VARIABLE_DEFINE, re.compile(r"X(?:开始|初始|起始)?(?:为|是)|X(?:的)?值")),
    _EffectPattern(EffectFactKind.CREST_OR_FAITH, re.compile(r"纹章|信仰")),
    _EffectPattern(EffectFactKind.DECK_OPERATION, re.compile(r"牌组|牌堆")),
    _EffectPattern(EffectFactKind.DISCARD, re.compile(r"舍弃|弃牌")),
    _EffectPattern(EffectFactKind.GRAVEYARD, re.compile(r"墓场|墓地")),
    _EffectPattern(EffectFactKind.SPECIAL_RESULT, re.compile(r"获得胜利|特殊胜利|胜利的卡牌|败北")),
)

_CONDITION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("self_evolved", "若本随从为进化后"),
    ("self_evolved", "若为进化后"),
    ("self_unevolved", "若本随从为进化前"),
    ("self_unevolved", "若为进化前"),
    ("if", "若"),
    ("if", "如果"),
    ("else", "否则"),
    ("combo", "连击"),
    ("rally", "协作"),
    ("super_ougi", "解放奥义"),
    ("ougi", "奥义"),
    ("spellboost", "魔力增幅"),
    ("earth_rite", "土之秘术"),
    ("necromancy", "唤灵"),
    ("kakusei", "觉醒"),
)

_COMPLEXITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("random", "随机"),
    ("selected_target", "选择"),
    ("all_scope", "所有"),
    ("all_scope", "全部"),
    ("variable_x", "X"),
    ("history_dependent", "本次对战"),
    ("history_dependent", "已使用"),
    ("history_dependent", "已被破坏"),
    ("continuous_trigger", "每当"),
    ("mode_choice", "模式"),
    ("special_result", "获得胜利"),
    ("combat_modifier", "伤害变为"),
    ("combat_modifier", "减少受到"),
    ("combat_modifier", "屏障"),
)

_SAFE_KINDS = {
    EffectFactKind.LEADER_DAMAGE,
    EffectFactKind.HEAL,
    EffectFactKind.STAT_BUFF,
    EffectFactKind.STAT_DEBUFF,
    EffectFactKind.KEYWORD_GAIN,
    EffectFactKind.KEYWORD_REMOVE,
}
_UNSAFE_KINDS = {
    EffectFactKind.SUMMON,
    EffectFactKind.ADD_TO_HAND,
    EffectFactKind.DRAW,
    EffectFactKind.DECK_OPERATION,
    EffectFactKind.SPECIAL_RESULT,
    EffectFactKind.AUTO_EVOLVE,
    EffectFactKind.CREST_OR_FAITH,
}


def extract_effect_facts(
    skill_desc: str,
    *,
    default_timing: EffectFactTiming = EffectFactTiming.UNKNOWN,
) -> tuple[CardEffectFact, ...]:
    if not skill_desc:
        return ()

    facts: list[CardEffectFact] = []
    for block in _split_effect_blocks(skill_desc):
        current_timing = default_timing
        current_conditions: frozenset[str] = frozenset()
        for segment in _split_effect_segments(block):
            segment_timing = _detect_timing(segment)
            if segment_timing is not EffectFactTiming.UNKNOWN:
                timing = segment_timing
                if segment_timing is not EffectFactTiming.SPELLBOOST:
                    current_timing = segment_timing
            else:
                timing = current_timing

            segment_conditions = _detect_conditions(segment)
            if segment_conditions and segment_timing is not EffectFactTiming.SPELLBOOST:
                current_conditions = segment_conditions
            conditions = segment_conditions if segment_timing is EffectFactTiming.SPELLBOOST else current_conditions

            complexity = _detect_complexity(segment)
            matched = False
            matched_kinds: set[EffectFactKind] = set()
            for effect_pattern in _EFFECT_PATTERNS:
                if "纹章" in segment and effect_pattern.kind in {EffectFactKind.BANISH, EffectFactKind.KEYWORD_GAIN}:
                    continue
                if "土之印" in segment and effect_pattern.kind is EffectFactKind.STAT_BUFF:
                    continue
                if effect_pattern.kind in matched_kinds:
                    continue
                if effect_pattern.kind is EffectFactKind.DECK_OPERATION and matched_kinds & {EffectFactKind.SUMMON, EffectFactKind.ADD_TO_HAND}:
                    continue
                if effect_pattern.kind is EffectFactKind.STAT_DEBUFF and EffectFactKind.STAT_BUFF in matched_kinds:
                    continue
                match = effect_pattern.pattern.search(segment)
                if match is None:
                    continue
                matched = True
                matched_kinds.add(effect_pattern.kind)
                fact = _build_fact(segment, timing, effect_pattern.kind, match, complexity, conditions)
                if "X" in segment and "X" in skill_desc and segment != skill_desc.strip():
                    fact = CardEffectFact(
                        timing=fact.timing,
                        kind=fact.kind,
                        amount=fact.amount,
                        count=fact.count,
                        target_side=fact.target_side,
                        target_zone=fact.target_zone,
                        target_type=fact.target_type,
                        targeting=fact.targeting,
                        value_source=fact.value_source,
                        conditions=fact.conditions,
                        safe_to_apply=fact.safe_to_apply,
                        requires_oracle=fact.requires_oracle,
                        confidence=fact.confidence,
                        raw_text=f"{fact.raw_text} {skill_desc.strip()}",
                        unsupported_reason=fact.unsupported_reason,
                    )
                facts.append(fact)
            if not matched and complexity:
                facts.append(CardEffectFact(
                    timing=timing,
                    kind=EffectFactKind.UNKNOWN_COMPLEX,
                    targeting=_detect_targeting(segment),
                    value_source=_detect_value_source(segment),
                    conditions=conditions,
                    safe_to_apply=False,
                    requires_oracle=True,
                    confidence=EffectFactConfidence.UNSUPPORTED_COMPLEX,
                    raw_text=segment,
                    unsupported_reason=",".join(sorted(complexity)),
                ))
    return _dedupe_facts(facts)


def _dedupe_facts(facts: list[CardEffectFact]) -> tuple[CardEffectFact, ...]:
    result: list[CardEffectFact] = []
    seen: set[tuple] = set()
    for fact in facts:
        key = (
            fact.timing,
            fact.kind,
            fact.amount,
            fact.count,
            fact.target_side,
            fact.target_zone,
            fact.target_type,
            fact.targeting,
            fact.value_source,
            fact.conditions,
            fact.raw_text,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return tuple(result)


def is_fact_applicable(fact: CardEffectFact, source_card: "CardView | None" = None) -> bool:
    if not fact.conditions:
        return True
    supported = {"self_evolved", "self_unevolved", "if"}
    if "self_evolved" in fact.conditions:
        if source_card is None or source_card.evolve_state <= 0:
            return False
    if "self_unevolved" in fact.conditions:
        if source_card is None or source_card.evolve_state != 0:
            return False
    for condition in fact.conditions:
        match = re.fullmatch(r"spellboost_threshold_(\d+)", condition)
        if match:
            supported.add(condition)
            if source_card is None:
                return False
            current = max(0, source_card.variable_x, source_card.spell_boost_count)
            if current < int(match.group(1)):
                return False
    unsupported = fact.conditions - supported
    if unsupported:
        return False
    return True


def _split_effect_blocks(skill_desc: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", skill_desc) if block.strip()]
    return blocks or [skill_desc.strip()]


def _split_effect_segments(skill_desc: str) -> list[str]:
    parts: list[str] = []
    for line in skill_desc.splitlines():
        for piece in re.split(r"(?<=。)", line):
            text = piece.strip()
            if text:
                parts.append(text)
    return parts or [skill_desc.strip()]


def _detect_timing(text: str) -> EffectFactTiming:
    for marker, timing in _BRACKET_TIMINGS:
        if marker in text:
            return timing
    for marker, timing in _FREE_TIMINGS:
        if marker in text:
            return timing
    return EffectFactTiming.UNKNOWN


def _build_fact(
    text: str,
    timing: EffectFactTiming,
    kind: EffectFactKind,
    match: re.Match[str],
    complexity: frozenset[str],
    conditions: frozenset[str],
) -> CardEffectFact:
    amount = _first_int(match)
    count = _detect_count(text)
    targeting = _detect_targeting(text)
    value_source = _detect_value_source(text)
    safe = _is_safe_to_apply(kind, targeting, value_source, complexity, conditions)
    return CardEffectFact(
        timing=timing,
        kind=kind,
        amount=amount,
        count=count,
        target_side=_detect_target_side(text),
        target_zone=_detect_target_zone(text),
        target_type=_detect_target_type(text),
        targeting=targeting,
        value_source=value_source,
        conditions=conditions,
        safe_to_apply=safe,
        requires_oracle=not safe,
        raw_text=text,
        unsupported_reason=",".join(sorted(complexity)) if not safe and complexity else "",
    )


def _first_int(match: re.Match[str]) -> int | None:
    for group in match.groups():
        if group == "X":
            return None
        if group and group.isdigit():
            return int(group)
    own = re.search(r"\d+", match.group(0))
    return int(own.group(0)) if own else None


def _detect_count(text: str) -> int | None:
    match = re.search(r"(\d+)个|(\d+)张|抽取(\d+)张|抽(\d+)张", text)
    return _first_int(match) if match else None


def _detect_target_side(text: str) -> str:
    if "对手" in text or "敌方" in text:
        return "opponent"
    if "自己" in text or "本随从" in text or "自己的" in text:
        return "self"
    return ""


def _detect_target_zone(text: str) -> str:
    if "战场" in text:
        return "field"
    if "手牌" in text and "返回手牌" not in text and "回到手牌" not in text:
        return "hand"
    if "牌组" in text or "牌堆" in text:
        return "deck"
    if "墓场" in text or "墓地" in text:
        return "graveyard"
    if "主战者" in text:
        return "leader"
    return ""


def _detect_target_type(text: str) -> str:
    if "纹章" in text or "信仰" in text:
        return "crest"
    if "主战者" in text:
        return "leader"
    if "随从" in text:
        return "follower"
    if "护符" in text:
        return "amulet"
    if "卡牌" in text or "卡" in text:
        return "card"
    return ""


def _detect_targeting(text: str) -> EffectTargeting:
    if "随机" in text:
        return EffectTargeting.RANDOM
    if "选择" in text:
        return EffectTargeting.SELECTED
    if "所有" in text or "全部" in text:
        return EffectTargeting.ALL
    if "本随从" in text:
        return EffectTargeting.SELF
    return EffectTargeting.NONE


def _detect_value_source(text: str) -> EffectValueSource:
    if "本次对战" in text or "已使用" in text or "已被破坏" in text:
        return EffectValueSource.HISTORY_BASED
    if "X" in text:
        return EffectValueSource.VARIABLE_X
    if "数量" in text or "次数" in text or "每" in text or "手牌数" in text or "连击" in text or "协作" in text:
        return EffectValueSource.COUNT_BASED
    if re.search(r"\d+", text):
        return EffectValueSource.FIXED
    return EffectValueSource.NONE


def _detect_conditions(text: str) -> frozenset[str]:
    conditions = []
    for tag in _structured_condition_tags(text):
        if tag not in conditions:
            conditions.append(tag)
    for tag, marker in _CONDITION_PATTERNS:
        if marker in text and tag not in conditions:
            conditions.append(tag)
    return frozenset(conditions)


def _structured_condition_tags(text: str) -> list[str]:
    tags: list[str] = []
    patterns = (
        ("combo_threshold", (r"连击[_ ]?(\d+)", r"连击(?:数)?(?:为|是)?(\d+)以上")),
        ("rally_threshold", (r"协作[_ ]?(\d+)", r"协作(?:数)?(?:为|是)?(\d+)以上")),
        ("ougi_threshold", (r"奥义[_ ]?(\d+)",)),
        ("super_ougi_threshold", (r"解放奥义[_ ]?(\d+)",)),
        ("earth_rite_cost", (r"土之秘术[_ ]?(\d+)",)),
        ("spellboost_threshold", (r"魔力增幅[_ ]?(\d+)", r"魔力增幅(?:次数)?(?:为|是)?(\d+)以上")),
        ("necromancy_threshold", (r"唤灵[_ ]?(\d+)", r"唤灵(?:为|是)?(\d+)以上")),
        ("graveyard_threshold", (r"墓[场地](?:数量|数)?(?:为|是)?(\d+)以上",)),
        ("hand_count_threshold", (r"手牌(?:数)?(?:为|是)?(\d+)张?以上", r"手牌(?:数)?(?:为|是)?(\d+)张")),
        ("field_count_threshold", (r"战场上(?:的)?(?:卡牌|随从)?(?:数|数量)?(?:为|是)?(\d+)以上",)),
        ("deck_count_threshold", (r"牌[组堆](?:剩余)?(?:张数|数量|数)?(?:为|是)?(\d+)以上",)),
        ("variable_x_threshold", (r"X(?:为|是)?(\d+)以上",)),
        ("history_played_total_threshold", (r"本次对战中已使用(?:的)?卡牌(?:数|数量)?(?:为|是)?(\d+)以上",)),
        ("history_destroyed_total_threshold", (r"本次对战中已被破坏(?:的)?(?:卡牌|随从)?(?:数|数量)?(?:为|是)?(\d+)以上",)),
    )
    for prefix, regexes in patterns:
        for regex in regexes:
            match = re.search(regex, text)
            if match:
                tags.append(f"{prefix}_{int(match.group(1))}")
                break
    if "觉醒" in text:
        tags.append("kakusei")
    return tags


def _detect_complexity(text: str) -> frozenset[str]:
    found = []
    for tag, marker in _COMPLEXITY_PATTERNS:
        if marker in text and tag not in found:
            found.append(tag)
    return frozenset(found)


def _is_safe_to_apply(
    kind: EffectFactKind,
    targeting: EffectTargeting,
    value_source: EffectValueSource,
    complexity: frozenset[str],
    conditions: frozenset[str],
) -> bool:
    if kind in _UNSAFE_KINDS:
        return False
    if kind not in _SAFE_KINDS:
        return False
    if targeting in {EffectTargeting.RANDOM, EffectTargeting.SELECTED, EffectTargeting.ALL, EffectTargeting.UNKNOWN}:
        return False
    if value_source not in {EffectValueSource.FIXED, EffectValueSource.NONE}:
        return False
    hard_complexity = complexity - {"selected_target"}
    if hard_complexity:
        return False
    if conditions - {"self_evolved", "self_unevolved"}:
        return False
    return True
