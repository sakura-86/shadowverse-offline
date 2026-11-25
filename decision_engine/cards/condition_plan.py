# -*- coding: utf-8 -*-
"""打出相关条件的静态计划（每 base_card_id 一份）。

背景：卡牌效果是否「绑定条件」无法从协议获得（`SkillEffectYellowCards` 只给出已达成的
结果），因此必须在本地由卡牌模板推导。本模块把 `UnifiedCardTemplate` 里与「现在打出
是否更值」相关的效果条件编译成可快速求值的计划，供 `ConditionProgressProjector`
在每次状态转移后刷新条件进度。

边界（显式排除，不静默）：

- 只收录 `trigger.event` 属于 `PLAY_RELEVANT_EVENTS`，或 `trigger.play_modes` 命中
  `PLAY_MODE_TOKENS` 的效果。`last_words` / `turn_end` / `on_attack` 等条件的达成
  与留牌决策不是同一维度，不进入本计划。
- 执行期参数条件（`subject` 以 `runtime.` 开头）与动作依赖条件
  （`selection_filter` / `event_material_filter` / `event_snapshot_exists`）在手牌
  状态下无法判定，单独计数排除，不混入 `unknown`。
- 模板缺失时 `coverage="unknown"`，表示「未建模」，绝不能当作「无条件」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from decision_engine.cards.templates import ConditionSpec, UnifiedCardTemplate

# 与「打出时效果强度」直接相关的事件。
# mode_choice_resolution / ability_resolution 承载「若满足条件则改为发动所有能力」这类
# 模式门控；on_spell_play 是法术卡自身的打出触发。
PLAY_RELEVANT_EVENTS = frozenset(
    {"fanfare", "enhance", "spellboost", "mode_choice_resolution", "ability_resolution", "on_spell_play"}
)
# 通过 play_modes 表达的打出模式。
PLAY_MODE_TOKENS = frozenset({"normal_play", "enhance", "accelerate", "crystal"})
# 依赖执行期动作参数的条件（如 selected_mode_options / is_enhance）。
_EXCLUDED_SUBJECT_PREFIXES = ("runtime.",)
# 依赖事件上下文或动作参数、无法在手牌状态下判定的条件种类。
_EXCLUDED_KINDS = frozenset({"selection_filter", "event_material_filter", "event_snapshot_exists"})

# 关键字型条件：卡面出现这些关键字即表示效果绑定条件，即使模板没有拆出结构化条件。
# (关键字, 条件 subject, 阈值正则列表)；阈值提取不到时记为不可判定，不猜数字。
_KEYWORD_THRESHOLD_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("连击", "me.play_count", (r"连击(?:数)?(?:为|是)?(\d+)以上", r"连击[_ ]?(\d+)")),
    ("协作", "me.rally", (r"协作(?:数)?(?:为|是)?(\d+)以上", r"协作[_ ]?(\d+)")),
    ("唤灵", "me.graveyard_count", (r"唤灵(?:为|是)?(\d+)以上", r"唤灵[_ ]?(\d+)")),
    ("土之秘术", "me.earth_sigil", (r"土之秘术[_ ]?(\d+)",)),
    ("魔力增幅", "source.spellboost_count", (r"魔力增幅(?:次数)?(?:为|是)?(\d+)以上", r"魔力增幅[_ ]?(\d+)")),
    ("信仰", "me.faith_value", (r"信仰(?:值)?(?:为|是)?(\d+)以上", r"信仰[_ ]?(\d+)")),
)
# 关键字出现但属于「赋予/增加/发动」语义时不是条件（例如「使自己的【连击】+1」）。
_KEYWORD_FALSE_POSITIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("连击", r"连击[】\]]?\+"),
    ("连击", r"(获得|给予)[^。]{0,10}连击"),
    ("魔力增幅", r"(发动|给予|获得)[^。]{0,10}魔力增幅"),
    ("信仰", r"(获得|给予|增加)[^。]{0,10}信仰"),
)

# 「留牌潜在价值」的另外两类来源：
# 1) 本卡牌被舍弃时发动 —— 留牌可被弃牌效果兑现；
# 2) 在手牌中因条件降费 —— 等待降费后以更低费用打出。
_SELF_DISCARD_EVENTS = frozenset({"on_discard_self"})
_SELF_DISCARD_TEXT = "本卡牌被舍弃时"
_HAND_COST_REDUCTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"本卡牌的费用\s*[-−]\s*(?:\d+|X)"),
    re.compile(r"本卡牌的费用(?:变为|减为|减少)\s*\d+"),
)


@dataclass(frozen=True)
class _IfClauseRule:
    """「若…」从句 → 可求值条件。value=None 表示取正则第 1 组。"""

    pattern: re.Pattern[str]
    kind: str
    subject: str
    operator: str
    value: int | tuple[int, ...] | None = None


# 只收录能明确映射到 snapshot 事实的从句。歧义模式（如「若为进化后」可能指本卡或场上）
# 不在这里处理，保持未建模，避免把错误状态喂给网络。
_IF_CLAUSE_RULES: tuple[_IfClauseRule, ...] = (
    _IfClauseRule(re.compile(r"战场上的?卡牌(?:的)?张数为(\d+)张?或以上"), "snapshot_threshold", "me.field_count", ">="),
    _IfClauseRule(re.compile(r"战场上有原始费用为(\d+)或以上的卡牌"), "snapshot_threshold", "me.field_max_original_cost", ">="),
    _IfClauseRule(re.compile(r"战场上有超进化后的随从"), "snapshot_threshold", "me.field_super_evolved_follower", ">=", 1),
    _IfClauseRule(re.compile(r"战场上有进化后的随从"), "snapshot_threshold", "me.field_evolved_follower", ">=", 1),
    _IfClauseRule(re.compile(r"主战者(?:的)?生命值为(\d+)或以下"), "snapshot_threshold", "me.leader_life", "<="),
    _IfClauseRule(re.compile(r"牌组中没有重复卡牌"), "snapshot_threshold", "me.deck_has_no_duplicates", "==", 1),
    _IfClauseRule(re.compile(r"自己的随从在自己的上一回合中攻击过主战者"), "snapshot_threshold", "me.follower_attacked_leader_previous_turn", "==", 1),
    _IfClauseRule(re.compile(r"本回合中自己的随从没有进行过攻击"), "snapshot_threshold", "me.attack_count", "==", 0),
    _IfClauseRule(re.compile(r"(?:本卡牌|卡牌)的费用为(\d+)"), "snapshot_threshold", "source.cost", "=="),
    _IfClauseRule(re.compile(r"已与本卡牌【融合】"), "snapshot_threshold", "source.is_fused", "==", 1),
    _IfClauseRule(re.compile(r"进入战场的自己的创造物·随从的种类为(\d+)种或以上"), "snapshot_threshold", "me.artifact_types_entered", ">="),
    _IfClauseRule(
        re.compile(r"本次对战中自己使用的卡牌的原始费用包含1到8"),
        "snapshot_threshold",
        "me.played_original_cost_set",
        "covers",
        (1, 2, 3, 4, 5, 6, 7, 8),
    ),
)


def _if_clause_conditions(text: str) -> list[ConditionSpec]:
    specs: list[ConditionSpec] = []
    for rule in _IF_CLAUSE_RULES:
        match = rule.pattern.search(text)
        if match is None:
            continue
        if rule.value is not None:
            value = rule.value
        else:
            value = int(match.group(1))
        specs.append(ConditionSpec(kind=rule.kind, subject=rule.subject, operator=rule.operator, value=value))
    return specs
# 固定语义关键字：阈值由规则决定，不依赖文本数字。
_KEYWORD_FIXED_RULES: tuple[tuple[str, ConditionSpec], ...] = (
    ("解放奥义", ConditionSpec(kind="text_condition", subject="super_ougi", operator="requires_review")),
    ("奥义", ConditionSpec(kind="text_condition", subject="ougi", operator="requires_review")),
    ("觉醒", ConditionSpec(kind="snapshot_bool", subject="me.is_kakusei", operator="==", value=1)),
)


@dataclass(frozen=True)
class ConditionPlan:
    """单张卡的打出条件计划。

    条件按 spec 分组：组内 AND（同一效果的多个门槛），组间 OR（同一张卡的不同强化
    效果/互斥分支）。这样 if/else 两个 spec 不会互相矛盾地把卡判成「永远未达成」。
    """

    base_card_id: int
    groups: tuple[tuple[ConditionSpec, ...], ...] = ()
    coverage: str = "unknown"  # "modeled" | "unknown"
    excluded_runtime: int = 0
    excluded_action_dependent: int = 0
    keyword_conditions: int = 0
    keyword_unresolved: int = 0
    if_clause_conditions: int = 0
    # 留牌潜在价值的另外两类静态来源（与「效果条件」维度独立）。
    has_discard_payoff: bool = False
    has_hand_cost_reduction: bool = False

    @property
    def keys(self) -> tuple[ConditionSpec, ...]:
        """扁平视图（诊断/兼容用）；求值必须按 groups。"""
        return tuple(condition for group in self.groups for condition in group)

    @property
    def has_conditional_effect(self) -> bool:
        return bool(self.groups)



def _condition_key(condition: ConditionSpec) -> tuple[Any, ...]:
    # value 可能是 list/dict（不可 hash），统一用 repr 作为去重键。
    return (condition.kind, condition.subject, condition.operator, repr(condition.value), condition.ref)


def _is_play_relevant(effect: Any) -> bool:
    trigger = getattr(effect, "trigger", None)
    if trigger is None:
        return False
    if trigger.event in PLAY_RELEVANT_EVENTS:
        return True
    return bool(set(trigger.play_modes) & PLAY_MODE_TOKENS)


def _is_keyword_condition_context(text: str, keyword: str) -> bool:
    """排除「赋予/增加/发动关键字」这类非条件语境。"""
    for false_positive_keyword, pattern in _KEYWORD_FALSE_POSITIVE_PATTERNS:
        if false_positive_keyword == keyword and re.search(pattern, text):
            return False
    return True


def _source_text(template: UnifiedCardTemplate) -> str:
    text = template.source_text
    return " ".join(filter(None, (text.clean_skill_desc, text.normal_desc, text.raw_skill_desc)))


def _keyword_condition_specs(text: str, existing_subjects: set[str]) -> tuple[list[ConditionSpec], int]:
    """从卡面关键字补出条件。

    【连击】【奥义】【解放奥义】【协作】【唤灵】【觉醒】【土之秘术】【魔力增幅】【信仰】
    本身就是条件标记：模板没有拆出结构化条件时，用关键字补一条，避免把这类卡误判成平卡。
    阈值型关键字（连击 3 以上等）从文本提取数字；提取不到时记为不可判定（UNKNOWN），
    不猜默认阈值。已被结构化条件覆盖的 subject 不重复添加。
    """
    specs: list[ConditionSpec] = []
    unresolved = 0
    remaining = text.replace("解放奥义", "")
    if "解放奥义" in text and "super_ougi" not in existing_subjects:
        specs.append(_KEYWORD_FIXED_RULES[0][1])
        existing_subjects.add("super_ougi")
    if "奥义" in remaining and "ougi" not in existing_subjects:
        specs.append(_KEYWORD_FIXED_RULES[1][1])
        existing_subjects.add("ougi")
    if "觉醒" in remaining and "me.is_kakusei" not in existing_subjects:
        specs.append(_KEYWORD_FIXED_RULES[2][1])
        existing_subjects.add("me.is_kakusei")
    for keyword, subject, patterns in _KEYWORD_THRESHOLD_RULES:
        if keyword not in remaining or subject in existing_subjects:
            continue
        if not _is_keyword_condition_context(remaining, keyword):
            continue
        value: int | None = None
        for pattern in patterns:
            match = re.search(pattern, remaining)
            if match:
                value = int(match.group(1))
                break
        if value is None:
            specs.append(
                ConditionSpec(
                    kind="text_condition",
                    subject=f"keyword_{keyword}",
                    operator="requires_review",
                )
            )
            unresolved += 1
        else:
            specs.append(ConditionSpec(kind="snapshot_threshold", subject=subject, operator=">=", value=value))
        existing_subjects.add(subject)
    return specs, unresolved


def build_condition_plan(template: UnifiedCardTemplate | None, base_card_id: int) -> ConditionPlan:
    """从卡牌模板构建条件计划。模板缺失时返回 coverage=unknown 的空计划。"""
    if template is None:
        return ConditionPlan(base_card_id=base_card_id, coverage="unknown")
    groups: list[tuple[ConditionSpec, ...]] = []
    seen_groups: set[tuple[Any, ...]] = set()
    known_subjects: set[str] = set()
    excluded_runtime = 0
    excluded_action_dependent = 0

    def add_group(group: list[ConditionSpec]) -> bool:
        if not group:
            return False
        signature = tuple(sorted(_condition_key(condition) for condition in group))
        if signature in seen_groups:
            return False
        seen_groups.add(signature)
        groups.append(tuple(group))
        known_subjects.update(condition.subject for condition in group)
        return True

    # 每个 play 相关 spec 的条件是一组；组间互斥分支（if/else）因此不会相加成矛盾条件。
    for effect in template.executable_effects:
        if not _is_play_relevant(effect):
            continue
        group: list[ConditionSpec] = []
        for condition in effect.conditions:
            if condition.kind in _EXCLUDED_KINDS:
                excluded_action_dependent += 1
                continue
            if condition.subject.startswith(_EXCLUDED_SUBJECT_PREFIXES):
                excluded_runtime += 1
                continue
            group.append(condition)
        add_group(group)

    source_text = _source_text(template)
    keyword_specs, keyword_unresolved = _keyword_condition_specs(source_text, known_subjects)
    keyword_conditions = len(keyword_specs) if add_group(list(keyword_specs)) else 0

    # 「若…」从句只在有 play 相关效果时解析：纹章/护符的回合结束条件不属于打出价值。
    if_clause_conditions = 0
    if any(_is_play_relevant(effect) for effect in template.executable_effects):
        for spec in _if_clause_conditions(source_text):
            if add_group([spec]):
                if_clause_conditions += 1

    return ConditionPlan(
        base_card_id=base_card_id,
        groups=tuple(groups),
        coverage="modeled",
        excluded_runtime=excluded_runtime,
        excluded_action_dependent=excluded_action_dependent,
        keyword_conditions=keyword_conditions,
        keyword_unresolved=keyword_unresolved,
        if_clause_conditions=if_clause_conditions,
        has_discard_payoff=(
            any(effect.trigger.event in _SELF_DISCARD_EVENTS for effect in template.executable_effects)
            or _SELF_DISCARD_TEXT in source_text
        ),
        has_hand_cost_reduction=any(pattern.search(source_text) for pattern in _HAND_COST_REDUCTION_PATTERNS),
    )


class ConditionPlanCache:
    """按 base_card_id 惰性构建并缓存条件计划。

    缓存生命周期与 LocalPredictor 实例一致，运行期不再做模板遍历或字符串解析。
    """

    def __init__(self, templates_by_base_card_id: Mapping[int, UnifiedCardTemplate] | None):
        self._templates = dict(templates_by_base_card_id or {})
        self._cache: dict[int, ConditionPlan] = {}

    @property
    def enabled(self) -> bool:
        """未加载任何卡牌模板时条件投影不启用。

        这类调用方（例如只测规则内核的 LocalPredictor()）没有条件知识，投影全部
        标 UNKNOWN 只会制造噪声；显式返回「未启用」，特征层按 unknown 输出。
        """
        return bool(self._templates)

    def get(self, base_card_id: int) -> ConditionPlan:
        plan = self._cache.get(base_card_id)
        if plan is None:
            plan = build_condition_plan(self._templates.get(base_card_id), base_card_id)
            self._cache[base_card_id] = plan
        return plan
