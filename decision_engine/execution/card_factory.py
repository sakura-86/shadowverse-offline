# -*- coding: utf-8 -*-
"""从卡牌知识构造可执行的 CardView。

``LocalMatchSetup``（self-play 建局）、自由盘面沙盒、fixture 构造都需要把
``UnifiedCardTemplate`` / ``BaseCardInfo`` 变成 ``CardView``。这条路径必须只有一份：
否则同一张卡在沙盒里「可打出爆能」、在 self-play 里却枚举不出来（或反过来），
spec 验收就失去了意义。

本模块只做「知识 → 状态视图」的投影，不解释也不执行效果。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from decision_engine.state.snapshot import CardType, CardView

from .state_ops import static_keyword_card_updates

# BaseCardMaster.Type → CardType，对齐 docs/reference/master-data-notes.md。
CARD_TYPE_BY_TYPE_ID: dict[int, CardType] = {
    1: CardType.FOLLOWER,
    2: CardType.AMULET,
    3: CardType.COUNTDOWN_AMULET,
    4: CardType.SPELL,
    5: CardType.CREST,
    6: CardType.COUNTDOWN_CREST,
    7: CardType.FAITH,
}

_COST_FIELDS = frozenset({"enhance_costs", "accelerate_costs", "crystal_costs"})


def card_type_for_type_id(type_id: int) -> CardType:
    """把协议/主数据 type_id 映射为 CardType；未知类型显式失败。"""
    try:
        return CARD_TYPE_BY_TYPE_ID[int(type_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"未知卡牌类型 type_id={type_id!r}，无法构造 CardView") from exc


def card_type_for_name(type_name: str) -> CardType:
    """主数据 type_name 的兜底解析（type_id 缺失时使用）。

    语义与 ``LocalMatchSetup`` 的历史实现一致：先精确匹配，再做子串匹配，
    最后归为 FOLLOWER。主数据里 type_name 是稳定的英文枚举名，不需要更复杂的解析。
    """
    normalized = str(type_name or "").strip().lower()
    if normalized in {"spell", "spell_evolved"}:
        return CardType.SPELL
    if normalized == "countdown_amulet":
        return CardType.COUNTDOWN_AMULET
    if "amulet" in normalized:
        return CardType.AMULET
    if normalized == "countdown_crest":
        return CardType.COUNTDOWN_CREST
    if "crest" in normalized:
        return CardType.CREST
    if normalized == "faith":
        return CardType.FAITH
    if "boon" in normalized:
        return CardType.AMULET  # boon 不在 1~7 范围，归为 amulet
    return CardType.FOLLOWER


def _play_mode_kind(mode: Any) -> str:
    kind = getattr(mode, "kind", None)
    if kind is None:
        return ""
    return str(getattr(kind, "value", kind))


def _play_mode_cost(mode: Any) -> int | None:
    cost = getattr(mode, "cost", None)
    value = getattr(cost, "value", None) if cost is not None else None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def play_mode_costs(template: Any) -> dict[str, tuple[int, ...]]:
    """从模板 play_modes 提取 enhance / accelerate / crystal 费用。

    缺少模板时返回空集合；调用方据此得到「这张卡没有这些打出方式」，
    而不是猜一个费用。
    """
    buckets: dict[str, set[int]] = {"enhance": set(), "accelerate": set(), "crystal": set()}
    for mode in getattr(template, "play_modes", ()) or ():
        kind = _play_mode_kind(mode)
        if kind not in ("enhance", "accelerate", "crystallize"):
            continue
        cost = _play_mode_cost(mode)
        if cost is None:
            continue
        buckets["crystal" if kind == "crystallize" else kind].add(cost)
    return {name: tuple(sorted(values)) for name, values in buckets.items()}


def template_has_trigger(template: Any, events: frozenset[str]) -> bool:
    """模板是否声明了某个触发事件（如 ``last_words`` / ``spellboost``）。"""
    for effect in getattr(template, "executable_effects", ()) or ():
        trigger = getattr(effect, "trigger", None)
        if trigger is not None and getattr(trigger, "event", "") in events:
            return True
    return False


def template_has_persistent_trigger(template: Any, events: frozenset[str]) -> bool:
    for persistent in getattr(template, "persistent_effects", ()) or ():
        trigger = getattr(persistent, "trigger", None)
        if trigger is not None and getattr(trigger, "event", "") in events:
            return True
    return False


def build_card_view(
    *,
    uid: int,
    base_card_id: int,
    card_id: int | None = None,
    template: Any | None = None,
    info: Any | None = None,
    skill_desc: str | None = None,
    supplement_info: Mapping[str, int] | None = None,
    field_entry: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> CardView:
    """构造一张 CardView。

    优先级：``overrides`` > 主数据/模板 > CardView 默认值。

    ``template`` 与 ``info`` 至少给一个，否则无法确定卡牌类型，直接报错而不是造一张
    全 0 的假卡（那会让枚举器产出并不存在的合法动作）。

    ``field_entry=True`` 表示这张卡是「刚进入战场」的形态：风暴/突进的随从会同时拿到
    ``can_attack`` / ``attacks_left``。建局/牌库构造（默认 False）只落静态关键词，
    把「能否攻击」留给回合开始与入场物化决定。
    """
    if base_card_id <= 0:
        raise ValueError(f"base_card_id 必须为正数，收到 {base_card_id!r}")
    if template is None and info is None:
        raise ValueError(f"base_card_id={base_card_id} 既没有模板也没有主数据，无法构造 CardView")

    identity = getattr(template, "identity", None)
    printed = getattr(template, "printed", None)

    type_id = int(getattr(info, "type_id", 0) or getattr(identity, "type_id", 0) or 0)
    if type_id in CARD_TYPE_BY_TYPE_ID:
        card_type = card_type_for_type_id(type_id)
    else:
        card_type = card_type_for_name(
            str(getattr(info, "card_type", "") or getattr(identity, "type_name", "") or "")
        )

    name = str(getattr(info, "name", "") or getattr(identity, "name", "") or "")
    class_id = int(getattr(info, "class_id", 0) or getattr(identity, "class_id", 0) or 0)
    resolved_card_id = int(
        card_id
        or getattr(identity, "normal_card_id", None)
        or getattr(info, "normal_card_id", 0)
        or base_card_id * 10
    )

    desc = str(skill_desc if skill_desc is not None else getattr(info, "skill_desc", "") or "")

    cost = int(getattr(info, "cost", 0) or 0) if info is not None else int(getattr(printed, "cost", 0) or 0)
    atk = int(getattr(info, "atk", 0) or 0) if info is not None else int(getattr(printed, "atk", 0) or 0)
    life = int(getattr(info, "life", 0) or 0) if info is not None else int(getattr(printed, "life", 0) or 0)
    countdown = int(getattr(printed, "countdown", 0) or 0) if printed is not None else 0

    tribes: tuple[int, ...] = ()
    for tribe in getattr(info, "tribes", ()) if info is not None else getattr(identity, "tribes", ()) or ():
        if isinstance(tribe, int):
            tribes += (tribe,)

    has_spell_boost = (
        template_has_trigger(template, frozenset({"spellboost"}))
        or template_has_persistent_trigger(template, frozenset({"spellboost"}))
        or "【魔力增幅时】" in desc
    )
    spellboost_reduces_cost = bool(
        template is not None
        and has_spell_boost
        and any(
            getattr(effect.trigger, "event", "") == "spellboost"
            and any(getattr(op, "op", "") == "change_cost" for op in effect.operations)
            for effect in template.executable_effects
            if getattr(effect, "trigger", None) is not None
        )
    )
    is_cant_attack = card_type.is_follower and "无法攻击" in desc
    is_earth_sigil = "土之印" in desc
    attack_limit = 0 if is_cant_attack or not card_type.is_follower else 1

    costs = play_mode_costs(template) if template is not None else {"enhance": (), "accelerate": (), "crystal": ()}

    card = CardView(
        uid=int(uid),
        card_id=resolved_card_id,
        base_card_id=int(base_card_id),
        name=name,
        class_id=class_id,
        cost=cost,
        atk=atk,
        life=life,
        max_life=life,
        card_type=card_type,
        countdown=countdown,
        tribes=tribes,
        has_cant_attack=is_cant_attack,
        has_activation=bool(template is not None and template_has_trigger(template, frozenset({"activate", "activation"}))),
        has_last_words=bool(
            template is not None
            and (
                template_has_trigger(template, frozenset({"last_words"}))
                or any(
                    str(getattr(kw, "keyword", "")) == "last_words"
                    for kw in getattr(template, "static_keywords", ()) or ()
                )
            )
        ),
        has_spell_boost=has_spell_boost,
        spellboost_reduces_cost=spellboost_reduces_cost,
        is_earth_sigil=is_earth_sigil,
        # 【土之印】护符不会被破坏：破坏类效果作用在土之印上是「消耗」而不是
        # 「破坏」，协议对它恒定下发 has_cant_destroy=true（replay_...1364304
        # st8 的 10031210 打出后即为 true）。土之印的消耗走 consume_earth_sigil op。
        has_cant_destroy=is_earth_sigil,
        attack_limit=attack_limit,
        enhance_costs=costs["enhance"],
        can_enhance=bool(costs["enhance"]),
        accelerate_costs=costs["accelerate"],
        crystal_costs=costs["crystal"],
        supplement_info=dict(supplement_info or {}),
    )

    if template is not None:
        updates: dict[str, Any] = {}
        for mode in getattr(template, "play_modes", ()) or ():
            kind = _play_mode_kind(mode)
            if kind == "activation":
                updates["has_activation"] = True
            elif kind == "special_action":
                updates["can_special_action"] = True
        keyword_updates = static_keyword_card_updates(card, template, enable_attacks=field_entry)
        if keyword_updates:
            updates.update(keyword_updates)
        if updates:
            card = replace(card, **updates)

    if overrides:
        unknown = sorted(set(overrides) - set(CardView.__dataclass_fields__))
        if unknown:
            raise ValueError(f"CardView 不存在这些字段: {unknown}")
        card = replace(card, **_coerce_overrides(overrides))

    return card


def _coerce_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """把前端传来的 JSON 值收敛成 CardView 字段类型。"""
    coerced: dict[str, Any] = {}
    for field_name, value in overrides.items():
        if field_name == "card_type":
            coerced[field_name] = value if isinstance(value, CardType) else CardType(str(value))
        elif field_name in _COST_FIELDS:
            coerced[field_name] = tuple(int(v) for v in (value or ()))
        elif field_name == "tribes":
            coerced[field_name] = tuple(int(v) for v in (value or ()))
        elif field_name == "fusion_list":
            coerced[field_name] = tuple(tuple(int(uid) for uid in group) for group in (value or ()))
        elif field_name == "supplement_info":
            coerced[field_name] = {str(k): int(v) for k, v in (value or {}).items()}
        elif field_name == "card_type" and isinstance(value, CardType):
            coerced[field_name] = value
        else:
            coerced[field_name] = value
    return coerced
