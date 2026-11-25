# -*- coding: utf-8 -*-
"""已知牌组池与 hidden draw 分支辅助工具。"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from decision_engine.cards import CardResolver, MasterDataProvider
from decision_engine.state.deck_instance import DeckInstanceState, make_deck_instance_state
from decision_engine.state.snapshot import BattleSnapshot, CardIdCount, CardType, CardView, DeckPoolCard, DeckPoolView


class DeckPoolError(RuntimeError):
    """牌组池解析或材料化失败。"""


_DEFAULT_CARD_DATABASE = Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"


def parse_practice_deck_pool(payload: dict[str, Any]) -> DeckPoolView:
    """从 practice_deck_state payload 解析我方已知牌组池。"""

    if not isinstance(payload, dict):
        raise DeckPoolError("practice_deck_state payload must be a dict")
    source = "practice_deck_state.user_deck"
    if payload.get("type") in {"practice_deck_state", "PracticeDeckState"}:
        data = payload.get("data")
        if not isinstance(data, dict):
            raise DeckPoolError("practice_deck_state payload requires data dict")
        user_deck = data.get("user_deck") or data.get("userDeck")
    elif "user_deck" in payload or "userDeck" in payload:
        user_deck = payload.get("user_deck") or payload.get("userDeck")
        source = "direct.user_deck"
    else:
        raise DeckPoolError(f"unsupported practice deck payload type: {payload.get('type')}")
    if not isinstance(user_deck, dict):
        raise DeckPoolError("practice_deck_state payload requires user_deck dict")

    raw_cards = user_deck.get("cards") or user_deck.get("Cards") or ()
    if not isinstance(raw_cards, list):
        raise DeckPoolError("practice_deck_state user_deck.cards must be a list")

    cards: list[DeckPoolCard] = []
    observed_total = _to_int(user_deck.get("card_count_total", user_deck.get("cardCountTotal", 0)))
    total_count = 0
    for raw_card in raw_cards:
        if not isinstance(raw_card, dict):
            continue
        card_id = _to_int(raw_card.get("card_id", raw_card.get("cardId", 0)))
        count = _to_int(raw_card.get("count", raw_card.get("Count", 0)))
        if card_id <= 0 or count <= 0:
            continue
        total_count += count
        cards.append(
            DeckPoolCard(
                card_id=card_id,
                count=count,
            )
        )

    if observed_total <= 0:
        observed_total = total_count

    status = "available"
    warnings: list[str] = []
    if observed_total != total_count:
        status = "invalid"
        warnings.append("practice_deck_state_card_count_total_mismatch")

    return DeckPoolView(
        source=source,
        cards=tuple(cards),
        observed_total=observed_total,
        remaining_count=total_count,
        status=status,
        warnings=tuple(warnings),
    )


def build_deck_instances_from_practice_deck(payload: dict[str, Any], *, resolver: CardResolver | None = None, owner: str = "me") -> DeckInstanceState:
    pool = parse_practice_deck_pool(payload)
    resolver = resolver or CardResolver(MasterDataProvider.from_json(_DEFAULT_CARD_DATABASE))
    cards: list[CardView] = []
    uid = -1
    for item in pool.cards:
        for _ in range(item.count):
            card = materialize_draw_card(item.card_id, BattleSnapshot(me=_empty_player(), opponent=_empty_player(), cards_by_uid={}), uid=uid, resolver=resolver)
            if card is None:
                warnings = (*pool.warnings, f"deck_instance_card_not_materializable:{item.card_id}")
                return DeckInstanceState(source=pool.source, status="invalid", warnings=warnings)
            cards.append(card)
            uid -= 1
    state = make_deck_instance_state(cards, owner=owner, source=pool.source)
    status = pool.status
    warnings = pool.warnings
    if pool.observed_total != len(cards):
        status = "invalid"
        warnings = (*warnings, "deck_instance_count_mismatch")
    return replace(state, status=status, warnings=warnings)


def reconcile_deck_instances_with_snapshot(state: DeckInstanceState, snapshot: BattleSnapshot, *, player: str = "me") -> DeckInstanceState:
    """基于当前 snapshot 从 draw_pile 扣除可确认已离开牌组的实例。"""

    view = snapshot.me if player == "me" else snapshot.opponent
    visible = Counter[int]()
    visible.update(_card_id for _card_id in _visible_card_ids(snapshot, view.hand))
    visible.update(_card_id for _card_id in _visible_card_ids(snapshot, view.field))
    visible.update(_card_id for _card_id in _counted_card_ids(view.played_card_ids))
    visible.update(_card_id for _card_id in _counted_card_ids(view.destroyed_card_ids))
    if not visible:
        return state

    by_id = state.by_instance_id()
    draw_pile = list(state.draw_pile)
    warnings: list[str] = list(state.warnings)
    for card_id, count in visible.items():
        remaining = count
        for instance_id in tuple(draw_pile):
            if remaining <= 0:
                break
            instance = by_id.get(instance_id)
            if instance is None:
                continue
            card = instance.card
            if card.card_id != card_id and card.base_card_id != card_id:
                continue
            draw_pile.remove(instance_id)
            remaining -= 1
        if remaining > 0:
            warnings.append(f"deck_instance_visible_count_exceeds_pool:{card_id}")
    status = state.status
    if len(draw_pile) != view.deck_count:
        status = "ambiguous" if status == "available" else status
        warnings.append("deck_instance_count_mismatch")
    return replace(state, draw_pile=tuple(draw_pile), status=status, warnings=tuple(dict.fromkeys(warnings)))



def reconcile_deck_pool_with_snapshot(pool: DeckPoolView, snapshot: BattleSnapshot, *, player: str = "me") -> DeckPoolView:
    """基于当前 snapshot 扣除可确认已离开牌组的卡。"""

    view = snapshot.me if player == "me" else snapshot.opponent
    remaining = Counter({card.card_id: card.count for card in pool.cards})
    warnings: list[str] = list(pool.warnings)

    visible = Counter[int]()
    visible.update(_card_id for _card_id in _visible_card_ids(snapshot, view.hand))
    visible.update(_card_id for _card_id in _visible_card_ids(snapshot, view.field))
    visible.update(_card_id for _card_id in _counted_card_ids(view.played_card_ids))
    visible.update(_card_id for _card_id in _counted_card_ids(view.destroyed_card_ids))

    for card_id, count in visible.items():
        if card_id not in remaining:
            warnings.append(f"deck_pool_unknown_visible_card_id:{card_id}")
            continue
        if count > remaining[card_id]:
            warnings.append(f"deck_pool_visible_count_exceeds_pool:{card_id}")
            remaining[card_id] = 0
            continue
        remaining[card_id] -= count
        if remaining[card_id] < 0:
            remaining[card_id] = 0

    cards = tuple(
        DeckPoolCard(card_id=card_id, count=count)
        for card_id, count in sorted(remaining.items())
        if count > 0
    )
    remaining_count = sum(card.count for card in cards)
    status = pool.status
    if remaining_count != view.deck_count:
        status = "ambiguous" if status == "available" else status
        warnings.append("deck_pool_count_mismatch")

    return replace(
        pool,
        cards=cards,
        remaining_count=remaining_count,
        status=status,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def materialize_draw_card(
    card_id: int,
    snapshot: BattleSnapshot,
    *,
    uid: int,
    resolver: CardResolver | None = None,
    templates: "Mapping[int, Any] | None" = None,
) -> CardView | None:
    """把 card_id 材料化为本地搜索分支里的 CardView。

    如果传入 templates（base_card_id -> UnifiedCardTemplate 映射），
    会从模板的 static_keywords 设置 has_guard / has_storm 等字段。
    未传入时自动从默认模板缓存加载。
    """

    if card_id <= 0:
        return None
    resolver = resolver or CardResolver()
    base = resolver.get_base_by_card_id(card_id)
    if base is None:
        return None
    card_type = _card_type_from_base(base.card_type, base.type_id)
    card = CardView(
        uid=uid,
        card_id=card_id,
        base_card_id=base.base_card_id,
        name=base.name,
        cost=base.cost,
        atk=base.atk,
        life=base.life,
        max_life=base.life,
        card_type=card_type,
        tribes=base.tribes,
    )
    # 应用 static_keywords
    if templates is None:
        templates = _default_templates_by_base_card_id()
    if templates is not None:
        template = templates.get(base.base_card_id)
        if template is not None:
            if template.static_keywords:
                overrides = _keyword_overrides_from_template(template)
                if overrides:
                    card = replace(card, **overrides)
            # 谢幕曲(last_words)是 executable effect 而非 static_keyword，
            # 需从模板 executable_effects 派生 has_last_words，否则物化出的卡
            # 会漏掉谢幕曲标记（护符教 history_same_name_pool 依赖此标记筛选
            # "拥有谢幕曲的护符"）。
            if not card.has_last_words and any(
                e.trigger.event == "last_words" for e in template.executable_effects
            ):
                card = replace(card, has_last_words=True)
            # 从 play_modes 推导 enhance / accelerate / crystallize 费用。
            # 用 PlayModeKind 枚举比较，避免与 local_match_setup 的字符串字面量漂移。
            from decision_engine.cards import PlayModeKind
            enhance_costs = tuple(sorted(
                int(pm.cost.value) for pm in template.play_modes
                if pm.kind == PlayModeKind.ENHANCE and pm.cost and pm.cost.value is not None
            ))
            if enhance_costs:
                card = replace(card, can_enhance=True, enhance_costs=enhance_costs)
            accelerate_costs = tuple(sorted(
                int(pm.cost.value) for pm in template.play_modes
                if pm.kind == PlayModeKind.ACCELERATE and pm.cost and pm.cost.value is not None
            ))
            if accelerate_costs:
                card = replace(card, can_accelerate=True, accelerate_costs=accelerate_costs)
            crystal_costs = tuple(sorted(
                int(pm.cost.value) for pm in template.play_modes
                if pm.kind == PlayModeKind.CRYSTALLIZE and pm.cost and pm.cost.value is not None
            ))
            if crystal_costs:
                card = replace(card, crystal_costs=crystal_costs)
    return card


@lru_cache(maxsize=1)
def _default_templates_by_base_card_id() -> "Mapping[int, Any] | None":
    """延迟加载默认模板库。失败时返回 None 而非阻断。"""
    try:
        from decision_engine.cards.template_loader import load_card_templates_by_base_card_id
        _TEMPLATES_PATH = Path(__file__).resolve().parents[1] / "data" / "public" / "card_templates.json"
        if _TEMPLATES_PATH.exists():
            return load_card_templates_by_base_card_id(_TEMPLATES_PATH)
    except Exception:
        pass
    return None


# keyword 到 CardView 字段的映射（与 LocalPredictor._sync_played_field_card_template_stats 保持一致）
_KEYWORD_TO_FIELD: dict[str, str] = {
    "guard": "has_guard",
    "storm": "has_storm",
    "rush": "has_rush",
    "bane": "has_bane",
    "drain": "has_drain",
    "ambush": "has_ambush",
    "cant_select": "has_cant_select",
    "intimidate": "has_cant_be_attacked",
    "induction": "has_induction",
    "cant_destroy": "has_cant_destroy",
    "ignore_guard_when_attacking": "can_ignore_guard",
    "barrier": "has_temp_shield",
    "temp_shield": "has_temp_shield",
    "last_words": "has_last_words",
    "damage_cut": "has_damage_cut",
}


def _keyword_overrides_from_template(template: Any) -> dict[str, Any]:
    """从模板 static_keywords 提取 CardView 字段覆盖。"""
    overrides: dict[str, Any] = {}
    for kw in template.static_keywords:
        field_name = _KEYWORD_TO_FIELD.get(kw.keyword)
        if field_name is not None:
            overrides[field_name] = True
    return overrides


def next_synthetic_uid(snapshot: BattleSnapshot, branch_index: int = 0) -> int:
    """为本地分支生成稳定的 synthetic uid。"""

    existing = max((uid for uid in snapshot.cards_by_uid if uid > 0), default=0)
    candidate = -(existing + branch_index + 1)
    while candidate in snapshot.cards_by_uid:
        candidate -= 1
    return candidate


def _empty_player():
    from decision_engine.state.snapshot import PlayerView

    return PlayerView(life=0, max_life=0, pp=0, max_pp=0)


def _visible_card_ids(snapshot: BattleSnapshot, uids: tuple[int, ...]) -> tuple[int, ...]:
    cards: list[int] = []
    for uid in uids:
        card = snapshot.cards_by_uid.get(uid)
        if card is None or card.card_id <= 0:
            continue
        cards.append(card.card_id)
    return tuple(cards)


def _counted_card_ids(items: tuple[CardIdCount, ...]) -> tuple[int, ...]:
    expanded: list[int] = []
    for item in items:
        if item.card_id <= 0 or item.count <= 0:
            continue
        expanded.extend([item.card_id] * item.count)
    return tuple(expanded)


def _card_type_from_base(card_type: str, type_id: int) -> CardType:
    normalized = (card_type or "").strip().lower()
    if normalized in {"spell", "spell_evolved"} or type_id == 4:
        return CardType.SPELL
    # 直接使用 type_id 映射（对齐 BaseCardMaster.Type）
    _TYPE_ID_MAP = {1: CardType.FOLLOWER, 2: CardType.AMULET, 3: CardType.COUNTDOWN_AMULET,
                    4: CardType.SPELL, 5: CardType.CREST, 6: CardType.COUNTDOWN_CREST, 7: CardType.FAITH}
    if type_id in _TYPE_ID_MAP:
        return _TYPE_ID_MAP[type_id]
    if "amulet" in normalized:
        return CardType.AMULET
    if "crest" in normalized:
        return CardType.CREST
    if "boon" in normalized:
        return CardType.AMULET
    return CardType.FOLLOWER


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
