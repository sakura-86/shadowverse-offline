# -*- coding: utf-8 -*-
"""起手留牌策略。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

TANTELLA = 10834110
DIMENSION_SHIFT = 10134310
STEEL_REST = 10833310
HOMEWORK = 10133310
ARO_TRANSFORM = 10531310
REUNION = 10832310
VERDANT_GRACE = 10831110

AETHER = 10264110
WILBERT = 10264120
SOFINA = 10564110
JEANNE = 10164120
GALLEON = 10464110
MAVEN = 10364120

DRAW2_RESOURCES = frozenset({STEEL_REST, HOMEWORK, ARO_TRANSFORM})
CONSUMABLE_SPELLBOOST = frozenset({REUNION, VERDANT_GRACE})
GUARD_HAVEN_CORE = frozenset({AETHER})
GUARD_HAVEN_ENGINE = frozenset({WILBERT, SOFINA, JEANNE, GALLEON, MAVEN})
GUARD_HAVEN_KEEP_LOW = frozenset({GALLEON, SOFINA})
GUARD_HAVEN_KEEP_MID = frozenset({WILBERT, MAVEN})

_DRAW2_PRIORITY = {
    STEEL_REST: 92.0,
    HOMEWORK: 88.0,
    ARO_TRANSFORM: 78.0,
}


@dataclass(frozen=True)
class MulliganCard:
    unique_id: int
    card_id: int
    base_card_id: int | None = None
    name: str = ""
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class MulliganConfig:
    aggressive_find_terminal: bool = False
    full_redraw_without_terminal: bool = False
    keep_one_draw2_resource: bool = True
    keep_dimension_with_terminal_or_resource: bool = True
    fallback_keep_unknown: bool = False


@dataclass(frozen=True)
class MulliganDecision:
    cards_to_change: list[int]
    cards_to_keep: list[int]
    card_scores: dict[int, float]
    reasons: dict[int, list[str]] = field(default_factory=dict)
    strategy_id: str = "spellboost_rune"


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _card_ids(card: MulliganCard) -> frozenset[int]:
    values = {card.card_id}
    if card.base_card_id is not None:
        values.add(card.base_card_id)
    return frozenset(values)


def _matches(card: MulliganCard, card_id: int) -> bool:
    return card_id in _card_ids(card)


def _matches_any(card: MulliganCard, card_ids: frozenset[int]) -> bool:
    return bool(_card_ids(card) & card_ids)


def _identity(card: MulliganCard) -> int:
    return card.base_card_id or card.card_id


def extract_mulligan_hand(state_update: dict[str, Any]) -> list[MulliganCard]:
    if not isinstance(state_update, dict) or state_update.get("type") != "state_update":
        return []
    data = state_update.get("data")
    if not isinstance(data, dict):
        return []
    players = data.get("players")
    if not isinstance(players, list) or not players or not isinstance(players[0], dict):
        return []
    hand_cards = players[0].get("hand_cards")
    if not isinstance(hand_cards, list):
        return []

    result: list[MulliganCard] = []
    for raw_card in hand_cards:
        if not isinstance(raw_card, dict):
            continue
        unique_id = _to_int(raw_card.get("unique_id", raw_card.get("uid")))
        card_id = _to_int(raw_card.get("card_id"))
        if unique_id is None or card_id is None:
            continue
        base_card_id = _to_int(raw_card.get("base_card_id"))
        name = raw_card.get("name")
        result.append(MulliganCard(
            unique_id=unique_id,
            card_id=card_id,
            base_card_id=base_card_id,
            name=name if isinstance(name, str) else "",
            raw=raw_card,
        ))
    return result


class SpellboostRuneMulliganStrategy:
    strategy_id = "spellboost_rune"

    def __init__(self, config: MulliganConfig | None = None):
        self.config = config or MulliganConfig()

    def decide(self, hand: Sequence[MulliganCard]) -> MulliganDecision:
        hand = list(hand)
        has_tantella = any(_matches(card, TANTELLA) for card in hand)
        draw2_cards = [card for card in hand if _matches_any(card, DRAW2_RESOURCES)]
        selected_draw2 = self._select_draw2(draw2_cards, has_tantella)
        has_resource_chain = has_tantella or selected_draw2 is not None
        selected_consumable = self._select_consumable(hand, has_tantella, selected_draw2 is not None)

        keep: set[int] = set()
        scores: dict[int, float] = {}
        reasons: dict[int, list[str]] = {}

        for card in hand:
            score, card_reasons = self._score_card(
                card,
                has_tantella=has_tantella,
                selected_draw2=selected_draw2,
                has_resource_chain=has_resource_chain,
                selected_consumable=selected_consumable,
            )
            scores[card.unique_id] = score
            reasons[card.unique_id] = card_reasons

            if self._should_keep(
                card,
                has_tantella=has_tantella,
                selected_draw2=selected_draw2,
                has_resource_chain=has_resource_chain,
                selected_consumable=selected_consumable,
            ):
                keep.add(card.unique_id)

        if self.config.full_redraw_without_terminal and not has_tantella:
            keep.clear()
            for card in hand:
                scores[card.unique_id] = min(scores[card.unique_id], 25.0)
                reasons[card.unique_id].append("full_redraw_without_terminal")

        if self.config.fallback_keep_unknown:
            for card in hand:
                if not self._is_known(card):
                    keep.add(card.unique_id)
                    reasons[card.unique_id].append("fallback_keep_unknown")

        cards_to_keep = [card.unique_id for card in hand if card.unique_id in keep]
        cards_to_change = [card.unique_id for card in hand if card.unique_id not in keep]
        return MulliganDecision(
            cards_to_change=cards_to_change,
            cards_to_keep=cards_to_keep,
            card_scores=scores,
            reasons=reasons,
            strategy_id=self.strategy_id,
        )

    def _select_draw2(self, draw2_cards: Sequence[MulliganCard], has_tantella: bool) -> MulliganCard | None:
        if not draw2_cards or not self.config.keep_one_draw2_resource:
            return None
        if self.config.aggressive_find_terminal and not has_tantella:
            return max(draw2_cards, key=lambda card: _DRAW2_PRIORITY.get(_identity(card), 0.0))
        return max(draw2_cards, key=lambda card: _DRAW2_PRIORITY.get(_identity(card), 0.0))

    def _select_consumable(
        self,
        hand: Sequence[MulliganCard],
        has_tantella: bool,
        has_draw2: bool,
    ) -> MulliganCard | None:
        if not has_tantella:
            return None
        candidates = [card for card in hand if _matches_any(card, CONSUMABLE_SPELLBOOST)]
        if not candidates:
            return None
        if has_draw2:
            return max(candidates, key=lambda card: 1 if _matches(card, VERDANT_GRACE) else 0)
        return max(candidates, key=lambda card: 1 if _matches(card, VERDANT_GRACE) else 0)

    def _score_card(
        self,
        card: MulliganCard,
        *,
        has_tantella: bool,
        selected_draw2: MulliganCard | None,
        has_resource_chain: bool,
        selected_consumable: MulliganCard | None,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        if _matches(card, TANTELLA):
            reasons.append("main_terminal_tantella")
            return 100.0, reasons

        if selected_draw2 is not None and card.unique_id == selected_draw2.unique_id:
            reasons.append("keep_one_draw2_resource")
            return _DRAW2_PRIORITY.get(_identity(card), 78.0), reasons

        if _matches_any(card, DRAW2_RESOURCES):
            reasons.append("extra_draw2_resource")
            return 58.0 if has_tantella else 45.0, reasons

        if _matches(card, DIMENSION_SHIFT):
            if has_tantella:
                reasons.append("dimension_otk_extension_with_tantella")
                return 76.0, reasons
            if has_resource_chain and self.config.keep_dimension_with_terminal_or_resource:
                reasons.append("dimension_kept_with_resource_chain")
                return 62.0, reasons
            reasons.append("dimension_without_terminal_or_resource")
            return 40.0, reasons

        if selected_consumable is not None and card.unique_id == selected_consumable.unique_id:
            reasons.append("low_cost_spellboost_for_tantella")
            return 60.0 if _matches(card, REUNION) else 58.0, reasons

        if _matches_any(card, CONSUMABLE_SPELLBOOST):
            if has_tantella or has_resource_chain:
                reasons.append("extra_consumable_spellboost")
                return 45.0, reasons
            reasons.append("consumable_spellboost_without_core")
            return 35.0, reasons

        reasons.append("non_core_or_high_cost")
        return 30.0, reasons

    def _should_keep(
        self,
        card: MulliganCard,
        *,
        has_tantella: bool,
        selected_draw2: MulliganCard | None,
        has_resource_chain: bool,
        selected_consumable: MulliganCard | None,
    ) -> bool:
        if _matches(card, TANTELLA):
            return True
        if selected_draw2 is not None and card.unique_id == selected_draw2.unique_id:
            return not (self.config.full_redraw_without_terminal and not has_tantella)
        if _matches(card, DIMENSION_SHIFT):
            if has_tantella:
                return True
            return has_resource_chain and self.config.keep_dimension_with_terminal_or_resource and not self.config.aggressive_find_terminal
        if selected_consumable is not None and card.unique_id == selected_consumable.unique_id:
            return has_tantella
        return False

    def _is_known(self, card: MulliganCard) -> bool:
        return (
            _matches(card, TANTELLA)
            or _matches(card, DIMENSION_SHIFT)
            or _matches_any(card, DRAW2_RESOURCES)
            or _matches_any(card, CONSUMABLE_SPELLBOOST)
        )


class GuardHavenMulliganStrategy:
    strategy_id = "guard_haven"

    def __init__(self, config: MulliganConfig | None = None):
        self.config = config or MulliganConfig()

    def decide(self, hand: Sequence[MulliganCard]) -> MulliganDecision:
        hand = list(hand)
        has_aether = any(_matches_any(card, GUARD_HAVEN_CORE) for card in hand)
        has_wilbert = any(_matches(card, WILBERT) for card in hand)
        early_curve_cards = [card for card in hand if _matches_any(card, GUARD_HAVEN_KEEP_LOW)]
        keep: set[int] = set()
        scores: dict[int, float] = {}
        reasons: dict[int, list[str]] = {}

        for card in hand:
            score, card_reasons = self._score_card(card, has_aether=has_aether, has_wilbert=has_wilbert, early_curve_count=len(early_curve_cards))
            scores[card.unique_id] = score
            reasons[card.unique_id] = card_reasons
            if self._should_keep(card, has_aether=has_aether, has_wilbert=has_wilbert, early_curve_count=len(early_curve_cards)):
                keep.add(card.unique_id)

        cards_to_keep = [card.unique_id for card in hand if card.unique_id in keep]
        cards_to_change = [card.unique_id for card in hand if card.unique_id not in keep]
        return MulliganDecision(
            cards_to_change=cards_to_change,
            cards_to_keep=cards_to_keep,
            card_scores=scores,
            reasons=reasons,
            strategy_id=self.strategy_id,
        )

    def _score_card(self, card: MulliganCard, *, has_aether: bool, has_wilbert: bool, early_curve_count: int) -> tuple[float, list[str]]:
        reasons: list[str] = []
        if _matches_any(card, GUARD_HAVEN_CORE):
            reasons.append("guard_haven_core_aether")
            return 100.0, reasons
        if _matches(card, WILBERT):
            if has_aether or early_curve_count >= 2:
                reasons.append("guard_haven_wilbert_with_support")
                return 88.0, reasons
            reasons.append("guard_haven_wilbert_without_support")
            return 56.0, reasons
        if _matches_any(card, GUARD_HAVEN_KEEP_LOW):
            reasons.append("guard_haven_early_curve")
            return 76.0 if has_aether else 68.0, reasons
        if _matches_any(card, GUARD_HAVEN_KEEP_MID):
            if has_aether:
                reasons.append("guard_haven_midgame_pressure_with_aether")
                return 62.0, reasons
            reasons.append("guard_haven_midgame_pressure")
            return 48.0, reasons
        if self._is_heavy(card):
            reasons.append("guard_haven_heavy_late_keep_out")
            return 22.0, reasons
        reasons.append("guard_haven_generic_filler")
        return 40.0 if has_aether else 36.0, reasons

    def _should_keep(self, card: MulliganCard, *, has_aether: bool, has_wilbert: bool, early_curve_count: int) -> bool:
        if _matches_any(card, GUARD_HAVEN_CORE):
            return True
        if _matches(card, WILBERT):
            return has_aether or early_curve_count >= 2
        if _matches_any(card, GUARD_HAVEN_KEEP_LOW):
            return has_aether or early_curve_count >= 1
        if _matches_any(card, GUARD_HAVEN_KEEP_MID):
            return has_aether and early_curve_count >= 1
        return False

    def _is_heavy(self, card: MulliganCard) -> bool:
        return _matches(card, JEANNE) or _matches(card, ARO_TRANSFORM) or _matches(card, DIMENSION_SHIFT) or _matches(card, REUNION)


def build_mulligan_command(
    state_update: dict[str, Any],
    strategy: SpellboostRuneMulliganStrategy | GuardHavenMulliganStrategy | None = None,
) -> tuple[dict[str, Any], MulliganDecision | None]:
    hand = extract_mulligan_hand(state_update)
    if not hand:
        return {"action": "mulligan", "cards_to_change": [], "force_execute": True}, None
    strategy = strategy or SpellboostRuneMulliganStrategy()
    decision = strategy.decide(hand)
    return {
        "action": "mulligan",
        "cards_to_change": decision.cards_to_change,
        "force_execute": True,
    }, decision
