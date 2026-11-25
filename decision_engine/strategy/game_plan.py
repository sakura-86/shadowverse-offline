# -*- coding: utf-8 -*-
"""体系主线与战术意图。

GamePlan 只描述当前对局主线、资源保留、阈值和顺序收益，
不执行卡牌效果，不判断协议合法性。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from decision_engine.cards.effect_facts import EffectFactKind, EffectFactTiming, is_fact_applicable

if TYPE_CHECKING:
    from decision_engine.cards.resolver import CardResolver
    from decision_engine.state.snapshot import BattleSnapshot, CardView
    from decision_engine.strategy.match_context import MatchContext


class OrderingRule(Enum):
    DRAW_BEFORE_HAND_BOOST = "draw_before_hand_boost"
    REFUND_BEFORE_COST_REDUCTION_PAYOFF = "refund_before_cost_reduction_payoff"
    PUSH_THRESHOLD_BEFORE_PAYOFF = "push_threshold_before_payoff"
    CLEAR_GUARD_BEFORE_TERMINAL = "clear_guard_before_terminal"


class TacticalWindowKind(Enum):
    EXTRA_PP_UNLOCKS_KEY_CARD = "extra_pp_unlocks_key_card"
    SUPER_EVOLVE_TERMINAL_WINDOW = "super_evolve_terminal_window"
    THRESHOLD_PUSH_WINDOW = "threshold_push_window"
    REFUND_DRAW_CHAIN_WINDOW = "refund_draw_chain_window"
    HAND_PRESSURE_WINDOW = "hand_pressure_window"
    SURVIVAL_WINDOW = "survival_window"
    PACKAGE_WINDOW = "package_window"


class HandPressureLevel(Enum):
    LOW = "low"
    SAFE_DRAW = "safe_draw"
    NEAR_FULL = "near_full"
    FULL = "full"


@dataclass(frozen=True)
class HandPressureSignal:
    level: HandPressureLevel
    hand_size: int
    draw_capacity: int = 0
    overflow_on_draw2: int = 0
    reason: str = ""

@dataclass(frozen=True)
class SurvivalSignal:
    opponent_attack: int = 0
    max_attacker: int = 0
    opponent_width: int = 0
    crisis: bool = False
    reason: str = ""


@dataclass(frozen=True)
class PackageHint:
    name: str
    card_uid: int = 0
    priority: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class ThresholdObjective:
    resource: str
    current: int
    target: int
    gap: int
    card_uid: int = 0
    priority: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class ResourceReservation:
    resource: str
    reserved_for: str
    opportunity_cost: float
    reason: str = ""


@dataclass(frozen=True)
class TacticalWindow:
    kind: TacticalWindowKind
    card_uid: int = 0
    value: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class TacticalIntent:
    archetype_id: str = ""
    main_plan: str = ""
    terminal_uids: tuple[int, ...] = ()
    missing_terminal: str = ""
    fallback_focus: str = ""
    threshold_objectives: tuple[ThresholdObjective, ...] = ()
    reserved_resources: tuple[ResourceReservation, ...] = ()
    ordering_rules: tuple[OrderingRule, ...] = ()
    tactical_windows: tuple[TacticalWindow, ...] = ()
    hand_pressure: HandPressureSignal | None = None
    survival: SurvivalSignal | None = None
    package_hints: tuple[PackageHint, ...] = ()
    estimated_burst: dict[str, float] | None = None
    reasons: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return bool(self.main_plan)

    def best_threshold_gap(self, resource: str) -> int | None:
        gaps = [item.gap for item in self.threshold_objectives if item.resource == resource]
        return min(gaps) if gaps else None

    def has_ordering_rule(self, rule: OrderingRule) -> bool:
        return rule in self.ordering_rules

    def reservation_cost(self, resource: str) -> float:
        return sum(item.opportunity_cost for item in self.reserved_resources if item.resource == resource)


class GamePlanAnalyzer:
    def analyze(self, snapshot: "BattleSnapshot") -> TacticalIntent:
        raise NotImplementedError


class NoopGamePlanAnalyzer(GamePlanAnalyzer):
    def analyze(self, snapshot: "BattleSnapshot") -> TacticalIntent:
        return TacticalIntent()


class SpellboostGamePlanAnalyzer(GamePlanAnalyzer):
    def __init__(self, *, context: "MatchContext", card_resolver: "CardResolver | None" = None):
        self._context = context
        self._resolver = card_resolver

    def analyze(self, snapshot: "BattleSnapshot") -> TacticalIntent:
        archetype = self._context.my_profile.archetype
        if archetype.archetype_id != "spellboost_rune" and "spellboost" not in archetype.effective_resources:
            return TacticalIntent()

        terminal = self._best_terminal(snapshot)
        ordering_rules = (
            OrderingRule.DRAW_BEFORE_HAND_BOOST,
            OrderingRule.REFUND_BEFORE_COST_REDUCTION_PAYOFF,
            OrderingRule.PUSH_THRESHOLD_BEFORE_PAYOFF,
            OrderingRule.CLEAR_GUARD_BEFORE_TERMINAL,
        )
        hand_pressure = self._hand_pressure(snapshot)
        survival = self._survival_signal(snapshot)
        package_hints = tuple(self._package_hints(snapshot))
        windows = self._tactical_windows(snapshot)
        if terminal is None:
            reasons = [
                "spellboost_main_plan:tantella_20_otk",
                "missing_terminal:tantella",
                "fallback_focus:draw/find_terminal",
            ]
            reasons.extend(window.reason for window in windows if window.reason)
            return TacticalIntent(
                archetype_id=archetype.archetype_id,
                main_plan="tantella_20_otk",
                missing_terminal="tantella",
                fallback_focus="draw/find_terminal",
                threshold_objectives=tuple(self._payoff_thresholds(snapshot)),
                ordering_rules=ordering_rules,
                tactical_windows=tuple(windows),
                hand_pressure=hand_pressure,
                survival=survival,
                package_hints=package_hints,
                estimated_burst={"tantella_with_sep": 16.0, "tantella_without_sep": 10.0},
                reasons=tuple(reasons),
            )

        terminal_x = max(0, terminal.variable_x, terminal.spell_boost_count)
        terminal_threshold = max(20, archetype.resource_thresholds.get("spellboost_for_tantella_20", 20))
        terminal_gap = max(0, terminal_threshold - terminal_x)

        thresholds: list[ThresholdObjective] = [
            ThresholdObjective(
                resource="spellboost",
                current=terminal_x,
                target=terminal_threshold,
                gap=terminal_gap,
                card_uid=terminal.uid,
                priority=12.0 if terminal_gap <= 3 else 8.0 if terminal_gap <= 8 else 4.0,
                reason="tantella_terminal",
            )
        ]
        thresholds.extend(self._payoff_thresholds(snapshot))

        reservations: list[ResourceReservation] = []
        if terminal_gap <= 15 and snapshot.me.sep > 0:
            reservations.append(
                ResourceReservation(
                    resource="sep",
                    reserved_for="tantella_terminal_burst",
                    opportunity_cost=6.0 if snapshot.me.sep == 1 else 4.0,
                    reason="tantella_burst_with_sep_about_16_without_sep_about_10",
                )
            )
        if snapshot.me.extra_pp is not None and not snapshot.me.extra_pp.is_used_extra_pp_this_turn:
            reservations.append(
                ResourceReservation(
                    resource="extra_pp",
                    reserved_for="key_spellboost_window",
                    opportunity_cost=3.0,
                    reason="save_extra_pp_unless_unlocks_key_window",
                )
            )

        reasons = [
            "spellboost_main_plan:tantella_20_otk",
            f"terminal_uid:{terminal.uid}",
            f"terminal_x:{terminal_x}",
            f"terminal_gap:{terminal_gap}",
        ]
        if terminal_gap == 0:
            reasons.append("terminal_ready_before_dimension_shift")
        reasons.extend(window.reason for window in windows if window.reason)

        return TacticalIntent(
            archetype_id=archetype.archetype_id,
            main_plan="tantella_20_otk",
            terminal_uids=(terminal.uid,),
            threshold_objectives=tuple(thresholds),
            reserved_resources=tuple(reservations),
            ordering_rules=ordering_rules,
            tactical_windows=tuple(windows),
            hand_pressure=hand_pressure,
            survival=survival,
            package_hints=package_hints,
            estimated_burst={"tantella_with_sep": 16.0, "tantella_without_sep": 10.0},
            reasons=tuple(reasons),
        )

    def _hand_pressure(self, snapshot: "BattleSnapshot") -> HandPressureSignal:
        hand_size = len(snapshot.me.hand)
        capacity = max(0, 9 - hand_size)
        capacity_after_play = max(0, 9 - max(0, hand_size - 1))
        overflow = max(0, 2 - capacity_after_play)
        if hand_size >= 9:
            return HandPressureSignal(HandPressureLevel.FULL, hand_size, capacity, overflow, "hand_full_natural_draw_risk")
        if hand_size >= 8:
            return HandPressureSignal(HandPressureLevel.NEAR_FULL, hand_size, capacity, overflow, "hand_near_full_draw_capacity_limited")
        if hand_size <= 6:
            return HandPressureSignal(HandPressureLevel.SAFE_DRAW, hand_size, capacity, 0, "safe_draw_capacity")
        return HandPressureSignal(HandPressureLevel.LOW, hand_size, capacity, 0, "moderate_hand_pressure")

    def _survival_signal(self, snapshot: "BattleSnapshot") -> SurvivalSignal:
        followers = [snapshot.cards_by_uid[uid] for uid in snapshot.opponent.field if uid in snapshot.cards_by_uid]
        attacks = [max(0, card.atk) for card in followers]
        total_attack = sum(attacks)
        max_attacker = max(attacks, default=0)
        width = len(followers)
        crisis = snapshot.me.life <= total_attack + 2 or (snapshot.me.life <= 10 and width > 0) or (snapshot.me.life <= 14 and width >= 4)
        reason = "survival_crisis_visible_attack" if crisis else "stable_life_total"
        return SurvivalSignal(
            opponent_attack=total_attack,
            max_attacker=max_attacker,
            opponent_width=width,
            crisis=crisis,
            reason=reason,
        )

    def _package_hints(self, snapshot: "BattleSnapshot") -> list[PackageHint]:
        hints: list[PackageHint] = []
        for uid in snapshot.me.hand:
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            roles = self._card_roles(card)
            current = max(0, card.variable_x, card.spell_boost_count)
            hand_spellboost_count = self._hand_spellboost_count(card)
            if "finisher" in roles:
                if current >= 20:
                    hints.append(PackageHint("tantella_x20_terminal", uid, 16.0, "tantella_generates_blue_four_and_grand_return"))
                elif current >= 10:
                    hints.append(PackageHint("tantella_x10_terminal", uid, 11.0, "tantella_generates_blue_four"))
                elif current >= 8:
                    hints.append(PackageHint("tantella_near_terminal", uid, 8.0, "preserve_and_push_tantella_threshold"))
            if "combo_enabler" in roles:
                hints.append(PackageHint("dimension_shift_timing", uid, 9.0, "evaluate_hand_reset_and_pp_reset_timing"))
            if "board_swing" in roles:
                priority = 9.0
                reason = "board_swing_package"
                if hand_spellboost_count >= 3:
                    priority = 13.0
                    reason = "anne_grea_body_token_hand_spellboost_package"
                hints.append(PackageHint("board_swing", uid, priority, reason))
            if "removal" in roles and snapshot.opponent.field:
                hints.append(PackageHint("removal_window", uid, 8.0, "visible_opponent_field_removal"))
            if self._is_steel_rest(card):
                if current >= 5:
                    hints.append(PackageHint("steel_rest_threshold_resource", uid, 10.0, "steel_rest_x5_draw_heal_pp_refund"))
                else:
                    hints.append(PackageHint("steel_rest_low_x_hold", uid, 8.0, "preserve_steel_rest_until_x5"))
        return hints

    @staticmethod
    def _is_steel_rest(card: "CardView") -> bool:
        return (card.base_card_id or card.card_id) == 10833310 or card.card_id == 10833310

    def _card_roles(self, card: "CardView") -> frozenset[str]:
        if card.card_id == 0:
            return frozenset()
        card_keys = {card.card_id, card.base_card_id}
        if self._resolver is not None:
            base = self._resolver.get_base_by_card_id(card.card_id)
            if base is not None:
                card_keys.add(base.base_card_id)
        roles = {
            role
            for role, card_ids in self._context.my_profile.archetype.card_roles.items()
            if any(card_key in card_ids for card_key in card_keys if card_key)
        }
        return frozenset(roles)

    def _best_terminal(self, snapshot: "BattleSnapshot") -> "CardView | None":
        terminals = []
        for uid in snapshot.me.hand:
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            if "finisher" in self._card_roles(card):
                terminals.append(card)
        if not terminals:
            return None

        def terminal_key(card: "CardView") -> tuple[int, int]:
            return (1 if card.has_storm else 0, max(0, card.variable_x, card.spell_boost_count))

        return max(terminals, key=terminal_key)

    def _facts(self, card: "CardView"):
        if self._resolver is None or card.card_id == 0:
            return ()
        return self._resolver.get_effect_facts(card.card_id)

    def _has_draw(self, card: "CardView") -> bool:
        return any(f.kind is EffectFactKind.DRAW and is_fact_applicable(f, card) for f in self._facts(card))

    def _has_pp_refund(self, card: "CardView") -> bool:
        return any(f.kind is EffectFactKind.PP_CHANGE and (f.amount or 0) > 0 and is_fact_applicable(f, card) for f in self._facts(card))

    def _hand_spellboost_count(self, card: "CardView") -> int:
        count = 0
        for fact in self._facts(card):
            if fact.timing not in {EffectFactTiming.ON_PLAY, EffectFactTiming.FANFARE, EffectFactTiming.ACTIVATION}:
                continue
            text = " ".join(part for part in (fact.raw_text, fact.unsupported_reason) if part)
            if "hand_spellboost" in text or "手牌" in text and "魔力增幅" in text:
                count = max(count, fact.amount or 1)
        return count

    def _has_hand_spellboost(self, card: "CardView") -> bool:
        return self._hand_spellboost_count(card) > 0

    def _payoff_thresholds(self, snapshot: "BattleSnapshot") -> list[ThresholdObjective]:
        objectives: list[ThresholdObjective] = []
        for uid in snapshot.me.hand:
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            current = max(0, card.variable_x, card.spell_boost_count)
            for target in (5, 10, 20):
                gap = target - current
                if 0 < gap <= 1:
                    objectives.append(
                        ThresholdObjective(
                            resource="spellboost",
                            current=current,
                            target=target,
                            gap=gap,
                            card_uid=card.uid,
                            priority=8.0 if target == 5 else 10.0 if target == 10 else 14.0,
                            reason=f"near_spellboost_threshold_{target}",
                        )
                    )
        return objectives

    def _tactical_windows(self, snapshot: "BattleSnapshot") -> list[TacticalWindow]:
        windows: list[TacticalWindow] = []
        legal = snapshot.legal
        for uid in snapshot.me.hand:
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            hand_spellboost_count = self._hand_spellboost_count(card)
            if self._has_draw(card) and hand_spellboost_count > 0:
                windows.append(
                    TacticalWindow(
                        kind=TacticalWindowKind.REFUND_DRAW_CHAIN_WINDOW if self._has_pp_refund(card) else TacticalWindowKind.THRESHOLD_PUSH_WINDOW,
                        card_uid=uid,
                        value=7.0 if self._has_pp_refund(card) else 4.0,
                        reason="draw_refund_hand_spellboost_chain" if self._has_pp_refund(card) else "draw_before_hand_spellboost",
                    )
                )
            if hand_spellboost_count >= 3:
                windows.append(
                    TacticalWindow(
                        kind=TacticalWindowKind.THRESHOLD_PUSH_WINDOW,
                        card_uid=uid,
                        value=10.0,
                        reason="strong_hand_spellboost_threshold_push",
                    )
                )
            if max(0, card.variable_x, card.spell_boost_count) >= 20 and "finisher" in self._card_roles(card):
                windows.append(
                    TacticalWindow(
                        kind=TacticalWindowKind.THRESHOLD_PUSH_WINDOW,
                        card_uid=uid,
                        value=12.0,
                        reason="terminal_ready_before_dimension_shift",
                    )
                )
            if max(0, card.variable_x, card.spell_boost_count) in {4, 9, 19}:
                windows.append(
                    TacticalWindow(
                        kind=TacticalWindowKind.THRESHOLD_PUSH_WINDOW,
                        card_uid=uid,
                        value=8.0,
                        reason="push_threshold_before_payoff",
                    )
                )
        if legal is not None:
            unlocked = legal.can_play_with_ex_pp - legal.can_play
            for uid in unlocked:
                card = snapshot.cards_by_uid.get(uid)
                if card is None:
                    continue
                roles = self._card_roles(card)
                if roles & {"engine", "board_swing", "spellboost_enabler", "finisher"}:
                    windows.append(
                        TacticalWindow(
                            kind=TacticalWindowKind.EXTRA_PP_UNLOCKS_KEY_CARD,
                            card_uid=uid,
                            value=9.0,
                            reason="extra_pp_unlocks_spellboost_key_card",
                        )
                    )
        return windows


class GamePlanAnalyzerFactory:
    @staticmethod
    def for_context(
        *,
        context: "MatchContext | None",
        card_resolver: "CardResolver | None" = None,
    ) -> GamePlanAnalyzer:
        if context is None:
            return NoopGamePlanAnalyzer()
        archetype = context.my_profile.archetype
        if archetype.archetype_id == "spellboost_rune" or "spellboost" in archetype.effective_resources:
            return SpellboostGamePlanAnalyzer(context=context, card_resolver=card_resolver)
        return NoopGamePlanAnalyzer()
