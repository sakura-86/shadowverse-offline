# -*- coding: utf-8 -*-
"""跨回合本地 transition 抽象。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from decision_engine.cards import CardResolver, MasterDataProvider
from decision_engine.simulation.deck_pool import DeckPoolView, materialize_draw_card, next_synthetic_uid
from decision_engine.simulation.opponent_deck_pool import OpponentDeckPoolBuilder, OpponentDeckPoolConfig
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot
from decision_engine.strategy.opponent_deck_identifier import OpponentDeckIdentifier, OpponentDeckIdentifierConfig

from .opponent_policy import OpponentPolicyConfig, OpponentTurnPolicy
from .rule_kernel import CoreRuleKernel, RuleKernelError


@dataclass(frozen=True)
class CrossTurnTransitionConfig:
    enabled: bool = False
    max_transitions_per_search_path: int = 1
    max_ally_turns: int | None = None
    skip_opponent_actions: bool = True
    allow_hidden_draws: bool = True
    hidden_draw_branching_enabled: bool = False
    hidden_draw_top_k: int = 0
    include_hidden_draw_residual: bool = True
    require_deck_pool_when_enabled: bool = True
    opponent_modeling_enabled: bool = False
    opponent_class_en: str | None = None
    opponent_candidate_top_k: int = 8
    opponent_action_beam_width: int = 4
    opponent_action_max_steps: int = 1
    include_opponent_no_action_branch: bool = True
    require_opponent_class_when_enabled: bool = True
    require_opponent_pool_when_enabled: bool = True
    opponent_deck_identification_enabled: bool = False
    opponent_deck_hypothesis_top_n: int = 5
    require_opponent_deck_hypothesis_when_enabled: bool = False


@dataclass(frozen=True)
class CrossTurnTransitionResult:
    snapshot: BattleSnapshot
    certainty: PredictionCertainty
    description: str
    assumptions: tuple[str, ...] = ()
    unresolved_effects: tuple[str, ...] = ()
    applied_steps: tuple[str, ...] = ()


class CrossTurnTransition:
    def __init__(
        self,
        *,
        kernel: CoreRuleKernel | None = None,
        card_resolver: CardResolver | None = None,
        opponent_pool_builder: OpponentDeckPoolBuilder | None = None,
        opponent_policy: OpponentTurnPolicy | None = None,
        opponent_deck_identifier: OpponentDeckIdentifier | None = None,
    ):
        self.kernel = kernel or CoreRuleKernel()
        if card_resolver is None:
            database_path = Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"
            card_resolver = CardResolver(MasterDataProvider.from_json(database_path))
        self.card_resolver = card_resolver
        self.opponent_pool_builder = opponent_pool_builder or OpponentDeckPoolBuilder()
        self.opponent_policy = opponent_policy or OpponentTurnPolicy(resolver=card_resolver)
        self.opponent_deck_identifier = opponent_deck_identifier or OpponentDeckIdentifier()

    def apply_after_ally_turn_end(
        self,
        snapshot: BattleSnapshot,
        config: CrossTurnTransitionConfig | None = None,
    ) -> CrossTurnTransitionResult:
        return self.apply_after_ally_turn_end_branches(snapshot, config=config)[0]

    def apply_after_ally_turn_end_branches(
        self,
        snapshot: BattleSnapshot,
        config: CrossTurnTransitionConfig | None = None,
    ) -> tuple[CrossTurnTransitionResult, ...]:
        config = config or CrossTurnTransitionConfig()
        if not snapshot.is_ally_turn:
            return (self._error(snapshot, "cross turn transition requires ally turn snapshot"),)
        if not snapshot.is_turn_ended:
            return (self._error(snapshot, "cross turn transition requires ended ally turn"),)

        base_assumptions = ("cross_turn_transition_enabled", "hidden_draws_without_card_identity", "no_random_or_generated_cards")
        base_unresolved = ("opponent_turn_start_effects", "opponent_turn_end_effects", "ally_turn_start_effects", "hidden_draw_card_identity")
        applied_steps: list[str] = []

        try:
            opponent_start = self.kernel.start_turn(snapshot, "opponent", drawn_card=None)
            current = opponent_start.snapshot
            applied_steps.append("opponent_start_turn")
            if self._has_battle_result(current):
                return (self._partial(current, "cross turn transition stopped after opponent start", base_assumptions, base_unresolved, tuple(applied_steps)),)
        except RuleKernelError as exc:
            return (
                CrossTurnTransitionResult(
                    snapshot=snapshot,
                    certainty=PredictionCertainty.PREDICT_ERROR,
                    description=str(exc),
                    assumptions=base_assumptions,
                    unresolved_effects=base_unresolved,
                    applied_steps=tuple(applied_steps),
                ),
            )

        opponent_branches = self._apply_opponent_turn_branches(current, config, base_assumptions, base_unresolved, tuple(applied_steps))
        results: list[CrossTurnTransitionResult] = []
        for branch in opponent_branches:
            if branch.certainty == PredictionCertainty.PREDICT_ERROR:
                results.append(branch)
                continue
            results.extend(self._advance_to_ally_turn(branch.snapshot, snapshot.me.deck_pool, config, branch.assumptions, branch.unresolved_effects, branch.applied_steps))
        return tuple(results)

    def _apply_opponent_turn_branches(
        self,
        current: BattleSnapshot,
        config: CrossTurnTransitionConfig,
        assumptions: tuple[str, ...],
        unresolved: tuple[str, ...],
        applied_steps: tuple[str, ...],
    ) -> tuple[CrossTurnTransitionResult, ...]:
        if not config.opponent_modeling_enabled:
            skipped_assumptions = assumptions + ("opponent_actions_skipped",)
            skipped_unresolved = unresolved + ("opponent_turn_actions",)
            try:
                opponent_end = self.kernel.end_turn(current, "opponent")
            except RuleKernelError as exc:
                return (
                    CrossTurnTransitionResult(
                        snapshot=current,
                        certainty=PredictionCertainty.PREDICT_ERROR,
                        description=str(exc),
                        assumptions=skipped_assumptions,
                        unresolved_effects=skipped_unresolved,
                        applied_steps=applied_steps,
                    ),
                )
            steps = applied_steps + ("opponent_end_turn",)
            if self._has_battle_result(opponent_end.snapshot):
                return (self._partial(opponent_end.snapshot, "cross turn transition stopped after opponent end", skipped_assumptions, skipped_unresolved, steps),)
            return (self._partial(opponent_end.snapshot, "opponent turn advanced without modeled actions", skipped_assumptions, skipped_unresolved, steps),)

        class_en = (config.opponent_class_en or "").strip()
        if not class_en and config.require_opponent_class_when_enabled:
            return (
                CrossTurnTransitionResult(
                    snapshot=current,
                    certainty=PredictionCertainty.PREDICT_ERROR,
                    description="opponent_class_required_for_modeling",
                    assumptions=assumptions + ("opponent_modeling_enabled",),
                    unresolved_effects=unresolved + ("opponent_class_unknown",),
                    applied_steps=applied_steps,
                ),
            )

        build = self._build_opponent_pool(current, class_en, config)
        if build.deck_pool is None:
            certainty = PredictionCertainty.PREDICT_ERROR if config.require_opponent_pool_when_enabled else build.certainty
            return (
                CrossTurnTransitionResult(
                    snapshot=current,
                    certainty=certainty,
                    description="; ".join(build.assumptions),
                    assumptions=assumptions + build.assumptions,
                    unresolved_effects=unresolved + build.unresolved_effects,
                    applied_steps=applied_steps,
                ),
            )

        policy_config = OpponentPolicyConfig(
            candidate_top_k=config.opponent_candidate_top_k,
            action_beam_width=config.opponent_action_beam_width,
            max_action_steps=config.opponent_action_max_steps,
            include_no_action_branch=config.include_opponent_no_action_branch,
        )
        branches = self.opponent_policy.expand(current, build.deck_pool, policy_config)
        results: list[CrossTurnTransitionResult] = []
        for branch in branches:
            branch_assumptions = assumptions + build.assumptions + branch.assumptions + (f"opponent_deck_pool_source={build.deck_pool.source}",)
            branch_unresolved = unresolved + build.unresolved_effects + branch.unresolved_effects
            branch_steps = applied_steps + tuple(f"opponent_action:{action.kind.value}:{action.source_uid}" for action in branch.actions)
            if branch.certainty == PredictionCertainty.PREDICT_ERROR:
                results.append(
                    CrossTurnTransitionResult(
                        snapshot=branch.snapshot,
                        certainty=PredictionCertainty.PREDICT_ERROR,
                        description=branch.description,
                        assumptions=branch_assumptions,
                        unresolved_effects=branch_unresolved,
                        applied_steps=branch_steps,
                    )
                )
                continue
            try:
                opponent_end = branch.snapshot if branch.snapshot.is_turn_ended else self.kernel.end_turn(branch.snapshot, "opponent").snapshot
            except RuleKernelError as exc:
                results.append(
                    CrossTurnTransitionResult(
                        snapshot=branch.snapshot,
                        certainty=PredictionCertainty.PREDICT_ERROR,
                        description=str(exc),
                        assumptions=branch_assumptions,
                        unresolved_effects=branch_unresolved,
                        applied_steps=branch_steps,
                    )
                )
                continue
            steps = branch_steps + ("opponent_end_turn",)
            results.append(self._partial(opponent_end, branch.description or "opponent turn modeled", branch_assumptions, branch_unresolved, steps))
        return tuple(results)

    def _build_opponent_pool(self, current: BattleSnapshot, class_en: str, config: CrossTurnTransitionConfig):
        pool_config = OpponentDeckPoolConfig(max_candidate_cards=config.opponent_candidate_top_k)
        if not config.opponent_deck_identification_enabled:
            return self.opponent_pool_builder.build_for_class(class_en, pool_config)
        hypotheses = self.opponent_deck_identifier.identify(
            current,
            opponent_class_en=class_en,
            config=OpponentDeckIdentifierConfig(
                format=pool_config.format,
                top_n=config.opponent_deck_hypothesis_top_n,
            ),
        )
        if hypotheses:
            return self.opponent_pool_builder.build_from_decks(
                tuple(hypothesis.deck for hypothesis in hypotheses),
                pool_config,
                source=f"opponent_deck_hypotheses:top={len(hypotheses)}",
                assumptions=(
                    "opponent_candidate_pool_from_deck_hypotheses",
                    f"opponent_deck_hypothesis_count={len(hypotheses)}",
                ),
            )
        if config.require_opponent_deck_hypothesis_when_enabled:
            return self.opponent_pool_builder._error(
                "opponent_deck_hypothesis_required_but_empty",
                unresolved=("opponent_deck_hypothesis_empty",),
            )
        fallback = self.opponent_pool_builder.build_for_class(class_en, pool_config)
        return replace(
            fallback,
            assumptions=fallback.assumptions + ("opponent_deck_identification_empty_fallback_to_class_pool",),
            unresolved_effects=fallback.unresolved_effects + ("opponent_deck_hypothesis_empty",),
        )

    def _advance_to_ally_turn(
        self,
        current: BattleSnapshot,
        initial_deck_pool: DeckPoolView | None,
        config: CrossTurnTransitionConfig,
        assumptions: tuple[str, ...],
        unresolved: tuple[str, ...],
        applied_steps: tuple[str, ...],
    ) -> tuple[CrossTurnTransitionResult, ...]:
        if self._has_battle_result(current):
            return (self._partial(current, "cross turn transition stopped after opponent turn", assumptions, unresolved, applied_steps),)
        if config.enabled and config.hidden_draw_branching_enabled and config.hidden_draw_top_k > 0:
            if initial_deck_pool is None or initial_deck_pool.status != "available":
                if config.require_deck_pool_when_enabled:
                    return (
                        CrossTurnTransitionResult(
                            snapshot=current,
                            certainty=PredictionCertainty.PREDICT_ERROR,
                            description="deck_pool_required_for_hidden_draw_branch",
                            assumptions=assumptions,
                            unresolved_effects=unresolved,
                            applied_steps=applied_steps,
                        ),
                    )
            else:
                branches = self._branch_hidden_draw(current, initial_deck_pool, config, assumptions, unresolved, applied_steps)
                if branches:
                    return branches
        try:
            ally_start = self.kernel.start_turn(current, "me", drawn_card=None)
        except RuleKernelError as exc:
            return (
                CrossTurnTransitionResult(
                    snapshot=current,
                    certainty=PredictionCertainty.PREDICT_ERROR,
                    description=str(exc),
                    assumptions=assumptions,
                    unresolved_effects=unresolved,
                    applied_steps=applied_steps,
                ),
            )
        return (
            self._partial(
                ally_start.snapshot,
                "cross turn transition advanced to next ally turn",
                assumptions,
                unresolved,
                applied_steps + ("ally_start_turn",),
            ),
        )

    def _should_branch_hidden_draw(self, snapshot: BattleSnapshot, config: CrossTurnTransitionConfig) -> bool:
        return bool(
            config.enabled
            and config.allow_hidden_draws
            and config.hidden_draw_branching_enabled
            and config.hidden_draw_top_k > 0
            and snapshot.me.deck_pool is not None
            and snapshot.me.deck_pool.status == "available"
        )

    def _branch_hidden_draw(
        self,
        current: BattleSnapshot,
        deck_pool: DeckPoolView,
        config: CrossTurnTransitionConfig,
        assumptions: tuple[str, ...],
        unresolved: tuple[str, ...],
        applied_steps: tuple[str, ...],
    ) -> tuple[CrossTurnTransitionResult, ...]:
        if deck_pool.status != "available":
            if config.require_deck_pool_when_enabled:
                return (
                    CrossTurnTransitionResult(
                        snapshot=current,
                        certainty=PredictionCertainty.PREDICT_ERROR,
                        description="deck_pool_required_for_hidden_draw_branch",
                        assumptions=assumptions,
                        unresolved_effects=unresolved,
                        applied_steps=applied_steps,
                    ),
                )
            return ()

        branches: list[CrossTurnTransitionResult] = []
        remaining_cards = sorted(deck_pool.cards, key=lambda card: (-card.count, card.card_id))
        for index, deck_card in enumerate(remaining_cards[: config.hidden_draw_top_k]):
            uid = next_synthetic_uid(current, index)
            card = materialize_draw_card(deck_card.card_id, current, uid=uid, resolver=self.card_resolver)
            if card is None:
                continue
            try:
                ally_start = self.kernel.start_turn(current, "me", drawn_card=card)
            except RuleKernelError as exc:
                branches.append(
                    CrossTurnTransitionResult(
                        snapshot=current,
                        certainty=PredictionCertainty.PREDICT_ERROR,
                        description=str(exc),
                        assumptions=assumptions + (f"hidden_draw_candidate_card_id={deck_card.card_id}",),
                        unresolved_effects=unresolved,
                        applied_steps=applied_steps,
                    )
                )
                continue
            updated_pool = self._consume_known_draw(deck_pool, deck_card.card_id)
            branches.append(
                self._partial(
                    self._attach_deck_pool(ally_start.snapshot, updated_pool),
                    f"cross turn transition branch hidden draw card_id={deck_card.card_id}",
                    assumptions + (
                        "hidden_draw_branch_from_deck_pool",
                        f"hidden_draw_candidate_card_id={deck_card.card_id}",
                        "draw_identity_is_branch_assumption_not_fact",
                    ),
                    tuple(item for item in unresolved if item != "hidden_draw_card_identity") + ("hidden_draw_identity_unverified",),
                    applied_steps + (f"ally_start_turn_branch_{deck_card.card_id}",),
                )
            )

        if config.include_hidden_draw_residual:
            try:
                ally_start = self.kernel.start_turn(current, "me", drawn_card=None)
            except RuleKernelError as exc:
                branches.append(
                    CrossTurnTransitionResult(
                        snapshot=current,
                        certainty=PredictionCertainty.PREDICT_ERROR,
                        description=str(exc),
                        assumptions=assumptions + ("hidden_draw_residual_branch",),
                        unresolved_effects=unresolved,
                        applied_steps=applied_steps,
                    )
                )
            else:
                branches.append(
                    self._partial(
                        self._attach_deck_pool(
                            ally_start.snapshot,
                            self._consume_unknown_draw(deck_pool),
                        ),
                        "cross turn transition residual hidden draw",
                        assumptions + ("hidden_draw_residual_branch", "hidden_draw_identity_withheld"),
                        unresolved,
                        applied_steps + ("ally_start_turn_residual",),
                    )
                )

        return tuple(branches)

    def _consume_known_draw(self, deck_pool: DeckPoolView, card_id: int) -> DeckPoolView:
        cards = []
        consumed = False
        for item in deck_pool.cards:
            if item.card_id == card_id and not consumed and item.count > 0:
                consumed = True
                if item.count > 1:
                    cards.append(replace(item, count=item.count - 1))
            else:
                cards.append(item)
        remaining_cards = tuple(card for card in cards if card.count > 0)
        return replace(
            deck_pool,
            cards=remaining_cards,
            remaining_count=max(0, deck_pool.remaining_count - 1),
            consumed_hidden_draws=deck_pool.consumed_hidden_draws + 1,
            warnings=deck_pool.warnings,
        )

    def _consume_unknown_draw(self, deck_pool: DeckPoolView) -> DeckPoolView:
        return replace(
            deck_pool,
            remaining_count=max(0, deck_pool.remaining_count - 1),
            consumed_hidden_draws=deck_pool.consumed_hidden_draws + 1,
            status="ambiguous" if deck_pool.status == "available" else deck_pool.status,
            warnings=deck_pool.warnings + ("hidden_draw_identity_withheld",),
        )

    def _attach_deck_pool(self, snapshot: BattleSnapshot, deck_pool: DeckPoolView) -> BattleSnapshot:
        return replace(snapshot, me=replace(snapshot.me, deck_pool=deck_pool))

    def _error(self, snapshot: BattleSnapshot, description: str) -> CrossTurnTransitionResult:
        return CrossTurnTransitionResult(snapshot=snapshot, certainty=PredictionCertainty.PREDICT_ERROR, description=description)

    def _partial(
        self,
        snapshot: BattleSnapshot,
        description: str,
        assumptions: tuple[str, ...],
        unresolved_effects: tuple[str, ...],
        applied_steps: tuple[str, ...],
    ) -> CrossTurnTransitionResult:
        return CrossTurnTransitionResult(
            snapshot=snapshot,
            certainty=PredictionCertainty.PARTIAL,
            description=description,
            assumptions=assumptions,
            unresolved_effects=unresolved_effects,
            applied_steps=applied_steps,
        )

    def _has_battle_result(self, snapshot: BattleSnapshot) -> bool:
        return snapshot.me.result != 0 or snapshot.opponent.result != 0
