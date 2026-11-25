# -*- coding: utf-8 -*-
"""Opponent turn branch policy over ambiguous meta deck pools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.cards import CardResolver
from decision_engine.simulation.opponent_hand_materializer import OpponentHandMaterializer
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot, DeckPoolCard, DeckPoolView

from .player_local_predictor import PlayerLocalPredictor

if TYPE_CHECKING:
    from decision_engine.actions.player_local_enumerator import PlayerLocalActionEnumerator


@dataclass(frozen=True)
class OpponentPolicyConfig:
    candidate_top_k: int = 8
    action_beam_width: int = 4
    max_action_steps: int = 1
    include_no_action_branch: bool = True


@dataclass(frozen=True)
class OpponentPolicyBranch:
    snapshot: BattleSnapshot
    actions: tuple[LegalAction, ...]
    certainty: PredictionCertainty
    assumptions: tuple[str, ...] = ()
    unresolved_effects: tuple[str, ...] = ()
    description: str = ""
    score: float = 0.0


class OpponentActionScorer:
    def score(self, before: BattleSnapshot, after: BattleSnapshot, action: LegalAction, certainty: PredictionCertainty) -> float:
        score = 0.0
        score += max(before.me.life - after.me.life, 0) * 20.0
        score += max(len(before.me.field) - len(after.me.field), 0) * 8.0
        score += max(len(after.opponent.field) - len(before.opponent.field), 0) * 4.0
        score += max(before.opponent.pp - after.opponent.pp, 0) * 2.0
        if action.kind is ActionKind.TURN_END:
            score -= 5.0
        if certainty == PredictionCertainty.PARTIAL:
            score -= 1.0
        if certainty == PredictionCertainty.UNKNOWN:
            score -= 5.0
        if certainty == PredictionCertainty.PREDICT_ERROR:
            score -= 1000.0
        return score


class OpponentTurnPolicy:
    def __init__(
        self,
        *,
        enumerator: PlayerLocalActionEnumerator | None = None,
        predictor: PlayerLocalPredictor | None = None,
        materializer: OpponentHandMaterializer | None = None,
        scorer: OpponentActionScorer | None = None,
        resolver: CardResolver | None = None,
    ):
        if enumerator is None:
            from decision_engine.actions.player_local_enumerator import PlayerLocalActionEnumerator

            enumerator = PlayerLocalActionEnumerator()
        self.enumerator = enumerator
        self._predictor = predictor
        self.materializer = materializer or OpponentHandMaterializer(resolver=resolver)
        self.scorer = scorer or OpponentActionScorer()

    @property
    def predictor(self) -> "PlayerLocalPredictor":
        if self._predictor is None:
            raise ValueError("OpponentTurnPolicy.predictor is None — pass PlayerLocalPredictor(LocalPredictor(templates_by_base_card_id=...)) at construction")
        return self._predictor

    def expand(
        self,
        snapshot: BattleSnapshot,
        deck_pool: DeckPoolView,
        config: OpponentPolicyConfig | None = None,
    ) -> tuple[OpponentPolicyBranch, ...]:
        config = config or OpponentPolicyConfig()
        branches: list[OpponentPolicyBranch] = []
        for index, candidate in enumerate(self._candidate_cards(snapshot, deck_pool, config)):
            hand_branch = self.materializer.materialize(snapshot, candidate, branch_index=index)
            if hand_branch.unresolved_effects:
                branches.append(
                    OpponentPolicyBranch(
                        snapshot=hand_branch.snapshot,
                        actions=(),
                        certainty=PredictionCertainty.PREDICT_ERROR,
                        assumptions=hand_branch.assumptions,
                        unresolved_effects=hand_branch.unresolved_effects,
                        description="opponent_candidate_cards_not_materializable",
                        score=-1000.0,
                    )
                )
                continue
            for action in self._candidate_actions(hand_branch.snapshot):
                prediction = self.predictor.predict(hand_branch.snapshot, action, player="opponent")
                score = self.scorer.score(hand_branch.snapshot, prediction.snapshot, action, prediction.certainty)
                branches.append(
                    OpponentPolicyBranch(
                        snapshot=prediction.snapshot,
                        actions=(action,),
                        certainty=prediction.certainty,
                        assumptions=hand_branch.assumptions
                        + prediction.assumptions
                        + ("opponent_modeling_enabled", "opponent_action_branch_from_meta_candidate"),
                        unresolved_effects=hand_branch.unresolved_effects + prediction.unresolved_effects,
                        description="; ".join(part for part in ("opponent action branch", prediction.description) if part),
                        score=score,
                    )
                )

        branches.sort(key=lambda branch: branch.score, reverse=True)
        kept = branches[: max(config.action_beam_width, 0)]
        if config.include_no_action_branch:
            kept.append(
                OpponentPolicyBranch(
                    snapshot=snapshot,
                    actions=(),
                    certainty=PredictionCertainty.PARTIAL,
                    assumptions=(
                        "opponent_modeling_enabled",
                        "opponent_no_action_residual_branch",
                        "opponent_actions_not_exhaustively_modelled",
                    ),
                    unresolved_effects=("opponent_turn_actions_residual",),
                    description="opponent no-action residual branch",
                    score=-10.0,
                )
            )
        if kept:
            return tuple(kept)
        return (
            OpponentPolicyBranch(
                snapshot=snapshot,
                actions=(),
                certainty=PredictionCertainty.UNKNOWN,
                assumptions=("opponent_modeling_enabled", "opponent_policy_no_action_candidates"),
                unresolved_effects=("opponent_turn_actions",),
                description="opponent policy produced no action candidates",
                score=-100.0,
            ),
        )

    def _candidate_cards(self, snapshot: BattleSnapshot, deck_pool: DeckPoolView, config: OpponentPolicyConfig) -> tuple[DeckPoolCard, ...]:
        effective_pp = self._effective_pp(snapshot)
        cards = sorted(
            deck_pool.cards,
            key=lambda card: (
                -self._curve_pressure_weight(card.cost, snapshot.opponent.pp, effective_pp),
                -card.count,
                -card.cost,
                card.card_id,
            ),
        )
        return tuple(cards[: max(config.candidate_top_k, 0)])

    def _effective_pp(self, snapshot: BattleSnapshot) -> int:
        extra_pp = snapshot.opponent.extra_pp
        if extra_pp is None or extra_pp.extra_pp <= 0 or extra_pp.is_used_extra_pp_this_turn:
            return snapshot.opponent.pp
        return snapshot.opponent.pp + 1

    def _curve_pressure_weight(self, cost: int, current_pp: int, effective_pp: int) -> float:
        if cost <= 0:
            return 0.2
        if cost == effective_pp and effective_pp > current_pp:
            return 1.0
        if cost == current_pp:
            return 0.95
        if cost == effective_pp:
            return 0.9
        if cost == effective_pp - 1:
            return 0.7
        if cost > effective_pp:
            return 0.1
        if cost <= 2 and effective_pp >= 5:
            return 0.35
        return 0.5

    def _candidate_actions(self, snapshot: BattleSnapshot) -> tuple[LegalAction, ...]:
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        return tuple(
            action
            for action in actions
            if action.kind
            in {
                ActionKind.PLAY,
                ActionKind.ATTACK_LEADER,
                ActionKind.ATTACK_FOLLOWER,
                ActionKind.EVOLVE,
                ActionKind.SUPER_EVOLVE,
                ActionKind.ACTIVATE,
                ActionKind.SPECIAL_ACTION,
                ActionKind.FUSION,
                ActionKind.EXTRA_PP,
                ActionKind.TURN_END,
            }
        )
