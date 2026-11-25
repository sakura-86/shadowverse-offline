# -*- coding: utf-8 -*-
"""Materialize hypothetical opponent hand cards for local branches."""

from __future__ import annotations

from dataclasses import dataclass, replace

from decision_engine.cards import CardResolver
from decision_engine.state.snapshot import BattleSnapshot, DeckPoolCard

from .deck_pool import materialize_draw_card, next_synthetic_uid


@dataclass(frozen=True)
class OpponentHandBranch:
    snapshot: BattleSnapshot
    candidate_card_id: int
    materialized_uid: int | None = None
    assumptions: tuple[str, ...] = ()
    unresolved_effects: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class OpponentHandMaterializer:
    def __init__(self, *, resolver: CardResolver | None = None):
        self.resolver = resolver

    def materialize(self, snapshot: BattleSnapshot, candidate: DeckPoolCard, *, branch_index: int = 0) -> OpponentHandBranch:
        uid = next_synthetic_uid(snapshot, branch_index)
        card = materialize_draw_card(candidate.card_id, snapshot, uid=uid, resolver=self.resolver)
        if card is None:
            return OpponentHandBranch(
                snapshot=snapshot,
                candidate_card_id=candidate.card_id,
                assumptions=(
                    "opponent_candidate_pool_from_meta",
                    f"opponent_candidate_card_id={candidate.card_id}",
                    "opponent_candidate_card_is_hypothetical",
                ),
                unresolved_effects=(f"opponent_candidate_materialization_failed:{candidate.card_id}",),
                warnings=(f"opponent_candidate_materialization_failed:{candidate.card_id}",),
            )

        cards_by_uid = dict(snapshot.cards_by_uid)
        cards_by_uid[uid] = card
        opponent = replace(snapshot.opponent, hand=tuple((*snapshot.opponent.hand, uid)))
        return OpponentHandBranch(
            snapshot=replace(snapshot, opponent=opponent, cards_by_uid=cards_by_uid),
            candidate_card_id=candidate.card_id,
            materialized_uid=uid,
            assumptions=(
                "opponent_candidate_pool_from_meta",
                f"opponent_candidate_card_id={candidate.card_id}",
                "opponent_candidate_card_is_hypothetical",
                "opponent_hand_identity_is_branch_assumption_not_fact",
                "opponent_candidate_card_may_represent_hidden_hand_or_draw",
            ),
        )
