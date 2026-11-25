# -*- coding: utf-8 -*-
"""Opponent policy tests."""

import unittest

from decision_engine.execution.local_predictor import LocalPredictor
from decision_engine.execution.opponent_policy import OpponentPolicyConfig, OpponentTurnPolicy
from decision_engine.execution.player_local_predictor import PlayerLocalPredictor
from decision_engine.simulation.opponent_hand_materializer import OpponentHandMaterializer
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, DeckPoolCard, DeckPoolView, ExtraPpView, PlayerView


class FakeBase:
    def __init__(self, card_id=101, cost=2):
        self.base_card_id = card_id
        self.name = "候选随从"
        self.cost = cost
        self.atk = 2
        self.life = 2
        self.card_type = "follower"
        self.type_id = 1
        self.tribes = (0,)


class FakeResolver:
    costs = {101: 2, 201: 2, 202: 5, 203: 6, 204: 7}

    def get_base_by_card_id(self, card_id):
        cost = self.costs.get(card_id)
        return FakeBase(card_id, cost) if cost is not None else None


class OpponentPolicyTest(unittest.TestCase):
    def test_expands_candidate_play_branch_and_residual(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, unique_id=10),
            opponent=PlayerView(life=20, max_life=20, pp=2, max_pp=2, unique_id=20),
            cards_by_uid={},
            is_ally_turn=False,
        )
        pool = DeckPoolView(source="meta", cards=(DeckPoolCard(card_id=101, count=3, cost=2),), status="ambiguous")
        policy = OpponentTurnPolicy(predictor=PlayerLocalPredictor(LocalPredictor()), materializer=OpponentHandMaterializer(resolver=FakeResolver()))

        branches = policy.expand(snapshot, pool, OpponentPolicyConfig(candidate_top_k=1, action_beam_width=1, include_no_action_branch=True))

        self.assertEqual(len(branches), 2)
        play_branch = branches[0]
        self.assertEqual(play_branch.certainty, PredictionCertainty.PARTIAL)
        self.assertEqual(play_branch.snapshot.opponent.field, (-1,))
        self.assertIn("opponent_action_branch_from_meta_candidate", play_branch.assumptions)
        self.assertIn("opponent_hand_identity_is_branch_assumption_not_fact", play_branch.assumptions)
        self.assertIn("opponent_no_action_residual_branch", branches[1].assumptions)

    def test_materialization_failure_is_error_branch(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2),
            opponent=PlayerView(life=20, max_life=20, pp=2, max_pp=2),
            cards_by_uid={},
            is_ally_turn=False,
        )
        pool = DeckPoolView(source="meta", cards=(DeckPoolCard(card_id=999, count=3, cost=2),), status="ambiguous")
        policy = OpponentTurnPolicy(predictor=PlayerLocalPredictor(LocalPredictor()), materializer=OpponentHandMaterializer(resolver=FakeResolver()))

        branches = policy.expand(snapshot, pool, OpponentPolicyConfig(candidate_top_k=1, action_beam_width=1, include_no_action_branch=False))

        self.assertEqual(branches[0].certainty, PredictionCertainty.PREDICT_ERROR)
        self.assertIn("opponent_candidate_materialization_failed:999", branches[0].unresolved_effects)
    def test_candidate_cards_prioritize_current_pp_high_cost_threat(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=5),
            opponent=PlayerView(life=20, max_life=20, pp=5, max_pp=5),
            cards_by_uid={},
            is_ally_turn=False,
        )
        pool = DeckPoolView(
            source="meta",
            cards=(
                DeckPoolCard(card_id=201, count=100, cost=2),
                DeckPoolCard(card_id=202, count=50, cost=5),
                DeckPoolCard(card_id=204, count=100, cost=7),
            ),
            status="ambiguous",
        )

        candidates = OpponentTurnPolicy()._candidate_cards(snapshot, pool, OpponentPolicyConfig(candidate_top_k=2))

        self.assertEqual([card.card_id for card in candidates], [202, 201])

    def test_candidate_cards_prioritize_extra_pp_unlock_window(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=5),
            opponent=PlayerView(life=20, max_life=20, pp=5, max_pp=5, extra_pp=ExtraPpView(extra_pp=1)),
            cards_by_uid={},
            is_ally_turn=False,
        )
        pool = DeckPoolView(
            source="meta",
            cards=(
                DeckPoolCard(card_id=201, count=100, cost=2),
                DeckPoolCard(card_id=202, count=100, cost=5),
                DeckPoolCard(card_id=203, count=50, cost=6),
            ),
            status="ambiguous",
        )

        candidates = OpponentTurnPolicy()._candidate_cards(snapshot, pool, OpponentPolicyConfig(candidate_top_k=3))

        self.assertEqual([card.card_id for card in candidates], [203, 202, 201])


if __name__ == "__main__":
    unittest.main()
