# -*- coding: utf-8 -*-
"""CrossTurnTransition 测试。"""

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.meta.models import CardSlot, DeckEntry
from decision_engine.strategy.opponent_deck_identifier import DeckHypothesis
from decision_engine.execution import CrossTurnTransition, CrossTurnTransitionConfig, LocalRollbackSearch, RollbackSearchConfig
from decision_engine.execution.opponent_policy import OpponentPolicyBranch
from decision_engine.simulation.opponent_deck_pool import OpponentDeckPoolBuildResult
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, DeckPoolCard, DeckPoolView, ExtraPpView, PlayerView


class CrossTurnTransitionTest(unittest.TestCase):
    def test_advances_from_ended_ally_turn_to_next_ally_turn_without_cards(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=3, turn=1, is_first_side=False),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=3, turn=0, is_first_side=True),
            cards_by_uid={},
            is_ally_turn=True,
            is_turn_ended=True,
        )

        result = CrossTurnTransition().apply_after_ally_turn_end(snapshot)

        self.assertEqual(result.certainty, PredictionCertainty.PARTIAL)
        self.assertTrue(result.snapshot.is_ally_turn)
        self.assertFalse(result.snapshot.is_turn_ended)
        self.assertEqual(result.snapshot.opponent.turn, 1)
        self.assertEqual(result.snapshot.me.turn, 2)
        self.assertEqual(result.snapshot.opponent.deck_count, 2)
        self.assertEqual(result.snapshot.me.deck_count, 2)
        self.assertEqual(result.snapshot.cards_by_uid, {})
        self.assertIn("opponent_actions_skipped", result.assumptions)
        self.assertIn("hidden_draws_without_card_identity", result.assumptions)
        self.assertIn("opponent_turn_actions", result.unresolved_effects)
        self.assertIn("hidden_draw_card_identity", result.unresolved_effects)
        self.assertEqual(result.applied_steps, ("opponent_start_turn", "opponent_end_turn", "ally_start_turn"))

    def test_hidden_draw_branch_uses_deck_pool(self):
        snapshot = BattleSnapshot(
            me=PlayerView(
                life=20,
                max_life=20,
                pp=0,
                max_pp=2,
                deck_count=2,
                turn=1,
                is_first_side=False,
                deck_pool=DeckPoolView(source="practice_deck_state.user_deck", cards=(DeckPoolCard(card_id=10001110, count=1), DeckPoolCard(card_id=10001111, count=1)), observed_total=2, remaining_count=2),
            ),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=2, turn=0, is_first_side=True),
            cards_by_uid={},
            is_ally_turn=True,
            is_turn_ended=True,
        )

        result = CrossTurnTransition().apply_after_ally_turn_end_branches(
            snapshot,
            config=CrossTurnTransitionConfig(enabled=True, hidden_draw_branching_enabled=True, hidden_draw_top_k=2),
        )

        self.assertEqual(len(result), 3)
        self.assertTrue(any("hidden_draw_branch_from_deck_pool" in item.assumptions for item in result))
        self.assertTrue(any("hidden_draw_residual_branch" in item.assumptions for item in result))
        candidate = next(item for item in result if any("hidden_draw_candidate_card_id=10001110" in a for a in item.assumptions))
        self.assertEqual(candidate.certainty, PredictionCertainty.PARTIAL)
        self.assertIn(candidate.snapshot.me.hand[0], candidate.snapshot.cards_by_uid)
        self.assertEqual(candidate.snapshot.cards_by_uid[candidate.snapshot.me.hand[0]].card_id, 10001110)

    def test_rejects_non_ended_or_non_ally_snapshot(self):
        transition = CrossTurnTransition()
        actionable = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=3),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=3),
            cards_by_uid={},
            is_ally_turn=True,
            is_turn_ended=False,
        )
        opponent_turn = BattleSnapshot(
            me=actionable.me,
            opponent=actionable.opponent,
            cards_by_uid={},
            is_ally_turn=False,
            is_turn_ended=True,
        )

        not_ended = transition.apply_after_ally_turn_end(actionable)
        not_ally = transition.apply_after_ally_turn_end(opponent_turn)

        self.assertEqual(not_ended.certainty, PredictionCertainty.PREDICT_ERROR)
        self.assertEqual(not_ended.snapshot, actionable)
        self.assertEqual(not_ally.certainty, PredictionCertainty.PREDICT_ERROR)
        self.assertEqual(not_ally.snapshot, opponent_turn)

    def test_reuses_core_extra_pp_grant_on_next_ally_start(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=6, deck_count=3, turn=5, is_first_side=False, extra_pp=ExtraPpView(extra_pp=0)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=6, deck_count=3, turn=5, is_first_side=True),
            cards_by_uid={},
            is_ally_turn=True,
            is_turn_ended=True,
        )

        result = CrossTurnTransition().apply_after_ally_turn_end(snapshot)

        self.assertEqual(result.certainty, PredictionCertainty.PARTIAL)
        self.assertEqual(result.snapshot.me.turn, 6)
        self.assertEqual(result.snapshot.me.extra_pp.extra_pp, 1)
        self.assertEqual(result.snapshot.me.extra_pp.restore_extra_pp_turn, 6)
        self.assertFalse(result.snapshot.me.extra_pp.is_used_extra_pp_this_turn)

    def test_opponent_modeling_requires_class_when_enabled(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=3, turn=1, is_first_side=False),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=3, turn=0, is_first_side=True),
            cards_by_uid={},
            is_ally_turn=True,
            is_turn_ended=True,
        )

        result = CrossTurnTransition().apply_after_ally_turn_end(
            snapshot,
            CrossTurnTransitionConfig(enabled=True, opponent_modeling_enabled=True),
        )

        self.assertEqual(result.certainty, PredictionCertainty.PREDICT_ERROR)
        self.assertEqual(result.description, "opponent_class_required_for_modeling")
        self.assertIn("opponent_class_unknown", result.unresolved_effects)

    def test_opponent_modeling_branch_replaces_skipped_assumption(self):
        class FakeBuilder:
            def build_for_class(self, class_en, config):
                return OpponentDeckPoolBuildResult(
                    deck_pool=DeckPoolView(source="fake_meta", cards=(DeckPoolCard(card_id=101, count=1),), status="ambiguous"),
                    certainty=PredictionCertainty.PARTIAL,
                    assumptions=("opponent_candidate_pool_from_meta", f"opponent_class_en={class_en}"),
                    unresolved_effects=("opponent_actual_deck_unknown",),
                )

        class FakePolicy:
            def expand(self, snapshot, deck_pool, config):
                return (
                    OpponentPolicyBranch(
                        snapshot=snapshot,
                        actions=(LegalAction(kind=ActionKind.PLAY, source_uid=123),),
                        certainty=PredictionCertainty.PARTIAL,
                        assumptions=("opponent_action_branch_from_meta_candidate",),
                        unresolved_effects=("opponent_hidden_hand_full_state",),
                        description="fake opponent action",
                    ),
                )

        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=3, turn=1, is_first_side=False),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=3, turn=0, is_first_side=True),
            cards_by_uid={},
            is_ally_turn=True,
            is_turn_ended=True,
        )

        result = CrossTurnTransition(opponent_pool_builder=FakeBuilder(), opponent_policy=FakePolicy()).apply_after_ally_turn_end(
            snapshot,
            CrossTurnTransitionConfig(enabled=True, opponent_modeling_enabled=True, opponent_class_en="Runecraft"),
        )

        self.assertEqual(result.certainty, PredictionCertainty.PARTIAL)
        self.assertNotIn("opponent_actions_skipped", result.assumptions)
        self.assertIn("opponent_action_branch_from_meta_candidate", result.assumptions)
        self.assertIn("opponent_deck_pool_source=fake_meta", result.assumptions)
        self.assertEqual(result.applied_steps, ("opponent_start_turn", "opponent_action:play:123", "opponent_end_turn", "ally_start_turn"))

    def test_opponent_deck_identification_uses_hypothesis_pool(self):
        deck = DeckEntry("Runecraft", 3, "match", "2026-07-10", "", "standard", "增幅", [CardSlot("101", 3, "A", 1, card_set_id=10008)])

        class FakeIdentifier:
            def identify(self, snapshot, *, opponent_class_en, config):
                return (DeckHypothesis(deck=deck, score=10.0, matched_card_ids=(101,), missing_seen_card_ids=()),)

        class FakeBuilder:
            def build_from_decks(self, decks, config, *, source, assumptions):
                self.decks = decks
                return OpponentDeckPoolBuildResult(
                    deck_pool=DeckPoolView(source="hypothesis_pool", cards=(DeckPoolCard(card_id=101, count=100),), status="ambiguous"),
                    certainty=PredictionCertainty.PARTIAL,
                    assumptions=assumptions + ("from_hypothesis_builder",),
                    unresolved_effects=(),
                )

            def build_for_class(self, class_en, config):
                raise AssertionError("class pool should not be used")

        class FakePolicy:
            def expand(self, snapshot, deck_pool, config):
                return (OpponentPolicyBranch(snapshot=snapshot, actions=(), certainty=PredictionCertainty.PARTIAL, assumptions=("policy_used",), description="policy"),)

        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=3, turn=1, is_first_side=False),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=3, turn=0, is_first_side=True),
            cards_by_uid={},
            is_ally_turn=True,
            is_turn_ended=True,
        )

        result = CrossTurnTransition(opponent_pool_builder=FakeBuilder(), opponent_policy=FakePolicy(), opponent_deck_identifier=FakeIdentifier()).apply_after_ally_turn_end(
            snapshot,
            CrossTurnTransitionConfig(enabled=True, opponent_modeling_enabled=True, opponent_class_en="Runecraft", opponent_deck_identification_enabled=True),
        )

        self.assertIn("opponent_candidate_pool_from_deck_hypotheses", result.assumptions)
        self.assertIn("from_hypothesis_builder", result.assumptions)
        self.assertIn("opponent_deck_pool_source=hypothesis_pool", result.assumptions)

    def test_opponent_deck_identification_required_empty_is_error(self):
        class EmptyIdentifier:
            def identify(self, snapshot, *, opponent_class_en, config):
                return ()

        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=3, turn=1, is_first_side=False),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=3, turn=0, is_first_side=True),
            cards_by_uid={},
            is_ally_turn=True,
            is_turn_ended=True,
        )

        result = CrossTurnTransition(opponent_deck_identifier=EmptyIdentifier()).apply_after_ally_turn_end(
            snapshot,
            CrossTurnTransitionConfig(enabled=True, opponent_modeling_enabled=True, opponent_class_en="Runecraft", opponent_deck_identification_enabled=True, require_opponent_deck_hypothesis_when_enabled=True),
        )

        self.assertEqual(result.certainty, PredictionCertainty.PREDICT_ERROR)
        self.assertIn("opponent_deck_hypothesis_empty", result.unresolved_effects)


if __name__ == "__main__":
    unittest.main()
