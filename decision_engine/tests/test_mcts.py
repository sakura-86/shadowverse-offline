# -*- coding: utf-8 -*-
"""Phase L8 rollback search 测试。"""

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.execution import CrossTurnTransitionConfig, LocalRollbackSearch, PartialBranchPolicy, PredictErrorPolicy, RollbackSearchConfig
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, DeckPoolCard, DeckPoolView, PlayerView


class RollbackSearchTest(unittest.TestCase):
    def make_cross_turn_snapshot(self):
        cards = {1: CardView(uid=1, card_id=101, base_card_id=101, name="下一回合随从", cost=3, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER)}
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=3, deck_count=3, turn=1, hand=(1,), is_first_side=False),
            opponent=PlayerView(life=5, max_life=20, pp=0, max_pp=3, deck_count=3, turn=0, is_first_side=True),
            cards_by_uid=cards,
            source="test",
            is_ally_turn=True,
        )

        def provider(current):
            if not current.is_ally_turn or current.is_turn_ended:
                return ()
            if current.me.pp == 0:
                return (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)
            return (LegalAction(kind=ActionKind.PLAY, source_uid=1, cost=3),)

        return snapshot, provider

    def make_snapshot(self):
        me = PlayerView(life=20, max_life=20, pp=3, max_pp=3, hand=(1,), field=(3,))
        opponent = PlayerView(life=5, max_life=20, pp=0, max_pp=3, field=(4,))
        cards = {
            1: CardView(uid=1, card_id=101, base_card_id=101, name="随从", cost=3, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER),
            3: CardView(uid=3, card_id=103, base_card_id=103, name="攻击者", atk=3, life=4, max_life=4, attacks_left=1, can_attack=True, can_attack_leader=True),
            4: CardView(uid=4, card_id=104, base_card_id=104, name="敌方", atk=1, life=2, max_life=2),
        }
        return BattleSnapshot(me=me, opponent=opponent, cards_by_uid=cards, source="test", is_ally_turn=True)

    def score(self, snapshot, certainty):
        penalty = 0
        if certainty == PredictionCertainty.PARTIAL:
            penalty = 1
        if certainty == PredictionCertainty.PREDICT_ERROR:
            penalty = 1000
        return -snapshot.opponent.life - penalty + len(snapshot.me.field)

    def test_search_selects_best_local_prediction_without_mutating_root(self):
        snapshot = self.make_snapshot()
        actions = (
            LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=3),
            LegalAction(kind=ActionKind.PLAY, source_uid=1, cost=3),
        )
        search = LocalRollbackSearch(lambda _snapshot: actions, self.score, config=RollbackSearchConfig(max_depth=1, beam_width=4, partial_penalty=0))

        result = search.search(snapshot)

        self.assertIsNotNone(result.best)
        assert result.best is not None
        self.assertEqual(result.best.actions[0].kind, ActionKind.ATTACK_LEADER)
        self.assertEqual(result.best.snapshot.opponent.life, 2)
        self.assertEqual(snapshot.opponent.life, 5)
        self.assertEqual(snapshot.me.hand, (1,))
        self.assertEqual(result.trace.predictions, 2)

    def test_predict_error_is_skipped_by_default(self):
        snapshot = self.make_snapshot()
        actions = (
            LegalAction(kind=ActionKind.FUSION, source_uid=1),
            LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=3),
        )
        search = LocalRollbackSearch(lambda _snapshot: actions, self.score, config=RollbackSearchConfig(max_depth=1))

        result = search.search(snapshot)

        self.assertEqual(result.trace.predict_error_skipped, 1)
        self.assertTrue(all(node.certainty != PredictionCertainty.PREDICT_ERROR for node in result.candidates))

    def test_predict_error_can_stop_search(self):
        snapshot = self.make_snapshot()
        search = LocalRollbackSearch(
            lambda _snapshot: (LegalAction(kind=ActionKind.FUSION, source_uid=1),),
            self.score,
            config=RollbackSearchConfig(max_depth=1, predict_error_policy=PredictErrorPolicy.STOP),
        )

        result = search.search(snapshot)

        self.assertEqual(result.trace.terminal_reason, "predict_error")
        self.assertEqual(result.trace.predictions, 1)

    def test_cross_turn_transition_allows_search_after_turn_end_when_enabled(self):
        snapshot, provider = self.make_cross_turn_snapshot()
        search = LocalRollbackSearch(
            provider,
            lambda current, _certainty: len(current.me.field),
            config=RollbackSearchConfig(max_depth=2, beam_width=4, partial_penalty=0, cross_turn_transition=CrossTurnTransitionConfig(enabled=True)),
        )

        result = search.search(snapshot)

        self.assertIsNotNone(result.best)
        assert result.best is not None
        self.assertEqual([action.kind for action in result.best.actions], [ActionKind.TURN_END, ActionKind.PLAY])
        self.assertNotIn(ActionKind.TURN_START, [action.kind for action in result.best.actions])
        self.assertEqual(result.trace.cross_turn_transitions, 1)
        self.assertEqual(result.trace.max_ally_turn_index_reached, 1)
        self.assertEqual(result.best.cross_turn_transitions, 1)
        self.assertEqual(result.best.ally_turn_index, 1)
        self.assertEqual(result.best.snapshot.me.field, (1,))
        self.assertIn("opponent_turn_actions", result.best.unresolved_effects)

    def test_cross_turn_transition_expands_deck_pool_hidden_draw_branches(self):
        snapshot = BattleSnapshot(
            me=PlayerView(
                life=20,
                max_life=20,
                pp=0,
                max_pp=2,
                deck_count=2,
                turn=1,
                is_first_side=False,
                deck_pool=DeckPoolView(
                    source="practice_deck_state.user_deck",
                    cards=(
                        DeckPoolCard(card_id=10001110, count=1),
                        DeckPoolCard(card_id=10001111, count=1),
                    ),
                    observed_total=2,
                    remaining_count=2,
                ),
            ),
            opponent=PlayerView(life=5, max_life=20, pp=0, max_pp=2, deck_count=2, turn=0, is_first_side=True),
            cards_by_uid={},
            source="test",
            is_ally_turn=True,
        )

        def provider(current):
            if not current.is_ally_turn or current.is_turn_ended:
                return ()
            if current.me.pp == 0:
                return (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)
            return tuple(LegalAction(kind=ActionKind.PLAY, source_uid=uid, cost=current.cards_by_uid[uid].cost) for uid in current.me.hand)

        result = LocalRollbackSearch(
            provider,
            lambda current, _certainty: len(current.me.hand) + len(current.me.field),
            config=RollbackSearchConfig(
                max_depth=2,
                beam_width=8,
                partial_penalty=0,
                cross_turn_transition=CrossTurnTransitionConfig(
                    enabled=True,
                    hidden_draw_branching_enabled=True,
                    hidden_draw_top_k=2,
                ),
            ),
        ).search(snapshot)

        turn_end_nodes = [node for node in result.candidates if [action.kind for action in node.actions] == [ActionKind.TURN_END]]
        self.assertEqual(len(turn_end_nodes), 3)
        self.assertEqual(result.trace.cross_turn_transitions, 1)
        self.assertEqual(result.trace.hidden_draw_branches, 2)
        self.assertEqual(result.trace.hidden_draw_residual_branches, 1)
        self.assertEqual(result.trace.hidden_draw_branch_errors, 0)
        self.assertTrue(any(10001110 in {card.card_id for card in node.snapshot.cards_by_uid.values()} for node in turn_end_nodes))
        self.assertTrue(any([action.kind for action in node.actions] == [ActionKind.TURN_END, ActionKind.PLAY] for node in result.candidates))

    def test_cross_turn_transition_counts_hidden_draw_branch_error_when_deck_pool_missing(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, deck_count=2, turn=1, is_first_side=False),
            opponent=PlayerView(life=5, max_life=20, pp=0, max_pp=2, deck_count=2, turn=0, is_first_side=True),
            cards_by_uid={},
            source="test",
            is_ally_turn=True,
        )

        result = LocalRollbackSearch(
            lambda current: (LegalAction(kind=ActionKind.TURN_END, source_uid=0),) if current.is_ally_turn and not current.is_turn_ended else (),
            self.score,
            config=RollbackSearchConfig(
                max_depth=1,
                beam_width=4,
                partial_penalty=0,
                cross_turn_transition=CrossTurnTransitionConfig(
                    enabled=True,
                    hidden_draw_branching_enabled=True,
                    hidden_draw_top_k=1,
                ),
            ),
        ).search(snapshot)

        self.assertEqual(result.trace.cross_turn_transitions, 1)
        self.assertEqual(result.trace.hidden_draw_branches, 0)
        self.assertEqual(result.trace.hidden_draw_residual_branches, 0)
        self.assertEqual(result.trace.hidden_draw_branch_errors, 1)
        self.assertEqual(result.trace.predict_error_skipped, 1)

    def test_cross_turn_transition_tracks_ally_turn_index(self):
        snapshot, provider = self.make_cross_turn_snapshot()

        result = LocalRollbackSearch(
            provider,
            lambda current, _certainty: len(current.me.field),
            config=RollbackSearchConfig(max_depth=2, beam_width=4, partial_penalty=0, cross_turn_transition=CrossTurnTransitionConfig(enabled=True)),
        ).search(snapshot)

        root = next(node for node in result.candidates if not node.actions)
        self.assertEqual(root.ally_turn_index, 0)
        self.assertIsNotNone(result.best)
        assert result.best is not None
        self.assertEqual(result.best.ally_turn_index, 1)
        self.assertEqual(result.best.cross_turn_transitions, 1)
        self.assertEqual(result.trace.max_ally_turn_index_reached, 1)

    def test_cross_turn_transition_respects_max_ally_turns_one(self):
        snapshot, provider = self.make_cross_turn_snapshot()

        result = LocalRollbackSearch(
            provider,
            lambda current, _certainty: len(current.me.field),
            config=RollbackSearchConfig(max_depth=2, beam_width=4, partial_penalty=0, cross_turn_transition=CrossTurnTransitionConfig(enabled=True, max_ally_turns=1)),
        ).search(snapshot)

        self.assertEqual(result.trace.cross_turn_transitions, 0)
        self.assertEqual(result.trace.cross_turn_budget_skipped, 1)
        self.assertFalse(any([action.kind for action in node.actions] == [ActionKind.TURN_END, ActionKind.PLAY] for node in result.candidates))

    def test_cross_turn_transition_allows_next_turn_with_max_ally_turns_two(self):
        snapshot, provider = self.make_cross_turn_snapshot()

        result = LocalRollbackSearch(
            provider,
            lambda current, _certainty: len(current.me.field),
            config=RollbackSearchConfig(max_depth=2, beam_width=4, partial_penalty=0, cross_turn_transition=CrossTurnTransitionConfig(enabled=True, max_ally_turns=2)),
        ).search(snapshot)

        self.assertTrue(any([action.kind for action in node.actions] == [ActionKind.TURN_END, ActionKind.PLAY] for node in result.candidates))
        self.assertEqual(result.trace.cross_turn_transitions, 1)
        self.assertEqual(result.trace.cross_turn_budget_skipped, 0)

    def test_cross_turn_transition_still_respects_max_transitions_per_search_path(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=3, deck_count=5, turn=1, is_first_side=False),
            opponent=PlayerView(life=5, max_life=20, pp=0, max_pp=3, deck_count=5, turn=0, is_first_side=True),
            cards_by_uid={},
            source="test",
            is_ally_turn=True,
        )

        def provider(current):
            if not current.is_ally_turn or current.is_turn_ended:
                return ()
            return (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)

        result = LocalRollbackSearch(
            provider,
            lambda current, _certainty: current.me.turn,
            config=RollbackSearchConfig(max_depth=3, beam_width=4, partial_penalty=0, cross_turn_transition=CrossTurnTransitionConfig(enabled=True, max_transitions_per_search_path=1, max_ally_turns=3)),
        ).search(snapshot)

        self.assertEqual(max(node.cross_turn_transitions for node in result.candidates), 1)
        self.assertEqual(result.trace.cross_turn_transitions, 1)
        self.assertEqual(result.trace.cross_turn_budget_skipped, 1)

    def test_cross_turn_transition_is_not_used_when_disabled(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=3, deck_count=3, turn=1),
            opponent=PlayerView(life=5, max_life=20, pp=0, max_pp=3, deck_count=3, turn=0),
            cards_by_uid={},
            source="test",
            is_ally_turn=True,
        )

        def provider(current):
            if current.is_turn_ended:
                return ()
            return (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)

        result = LocalRollbackSearch(provider, self.score, config=RollbackSearchConfig(max_depth=2, beam_width=4, partial_penalty=0)).search(snapshot)

        self.assertEqual(result.trace.cross_turn_transitions, 0)
        self.assertTrue(result.best is None or all(action.kind is ActionKind.TURN_END for action in result.best.actions))

    def test_partial_policy_can_skip_or_penalize(self):
        snapshot = self.make_snapshot()
        actions = (LegalAction(kind=ActionKind.PLAY, source_uid=1, cost=3),)

        skipped = LocalRollbackSearch(
            lambda _snapshot: actions,
            self.score,
            config=RollbackSearchConfig(max_depth=1, partial_policy=PartialBranchPolicy.SKIP),
        ).search(snapshot)
        self.assertEqual(skipped.trace.partial_skipped, 1)
        self.assertEqual(skipped.trace.terminal_reason, "no_next_candidates")

        penalized = LocalRollbackSearch(
            lambda _snapshot: actions,
            self.score,
            config=RollbackSearchConfig(max_depth=1, partial_policy=PartialBranchPolicy.PENALIZE, partial_penalty=50),
        ).search(snapshot)
        action_nodes = [node for node in penalized.candidates if node.actions]
        self.assertEqual(len(action_nodes), 1)
        self.assertLess(action_nodes[0].score, self.score(action_nodes[0].snapshot, PredictionCertainty.PARTIAL))


if __name__ == "__main__":
    unittest.main()
