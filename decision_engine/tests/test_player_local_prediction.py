# -*- coding: utf-8 -*-
"""Player-aware local prediction tests."""

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.actions.player_local_enumerator import PlayerLocalActionEnumerator
from decision_engine.execution.local_predictor import LocalPredictor
from decision_engine.execution.player_local_predictor import PlayerLocalPredictor
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView


class PlayerLocalPredictionTest(unittest.TestCase):
    def make_snapshot(self):
        return BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, unique_id=10),
            opponent=PlayerView(life=20, max_life=20, pp=2, max_pp=2, hand=(1,), field=(2,), unique_id=20),
            cards_by_uid={
                1: CardView(uid=1, card_id=101, base_card_id=101, name="对手随从", cost=2, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER),
                2: CardView(uid=2, card_id=102, base_card_id=102, name="对手攻击者", atk=3, life=3, max_life=3, can_attack=True, can_attack_leader=True, attacks_left=1),
            },
            is_ally_turn=False,
        )

    def test_enumerates_opponent_actions(self):
        actions = PlayerLocalActionEnumerator().enumerate(self.make_snapshot(), player="opponent")

        self.assertTrue(any(action.kind is ActionKind.PLAY and action.source_uid == 1 for action in actions))
        attack = next(action for action in actions if action.kind is ActionKind.ATTACK_LEADER)
        self.assertEqual(attack.target_groups, ((10,),))

    def test_predicts_opponent_play_in_original_perspective(self):
        snapshot = self.make_snapshot()
        prediction = PlayerLocalPredictor(LocalPredictor()).predict(snapshot, LegalAction(kind=ActionKind.PLAY, source_uid=1, cost=2), player="opponent")

        self.assertEqual(prediction.snapshot.me.hand, ())
        self.assertEqual(prediction.snapshot.opponent.hand, ())
        self.assertEqual(prediction.snapshot.opponent.field, (2, 1))
        self.assertEqual(prediction.snapshot.opponent.pp, 0)
        self.assertIn("player_view_swapped_for_opponent_prediction", prediction.assumptions)
        self.assertEqual(snapshot.opponent.hand, (1,))

    def test_predicts_opponent_attack_leader(self):
        snapshot = self.make_snapshot()
        prediction = PlayerLocalPredictor(LocalPredictor()).predict(snapshot, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=2), player="opponent")

        self.assertEqual(prediction.snapshot.me.life, 17)
        self.assertEqual(prediction.snapshot.opponent.life, 20)
        self.assertEqual(prediction.snapshot.cards_by_uid[2].attacks_left, 0)


if __name__ == "__main__":
    unittest.main()
