# -*- coding: utf-8 -*-
"""Training state feature tests."""

import unittest

from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, CrestView, ExtraPpView, PlayerView
from decision_engine.training.feature_schema import TRAINING_FEATURE_SCHEMA_VERSION
from decision_engine.training.state_features import state_features


class TrainingStateFeaturesTest(unittest.TestCase):
    def test_state_features_encode_snapshot_without_uid_leakage(self) -> None:
        snapshot = BattleSnapshot(
            me=PlayerView(
                life=18,
                max_life=20,
                pp=3,
                max_pp=5,
                ep=1,
                sep=0,
                deck_count=22,
                turn=5,
                hand=(101,),
                field=(201,),
                rally=4,
                play_count=2,
                graveyard_count=7,
                evolve_count=3,
                is_kakusei=True,
                super_evolve_turn=4,
                follower_attacked_leader_previous_turn=True,
                crests=(CrestView(card_id=9001, countdown=2),),
                extra_pp=ExtraPpView(extra_pp=1, preparation_extra_pp=1, extra_pp_state=2),
            ),
            opponent=PlayerView(life=15, max_life=20, pp=2, max_pp=5, hand=(301,), field=(401,), deck_count=20),
            cards_by_uid={
                101: CardView(uid=101, card_id=1001, base_card_id=1000, name="Hand", cost=2, card_type=CardType.SPELL, has_spell_boost=True, spell_boost_count=3, variable_x=6),
                201: CardView(uid=201, card_id=2001, base_card_id=2000, name="Field", cost=3, atk=3, life=4, max_life=4, has_guard=True, card_type=CardType.FOLLOWER),
                301: CardView(uid=301, card_id=3001, base_card_id=3000, name="Opp Hand", cost=1),
                401: CardView(uid=401, card_id=4001, base_card_id=4000, name="Opp Field", cost=2, atk=2, life=2, max_life=2),
            },
            is_ally_turn=True,
        )

        features = state_features(snapshot)

        self.assertEqual(features["schema"], "StateFeatures")
        self.assertEqual(features["version"], TRAINING_FEATURE_SCHEMA_VERSION)
        self.assertEqual(features["player"]["life"], 18)
        self.assertEqual(features["player"]["play_count"], 2)
        self.assertEqual(features["player"]["graveyard_count"], 7)
        self.assertEqual(features["player"]["rally"], 4)
        self.assertEqual(features["player"]["evolve_count"], 3)
        self.assertTrue(features["player"]["is_kakusei"])
        self.assertTrue(features["player"]["super_evolve_unlocked"])
        self.assertTrue(features["player"]["follower_attacked_leader_previous_turn"])
        self.assertTrue(features["player"]["extra_pp_available"])
        self.assertEqual(features["player"]["hand_cards"][0]["variable_x"], 6)
        self.assertEqual(features["player"]["hand_cards"][0]["original_cost"], 2)
        self.assertEqual(features["player"]["hand_cards"][0]["base_card_id"], 1000)
        self.assertEqual(features["player"]["crests"][0]["card_id"], 9001)
        self.assertEqual(features["pressure"]["opponent_field_attack_total"], 2)
        self.assertNotIn("uid", str(features))
        self.assertNotIn("unique_id", str(features))

    def test_state_features_support_opponent_perspective(self) -> None:
        snapshot = BattleSnapshot(
            me=PlayerView(life=10, max_life=20, pp=1, max_pp=1),
            opponent=PlayerView(life=19, max_life=20, pp=4, max_pp=4, turn=4),
            cards_by_uid={},
            is_ally_turn=False,
        )

        features = state_features(snapshot, perspective="opponent")

        self.assertEqual(features["perspective"], "opponent")
        self.assertEqual(features["player"]["life"], 19)
        self.assertEqual(features["opponent"]["life"], 10)


if __name__ == "__main__":
    unittest.main()
