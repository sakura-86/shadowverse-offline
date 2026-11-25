# -*- coding: utf-8 -*-
"""Training action feature tests."""

import unittest

from decision_engine.actions.model import ActionKind, LegalAction, PlayKind
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView
from decision_engine.training.action_features import action_features, legal_actions_features
from decision_engine.training.tensorization import ACTION_VECTOR_FEATURE_ORDER, action_vectors_from_action_features


class TrainingActionFeaturesTest(unittest.TestCase):
    def test_action_features_encode_source_target_without_uid_leakage(self) -> None:
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=3, max_pp=3, hand=(10,), field=(11,)),
            opponent=PlayerView(life=20, max_life=20, pp=2, max_pp=2, field=(20,)),
            cards_by_uid={
                10: CardView(uid=10, card_id=1001, base_card_id=1000, name="Spell", cost=3, card_type=CardType.SPELL),
                11: CardView(uid=11, card_id=1101, base_card_id=1100, name="Follower", cost=2, atk=2, life=2, max_life=2, can_attack=True),
                20: CardView(uid=20, card_id=2001, base_card_id=2000, name="Guard", cost=2, atk=1, life=3, max_life=3, has_guard=True),
            },
        )
        action = LegalAction(kind=ActionKind.PLAY, source_uid=10, play_kind=PlayKind.NORMAL, cost=3, target_groups=((20,),), selected_indexes=(1,))

        features = action_features(snapshot, action)

        self.assertEqual(features["schema"], "ActionFeatures")
        self.assertEqual(features["action_kind"], "play")
        self.assertTrue(features["uses_full_pp"])
        self.assertEqual(features["source"]["zone"], "me_hand")
        self.assertEqual(features["source"]["base_card_id"], 1000)
        self.assertEqual(features["targets"][0]["side"], "opponent")
        self.assertEqual(features["targets"][0]["base_card_id"], 2000)
        self.assertEqual(features["target_count"], 1)
        self.assertEqual(features["selected_index_count"], 1)
        self.assertNotIn("source_uid", str(features))
        self.assertNotIn("target_uid", str(features))
        self.assertNotIn("uid", str(features))

    def test_legal_actions_features_preserve_order(self) -> None:
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(11,)),
            opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1),
            cards_by_uid={11: CardView(uid=11, card_id=1101, base_card_id=1100, cost=1, atk=1, life=1, max_life=1)},
        )
        actions = [
            LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=11),
            LegalAction(kind=ActionKind.TURN_END, source_uid=0),
        ]

        features = legal_actions_features(snapshot, actions)

        self.assertEqual([item["action_kind"] for item in features], ["attack_leader", "turn_end"])
        self.assertTrue(features[0]["is_combat_action"])
        self.assertTrue(features[1]["is_terminal_action"])

    def test_play_kind_is_encoded_as_symmetric_one_hot_features(self) -> None:
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=10, max_pp=10, hand=(10,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0),
            cards_by_uid={10: CardView(uid=10, card_id=1001, base_card_id=1000, cost=1)},
        )
        expected = {
            PlayKind.NORMAL: "is_normal_play",
            PlayKind.ENHANCE: "is_enhance",
            PlayKind.ACCELERATE: "is_accelerate",
            PlayKind.CRYSTAL: "is_crystal",
        }
        for play_kind, active_key in expected.items():
            with self.subTest(play_kind=play_kind):
                features = action_features(
                    snapshot,
                    LegalAction(kind=ActionKind.PLAY, source_uid=10, play_kind=play_kind, cost=1),
                )
                self.assertEqual(features["play_kind"], play_kind.value)
                self.assertNotIn("play_kind", ACTION_VECTOR_FEATURE_ORDER)
                vector = dict(zip(ACTION_VECTOR_FEATURE_ORDER, action_vectors_from_action_features([features])[0]))
                for key in expected.values():
                    self.assertEqual(features[key], key == active_key)
                    self.assertEqual(vector[key], float(key == active_key))

    def test_action_source_reuses_full_card_feature_contract(self) -> None:
        """动作 source 的 card stat 必须和 state hand token 一样看见 X 与原始费用。"""
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=2, max_pp=2, hand=(10,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0),
            cards_by_uid={
                # 10662210 的印刷费用为2；模拟经降费后的当前费用0。
                10: CardView(uid=10, card_id=10662210, base_card_id=10662210, cost=0, variable_x=7),
            },
        )
        source = action_features(snapshot, LegalAction(kind=ActionKind.PLAY, source_uid=10, cost=0))["source"]
        self.assertEqual(source["variable_x"], 7)
        self.assertEqual(source["original_cost"], 2)


if __name__ == "__main__":
    unittest.main()

