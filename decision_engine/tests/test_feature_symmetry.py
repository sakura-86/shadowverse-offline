# -*- coding: utf-8 -*-
"""Tests for perspective symmetry in feature encoding.

验证 state_features 和 action_features 在 perspective='me' 和 perspective='opponent'
下产生正确对称的特征。
"""

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, LegalView, PlayerView
from decision_engine.training.action_features import action_features, legal_actions_features
from decision_engine.training.state_features import state_features


def _snapshot() -> BattleSnapshot:
    """构造一个非对称 snapshot（双方状态不同，便于验证翻转）。"""
    my_hand = CardView(uid=1, card_id=100, base_card_id=100, cost=2, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER)
    my_field = CardView(uid=2, card_id=200, base_card_id=200, atk=3, life=4, max_life=4, card_type=CardType.FOLLOWER, can_attack=True, attacks_left=1)
    opp_field = CardView(uid=10, card_id=1000, base_card_id=1000, atk=5, life=5, max_life=5, card_type=CardType.FOLLOWER, has_guard=True)
    opp_hand = CardView(uid=11, card_id=1100, base_card_id=1100, cost=3, card_type=CardType.SPELL)

    return BattleSnapshot(
        me=PlayerView(life=18, max_life=20, pp=5, max_pp=5, ep=1, hand=(1,), field=(2,), deck_count=15, turn=4, unique_id=1),
        opponent=PlayerView(life=12, max_life=20, pp=3, max_pp=7, ep=2, hand=(11,), field=(10,), deck_count=8, turn=3, unique_id=2),
        cards_by_uid={1: my_hand, 2: my_field, 10: opp_field, 11: opp_hand},
        is_ally_turn=True,
        legal=LegalView(can_play=frozenset({1}), can_attack_leader=frozenset({2})),
    )


class TestStateFeaturesPerspective(unittest.TestCase):

    def test_me_perspective_player_is_me(self):
        """perspective='me' 时 player 应该对应 snapshot.me。"""
        snapshot = _snapshot()
        features = state_features(snapshot, perspective="me")
        self.assertEqual(features["player"]["life"], 18)
        self.assertEqual(features["opponent"]["life"], 12)

    def test_opponent_perspective_player_is_opponent(self):
        """perspective='opponent' 时 player 应该对应 snapshot.opponent。"""
        snapshot = _snapshot()
        features = state_features(snapshot, perspective="opponent")
        self.assertEqual(features["player"]["life"], 12)
        self.assertEqual(features["opponent"]["life"], 18)

    def test_is_ally_turn_flipped_for_opponent(self):
        """perspective='opponent' 时 is_ally_turn 应翻转。"""
        snapshot = _snapshot()  # is_ally_turn=True (我方回合)
        me_features = state_features(snapshot, perspective="me")
        opp_features = state_features(snapshot, perspective="opponent")

        self.assertTrue(me_features["is_ally_turn"])  # 我方回合，我方视角=True
        self.assertFalse(opp_features["is_ally_turn"])  # 我方回合，对方视角=False

    def test_legal_summary_hidden_for_opponent(self):
        """perspective='opponent' 时 legal_summary 应为 present=False。"""
        snapshot = _snapshot()
        me_features = state_features(snapshot, perspective="me")
        opp_features = state_features(snapshot, perspective="opponent")

        self.assertTrue(me_features["legal_summary"]["present"])
        self.assertFalse(opp_features["legal_summary"]["present"])

    def test_symmetry_player_opponent_swap(self):
        """perspective 翻转后，player 和 opponent 的内容应完全互换。"""
        snapshot = _snapshot()
        me_features = state_features(snapshot, perspective="me")
        opp_features = state_features(snapshot, perspective="opponent")

        # me 视角的 player.life == opponent 视角的 opponent.life
        self.assertEqual(me_features["player"]["life"], opp_features["opponent"]["life"])
        self.assertEqual(me_features["opponent"]["life"], opp_features["player"]["life"])
        # PP
        self.assertEqual(me_features["player"]["pp"], opp_features["opponent"]["pp"])
        self.assertEqual(me_features["opponent"]["pp"], opp_features["player"]["pp"])
        # hand_count
        self.assertEqual(me_features["player"]["hand_count"], opp_features["opponent"]["hand_count"])

    def test_pressure_features_flipped(self):
        """pressure 特征也应随 perspective 翻转。"""
        snapshot = _snapshot()
        me_features = state_features(snapshot, perspective="me")
        opp_features = state_features(snapshot, perspective="opponent")

        # 我方视角的 player_field_attack_total == 对方视角的 opponent_field_attack_total
        self.assertEqual(
            me_features["pressure"]["player_field_attack_total"],
            opp_features["pressure"]["opponent_field_attack_total"],
        )


class TestActionFeaturesPerspective(unittest.TestCase):

    def test_source_zone_me_hand_for_me_perspective(self):
        """me 视角下，source 在 me.hand 中应标记为 me_hand。"""
        snapshot = _snapshot()
        action = LegalAction(kind=ActionKind.PLAY, source_uid=1, cost=2)
        features = action_features(snapshot, action, perspective="me")
        self.assertEqual(features["source"]["zone"], "me_hand")

    def test_source_zone_flipped_for_opponent_perspective(self):
        """opponent 视角下，source 在 snapshot.opponent.hand（= perspective 的 me）应标记为 me_hand。"""
        snapshot = _snapshot()
        # 对方手牌 uid=11，从对方视角看是 me_hand
        action = LegalAction(kind=ActionKind.PLAY, source_uid=11, cost=3)
        features = action_features(snapshot, action, perspective="opponent")
        self.assertEqual(features["source"]["zone"], "me_hand")

    def test_target_side_flipped_for_opponent_perspective(self):
        """opponent 视角下，target 在 snapshot.me.field 应标记为 opponent。"""
        snapshot = _snapshot()
        # 对方打我方场上随从 uid=2
        action = LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=10, target_groups=((2,),))
        features = action_features(snapshot, action, perspective="opponent")
        self.assertEqual(features["source"]["zone"], "me_field")  # uid=10 在 opp.field = perspective's me_field
        self.assertEqual(features["targets"][0]["side"], "opponent")  # uid=2 在 me.field = perspective's opponent
        self.assertEqual(features["targets"][0]["zone"], "opponent_field")

    def test_pp_before_action_uses_perspective(self):
        """pp_before_action 应使用 perspective 对应的 PP。"""
        snapshot = _snapshot()
        action = LegalAction(kind=ActionKind.PLAY, source_uid=11, cost=3)
        me_features = action_features(snapshot, action, perspective="me")
        opp_features = action_features(snapshot, action, perspective="opponent")

        self.assertEqual(me_features["pp_before_action"], 5)  # me.pp
        self.assertEqual(opp_features["pp_before_action"], 3)  # opponent.pp

    def test_legal_actions_features_batch(self):
        """legal_actions_features 批量接口应正确传递 perspective。"""
        snapshot = _snapshot()
        actions = [
            LegalAction(kind=ActionKind.PLAY, source_uid=11, cost=3),
            LegalAction(kind=ActionKind.TURN_END, source_uid=0),
        ]
        features = legal_actions_features(snapshot, actions, perspective="opponent")
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0]["pp_before_action"], 3)
        self.assertEqual(features[0]["source"]["zone"], "me_hand")  # uid=11 in opp.hand = perspective's me_hand


if __name__ == "__main__":
    unittest.main()
