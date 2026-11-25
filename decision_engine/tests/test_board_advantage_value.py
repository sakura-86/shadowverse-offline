# -*- coding: utf-8 -*-
"""Tests for BoardAdvantageValue."""

import unittest
from math import tanh

from decision_engine.planning.search_value import BoardAdvantageValue, ValueResult
from decision_engine.state.snapshot import (
    BattleSnapshot,
    CardType,
    CardView,
    LegalView,
    PlayerView,
)


def _follower(uid: int, atk: int, life: int) -> CardView:
    return CardView(uid=uid, atk=atk, life=life, card_type=CardType.FOLLOWER)


def _amulet(uid: int) -> CardView:
    return CardView(uid=uid, atk=0, life=0, card_type=CardType.AMULET, countdown=3)


def _snapshot(
    *,
    my_field_cards: list[CardView] | None = None,
    opp_field_cards: list[CardView] | None = None,
    my_life: int = 20,
    opp_life: int = 20,
) -> BattleSnapshot:
    my_field_cards = my_field_cards or []
    opp_field_cards = opp_field_cards or []
    cards_by_uid: dict[int, CardView] = {}
    my_field: list[int] = []
    opp_field: list[int] = []
    for card in my_field_cards:
        cards_by_uid[card.uid] = card
        my_field.append(card.uid)
    for card in opp_field_cards:
        cards_by_uid[card.uid] = card
        opp_field.append(card.uid)
    return BattleSnapshot(
        me=PlayerView(life=my_life, max_life=20, pp=3, max_pp=3, field=tuple(my_field)),
        opponent=PlayerView(life=opp_life, max_life=20, pp=3, max_pp=3, field=tuple(opp_field)),
        cards_by_uid=cards_by_uid,
        legal=LegalView(),
    )


class TestBoardAdvantageValue(unittest.TestCase):
    def setUp(self):
        self.value = BoardAdvantageValue()

    def test_empty_board_equal_life(self):
        """双方空场等血，评分应为 0。"""
        snap = _snapshot()
        result = self.value.evaluate(snap, player="me")
        self.assertEqual(result.value, 0.0)
        self.assertFalse(result.terminal)
        self.assertEqual(result.source, "board_advantage")

    def test_my_board_only(self):
        """只有我方有场面，评分应为正。"""
        snap = _snapshot(my_field_cards=[_follower(1, 3, 5)])
        result = self.value.evaluate(snap, player="me")
        # 有效威胁 = min(3, 5) = 3
        # board_score = 1.0 * 3 - 2.0 * 0 = 3.0
        # life_score = 0.3 * (20 - 20) = 0
        # raw = 3.0 / 10 = 0.3
        # value = tanh(0.3)
        expected = tanh(3.0 / 10.0)
        self.assertAlmostEqual(result.value, expected, places=6)
        self.assertGreater(result.value, 0.0)

    def test_opp_board_only(self):
        """只有对方有场面，评分应为负，且绝对值更大（2x 权重）。"""
        snap = _snapshot(opp_field_cards=[_follower(1, 3, 5)])
        result = self.value.evaluate(snap, player="me")
        # 有效威胁 = min(3, 5) = 3
        # board_score = 1.0 * 0 - 2.0 * 3 = -6.0
        # raw = -6.0 / 10 = -0.6
        expected = tanh(-6.0 / 10.0)
        self.assertAlmostEqual(result.value, expected, places=6)
        self.assertLess(result.value, 0.0)

    def test_opp_weight_double(self):
        """同等场面时，对方留场的扣分更重。"""
        my_cards = [_follower(1, 3, 3)]
        opp_cards = [_follower(2, 3, 3)]
        snap = _snapshot(my_field_cards=my_cards, opp_field_cards=opp_cards)
        result = self.value.evaluate(snap, player="me")
        # my_threat = 3, opp_threat = 3
        # board_score = 1*3 - 2*3 = -3
        # 对称场面下评分为负，反映"留场对对方更有利"的设计意图
        self.assertLess(result.value, 0.0)

    def test_high_atk_low_life(self):
        """高攻低血随从有效威胁低。"""
        snap = _snapshot(my_field_cards=[_follower(1, 8, 1)])
        result = self.value.evaluate(snap, player="me")
        # 有效威胁 = min(8, 1) = 1
        # board_score = 1.0 * 1 = 1.0
        expected = tanh(1.0 / 10.0)
        self.assertAlmostEqual(result.value, expected, places=6)

    def test_zero_atk_high_life(self):
        """0 攻随从不构成威胁。"""
        snap = _snapshot(my_field_cards=[_follower(1, 0, 5)])
        result = self.value.evaluate(snap, player="me")
        # min(0, 5) 但 atk=0 直接 0
        # board_score = 0
        self.assertEqual(result.value, 0.0)

    def test_amulets_ignored(self):
        """护符不计入场面评分。"""
        snap = _snapshot(my_field_cards=[_amulet(1)], opp_field_cards=[_amulet(2)])
        result = self.value.evaluate(snap, player="me")
        self.assertEqual(result.value, 0.0)

    def test_life_advantage(self):
        """血量差作为辅助信号。"""
        snap = _snapshot(my_life=20, opp_life=10)
        result = self.value.evaluate(snap, player="me")
        # board_score = 0, life_score = 0.3 * (20 - 10) = 3.0
        # raw = 3.0 / 10 = 0.3
        expected = tanh(3.0 / 10.0)
        self.assertAlmostEqual(result.value, expected, places=6)
        self.assertGreater(result.value, 0.0)

    def test_terminal_win(self):
        """对方血量 <= 0 时应返回终局 +1。"""
        snap = _snapshot(opp_life=0)
        result = self.value.evaluate(snap, player="me")
        self.assertEqual(result.value, 1.0)
        self.assertTrue(result.terminal)

    def test_terminal_lose(self):
        """我方血量 <= 0 时应返回终局 -1。"""
        snap = _snapshot(my_life=0)
        result = self.value.evaluate(snap, player="me")
        self.assertEqual(result.value, -1.0)
        self.assertTrue(result.terminal)

    def test_opponent_perspective(self):
        """player=opponent 时视角反转。"""
        snap = _snapshot(my_field_cards=[_follower(1, 4, 4)])
        result_me = self.value.evaluate(snap, player="me")
        result_opp = self.value.evaluate(snap, player="opponent")
        # 对于 opponent，我方场面变成"对方场面"，应该是负面信号
        self.assertGreater(result_me.value, 0.0)
        self.assertLess(result_opp.value, 0.0)

    def test_value_bounded(self):
        """极端场面评分仍在 [-1, 1] 范围内。"""
        # 5 个 10/10 随从
        big_board = [_follower(i, 10, 10) for i in range(5)]
        snap = _snapshot(my_field_cards=big_board)
        result = self.value.evaluate(snap, player="me")
        self.assertLessEqual(result.value, 1.0)
        self.assertGreaterEqual(result.value, -1.0)


if __name__ == "__main__":
    unittest.main()
