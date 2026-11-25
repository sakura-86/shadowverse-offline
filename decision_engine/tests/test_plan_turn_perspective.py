# -*- coding: utf-8 -*-
"""plan_turn 的决策方视角换算（回归门）。

背景：``turn_planner`` 曾用 ``not snapshot.is_ally_turn`` 判断"回合是否结束"。
``is_ally_turn`` 是"**我方**回合"，对手回合时为 False，于是 ``plan_turn`` 对
``player="opponent"`` 永远判成 turn_over、返回空路线——同一策略在 me 侧会攻击/进化，
在 opponent 侧只会打牌。该 bug 导致 plan-vs-plan 24 对全判 candidate 胜（对手没动）。

这些用例锁住换算本身；只要有一个反了，对手侧就会重新空转。
"""

from __future__ import annotations

import unittest

from decision_engine.planning.turn_planner import is_deciding_player_turn


class _Snap:
    def __init__(self, ally_turn: bool) -> None:
        self.is_ally_turn = ally_turn


class DecidingPlayerTurnTest(unittest.TestCase):
    def test_me_on_my_turn(self) -> None:
        self.assertTrue(is_deciding_player_turn(_Snap(True), "me"))

    def test_opponent_on_my_turn(self) -> None:
        self.assertFalse(is_deciding_player_turn(_Snap(True), "opponent"))

    def test_me_on_opponent_turn(self) -> None:
        self.assertFalse(is_deciding_player_turn(_Snap(False), "me"))

    def test_opponent_on_opponent_turn(self) -> None:
        # 修复前这里返回 False，导致 plan_turn 在对手回合永远空转。
        self.assertTrue(is_deciding_player_turn(_Snap(False), "opponent"))

    def test_default_player_is_me(self) -> None:
        self.assertTrue(is_deciding_player_turn(_Snap(True)))
        self.assertFalse(is_deciding_player_turn(_Snap(False)))

    def test_missing_flag_defaults_to_ally_turn(self) -> None:
        class _Bare:
            pass

        self.assertTrue(is_deciding_player_turn(_Bare(), "me"))
        self.assertFalse(is_deciding_player_turn(_Bare(), "opponent"))


if __name__ == "__main__":
    unittest.main()
