# -*- coding: utf-8 -*-
"""无事可做时的 PP 花完顺序（纯函数，不跑对局）。"""

from __future__ import annotations

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.turn_plan_arm import rank_burn_pp_candidates


def _play(cost: int | None, uid: int = 1) -> LegalAction:
    return LegalAction(kind=ActionKind.PLAY, source_uid=uid, cost=cost)


class RankBurnPpTest(unittest.TestCase):
    def test_highest_cost_first(self) -> None:
        actions = (_play(1), _play(5), _play(3))
        self.assertEqual(rank_burn_pp_candidates(actions, pp=10), [1, 2, 0])

    def test_only_affordable(self) -> None:
        actions = (_play(1), _play(5), _play(3))
        # pp=4：5 费打不出，剩下 3 费、1 费按降序。
        self.assertEqual(rank_burn_pp_candidates(actions, pp=4), [2, 0])

    def test_equal_cost_keeps_enumeration_order(self) -> None:
        actions = (_play(2, uid=1), _play(2, uid=2), _play(2, uid=3))
        self.assertEqual(rank_burn_pp_candidates(actions, pp=2), [0, 1, 2])

    def test_none_cost_is_treated_as_zero(self) -> None:
        actions = (_play(None), _play(2))
        self.assertEqual(rank_burn_pp_candidates(actions, pp=0), [0])
        self.assertEqual(rank_burn_pp_candidates(actions, pp=2), [1, 0])

    def test_non_play_actions_excluded(self) -> None:
        actions = (
            LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=1),
            _play(3),
            LegalAction(kind=ActionKind.TURN_END, source_uid=0),
        )
        self.assertEqual(rank_burn_pp_candidates(actions, pp=3), [1])

    def test_zero_pp_yields_only_free_cards(self) -> None:
        actions = (_play(1), _play(0), _play(2))
        self.assertEqual(rank_burn_pp_candidates(actions, pp=0), [1])

    def test_empty_actions(self) -> None:
        self.assertEqual(rank_burn_pp_candidates((), pp=5), [])


if __name__ == "__main__":
    unittest.main()
