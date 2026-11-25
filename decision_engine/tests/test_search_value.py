# -*- coding: utf-8 -*-
"""SearchValue tests."""

from __future__ import annotations

import unittest

from decision_engine.planning.search_value import TerminalValue, ValueResult
from decision_engine.state.snapshot import BattleSnapshot, PlayerView


class SearchValueTest(unittest.TestCase):
    def test_terminal_value_returns_neutral_non_terminal_baseline(self) -> None:
        result = TerminalValue().evaluate(_snapshot(), player="me")

        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.source, "non_terminal_zero_baseline")
        self.assertFalse(result.terminal)

    def test_terminal_value_uses_player_perspective_for_terminal_states(self) -> None:
        me_win = TerminalValue().evaluate(_snapshot(me_life=20, opponent_life=0), player="me")
        opponent_win = TerminalValue().evaluate(_snapshot(me_life=0, opponent_life=20), player="me")

        self.assertEqual(me_win.value, 1.0)
        self.assertEqual(opponent_win.value, -1.0)
        self.assertTrue(me_win.terminal)
        self.assertTrue(opponent_win.terminal)

    def test_value_result_validation_rejects_out_of_range_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "in \[-1.0, 1.0\]"):
            ValueResult(value=1.5, source="x").validate()


def _snapshot(*, me_life: int = 20, opponent_life: int = 20) -> BattleSnapshot:
    return BattleSnapshot(
        me=PlayerView(life=me_life, max_life=20, pp=1, max_pp=1),
        opponent=PlayerView(life=opponent_life, max_life=20, pp=1, max_pp=1),
        cards_by_uid={},
    )


if __name__ == "__main__":
    unittest.main()
