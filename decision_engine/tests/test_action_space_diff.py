# -*- coding: utf-8 -*-
"""本地/协议动作空间 diff 测试。"""

import unittest

from decision_engine.actions.action_space_diff import diff_local_protocol_actions
from decision_engine.actions.model import ActionKind
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, LegalView, PlayerView


def _snapshot(*, legal, pp=2, hand=(1,), field=(), opponent_field=()):
    cards = {1: CardView(uid=1, cost=2, card_type=CardType.FOLLOWER)}
    cards.update({uid: CardView(uid=uid, atk=1, life=1) for uid in field})
    cards.update({uid: CardView(uid=uid, atk=1, life=1) for uid in opponent_field})
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=pp, max_pp=pp, hand=hand, field=field),
        opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=pp, unique_id=99, field=opponent_field),
        cards_by_uid=cards,
        legal=legal,
        is_ally_turn=True,
    )


class ActionSpaceDiffTest(unittest.TestCase):
    def test_reports_matched_local_and_protocol_actions(self):
        snapshot = _snapshot(legal=LegalView(can_play=frozenset({1}), can_end_turn=True))

        report = diff_local_protocol_actions(snapshot)

        self.assertEqual(report.local_count, 2)
        self.assertEqual(report.protocol_count, 2)
        self.assertEqual(report.missing, ())
        self.assertEqual(report.extra, ())
        self.assertEqual({action.kind for action in report.matched}, {ActionKind.PLAY, ActionKind.TURN_END})

    def test_reports_missing_protocol_action_without_local_fallback(self):
        snapshot = _snapshot(legal=LegalView(can_play=frozenset({1}), can_activate=frozenset({2}), can_end_turn=True), field=(2,))

        report = diff_local_protocol_actions(snapshot)

        self.assertEqual([(action.kind, action.source_uid) for action in report.missing], [(ActionKind.ACTIVATE, 2)])
        self.assertTrue(all(action.legality_source == "protocol" for action in report.missing))

    def test_reports_extra_local_action_without_protocol_play_list(self):
        snapshot = _snapshot(legal=LegalView())

        report = diff_local_protocol_actions(snapshot)

        self.assertEqual([(action.kind, action.source_uid) for action in report.extra], [(ActionKind.PLAY, 1), (ActionKind.TURN_END, 0)])
        self.assertTrue(all(action.legality_source == "local" for action in report.extra))


if __name__ == "__main__":
    unittest.main()
