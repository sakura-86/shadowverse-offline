# -*- coding: utf-8 -*-
"""Opponent observed-card extraction tests."""

import unittest

from decision_engine.state.snapshot import BattleSnapshot, CardIdCount, CardView, PlayerView
from decision_engine.strategy.opponent_observation import extract_opponent_observed_cards


class OpponentObservationTest(unittest.TestCase):
    def test_extracts_visible_and_history_cards_with_sources(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=1),
            opponent=PlayerView(
                life=20,
                max_life=20,
                pp=0,
                max_pp=1,
                hand=(1,),
                field=(2,),
                played_card_ids=(CardIdCount(card_id=101, count=2), CardIdCount(card_id=103, count=1)),
                destroyed_card_ids=(CardIdCount(card_id=102, count=1),),
            ),
            cards_by_uid={
                1: CardView(uid=1, card_id=101, base_card_id=101, name="可见手牌"),
                2: CardView(uid=2, card_id=102, base_card_id=102, name="场上"),
            },
        )

        observed = extract_opponent_observed_cards(snapshot)

        self.assertEqual(observed.card_ids, (101, 102, 103))
        self.assertEqual(observed.sources[101], ("played", "visible_hand"))
        self.assertEqual(observed.sources[102], ("destroyed", "field"))
        self.assertEqual(observed.sources[103], ("played",))
        self.assertEqual(observed.warnings, ())

    def test_missing_visible_uid_is_warning(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=1),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=1, field=(99,)),
            cards_by_uid={},
        )

        observed = extract_opponent_observed_cards(snapshot)

        self.assertEqual(observed.card_ids, ())
        self.assertIn("opponent_observed_uid_missing:field:99", observed.warnings)


if __name__ == "__main__":
    unittest.main()
