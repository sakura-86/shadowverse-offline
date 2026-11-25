# -*- coding: utf-8 -*-
"""Deck pool parsing tests."""

import unittest

from decision_engine.simulation.deck_pool import DeckPoolError, parse_practice_deck_pool, reconcile_deck_pool_with_snapshot
from decision_engine.state.snapshot import BattleSnapshot, CardIdCount, CardType, CardView, PlayerView


class DeckPoolTest(unittest.TestCase):
    def test_parse_practice_deck_pool(self):
        pool = parse_practice_deck_pool({
            "type": "practice_deck_state",
            "data": {
                "user_deck": {
                    "card_count_total": 3,
                    "cards": [
                        {"card_id": 101, "count": 2},
                        {"card_id": 102, "count": 1},
                    ],
                }
            },
        })

        self.assertEqual(pool.source, "practice_deck_state.user_deck")
        self.assertEqual(pool.observed_total, 3)
        self.assertEqual(pool.remaining_count, 3)
        self.assertEqual(pool.status, "available")
        self.assertEqual(tuple((card.card_id, card.count) for card in pool.cards), ((101, 2), (102, 1)))

    def test_parse_practice_deck_pool_accepts_direct_user_deck_payload(self):
        pool = parse_practice_deck_pool({
            "user_deck": {
                "card_count_total": 2,
                "cards": [{"card_id": 201, "count": 2}],
            }
        })

        self.assertEqual(pool.source, "direct.user_deck")
        self.assertEqual(pool.observed_total, 2)
        self.assertEqual(pool.remaining_count, 2)
        self.assertEqual(tuple((card.card_id, card.count) for card in pool.cards), ((201, 2),))

    def test_parse_practice_deck_pool_rejects_wrong_type(self):
        with self.assertRaises(DeckPoolError):
            parse_practice_deck_pool({"type": "action_request", "data": {}})

    def test_reconcile_deck_pool_with_snapshot(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=3, deck_count=2, hand=(1,), field=(2,), played_card_ids=(CardIdCount(card_id=101, count=1),), destroyed_card_ids=(CardIdCount(card_id=102, count=1),)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=3, deck_count=3),
            cards_by_uid={1: CardView(uid=1, card_id=101, base_card_id=101, name="A", cost=1, card_type=CardType.FOLLOWER), 2: CardView(uid=2, card_id=102, base_card_id=102, name="B", cost=2, card_type=CardType.SPELL)},
        )
        pool = parse_practice_deck_pool({
            "type": "practice_deck_state",
            "data": {"user_deck": {"card_count_total": 3, "cards": [{"card_id": 101, "count": 2}, {"card_id": 102, "count": 1}]}}
        })

        reconciled = reconcile_deck_pool_with_snapshot(pool, snapshot)

        self.assertEqual(reconciled.remaining_count, 0)
        self.assertEqual(reconciled.status, "ambiguous")
        self.assertIn("deck_pool_count_mismatch", reconciled.warnings)
