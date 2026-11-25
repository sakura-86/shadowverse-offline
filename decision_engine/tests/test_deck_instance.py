# -*- coding: utf-8 -*-
"""Deck instance lifecycle tests."""

import unittest
from dataclasses import replace

from decision_engine.simulation.deck_pool import build_deck_instances_from_practice_deck, reconcile_deck_instances_with_snapshot
from decision_engine.state.deck_instance import make_deck_instance_state
from decision_engine.state.snapshot import BattleSnapshot, CardIdCount, CardType, CardView, PlayerView


class DeckInstanceStateTest(unittest.TestCase):
    def test_make_deck_instance_state_creates_precise_instances(self):
        state = make_deck_instance_state((
            CardView(uid=-1, card_id=101, base_card_id=101, name="A", cost=1, card_type=CardType.SPELL),
            CardView(uid=-2, card_id=101, base_card_id=101, name="A", cost=1, card_type=CardType.SPELL),
        ), source="test")

        self.assertEqual(tuple(instance.instance_id for instance in state.instances), (1, 2))
        self.assertEqual(state.draw_pile, (1, 2))
        self.assertEqual(tuple(instance.uid for instance in state.instances), (None, None))
        self.assertEqual(tuple(instance.card.card_id for instance in state.instances), (101, 101))

    def test_move_preserves_instance_dynamic_state(self):
        boosted = CardView(uid=41, card_id=10134310, base_card_id=10134310, name="超越次元", cost=4, card_type=CardType.SPELL, has_spell_boost=True, spell_boost_count=14)
        state = make_deck_instance_state((boosted,), source="test")
        in_hand = state.move(1, "hand", uid=41, card=boosted, visibility="visible")
        returned = in_hand.move(1, "deck", visibility="hidden")
        drawn = returned.move(1, "hand", uid=41, visibility="visible")

        instance = drawn.by_instance_id()[1]
        self.assertEqual(drawn.draw_pile, ())
        self.assertEqual(drawn.hand, (1,))
        self.assertEqual(instance.card.uid, 41)
        self.assertEqual(instance.card.cost, 4)
        self.assertEqual(instance.card.spell_boost_count, 14)

    def test_update_card_updates_visible_instance(self):
        card = CardView(uid=7, card_id=101, base_card_id=101, name="A", cost=3, card_type=CardType.SPELL, has_spell_boost=True, spell_boost_count=2)
        state = make_deck_instance_state((card,), source="test").move(1, "hand", uid=7, card=card, visibility="visible")

        updated = state.update_card(7, replace(card, cost=1, spell_boost_count=4))

        instance = updated.by_uid()[7]
        self.assertEqual(instance.card.cost, 1)
        self.assertEqual(instance.card.spell_boost_count, 4)

    def test_reconcile_deck_instances_with_snapshot_removes_visible_cards_from_draw_pile(self):
        first = CardView(uid=-1, card_id=101, base_card_id=101, name="A", cost=1, card_type=CardType.FOLLOWER)
        second = CardView(uid=-2, card_id=102, base_card_id=102, name="B", cost=2, card_type=CardType.FOLLOWER)
        third = CardView(uid=-3, card_id=103, base_card_id=103, name="C", cost=3, card_type=CardType.FOLLOWER)
        state = make_deck_instance_state((first, second, third), source="test")
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=0, deck_count=1, hand=(10,), played_card_ids=(CardIdCount(card_id=102, count=1),)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0),
            cards_by_uid={10: CardView(uid=10, card_id=101, base_card_id=101, name="A", card_type=CardType.FOLLOWER)},
        )

        reconciled = reconcile_deck_instances_with_snapshot(state, snapshot)

        self.assertEqual(reconciled.status, "available")
        self.assertEqual(reconciled.draw_pile, (3,))

    def test_build_deck_instances_from_practice_deck_expands_counts(self):
        state = build_deck_instances_from_practice_deck({
            "type": "practice_deck_state",
            "data": {
                "user_deck": {
                    "card_count_total": 3,
                    "cards": [
                        {"card_id": 10133310, "count": 2},
                        {"card_id": 10134310, "count": 1},
                    ],
                }
            },
        })

        self.assertEqual(state.source, "practice_deck_state.user_deck")
        self.assertEqual(state.status, "available")
        self.assertEqual(len(state.instances), 3)
        self.assertEqual(state.draw_pile, (1, 2, 3))
        self.assertEqual(tuple(instance.card.card_id for instance in state.instances), (10133310, 10133310, 10134310))


if __name__ == "__main__":
    unittest.main()
