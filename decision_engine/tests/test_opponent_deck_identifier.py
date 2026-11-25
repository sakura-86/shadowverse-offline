# -*- coding: utf-8 -*-
"""Opponent deck identifier tests."""

import unittest

from decision_engine.meta.models import CardSlot, DeckEntry
from decision_engine.state.snapshot import BattleSnapshot, CardIdCount, CardView, PlayerView
from decision_engine.strategy.opponent_deck_identifier import OpponentDeckIdentifier, OpponentDeckIdentifierConfig


class FakeDataSource:
    def __init__(self, decks):
        self.decks = decks

    def load_all_decks(self):
        return self.decks


class OpponentDeckIdentifierTest(unittest.TestCase):
    def make_snapshot(self):
        return BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=1),
            opponent=PlayerView(
                life=20,
                max_life=20,
                pp=0,
                max_pp=1,
                field=(1,),
                played_card_ids=(CardIdCount(card_id=102, count=1),),
            ),
            cards_by_uid={1: CardView(uid=1, card_id=101, base_card_id=101, name="已见")},
        )

    def test_seen_cards_rank_matching_deck_first(self):
        decks = [
            DeckEntry("Witch", 5, "miss", "2026-07-10", "", "standard", "A", [CardSlot("201", 3, "X", 1)]),
            DeckEntry("Witch", 1, "match", "2026-07-01", "", "standard", "B", [CardSlot("101", 3, "A", 1), CardSlot("102", 2, "B", 2)]),
        ]

        hypotheses = OpponentDeckIdentifier(FakeDataSource(decks)).identify(self.make_snapshot(), opponent_class_en="Witch")

        self.assertEqual(hypotheses[0].deck.player_name, "match")
        self.assertEqual(hypotheses[0].matched_card_ids, (101, 102))
        self.assertEqual(hypotheses[0].missing_seen_card_ids, ())
        self.assertEqual(hypotheses[1].missing_seen_card_ids, (101, 102))

    def test_without_seen_cards_uses_prior_and_marks_assumption(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=1),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=1),
            cards_by_uid={},
        )
        decks = [
            DeckEntry("Witch", 1, "low", "2026-07-01", "", "standard", "A", [CardSlot("101", 3, "A", 1)]),
            DeckEntry("Witch", 10, "high", "2026-07-01", "", "standard", "B", [CardSlot("102", 3, "B", 1)]),
        ]

        hypotheses = OpponentDeckIdentifier(FakeDataSource(decks)).identify(snapshot, opponent_class_en="Witch")

        self.assertEqual(hypotheses[0].deck.player_name, "high")
        self.assertIn("opponent_deck_identification_without_seen_cards", hypotheses[0].assumptions)

    def test_filters_class_and_format(self):
        decks = [
            DeckEntry("Royal", 20, "sword", "2026-07-01", "", "standard", "A", [CardSlot("101", 3, "A", 1)]),
            DeckEntry("Witch", 20, "unlimited", "2026-07-01", "", "unlimited", "B", [CardSlot("101", 3, "A", 1)]),
            DeckEntry("Witch", 1, "standard", "2026-07-01", "", "standard", "C", [CardSlot("101", 3, "A", 1)]),
        ]

        hypotheses = OpponentDeckIdentifier(FakeDataSource(decks)).identify(self.make_snapshot(), opponent_class_en="Witch", config=OpponentDeckIdentifierConfig(format="standard"))

        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(hypotheses[0].deck.player_name, "standard")


if __name__ == "__main__":
    unittest.main()
