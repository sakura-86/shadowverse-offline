# -*- coding: utf-8 -*-
"""Opponent meta deck pool tests."""

import unittest

from decision_engine.meta.models import CardSlot, DeckEntry
from decision_engine.simulation.opponent_deck_pool import OpponentDeckPoolBuilder, OpponentDeckPoolConfig
from decision_engine.simulation.predictor import PredictionCertainty


class FakeDataSource:
    def __init__(self, decks, latest_set_id=10008):
        self.decks = decks
        self.latest_set_id = latest_set_id

    def load_all_decks(self):
        return self.decks

    def get_latest_set_id(self):
        return self.latest_set_id


class OpponentDeckPoolTest(unittest.TestCase):
    def test_builds_ambiguous_pool_with_adoption_rate_for_class(self):
        decks = [
            DeckEntry("Witch", 10 - index, f"P{index}", f"2026-07-{10 - index:02d}", "", "standard", "增幅", [CardSlot("1001", 3, "A", 2, "法术", 10008), CardSlot("1002", 2 if index < 5 else 1, "B", 3, "随从", 10008)])
            for index in range(10)
        ]
        decks.append(DeckEntry("Royal", 20, "C", "2026-07-11", "", "standard", "皇家", [CardSlot("2001", 3, "C", 1, "随从", 10008)]))
        decks.append(DeckEntry("Witch", 20, "D", "2026-07-11", "", "unlimited", "无限", [CardSlot("9999", 3, "D", 1, "随从", 10001)]))

        result = OpponentDeckPoolBuilder(FakeDataSource(decks)).build_for_class("Witch", OpponentDeckPoolConfig(max_decks_per_class=10))

        self.assertEqual(result.certainty, PredictionCertainty.PARTIAL)
        self.assertIsNotNone(result.deck_pool)
        assert result.deck_pool is not None
        self.assertEqual(result.deck_pool.status, "ambiguous")
        self.assertIn("class=Witch", result.deck_pool.source)
        self.assertIn("sets=10004-10008", result.deck_pool.source)
        self.assertEqual([card.card_id for card in result.deck_pool.cards], [1001, 1002])
        self.assertEqual(result.deck_pool.cards[0].count, 100)
        self.assertEqual(result.deck_pool.cards[1].count, 50)
        self.assertIn("opponent_deck_pool_count_is_adoption_rate_percent", result.assumptions)

    def test_empty_class_is_unknown(self):
        result = OpponentDeckPoolBuilder(FakeDataSource([])).build_for_class("Witch")

        self.assertEqual(result.certainty, PredictionCertainty.UNKNOWN)
        self.assertIsNone(result.deck_pool)
        self.assertIn("opponent_candidate_pool_empty", result.unresolved_effects)

    def test_invalid_card_id_is_warning(self):
        decks = [DeckEntry("Witch", 1, "A", "2026-07-10", "", "standard", "增幅", [CardSlot("bad", 1, "Bad", 1)])]

        result = OpponentDeckPoolBuilder(FakeDataSource(decks)).build_for_class("Witch")

        self.assertEqual(result.certainty, PredictionCertainty.UNKNOWN)
        self.assertIn("opponent_candidate_card_id_invalid:bad", result.warnings)

    def test_standard_uses_latest_five_sets_and_unlimited_uses_all_sets(self):
        decks = [
            DeckEntry("Witch", 10, "A", "2026-07-10", "", "standard", "增幅", [CardSlot("1001", 3, "新卡", 2, "法术", 10008), CardSlot("1002", 3, "旧卡", 2, "法术", 10003)]),
            DeckEntry("Witch", 10, "B", "2026-07-10", "无限", "unlimited", "无限", [CardSlot("1001", 3, "新卡", 2, "法术", 10008), CardSlot("1002", 3, "旧卡", 2, "法术", 10003), CardSlot("1003", 3, "太旧", 2, "法术", 10000)]),
        ]

        standard = OpponentDeckPoolBuilder(FakeDataSource(decks, latest_set_id=10008)).build_for_class("Witch", OpponentDeckPoolConfig(format="standard"))
        unlimited = OpponentDeckPoolBuilder(FakeDataSource(decks, latest_set_id=10008)).build_for_class("Witch", OpponentDeckPoolConfig(format="unlimited"))

        assert standard.deck_pool is not None
        assert unlimited.deck_pool is not None
        self.assertEqual([card.card_id for card in standard.deck_pool.cards], [1001])
        self.assertEqual([card.card_id for card in unlimited.deck_pool.cards], [1001, 1002])
        self.assertIn("opponent_candidate_card_out_of_format:1002:set=10003", standard.warnings)
        self.assertIn("opponent_candidate_card_out_of_format:1003:set=10000", unlimited.warnings)

    def test_build_from_deck_hypotheses_recomputes_pool(self):
        decks = (
            DeckEntry("Witch", 10, "A", "2026-07-10", "", "standard", "增幅", [CardSlot("1001", 3, "A", 2, "法术", 10008), CardSlot("1002", 1, "B", 3, "随从", 10008)]),
            DeckEntry("Witch", 8, "B", "2026-07-09", "", "standard", "增幅", [CardSlot("1001", 3, "A", 2, "法术", 10008)]),
        )

        result = OpponentDeckPoolBuilder(FakeDataSource([], latest_set_id=10008)).build_from_decks(decks, OpponentDeckPoolConfig())

        self.assertEqual(result.certainty, PredictionCertainty.PARTIAL)
        self.assertIn("opponent_candidate_pool_from_deck_hypotheses", result.assumptions)
        self.assertIn("opponent_deck_hypothesis_count=2", result.assumptions)
        assert result.deck_pool is not None
        self.assertIn("opponent_deck_hypotheses", result.deck_pool.source)
        self.assertEqual([card.card_id for card in result.deck_pool.cards], [1001, 1002])
        self.assertEqual(result.deck_pool.cards[0].count, 100)
        self.assertEqual(result.deck_pool.cards[1].count, 17)

    def test_build_from_empty_hypotheses_is_unknown(self):
        result = OpponentDeckPoolBuilder(FakeDataSource([], latest_set_id=10008)).build_from_decks(())

        self.assertEqual(result.certainty, PredictionCertainty.UNKNOWN)
        self.assertIsNone(result.deck_pool)
        self.assertIn("opponent_candidate_pool_empty", result.unresolved_effects)


if __name__ == "__main__":
    unittest.main()
