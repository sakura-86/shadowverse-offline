# -*- coding: utf-8 -*-
"""起手留牌策略测试。"""

import unittest

from decision_engine.strategy.mulligan import (
    ARO_TRANSFORM,
    DIMENSION_SHIFT,
    HOMEWORK,
    REUNION,
    STEEL_REST,
    TANTELLA,
    VERDANT_GRACE,
    AETHER,
    WILBERT,
    JEANNE,
    GALLEON,
    SOFINA,
    MAVEN,
    MulliganCard,
    MulliganConfig,
    GuardHavenMulliganStrategy,
    SpellboostRuneMulliganStrategy,
    build_mulligan_command,
    extract_mulligan_hand,
)

KUON = 10134110
ARO = 10534120


def _card(uid: int, card_id: int, base_card_id: int | None = None) -> dict:
    return {
        "unique_id": uid,
        "card_id": card_id,
        "base_card_id": base_card_id if base_card_id is not None else card_id,
    }


def _state(cards: list[dict]) -> dict:
    return {
        "type": "state_update",
        "data": {
            "players": [
                {
                    "hand_cards": cards,
                }
            ]
        },
    }


def _mcard(uid: int, card_id: int, base_card_id: int | None = None) -> MulliganCard:
    return MulliganCard(unique_id=uid, card_id=card_id, base_card_id=base_card_id or card_id)


class ExtractMulliganHandTest(unittest.TestCase):
    def test_extracts_unique_id_and_uid(self):
        state = _state([
            _card(1, TANTELLA),
            {"uid": 2, "card_id": STEEL_REST, "base_card_id": STEEL_REST},
            {"card_id": HOMEWORK, "base_card_id": HOMEWORK},
        ])
        hand = extract_mulligan_hand(state)
        self.assertEqual([card.unique_id for card in hand], [1, 2])
        self.assertEqual([card.card_id for card in hand], [TANTELLA, STEEL_REST])

    def test_non_state_update_returns_empty(self):
        self.assertEqual(extract_mulligan_hand({"type": "practice_deck_state", "data": {}}), [])


class SpellboostRuneMulliganStrategyTest(unittest.TestCase):
    def test_keeps_tantella_and_changes_other_high_cost_cards(self):
        hand = [_mcard(10, TANTELLA), _mcard(11, KUON), _mcard(12, ARO)]
        decision = SpellboostRuneMulliganStrategy().decide(hand)
        self.assertIn(10, decision.cards_to_keep)
        self.assertIn(11, decision.cards_to_change)
        self.assertIn(12, decision.cards_to_change)
        self.assertEqual(decision.card_scores[10], 100.0)

    def test_keeps_one_draw2_resource_above_reunion(self):
        hand = [_mcard(20, STEEL_REST), _mcard(21, HOMEWORK), _mcard(22, REUNION)]
        decision = SpellboostRuneMulliganStrategy().decide(hand)
        self.assertIn(20, decision.cards_to_keep)
        self.assertIn(21, decision.cards_to_change)
        self.assertIn(22, decision.cards_to_change)
        self.assertGreater(decision.card_scores[20], decision.card_scores[22])

    def test_keeps_dimension_with_tantella(self):
        hand = [_mcard(30, TANTELLA), _mcard(31, DIMENSION_SHIFT)]
        decision = SpellboostRuneMulliganStrategy().decide(hand)
        self.assertIn(30, decision.cards_to_keep)
        self.assertIn(31, decision.cards_to_keep)
        self.assertGreaterEqual(decision.card_scores[31], 70.0)

    def test_changes_dimension_without_terminal_or_resource(self):
        hand = [_mcard(40, DIMENSION_SHIFT), _mcard(41, KUON)]
        decision = SpellboostRuneMulliganStrategy().decide(hand)
        self.assertIn(40, decision.cards_to_change)
        self.assertLess(decision.card_scores[40], 50.0)

    def test_keeps_dimension_with_draw2_resource_by_default(self):
        hand = [_mcard(50, DIMENSION_SHIFT), _mcard(51, STEEL_REST)]
        decision = SpellboostRuneMulliganStrategy().decide(hand)
        self.assertIn(51, decision.cards_to_keep)
        self.assertIn(50, decision.cards_to_keep)
        self.assertGreater(decision.card_scores[50], 50.0)

    def test_consumable_spellboost_depends_on_terminal(self):
        no_terminal = [_mcard(60, REUNION), _mcard(61, VERDANT_GRACE)]
        decision = SpellboostRuneMulliganStrategy().decide(no_terminal)
        self.assertIn(60, decision.cards_to_change)
        self.assertIn(61, decision.cards_to_change)

        with_terminal = [_mcard(62, TANTELLA), _mcard(63, REUNION), _mcard(64, VERDANT_GRACE)]
        decision = SpellboostRuneMulliganStrategy().decide(with_terminal)
        self.assertIn(62, decision.cards_to_keep)
        kept_consumables = {63, 64} & set(decision.cards_to_keep)
        self.assertEqual(len(kept_consumables), 1)

    def test_full_redraw_without_terminal_changes_everything(self):
        hand = [_mcard(70, STEEL_REST), _mcard(71, DIMENSION_SHIFT), _mcard(72, REUNION)]
        default_decision = SpellboostRuneMulliganStrategy().decide(hand)
        self.assertIn(70, default_decision.cards_to_keep)

        aggressive = SpellboostRuneMulliganStrategy(MulliganConfig(full_redraw_without_terminal=True)).decide(hand)
        self.assertEqual(set(aggressive.cards_to_change), {70, 71, 72})
        self.assertEqual(aggressive.cards_to_keep, [])

    def test_matches_base_card_id(self):
        hand = [_mcard(80, 70134111, TANTELLA), _mcard(81, 10833311, STEEL_REST)]
        decision = SpellboostRuneMulliganStrategy().decide(hand)
        self.assertIn(80, decision.cards_to_keep)
        self.assertIn(81, decision.cards_to_keep)


class GuardHavenMulliganStrategyTest(unittest.TestCase):
    def test_keeps_aether_and_early_curve(self):
        hand = [
            _mcard(100, AETHER),
            _mcard(101, GALLEON),
            _mcard(102, SOFINA),
            _mcard(103, JEANNE),
        ]
        decision = GuardHavenMulliganStrategy().decide(hand)
        self.assertIn(100, decision.cards_to_keep)
        self.assertIn(101, decision.cards_to_keep)
        self.assertIn(102, decision.cards_to_keep)
        self.assertIn(103, decision.cards_to_change)
        self.assertEqual(decision.strategy_id, "guard_haven")
        self.assertGreater(decision.card_scores[100], decision.card_scores[103])

    def test_keeps_wilbert_with_aether_but_not_as_naked_priority(self):
        supported = [
            _mcard(110, AETHER),
            _mcard(111, WILBERT),
            _mcard(112, GALLEON),
        ]
        naked = [
            _mcard(120, WILBERT),
            _mcard(121, JEANNE),
            _mcard(122, MAVEN),
        ]
        supported_decision = GuardHavenMulliganStrategy().decide(supported)
        naked_decision = GuardHavenMulliganStrategy().decide(naked)
        self.assertIn(110, supported_decision.cards_to_keep)
        self.assertIn(111, supported_decision.cards_to_keep)
        self.assertIn(120, naked_decision.cards_to_change)
        self.assertLess(naked_decision.card_scores[120], supported_decision.card_scores[111])

    def test_changes_heavy_late_cards_without_core_or_curve(self):
        hand = [
            _mcard(130, JEANNE),
            _mcard(131, MAVEN),
            _mcard(132, 10164120),
        ]
        decision = GuardHavenMulliganStrategy().decide(hand)
        self.assertEqual(decision.cards_to_keep, [])
        self.assertEqual(set(decision.cards_to_change), {130, 131, 132})


class BuildMulliganCommandTest(unittest.TestCase):
    def test_uses_unique_ids_not_card_ids(self):
        state = _state([
            _card(90, TANTELLA),
            _card(91, KUON),
            _card(92, DIMENSION_SHIFT),
        ])
        command, decision = build_mulligan_command(state, SpellboostRuneMulliganStrategy())
        self.assertEqual(command["action"], "mulligan")
        self.assertTrue(command["force_execute"])
        self.assertEqual(command["cards_to_change"], [91])
        self.assertIsNotNone(decision)
        self.assertNotIn(KUON, command["cards_to_change"])

    def test_guard_haven_command_uses_guard_haven_strategy(self):
        state = _state([
            _card(140, AETHER),
            _card(141, WILBERT),
            _card(142, JEANNE),
        ])
        command, decision = build_mulligan_command(state, GuardHavenMulliganStrategy())
        self.assertEqual(command["action"], "mulligan")
        self.assertTrue(command["force_execute"])
        self.assertIsNotNone(decision)
        self.assertEqual(decision.strategy_id, "guard_haven")

    def test_empty_hand_falls_back_to_empty_mulligan(self):
        command, decision = build_mulligan_command({"type": "state_update", "data": {}})
        self.assertEqual(command["cards_to_change"], [])
        self.assertIsNone(decision)


if __name__ == "__main__":
    unittest.main()
