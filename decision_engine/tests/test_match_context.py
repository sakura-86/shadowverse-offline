# -*- coding: utf-8 -*-
"""MatchContext / OpponentIdentifier / MetaSnapshot 测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from decision_engine.strategy.match_context import (
    ArchetypeProfile,
    DeckProfile,
    MatchContext,
    OpponentProfile,
)
from decision_engine.strategy.context_manager import MatchContextManager
from decision_engine.strategy.meta_snapshot import MetaSnapshot, MetaSnapshotLoader
from decision_engine.strategy.opponent_identifier import OpponentIdentifier


def _rally_sword() -> ArchetypeProfile:
    return ArchetypeProfile(
        archetype_id="rally_sword",
        class_id=2,
        display_name="连携皇",
        effective_resources=frozenset(["rally", "play_count"]),
        resource_thresholds={"rally": 20},
        key_card_ids=frozenset([10722110, 10724110, 10824120]),
        card_roles={"finisher": frozenset([10824120]), "engine": frozenset([10722110])},
        win_condition="rally_buff_storm",
        aggro_level=0.6,
        estimated_kill_turn=7,
    )


def _needle_dragon() -> ArchetypeProfile:
    return ArchetypeProfile(
        archetype_id="needle_dragon",
        class_id=4,
        display_name="扎针龙",
        effective_resources=frozenset(["max_pp", "ep", "sep"]),
        resource_thresholds={"max_pp": 7},
        key_card_ids=frozenset([10543310, 10042310, 10344120]),
        win_condition="storm_face",
        aggro_level=0.4,
        estimated_kill_turn=8,
    )


class ArchetypeProfileTest(unittest.TestCase):
    def test_frozen_dataclass(self):
        arch = _rally_sword()
        self.assertEqual(arch.archetype_id, "rally_sword")
        self.assertIn("rally", arch.effective_resources)
        self.assertEqual(arch.resource_thresholds["rally"], 20)
        self.assertIn(10824120, arch.card_roles["finisher"])


class DeckProfileTest(unittest.TestCase):
    def test_construction(self):
        deck = DeckProfile(
            class_id=2,
            archetype=_rally_sword(),
            deck_card_ids=frozenset([10722110, 10724110, 10824120, 10021110]),
            deck_card_counts={10722110: 3, 10724110: 2, 10824120: 3, 10021110: 1},
        )
        self.assertEqual(deck.class_id, 2)
        self.assertIn(10722110, deck.deck_card_ids)
        self.assertEqual(deck.deck_card_counts[10724110], 2)


class OpponentProfileTest(unittest.TestCase):
    def test_observe_and_primary(self):
        opp = OpponentProfile(class_id=4)
        self.assertIsNone(opp.primary_archetype)
        self.assertEqual(opp.estimated_kill_turn, 8)

        opp.observe(10543310)
        opp.observe(10543310)
        self.assertEqual(len(opp.observed_card_ids), 1)

        dragon = _needle_dragon()
        opp.likely_archetypes = [(dragon, 0.9)]
        self.assertEqual(opp.primary_archetype, dragon)
        self.assertEqual(opp.estimated_kill_turn, 8)


class MatchContextTest(unittest.TestCase):
    def test_construction(self):
        ctx = MatchContext(
            my_profile=DeckProfile(class_id=2, archetype=_rally_sword()),
            opponent_profile=OpponentProfile(class_id=4),
            turn=5,
        )
        self.assertEqual(ctx.turn, 5)
        self.assertEqual(ctx.my_profile.archetype.archetype_id, "rally_sword")


class MetaSnapshotLoaderTest(unittest.TestCase):
    def test_load_from_json(self):
        data = {
            "archetypes": [
                {
                    "archetype_id": "rally_sword",
                    "class_id": 2,
                    "display_name": "连携皇",
                    "effective_resources": ["rally", "play_count"],
                    "resource_thresholds": {"rally": 20},
                    "key_card_ids": [10722110, 10724110],
                    "card_roles": {"finisher": [10824120], "engine": [10722110, 10724110]},
                    "win_condition": "rally_buff_storm",
                    "aggro_level": 0.6,
                    "estimated_kill_turn": 7,
                },
                {
                    "archetype_id": "needle_dragon",
                    "class_id": 4,
                    "display_name": "扎针龙",
                    "effective_resources": ["max_pp"],
                    "resource_thresholds": {"max_pp": 7},
                    "key_card_ids": [10543310],
                    "win_condition": "storm_face",
                    "aggro_level": 0.4,
                    "estimated_kill_turn": 8,
                },
            ]
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        meta = MetaSnapshotLoader.from_json(path)
        Path(path).unlink()

        self.assertEqual(len(meta.archetypes), 2)
        self.assertIn("rally_sword", meta.archetypes)
        self.assertEqual(meta.archetypes["rally_sword"].class_id, 2)
        self.assertIn(10824120, meta.archetypes["rally_sword"].card_roles["finisher"])

        sword_archs = meta.get_archetypes_for_class(2)
        self.assertEqual(len(sword_archs), 1)

        dragon_archs = meta.get_archetypes_for_class(4)
        self.assertEqual(len(dragon_archs), 1)

    def test_missing_file_returns_empty(self):
        meta = MetaSnapshotLoader.from_json("/nonexistent/path.json")
        self.assertEqual(len(meta.archetypes), 0)

    def test_spellboost_deck_context_loads_roles_and_counts(self):
        manager = MatchContextManager.from_deck_json(
            "decision_engine/data/decks/10008/my_deck_spellboost_rune.json",
            opponent_class_id=3,
            meta_path="decision_engine/strategy/archetype_definitions.json",
        )
        profile = manager.context.my_profile

        self.assertEqual(profile.archetype.archetype_id, "spellboost_rune")
        self.assertIn("spellboost", profile.archetype.effective_resources)
        self.assertEqual(profile.archetype.resource_thresholds["spellboost_for_transcend"], 18)
        self.assertEqual(profile.archetype.resource_thresholds["spellboost_for_tantella_20"], 20)
        self.assertEqual(profile.deck_card_counts[10834110], 3)
        self.assertEqual(profile.deck_card_counts[10534120], 2)
        self.assertIn(10834110, profile.archetype.card_roles["finisher"])


class OpponentIdentifierTest(unittest.TestCase):
    def setUp(self):
        self.meta = MetaSnapshot(
            archetypes={
                "rally_sword": _rally_sword(),
                "needle_dragon": _needle_dragon(),
            },
            class_archetypes={
                2: (_rally_sword(),),
                4: (_needle_dragon(),),
            },
        )
        self.identifier = OpponentIdentifier(self.meta)

    def test_no_observations_gives_equal_prob(self):
        result = self.identifier.identify(2, [])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][1], 1.0)

    def test_matching_key_cards_increases_prob(self):
        result = self.identifier.identify(4, [10543310, 10042310])
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0][1], 0.0)
        self.assertEqual(result[0][0].archetype_id, "needle_dragon")

    def test_unknown_class_returns_empty(self):
        result = self.identifier.identify(99, [10543310])
        self.assertEqual(result, [])

    def test_update_opponent(self):
        opp = OpponentProfile(class_id=4)
        opp.observe(10543310)
        self.identifier.update_opponent(opp)
        self.assertIsNotNone(opp.primary_archetype)
        self.assertEqual(opp.primary_archetype.archetype_id, "needle_dragon")


if __name__ == "__main__":
    unittest.main()
