# -*- coding: utf-8 -*-
"""Oracle interface skeleton 测试。"""

import unittest

from decision_engine.actions.model import ActionKind, LegalAction, PlannedAction, PlayKind
from decision_engine.simulation import MockOracle, Oracle, action_key
from decision_engine.state.snapshot import BattleSnapshot, PlayerView


def _snapshot(life=20, source="test"):
    return BattleSnapshot(
        me=PlayerView(life=life, max_life=20, pp=10, max_pp=10),
        opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0),
        cards_by_uid={},
        source=source,
    )


class OracleInterfaceTest(unittest.TestCase):
    def test_current_returns_initial_snapshot(self):
        snapshot = _snapshot(life=17)
        oracle = MockOracle(snapshot)

        self.assertIs(oracle.current(), snapshot)
        self.assertIsInstance(oracle, Oracle)

    def test_apply_transition_updates_current_snapshot(self):
        start = _snapshot(life=20, source="start")
        after = _snapshot(life=18, source="after")
        action = PlannedAction(kind=ActionKind.PLAY, source_uid=101, play_kind=PlayKind.NORMAL)
        oracle = MockOracle(start, {action_key(action): after})

        self.assertIs(oracle.apply(action), after)
        self.assertIs(oracle.current(), after)

    def test_unmatched_action_raises_clear_exception_without_simulation(self):
        action = PlannedAction(kind=ActionKind.PLAY, source_uid=101, play_kind=PlayKind.NORMAL)
        oracle = MockOracle(_snapshot())

        with self.assertRaisesRegex(KeyError, "MockOracle has no transition"):
            oracle.apply(action)

    def test_supports_legal_action_and_planned_action_with_same_key(self):
        start = _snapshot(source="start")
        after_attack = _snapshot(life=19, source="after_attack")
        legal_action = LegalAction(
            kind=ActionKind.ATTACK_FOLLOWER,
            source_uid=201,
            target_groups=((301,),),
        )
        planned_action = PlannedAction(
            kind=ActionKind.ATTACK_FOLLOWER,
            source_uid=201,
            target_unique_id_by_skills=((301,),),
        )
        oracle = MockOracle(start, [(legal_action, after_attack)])

        self.assertEqual(action_key(legal_action), action_key(planned_action))
        self.assertIs(oracle.apply(planned_action), after_attack)
        self.assertIs(oracle.current(), after_attack)

    def test_action_key_includes_materials_and_play_kind(self):
        normal = LegalAction(kind=ActionKind.PLAY, source_uid=401, play_kind=PlayKind.NORMAL)
        enhance = LegalAction(kind=ActionKind.PLAY, source_uid=401, play_kind=PlayKind.ENHANCE)
        fusion_a = LegalAction(kind=ActionKind.FUSION, source_uid=501, material_uids=(1, 2))
        fusion_b = LegalAction(kind=ActionKind.FUSION, source_uid=501, material_uids=(2, 1))

        self.assertNotEqual(action_key(normal), action_key(enhance))
        self.assertNotEqual(action_key(fusion_a), action_key(fusion_b))


if __name__ == "__main__":
    unittest.main()
