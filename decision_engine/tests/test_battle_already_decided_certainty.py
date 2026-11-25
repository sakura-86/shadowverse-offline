# -*- coding: utf-8 -*-
"""胜负已判定后不再结算效果，certainty 不应保留 PARTIAL。"""

from __future__ import annotations

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.execution import LocalPredictor
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot, PlayerView


class BattleAlreadyDecidedTest(unittest.TestCase):
    def test_decided_battle_drops_partial_and_unresolved(self) -> None:
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=0),
            opponent=PlayerView(life=0, max_life=20, pp=0, max_pp=0, result=104),
            cards_by_uid={},
            source="test",
        )

        result = LocalPredictor()._apply_event_queue_effects(
            after_core=snapshot,
            action=LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=3),
            events=(),
            default_certainty=PredictionCertainty.PARTIAL,
            default_description="local attack leader core rules only",
            default_assumptions=("attack_effect_candidates_present",),
            default_unresolved=("attack_effect_candidates_present",),
        )

        self.assertEqual(result.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(result.unresolved_effects, ())
        self.assertIn("battle_already_decided", result.assumptions)

    def test_decided_battle_keeps_higher_certainty(self) -> None:
        snapshot = BattleSnapshot(
            me=PlayerView(life=0, max_life=20, pp=0, max_pp=0, result=103),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0),
            cards_by_uid={},
            source="test",
        )

        result = LocalPredictor()._apply_event_queue_effects(
            after_core=snapshot,
            action=LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=3),
            events=(),
            default_certainty=PredictionCertainty.EXACT,
            default_description="exact",
            default_assumptions=(),
            default_unresolved=(),
        )

        self.assertEqual(result.certainty, PredictionCertainty.EXACT)


if __name__ == "__main__":
    unittest.main()
