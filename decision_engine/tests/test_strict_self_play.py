# -*- coding: utf-8 -*-
"""Strict self-play contract tests."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.diff_value import DiffSearchValue
from decision_engine.planning.strict_self_play import (
    STRICT_CERTAINTY_VALUES,
    StrictSelfPlayConfig,
    StrictSelfPlayRunner,
    _is_strict_certainty,
    _valid_policy,
    diff_value_factory,
    terminal_value_factory,
)
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot, PlayerView


class StrictSelfPlayContractTest(unittest.TestCase):
    def test_diff_factory_binds_current_turn_start_and_player(self) -> None:
        turn_start = _snapshot(me_life=20, opponent_life=18)

        value = diff_value_factory(turn_start, "opponent")

        self.assertIsInstance(value, DiffSearchValue)
        self.assertEqual(value._turn_start, turn_start)
        self.assertEqual(value._player, "opponent")

    def test_baseline_factory_is_not_diff(self) -> None:
        value = terminal_value_factory(_snapshot(), "me")

        self.assertNotIsInstance(value, DiffSearchValue)
        self.assertEqual(value.evaluate(_snapshot(), player="me").source, "non_terminal_zero_baseline")

    def test_strict_runner_forces_single_search_turn(self) -> None:
        runner = StrictSelfPlayRunner(
            predictor=_Predictor(),
            value_factory=diff_value_factory,
            arm="diff",
            config=StrictSelfPlayConfig(simulations=4, max_depth=2),
        )

        search = runner._build_search(diff_value_factory(_snapshot(), "me"), seed=10, turn_index=2)

        self.assertEqual(search.config.max_turns, 1)
        self.assertTrue(search.config.reject_predict_error)

    def test_policy_rejects_nonzero_visits_on_rejected_edge(self) -> None:
        self.assertFalse(_valid_policy((2, 1), (2 / 3, 1 / 3), {"root": [{"rejected": True}, {"rejected": False}]}))
        self.assertTrue(_valid_policy((0, 3), (0.0, 1.0), {"root": [{"rejected": True}, {"rejected": False}]}))

    def test_strict_certainties_use_stable_protocol_values(self) -> None:
        self.assertEqual(STRICT_CERTAINTY_VALUES, {"exact", "spec_only"})
        self.assertTrue(_is_strict_certainty(PredictionCertainty.EXACT))
        self.assertTrue(_is_strict_certainty(PredictionCertainty.SPEC_ONLY))
        self.assertTrue(_is_strict_certainty("spec_only"))
        self.assertFalse(_is_strict_certainty(PredictionCertainty.PARTIAL))
        self.assertFalse(_is_strict_certainty(PredictionCertainty.UNKNOWN))
        self.assertFalse(_is_strict_certainty(PredictionCertainty.PREDICT_ERROR))


@dataclass
class _Predictor:
    def predict_with_replay_context(self, snapshot, action, *, player="me", generated_cards=(), execution_context=None):
        del action, player, generated_cards, execution_context
        return _Prediction(snapshot)


@dataclass
class _Prediction:
    snapshot: BattleSnapshot
    certainty: PredictionCertainty = PredictionCertainty.EXACT
    unresolved_effects: tuple[str, ...] = ()


def _snapshot(*, me_life: int = 20, opponent_life: int = 20) -> BattleSnapshot:
    return BattleSnapshot(
        me=PlayerView(life=me_life, max_life=20, pp=1, max_pp=1),
        opponent=PlayerView(life=opponent_life, max_life=20, pp=1, max_pp=1),
        cards_by_uid={},
    )


if __name__ == "__main__":
    unittest.main()
