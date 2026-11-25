# -*- coding: utf-8 -*-
"""SearchPrior tests."""

from __future__ import annotations

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.search_prior import PriorResult, UniformSearchPrior
from decision_engine.state.snapshot import BattleSnapshot, PlayerView


class SearchPriorTest(unittest.TestCase):
    def test_uniform_search_prior_normalizes_action_weights(self) -> None:
        prior = UniformSearchPrior().priors(_snapshot(), _actions(), player="me")

        self.assertEqual(prior.source, "uniform")
        self.assertEqual(prior.priors, (0.5, 0.5))
        self.assertEqual(prior.diagnostics["action_count"], 2)

    def test_uniform_search_prior_rejects_empty_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one legal action"):
            UniformSearchPrior().priors(_snapshot(), (), player="me")

    def test_prior_result_validation_rejects_bad_length_and_sum(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal legal action count"):
            PriorResult(priors=(1.0,), source="x").validate(2)
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            PriorResult(priors=(0.2, 0.2), source="x").validate(2)


def _snapshot() -> BattleSnapshot:
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=1, max_pp=1),
        opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1),
        cards_by_uid={},
    )


def _actions() -> tuple[LegalAction, ...]:
    return (
        LegalAction(kind=ActionKind.TURN_END, source_uid=0),
        LegalAction(kind=ActionKind.TURN_END, source_uid=0),
    )


if __name__ == "__main__":
    unittest.main()
