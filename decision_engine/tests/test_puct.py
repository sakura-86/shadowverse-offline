# -*- coding: utf-8 -*-
"""PUCT baseline tests."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.puct import PuctSearch
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView


class PuctTest(unittest.TestCase):
    def test_puct_search_returns_visit_counts_and_policy_target(self) -> None:
        snapshot = _snapshot()
        result = PuctSearch(predictor=_Predictor(), config=_config(12)).search(snapshot, player="me")

        self.assertEqual(len(result.actions), 2)
        self.assertEqual(len(result.visit_counts), 2)
        self.assertEqual(sum(result.visit_counts), 12)
        self.assertAlmostEqual(sum(result.policy_target), 1.0)
        self.assertIn("root", result.trace)
        self.assertEqual(result.diagnostics["action_count"], 2)
        self.assertEqual(result.diagnostics["prior_source"], "uniform")
        self.assertGreater(result.diagnostics["action_predictor_call_count"], 0)
        self.assertEqual(result.diagnostics["turn_transition_count"], 0)
        self.assertEqual(
            result.diagnostics["predictor_call_count"],
            result.diagnostics["action_predictor_call_count"]
            + result.diagnostics["turn_transition_predictor_call_count"],
        )

    def test_puct_search_selects_stable_root_index_on_ties(self) -> None:
        snapshot = _snapshot()
        result = PuctSearch(predictor=_Predictor(), config=_config(1)).search(snapshot, player="me")

        self.assertEqual(result.selected_action_index, 0)

    def test_puct_search_rejects_predict_error_action(self) -> None:
        snapshot = _snapshot()
        result = PuctSearch(predictor=_FirstActionErrorPredictor(), config=_config(4, reject_predict_error=True)).search(snapshot, player="me")

        self.assertEqual(result.selected_action_index, 1)
        self.assertEqual(result.visit_counts[0], 0)
        self.assertTrue(result.trace["root"][0]["rejected"])
        self.assertEqual(result.trace["root"][0]["rejection_reason"], "predict_error")

    def test_puct_search_rejects_empty_legal_actions(self) -> None:
        snapshot = BattleSnapshot(me=PlayerView(life=20, max_life=20, pp=0, max_pp=0), opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0), cards_by_uid={})

        with self.assertRaisesRegex(ValueError, "at least one legal action"):
            PuctSearch(predictor=_Predictor(), config=_config(4)).search(snapshot, player="me")


@dataclass
class _Prediction:
    snapshot: BattleSnapshot
    certainty: object
    description: str = ""
    assumptions: tuple[str, ...] = ()
    unresolved_effects: tuple[str, ...] = ()


class _Predictor:
    def predict(self, snapshot: BattleSnapshot, action: LegalAction, *, player: str = "me") -> _Prediction:
        del action, player
        return _Prediction(snapshot=snapshot, certainty=_certainty())


class _FirstActionErrorPredictor:
    def predict(self, snapshot: BattleSnapshot, action: LegalAction, *, player: str = "me") -> _Prediction:
        del player
        certainty = PredictionCertainty.PREDICT_ERROR if action.kind is ActionKind.PLAY else _certainty()
        return _Prediction(snapshot=snapshot, certainty=certainty)


class _Certainty:
    value = "exact"



def _certainty() -> _Certainty:
    return _Certainty()


def _config(simulations: int, *, reject_predict_error: bool = False):
    from decision_engine.planning.puct import PuctConfig

    return PuctConfig(simulations=simulations, max_depth=1, c_puct=1.5, discount=1.0, reject_predict_error=reject_predict_error)


def _snapshot() -> BattleSnapshot:
    cards = {
        1: CardView(uid=1, card_id=1, base_card_id=1, cost=1, card_type=CardType.SPELL),
    }
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, hand=(1,)),
        opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0),
        cards_by_uid=cards,
        is_ally_turn=True,
    )


if __name__ == "__main__":
    unittest.main()
