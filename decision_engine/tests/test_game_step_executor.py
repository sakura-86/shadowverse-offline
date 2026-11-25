"""GameStepExecutor deckout regression tests."""

import unittest

from decision_engine.execution.game_step_executor import GameStepExecutor
from decision_engine.execution.local_predictor import LocalPredictor
from decision_engine.execution.player_local_predictor import PlayerLocalPredictor
from decision_engine.execution.rule_kernel import BATTLE_RESULT_DECK_OUT_LOSE, BATTLE_RESULT_DECK_OUT_WIN
from decision_engine.state.snapshot import BattleSnapshot, PlayerView


class GameStepExecutorDeckoutTest(unittest.TestCase):
    def test_turn_transition_preserves_empty_deck_and_resolves_deckout(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, deck_count=3),
            opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1, deck_count=0),
            cards_by_uid={},
            is_turn_ended=True,
        )
        executor = GameStepExecutor(
            predictor=PlayerLocalPredictor(LocalPredictor()),
            seed_base=42,
        )

        result = executor.execute_turn_transition(snapshot, ending_player="me")

        self.assertEqual(result.snapshot.opponent.deck_count, 0)
        self.assertTrue(result.snapshot.is_turn_ended)
        self.assertEqual(result.snapshot.me.result, BATTLE_RESULT_DECK_OUT_WIN)
        self.assertEqual(result.snapshot.opponent.result, BATTLE_RESULT_DECK_OUT_LOSE)
        self.assertIsNotNone(result.turn_start_prediction)


if __name__ == "__main__":
    unittest.main()
