# -*- coding: utf-8 -*-
"""PPO 起势验证的单元测试：GAE 数值与 reward 回填口径。"""

from __future__ import annotations

import unittest

from decision_engine.planning.search_ab_match import GameOutcome
from decision_engine.training.ppo_rollout import (
    PPORolloutCollector,
    RolloutStep,
)
from decision_engine.training.ppo_trainer import compute_gae, roc_auc


def _step(*, value: float, reward: float = 0.0, player: str = "me") -> RolloutStep:
    return RolloutStep(
        sample_id="s",
        player=player,
        example={},
        selected_index=0,
        log_prob=0.0,
        value=value,
        executable_count=2,
        executable_indices=(0, 1),
        turn_index=1,
        step_index=0,
        reward=reward,
    )


class ComputeGAETest(unittest.TestCase):
    def test_two_step_episode_matches_hand_computation(self) -> None:
        # gamma=0.9, lam=1.0，只有最后一步有 reward=+1。
        episode = [_step(value=0.5), _step(value=0.4, reward=1.0)]
        result = compute_gae([episode], gamma=0.9, lam=1.0)

        # 末步：delta = 1.0 + 0 - 0.4 = 0.6
        # 首步：delta = 0 + 0.9*0.4 - 0.5 = -0.14；adv = -0.14 + 0.9*1.0*0.6 = 0.4
        self.assertAlmostEqual(result["advantages"][1], 0.6, places=6)
        self.assertAlmostEqual(result["advantages"][0], 0.4, places=6)
        # return = advantage + value
        self.assertAlmostEqual(result["returns"][1], 1.0, places=6)
        self.assertAlmostEqual(result["returns"][0], 0.9, places=6)

    def test_sparse_reward_alone_still_produces_signal(self) -> None:
        # 中间步 reward 全 0：advantage 完全由 value 的时间差分决定（本实验的前提）。
        episode = [_step(value=0.1), _step(value=0.3), _step(value=0.9, reward=1.0)]
        result = compute_gae([episode], gamma=1.0, lam=1.0)
        # 末步 delta = 1.0 - 0.9 = 0.1；逐步向前累积
        self.assertAlmostEqual(result["advantages"][2], 0.1, places=6)
        self.assertAlmostEqual(result["advantages"][1], 0.1 + (0.9 - 0.3), places=6)
        self.assertAlmostEqual(result["advantages"][0], 0.1 + 0.6 + (0.3 - 0.1), places=6)


def _game(*, accepted: bool = True, winner_side: str | None = "me") -> GameOutcome:
    return GameOutcome(
        game_id="g", seed=1, first_player="me", candidate_side="me",
        accepted=accepted, winner_side=winner_side,
        candidate_won=(winner_side == "me") if accepted and winner_side else None,
        reason="terminal:me" if accepted else "no_terminal",
        turns=10, steps=20, diagnostics={},
    )


class RolloutCollectorTest(unittest.TestCase):
    def test_only_last_step_gets_terminal_reward(self) -> None:
        collector = PPORolloutCollector(actor_side="me")
        collector._pending["g"] = [_step(value=0.1), _step(value=0.2), _step(value=0.3)]
        collector.finalize_game(_game(winner_side="me"))

        steps = collector.batch.all_steps()
        self.assertEqual([s.reward for s in steps], [0.0, 0.0, 1.0])

    def test_loser_gets_negative_terminal_reward(self) -> None:
        collector = PPORolloutCollector(actor_side="me")
        collector._pending["g"] = [_step(value=0.1), _step(value=0.2)]
        collector.finalize_game(_game(winner_side="opponent"))
        self.assertEqual([s.reward for s in collector.batch.all_steps()], [0.0, -1.0])

    def test_rejected_game_is_dropped_not_zero_filled(self) -> None:
        collector = PPORolloutCollector(actor_side="me")
        collector._pending["g"] = [_step(value=0.1), _step(value=0.2)]
        collector.finalize_game(_game(accepted=False, winner_side=None))

        self.assertEqual(collector.batch.all_steps(), [])
        self.assertEqual(collector.batch.dropped_episodes, 1)

    def test_accepted_game_without_winner_is_dropped(self) -> None:
        collector = PPORolloutCollector(actor_side="me")
        collector._pending["g"] = [_step(value=0.1)]
        collector.finalize_game(_game(accepted=True, winner_side=None))
        self.assertEqual(collector.batch.all_steps(), [])
        self.assertEqual(collector.batch.dropped_episodes, 1)

    def test_outcome_backfilled_from_winner(self) -> None:
        collector = PPORolloutCollector(actor_side="me")
        collector._pending["g"] = [_step(value=0.1), _step(value=0.2)]
        collector.finalize_game(_game(accepted=True, winner_side="me"))
        self.assertEqual([step.outcome for step in collector.batch.all_steps()], [1.0, 1.0])

    def test_outcome_negative_for_loss(self) -> None:
        collector = PPORolloutCollector(actor_side="me")
        collector._pending["g"] = [_step(value=0.1)]
        collector.finalize_game(_game(accepted=True, winner_side="opponent"))
        self.assertEqual([step.outcome for step in collector.batch.all_steps()], [-1.0])


class RocAucTest(unittest.TestCase):
    def test_perfect_separation(self) -> None:
        self.assertAlmostEqual(roc_auc([2.0, 3.0, -1.0, -2.0], [1.0, 1.0, -1.0, -1.0]), 1.0, places=6)

    def test_reversed_separation(self) -> None:
        self.assertAlmostEqual(roc_auc([-2.0, -3.0, 1.0, 2.0], [1.0, 1.0, -1.0, -1.0]), 0.0, places=6)

    def test_ties_give_half(self) -> None:
        self.assertAlmostEqual(roc_auc([1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0]), 0.5, places=6)

    def test_single_class_is_neutral(self) -> None:
        self.assertAlmostEqual(roc_auc([1.0, 2.0], [1.0, 1.0]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
