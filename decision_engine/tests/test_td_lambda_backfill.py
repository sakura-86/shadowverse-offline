# -*- coding: utf-8 -*-
"""TD(λ) / n-step λ-return 回填的数学正确性测试。

核心验收：λ=1 必须严格退化为 mc_discounted（前向 λ-return 的理论要求），
且折现按全局 turn_index 差（与 mc 同一把尺子）。
"""

import unittest
from dataclasses import dataclass

from decision_engine.planning.value_target_backfill import (
    compute_td_lambda_for_episode,
    compute_mc_discounted_for_episode,
)
from decision_engine.state.snapshot import BattleSnapshot, PlayerView


@dataclass
class _FakeStep:
    sample_id: str
    player: str
    turn_index: int
    snapshot: object
    prediction: object = None


def _snap(my_life: int = 18, opp_life: int = 18) -> BattleSnapshot:
    return BattleSnapshot(
        me=PlayerView(life=my_life, max_life=20, pp=5, max_pp=5, turn=1, deck_count=20),
        opponent=PlayerView(life=opp_life, max_life=20, pp=5, max_pp=5, turn=1, deck_count=20),
        cards_by_uid={}, source="t", is_ally_turn=True,
    )


def _make_episode(seq):
    return [_FakeStep("s%d" % i, pl, t, _snap()) for i, (pl, t) in enumerate(seq)]


class TdLambdaTest(unittest.TestCase):
    # me 先手、全局终局落在 opponent 回合
    SEQ_END_OPP = [("me", 1), ("opponent", 2), ("me", 3), ("opponent", 4), ("me", 5), ("opponent", 6)]
    # 全局终局落在 me 回合
    SEQ_END_ME = [("me", 1), ("opponent", 2), ("me", 3), ("opponent", 4), ("me", 5)]

    def test_lambda_one_equals_mc_discounted(self):
        """λ=1 必须严格等于 mc_discounted（两种终局归属 × 多个 γ）。"""
        for seq in (self.SEQ_END_OPP, self.SEQ_END_ME):
            steps = _make_episode(seq)
            for winner in ("me", "opponent"):
                for gamma in (0.80, 0.90, 0.97, 1.0):
                    td = compute_td_lambda_for_episode(steps, winner, gamma=gamma, lam=1.0)
                    mc = compute_mc_discounted_for_episode(steps, winner, gamma=gamma)
                    self.assertEqual(set(td), set(mc))
                    for sid in td:
                        self.assertAlmostEqual(
                            td[sid], mc[sid], places=10,
                            msg=f"seq={seq} winner={winner} γ={gamma} sid={sid}",
                        )

    def test_lambda_monotonic_interpolation(self):
        """固定早期帧，value 随 λ 从 0→1 单调从 1-step TD 过渡到 MC。"""
        steps = _make_episode(self.SEQ_END_OPP)
        v0 = compute_td_lambda_for_episode(steps, "me", gamma=0.9, lam=0.0)["s0"]
        v5 = compute_td_lambda_for_episode(steps, "me", gamma=0.9, lam=0.5)["s0"]
        v1 = compute_td_lambda_for_episode(steps, "me", gamma=0.9, lam=1.0)["s0"]
        # 单调（本构造下三者递增）
        self.assertLessEqual(v0, v5)
        self.assertLessEqual(v5, v1)

    def test_no_winner_returns_empty(self):
        steps = _make_episode(self.SEQ_END_OPP)
        self.assertEqual(compute_td_lambda_for_episode(steps, None, gamma=0.9, lam=0.7), {})

    def test_value_in_range(self):
        """所有 λ-return 落在 [-1, 1]。"""
        steps = _make_episode(self.SEQ_END_OPP)
        for lam in (0.0, 0.3, 0.7, 1.0):
            out = compute_td_lambda_for_episode(steps, "me", gamma=0.97, lam=lam)
            for v in out.values():
                self.assertGreaterEqual(v, -1.0 - 1e-9)
                self.assertLessEqual(v, 1.0 + 1e-9)

    def test_all_frames_covered(self):
        """每个 sample_id 都被回填。"""
        steps = _make_episode(self.SEQ_END_OPP)
        out = compute_td_lambda_for_episode(steps, "me", gamma=0.97, lam=0.7)
        self.assertEqual(set(out), {s.sample_id for s in steps})


if __name__ == "__main__":
    unittest.main()
