# -*- coding: utf-8 -*-
"""value_target_backfill 单元测试:4回合窗口定位。"""
import unittest
from dataclasses import dataclass
from typing import Any
from decision_engine.planning.value_target_backfill import compute_targets_for_episode, compute_targets_multi_window
from decision_engine.planning.value_target import ValueTargetWeights, compute_value_target
from decision_engine.state.snapshot import BattleSnapshot, PlayerView, LegalView

W = ValueTargetWeights()


def _snap(my_life=20, opp_life=20):
    def mk(life): return PlayerView(life=life, max_life=20, pp=10, max_pp=10)
    return BattleSnapshot(me=mk(my_life), opponent=mk(opp_life), cards_by_uid={}, legal=LegalView())


@dataclass
class _Pred:
    snapshot: Any

@dataclass
class _Step:
    player: str
    turn_index: int
    snapshot: Any
    prediction: Any
    sample_id: str


def _mk_step(player, turn, before, after, sid):
    return _Step(player=player, turn_index=turn, snapshot=before,
                 prediction=_Pred(snapshot=after), sample_id=sid)


class TestBackfillWindow(unittest.TestCase):
    def test_window_locates_C_end(self):
        """回合序列 me(A) opp(B) me(C) opp(D) → A帧的C_end = C段after。"""
        # A: 我方回合,对方满血20; C段结束时对方10(A的随从在C打了10脸)
        steps = [
            _mk_step("me", 1, _snap(opp_life=20), _snap(opp_life=20), "A1"),   # A: 决策前对方20
            _mk_step("opponent", 2, _snap(opp_life=20), _snap(opp_life=20), "B1"),
            _mk_step("me", 3, _snap(opp_life=20), _snap(opp_life=10), "C1"),   # C: 打脸到10
            _mk_step("opponent", 4, _snap(opp_life=10), _snap(opp_life=10), "D1"),
        ]
        targets = compute_targets_for_episode(steps, weights=W)
        # A1的窗口: start=A1.snapshot(opp20) → C_end=C段after(opp10) → 净打脸10
        expected_A = compute_value_target(_snap(opp_life=20), _snap(opp_life=10), "me", weights=W)
        self.assertAlmostEqual(targets["A1"], expected_A, places=5)
        self.assertGreater(targets["A1"], 0)  # 打脸=正

    def test_early_end_uses_last_after(self):
        """对局提前结束(无下下个我方回合) → C_end用整局末尾after。"""
        steps = [
            _mk_step("me", 1, _snap(opp_life=20), _snap(opp_life=15), "A1"),
            _mk_step("opponent", 2, _snap(opp_life=15), _snap(opp_life=15), "B1"),
            # 没有第2个me回合 → A1的C_end用最后帧after(B1的after=opp15)
        ]
        targets = compute_targets_for_episode(steps, weights=W)
        expected = compute_value_target(_snap(opp_life=20), _snap(opp_life=15), "me", weights=W)
        self.assertAlmostEqual(targets["A1"], expected, places=5)

    def test_opponent_frames_root_is_opponent(self):
        """对方回合的帧,root_player=opponent。"""
        steps = [
            _mk_step("me", 1, _snap(), _snap(), "A1"),
            _mk_step("opponent", 2, _snap(my_life=20), _snap(my_life=20), "B1"),
            _mk_step("me", 3, _snap(), _snap(), "C1"),
            _mk_step("opponent", 4, _snap(my_life=20), _snap(my_life=12), "D1"),  # 对方打我脸
        ]
        targets = compute_targets_for_episode(steps, weights=W)
        # B1: root=opponent, 窗口B→D, D段对方(=我方me)掉血... root视角opponent
        self.assertIn("B1", targets)

    def test_all_frames_get_target(self):
        steps = [
            _mk_step("me", 1, _snap(), _snap(), "A1"),
            _mk_step("me", 1, _snap(), _snap(), "A2"),  # 同回合第2帧
            _mk_step("opponent", 2, _snap(), _snap(), "B1"),
            _mk_step("me", 3, _snap(), _snap(), "C1"),
        ]
        targets = compute_targets_for_episode(steps, weights=W)
        self.assertEqual(set(targets.keys()), {"A1", "A2", "B1", "C1"})


class TestLookaheadWindow(unittest.TestCase):
    """窗口长度 lookahead_my_turns 定位。序列 A我 B敌 C我 D敌 E我。"""

    def _seq(self):
        # A(opp20) → C打脸到15 → E打脸到8。用于区分 seen==1(到C)和 seen==2(到E)。
        return [
            _mk_step("me", 1, _snap(opp_life=20), _snap(opp_life=20), "A1"),
            _mk_step("opponent", 2, _snap(opp_life=20), _snap(opp_life=20), "B1"),
            _mk_step("me", 3, _snap(opp_life=20), _snap(opp_life=15), "C1"),
            _mk_step("opponent", 4, _snap(opp_life=15), _snap(opp_life=15), "D1"),
            _mk_step("me", 5, _snap(opp_life=15), _snap(opp_life=8), "E1"),
        ]

    def test_lookahead1_stops_at_C(self):
        """seen==1: A 的窗口终点=C段after(opp15) → 净打脸5。"""
        t = compute_targets_for_episode(self._seq(), weights=W, lookahead_my_turns=1)
        exp = compute_value_target(_snap(opp_life=20), _snap(opp_life=15), "me", weights=W)
        self.assertAlmostEqual(t["A1"], exp, places=5)

    def test_lookahead2_stops_at_E(self):
        """seen==2: A 的窗口终点=E段after(opp8) → 净打脸12。"""
        t = compute_targets_for_episode(self._seq(), weights=W, lookahead_my_turns=2)
        exp = compute_value_target(_snap(opp_life=20), _snap(opp_life=8), "me", weights=W)
        self.assertAlmostEqual(t["A1"], exp, places=5)

    def test_lookahead1_differs_from_lookahead2(self):
        """两个窗口对同一帧给出不同 value(长窗口含更多后续兑现)。"""
        t1 = compute_targets_for_episode(self._seq(), weights=W, lookahead_my_turns=1)
        t2 = compute_targets_for_episode(self._seq(), weights=W, lookahead_my_turns=2)
        self.assertNotAlmostEqual(t1["A1"], t2["A1"], places=3)
        self.assertGreater(t2["A1"], t1["A1"])  # E打脸更多,长窗口value更高


class TestMultiWindow(unittest.TestCase):
    def _seq(self):
        return [
            _mk_step("me", 1, _snap(opp_life=20), _snap(opp_life=20), "A1"),
            _mk_step("opponent", 2, _snap(opp_life=20), _snap(opp_life=20), "B1"),
            _mk_step("me", 3, _snap(opp_life=20), _snap(opp_life=15), "C1"),
            _mk_step("opponent", 4, _snap(opp_life=15), _snap(opp_life=15), "D1"),
            _mk_step("me", 5, _snap(opp_life=15), _snap(opp_life=8), "E1"),
        ]

    def test_multi_window_weighted_average(self):
        """0.8·seen1 + 0.2·seen2 = 加权平均(权重归一)。"""
        seq = self._seq()
        t1 = compute_targets_for_episode(seq, weights=W, lookahead_my_turns=1)
        t2 = compute_targets_for_episode(seq, weights=W, lookahead_my_turns=2)
        tm = compute_targets_multi_window(seq, [(1, 0.8), (2, 0.2)], weights=W)
        exp = (0.8 * t1["A1"] + 0.2 * t2["A1"]) / 1.0
        self.assertAlmostEqual(tm["A1"], exp, places=5)
        # 加权值介于两个窗口之间
        self.assertTrue(min(t1["A1"], t2["A1"]) <= tm["A1"] <= max(t1["A1"], t2["A1"]))

    def test_multi_window_weight_normalization(self):
        """非归一权重(0.8/0.2和=1)与(4/1和=5)结果相同(自动归一)。"""
        seq = self._seq()
        a = compute_targets_multi_window(seq, [(1, 0.8), (2, 0.2)], weights=W)
        b = compute_targets_multi_window(seq, [(1, 4.0), (2, 1.0)], weights=W)
        self.assertAlmostEqual(a["A1"], b["A1"], places=5)

    def test_multi_window_all_frames(self):
        tm = compute_targets_multi_window(self._seq(), [(1, 0.8), (2, 0.2)], weights=W)
        self.assertIn("A1", tm)
        self.assertTrue(all(-1.0 <= v <= 1.0 for v in tm.values()))


if __name__ == "__main__":
    unittest.main()
