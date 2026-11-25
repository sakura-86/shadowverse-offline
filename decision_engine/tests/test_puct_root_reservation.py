# -*- coding: utf-8 -*-
"""Tests for PUCT root action reservation (低 prior 根动作强制访问).

覆盖：
- 默认关闭（root_reservation_ratio=0）时完全保持原行为，不改变 visit 分布与执行数。
- 开启时低 prior 的非 turn_end 根边至少被访问 min_visits 次。
- reservation + deepening + UCB 合计严格不超过且等于 simulations。
- diagnostics 报告 reservation_eligible / requested / completed。
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, Sequence

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.puct import PuctConfig, PuctSearch
from decision_engine.planning.search_prior import PriorResult
from decision_engine.planning.search_value import BoardAdvantageValue
from decision_engine.state.snapshot import (
    BattleSnapshot,
    CardType,
    CardView,
    LegalView,
    PlayerView,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _action(kind: str, uid: int = 0) -> LegalAction:
    return LegalAction(kind=ActionKind(kind), source_uid=uid)


def _snapshot() -> BattleSnapshot:
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=5, max_pp=5),
        opponent=PlayerView(life=20, max_life=20, pp=5, max_pp=5),
        cards_by_uid={},
        legal=LegalView(),
    )


@dataclass
class _MockPrediction:
    snapshot: BattleSnapshot
    certainty: Any = None
    unresolved_effects: tuple = ()


class _StaticPredictor:
    """任何动作都返回原 snapshot（无状态变化，估值恒定）。"""

    def predict(self, snapshot: BattleSnapshot, action: LegalAction, *, player: str = "me"):
        del action, player
        return _MockPrediction(snapshot=snapshot)


class _FixedEnumerator:
    """始终返回固定动作集：两个 play + 一个 turn_end。"""

    def __init__(self, actions: list[LegalAction]):
        self._actions = actions

    def enumerate(self, snapshot: BattleSnapshot, *, player: str = "me"):
        del snapshot, player
        return list(self._actions)


class _FixedPrior:
    """按 action.source_uid 返回预设 prior，使某个非 turn_end 动作 prior 极低。"""

    source = "fixed_test_prior"

    def __init__(self, prior_by_uid: dict[int, float]):
        self._prior_by_uid = prior_by_uid

    def priors(self, snapshot: BattleSnapshot, actions: Sequence[LegalAction], *, player: str) -> PriorResult:
        del snapshot, player
        raw = [self._prior_by_uid.get(a.source_uid, 0.0) for a in actions]
        total = sum(raw)
        if total <= 0:
            n = len(raw)
            values = tuple(1.0 / n for _ in raw)
        else:
            values = tuple(v / total for v in raw)
        result = PriorResult(priors=values, source=self.source)
        result.validate(len(actions))
        return result


# ---------------------------------------------------------------------------
# 固定动作集：uid=1 高 prior play, uid=2 低 prior play, uid=0 turn_end
# ---------------------------------------------------------------------------


def _actions() -> list[LegalAction]:
    return [
        _action("play", uid=1),   # index 0: 高 prior
        _action("play", uid=2),   # index 1: 极低 prior（目标 reservation 边）
        _action("turn_end", uid=0),  # index 2: turn_end，不参与 reservation
    ]


def _prior_map() -> dict[int, float]:
    # uid=1 高，uid=2 极低，turn_end(uid=0) 中等；归一化后 index1(uid=2) 最低
    return {1: 0.90, 2: 0.01, 0: 0.09}


def _make_search(config: PuctConfig) -> PuctSearch:
    return PuctSearch(
        enumerator=_FixedEnumerator(_actions()),
        child_enumerator=_FixedEnumerator(_actions()),
        predictor=_StaticPredictor(),
        prior=_FixedPrior(_prior_map()),
        value=BoardAdvantageValue(),
        config=config,
    )


# 关闭 dirichlet 噪声与 high-prior deepening，隔离 reservation 行为的基线配置。
def _base_config(simulations: int, **overrides) -> PuctConfig:
    params = dict(
        simulations=simulations,
        max_depth=1,
        c_puct=1.5,
        dirichlet_epsilon=0.0,
        prior_deepening_threshold=0.0,   # 关闭 high-prior deepening
        prior_deepening_ratio=0.0,
        root_reservation_ratio=0.0,      # 默认关闭 reservation
        root_reservation_min_visits=0,
    )
    params.update(overrides)
    return PuctConfig(**params)


class TestPuctRootReservation(unittest.TestCase):

    def test_default_off_keeps_original_behavior(self):
        """默认 root_reservation_ratio=0 时不预留：低 prior 边在小预算下不被访问，
        executed_simulations 等于配置，且 reservation 诊断全为 0。"""
        config = _base_config(8)
        result = _make_search(config).search(_snapshot(), player="me")

        # 低 prior 的 play(uid=2, index=1) 在纯 UCB + 小预算下不会被访问
        self.assertEqual(result.visit_counts[1], 0)
        self.assertEqual(result.diagnostics["executed_simulations"], 8)
        self.assertEqual(sum(result.visit_counts), 8)
        self.assertEqual(result.diagnostics["reservation_requested"], 0)
        self.assertEqual(result.diagnostics["reservation_completed"], 0)
        self.assertEqual(result.diagnostics["reservation_eligible"], 0)

    def test_default_off_identical_to_no_reservation_fields(self):
        """开启前后，关闭配置下 visit 分布应与不含 reservation 的旧语义一致（回归保护）。"""
        config = _base_config(16)
        result = _make_search(config).search(_snapshot(), player="me")
        # 全部预算落在常规 UCB，低 prior 边始终 0
        self.assertEqual(result.visit_counts[1], 0)
        self.assertEqual(sum(result.visit_counts), 16)

    def test_reservation_visits_low_prior_edge(self):
        """开启 reservation 后，低 prior 的非 turn_end 根边至少被访问 min_visits 次。"""
        config = _base_config(
            8,
            root_reservation_ratio=0.5,
            root_reservation_min_visits=1,
        )
        result = _make_search(config).search(_snapshot(), player="me")

        # 低 prior play(index=1) 现在至少被访问一次
        self.assertGreaterEqual(result.visit_counts[1], 1)
        # 高 prior play(index=0) 也是 eligible，应同样被预留
        self.assertGreaterEqual(result.visit_counts[0], 1)

    def test_reservation_min_visits_multiple(self):
        """min_visits=2 时每个 eligible 边至少访问 2 次（预算足够时）。"""
        config = _base_config(
            32,
            root_reservation_ratio=0.5,
            root_reservation_min_visits=2,
        )
        result = _make_search(config).search(_snapshot(), player="me")

        # 两个 play 边（eligible）各自至少 2 次
        self.assertGreaterEqual(result.visit_counts[1], 2)
        self.assertGreaterEqual(result.visit_counts[0], 2)

    def test_executed_simulations_equals_config(self):
        """reservation + UCB 合计严格等于 simulations（不超过、尽量等于）。"""
        config = _base_config(
            20,
            root_reservation_ratio=0.4,
            root_reservation_min_visits=1,
        )
        result = _make_search(config).search(_snapshot(), player="me")

        self.assertEqual(result.diagnostics["executed_simulations"], 20)
        self.assertEqual(sum(result.visit_counts), 20)

    def test_executed_simulations_with_deepening_and_reservation(self):
        """reservation + high-prior deepening + UCB 三阶段合计严格不超过且等于 simulations。"""
        config = _base_config(
            24,
            prior_deepening_threshold=0.15,
            prior_deepening_ratio=0.3,
            prior_deepening_sims=2,
            root_reservation_ratio=0.3,
            root_reservation_min_visits=1,
        )
        result = _make_search(config).search(_snapshot(), player="me")

        self.assertLessEqual(result.diagnostics["executed_simulations"], 24)
        self.assertEqual(result.diagnostics["executed_simulations"], 24)
        self.assertEqual(sum(result.visit_counts), 24)

    def test_reservation_diagnostics_reported(self):
        """diagnostics 报告 reservation_eligible / requested / completed。"""
        config = _base_config(
            12,
            root_reservation_ratio=0.5,
            root_reservation_min_visits=1,
        )
        result = _make_search(config).search(_snapshot(), player="me")

        diag = result.diagnostics
        # 两个 play 是 eligible（非 turn_end、未 reject）
        self.assertEqual(diag["reservation_eligible"], 2)
        self.assertGreaterEqual(diag["reservation_requested"], 1)
        self.assertGreaterEqual(diag["reservation_completed"], 1)
        # completed 不超过 requested
        self.assertLessEqual(diag["reservation_completed"], diag["reservation_requested"])

    def test_reservation_prefers_low_prior_first(self):
        """预算不足以覆盖全部 eligible 时，低 prior 边优先获得预留。"""
        # budget_cap = int(6 * 0.2) = 1，只够预留 1 个 edge → 应给最低 prior 的 index=1
        config = _base_config(
            6,
            root_reservation_ratio=0.2,
            root_reservation_min_visits=1,
        )
        result = _make_search(config).search(_snapshot(), player="me")

        self.assertEqual(result.diagnostics["reservation_completed"], 1)
        # 最低 prior 的 play(index=1) 被优先预留访问
        self.assertGreaterEqual(result.visit_counts[1], 1)


if __name__ == "__main__":
    unittest.main()
