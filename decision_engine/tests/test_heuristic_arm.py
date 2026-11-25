# -*- coding: utf-8 -*-
"""L1 启发式对手的单元测试（纯逻辑，不跑真实对局）。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from decision_engine.actions.model import ActionKind
from decision_engine.planning.heuristic_arm import (
    HeuristicArmSelector,
    _first_target_uid,
    _views,
)


class FirstTargetUidTest(unittest.TestCase):
    def test_returns_first_uid_of_first_non_empty_group(self) -> None:
        action = SimpleNamespace(target_groups=((), (7, 8), (9,)))
        self.assertEqual(_first_target_uid(action), 7)

    def test_returns_none_when_no_targets(self) -> None:
        self.assertIsNone(_first_target_uid(SimpleNamespace(target_groups=())))
        self.assertIsNone(_first_target_uid(SimpleNamespace(target_groups=((), ()))))


class ViewsTest(unittest.TestCase):
    def test_perspective_swaps_sides(self) -> None:
        snapshot = SimpleNamespace(me="M", opponent="O")
        self.assertEqual(_views(snapshot, "me"), ("M", "O"))
        self.assertEqual(_views(snapshot, "opponent"), ("O", "M"))


class HeuristicPriorityTest(unittest.TestCase):
    """用假 enumerator/predictor 验证优先级顺序。"""

    def _selector(self, actions, *, prediction_ok: bool = True):
        class _Enumerator:
            def enumerate(self, snapshot, *, player):
                return actions

        class _Predictor:
            def predict(self, snapshot, action, *, player):
                if not prediction_ok:
                    raise RuntimeError("boom")
                return object()

        selector = HeuristicArmSelector(
            _Predictor(), _Enumerator(),
            prediction_filter=lambda _prediction: prediction_ok,
        )
        return selector

    def _snapshot(self):
        # 我方一个 3/2 可攻击随从；敌方 10 血、一个 2/2 随从。
        cards = {
            1: SimpleNamespace(atk=3, life=2, can_attack=True),
            2: SimpleNamespace(atk=2, life=2, can_attack=True),
        }
        return SimpleNamespace(
            me=SimpleNamespace(life=20, field=(1,), pp=3),
            opponent=SimpleNamespace(life=10, field=(2,)),
            cards_by_uid=cards,
        )

    def test_lethal_leader_attack_has_top_priority(self) -> None:
        actions = (
            SimpleNamespace(kind=ActionKind.PLAY, source_uid=5, cost=5, target_groups=()),
            SimpleNamespace(kind=ActionKind.ATTACK_LEADER, source_uid=1, cost=None, target_groups=()),
        )
        # 敌方 10 血、我方场攻 3，不构成斩杀 -> 应该先出牌。
        selector = self._selector(actions)
        result = selector.select(self._snapshot(), player="me", branch_seed=0)
        self.assertEqual(result.selected_action_index, 0)

    def test_play_prefers_highest_cost(self) -> None:
        actions = (
            SimpleNamespace(kind=ActionKind.PLAY, source_uid=5, cost=2, target_groups=()),
            SimpleNamespace(kind=ActionKind.PLAY, source_uid=6, cost=7, target_groups=()),
            SimpleNamespace(kind=ActionKind.PLAY, source_uid=7, cost=4, target_groups=()),
        )
        selector = self._selector(actions)
        result = selector.select(self._snapshot(), player="me", branch_seed=0)
        self.assertEqual(result.actions[result.selected_action_index].source_uid, 6)

    def test_no_strict_action_returns_none_index(self) -> None:
        actions = (
            SimpleNamespace(kind=ActionKind.PLAY, source_uid=5, cost=2, target_groups=()),
        )
        selector = self._selector(actions, prediction_ok=False)
        result = selector.select(self._snapshot(), player="me", branch_seed=0)
        self.assertIsNone(result.selected_action_index)
        self.assertEqual(result.diagnostics.get("reject"), "no_strict_action")

    def test_empty_actions_returns_none_index(self) -> None:
        selector = self._selector(())
        result = selector.select(self._snapshot(), player="me", branch_seed=0)
        self.assertIsNone(result.selected_action_index)
        self.assertEqual(result.diagnostics.get("reject"), "no_legal_actions")


if __name__ == "__main__":
    unittest.main()
