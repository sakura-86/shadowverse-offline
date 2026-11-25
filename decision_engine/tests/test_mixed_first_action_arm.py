# -*- coding: utf-8 -*-
"""``MixedFirstActionSelector``：每回合第一步由模型 prior 决定，其余交给 ``plan_turn``。

这一层的契约只有三条，任何一条破了都会让 A/B 结论失真：

1. 每回合**第一次** ``select`` 必须是 ``phase=model_first``，且**不**调用 ``plan_turn``；
2. 同回合后续 ``select`` 必须是 ``phase=plan_rest``，且走 ``plan_turn``；
3. 进入新回合后重置回 ``model_first``。

``plan_turn`` 用 mock 替换：它需要完整 snapshot + 真实本地预测器，本测试只验证
"谁做决定"，不验证规划质量（那由 ``test_turn_plan_perspective`` 等覆盖）。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from decision_engine.planning.turn_plan_arm import (
    MixedFirstActionConfig,
    MixedFirstActionSelector,
    TurnPlanArmConfig,
    build_mixed_first_action_arm,
)


def _action(uid: int) -> SimpleNamespace:
    return SimpleNamespace(
        kind="play",
        source_uid=uid,
        play_kind=None,
        target_groups=(),
        random_target_groups=(),
        selected_indexes=(),
        material_uids=(),
        variant=None,
        cost=1,
    )


def _snapshot(turn: int) -> SimpleNamespace:
    return SimpleNamespace(
        me=SimpleNamespace(turn=turn, pp=10),
        opponent=SimpleNamespace(turn=turn, pp=0),
        is_ally_turn=True,
    )


class _FixedEnumerator:
    def __init__(self, actions) -> None:
        self._actions = tuple(actions)

    def enumerate(self, snapshot, *, player):
        return list(self._actions)


class _FixedPrior:
    """固定 prior 向量；argmax 落在 ``self.expected_index``。"""

    def __init__(self, priors) -> None:
        self._priors = tuple(priors)

    def priors(self, snapshot, actions, *, player):
        return SimpleNamespace(priors=list(self._priors))


class _ExactPredictor:
    def predict(self, snapshot, action, *, player):
        return SimpleNamespace(certainty="exact", unresolved_effects=())


def _selector(actions, priors) -> MixedFirstActionSelector:
    return MixedFirstActionSelector(
        _ExactPredictor(),
        _FixedEnumerator(actions),
        plan_config=TurnPlanArmConfig(max_actions=12),
        prior=_FixedPrior(priors),
    )


class MixedFirstActionSelectorTest(unittest.TestCase):
    def test_first_step_is_model_rest_is_plan(self) -> None:
        actions = (_action(1), _action(2))
        selector = _selector(actions, [0.1, 0.9])

        with patch("decision_engine.planning.turn_plan_arm.plan_turn") as fake_plan:
            fake_plan.return_value = SimpleNamespace(
                route=(actions[0],), score=None, nodes=1, truncated=False
            )

            first = selector.select(_snapshot(3), player="me", branch_seed=1)
            self.assertEqual(first.diagnostics["phase"], "model_first")
            self.assertEqual(first.selected_action_index, 1, "模型应选 prior 最高的动作")
            self.assertEqual(fake_plan.call_count, 0, "首动作不该触发 plan_turn")

            second = selector.select(_snapshot(3), player="me", branch_seed=1)
            self.assertEqual(second.diagnostics["phase"], "plan_rest")
            self.assertEqual(second.selected_action_index, 0, "其余步应跟随 plan_turn 路线")
            self.assertEqual(fake_plan.call_count, 1)

        self.assertEqual(selector.model_first_decisions, 1)
        self.assertEqual(selector.plan_decisions, 1)

    def test_new_turn_resets_to_model(self) -> None:
        actions = (_action(1), _action(2))
        selector = _selector(actions, [0.9, 0.1])

        with patch("decision_engine.planning.turn_plan_arm.plan_turn") as fake_plan:
            fake_plan.return_value = SimpleNamespace(
                route=(actions[0],), score=None, nodes=1, truncated=False
            )

            selector.select(_snapshot(3), player="me", branch_seed=1)
            selector.select(_snapshot(3), player="me", branch_seed=1)
            third = selector.select(_snapshot(4), player="me", branch_seed=1)

        self.assertEqual(third.diagnostics["phase"], "model_first")
        self.assertEqual(selector.model_first_decisions, 2)
        self.assertEqual(selector.plan_decisions, 1)

    def test_two_players_keep_separate_turn_keys(self) -> None:
        """同一 turn 号下 me / opponent 是不同回合，不能共用 ``_first_done``。"""
        actions = (_action(1), _action(2))
        selector = _selector(actions, [0.9, 0.1])

        with patch("decision_engine.planning.turn_plan_arm.plan_turn") as fake_plan:
            fake_plan.return_value = SimpleNamespace(
                route=(actions[0],), score=None, nodes=1, truncated=False
            )
            me_first = selector.select(_snapshot(3), player="me", branch_seed=1)
            opp_first = selector.select(_snapshot(3), player="opponent", branch_seed=1)

        self.assertEqual(me_first.diagnostics["phase"], "model_first")
        self.assertEqual(opp_first.diagnostics["phase"], "model_first")

    def test_build_arm_exposes_config_hash(self) -> None:
        arm = build_mixed_first_action_arm(
            "mixed",
            MixedFirstActionConfig(max_actions=12),
            enumerator_factory=lambda _predictor: _FixedEnumerator([_action(1)]),
            prior=_FixedPrior([1.0]),
        )
        self.assertEqual(arm.name, "mixed")
        self.assertEqual(arm.config["arm"], "mixed_first_action")
        self.assertEqual(arm.config["max_actions"], 12)
        self.assertTrue(arm.config_hash())


if __name__ == "__main__":
    unittest.main()
