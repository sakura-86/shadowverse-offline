# -*- coding: utf-8 -*-
"""snapshot_exists 条件（含 exclude_source / original_cost_eq_N 过滤）测试。"""

from __future__ import annotations

import unittest

from decision_engine.cards.templates import ConditionSpec
from decision_engine.execution.condition_evaluator import ConditionEvaluationContext, ConditionEvaluator
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView


def _card(uid: int, base_card_id: int, *, cost: int, card_type: CardType = CardType.FOLLOWER) -> CardView:
    return CardView(uid=uid, card_id=base_card_id, base_card_id=base_card_id, name=f"c{base_card_id}", cost=cost, card_type=card_type)


def _snapshot(cards: dict[int, CardView], field: tuple[int, ...]) -> BattleSnapshot:
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=0, max_pp=0, field=field),
        opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0),
        cards_by_uid=cards,
        source="test",
    )


class SnapshotExistsConditionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = ConditionEvaluator()
        # 10001110 不屈的剑斗士（印刷费用 2）；10501110 挥毫的怪物（印刷费用 1）；10624120 龙之启示（印刷费用 2）
        self.snapshot = _snapshot(
            {
                1: _card(1, 10001110, cost=2),
                2: _card(2, 10501110, cost=1),
                3: _card(3, 10624120, cost=2),
            },
            field=(1, 3),
        )

    def test_original_cost_filter_matches_other_card(self) -> None:
        condition = ConditionSpec(
            kind="snapshot_exists",
            subject="me.field",
            operator="exists",
            value=True,
            ref="filters:exclude_source_original_cost_eq_2",
        )
        context = ConditionEvaluationContext(action=None, source_card=self.snapshot.cards_by_uid[2])

        self.assertIs(self.evaluator.evaluate(self.snapshot, None, condition, context), True)

    def test_exclude_source_removes_only_source(self) -> None:
        # 场上只有 source 自己（uid=1），排除后不应存在。
        snapshot = _snapshot({1: _card(1, 10001110, cost=2)}, field=(1,))
        condition = ConditionSpec(
            kind="snapshot_exists",
            subject="me.field",
            operator="exists",
            value=True,
            ref="filters:exclude_source_original_cost_eq_2",
        )
        context = ConditionEvaluationContext(action=None, source_card=snapshot.cards_by_uid[1])

        self.assertIs(self.evaluator.evaluate(snapshot, None, condition, context), False)

    def test_unsupported_filter_returns_none(self) -> None:
        condition = ConditionSpec(
            kind="snapshot_exists",
            subject="me.field",
            operator="exists",
            value=True,
            ref="filters:some_unknown_filter",
        )
        context = ConditionEvaluationContext(action=None)

        self.assertIsNone(self.evaluator.evaluate(self.snapshot, None, condition, context))


if __name__ == "__main__":
    unittest.main()
