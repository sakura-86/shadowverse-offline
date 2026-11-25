# -*- coding: utf-8 -*-
"""`me/opponent.field_has_base_card_id` 条件测试。

对应卡面：「自己的回合开始时，若自己的战场上没有『侠气贯流』，则召唤1张…」
目标卡 id 由 `condition.ref` 传入（subject 是静态字符串，不能自带参数）。
"""

from __future__ import annotations

import unittest

from decision_engine.cards.templates import ConditionSpec
from decision_engine.execution.condition_evaluator import ConditionEvaluationContext, ConditionEvaluator
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView


def _card(uid: int, base_card_id: int) -> CardView:
    return CardView(uid=uid, card_id=base_card_id, base_card_id=base_card_id, name=f"c{base_card_id}", card_type=CardType.AMULET)


def _snapshot(*, me_field=(), opp_field=(), cards=None) -> BattleSnapshot:
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=0, max_pp=0, field=tuple(me_field)),
        opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0, field=tuple(opp_field)),
        cards_by_uid=cards or {},
        source="test",
    )


class FieldHasBaseCardIdConditionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = ConditionEvaluator()
        self.cards = {
            1: _card(1, 86714210),
            2: _card(2, 90021210),
        }

    def _value(self, snapshot: BattleSnapshot, subject: str, ref) -> int | None:
        condition = ConditionSpec(kind="snapshot_threshold", subject=subject, operator="==", value=0, ref=ref)
        return self.evaluator._condition_subject_value(
            snapshot, subject, ConditionEvaluationContext(action=None), condition
        )

    def test_returns_one_when_card_present_on_own_field(self) -> None:
        snapshot = _snapshot(me_field=(1, 2), cards=self.cards)
        self.assertEqual(self._value(snapshot, "me.field_has_base_card_id", "86714210"), 1)

    def test_returns_zero_when_card_absent(self) -> None:
        snapshot = _snapshot(me_field=(2,), cards=self.cards)
        self.assertEqual(self._value(snapshot, "me.field_has_base_card_id", "86714210"), 0)

    def test_opponent_prefix_checks_opponent_field(self) -> None:
        snapshot = _snapshot(me_field=(2,), opp_field=(1,), cards=self.cards)
        self.assertEqual(self._value(snapshot, "me.field_has_base_card_id", "86714210"), 0)
        self.assertEqual(self._value(snapshot, "opponent.field_has_base_card_id", "86714210"), 1)

    def test_missing_ref_is_unsupported_not_zero(self) -> None:
        snapshot = _snapshot(me_field=(1,), cards=self.cards)
        self.assertIsNone(self._value(snapshot, "me.field_has_base_card_id", None))

    def test_non_numeric_ref_is_unsupported(self) -> None:
        snapshot = _snapshot(me_field=(1,), cards=self.cards)
        self.assertIsNone(self._value(snapshot, "me.field_has_base_card_id", "not-an-id"))

    def test_snapshot_threshold_equality_zero_matches_when_absent(self) -> None:
        """完整求值：`== 0` 在场上没有该卡时为真。"""
        snapshot = _snapshot(me_field=(2,), cards=self.cards)
        condition = ConditionSpec(
            kind="snapshot_threshold",
            subject="me.field_has_base_card_id",
            operator="==",
            value=0,
            ref="86714210",
        )
        result = self.evaluator.evaluate(snapshot, None, condition, ConditionEvaluationContext(action=None))
        self.assertIs(result, True)

    def test_snapshot_threshold_equality_zero_rejects_when_present(self) -> None:
        snapshot = _snapshot(me_field=(1,), cards=self.cards)
        condition = ConditionSpec(
            kind="snapshot_threshold",
            subject="me.field_has_base_card_id",
            operator="==",
            value=0,
            ref="86714210",
        )
        result = self.evaluator.evaluate(snapshot, None, condition, ConditionEvaluationContext(action=None))
        self.assertIs(result, False)


if __name__ == "__main__":
    unittest.main()
