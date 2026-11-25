# -*- coding: utf-8 -*-
"""通用换牌策略（CurveMulliganSelector）的契约测试。

只测"给定手牌费用 → 换掉哪些 uid"，不依赖真实卡牌数据与对局执行。
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from decision_engine.planning.mulligan_policy import (
    CurveMulliganConfig,
    CurveMulliganSelector,
    build_curve_mulligan_selector,
)


@dataclass(frozen=True)
class _Card:
    uid: int
    cost: int


@dataclass(frozen=True)
class _View:
    hand: tuple[int, ...]


@dataclass(frozen=True)
class _Snapshot:
    me: _View
    opponent: _View
    cards_by_uid: dict = field(default_factory=dict)


def _snapshot(me_costs: tuple[int, ...], opp_costs: tuple[int, ...] = ()) -> _Snapshot:
    cards: dict = {}
    uid = 1
    me_hand = []
    for cost in me_costs:
        cards[uid] = _Card(uid=uid, cost=cost)
        me_hand.append(uid)
        uid += 1
    opp_hand = []
    for cost in opp_costs:
        cards[uid] = _Card(uid=uid, cost=cost)
        opp_hand.append(uid)
        uid += 1
    return _Snapshot(me=_View(hand=tuple(me_hand)), opponent=_View(hand=tuple(opp_hand)), cards_by_uid=cards)


class CurveMulliganSelectorTest(unittest.TestCase):
    def test_keeps_full_low_curve(self) -> None:
        selector = CurveMulliganSelector()
        snap = _snapshot((1, 2, 3, 4))
        self.assertEqual(selector.select_mulligan(snap, player="me"), ())

    def test_changes_expensive_cards(self) -> None:
        selector = CurveMulliganSelector()
        snap = _snapshot((1, 7, 9, 2))
        # 7 和 9 费都超过 keep_max_cost=4，按从贵到便宜返回
        self.assertEqual(selector.select_mulligan(snap, player="me"), (3, 2))

    def test_respects_max_changes(self) -> None:
        selector = CurveMulliganSelector(CurveMulliganConfig(max_changes=2))
        snap = _snapshot((6, 7, 8, 9))
        self.assertEqual(selector.select_mulligan(snap, player="me"), (4, 3))

    def test_max_changes_zero_changes_nothing(self) -> None:
        selector = CurveMulliganSelector(CurveMulliganConfig(max_changes=0))
        snap = _snapshot((6, 7, 8, 9))
        self.assertEqual(selector.select_mulligan(snap, player="me"), ())

    def test_targets_opponent_side_by_player(self) -> None:
        selector = CurveMulliganSelector()
        snap = _snapshot((1, 2), (8, 1))
        self.assertEqual(selector.select_mulligan(snap, player="me"), ())
        self.assertEqual(selector.select_mulligan(snap, player="opponent"), (3,))

    def test_missing_card_is_skipped_not_treated_as_zero_cost(self) -> None:
        selector = CurveMulliganSelector()
        # hand 里有一个 uid 不在 cards_by_uid 中：跳过它，不猜费用、不崩溃。
        snap = _Snapshot(me=_View(hand=(1, 999)), opponent=_View(hand=()), cards_by_uid={1: _Card(uid=1, cost=3)})
        self.assertEqual(selector.select_mulligan(snap, player="me"), ())

    def test_threshold_boundary_is_inclusive(self) -> None:
        selector = CurveMulliganSelector(CurveMulliganConfig(keep_max_cost=4))
        snap = _snapshot((4, 5))
        # cost=4 保留（<=4），cost=5 换掉
        self.assertEqual(selector.select_mulligan(snap, player="me"), (2,))

    def test_rejects_invalid_player(self) -> None:
        selector = CurveMulliganSelector()
        with self.assertRaises(ValueError):
            selector.select_mulligan(_snapshot((1,)), player="third")

    def test_rejects_inverted_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            CurveMulliganSelector(CurveMulliganConfig(always_keep_max_cost=5, keep_max_cost=3))

    def test_builder_matches_constructor(self) -> None:
        built = build_curve_mulligan_selector(max_changes=1)
        snap = _snapshot((6, 7))
        self.assertEqual(built.select_mulligan(snap, player="me"), (2,))


if __name__ == "__main__":
    unittest.main()
