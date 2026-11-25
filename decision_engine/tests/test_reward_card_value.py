"""分档奖励卡定价：解锁时必须把被没收的势守恒地转移到奖励卡上。

背景：坦忒拉（10834110，阈值 10/20）打出时各给一张 **1 费**奖励卡——苍奏之四
（90034340，全场 5 伤）与宏大的回归（90034350，双击）。这两张卡牌效约等于 5 费卡，
若按普通手牌 w_hand=1.5PP 计，解锁瞬间势会凭空蒸发：没收 Φ(20)=10PP 只换回 3PP，
净 −7PP ⇒ "越攒越不该打"，与真实收益方向相反。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from decision_engine.cards.spellboost import (
    boost_reward_card_spans,
    build_boost_reward_card_spans,
)
from decision_engine.planning.incremental_value import (
    DEFAULT_WEIGHTS,
    _boost_potential,
    _hand_value,
)

TANTRA = 10834110
CANGZOU = 90034340   # 苍奏之四：10 档奖励
HONGDA = 90034350    # 宏大的回归：20 档奖励


def _thr(card_id: int) -> tuple[int, ...]:
    return (10, 20) if card_id == TANTRA else ()


def _card(card_id: int):
    return SimpleNamespace(base_card_id=card_id)


def _hand(cards, weights=DEFAULT_WEIGHTS) -> float:
    """把卡列表包成 (snapshot, view) 调 _hand_value——view.hand 存的是 uid。"""
    by_uid = {i: card for i, card in enumerate(cards)}
    snapshot = SimpleNamespace(cards_by_uid=by_uid)
    view = SimpleNamespace(hand=list(by_uid))
    return _hand_value(snapshot, view, weights)


class RewardCardSpanTableTest(unittest.TestCase):
    def test_tantra_reward_cards_carry_ten_boost_spans(self) -> None:
        spans = build_boost_reward_card_spans()
        self.assertEqual(spans.get(CANGZOU), 10)
        self.assertEqual(spans.get(HONGDA), 10)

    def test_cache_matches_builder(self) -> None:
        self.assertEqual(boost_reward_card_spans(), build_boost_reward_card_spans())


class RewardCardConservationTest(unittest.TestCase):
    def test_two_reward_cards_equal_phi_at_cap(self) -> None:
        """核心守恒：没收 Φ(20)=100 分，两张奖励卡正好承接 100 分。"""
        phi_at_cap = _boost_potential(20, TANTRA, _thr, DEFAULT_WEIGHTS)
        hand_gain = _hand([_card(CANGZOU), _card(HONGDA)], DEFAULT_WEIGHTS)
        self.assertAlmostEqual(phi_at_cap, 100.0)
        self.assertAlmostEqual(hand_gain, 100.0)
        self.assertAlmostEqual(phi_at_cap, hand_gain)

    def test_single_reward_card_matches_mid_tier_increment(self) -> None:
        """10 档只给苍奏之四：它承接 50 分，即 Φ(10) 中属于 10 档那部分。"""
        self.assertAlmostEqual(_hand([_card(CANGZOU)], DEFAULT_WEIGHTS), 50.0)

    def test_normal_card_still_uses_w_hand(self) -> None:
        self.assertAlmostEqual(_hand([_card(TANTRA)], DEFAULT_WEIGHTS), 15.0)

    def test_reward_value_scales_with_w_boost(self) -> None:
        """奖励卡价值随 w_boost 缩放，否则与没收额脱钩。"""
        from dataclasses import replace
        w = replace(DEFAULT_WEIGHTS, w_boost=2.5)
        self.assertAlmostEqual(_hand([_card(CANGZOU)], w), 25.0)
        self.assertAlmostEqual(_boost_potential(20, TANTRA, _thr, w), 50.0)


class CashOutMonotonicityTest(unittest.TestCase):
    """打出时"没收 − 奖励卡承接"的净额，不应随 sb 增大而变差（旧口径的缺陷）。"""

    def _net_clawback(self, sb: int) -> float:
        """没收额 − 已解锁奖励卡承接额（越小越好；负 = 打出反而赚）。"""
        clawback = _boost_potential(sb, TANTRA, _thr, DEFAULT_WEIGHTS)
        gained = 0.0
        if sb >= 10:
            gained += _hand([_card(CANGZOU)], DEFAULT_WEIGHTS)
        if sb >= 20:
            gained += _hand([_card(HONGDA)], DEFAULT_WEIGHTS)
        return clawback - gained

    def test_full_tier_is_fully_covered(self) -> None:
        self.assertAlmostEqual(self._net_clawback(20), 0.0)

    def test_mid_tier_loses_only_the_tail_progress(self) -> None:
        """10 档兑现亏 25 分 = 为 20 档积累的一半进度（Φ(10)−50）。"""
        self.assertAlmostEqual(self._net_clawback(10), 25.0)

    def test_no_longer_penalises_waiting(self) -> None:
        """满档打出的净扣减（0）不得高于低档（25），即"越攒越亏"已被修复。"""
        self.assertLessEqual(self._net_clawback(20), self._net_clawback(10))


if __name__ == "__main__":
    unittest.main()
