# -*- coding: utf-8 -*-
"""增幅"预支回报"势：在手即拥有完成积累的回报，提前打出则丢失它。

设计（用户提出）：保留原有的"每点增幅积累势"不变，另外给每张在手、有饱和点的
增幅卡一个**常量预支项 R = 最高阈值总 reward**。于是：

- 攒到几档都一样，只要卡在手就带着 R；
- 提前打出 → 卡离手 → R 与已攒进度一起消失 = 付出 R 的代价；
- 攒满 → 进度势(R) + 预支(R) = 2R（翻倍）；
- 兑现 → 一并扣除。

关键性质：R 是**状态函数**上的常量项，所以 `_boost_marginal_value`（两点差）完全不变，
"被增幅时积累势"的行为原样保留，且首尾差 == 逐边累积（PBRS 成立）。
"""

from __future__ import annotations

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.incremental_value import (
    StepRewardWeights,
    _boost_marginal_value,
    _boost_potential,
    default_threshold_fn,
    step_reward,
)
from decision_engine.state.snapshot import (
    BattleSnapshot, CardType, CardView, LegalView, PlayerView,
)

THRESHOLD_FN = default_threshold_fn()
BID = 10834110  # 10/20 档
BASE_W = 5.0


def _w(**overrides) -> StepRewardWeights:
    """本文件的默认：预支开启（生产默认关闭，见 StepRewardWeights 注释）。"""
    base: dict[str, float] = {"w_boost_anticipation": 1.0}
    base.update(overrides)
    return StepRewardWeights(**base)


class AnticipationPotentialTest(unittest.TestCase):
    def test_zero_progress_has_no_anticipation_in_progressive_mode(self) -> None:
        """渐进模式（默认 power=2）：一点没攒的卡不该被预支，否则会把卡变成不动产。"""
        w = _w()
        self.assertAlmostEqual(_boost_potential(0, BID, THRESHOLD_FN, w), 0.0)

    def test_constant_mode_when_power_zero(self) -> None:
        """power=0 退回"整份常量"语义：零进度也给 R（§12.20 的行为，已知会诱发囤积）。"""
        w = _w(w_boost_anticipation_power=0.0)
        self.assertAlmostEqual(_boost_potential(0, BID, THRESHOLD_FN, w), 20 * BASE_W)

    def test_progressive_anticipation_is_monotone(self) -> None:
        w = _w()
        values = [_boost_potential(sb, BID, THRESHOLD_FN, w) for sb in (0, 5, 10, 15, 20)]
        self.assertEqual(values, sorted(values))

    def test_full_progress_is_double(self) -> None:
        w = _w()
        full = _boost_potential(20, BID, THRESHOLD_FN, w)
        progress_only = _boost_potential(
            20, BID, THRESHOLD_FN, _w(w_boost_anticipation=0.0)
        )
        self.assertAlmostEqual(full, 2 * progress_only)

    def test_constant_mode_keeps_marginal_unchanged(self) -> None:
        """整份常量模式（power=0）下边际必须与关闭预支时完全一致。"""
        with_pre = _w(w_boost_anticipation_power=0.0)
        without = _w(w_boost_anticipation=0.0)
        for prev_sb, next_sb in ((0, 1), (3, 7), (9, 10), (10, 20)):
            with self.subTest(prev=prev_sb, next=next_sb):
                self.assertAlmostEqual(
                    _boost_marginal_value(prev_sb, next_sb, BID, THRESHOLD_FN, with_pre),
                    _boost_marginal_value(prev_sb, next_sb, BID, THRESHOLD_FN, without),
                )

    def test_progressive_mode_adds_marginal_near_threshold(self) -> None:
        """渐进模式下，接近满档的边际应高于进度势本身的边际（临门一脚被放大）。"""
        progressive = _w()
        progress_only = _w(w_boost_anticipation=0.0)
        near = _boost_marginal_value(18, 19, BID, THRESHOLD_FN, progressive)
        baseline = _boost_marginal_value(18, 19, BID, THRESHOLD_FN, progress_only)
        self.assertGreater(near, baseline)

    def test_anticipation_off_restores_old_semantics(self) -> None:
        w = _w(w_boost_anticipation=0.0)
        self.assertAlmostEqual(_boost_potential(0, BID, THRESHOLD_FN, w), 0.0)

    def test_no_cap_card_has_no_anticipation(self) -> None:
        """线性成长型（无阈值）不应被预支，否则会凭空抬高所有增幅卡。"""
        w = _w()
        self.assertAlmostEqual(_boost_potential(0, 10131320, THRESHOLD_FN, w), 0.0)
        self.assertAlmostEqual(
            _boost_potential(3, 10131320, THRESHOLD_FN, w), 3 * w.w_boost_linear
        )


def _snap(*, hand=(), life=20, opp_life=20) -> BattleSnapshot:
    cards, h = {}, []
    for card in hand:
        cards[card.uid] = card
        h.append(card.uid)
    return BattleSnapshot(
        me=PlayerView(life=life, max_life=20, pp=10, max_pp=10, hand=tuple(h)),
        opponent=PlayerView(life=opp_life, max_life=20, pp=10, max_pp=10),
        cards_by_uid=cards, legal=LegalView(), is_ally_turn=True,
    )


def _boost_card(uid: int, sb: int) -> CardView:
    return CardView(uid=uid, card_id=BID, base_card_id=BID, atk=5, life=5, max_life=5,
                    cost=6, card_type=CardType.FOLLOWER,
                    has_spell_boost=True, spell_boost_count=sb)


class EarlyCashOutCostTest(unittest.TestCase):
    """端到端：提前打出这张卡，要比"留着"多丢一份 R。"""

    def test_playing_early_costs_anticipation(self) -> None:
        """开启预支后，"提前打出"的代价恰好比不开启时多出 R。"""
        action = LegalAction(kind=ActionKind.PLAY, source_uid=1, description="")
        before = _snap(hand=[_boost_card(1, 8)])
        after = _snap(hand=[])
        with_anticipation = step_reward(
            before, action, after, "me",
            weights=_w(w_boost_anticipation_power=0.0), threshold_fn=THRESHOLD_FN,
        )
        without = step_reward(
            before, action, after, "me", weights=_w(w_boost_anticipation=0.0),
            threshold_fn=THRESHOLD_FN,
        )
        anticipation = 20 * BASE_W
        self.assertAlmostEqual(with_anticipation - without, -anticipation)

    def test_holding_through_turn_end_costs_nothing(self) -> None:
        w = _w()
        action = LegalAction(kind=ActionKind.TURN_END, source_uid=0, description="")
        before = _snap(hand=[_boost_card(1, 8)])
        after = _snap(hand=[_boost_card(1, 8)])
        reward = step_reward(before, action, after, "me", weights=w, threshold_fn=THRESHOLD_FN)
        self.assertAlmostEqual(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
