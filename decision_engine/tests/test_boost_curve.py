"""阈值合并（boost_single_tier）与凸进度势（w_boost_curve_power）的形状测试。"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace

from decision_engine.planning.incremental_value import (
    DEFAULT_WEIGHTS,
    StepRewardWeights,
    _boost_potential,
    weights_from_env,
)

# 坦忒拉 10834110：阈值 10 / 20，w_boost=5 ⇒ reward 各 50 分，总量 100 分 = 10 PP。
TANTRA = 10834110
THRESHOLDS = (10, 20)


def _thr(base_card_id: int) -> tuple[int, ...]:
    return THRESHOLDS if base_card_id == TANTRA else ()


def _phi(sb: int, w: StepRewardWeights) -> float:
    return _boost_potential(sb, TANTRA, _thr, w)


class BoostCurveDefaultsTest(unittest.TestCase):
    def test_default_keeps_multi_tier_shape(self) -> None:
        """默认参数必须与旧行为逐点一致（10 档折点仍在）。"""
        self.assertEqual(_phi(0, DEFAULT_WEIGHTS), 0.0)
        self.assertAlmostEqual(_phi(5, DEFAULT_WEIGHTS), 37.5)
        self.assertAlmostEqual(_phi(10, DEFAULT_WEIGHTS), 75.0)
        self.assertAlmostEqual(_phi(15, DEFAULT_WEIGHTS), 87.5)
        self.assertAlmostEqual(_phi(20, DEFAULT_WEIGHTS), 100.0)

    def test_default_marginal_collapses_after_mid_tier(self) -> None:
        """默认口径的缺陷本身要被固定下来：跨过 10 档后边际从 7.5 掉到 2.5。"""
        before = _phi(10, DEFAULT_WEIGHTS) - _phi(9, DEFAULT_WEIGHTS)
        after = _phi(11, DEFAULT_WEIGHTS) - _phi(10, DEFAULT_WEIGHTS)
        self.assertAlmostEqual(before, 7.5)
        self.assertAlmostEqual(after, 2.5)


class BoostSingleTierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.w = replace(DEFAULT_WEIGHTS, boost_single_tier=True)

    def test_total_is_conserved(self) -> None:
        self.assertAlmostEqual(_phi(20, self.w), 100.0)

    def test_curve_is_linear_with_constant_marginal(self) -> None:
        for sb in range(0, 21):
            self.assertAlmostEqual(_phi(sb, self.w), 5.0 * sb)
        first = _phi(1, self.w) - _phi(0, self.w)
        last = _phi(20, self.w) - _phi(19, self.w)
        self.assertAlmostEqual(first, 5.0)
        self.assertAlmostEqual(first, last)

    def test_mid_tier_is_no_longer_a_breakpoint(self) -> None:
        """单档化后 9→10→11 的边际必须一致（旧口径是 7.5 / 2.5 的断崖）。"""
        d9 = _phi(10, self.w) - _phi(9, self.w)
        d10 = _phi(11, self.w) - _phi(10, self.w)
        self.assertAlmostEqual(d9, d10)


class BoostCurvePowerTest(unittest.TestCase):
    def test_power_two_moves_value_to_the_tail(self) -> None:
        w = replace(DEFAULT_WEIGHTS, w_boost_curve_power=2.0)
        # 线性是 75；凸形把 10 档处压到 50 + 50*0.25 = 62.5
        self.assertAlmostEqual(_phi(10, w), 62.5)
        self.assertAlmostEqual(_phi(20, w), 100.0)  # 总量守恒
        late = _phi(20, w) - _phi(19, w)
        early = _phi(1, w) - _phi(0, w)
        self.assertGreater(late, early)

    def test_power_two_with_single_tier(self) -> None:
        w = replace(DEFAULT_WEIGHTS, boost_single_tier=True, w_boost_curve_power=2.0)
        self.assertAlmostEqual(_phi(10, w), 25.0)
        self.assertAlmostEqual(_phi(20, w), 100.0)
        self.assertAlmostEqual(_phi(0, w), 0.0)

    def test_monotone_non_decreasing(self) -> None:
        for power in (0.5, 1.0, 2.0, 3.0):
            w = replace(DEFAULT_WEIGHTS, w_boost_curve_power=power)
            prev = _phi(0, w)
            for sb in range(1, 25):
                cur = _phi(sb, w)
                self.assertGreaterEqual(cur, prev - 1e-9)


class BoostCurveNoThresholdTest(unittest.TestCase):
    def test_linear_card_is_unaffected_by_curve_params(self) -> None:
        for w in (
            DEFAULT_WEIGHTS,
            replace(DEFAULT_WEIGHTS, boost_single_tier=True),
            replace(DEFAULT_WEIGHTS, w_boost_curve_power=3.0),
        ):
            self.assertAlmostEqual(_phi_linear(7, w), 35.0)


def _phi_linear(sb: int, w: StepRewardWeights) -> float:
    return _boost_potential(sb, 99999999, _thr, w)


class BoostGateRewardTest(unittest.TestCase):
    """闸门奖励：跨阈一次性发放，且不进 Φ（所以打出时不回收）。"""

    def test_off_by_default(self) -> None:
        self.assertEqual(_gate(9, 10, DEFAULT_WEIGHTS), 0.0)

    def test_only_pays_on_the_crossing_step(self) -> None:
        w = replace(DEFAULT_WEIGHTS, w_boost_gate=1.0)
        self.assertAlmostEqual(_gate(9, 10, w), 50.0)    # 跨 10 档：reward(10)=10×5
        self.assertAlmostEqual(_gate(10, 11, w), 0.0)    # 没跨新阈值
        self.assertAlmostEqual(_gate(19, 20, w), 50.0)   # 跨 20 档：reward(20)=10×5
        self.assertAlmostEqual(_gate(20, 21, w), 0.0)

    def test_crossing_two_tiers_in_one_step_pays_both(self) -> None:
        w = replace(DEFAULT_WEIGHTS, w_boost_gate=1.0)
        self.assertAlmostEqual(_gate(9, 21, w), 100.0)

    def test_coefficient_scales_it(self) -> None:
        w = replace(DEFAULT_WEIGHTS, w_boost_gate=2.0)
        self.assertAlmostEqual(_gate(9, 10, w), 100.0)

    def test_gate_reward_is_not_reclaimed_when_card_leaves_hand(self) -> None:
        """闸门奖励进了 reward 就再也不扣回：这是它与进度势的关键区别。

        进度势 Φ(20)=100，打出时 3b 块扣 100；闸门只发 100，不产生持有势，
        所以 `_boost_potential` 在 gate 开启时仍是 100（与 gate 无关），回收额为 0。
        """
        w = replace(DEFAULT_WEIGHTS, w_boost_gate=1.0)
        self.assertAlmostEqual(_phi(20, w), 100.0)  # Φ 不受 gate 影响
        # 回收量由 w_boost_potential_scale 决定，与 gate 无关
        self.assertAlmostEqual(_gate(0, 20, w), 100.0)  # 一次跨越两档共发 100


def _gate(prev_sb: int, next_sb: int, w: StepRewardWeights) -> float:
    from decision_engine.planning.incremental_value import _boost_gate_reward
    return _boost_gate_reward(prev_sb, next_sb, TANTRA, _thr, w)


class BoostReleaseAtCapTest(unittest.TestCase):
    """满档释放：只有攒到最高阈值的卡在打出时不再回收持有势。"""

    def test_at_cap_detection(self) -> None:
        from decision_engine.planning.incremental_value import _at_boost_cap
        self.assertTrue(_at_boost_cap(20, TANTRA, _thr))
        self.assertTrue(_at_boost_cap(25, TANTRA, _thr))
        self.assertFalse(_at_boost_cap(19, TANTRA, _thr))
        self.assertFalse(_at_boost_cap(0, TANTRA, _thr))
        # 无阈值卡永不"满档"，否则线性增幅卡会被误判为免费打出
        self.assertFalse(_at_boost_cap(99, 99999999, _thr))

    def test_default_off_keeps_clawback(self) -> None:
        self.assertFalse(DEFAULT_WEIGHTS.boost_release_at_cap)

    def test_release_keeps_low_tier_costly(self) -> None:
        """释放只作用于满档：Φ(19)=97.5 仍会被回收，所以早打依然有代价。"""
        self.assertAlmostEqual(_phi(19, DEFAULT_WEIGHTS), 97.5)
        self.assertAlmostEqual(_phi(20, DEFAULT_WEIGHTS), 100.0)


class BoostCurveEnvTest(unittest.TestCase):
    """实验编排靠 EP_* 环境变量，布尔字段必须能被 1/0 覆盖。"""

    def setUp(self) -> None:
        self._saved = dict(os.environ)

    def tearDown(self) -> None:
        for key in ("EP_BOOST_SINGLE_TIER", "EP_W_BOOST_CURVE_POWER"):
            os.environ.pop(key, None)
        os.environ.update(self._saved)

    def test_env_can_enable_single_tier_and_curve(self) -> None:
        os.environ["EP_BOOST_SINGLE_TIER"] = "1"
        os.environ["EP_W_BOOST_CURVE_POWER"] = "2.5"
        w = weights_from_env()
        self.assertTrue(w.boost_single_tier)
        self.assertAlmostEqual(w.w_boost_curve_power, 2.5)

    def test_env_bool_accepts_false_strings(self) -> None:
        os.environ["EP_BOOST_SINGLE_TIER"] = "false"
        self.assertFalse(weights_from_env().boost_single_tier)

    def test_unset_env_keeps_defaults(self) -> None:
        os.environ.pop("EP_BOOST_SINGLE_TIER", None)
        os.environ.pop("EP_W_BOOST_CURVE_POWER", None)
        w = weights_from_env()
        self.assertFalse(w.boost_single_tier)
        self.assertAlmostEqual(w.w_boost_curve_power, 1.0)


if __name__ == "__main__":
    unittest.main()
