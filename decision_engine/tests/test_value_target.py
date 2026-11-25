# -*- coding: utf-8 -*-
"""value_target 单元测试：4回合窗口净资源交换 + 先后手节奏基准。"""
import unittest
from math import tanh
from decision_engine.planning.value_target import (
    ValueTargetWeights, compute_value_target, compute_value_target_net,
)
from decision_engine.state.snapshot import BattleSnapshot, PlayerView, LegalView, ExtraPpView

W = ValueTargetWeights()


def _snap(*, my_life=20, opp_life=20, my_hand=0, opp_hand=0, my_ep=0, opp_ep=0,
          my_sep=0, opp_sep=0, my_xpp=0, opp_xpp=0, my_result=0, opp_result=0,
          my_first=True, opp_first=False):
    """默认 root(me)侧为先手，方便资源类测试隔离先后手基准（统一 +w_first）。"""
    def mk(life, hand_n, ep, sep, xpp, result, first):
        return PlayerView(
            life=life, max_life=20, pp=10, max_pp=10, ep=ep, sep=sep,
            hand=tuple(range(hand_n)), result=result, is_first_side=first,
            extra_pp=ExtraPpView(extra_pp=xpp) if xpp else None,
        )
    return BattleSnapshot(
        me=mk(my_life, my_hand, my_ep, my_sep, my_xpp, my_result, my_first),
        opponent=mk(opp_life, opp_hand, opp_ep, opp_sep, opp_xpp, opp_result, opp_first),
        cards_by_uid={}, legal=LegalView(),
    )


# root=me 且 me 为先手时的基准偏移（所有资源类测试期望都要加上）
FB = W.w_first  # first-side bonus (+w_first)


class TestValueTargetNet(unittest.TestCase):
    """资源交换项测试（root=me 设为先手，期望值含 +w_first 基准）。"""

    def test_net_face_damage(self):
        start = _snap(opp_life=20)
        end = _snap(opp_life=10)
        net = compute_value_target_net(start, end, "me", weights=W)
        self.assertAlmostEqual(net, 10 * W.w_life + FB, places=5)

    def test_net_symmetric_life(self):
        start = _snap(my_life=20, opp_life=20)
        end = _snap(my_life=18, opp_life=14)
        net = compute_value_target_net(start, end, "me", weights=W)
        self.assertAlmostEqual(net, (6 - 2) * W.w_life + FB, places=5)

    def test_no_phantom_loss_from_heal(self):
        """掉血后回血净0(资源项)；总 NET = 先后手基准 +w_first。"""
        start = _snap(my_life=20)
        end = _snap(my_life=20)
        net = compute_value_target_net(start, end, "me", weights=W)
        self.assertAlmostEqual(net, FB, places=5)  # 资源项0，仅剩基准

    def test_net_opponent_ep_consumed(self):
        start = _snap(opp_ep=2)
        end = _snap(opp_ep=0)
        net = compute_value_target_net(start, end, "me", weights=W)
        self.assertAlmostEqual(net, 2 * W.w_ep + FB, places=5)

    def test_net_my_sep_consumed_negative(self):
        start = _snap(my_sep=1)
        end = _snap(my_sep=0)
        net = compute_value_target_net(start, end, "me", weights=W)
        self.assertAlmostEqual(net, -W.w_sep + FB, places=5)

    def test_net_hand_advantage(self):
        start = _snap(my_hand=3, opp_hand=3)
        end = _snap(my_hand=4, opp_hand=2)
        net = compute_value_target_net(start, end, "me", weights=W)
        self.assertAlmostEqual(net, (4 - 2 - 0) * W.w_hand + FB, places=5)

    def test_net_extra_pp_held(self):
        start = _snap()
        end = _snap(my_xpp=1)
        net = compute_value_target_net(start, end, "me", weights=W)
        self.assertAlmostEqual(net, W.w_xpp + FB, places=5)

    def test_terminal_win(self):
        """终局不叠加先后手基准，直接 +v_win。"""
        start = _snap(opp_life=5)
        end = _snap(opp_life=0)
        net = compute_value_target_net(start, end, "me", weights=W)
        self.assertAlmostEqual(net, W.v_win, places=5)

    def test_terminal_loss(self):
        start = _snap(my_life=3)
        end = _snap(my_life=0)
        net = compute_value_target_net(start, end, "me", weights=W)
        self.assertAlmostEqual(net, -W.v_win, places=5)


class TestFirstSideTempo(unittest.TestCase):
    """先后手节奏基准（PP节奏框架）：先手[N,N]持平/后手[N,N+1]慢一费/额外PP临时反转。"""

    def test_first_side_bonus_only(self):
        """先手、无任何额外PP、无交换 → NET=+w_first(纯先手基准)。"""
        s = _snap(my_first=True, opp_first=False)
        net = compute_value_target_net(s, s, "me", weights=W)
        self.assertAlmostEqual(net, W.w_first, places=5)

    def test_real_opening_first_side_net_zero(self):
        """真实开局(先手视角):对方后手持1额外PP。
        先手基准 +w_first，对方额外PP项 w_xpp×(0−1)=−w_xpp → NET=0([N,N]持平)。"""
        s = _snap(my_first=True, opp_first=False, opp_xpp=1)
        net = compute_value_target_net(s, s, "me", weights=W)
        self.assertAlmostEqual(net, 0.0, places=5)

    def test_real_opening_second_side_net_zero(self):
        """真实开局(后手视角):我方后手持1额外PP。
        后手基准 −w_first，我方额外PP +w_xpp → NET=0(额外PP临时反转[N+1,N+1])。"""
        s = _snap(my_first=False, opp_first=True, my_xpp=1)
        net = compute_value_target_net(s, s, "me", weights=W)
        self.assertAlmostEqual(net, 0.0, places=5)

    def test_second_side_no_xpp_is_negative_one_pp(self):
        """后手、无额外PP、无交换 → NET=−w_first([N,N+1]慢一费=−1PP)。"""
        s = _snap(my_first=False, opp_first=True)
        net = compute_value_target_net(s, s, "me", weights=W)
        self.assertAlmostEqual(net, -W.w_first, places=5)

    def test_first_side_slightly_better_than_second(self):
        """同样无交换,先手 NET > 后手(无额外PP) NET。"""
        s1 = _snap(my_first=True, opp_first=False)
        net_first = compute_value_target_net(s1, s1, "me", weights=W)
        s2 = _snap(my_first=False, opp_first=True)
        net_second = compute_value_target_net(s2, s2, "me", weights=W)
        self.assertGreater(net_first, net_second)
        self.assertAlmostEqual(net_first - net_second, 2 * W.w_first, places=5)

    def test_xpp_consumed_value_conserved(self):
        """后手用掉额外PP多打1费→变成场面/血量(此处用打脸1费=+w_life×...)。
        额外PP归0(−不再+w_xpp),但兑现的收益被血量项接住 → 不丢分。"""
        # start: 后手持1额外PP; end: 额外PP用掉(0),但打了对方相当于1PP的脸(1PP=10分=1血这里简化打1血)
        s = _snap(my_first=False, opp_first=True, my_xpp=1, opp_life=20)
        e = _snap(my_first=False, opp_first=True, my_xpp=0, opp_life=19)  # 用额外PP打1脸
        net = compute_value_target_net(s, e, "me", weights=W)
        # 资源: +1×w_life(打脸) + 0(xpp归0); 基准 −w_first
        self.assertAlmostEqual(net, W.w_life - W.w_first, places=5)
        # 关键:额外PP兑现成打脸后,净值不低于"持有额外PP不动"(=−w_first+0... 实为0)
        # 持有不动 NET=0; 用掉打1血 NET = w_life - w_first = 10-10 = 0 → 至少持平
        self.assertGreaterEqual(net, 0.0 - 1e-9)


class TestValueTargetTanh(unittest.TestCase):
    def test_tanh_range(self):
        start = _snap(opp_life=20)
        end = _snap(opp_life=15)
        v = compute_value_target(start, end, "me", weights=W)
        self.assertTrue(-1.0 < v < 1.0)
        self.assertAlmostEqual(v, tanh((5 * W.w_life + FB) / W.scale), places=5)

    def test_terminal_saturates(self):
        start = _snap(opp_life=5)
        end = _snap(opp_life=0)
        v = compute_value_target(start, end, "me", weights=W)
        # 终局 NET=+v_win → tanh(v_win/scale)，接近但不必到 0.99（scale=90 时 =0.977）
        self.assertAlmostEqual(v, tanh(W.v_win / W.scale), places=5)
        self.assertGreater(v, 0.95)

    def test_root_player_symmetry(self):
        """me(先手)视角 vs opponent(后手)视角:资源零和,先后手基准也零和 → 整体对称。"""
        start = _snap(opp_life=20, my_first=True, opp_first=False)
        end = _snap(opp_life=10, my_first=True, opp_first=False)
        v_me = compute_value_target(start, end, "me", weights=W)
        v_opp = compute_value_target(start, end, "opponent", weights=W)
        self.assertAlmostEqual(v_me, -v_opp, places=5)


if __name__ == "__main__":
    unittest.main()
