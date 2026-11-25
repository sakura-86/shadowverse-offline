# -*- coding: utf-8 -*-
"""Tests for incremental_value.step_reward（逐动作即时收益）。

S0：各项量纲、增幅逐动作不丢（变身/打出）、跨回合符号、终局。
"""

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.incremental_value import (
    StepRewardWeights,
    step_reward,
    _boost_marginal_value,
    build_spellboost_threshold_table,
    make_threshold_fn,
    default_threshold_fn,
)
from decision_engine.state.snapshot import (
    BattleSnapshot, CardType, CardView, CrestView, ExtraPpView, LegalView, PlayerView,
)

W = StepRewardWeights()


def _action(kind="play", uid=0):
    return LegalAction(kind=ActionKind(kind), source_uid=uid, description="")


def _card(uid, *, atk=0, life=0, base=0, ctype=CardType.FOLLOWER, sb=0, has_sb=False, cost=0):
    return CardView(uid=uid, base_card_id=base, atk=atk, life=life, max_life=life,
                    card_type=ctype, spell_boost_count=sb, has_spell_boost=has_sb, cost=cost)


def _snap(*, my_life=20, opp_life=20, my_field=None, opp_field=None, my_hand=None,
          my_ep=0, my_sep=0, my_max_pp=10, opp_ep=0, opp_sep=0,
          my_crests=(), opp_crests=()):
    my_field = my_field or []
    opp_field = opp_field or []
    my_hand = my_hand or []
    cards = {}
    mf, of, mh = [], [], []
    for c in my_field: cards[c.uid] = c; mf.append(c.uid)
    for c in opp_field: cards[c.uid] = c; of.append(c.uid)
    for c in my_hand: cards[c.uid] = c; mh.append(c.uid)
    return BattleSnapshot(
        me=PlayerView(life=my_life, max_life=20, pp=10, max_pp=my_max_pp,
                      field=tuple(mf), hand=tuple(mh), ep=my_ep, sep=my_sep, crests=tuple(my_crests)),
        opponent=PlayerView(life=opp_life, max_life=20, pp=10, max_pp=10, field=tuple(of),
                            ep=opp_ep, sep=opp_sep, crests=tuple(opp_crests)),
        cards_by_uid=cards, legal=LegalView(),
    )


class TestBasicRewards(unittest.TestCase):
    def test_board_gain(self):
        """打出3/3：手牌张数 -1（-w_hand）+ 新增随从 board 信号。"""
        prev = _snap(my_hand=[_card(1, atk=3, life=3, base=100)])
        nxt = _snap(my_field=[_card(1, atk=3, life=3, base=100)])
        r = step_reward(prev, _action("play", 1), nxt, "me", weights=W)
        board = W.w_board_count + 3 * W.w_board_atk + 3 * W.w_board_life
        self.assertAlmostEqual(r, -W.w_hand + board, places=5)

    def test_face_damage(self):
        """打脸：对方 life 20→15。"""
        prev = _snap()
        nxt = _snap(opp_life=15)
        r = step_reward(prev, _action("attack_leader"), nxt, "me", weights=W)
        self.assertAlmostEqual(r, 5 * W.w_face, places=5)

    def test_persistent_crest_acquisition_uses_discounted_immediate_value(self):
        """10264122获取即时记半张6费来源卡，长期部分只交由horizon处理。"""
        prev = _snap()
        nxt = _snap(my_crests=(CrestView(card_id=10264122),))
        r = step_reward(prev, _action("evolve", 1), nxt, "me", weights=W)
        self.assertAlmostEqual(r, 3 * W.w_crest, places=5)

    def test_clear(self):
        """解场：对方 4/4 随从消失 → board 读数之差为正（§11 过程即计）。"""
        prev = _snap(opp_field=[_card(9, atk=4, life=4, base=200)])
        nxt = _snap()
        r = step_reward(prev, _action("attack_follower"), nxt, "me", weights=W)
        board = W.w_board_count + 4 * W.w_board_atk + 4 * W.w_board_life
        self.assertAlmostEqual(r, board, places=5)

    def test_heal(self):
        prev = _snap(my_life=15)
        nxt = _snap(my_life=20)
        r = step_reward(prev, _action(), nxt, "me", weights=W)
        self.assertAlmostEqual(r, 5 * W.w_heal, places=5)

    def test_terminal_win(self):
        prev = _snap(opp_life=3)
        nxt = _snap(opp_life=0)
        r = step_reward(prev, _action("attack_leader"), nxt, "me", weights=W)
        self.assertAlmostEqual(r, W.v_win, places=5)

    def test_terminal_loss(self):
        prev = _snap(my_life=3)
        nxt = _snap(my_life=0)
        r = step_reward(prev, _action(), nxt, "me", weights=W)
        self.assertAlmostEqual(r, -W.v_win, places=5)


class TestCrossTurnSign(unittest.TestCase):
    def test_opponent_face_damage_is_negative_for_root(self):
        """跨回合符号：对手打我方脸，从 root(me) 视角是负收益。"""
        prev = _snap(my_life=20)
        nxt = _snap(my_life=14)  # 我方被打 6
        r = step_reward(prev, _action("attack_leader"), nxt, "me", weights=W)
        self.assertAlmostEqual(r, -6 * W.w_face, places=5)

    def test_opponent_clears_my_board_is_negative(self):
        """对手清我方随从：board 读数之差为负（§11 过程即计）。"""
        prev = _snap(my_field=[_card(1, atk=3, life=3, base=100)])
        nxt = _snap()
        r = step_reward(prev, _action("attack_follower"), nxt, "me", weights=W)
        board = W.w_board_count + 3 * W.w_board_atk + 3 * W.w_board_life
        self.assertAlmostEqual(r, -board, places=5)


class TestSpellboost(unittest.TestCase):
    def test_boost_increment_scored(self):
        """打法术使手牌其他牌增幅+1（留在手牌），有正收益。"""
        # 手牌：一张被打的法术(uid=1)离手，两张受增幅牌(uid=2,3)留手且 sb 0→1
        prev = _snap(my_hand=[
            _card(1, base=10133310, ctype=CardType.SPELL, has_sb=True, sb=0),
            _card(2, base=10131320, ctype=CardType.SPELL, has_sb=True, sb=0),
            _card(3, base=10833310, ctype=CardType.SPELL, has_sb=True, sb=0),
        ])
        nxt = _snap(my_hand=[
            _card(2, base=10131320, ctype=CardType.SPELL, has_sb=True, sb=1),
            _card(3, base=10833310, ctype=CardType.SPELL, has_sb=True, sb=1),
        ])
        # 无 threshold_fn → 线性：2张牌各 +1 × w_boost_linear
        r = step_reward(prev, _action("play", 1), nxt, "me", weights=W)
        # 打出1张手牌：手牌张数 3→2 = -w_hand；增幅收益 2 × w_boost_linear
        self.assertAlmostEqual(r, -W.w_hand + 2 * W.w_boost_linear, places=5)

    def test_boost_with_threshold_marginal(self):
        """有阈值增幅：朝阈值推进计分，饱和后归零。"""
        def thr(bid):
            return (5,) if bid == 10833310 else ()
        # 一张牌 sb 4→6（跨过阈值5），有效段 = min(6,5)-min(4,5) = 5-4 = 1
        prev = _snap(my_hand=[_card(2, base=10833310, ctype=CardType.SPELL, has_sb=True, sb=4)])
        nxt = _snap(my_hand=[_card(2, base=10833310, ctype=CardType.SPELL, has_sb=True, sb=6)])
        r = step_reward(prev, _action(), nxt, "me", weights=W, threshold_fn=thr)
        self.assertAlmostEqual(r, 1 * W.w_boost, places=5)  # 只有朝阈值的1段有效

    def test_boost_saturated_no_value(self):
        """已达阈值再增幅，边际归零（别囤饱和牌）。"""
        def thr(bid): return (5,)
        prev = _snap(my_hand=[_card(2, base=999, ctype=CardType.SPELL, has_sb=True, sb=5)])
        nxt = _snap(my_hand=[_card(2, base=999, ctype=CardType.SPELL, has_sb=True, sb=8)])
        r = step_reward(prev, _action(), nxt, "me", weights=W, threshold_fn=thr)
        self.assertAlmostEqual(r, 0.0, places=5)

    def test_boost_not_lost_on_transform(self):
        """变身丢失场景：逐动作调用时，增幅在变身前的当步已计入。

        模拟：某步 sb 4→5（uid 不变，base_card_id 尚未变），该步计入朝阈值推进。
        （变身发生在此步之后的独立步，不影响本步已计的增幅。）
        """
        def thr(bid): return (5,)
        prev = _snap(my_hand=[_card(7, base=86233310, ctype=CardType.SPELL, has_sb=True, sb=4)])
        nxt = _snap(my_hand=[_card(7, base=86233310, ctype=CardType.SPELL, has_sb=True, sb=5)])
        r = step_reward(prev, _action(), nxt, "me", weights=W, threshold_fn=thr)
        self.assertAlmostEqual(r, 1 * W.w_boost, places=5)

    def test_early_cash_in_loses_higher_tier_progress(self):
        """核心机制：多阈值卡在低档兑现（卡离手）会丢失为高档积攒的进度势 → 净亏。

        坦忒拉(10,20)，sb=10 持有态势 = 10档满(10*w_boost) + 20档进度10/20(5*w_boost)
        = 15*w_boost。若在 sb=10 打出（卡离手，手牌不再含它），其持有态势全部蒸发；
        若生成的衍生物（苍奏之四，10档解锁）只值 10档的 10*w_boost，则净丢失 20档的
        5*w_boost —— 这正是"忍到20"的真实机会成本（telescoping，无偏）。
        """
        def thr(bid): return (10, 20)
        from dataclasses import replace

        from decision_engine.planning.incremental_value import _boost_potential
        # 这里验证的是"进度势"本身，关闭预支项（预支是 §12.20 的独立机制，另有专门测试）
        w_progress = replace(W, w_boost_anticipation=0.0)
        pot10 = _boost_potential(10, 10834110, thr, w_progress)
        pot20 = _boost_potential(20, 10834110, thr, w_progress)
        # sb=10 时为20档已攒进度 = pot10 中属于20档的部分 = 20档reward*10/20 = 10*w_boost*0.5
        self.assertAlmostEqual(pot10, 15 * W.w_boost, places=5)   # 10(满) + 5(20档半程)
        self.assertAlmostEqual(pot20, 20 * W.w_boost, places=5)   # 10 + 10(20档满)
        # 在10兑现：丢失持有态15*w_boost；忍到20兑现：丢失20*w_boost但拿满额解锁。
        # 关键：sb=10 时的持有态(15)已高于"仅10档满"(10)，多出的5就是20档进度，
        # 提前兑现即蒸发 → 搜索看到"继续攒不亏、提前兑现亏5*w_boost"。
        self.assertGreater(pot10, 10 * W.w_boost, "sb=10 持有态须含为20档攒的进度势")


class TestBoostMarginal(unittest.TestCase):
    def test_marginal_below_threshold(self):
        self.assertAlmostEqual(_boost_marginal_value(0, 3, 1, lambda b: (5,), W), 3 * W.w_boost)

    def test_marginal_crossing_threshold(self):
        # 3→7 跨阈值5：有效段 = min(7,5)-min(3,5)=5-3=2
        self.assertAlmostEqual(_boost_marginal_value(3, 7, 1, lambda b: (5,), W), 2 * W.w_boost)

    def test_marginal_above_threshold(self):
        self.assertAlmostEqual(_boost_marginal_value(5, 9, 1, lambda b: (5,), W), 0.0)

    def test_marginal_no_threshold_linear(self):
        self.assertAlmostEqual(_boost_marginal_value(0, 3, 1, None, W), 3 * W.w_boost_linear)

    def test_marginal_multi_threshold_uses_max(self):
        # 坦忒拉阈值(10,20)，多阈值进度势叠加（非旧的朝max线性）：
        # sb=12: 10档 reward=10*w_boost, 进度满 → 10*w_boost
        #        20档 reward=(20-10)*w_boost=10*w_boost, 进度12/20 → 6*w_boost
        #        total = 16*w_boost；sb=0 → 0；边际 = 16*w_boost
        self.assertAlmostEqual(_boost_marginal_value(0, 12, 1, lambda b: (10, 20), W),
                               16 * W.w_boost)


class TestThresholdTable(unittest.TestCase):
    """S1：从真实卡牌数据库解析增幅阈值。"""

    def test_known_card_thresholds(self):
        table = build_spellboost_threshold_table()
        self.assertEqual(table.get(10133310), (5,))
        self.assertEqual(table.get(10833310), (5,))
        self.assertEqual(table.get(10834110), (10, 20))

    def test_linear_card_not_in_table(self):
        table = build_spellboost_threshold_table()
        self.assertNotIn(10131320, table)  # 暴风破线性无阈值

    def test_make_threshold_fn(self):
        fn = make_threshold_fn({10834110: (10, 20)})
        self.assertEqual(fn(10834110), (10, 20))
        self.assertEqual(fn(99999999), ())

    def test_default_threshold_fn_integrates_with_step_reward(self):
        fn = default_threshold_fn()
        prev = _snap(my_hand=[_card(1, base=10834110, ctype=CardType.FOLLOWER, has_sb=True, sb=0)])
        nxt = _snap(my_hand=[_card(1, base=10834110, ctype=CardType.FOLLOWER, has_sb=True, sb=3)])
        r = step_reward(prev, _action(), nxt, "me", weights=W, threshold_fn=fn)
        # 坦忒拉(10,20)多阈值进度势: sb=3 → 10档 w_boost*10*3/10 + 20档 w_boost*10*3/20
        #   = 3*w_boost + 1.5*w_boost = 4.5*w_boost
        self.assertAlmostEqual(r, 4.5 * W.w_boost, places=5)


class TestNoRegression(unittest.TestCase):
    """S4 无回退：确认新价值保留旧 diff 修好的关键行为（空融合判负）。"""

    def test_empty_fusion_is_negative(self):
        """空融合：消耗手牌但无场面/打脸/增幅收益 → 净负（不回潮）。"""
        prev = _snap(my_hand=[
            _card(1, base=1, ctype=CardType.FOLLOWER),
            _card(2, base=2, ctype=CardType.FOLLOWER),
            _card(3, base=3, ctype=CardType.FOLLOWER),
        ])
        nxt = _snap()  # 手牌空、场面空、无收益
        r = step_reward(prev, _action("fusion"), nxt, "me", weights=W)
        # 3 张手牌张数读数 × -w_hand，无任何正收益
        self.assertAlmostEqual(r, -3 * W.w_hand, places=5)
        self.assertLess(r, 0, "空融合必须判负（无回退）")

    def test_good_play_beats_empty_fusion(self):
        """打随从：手牌 -1 但 board 过程项 +随从信号；空融合只有手牌成本 → 打随从更优。"""
        from decision_engine.planning.incremental_value import horizon_leaf_residual
        prev = _snap(my_hand=[_card(1, atk=4, life=4, base=100)])
        good = _snap(my_field=[_card(1, atk=4, life=4, base=100)])  # 打出4/4
        empty = _snap()  # 手牌消耗无收益
        r_good = step_reward(prev, _action("play", 1), good, "me", weights=W)
        r_empty = step_reward(prev, _action("fusion", 1), empty, "me", weights=W)
        board = W.w_board_count + 4 * W.w_board_atk + 4 * W.w_board_life
        self.assertAlmostEqual(r_good, -W.w_hand + board, places=5)
        self.assertAlmostEqual(r_empty, -W.w_hand, places=5)
        # board 过程即计后，打随从的即时收益已显著优于空融合（叶残值不再含身材）。
        self.assertGreater(r_good, r_empty, "打随从应优于空融合")
        res_good = r_good + horizon_leaf_residual(good, "me")
        res_empty = r_empty + horizon_leaf_residual(empty, "me")
        self.assertGreater(res_good, res_empty, "打随从总价值应优于空融合")


class TestEvolveCost(unittest.TestCase):
    """方案A：进化消耗EP记负成本，避免浪费进化点。"""

    def test_evolve_empty_follower_is_net_negative(self):
        """进化一个无进化效果的随从：+2/+2身材(board 增量) - 1EP成本 → 净负。"""
        prev = _snap(my_ep=1, my_field=[_card(1, atk=3, life=3, base=100)])
        nxt = _snap(my_ep=0, my_field=[_card(1, atk=5, life=5, base=100)])
        r = step_reward(prev, _action("evolve", 1), nxt, "me", weights=W)
        # +2 atk + +2 life 的 board 增量；EP消耗 -w_ep。
        board_gain = 2 * W.w_board_atk + 2 * W.w_board_life
        self.assertAlmostEqual(r, -W.w_ep + board_gain, places=5)
        self.assertLess(r, 0, "进化空白随从应净负(避免浪费进化点)")

    def test_evolve_with_clear_effect_offsets_cost(self):
        """进化带解场/打脸效果：打脸收益抵消EP成本。进化打3脸: +3*w_face - w_ep + board。"""
        prev = _snap(my_ep=1, opp_life=20, my_field=[_card(1, atk=3, life=3, base=100)])
        nxt = _snap(my_ep=0, opp_life=17, my_field=[_card(1, atk=5, life=5, base=100)])
        r = step_reward(prev, _action("evolve", 1), nxt, "me", weights=W)
        board_gain = 2 * W.w_board_atk + 2 * W.w_board_life
        self.assertAlmostEqual(r, 3 * W.w_face - W.w_ep + board_gain, places=5)
        self.assertGreater(r, 0, "进化带打脸效果应净正")

    def test_ep_recover_is_positive(self):
        """回EP记正收益(对称)。"""
        prev = _snap(my_ep=0)
        nxt = _snap(my_ep=1)
        r = step_reward(prev, _action(), nxt, "me", weights=W)
        self.assertAlmostEqual(r, W.w_ep, places=5)

    def test_opponent_ep_consumed_is_positive_for_me(self):
        """对方消耗EP(被逼进化解场) → 我方赚(记正 w_ep)。"""
        prev = _snap(opp_ep=1)
        nxt = _snap(opp_ep=0)
        r = step_reward(prev, _action(), nxt, "me", weights=W)
        # 对方ep 1→0 消耗1点 → 我方 +w_ep
        self.assertAlmostEqual(r, W.w_ep, places=5)

    def test_opponent_sep_consumed_is_positive_for_me(self):
        """对方消耗SEP → 我方赚(记正 w_sep)。"""
        prev = _snap(opp_sep=1)
        nxt = _snap(opp_sep=0)
        r = step_reward(prev, _action(), nxt, "me", weights=W)
        self.assertAlmostEqual(r, W.w_sep, places=5)


class TestHorizonResidual(unittest.TestCase):
    """§11：地平线叶残值不再含场面身材（已由 step_reward 的 board 维度过程即计）。"""

    def test_pp_of_stat_calibration(self):
        """pp_of_stat 纯函数保留（9身材≈3费）；horizon 已不再调用它，仅作工具函数。"""
        from decision_engine.planning.incremental_value import pp_of_stat
        self.assertAlmostEqual(pp_of_stat(9) * 10, 30.0, delta=1.0)

    def test_pp_of_stat_superlinear(self):
        """超线性工具函数自身仍成立（不参与 horizon）。"""
        from decision_engine.planning.incremental_value import pp_of_stat
        self.assertGreater(pp_of_stat(18), pp_of_stat(9))
        self.assertGreater(pp_of_stat(9), pp_of_stat(2))

    def test_leaf_residual_board_not_counted(self):
        """§11：我方/对方场面身材不再进叶残值（防与 board 过程项双重计价）→ 叶残值=0。"""
        from decision_engine.planning.incremental_value import horizon_leaf_residual
        mine = _snap(my_field=[_card(1, atk=5, life=4, base=100)])
        theirs = _snap(opp_field=[_card(9, atk=5, life=4, base=200)])
        self.assertAlmostEqual(horizon_leaf_residual(mine, "me"), 0.0, places=5)
        self.assertAlmostEqual(horizon_leaf_residual(theirs, "me"), 0.0, places=5)

    def test_leaf_residual_persistent_crest_is_symmetric(self):
        """存活10264122在horizon提供一次性长期能力残值，双方对称（纹章残值保留）。"""
        from decision_engine.planning.incremental_value import horizon_leaf_residual
        mine = _snap(my_crests=(CrestView(card_id=10264122),))
        theirs = _snap(opp_crests=(CrestView(card_id=10264122),))
        self.assertAlmostEqual(horizon_leaf_residual(mine, "me"), 20.0, places=5)
        self.assertAlmostEqual(horizon_leaf_residual(theirs, "me"), -20.0, places=5)


class TestExtraPpOptionResidual(unittest.TestCase):
    """E3：Extra PP 保守 horizon option residual。

    只在 horizon_leaf_residual 传入 next_player 时加入；只对 next_player 可立即
    行动的一方评估，对 root_player 对称正/负；仅当该方后手、有未用 held token、
    且已知手牌存在恰被 pp+1 解锁的 play mode 时给保守固定 option 值。
    """

    def _extra_pp(self, held=1, used=False):
        return ExtraPpView(extra_pp=held, is_used_extra_pp_this_turn=used)

    def _e3_snap(
        self,
        *,
        side="me",
        is_first_side=False,
        pp=3,
        turn=4,
        held=1,
        used=False,
        unlock_cost=4,
        hand_cost=None,
        no_token=False,
    ):
        """构造一方持有 extra_pp、手牌含 pp+1 解锁 play mode 的局面。

        unlock_cost=pp+1 时手牌牌费恰好被 +1 PP 解锁；hand_cost 可覆盖为不可解锁。
        no_token=True 时该方无 extra_pp 对象。
        """
        cost = hand_cost if hand_cost is not None else unlock_cost
        extra = None if no_token else self._extra_pp(held=held, used=used)
        hand_card = _card(1, base=100, ctype=CardType.SPELL, cost=cost)
        actor = PlayerView(
            life=20, max_life=20, pp=pp, max_pp=10, turn=turn,
            is_first_side=is_first_side, hand=(1,), extra_pp=extra,
        )
        other = PlayerView(life=20, max_life=20, pp=pp, max_pp=10, turn=turn)
        cards = {1: hand_card}
        if side == "me":
            return BattleSnapshot(me=actor, opponent=other, cards_by_uid=cards, legal=LegalView())
        return BattleSnapshot(me=other, opponent=actor, cards_by_uid=cards, legal=LegalView())

    def test_default_call_unchanged(self):
        """默认调用（不传 next_player）：既有数值不变，option residual 不加入。"""
        from decision_engine.planning.incremental_value import horizon_leaf_residual
        snap = self._e3_snap(side="me", pp=3, unlock_cost=4)
        # 场面空、无纹章 → 既有残值为 0；不传 next_player 保持 0。
        self.assertAlmostEqual(horizon_leaf_residual(snap, "me"), 0.0, places=5)

    def test_option_positive_when_next_player_is_root_and_can_act(self):
        """next_player 可行动且为 root_player：正的保守 option 值。"""
        from decision_engine.planning.incremental_value import (
            horizon_leaf_residual, _EXTRA_PP_OPTION_RESIDUAL,
        )
        snap = self._e3_snap(side="me", pp=3, unlock_cost=4)
        r = horizon_leaf_residual(snap, "me", next_player="me")
        self.assertAlmostEqual(r, _EXTRA_PP_OPTION_RESIDUAL, places=5)

    def test_option_symmetric_negative_for_opponent(self):
        """正负对称：token 在对手方、next_player 为对手 → root 视角负。"""
        from decision_engine.planning.incremental_value import (
            horizon_leaf_residual, _EXTRA_PP_OPTION_RESIDUAL,
        )
        snap = self._e3_snap(side="opponent", pp=3, unlock_cost=4)
        r = horizon_leaf_residual(snap, "me", next_player="opponent")
        self.assertAlmostEqual(r, -_EXTRA_PP_OPTION_RESIDUAL, places=5)

    def test_option_value_at_most_half_of_one_pp(self):
        """保守约束：option 值 <= 一费价值的一半（一费=w_pp_temp=10）。"""
        from decision_engine.planning.incremental_value import _EXTRA_PP_OPTION_RESIDUAL
        self.assertLessEqual(_EXTRA_PP_OPTION_RESIDUAL, W.w_pp_temp / 2.0 + 1e-9)

    def test_no_unlock_hand_not_counted(self):
        """手牌无恰被 pp+1 解锁的 play mode（费用不匹配）→ 不计。"""
        from decision_engine.planning.incremental_value import horizon_leaf_residual
        # pp=3，手牌费=2（现在就买得起，不是 pp+1 解锁）；费=6（+1 也买不起）
        cheap = self._e3_snap(side="me", pp=3, hand_cost=2)
        far = self._e3_snap(side="me", pp=3, hand_cost=6)
        self.assertAlmostEqual(horizon_leaf_residual(cheap, "me", next_player="me"), 0.0, places=5)
        self.assertAlmostEqual(horizon_leaf_residual(far, "me", next_player="me"), 0.0, places=5)

    def test_unlock_via_enhance_play_mode(self):
        """enhance play mode 费用恰为 pp+1 也算解锁。"""
        from decision_engine.planning.incremental_value import (
            horizon_leaf_residual, _EXTRA_PP_OPTION_RESIDUAL,
        )
        # 普通费=2（现在买得起，不解锁），enhance 费=4=pp+1（被解锁）
        card = CardView(uid=1, base_card_id=100, card_type=CardType.SPELL, cost=2,
                        can_enhance=True, enhance_costs=(4,))
        actor = PlayerView(life=20, max_life=20, pp=3, max_pp=10, turn=4,
                           is_first_side=False, hand=(1,), extra_pp=self._extra_pp())
        other = PlayerView(life=20, max_life=20, pp=3, max_pp=10, turn=4)
        snap = BattleSnapshot(me=actor, opponent=other, cards_by_uid={1: card}, legal=LegalView())
        self.assertAlmostEqual(
            horizon_leaf_residual(snap, "me", next_player="me"), _EXTRA_PP_OPTION_RESIDUAL, places=5,
        )

    def test_full_field_follower_normal_mode_not_counted(self):
        """满场时，费用恰为 pp+1 的普通随从仍不可打出，不能形成 option。"""
        from decision_engine.planning.incremental_value import horizon_leaf_residual
        card = _card(1, base=100, ctype=CardType.FOLLOWER, cost=4)
        actor = PlayerView(
            life=20, max_life=20, pp=3, max_pp=10, turn=4,
            is_first_side=False, hand=(1,), field=(10, 11, 12, 13, 14),
            extra_pp=self._extra_pp(),
        )
        other = PlayerView(life=20, max_life=20, pp=3, max_pp=10, turn=4)
        snap = BattleSnapshot(me=actor, opponent=other, cards_by_uid={1: card}, legal=LegalView())
        self.assertAlmostEqual(horizon_leaf_residual(snap, "me", next_player="me"), 0.0, places=5)

    def test_spent_token_not_counted(self):
        """本回合已使用 extra_pp（is_used_extra_pp_this_turn）→ 不再是未来 option。"""
        from decision_engine.planning.incremental_value import horizon_leaf_residual
        snap = self._e3_snap(side="me", pp=3, unlock_cost=4, used=True)
        self.assertAlmostEqual(horizon_leaf_residual(snap, "me", next_player="me"), 0.0, places=5)

    def test_no_token_not_counted(self):
        """无 held token（extra_pp 对象缺失或 0）→ 不计。"""
        from decision_engine.planning.incremental_value import horizon_leaf_residual
        no_obj = self._e3_snap(side="me", pp=3, unlock_cost=4, no_token=True)
        zero = self._e3_snap(side="me", pp=3, unlock_cost=4, held=0)
        self.assertAlmostEqual(horizon_leaf_residual(no_obj, "me", next_player="me"), 0.0, places=5)
        self.assertAlmostEqual(horizon_leaf_residual(zero, "me", next_player="me"), 0.0, places=5)

    def test_first_side_not_counted(self):
        """先手方不获得 extra_pp → 不计。"""
        from decision_engine.planning.incremental_value import horizon_leaf_residual
        snap = self._e3_snap(side="me", pp=3, unlock_cost=4, is_first_side=True)
        self.assertAlmostEqual(horizon_leaf_residual(snap, "me", next_player="me"), 0.0, places=5)

    def test_t5_counted_when_currently_acting(self):
        """T5 当前该方可行动（next_player 为 token 持有方）→ 计入 option。"""
        from decision_engine.planning.incremental_value import (
            horizon_leaf_residual, _EXTRA_PP_OPTION_RESIDUAL,
        )
        snap = self._e3_snap(side="me", pp=3, turn=5, unlock_cost=4)
        r = horizon_leaf_residual(snap, "me", next_player="me")
        self.assertAlmostEqual(r, _EXTRA_PP_OPTION_RESIDUAL, places=5)

    def test_t5_token_not_counted_when_next_player_is_other_side(self):
        """T5 结束后 next_player 为另一方 → 该方持有的即将被覆盖 token 不计。

        token 在 me（T5 后手），但 next_player=opponent（me 已结束回合），
        只评估 opponent 一方；opponent 无 token → 0，不误奖即将被覆盖的 me token。
        """
        from decision_engine.planning.incremental_value import horizon_leaf_residual
        snap = self._e3_snap(side="me", pp=3, turn=5, unlock_cost=4)
        r = horizon_leaf_residual(snap, "me", next_player="opponent")
        self.assertAlmostEqual(r, 0.0, places=5)

    def test_non_participant_next_player_zero(self):
        """未知 / 非 me/opponent 的 next_player → 0。"""
        from decision_engine.planning.incremental_value import horizon_leaf_residual
        snap = self._e3_snap(side="me", pp=3, unlock_cost=4)
        self.assertAlmostEqual(horizon_leaf_residual(snap, "me", next_player="nobody"), 0.0, places=5)


class TestIncrementalSearchValue(unittest.TestCase):
    """IncrementalSearchValue：逐动作 PUCT 的 PP 本位势 leaf（§11，取代 MacroDrivenSearch）。"""

    def test_leaf_value_in_range_and_positive_for_good_board(self):
        from decision_engine.planning.incremental_value import IncrementalSearchValue
        ts = _snap()
        v = IncrementalSearchValue(ts, player="me")  # 无 predictor → virtual turn_end 退化
        leaf = _snap(my_field=[_card(1, atk=5, life=4, base=100)], opp_life=17)  # 铺场+打脸
        r = v.evaluate(leaf, player="me")
        self.assertFalse(r.terminal)
        self.assertGreater(r.value, 0.0)
        self.assertLessEqual(r.value, 1.0)
        self.assertGreaterEqual(r.value, -1.0)

    def test_terminal_win_loss(self):
        from decision_engine.planning.incremental_value import IncrementalSearchValue
        v = IncrementalSearchValue(_snap(), player="me")
        self.assertEqual(v.evaluate(_snap(opp_life=0), player="me").value, 1.0)
        self.assertEqual(v.evaluate(_snap(my_life=0), player="me").value, -1.0)

    def test_path_independence_equiv_endpoints(self):
        """势路径无关：leaf value 只依赖 turn_start 与 leaf 的状态差，与中间路径无关。

        同一 turn_start、同一 leaf 状态，无论构造顺序，evaluate 结果一致。
        """
        from decision_engine.planning.incremental_value import IncrementalSearchValue
        ts = _snap(my_hand=[_card(1, base=10834110, ctype=CardType.SPELL, has_sb=True, sb=0)])
        v = IncrementalSearchValue(ts, player="me")
        leaf = _snap(my_hand=[_card(1, base=10834110, ctype=CardType.SPELL, has_sb=True, sb=12)],
                     opp_life=17)
        r1 = v.evaluate(leaf, player="me").value
        r2 = v.evaluate(leaf, player="me").value
        self.assertAlmostEqual(r1, r2, places=9)

    def test_boost_accumulation_raises_value(self):
        """攒增幅（朝阈值推进）使 leaf 势上升 → value 更高。"""
        from decision_engine.planning.incremental_value import IncrementalSearchValue
        ts = _snap(my_hand=[_card(1, base=10834110, ctype=CardType.SPELL, has_sb=True, sb=0)])
        v = IncrementalSearchValue(ts, player="me")
        low = _snap(my_hand=[_card(1, base=10834110, ctype=CardType.SPELL, has_sb=True, sb=2)])
        high = _snap(my_hand=[_card(1, base=10834110, ctype=CardType.SPELL, has_sb=True, sb=9)])
        self.assertGreater(v.evaluate(high, player="me").value, v.evaluate(low, player="me").value)


if __name__ == "__main__":
    unittest.main()
