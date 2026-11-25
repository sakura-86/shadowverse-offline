# -*- coding: utf-8 -*-
"""Tests for macro_static_value（P4 静态资源 Macro evaluator）。

覆盖：手牌边际曲线、资源对称、终局压倒、extra pp 生命周期分级、
纹章通用转换估值（6PP+EP=50 / 7PP+SEP=65）、board residue 默认 0/可注入。
"""

import unittest

from decision_engine.planning.macro_static_value import (
    DEFAULT_HAND_MARGINAL,
    DEFAULT_WEIGHTS,
    MacroStaticWeights,
    ExtraPpStatus,
    classify_extra_pp,
    default_extra_pp_redeemable,
    zero_crest_valuation,
    CrestSourceCost,
    GenericCrestConversionValuation,
    static_side_score,
    macro_static_value,
    _hand_score,
)
from decision_engine.state.snapshot import (
    BattleSnapshot, CardView, CrestView, ExtraPpView, LegalView, PlayerView,
)

W = DEFAULT_WEIGHTS


def _player(
    *, life=20, max_pp=10, ep=0, sep=0, hand=(), field=(),
    is_first_side=False, turn=1, extra_pp=None, crests=(), result=0,
):
    return PlayerView(
        life=life, max_life=20, pp=max_pp, max_pp=max_pp, ep=ep, sep=sep,
        hand=tuple(hand), field=tuple(field), is_first_side=is_first_side,
        turn=turn, extra_pp=extra_pp, crests=tuple(crests), result=result,
    )


def _snap(me=None, opponent=None, cards=None):
    me = me or _player()
    opponent = opponent or _player()
    return BattleSnapshot(
        me=me, opponent=opponent,
        cards_by_uid=cards or {}, legal=LegalView(),
    )


class TestHandMarginalCurve(unittest.TestCase):
    def test_default_curve_values(self):
        self.assertEqual(DEFAULT_HAND_MARGINAL, (15, 15, 12, 9, 7, 5, 3, 1, 0))

    def test_marginal_accumulation(self):
        # 前 3 张：15+15+12 = 42
        self.assertEqual(_hand_score(3, DEFAULT_HAND_MARGINAL), 42.0)
        # 全 8 张（第 9 张边际 0）：15+15+12+9+7+5+3+1 = 67
        self.assertEqual(_hand_score(8, DEFAULT_HAND_MARGINAL), 67.0)

    def test_ninth_card_zero_marginal(self):
        eight = _hand_score(8, DEFAULT_HAND_MARGINAL)
        nine = _hand_score(9, DEFAULT_HAND_MARGINAL)
        ten = _hand_score(10, DEFAULT_HAND_MARGINAL)
        self.assertEqual(nine, eight)   # 第 9 张边际 0
        self.assertEqual(ten, eight)    # 第 10 张也 0
        self.assertEqual(nine - eight, 0.0)

    def test_empty_hand(self):
        self.assertEqual(_hand_score(0, DEFAULT_HAND_MARGINAL), 0.0)

    def test_side_score_uses_hand_curve(self):
        hand = [CardView(uid=i) for i in range(1, 4)]  # 3 张
        cards = {c.uid: c for c in hand}
        snap = _snap(me=_player(hand=[c.uid for c in hand]), cards=cards)
        # HP 20*10=200, max_pp 10*10=100, 手牌 42
        score = static_side_score(snap, snap.me, weights=W)
        self.assertAlmostEqual(score, 200 + 100 + 42, places=5)


class TestResourceWeightsAndSymmetry(unittest.TestCase):
    def test_default_resource_weights(self):
        self.assertEqual(W.w_hp, 10.0)
        self.assertEqual(W.w_max_pp, 10.0)
        self.assertEqual(W.w_ep, 20.0)
        self.assertEqual(W.w_sep, 30.0)
        self.assertEqual(W.v_win, 1000.0)
        self.assertEqual(W.w_board_residue, 0.0)

    def test_side_score_resource_terms(self):
        snap = _snap(me=_player(life=15, max_pp=8, ep=2, sep=1))
        # 15*10 + 8*10 + 2*20 + 1*30 = 150+80+40+30 = 300
        self.assertAlmostEqual(static_side_score(snap, snap.me, weights=W), 300.0, places=5)

    def test_symmetric_balanced_state(self):
        snap = _snap(me=_player(), opponent=_player())
        self.assertAlmostEqual(macro_static_value(snap, "me"), 0.0, places=5)
        self.assertAlmostEqual(macro_static_value(snap, "opponent"), 0.0, places=5)

    def test_me_opponent_antisymmetry(self):
        me = _player(life=18, ep=1)
        opp = _player(life=12, sep=1)
        snap = _snap(me=me, opponent=opp)
        v_me = macro_static_value(snap, "me")
        v_opp = macro_static_value(snap, "opponent")
        self.assertAlmostEqual(v_me, -v_opp, places=5)
        self.assertNotAlmostEqual(v_me, 0.0, places=5)

    def test_advantage_direction(self):
        me = _player(life=20)
        opp = _player(life=10)
        snap = _snap(me=me, opponent=opp)
        # 我方 HP 高 10 → +100
        self.assertAlmostEqual(macro_static_value(snap, "me"), 100.0, places=5)

    def test_invalid_player(self):
        snap = _snap()
        with self.assertRaises(ValueError):
            macro_static_value(snap, "both")


class TestTerminalDominates(unittest.TestCase):
    def test_win_returns_v_win(self):
        snap = _snap(me=_player(life=20), opponent=_player(life=0))
        self.assertEqual(macro_static_value(snap, "me"), W.v_win)
        self.assertEqual(macro_static_value(snap, "opponent"), -W.v_win)

    def test_loss_returns_neg_v_win(self):
        snap = _snap(me=_player(life=0), opponent=_player(life=20))
        self.assertEqual(macro_static_value(snap, "me"), -W.v_win)

    def test_simultaneous_zero(self):
        snap = _snap(me=_player(life=0), opponent=_player(life=0))
        self.assertEqual(macro_static_value(snap, "me"), 0.0)

    def test_terminal_dominates_intermediate(self):
        """终局值必须压倒任何可达的中间分累加。"""
        # 构造中间分极大的一侧：满血 + 满手牌 + 高资源
        hand = list(range(1, 9))
        cards = {i: CardView(uid=i) for i in hand}
        rich = _player(life=20, max_pp=10, ep=3, sep=3, hand=hand)
        snap_mid = _snap(me=rich, opponent=_player(life=1), cards=cards)
        mid = macro_static_value(snap_mid, "me")
        # 中间分应远小于 v_win（终局压倒）
        self.assertLess(mid, W.v_win)
        # 而实际终局（对方 0 血）直接给 v_win
        snap_win = _snap(me=rich, opponent=_player(life=0), cards=cards)
        self.assertEqual(macro_static_value(snap_win, "me"), W.v_win)
        self.assertGreater(macro_static_value(snap_win, "me"), mid)

    def test_deckout_result_terminal(self):
        # result 非 0 且对方 result=0 → me 胜
        snap = _snap(me=_player(result=1), opponent=_player(result=0))
        self.assertEqual(macro_static_value(snap, "me"), W.v_win)


class TestExtraPpLifecycle(unittest.TestCase):
    def _extra(self, *, held=1, used=False):
        return ExtraPpView(extra_pp=held, is_used_extra_pp_this_turn=used)

    def test_first_side_no_option(self):
        v = _player(is_first_side=True, turn=3, extra_pp=self._extra())
        snap = _snap(me=v)
        self.assertEqual(classify_extra_pp(snap, snap.me), ExtraPpStatus.NONE)

    def test_no_token_none(self):
        v = _player(is_first_side=False, turn=3, extra_pp=ExtraPpView(extra_pp=0))
        snap = _snap(me=v)
        self.assertEqual(classify_extra_pp(snap, snap.me), ExtraPpStatus.NONE)

    def test_used_this_turn_none(self):
        """本回合已用：不再是未来 option（且不从 is_used 反推已消费价值）。"""
        v = _player(is_first_side=False, turn=3, extra_pp=self._extra(used=True))
        snap = _snap(me=v)
        self.assertEqual(classify_extra_pp(snap, snap.me), ExtraPpStatus.NONE)

    def test_hold_early_turn(self):
        v = _player(is_first_side=False, turn=3, extra_pp=self._extra())
        snap = _snap(me=v)
        self.assertEqual(classify_extra_pp(snap, snap.me), ExtraPpStatus.HOLD)

    def test_expiring_t5_second_side(self):
        """T5 后手持有、即将被覆盖 → EXPIRING。"""
        v = _player(is_first_side=False, turn=5, extra_pp=self._extra())
        snap = _snap(me=v)
        self.assertEqual(classify_extra_pp(snap, snap.me), ExtraPpStatus.EXPIRING)

    def test_expiring_option_value_below_hold(self):
        """合同要点 4：T5 后手持有 EXPIRING 的分值应低于普通 HOLD。"""
        hold = _player(is_first_side=False, turn=3, extra_pp=self._extra())
        expiring = _player(is_first_side=False, turn=5, extra_pp=self._extra())
        snap_hold = _snap(me=hold)
        snap_exp = _snap(me=expiring)
        s_hold = static_side_score(snap_hold, snap_hold.me, weights=W)
        s_exp = static_side_score(snap_exp, snap_exp.me, weights=W)
        # 两者 turn 不同不影响静态分中的其它项（本模块 static score 不含 turn 项），
        # 差异只来自 extra_pp option 分级。
        self.assertLess(s_exp, s_hold)
        self.assertAlmostEqual(s_hold - s_exp, W.w_extra_pp_hold - W.w_extra_pp_expiring, places=5)

    def test_default_predicate_no_guess(self):
        """默认可兑现 predicate 不猜：即便持有也不会升级为 REDEEMABLE。"""
        v = _player(is_first_side=False, turn=3, extra_pp=self._extra())
        snap = _snap(me=v)
        self.assertFalse(default_extra_pp_redeemable(snap, snap.me))
        self.assertEqual(classify_extra_pp(snap, snap.me), ExtraPpStatus.HOLD)

    def test_injected_redeemable_predicate(self):
        """可注入 predicate 判定可兑现 → REDEEMABLE（值高于 HOLD）。"""
        v = _player(is_first_side=False, turn=3, extra_pp=self._extra())
        snap = _snap(me=v)
        status = classify_extra_pp(snap, snap.me, redeemable=lambda s, view: True)
        self.assertEqual(status, ExtraPpStatus.REDEEMABLE)
        self.assertGreater(W.w_extra_pp_redeemable, W.w_extra_pp_hold)


class TestCrestValuation(unittest.TestCase):
    def test_default_no_guess(self):
        snap = _snap(me=_player(crests=(CrestView(card_id=12345),)))
        # 默认 provider 返回 0：纹章不加分
        base = static_side_score(_snap(me=_player()), _snap(me=_player()).me, weights=W)
        with_crest = static_side_score(snap, snap.me, weights=W, crest_valuation=zero_crest_valuation)
        self.assertAlmostEqual(with_crest, base, places=5)

    def test_generic_conversion_6pp_ep(self):
        """通用转换估值：6PP + EP = 50（pp_unit=5, w_ep=20 → 6*5+20=50）。"""
        provider = GenericCrestConversionValuation(
            {777: CrestSourceCost(source_pp=6.0, source_ep=1.0)},
            pp_unit_value=5.0,
        )
        snap = _snap(me=_player(crests=(CrestView(card_id=777),)))
        self.assertAlmostEqual(provider(snap, snap.me), 50.0, places=5)

    def test_generic_conversion_7pp_sep(self):
        """通用转换估值：7PP + SEP = 65（pp_unit=5, w_sep=30 → 7*5+30=65）。"""
        provider = GenericCrestConversionValuation(
            {888: CrestSourceCost(source_pp=7.0, source_sep=1.0)},
            pp_unit_value=5.0,
        )
        snap = _snap(me=_player(crests=(CrestView(card_id=888),)))
        self.assertAlmostEqual(provider(snap, snap.me), 65.0, places=5)

    def test_generic_conversion_unknown_crest_zero(self):
        """未在成本表中的纹章不猜，记 0。"""
        provider = GenericCrestConversionValuation(
            {777: CrestSourceCost(source_pp=6.0)}, pp_unit_value=5.0,
        )
        snap = _snap(me=_player(crests=(CrestView(card_id=999),)))
        self.assertAlmostEqual(provider(snap, snap.me), 0.0, places=5)

    def test_generic_conversion_no_hardcoded_card_id(self):
        """provider 不硬编码任何卡：同一成本换不同 card_id 结果一致。"""
        cost = CrestSourceCost(source_pp=6.0, source_ep=1.0)
        p1 = GenericCrestConversionValuation({111: cost}, pp_unit_value=5.0)
        p2 = GenericCrestConversionValuation({222: cost}, pp_unit_value=5.0)
        s1 = _snap(me=_player(crests=(CrestView(card_id=111),)))
        s2 = _snap(me=_player(crests=(CrestView(card_id=222),)))
        self.assertAlmostEqual(p1(s1, s1.me), p2(s2, s2.me), places=5)

    def test_crest_valuation_added_to_side_score(self):
        provider = GenericCrestConversionValuation(
            {777: CrestSourceCost(source_pp=6.0, source_ep=1.0)}, pp_unit_value=5.0,
        )
        snap = _snap(me=_player(crests=(CrestView(card_id=777),)))
        base = static_side_score(snap, snap.me, weights=W, crest_valuation=zero_crest_valuation)
        with_crest = static_side_score(snap, snap.me, weights=W, crest_valuation=provider)
        self.assertAlmostEqual(with_crest - base, 50.0, places=5)


class TestBoardResidue(unittest.TestCase):
    def test_default_board_residue_zero(self):
        """board residue 参数默认 0：场面不加分。"""
        field = [CardView(uid=1, atk=5, life=5)]
        cards = {c.uid: c for c in field}
        snap = _snap(me=_player(field=[c.uid for c in field]), cards=cards)
        empty = _snap(me=_player())
        s_board = static_side_score(snap, snap.me, weights=W)
        s_empty = static_side_score(empty, empty.me, weights=W)
        # 默认 w_board_residue=0 且默认 provider=0 → 场面不影响静态分
        self.assertAlmostEqual(s_board, s_empty, places=5)

    def test_injected_board_valuation(self):
        """可注入 BoardValuationProvider + 非零权重 → 场面残值生效。"""
        field = [CardView(uid=1, atk=5, life=5)]
        cards = {c.uid: c for c in field}
        snap = _snap(me=_player(field=[c.uid for c in field]), cards=cards)

        def board_provider(s, view):
            total = 0.0
            for uid in view.field:
                c = s.cards_by_uid.get(uid)
                if c is not None:
                    total += (c.atk or 0) + (c.life or 0)
            return total

        weights = MacroStaticWeights(w_board_residue=0.5)
        base = static_side_score(snap, snap.me, weights=weights)  # 默认 zero provider
        with_board = static_side_score(
            snap, snap.me, weights=weights, board_valuation=board_provider,
        )
        # 5/5 = 10 身材 * 0.5 = 5
        self.assertAlmostEqual(with_board - base, 5.0, places=5)


class TestConfigurableWeights(unittest.TestCase):
    def test_custom_weights_applied(self):
        weights = MacroStaticWeights(w_hp=1.0, w_max_pp=2.0, w_ep=3.0, w_sep=4.0)
        snap = _snap(me=_player(life=10, max_pp=5, ep=1, sep=1))
        # 10*1 + 5*2 + 1*3 + 1*4 = 10+10+3+4 = 27
        self.assertAlmostEqual(static_side_score(snap, snap.me, weights=weights), 27.0, places=5)

    def test_custom_hand_marginal(self):
        weights = MacroStaticWeights(hand_marginal=(100, 0))
        self.assertEqual(_hand_score(1, weights.hand_marginal), 100.0)
        self.assertEqual(_hand_score(3, weights.hand_marginal), 100.0)  # 第 2、3 张边际 0


if __name__ == "__main__":
    unittest.main()
