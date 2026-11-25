# -*- coding: utf-8 -*-
"""FutureResourceSignature 单元测试。

覆盖 P2 验收要点：
- 左右视角对称（me 视角 ally == opponent 视角 enemy 的对应字段）；
- 对手手牌 base-id 不泄漏（对手侧 known_hand 恒为 None）；
- EP / SEP / extra_pp lifecycle / crest 身份的关键差异会区分；
- 己方 known hand base-id multiset；
- extra_pp 单帧不臆测 SPENT（无 TurnLine 历史时只落 NONE/HOLD/EXPIRING/UNKNOWN）。
"""

import unittest

from decision_engine.planning.future_resource_signature import (
    EXTRA_PP_SIG_EXPIRING,
    EXTRA_PP_SIG_HOLD,
    EXTRA_PP_SIG_NONE,
    EXTRA_PP_SIG_SPENT,
    EXTRA_PP_SIG_UNKNOWN,
    FutureResourceSignature,
)
from decision_engine.state.snapshot import (
    BattleSnapshot,
    CardType,
    CardView,
    CrestView,
    ExtraPpView,
    LegalView,
    PlayerView,
)


def _card(uid: int, base_card_id: int, card_type: CardType = CardType.SPELL) -> CardView:
    return CardView(uid=uid, card_id=base_card_id, base_card_id=base_card_id, card_type=card_type)


def _snapshot(
    *,
    me: PlayerView,
    opponent: PlayerView,
    cards: dict[int, CardView] | None = None,
) -> BattleSnapshot:
    return BattleSnapshot(
        me=me,
        opponent=opponent,
        cards_by_uid=cards or {},
        legal=LegalView(),
    )


class FutureResourceSignatureTest(unittest.TestCase):
    def _make_snapshot(self):
        # 己方手牌：uid 101->base 5001, uid 102->base 5002
        my_hand = {101: _card(101, 5001), 102: _card(102, 5002)}
        # 对手手牌：uid 201->base 9001（本应绝不泄漏）
        opp_hand = {201: _card(201, 9001), 202: _card(202, 9002)}
        cards = {**my_hand, **opp_hand}
        me = PlayerView(
            life=20, max_life=20, pp=5, max_pp=7, ep=2, sep=1, deck_count=25,
            is_first_side=False, turn=4,
            hand=(101, 102),
            crests=(CrestView(card_id=6001, countdown=3, faith_value=0),),
            extra_pp=ExtraPpView(extra_pp=1),
        )
        opponent = PlayerView(
            life=18, max_life=20, pp=6, max_pp=8, ep=3, sep=0, deck_count=22,
            is_first_side=True, turn=4,
            hand=(201, 202),
            crests=(CrestView(card_id=7001, countdown=1, faith_value=2),),
        )
        return _snapshot(me=me, opponent=opponent, cards=cards)

    def test_perspective_symmetry(self):
        """me 视角的 enemy 应等于 opponent 视角的 ally，反之亦然。"""
        snap = self._make_snapshot()
        me_view = FutureResourceSignature.from_snapshot(snap, "me")
        opp_view = FutureResourceSignature.from_snapshot(snap, "opponent")

        # me 视角己方(=snapshot.me) 生命/资源，应与 opponent 视角对手(=snapshot.me) 对应字段一致，
        # 但 known_hand 只在"己方视角"暴露，故这里比较不含 known_hand 的字段。
        self.assertEqual(me_view.ally.life, opp_view.enemy.life)
        self.assertEqual(me_view.ally.ep, opp_view.enemy.ep)
        self.assertEqual(me_view.ally.sep, opp_view.enemy.sep)
        self.assertEqual(me_view.ally.max_pp, opp_view.enemy.max_pp)
        self.assertEqual(me_view.ally.deck_count, opp_view.enemy.deck_count)
        self.assertEqual(me_view.ally.crest_identity, opp_view.enemy.crest_identity)
        self.assertEqual(me_view.ally.hand_count, opp_view.enemy.hand_count)
        self.assertEqual(me_view.ally.extra_pp_lifecycle, opp_view.enemy.extra_pp_lifecycle)

        # 对称：me 视角对手(=snapshot.opponent) 应等于 opponent 视角己方(=snapshot.opponent)。
        self.assertEqual(me_view.enemy.life, opp_view.ally.life)
        self.assertEqual(me_view.enemy.crest_identity, opp_view.ally.crest_identity)
        self.assertEqual(me_view.enemy.hand_count, opp_view.ally.hand_count)

    def test_opponent_hand_ids_not_leaked(self):
        """对手侧 known_hand 恒为 None，只暴露 count；己方侧暴露 base-id multiset。"""
        snap = self._make_snapshot()
        me_view = FutureResourceSignature.from_snapshot(snap, "me")

        # 己方 known hand = base id 多重集（排序）。
        self.assertEqual(me_view.ally.known_hand, (5001, 5002))
        self.assertEqual(me_view.ally.hand_count, 2)

        # 对手 known_hand 必须为 None，且 count 正确。
        self.assertIsNone(me_view.enemy.known_hand)
        self.assertEqual(me_view.enemy.hand_count, 2)

        # 从 opponent 视角看，snapshot.me 成为对手，其 known_hand 也必须为 None
        # （视角切换后仍不泄漏当前视角对手的手牌）。
        opp_view = FutureResourceSignature.from_snapshot(snap, "opponent")
        self.assertIsNone(opp_view.enemy.known_hand)
        # 而 opponent 视角己方(=snapshot.opponent) 现在暴露自己的 known hand。
        self.assertEqual(opp_view.ally.known_hand, (9001, 9002))

    def test_known_hand_multiset_counts_duplicates(self):
        """己方 known hand 是多重集，重复 base-id 保留。"""
        cards = {1: _card(1, 5001), 2: _card(2, 5001), 3: _card(3, 5002)}
        me = PlayerView(life=20, max_life=20, pp=1, max_pp=1, hand=(1, 2, 3), is_first_side=True)
        opp = PlayerView(life=20, max_life=20, pp=1, max_pp=1)
        view = FutureResourceSignature.from_snapshot(_snapshot(me=me, opponent=opp, cards=cards), "me")
        self.assertEqual(view.ally.known_hand, (5001, 5001, 5002))

    def test_ep_sep_difference_distinguishes(self):
        me_a = PlayerView(life=20, max_life=20, pp=1, max_pp=1, ep=2, sep=1, is_first_side=True)
        me_b = PlayerView(life=20, max_life=20, pp=1, max_pp=1, ep=1, sep=1, is_first_side=True)
        me_c = PlayerView(life=20, max_life=20, pp=1, max_pp=1, ep=2, sep=0, is_first_side=True)
        opp = PlayerView(life=20, max_life=20, pp=1, max_pp=1)
        sig_a = FutureResourceSignature.from_snapshot(_snapshot(me=me_a, opponent=opp), "me")
        sig_b = FutureResourceSignature.from_snapshot(_snapshot(me=me_b, opponent=opp), "me")
        sig_c = FutureResourceSignature.from_snapshot(_snapshot(me=me_c, opponent=opp), "me")
        self.assertNotEqual(sig_a, sig_b)  # EP 差异
        self.assertNotEqual(sig_a, sig_c)  # SEP 差异

    def test_crest_identity_difference_distinguishes(self):
        opp = PlayerView(life=20, max_life=20, pp=1, max_pp=1)
        me_a = PlayerView(
            life=20, max_life=20, pp=1, max_pp=1, is_first_side=True,
            crests=(CrestView(card_id=6001, countdown=3, faith_value=0),),
        )
        me_b = PlayerView(
            life=20, max_life=20, pp=1, max_pp=1, is_first_side=True,
            crests=(CrestView(card_id=6001, countdown=2, faith_value=0),),  # countdown 不同
        )
        me_c = PlayerView(
            life=20, max_life=20, pp=1, max_pp=1, is_first_side=True,
            crests=(CrestView(card_id=6002, countdown=3, faith_value=0),),  # card_id 不同
        )
        sig_a = FutureResourceSignature.from_snapshot(_snapshot(me=me_a, opponent=opp), "me")
        sig_b = FutureResourceSignature.from_snapshot(_snapshot(me=me_b, opponent=opp), "me")
        sig_c = FutureResourceSignature.from_snapshot(_snapshot(me=me_c, opponent=opp), "me")
        self.assertNotEqual(sig_a, sig_b)
        self.assertNotEqual(sig_a, sig_c)

    def test_crest_identity_order_independent(self):
        """纹章排列顺序不影响身份签名（排序稳定）。"""
        opp = PlayerView(life=20, max_life=20, pp=1, max_pp=1)
        c1 = CrestView(card_id=6001, countdown=3, faith_value=0)
        c2 = CrestView(card_id=6002, countdown=1, faith_value=1)
        me_a = PlayerView(life=20, max_life=20, pp=1, max_pp=1, is_first_side=True, crests=(c1, c2))
        me_b = PlayerView(life=20, max_life=20, pp=1, max_pp=1, is_first_side=True, crests=(c2, c1))
        sig_a = FutureResourceSignature.from_snapshot(_snapshot(me=me_a, opponent=opp), "me")
        sig_b = FutureResourceSignature.from_snapshot(_snapshot(me=me_b, opponent=opp), "me")
        self.assertEqual(sig_a, sig_b)

    def test_extra_pp_lifecycle_first_side_is_none(self):
        opp = PlayerView(life=20, max_life=20, pp=1, max_pp=1)
        me = PlayerView(
            life=20, max_life=20, pp=1, max_pp=1, is_first_side=True, turn=4,
            extra_pp=ExtraPpView(extra_pp=1),  # 即使协议给了 token，先手也判 NONE
        )
        sig = FutureResourceSignature.from_snapshot(_snapshot(me=me, opponent=opp), "me")
        self.assertEqual(sig.ally.extra_pp_lifecycle, EXTRA_PP_SIG_NONE)

    def test_extra_pp_lifecycle_hold_and_none(self):
        opp = PlayerView(life=20, max_life=20, pp=1, max_pp=1)
        # 后手持有 token → HOLD
        me_hold = PlayerView(
            life=20, max_life=20, pp=1, max_pp=1, is_first_side=False, turn=4,
            extra_pp=ExtraPpView(extra_pp=1),
        )
        # 后手未持有 token → NONE
        me_none = PlayerView(
            life=20, max_life=20, pp=1, max_pp=1, is_first_side=False, turn=4,
            extra_pp=ExtraPpView(extra_pp=0),
        )
        sig_hold = FutureResourceSignature.from_snapshot(_snapshot(me=me_hold, opponent=opp), "me")
        sig_none = FutureResourceSignature.from_snapshot(_snapshot(me=me_none, opponent=opp), "me")
        self.assertEqual(sig_hold.ally.extra_pp_lifecycle, EXTRA_PP_SIG_HOLD)
        self.assertEqual(sig_none.ally.extra_pp_lifecycle, EXTRA_PP_SIG_NONE)
        self.assertNotEqual(sig_hold, sig_none)  # lifecycle 差异会区分

    def test_extra_pp_lifecycle_expiring_t5_second_side(self):
        opp = PlayerView(life=20, max_life=20, pp=1, max_pp=1)
        me = PlayerView(
            life=20, max_life=20, pp=1, max_pp=1, is_first_side=False, turn=5,
            extra_pp=ExtraPpView(extra_pp=1),
        )
        sig = FutureResourceSignature.from_snapshot(_snapshot(me=me, opponent=opp), "me")
        self.assertEqual(sig.ally.extra_pp_lifecycle, EXTRA_PP_SIG_EXPIRING)

    def test_extra_pp_lifecycle_unknown_when_view_missing(self):
        """后手但 extra_pp 视图缺失 → UNKNOWN，保守分离，不臆测。"""
        opp = PlayerView(life=20, max_life=20, pp=1, max_pp=1)
        me = PlayerView(
            life=20, max_life=20, pp=1, max_pp=1, is_first_side=False, turn=4,
            extra_pp=None,
        )
        sig = FutureResourceSignature.from_snapshot(_snapshot(me=me, opponent=opp), "me")
        self.assertEqual(sig.ally.extra_pp_lifecycle, EXTRA_PP_SIG_UNKNOWN)

    def test_never_infers_spent_from_single_snapshot(self):
        """单帧（无 TurnLine 历史）绝不产出 SPENT。"""
        opp = PlayerView(life=20, max_life=20, pp=1, max_pp=1)
        # 即便 is_used_extra_pp_this_turn=True 的瞬态，也不得判 SPENT。
        me = PlayerView(
            life=20, max_life=20, pp=1, max_pp=1, is_first_side=False, turn=4,
            extra_pp=ExtraPpView(extra_pp=1, is_used_extra_pp_this_turn=True),
        )
        sig = FutureResourceSignature.from_snapshot(_snapshot(me=me, opponent=opp), "me")
        self.assertNotEqual(sig.ally.extra_pp_lifecycle, EXTRA_PP_SIG_SPENT)
        self.assertEqual(sig.ally.extra_pp_lifecycle, EXTRA_PP_SIG_HOLD)

    def test_invalid_perspective_raises(self):
        me = PlayerView(life=20, max_life=20, pp=1, max_pp=1)
        opp = PlayerView(life=20, max_life=20, pp=1, max_pp=1)
        with self.assertRaises(ValueError):
            FutureResourceSignature.from_snapshot(_snapshot(me=me, opponent=opp), "spectator")  # type: ignore[arg-type]

    def test_hashable_and_deterministic(self):
        snap = self._make_snapshot()
        a = FutureResourceSignature.from_snapshot(snap, "me")
        b = FutureResourceSignature.from_snapshot(snap, "me")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        _ = {a, b}  # 可作为 set 元素


if __name__ == "__main__":
    unittest.main()
