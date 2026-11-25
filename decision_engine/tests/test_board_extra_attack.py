# -*- coding: utf-8 -*-
"""board 维度的"额外攻击期权"：双击/多次攻击的预期伤害必须可见。

背景：规则内核与 effect spec 早就支持「1回合可以攻击2次」（90034350 宏大的回归 →
modify_attack_count，attacks_left 1→2），但势函数只读 count/atk/life，
于是"给双击"这个动作在势里是 +0——20 档兑现的质变被低估。

这里锁定三件事：
1. 期权只在"当前行动方且能攻击"时计入，不会给对手残留次数虚高；
2. 期权与血量维度精确对冲（给的时候记、打的时候扣），不双重计价；
3. 打满两刀的累计收益 = 两刀的真实伤害。
"""

from __future__ import annotations

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.incremental_value import (
    StepRewardWeights,
    _board_signal,
    step_reward,
)
from decision_engine.state.snapshot import (
    BattleSnapshot, CardType, CardView, LegalView, PlayerView,
)

W = StepRewardWeights()


def _follower(uid: int, *, atk: int = 5, life: int = 5, attacks_left: int = 1,
              can_attack: bool = True) -> CardView:
    return CardView(
        uid=uid, card_id=900000 + uid, base_card_id=900000 + uid,
        atk=atk, life=life, max_life=life, card_type=CardType.FOLLOWER,
        can_attack=can_attack, can_attack_leader=can_attack,
        attacks_left=attacks_left, attack_limit=max(1, attacks_left),
    )


def _snap(*, my_field=(), opp_field=(), my_life: int = 20, opp_life: int = 20,
          is_ally_turn: bool = True) -> BattleSnapshot:
    cards = {}
    mf, of = [], []
    for card in my_field:
        cards[card.uid] = card
        mf.append(card.uid)
    for card in opp_field:
        cards[card.uid] = card
        of.append(card.uid)
    return BattleSnapshot(
        me=PlayerView(life=my_life, max_life=20, pp=10, max_pp=10, field=tuple(mf)),
        opponent=PlayerView(life=opp_life, max_life=20, pp=10, max_pp=10, field=tuple(of)),
        cards_by_uid=cards, legal=LegalView(), is_ally_turn=is_ally_turn,
    )


class BoardExtraAttackSignalTest(unittest.TestCase):
    def test_double_attack_adds_expected_damage_option(self) -> None:
        single = _snap(my_field=[_follower(1, atk=5, attacks_left=1)])
        double = _snap(my_field=[_follower(1, atk=5, attacks_left=2)])
        delta = _board_signal(double, double.me, W) - _board_signal(single, single.me, W)
        self.assertAlmostEqual(delta, 5 * W.w_board_extra_attack)
        self.assertAlmostEqual(delta, 5 * W.w_face, msg="5 攻双击的期权应等于 5 点伤害")

    def test_cant_attack_does_not_earn_option(self) -> None:
        stuck = _snap(my_field=[_follower(1, atk=5, attacks_left=2, can_attack=False)])
        single = _snap(my_field=[_follower(1, atk=5, attacks_left=1)])
        self.assertAlmostEqual(
            _board_signal(stuck, stuck.me, W), _board_signal(single, single.me, W)
        )

    def test_non_active_side_does_not_earn_option(self) -> None:
        """对手回合里我方随从残留的 attacks_left=2 不是当前威胁。"""
        snap = _snap(my_field=[_follower(1, atk=5, attacks_left=2)], is_ally_turn=False)
        baseline = _snap(my_field=[_follower(1, atk=5, attacks_left=1)], is_ally_turn=False)
        self.assertAlmostEqual(
            _board_signal(snap, snap.me, W), _board_signal(baseline, baseline.me, W)
        )

    def test_opponent_side_earns_option_on_their_turn(self) -> None:
        snap = _snap(opp_field=[_follower(2, atk=5, attacks_left=2)], is_ally_turn=False)
        baseline = _snap(opp_field=[_follower(2, atk=5, attacks_left=1)], is_ally_turn=False)
        delta = _board_signal(snap, snap.opponent, W) - _board_signal(baseline, baseline.opponent, W)
        self.assertAlmostEqual(delta, 5 * W.w_board_extra_attack)

    def test_zero_attacks_left_is_not_negative(self) -> None:
        spent = _snap(my_field=[_follower(1, atk=5, attacks_left=0)])
        single = _snap(my_field=[_follower(1, atk=5, attacks_left=1)])
        self.assertAlmostEqual(
            _board_signal(spent, spent.me, W), _board_signal(single, single.me, W)
        )


class ExtraAttackStepRewardTest(unittest.TestCase):
    """端到端 step_reward：期权在"给"的时候记、在"打"的时候扣回。"""

    def test_granting_double_attack_scores_option(self) -> None:
        prev = _snap(my_field=[_follower(1, atk=5, attacks_left=1)])
        nxt = _snap(my_field=[_follower(1, atk=5, attacks_left=2)])
        action = LegalAction(kind=ActionKind.PLAY, source_uid=90034350, description="")
        reward = step_reward(prev, action, nxt, "me", weights=W)
        self.assertAlmostEqual(reward, 5 * W.w_board_extra_attack)

    def test_first_attack_swaps_option_for_damage(self) -> None:
        prev = _snap(my_field=[_follower(1, atk=5, attacks_left=2)], opp_life=20)
        nxt = _snap(my_field=[_follower(1, atk=5, attacks_left=1)], opp_life=15)
        action = LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=1, description="")
        reward = step_reward(prev, action, nxt, "me", weights=W)
        self.assertAlmostEqual(reward, 0.0, msg="血量 +5PP 与期权 −5PP 对冲")

    def test_second_attack_scores_full_damage(self) -> None:
        prev = _snap(my_field=[_follower(1, atk=5, attacks_left=1)], opp_life=15)
        nxt = _snap(my_field=[_follower(1, atk=5, attacks_left=0)], opp_life=10)
        action = LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=1, description="")
        reward = step_reward(prev, action, nxt, "me", weights=W)
        self.assertAlmostEqual(reward, 5 * W.w_face)

    def test_full_sequence_equals_two_attacks_of_damage(self) -> None:
        """给双击 + 打两刀的总收益 = 两刀真实伤害（10 点 = 10PP）。"""
        action_grant = LegalAction(kind=ActionKind.PLAY, source_uid=90034350, description="")
        action_attack = LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=1, description="")
        s0 = _snap(my_field=[_follower(1, atk=5, attacks_left=1)], opp_life=20)
        s1 = _snap(my_field=[_follower(1, atk=5, attacks_left=2)], opp_life=20)
        s2 = _snap(my_field=[_follower(1, atk=5, attacks_left=1)], opp_life=15)
        s3 = _snap(my_field=[_follower(1, atk=5, attacks_left=0)], opp_life=10)
        total = (
            step_reward(s0, action_grant, s1, "me", weights=W)
            + step_reward(s1, action_attack, s2, "me", weights=W)
            + step_reward(s2, action_attack, s3, "me", weights=W)
        )
        self.assertAlmostEqual(total, 10 * W.w_face)


if __name__ == "__main__":
    unittest.main()
