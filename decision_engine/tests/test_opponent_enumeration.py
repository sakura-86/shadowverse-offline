# -*- coding: utf-8 -*-
"""Tests for opponent perspective action enumeration.

验证 PlayerLocalActionEnumerator(player='opponent') 通过 opponent_as_me 视角交换后，
能正确枚举对方的 play/attack/evolve/super_evolve/activate/turn_end 等动作。
"""

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.actions.player_local_enumerator import PlayerLocalActionEnumerator
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView


def _follower(uid: int, atk: int, life: int, **kwargs) -> CardView:
    return CardView(
        uid=uid, card_id=uid * 100, base_card_id=uid * 100,
        atk=atk, life=life, max_life=life,
        card_type=CardType.FOLLOWER, **kwargs,
    )


def _spell(uid: int, cost: int) -> CardView:
    return CardView(uid=uid, card_id=uid * 100, base_card_id=uid * 100, cost=cost, card_type=CardType.SPELL)


class TestOpponentEnumeration(unittest.TestCase):
    """对方视角动作枚举集成测试。"""

    def setUp(self):
        self.enumerator = PlayerLocalActionEnumerator()

    def _snapshot(
        self,
        *,
        my_field: tuple[CardView, ...] = (),
        my_hand: tuple[CardView, ...] = (),
        opp_field: tuple[CardView, ...] = (),
        opp_hand: tuple[CardView, ...] = (),
        my_life: int = 20,
        opp_life: int = 20,
        opp_pp: int = 3,
        opp_ep: int = 0,
        opp_sep: int = 0,
        opp_turn: int = 3,
        opp_evolve_turn: int = 2,
        opp_super_evolve_turn: int = 0,
        opp_is_evolved_this_turn: bool = False,
    ) -> BattleSnapshot:
        cards: dict[int, CardView] = {}
        my_uids = []
        opp_uids = []
        my_hand_uids = []
        opp_hand_uids = []
        for c in my_field:
            cards[c.uid] = c
            my_uids.append(c.uid)
        for c in my_hand:
            cards[c.uid] = c
            my_hand_uids.append(c.uid)
        for c in opp_field:
            cards[c.uid] = c
            opp_uids.append(c.uid)
        for c in opp_hand:
            cards[c.uid] = c
            opp_hand_uids.append(c.uid)

        return BattleSnapshot(
            me=PlayerView(
                life=my_life, max_life=20, pp=0, max_pp=3,
                hand=tuple(my_hand_uids), field=tuple(my_uids), unique_id=1,
            ),
            opponent=PlayerView(
                life=opp_life, max_life=20, pp=opp_pp, max_pp=opp_pp,
                hand=tuple(opp_hand_uids), field=tuple(opp_uids), unique_id=2,
                turn=opp_turn, ep=opp_ep, sep=opp_sep,
                evolve_turn=opp_evolve_turn, super_evolve_turn=opp_super_evolve_turn,
                is_evolved_this_turn=opp_is_evolved_this_turn,
                deck_count=10,
            ),
            cards_by_uid=cards,
            is_ally_turn=False,
            is_turn_ended=False,
        )

    def _kinds(self, actions: list[LegalAction]) -> list[str]:
        return [a.kind.value for a in actions]

    # ------------------------------------------------------------------
    # 基本枚举
    # ------------------------------------------------------------------

    def test_opponent_always_has_turn_end(self):
        """对方任何时候都有 turn_end 可选。"""
        snapshot = self._snapshot()
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        self.assertIn("turn_end", self._kinds(actions))

    def test_opponent_empty_field_and_hand_only_turn_end(self):
        """对方空手空场只有 turn_end。"""
        snapshot = self._snapshot(opp_pp=0)
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        self.assertEqual(self._kinds(actions), ["turn_end"])

    # ------------------------------------------------------------------
    # 攻击
    # ------------------------------------------------------------------

    def test_opponent_attacks_face_without_guard(self):
        """无守护时对方可打脸。"""
        attacker = _follower(10, atk=3, life=3, can_attack=True, can_attack_leader=True, attacks_left=1)
        snapshot = self._snapshot(opp_field=(attacker,))
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        self.assertIn("attack_leader", self._kinds(actions))

    def test_opponent_attacks_follower_without_guard(self):
        """无守护时对方可打随从。"""
        attacker = _follower(10, atk=3, life=3, can_attack=True, can_attack_leader=True, attacks_left=1)
        my_follower = _follower(20, atk=2, life=2)
        snapshot = self._snapshot(opp_field=(attacker,), my_field=(my_follower,))
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        attack_follower = [a for a in actions if a.kind.value == "attack_follower"]
        self.assertEqual(len(attack_follower), 1)
        self.assertEqual(attack_follower[0].target_groups, ((20,),))

    def test_opponent_blocked_by_guard(self):
        """有守护时对方只能打守护，不能打脸或其他随从。"""
        attacker = _follower(10, atk=3, life=3, can_attack=True, can_attack_leader=True, attacks_left=1)
        guard = _follower(20, atk=1, life=5, has_guard=True)
        other = _follower(21, atk=2, life=2)
        snapshot = self._snapshot(opp_field=(attacker,), my_field=(guard, other))
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        kinds = self._kinds(actions)
        self.assertNotIn("attack_leader", kinds)
        attack_follower = [a for a in actions if a.kind.value == "attack_follower"]
        # 只能打守护
        targets = [a.target_groups[0][0] for a in attack_follower]
        self.assertEqual(targets, [20])

    def test_opponent_cannot_attack_ambush(self):
        """潜行随从不能被攻击。"""
        attacker = _follower(10, atk=3, life=3, can_attack=True, can_attack_leader=True, attacks_left=1)
        ambush = _follower(20, atk=4, life=4, has_ambush=True)
        normal = _follower(21, atk=2, life=2)
        snapshot = self._snapshot(opp_field=(attacker,), my_field=(ambush, normal))
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        attack_follower = [a for a in actions if a.kind.value == "attack_follower"]
        # 只能打 normal
        targets = [a.target_groups[0][0] for a in attack_follower]
        self.assertNotIn(20, targets)
        self.assertIn(21, targets)

    def test_opponent_cannot_attack_cant_be_attacked(self):
        """has_cant_be_attacked 的随从不能被攻击。"""
        attacker = _follower(10, atk=3, life=3, can_attack=True, can_attack_leader=True, attacks_left=1)
        untargetable = _follower(20, atk=4, life=4, has_cant_be_attacked=True)
        snapshot = self._snapshot(opp_field=(attacker,), my_field=(untargetable,))
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        attack_follower = [a for a in actions if a.kind.value == "attack_follower"]
        self.assertEqual(len(attack_follower), 0)
        # 但可打脸（无守护）
        self.assertIn("attack_leader", self._kinds(actions))

    def test_opponent_no_attack_when_cant_attack(self):
        """has_cant_attack 或 attacks_left=0 不能攻击。"""
        cant_attack = _follower(10, atk=3, life=3, can_attack=False, can_attack_leader=False, attacks_left=0)
        snapshot = self._snapshot(opp_field=(cant_attack,))
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        kinds = self._kinds(actions)
        self.assertNotIn("attack_leader", kinds)
        self.assertNotIn("attack_follower", kinds)

    # ------------------------------------------------------------------
    # 出牌
    # ------------------------------------------------------------------

    def test_opponent_plays_spell_with_enough_pp(self):
        """PP 够时对方可出法术。"""
        hand_card = _spell(11, cost=2)
        snapshot = self._snapshot(opp_hand=(hand_card,), opp_pp=3)
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        play_actions = [a for a in actions if a.kind.value == "play"]
        self.assertEqual(len(play_actions), 1)
        self.assertEqual(play_actions[0].source_uid, 11)

    def test_opponent_cannot_play_when_pp_insufficient(self):
        """PP 不够时不能出牌。"""
        hand_card = _spell(11, cost=5)
        snapshot = self._snapshot(opp_hand=(hand_card,), opp_pp=3)
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        play_actions = [a for a in actions if a.kind.value == "play"]
        self.assertEqual(len(play_actions), 0)

    def test_opponent_cannot_play_follower_when_field_full(self):
        """场满时不能出随从。"""
        hand_card = _follower(11, atk=1, life=1, cost=1)
        field = tuple(_follower(i, atk=1, life=1, can_attack=False, attacks_left=0) for i in range(1, 6))
        snapshot = self._snapshot(opp_field=field, opp_hand=(hand_card,), opp_pp=3)
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        play_actions = [a for a in actions if a.kind.value == "play"]
        self.assertEqual(len(play_actions), 0)

    def test_opponent_can_play_spell_when_field_full(self):
        """场满时仍可出法术。"""
        hand_card = _spell(11, cost=1)
        field = tuple(_follower(i, atk=1, life=1, can_attack=False, attacks_left=0) for i in range(1, 6))
        snapshot = self._snapshot(opp_field=field, opp_hand=(hand_card,), opp_pp=3)
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        play_actions = [a for a in actions if a.kind.value == "play"]
        self.assertEqual(len(play_actions), 1)

    # ------------------------------------------------------------------
    # 进化
    # ------------------------------------------------------------------

    def test_opponent_evolve_with_ep(self):
        """对方有 EP 且满足进化回合条件时可进化。"""
        unevolved = _follower(10, atk=2, life=2, evolve_state=0, can_evolve=True)
        snapshot = self._snapshot(opp_field=(unevolved,), opp_ep=1)
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        self.assertIn("evolve", self._kinds(actions))

    def test_opponent_cannot_evolve_without_ep(self):
        """对方没有 EP 不能进化。"""
        unevolved = _follower(10, atk=2, life=2, evolve_state=0, can_evolve=True)
        snapshot = self._snapshot(opp_field=(unevolved,), opp_ep=0)
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        self.assertNotIn("evolve", self._kinds(actions))

    def test_opponent_cannot_evolve_twice_in_turn(self):
        """同一回合已进化过不能再进化。"""
        unevolved = _follower(10, atk=2, life=2, evolve_state=0, can_evolve=True)
        snapshot = self._snapshot(opp_field=(unevolved,), opp_ep=2, opp_is_evolved_this_turn=True)
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        self.assertNotIn("evolve", self._kinds(actions))

    def test_opponent_cannot_evolve_already_evolved(self):
        """已进化的随从不能再进化。"""
        evolved = _follower(10, atk=4, life=4, evolve_state=1, can_evolve=False)
        snapshot = self._snapshot(opp_field=(evolved,), opp_ep=1)
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        self.assertNotIn("evolve", self._kinds(actions))

    # ------------------------------------------------------------------
    # 超进化
    # ------------------------------------------------------------------

    def test_opponent_super_evolve_with_sep(self):
        """对方有 SEP 且满足条件时可超进化。"""
        unevolved = _follower(10, atk=2, life=2, evolve_state=0, can_super_evolve=True)
        snapshot = self._snapshot(opp_field=(unevolved,), opp_sep=1, opp_super_evolve_turn=2)
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        self.assertIn("super_evolve", self._kinds(actions))

    def test_opponent_cannot_super_evolve_without_sep(self):
        """对方没有 SEP 不能超进化。"""
        unevolved = _follower(10, atk=2, life=2, evolve_state=0, can_super_evolve=True)
        snapshot = self._snapshot(opp_field=(unevolved,), opp_sep=0, opp_super_evolve_turn=2)
        actions = self.enumerator.enumerate(snapshot, player="opponent")
        self.assertNotIn("super_evolve", self._kinds(actions))

    # ------------------------------------------------------------------
    # 不枚举我方回合
    # ------------------------------------------------------------------

    def test_returns_empty_for_me_when_not_ally_turn(self):
        """is_ally_turn=False 时，player='me' 枚举应为空。"""
        snapshot = self._snapshot()
        actions = self.enumerator.enumerate(snapshot, player="me")
        self.assertEqual(actions, [])


if __name__ == "__main__":
    unittest.main()
