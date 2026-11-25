# -*- coding: utf-8 -*-
"""动作 pointer 表示测试。"""

import unittest

from decision_engine.actions.local_enumerator import LocalActionEnumerator
from decision_engine.actions.model import ActionKind, LegalAction, PlayKind
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView
from decision_engine.training.action_pointer import (
    ActionHead,
    PointerActionError,
    group_by_head,
    max_slot_count,
    resolve_pointer_action,
    slot_candidates,
    slot_specs,
    source_candidates,
    to_pointer_action,
)


class ActionPointerConversionTest(unittest.TestCase):
    def test_attack_leader_and_follower_share_one_head(self):
        """攻击主战者与攻击随从是同一个动作头，靠目标池区分。"""
        leader = LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=5, target_groups=((0,),), target_group_specs=("atk_leader",))
        follower = LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=5, target_groups=((7,),), target_group_specs=("atk_follower",))

        self.assertIs(to_pointer_action(leader).head, ActionHead.ATTACK)
        self.assertIs(to_pointer_action(follower).head, ActionHead.ATTACK)
        # 原始 kind 保留，反查时仍能区分打脸与打随从
        self.assertIs(to_pointer_action(leader).kind, ActionKind.ATTACK_LEADER)
        self.assertIs(to_pointer_action(follower).kind, ActionKind.ATTACK_FOLLOWER)

    def test_count_n_is_one_slot_with_arity_n(self):
        """``count=2`` 是一个槽选 2 个，不是两个槽。"""
        action = LegalAction(
            kind=ActionKind.PLAY,
            source_uid=10,
            play_kind=PlayKind.NORMAL,
            target_groups=((3, 4),),
            target_group_specs=("vira_fanfare",),
        )

        pointer = to_pointer_action(action)

        self.assertEqual(len(pointer.slots), 1)
        self.assertEqual(pointer.slots[0].arity, 2)
        self.assertEqual(pointer.slots[0].spec_id, "vira_fanfare")
        self.assertEqual(pointer.pointer_calls, 2)

    def test_slot_arity_mismatch_raises(self):
        """手写 PointerAction 时 arity 必须与实际 group 长度一致。"""
        from decision_engine.training.action_pointer import ActionSlot, PointerAction

        with self.assertRaises(PointerActionError):
            PointerAction(
                head=ActionHead.PLAY,
                kind=ActionKind.PLAY,
                source_uid=1,
                play_kind=PlayKind.NORMAL,
                targets=((2, 3),),
                slots=(ActionSlot(spec_id="x", arity=1),),
            )


class ActionPointerRoundTripTest(unittest.TestCase):
    def test_round_trip_single_slot(self):
        action = LegalAction(
            kind=ActionKind.PLAY,
            source_uid=10,
            play_kind=PlayKind.NORMAL,
            cost=3,
            target_groups=((1,),),
            target_group_specs=("spec_a",),
        )

        self.assertEqual(resolve_pointer_action(to_pointer_action(action), [action]), action)

    def test_round_trip_multi_slot(self):
        action = LegalAction(
            kind=ActionKind.PLAY,
            source_uid=10,
            play_kind=PlayKind.NORMAL,
            cost=3,
            target_groups=((11,), (2,)),
            target_group_specs=("10642310_auto_001", "10642310_auto_002"),
        )

        pointer = to_pointer_action(action)

        self.assertEqual(len(pointer.slots), 2)
        self.assertEqual(pointer.target_specs, ("10642310_auto_001", "10642310_auto_002"))
        self.assertEqual(resolve_pointer_action(pointer, [action]), action)

    def test_round_trip_ignores_description_and_legality_source(self):
        """description / legality_source / confidence 不参与动作身份。"""
        original = LegalAction(kind=ActionKind.TURN_END, source_uid=0, description="end turn", legality_source="local")
        equivalent = LegalAction(kind=ActionKind.TURN_END, source_uid=0, description="别的描述", legality_source="protocol")

        self.assertEqual(resolve_pointer_action(to_pointer_action(original), [equivalent]), equivalent)

    def test_round_trip_enhance_vs_normal_stays_distinct(self):
        """同源的爆能强化与普通打出必须能区分开。"""
        normal = LegalAction(kind=ActionKind.PLAY, source_uid=10, play_kind=PlayKind.NORMAL, cost=3)
        enhance = LegalAction(kind=ActionKind.PLAY, source_uid=10, play_kind=PlayKind.ENHANCE, cost=7)

        self.assertEqual(resolve_pointer_action(to_pointer_action(normal), [normal, enhance]), normal)
        self.assertEqual(resolve_pointer_action(to_pointer_action(enhance), [normal, enhance]), enhance)

    def test_target_group_specs_length_mismatch_raises(self):
        """specs 与 groups 不等长说明枚举器漏填，不能按位置猜。"""
        action = LegalAction(kind=ActionKind.PLAY, source_uid=1, target_groups=((2,), (3,)), target_group_specs=("only_one",))

        with self.assertRaises(PointerActionError):
            to_pointer_action(action)

    def test_resolve_raises_when_pointer_not_in_legal_actions(self):
        """反查失败必须显式报错，不退回默认动作。"""
        pointer = to_pointer_action(
            LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=5, target_groups=((7,),), target_group_specs=("x",))
        )
        other = LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=5, target_groups=((8,),), target_group_specs=("x",))

        with self.assertRaises(PointerActionError):
            resolve_pointer_action(pointer, [other])


class ActionPointerCandidateHelpersTest(unittest.TestCase):
    def _actions(self):
        return [
            LegalAction(kind=ActionKind.PLAY, source_uid=10, target_groups=((1,), (2,)), target_group_specs=("a", "b")),
            LegalAction(kind=ActionKind.PLAY, source_uid=11, target_groups=((3,),), target_group_specs=("a",)),
            LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=10, target_groups=((0,),), target_group_specs=("atk",)),
        ]

    def test_group_by_head(self):
        grouped = group_by_head(self._actions())

        self.assertEqual(set(grouped), {ActionHead.PLAY, ActionHead.ATTACK})
        self.assertEqual(len(grouped[ActionHead.PLAY]), 2)
        self.assertEqual(len(grouped[ActionHead.ATTACK]), 1)

    def test_source_and_slot_candidates(self):
        plays = group_by_head(self._actions())[ActionHead.PLAY]

        self.assertEqual(source_candidates(plays), (10, 11))
        self.assertEqual(slot_candidates(plays, 0), (1, 3))
        # 槽 1 只有第一个动作有
        self.assertEqual(slot_candidates(plays, 1), (2,))
        self.assertEqual(slot_specs(plays, 0), ("a",))
        self.assertEqual(slot_specs(plays, 1), ("b",))

    def test_max_slot_count(self):
        self.assertEqual(max_slot_count(self._actions()), 2)
        self.assertEqual(max_slot_count([]), 0)


def _multi_slot_snapshot() -> BattleSnapshot:
    """10642310（赤流）的 fanfare 有两个 selected_* spec：弃自己 1 张手牌 + 选对方 1 个随从。"""
    cards = {
        1: CardView(uid=1, card_type=CardType.FOLLOWER, atk=1, life=1),
        2: CardView(uid=2, card_type=CardType.FOLLOWER, atk=1, life=1),
        10: CardView(uid=10, base_card_id=10642310, cost=3, card_type=CardType.FOLLOWER, atk=2, life=2),
        11: CardView(uid=11, cost=1, card_type=CardType.SPELL),
        12: CardView(uid=12, cost=1, card_type=CardType.SPELL),
    }
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=20, max_pp=20, hand=(10, 11, 12)),
        opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=20, unique_id=99, field=(1, 2)),
        cards_by_uid=cards,
        legal=None,
        is_ally_turn=True,
        is_turn_ended=False,
    )


class ActionPointerEnumeratorIntegrationTest(unittest.TestCase):
    """用真实枚举器产出验证：每个合法动作都能无损往返。"""

    def test_every_enumerated_action_round_trips(self):
        actions = LocalActionEnumerator().enumerate(_multi_slot_snapshot())

        self.assertTrue(actions)
        for action in actions:
            self.assertEqual(len(action.target_group_specs), len(action.target_groups), action)
            pointer = to_pointer_action(action)
            self.assertEqual(resolve_pointer_action(pointer, actions), action)

    def test_multi_slot_action_exposes_two_distinct_slots(self):
        actions = [
            a
            for a in LocalActionEnumerator().enumerate(_multi_slot_snapshot())
            if a.kind is ActionKind.PLAY and a.source_uid == 10
        ]
        self.assertTrue(actions)
        pointer = to_pointer_action(actions[0])

        self.assertEqual(len(pointer.slots), 2)
        self.assertEqual(pointer.target_specs, ("10642310_auto_001", "10642310_auto_002"))
        self.assertEqual([slot.arity for slot in pointer.slots], [1, 1])


def _attack_snapshot() -> BattleSnapshot:
    """我方一个可攻击随从、对方一个可被攻击的随从 + 主战者。"""
    cards = {
        1: CardView(uid=1, card_type=CardType.FOLLOWER, atk=1, life=1, attacks_left=1),
        20: CardView(uid=20, card_type=CardType.FOLLOWER, atk=3, life=3, attacks_left=1, can_attack=True, can_attack_leader=True),
    }
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=0, max_pp=20, field=(20,)),
        opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=20, unique_id=99, field=(1,)),
        cards_by_uid=cards,
        legal=None,
        is_ally_turn=True,
        is_turn_ended=False,
    )


class ActionPointerAttackTest(unittest.TestCase):
    def test_attack_actions_carry_spec_ids_and_round_trip(self):
        """攻击动作也必须带 spec 标识；打随从与打脸归并到同一个 ActionHead.ATTACK。"""
        actions = LocalActionEnumerator().enumerate(_attack_snapshot())
        attacks = [a for a in actions if a.kind in (ActionKind.ATTACK_FOLLOWER, ActionKind.ATTACK_LEADER)]

        self.assertTrue(attacks, "预期至少枚举出攻击动作")
        for action in attacks:
            self.assertEqual(len(action.target_group_specs), len(action.target_groups), action)
            self.assertEqual(to_pointer_action(action).head, ActionHead.ATTACK)
            self.assertEqual(resolve_pointer_action(to_pointer_action(action), actions), action)

        specs_by_kind = {action.kind: action.target_group_specs for action in attacks}
        self.assertEqual(specs_by_kind[ActionKind.ATTACK_FOLLOWER], ("attack_follower",))
        self.assertEqual(specs_by_kind[ActionKind.ATTACK_LEADER], ("attack_leader",))


if __name__ == "__main__":
    unittest.main()
