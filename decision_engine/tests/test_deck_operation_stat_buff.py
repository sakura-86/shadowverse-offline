# -*- coding: utf-8 -*-
"""`deck_operation kind=stat_buff`（牌组内随从 +X/+Y）测试。

覆盖：
1. 只作用于仍在 draw_pile 的实例——「牌组中」不含已抽到手上的卡；
2. 按 filters 限定（card_type=follower 时护符/法术不变）；
3. buff 随抽牌带进手牌（draw_next_from_deck_instances 直接读 instance.card）；
4. deck_instances 缺失（replay 协议不给牌组构成）时显式失败，不静默跳过。
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from decision_engine.cards.templates import OperationSpec
from decision_engine.execution.effect_interpreter import EffectInterpretationContext, EffectInterpreter, EffectInterpreterError
from decision_engine.execution.local_match_setup import draw_next_from_deck_instances
from decision_engine.state.deck_instance import make_deck_instance_state
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView


def _card(uid: int | None, base_card_id: int, card_type: CardType, *, atk: int = 1, life: int = 1) -> CardView:
    return CardView(
        uid=uid if uid is not None else 0,
        card_id=base_card_id,
        base_card_id=base_card_id,
        name=f"c{base_card_id}",
        card_type=card_type,
        atk=atk,
        life=life,
        max_life=life,
        cost=2,
    )


def _snapshot(me_deck, *, hand=()):
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=3, max_pp=3, hand=tuple(hand), deck_count=len(me_deck.draw_pile) if me_deck else 0, deck_instances=me_deck),
        opponent=PlayerView(life=20, max_life=20, pp=3, max_pp=3),
        cards_by_uid={},
        source="test",
    )


DECK_OP = OperationSpec(
    op="deck_operation",
    target_ref="self_deck",
    params={
        "kind": "stat_buff",
        "attack": 1,
        "life": 1,
        "filters": [
            {"field": "zone", "operator": "==", "value": "deck"},
            {"field": "card_type", "operator": "==", "value": "follower"},
        ],
    },
)


class DeckStatBuffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.interpreter = EffectInterpreter()

    def _run(self, snapshot):
        return self.interpreter.apply_operations(
            snapshot,
            (DECK_OP,),
            EffectInterpretationContext(source_uid=1, player="me"),
        )

    def test_buffs_followers_in_draw_pile_only(self) -> None:
        deck = make_deck_instance_state((
            _card(None, 1001, CardType.FOLLOWER, atk=2, life=2),
            _card(None, 1002, CardType.SPELL, atk=0, life=0),
            _card(None, 1003, CardType.FOLLOWER, atk=3, life=3),
        ))
        result = self._run(_snapshot(deck))

        by_base = {inst.card.base_card_id: inst.card for inst in result.snapshot.me.deck_instances.instances}
        self.assertEqual((by_base[1001].atk, by_base[1001].life, by_base[1001].max_life), (3, 3, 3))
        self.assertEqual((by_base[1003].atk, by_base[1003].life, by_base[1003].max_life), (4, 4, 4))
        # 法术不受 card_type=follower 过滤影响
        self.assertEqual((by_base[1002].atk, by_base[1002].life), (0, 0))

    def test_skips_instances_already_left_the_deck(self) -> None:
        deck = make_deck_instance_state((
            _card(7, 1001, CardType.FOLLOWER, atk=2, life=2),
            _card(None, 1003, CardType.FOLLOWER, atk=3, life=3),
        )).move(1, "hand", uid=7)
        result = self._run(_snapshot(deck))

        by_base = {inst.card.base_card_id: inst.card for inst in result.snapshot.me.deck_instances.instances}
        # 已在手上的 1001 不该被 buff
        self.assertEqual((by_base[1001].atk, by_base[1001].life), (2, 2))
        self.assertEqual((by_base[1003].atk, by_base[1003].life), (4, 4))

    def test_buff_carries_into_hand_when_drawn(self) -> None:
        deck = make_deck_instance_state((
            _card(None, 1001, CardType.FOLLOWER, atk=2, life=2),
        ))
        buffed = self._run(_snapshot(deck)).snapshot
        drawn, after = draw_next_from_deck_instances(buffed, "me")

        self.assertIsNotNone(drawn)
        self.assertEqual((drawn.atk, drawn.life, drawn.max_life), (3, 3, 3))
        # draw_next_from_deck_instances 只负责把实例从 draw_pile 移到 hand，
        # deck_count 由调用方（game_step_executor / local_self_play）维护。
        self.assertEqual(after.me.deck_instances.draw_pile, ())

    def test_raises_when_deck_composition_missing(self) -> None:
        """replay 只给 deck_count，没有 deck_instances —— 必须显式失败。"""
        with self.assertRaises(EffectInterpreterError):
            self._run(_snapshot(None))

    def test_no_matching_cards_is_a_noop(self) -> None:
        deck = make_deck_instance_state((
            _card(None, 1002, CardType.SPELL, atk=0, life=0),
        ))
        result = self._run(_snapshot(deck))
        # 没有匹配的随从时不应改动任何实例
        self.assertEqual(
            [(inst.card.base_card_id, inst.card.atk) for inst in result.snapshot.me.deck_instances.instances],
            [(1002, 0)],
        )

    def test_attack_floor_is_zero(self) -> None:
        deck = make_deck_instance_state((
            _card(None, 1001, CardType.FOLLOWER, atk=0, life=2),
        ))
        debuff = replace(DECK_OP, params={"kind": "stat_buff", "attack": -2, "life": 0,
                                          "filters": [{"field": "card_type", "operator": "==", "value": "follower"}]})
        result = self.interpreter.apply_operations(
            _snapshot(deck), (debuff,), EffectInterpretationContext(source_uid=1, player="me")
        )
        instance = result.snapshot.me.deck_instances.instances[0]
        self.assertEqual(instance.card.atk, 0)
        self.assertEqual(instance.card.life, 2)


if __name__ == "__main__":
    unittest.main()
