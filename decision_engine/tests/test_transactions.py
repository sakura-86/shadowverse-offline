# -*- coding: utf-8 -*-
"""Phase L3 可回滚事务测试。"""

import unittest
from dataclasses import replace

from decision_engine.execution import (
    AddCard,
    InsertZoneUid,
    ModifyCardCounter,
    ModifyPlayerCounter,
    MoveZoneUid,
    RemoveZoneUidAt,
    RollbackTransactionLog,
    SetCardField,
    SetPlayerField,
    TransactionGroup,
)
from decision_engine.state.snapshot import BattleSnapshot, CardView, PlayerView


class BrokenTransaction:
    name = "broken"

    def apply(self, snapshot):
        raise RuntimeError("boom")

    def inverse(self, before, after):
        raise AssertionError("inverse should not be called")


class TransactionTest(unittest.TestCase):
    def make_snapshot(self):
        me = PlayerView(life=20, max_life=20, pp=3, max_pp=5, ep=1, sep=2, hand=(1, 2), field=(3,))
        opponent = PlayerView(life=18, max_life=20, pp=4, max_pp=5, field=(4,))
        cards = {
            1: CardView(uid=1, card_id=101, base_card_id=101, name="手牌1", cost=2),
            2: CardView(uid=2, card_id=102, base_card_id=102, name="手牌2", cost=3),
            3: CardView(uid=3, card_id=103, base_card_id=103, name="场上1", atk=2, life=3, max_life=3),
            4: CardView(uid=4, card_id=104, base_card_id=104, name="敌方1", atk=4, life=5, max_life=5),
        }
        return BattleSnapshot(me=me, opponent=opponent, cards_by_uid=cards, source="test", is_ally_turn=True)

    def test_player_and_card_counter_rollback(self):
        snapshot = self.make_snapshot()
        log = RollbackTransactionLog()

        current = log.apply_all(
            snapshot,
            (
                ModifyPlayerCounter("me", "life", -3),
                ModifyPlayerCounter("me", "pp", -2),
                ModifyCardCounter(3, "life", -1),
            ),
        )

        self.assertEqual(current.me.life, 17)
        self.assertEqual(current.me.pp, 1)
        self.assertEqual(current.cards_by_uid[3].life, 2)
        self.assertEqual(log.rollback(), snapshot)

    def test_set_fields_and_zone_operations_rollback(self):
        snapshot = self.make_snapshot()
        log = RollbackTransactionLog()

        current = log.apply_all(
            snapshot,
            (
                SetPlayerField("opponent", "life", 10),
                SetCardField(3, "has_storm", True),
                MoveZoneUid("me", "hand", 0, "me", "field"),
            ),
        )

        self.assertEqual(current.opponent.life, 10)
        self.assertTrue(current.cards_by_uid[3].has_storm)
        self.assertEqual(current.me.hand, (2,))
        self.assertEqual(current.me.field, (3, 1))
        self.assertEqual(log.rollback(), snapshot)

    def test_add_card_and_insert_remove_rollback(self):
        snapshot = self.make_snapshot()
        new_card = CardView(uid=9, card_id=109, base_card_id=109, name="新卡")
        log = RollbackTransactionLog()

        current = log.apply_all(
            snapshot,
            (
                AddCard(new_card),
                InsertZoneUid("me", "hand", 9, 1),
                RemoveZoneUidAt("me", "field", 0, 3),
            ),
        )

        self.assertIn(9, current.cards_by_uid)
        self.assertEqual(current.me.hand, (1, 9, 2))
        self.assertEqual(current.me.field, ())
        self.assertEqual(log.rollback(), snapshot)

    def test_transaction_group_rolls_back_on_failure(self):
        snapshot = self.make_snapshot()
        group = TransactionGroup(
            name="failing_group",
            transactions=(
                ModifyPlayerCounter("me", "life", -5),
                BrokenTransaction(),
            ),
        )

        with self.assertRaises(RuntimeError):
            group.apply(snapshot)

        self.assertEqual(snapshot, self.make_snapshot())

    def test_transaction_log_requires_current_snapshot(self):
        snapshot = self.make_snapshot()
        log = RollbackTransactionLog()
        current = log.apply(snapshot, ModifyPlayerCounter("me", "life", -1))
        other = replace(current, source="other")

        with self.assertRaises(Exception):
            log.apply(other, ModifyPlayerCounter("me", "life", -1))


if __name__ == "__main__":
    unittest.main()
