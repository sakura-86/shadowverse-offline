# -*- coding: utf-8 -*-
"""Phase L5 事件队列测试。"""

import unittest

from decision_engine.execution import (
    EventHandlerResult,
    EventKind,
    EventQueue,
    ExecutionEvent,
    ModifyPlayerCounter,
    event_names,
)
from decision_engine.state.snapshot import BattleSnapshot, PlayerView


class DamageThenEmitHandler:
    def handle(self, snapshot, event):
        if event.kind == EventKind.DAMAGE:
            return EventHandlerResult(
                snapshot=snapshot,
                transactions=(ModifyPlayerCounter("opponent", "life", -int(event.payload["amount"])),),
                emitted_events=(ExecutionEvent(EventKind.LAST_WORDS, source_uid=event.target_uid),),
            )
        return EventHandlerResult(snapshot=snapshot)


class FailingHandler:
    def handle(self, snapshot, event):
        if event.kind == EventKind.LAST_WORDS:
            raise RuntimeError("handler failed")
        return EventHandlerResult(
            snapshot=snapshot,
            transactions=(ModifyPlayerCounter("opponent", "life", -1),),
            emitted_events=(ExecutionEvent(EventKind.LAST_WORDS),),
        )


class MultiStepEmitHandler:
    def handle(self, snapshot, event):
        if event.name == "play_card":
            return EventHandlerResult(snapshot=snapshot, emitted_events=(ExecutionEvent(EventKind.LAST_WORDS, source_uid=1), ExecutionEvent(EventKind.DAMAGE, source_uid=2)))
        if event.name == "last_words":
            return EventHandlerResult(snapshot=snapshot, emitted_events=(ExecutionEvent(EventKind.SUMMON, source_uid=3),))
        return EventHandlerResult(snapshot=snapshot)


class EventQueueTest(unittest.TestCase):
    def make_snapshot(self):
        return BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1),
            opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1),
            cards_by_uid={},
            source="test",
        )

    def test_fifo_pop_and_event_names(self):
        queue = EventQueue([
            ExecutionEvent(EventKind.PLAY_CARD),
            ExecutionEvent(EventKind.FANFARE),
        ])

        self.assertEqual(queue.pop(), "play_card")
        self.assertEqual(queue.pop(), "fanfare")
        self.assertEqual(len(queue), 0)

    def test_dispatch_processes_appended_events_in_fifo_order(self):
        snapshot = self.make_snapshot()
        queue = EventQueue([ExecutionEvent(EventKind.DAMAGE, source_uid=1, target_uid=2, payload={"amount": 3})])

        result = queue.dispatch(snapshot, [DamageThenEmitHandler()])

        self.assertEqual(result.snapshot.opponent.life, 17)
        self.assertEqual(event_names(result.processed_events), ("damage", "last_words"))
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_dispatch_processes_nested_appended_events_at_queue_tail(self):
        snapshot = self.make_snapshot()
        queue = EventQueue([ExecutionEvent(EventKind.PLAY_CARD)])

        result = queue.dispatch(snapshot, [MultiStepEmitHandler()])

        self.assertEqual(event_names(result.processed_events), ("play_card", "last_words", "damage", "summon"))
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_dispatch_failure_rolls_back_transactions(self):
        snapshot = self.make_snapshot()
        queue = EventQueue([ExecutionEvent(EventKind.DAMAGE, payload={"amount": 1})])

        with self.assertRaises(RuntimeError):
            queue.dispatch(snapshot, [FailingHandler()])

        self.assertEqual(snapshot.opponent.life, 20)


if __name__ == "__main__":
    unittest.main()
