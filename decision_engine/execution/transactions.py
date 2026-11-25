# -*- coding: utf-8 -*-
"""可回滚事务日志。

事务以不可变 BattleSnapshot 为输入并返回新 BattleSnapshot；rollback 按反向事务恢复。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from decision_engine.state.snapshot import BattleSnapshot


class TransactionApplyError(RuntimeError):
    pass


class SnapshotTransaction(Protocol):
    name: str

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        ...

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> "SnapshotTransaction":
        ...


@dataclass(frozen=True)
class AppliedTransaction:
    forward: SnapshotTransaction
    backward: SnapshotTransaction


@dataclass(frozen=True)
class TransactionGroup:
    name: str
    transactions: tuple[SnapshotTransaction, ...]

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        log = RollbackTransactionLog()
        try:
            return log.apply_all(snapshot, self.transactions)
        except Exception as exc:
            try:
                log.rollback()
            except Exception as rollback_exc:
                raise TransactionApplyError(f"{self.name} failed and rollback failed: {rollback_exc}") from exc
            raise

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> "TransactionGroup":
        group_log = RollbackTransactionLog()
        group_log.apply_all(before, self.transactions)
        return TransactionGroup(
            name=f"rollback:{self.name}",
            transactions=tuple(applied.backward for applied in reversed(group_log.applied)),
        )


@dataclass
class RollbackTransactionLog:
    root: BattleSnapshot | None = None
    current: BattleSnapshot | None = None
    applied: list[AppliedTransaction] = field(default_factory=list)

    def apply(self, snapshot: BattleSnapshot, transaction: SnapshotTransaction) -> BattleSnapshot:
        if self.root is None:
            self.root = snapshot
            self.current = snapshot
        elif self.current is not snapshot:
            raise TransactionApplyError("transaction log can only continue from its current snapshot")

        before = snapshot
        after = transaction.apply(before)
        backward = transaction.inverse(before, after)
        self.applied.append(AppliedTransaction(forward=transaction, backward=backward))
        self.current = after
        return after

    def apply_all(self, snapshot: BattleSnapshot, transactions: tuple[SnapshotTransaction, ...] | list[SnapshotTransaction]) -> BattleSnapshot:
        current = snapshot
        for transaction in transactions:
            current = self.apply(current, transaction)
        return current

    def rollback(self) -> BattleSnapshot:
        if self.current is None:
            raise TransactionApplyError("cannot rollback an empty transaction log")
        current = self.current
        while self.applied:
            applied = self.applied.pop()
            current = applied.backward.apply(current)
        self.current = current
        return current

    def clear(self) -> None:
        self.root = None
        self.current = None
        self.applied.clear()
