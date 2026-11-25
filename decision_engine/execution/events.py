# -*- coding: utf-8 -*-
"""Phase L5 事件队列。

事件队列只负责时点顺序与分发，不解释具体卡牌效果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from decision_engine.state.snapshot import BattleSnapshot

from .transactions import RollbackTransactionLog, SnapshotTransaction


class EventKind(Enum):
    CONSUME_PP = "consume_pp"
    CONSUME_EP = "consume_ep"
    CONSUME_SEP = "consume_sep"
    PLAY_CARD = "play_card"
    PLAY_SPELL = "play_spell"
    ENHANCE = "enhance"
    ACCELERATE = "accelerate"
    CRYSTAL = "crystal"
    ENTER_FIELD = "enter_field"
    FANFARE = "fanfare"
    SUMMON = "summon"
    SUMMON_FAILED_FIELD_FULL = "summon_failed_field_full"
    DRAW = "draw"
    DRAW_HIDDEN = "draw_hidden"
    OVERDRAW = "overdraw"
    DECKOUT = "deckout"
    MOVE_TO_GRAVEYARD = "move_to_graveyard"
    DESTROY = "destroy"
    BANISH = "banish"
    BOUNCE = "bounce"
    RETURN_TO_DECK = "return_to_deck"
    FLOOD = "flood"
    TRANSFORM = "transform"
    MANUAL_EVOLVE = "manual_evolve"
    MANUAL_SUPER_EVOLVE = "manual_super_evolve"
    BECOMES_EVOLVED = "becomes_evolved"
    BECOMES_SUPER_EVOLVED = "becomes_super_evolved"
    ACTIVATE = "activate"
    SPECIAL_ACTION = "special_action"
    ATTACK_LEADER = "attack_leader"
    ATTACK_FOLLOWER = "attack_follower"
    CLASH = "clash"
    DAMAGE = "damage"
    HEAL = "heal"
    TEMP_SHIELD = "temp_shield"
    BANE = "bane"
    DESTROYED_BY_DAMAGE = "destroyed_by_damage"
    LAST_WORDS = "last_words"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    REFILL_PP = "refill_pp"
    COUNTDOWN = "countdown"
    SKYBOUND_ART = "skybound_art"
    NECROMANCY = "necromancy"
    EARTH_RITE = "earth_rite"
    FUSION = "fusion"
    REPLACE_HAND_CARD = "replace_hand_card"
    COPY_CARD = "copy_card"
    DISCARD = "discard"
    REANIMATE = "reanimate"
    KEYWORD_REMOVED = "keyword_removed"
    # 「使其失去所有能力」：被指向的卡在此之后不再拥有任何能力（含谢幕曲）。
    # 必须作为事件发出：同一 effect 内「先失去所有能力、再造成致死伤害」时，
    # 谢幕曲不应发动，消费端只能靠事件时序判断，不能回看 step 前的快照。
    ABILITY_REMOVED = "ability_removed"
    EXTRA_PP = "extra_pp"
    CONSUME_EXTRA_PP = "consume_extra_pp"
    SPELLBOOST = "spellboost"
    SOURCE_BUFFED = "on_source_buffed"
    TURN_START_FIELD_LAST_WORDS = "turn_start_field_last_words"
    TURN_START_CREST_LAST_WORDS = "turn_start_crest_last_words"
    TURN_START_DECK = "turn_start_deck"
    TURN_END_DECK = "turn_end_deck"
    TURN_START_FIELD_EFFECTS = "turn_start_field_effects"
    TURN_START_CREST_EFFECTS = "turn_start_crest_effects"


@dataclass(frozen=True)
class ExecutionEvent:
    kind: EventKind | str
    source_uid: int | None = None
    target_uid: int | None = None
    player: str | None = None
    payload: dict[str, object] = field(default_factory=dict)

    @property
    def name(self) -> str:
        if isinstance(self.kind, EventKind):
            return self.kind.value
        return str(self.kind)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.name == other
        return super().__eq__(other)


@dataclass(frozen=True)
class EventHandlerResult:
    snapshot: BattleSnapshot
    transactions: tuple[SnapshotTransaction, ...] = ()
    emitted_events: tuple[ExecutionEvent, ...] = ()
    priority_emitted_events: tuple[ExecutionEvent, ...] = ()


class EventHandler(Protocol):
    def handle(self, snapshot: BattleSnapshot, event: ExecutionEvent) -> EventHandlerResult:
        ...


class EventQueueError(RuntimeError):
    pass


@dataclass(frozen=True)
class EventDispatchResult:
    snapshot: BattleSnapshot
    transaction_log: RollbackTransactionLog
    processed_events: tuple[ExecutionEvent, ...]


class EventQueue:
    def __init__(self, events: tuple[ExecutionEvent, ...] | list[ExecutionEvent] = ()):
        self._events = list(events)

    def push(self, event: ExecutionEvent) -> None:
        self._events.append(event)

    def extend(self, events: tuple[ExecutionEvent, ...] | list[ExecutionEvent]) -> None:
        self._events.extend(events)

    def prepend(self, events: tuple[ExecutionEvent, ...] | list[ExecutionEvent]) -> None:
        self._events[0:0] = list(events)

    def pop(self) -> ExecutionEvent:
        if not self._events:
            raise EventQueueError("cannot pop from empty event queue")
        return self._events.pop(0)

    def __len__(self) -> int:
        return len(self._events)

    def dispatch(self, snapshot: BattleSnapshot, handlers: tuple[EventHandler, ...] | list[EventHandler]) -> EventDispatchResult:
        current = snapshot
        log = RollbackTransactionLog(root=snapshot, current=snapshot)
        processed: list[ExecutionEvent] = []
        try:
            while self._events:
                event = self.pop()
                processed.append(event)
                for handler in handlers:
                    result = handler.handle(current, event)
                    if result.transactions:
                        current = log.apply_all(current, result.transactions)
                    else:
                        current = result.snapshot
                    if result.priority_emitted_events:
                        self.prepend(result.priority_emitted_events)
                    self.extend(result.emitted_events)
        except Exception as exc:
            try:
                log.rollback()
            except Exception as rollback_exc:
                raise EventQueueError(f"event dispatch failed and rollback failed: {rollback_exc}") from exc
            raise
        # Diagnostic: check zone/cards_by_uid consistency
        from .diagnostic_layer import check_zone_cards_consistency
        issues = check_zone_cards_consistency(current)
        if issues:
            import logging
            _diag_logger = logging.getLogger(__name__)
            for issue in issues:
                _diag_logger.warning(f"zone/cards_by_uid inconsistency after event dispatch: {issue}")
        return EventDispatchResult(snapshot=current, transaction_log=log, processed_events=tuple(processed))


def event_names(events: tuple[ExecutionEvent, ...] | list[ExecutionEvent]) -> tuple[str, ...]:
    return tuple(event.name for event in events)
