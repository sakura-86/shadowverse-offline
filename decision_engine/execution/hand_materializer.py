# -*- coding: utf-8 -*-
"""统一手牌知识物化辅助。"""

from __future__ import annotations

from dataclasses import replace

from decision_engine.state.hand_knowledge import known_generated_hand_card, unknown_opponent_hand_copy
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView

from .events import EventKind, ExecutionEvent
from .state_ops import AddCard, InsertZoneUid, SetHandKnowledge
from .transactions import SnapshotTransaction

UNKNOWN_OPPONENT_HAND_COPY_NAME = "UNKNOWN_OPPONENT_HAND_COPY"


def make_unknown_opponent_hand_copy_card(uid: int, *, cost: int = 0) -> CardView:
    return CardView(
        uid=uid,
        card_id=0,
        base_card_id=0,
        name=UNKNOWN_OPPONENT_HAND_COPY_NAME,
        cost=max(0, cost),
        card_type=CardType.SPELL,
        content_unknown=1,
    )


def copy_random_opponent_hand_to_self(
    snapshot: BattleSnapshot,
    *,
    player: str,
    copied_uid: int | None,
    placeholder_uid: int,
    source_uid: int | None,
    source_effect_id: str | None = None,
    cost_delta: int = 0,
) -> tuple[tuple[SnapshotTransaction, ...], tuple[ExecutionEvent, ...]]:
    opponent_view = snapshot.opponent if player == "me" else snapshot.me
    if copied_uid is not None and copied_uid not in opponent_view.hand:
        raise ValueError("copy_from_opponent_hand target must be in opponent hand")
    # Opponent hand identities may be hidden in local snapshots; absence of a visible slot is still a valid unknown copy.
    if placeholder_uid in snapshot.cards_by_uid:
        raise ValueError("copy_from_opponent_hand generated uid already exists")
    placeholder = make_unknown_opponent_hand_copy_card(placeholder_uid, cost=cost_delta)
    knowledge = unknown_opponent_hand_copy(
        uid=placeholder_uid,
        owner="me" if player == "me" else "opponent",
        source_uid=source_uid,
        source_effect_id=source_effect_id,
    )
    return (
        (
            AddCard(placeholder),
            InsertZoneUid(player, "hand", placeholder.uid),
            SetHandKnowledge(knowledge),
        ),
        (ExecutionEvent(EventKind.COPY_CARD, source_uid=copied_uid, target_uid=placeholder.uid, player=player),),
    )


def mark_known_generated_hand_card(
    card: CardView,
    *,
    player: str,
    source_uid: int | None,
    source_effect_id: str | None = None,
) -> SetHandKnowledge:
    return SetHandKnowledge(known_generated_hand_card(
        uid=card.uid,
        owner="me" if player == "me" else "opponent",
        base_card_id=card.base_card_id,
        card_id=card.card_id,
        source_uid=source_uid,
        source_effect_id=source_effect_id,
    ))


def with_replay_bound_identity(placeholder: CardView, actual: CardView) -> CardView:
    return replace(
        placeholder,
        card_id=actual.card_id,
        base_card_id=actual.base_card_id,
        name=actual.name,
        cost=actual.cost,
        atk=actual.atk,
        life=actual.life,
        max_life=actual.max_life,
        card_type=actual.card_type,
    )
