# -*- coding: utf-8 -*-
"""Precise deck instance lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from decision_engine.state.snapshot import CardView


@dataclass(frozen=True)
class DeckInstance:
    instance_id: int
    card: CardView
    owner: str = "me"
    zone: str = "deck"
    uid: int | None = None
    origin: str = "deck"
    order_hint: int | None = None
    visibility: str = "hidden"


@dataclass(frozen=True)
class DeckInstanceState:
    instances: tuple[DeckInstance, ...] = ()
    draw_pile: tuple[int, ...] = ()
    hand: tuple[int, ...] = ()
    field: tuple[int, ...] = ()
    graveyard: tuple[int, ...] = ()
    banished: tuple[int, ...] = ()
    source: str = ""
    status: str = "available"
    warnings: tuple[str, ...] = ()

    def by_instance_id(self) -> dict[int, DeckInstance]:
        return {instance.instance_id: instance for instance in self.instances}

    def by_uid(self) -> dict[int, DeckInstance]:
        return {instance.uid: instance for instance in self.instances if instance.uid is not None}

    def move(self, instance_id: int, zone: str, *, uid: int | None = None, card: CardView | None = None, visibility: str | None = None, draw_index: int | None = None) -> "DeckInstanceState":
        by_id = self.by_instance_id()
        if instance_id not in by_id:
            raise KeyError(f"deck instance not found: {instance_id}")
        current = by_id[instance_id]
        next_uid = current.uid if uid is None else uid
        next_card = current.card if card is None else card
        if uid is not None and next_card.uid != uid:
            next_card = replace(next_card, uid=uid)
        next_visibility = current.visibility if visibility is None else visibility
        instances = tuple(
            replace(current, zone=zone, uid=next_uid, card=next_card, visibility=next_visibility) if item.instance_id == instance_id else item
            for item in self.instances
        )
        zones = {
            "draw_pile": list(self.draw_pile),
            "hand": list(self.hand),
            "field": list(self.field),
            "graveyard": list(self.graveyard),
            "banished": list(self.banished),
        }
        for values in zones.values():
            if instance_id in values:
                values.remove(instance_id)
        target = "draw_pile" if zone == "deck" else zone
        if target not in zones:
            raise ValueError(f"unsupported deck instance zone: {zone}")
        if draw_index is None:
            zones[target].append(instance_id)
        else:
            zones[target].insert(draw_index, instance_id)
        return replace(
            self,
            instances=instances,
            draw_pile=tuple(zones["draw_pile"]),
            hand=tuple(zones["hand"]),
            field=tuple(zones["field"]),
            graveyard=tuple(zones["graveyard"]),
            banished=tuple(zones["banished"]),
        )

    def update_card(self, uid: int, card: CardView) -> "DeckInstanceState":
        instances = tuple(
            replace(instance, card=card, uid=card.uid) if instance.uid == uid else instance
            for instance in self.instances
        )
        return replace(self, instances=instances)


def make_deck_instance_state(cards: Iterable[CardView], *, owner: str = "me", source: str = "") -> DeckInstanceState:
    instances = tuple(
        DeckInstance(
            instance_id=index + 1,
            card=card,
            owner=owner,
            zone="deck",
            uid=None,
            origin="deck",
            order_hint=index,
            visibility="hidden",
        )
        for index, card in enumerate(cards)
    )
    return DeckInstanceState(
        instances=instances,
        draw_pile=tuple(instance.instance_id for instance in instances),
        source=source,
        status="available",
    )
