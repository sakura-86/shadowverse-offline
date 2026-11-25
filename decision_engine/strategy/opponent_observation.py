# -*- coding: utf-8 -*-
"""Opponent observed-card extraction for deck identification."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from decision_engine.state.snapshot import BattleSnapshot, CardIdCount


@dataclass(frozen=True)
class OpponentObservedCards:
    card_ids: tuple[int, ...]
    sources: dict[int, tuple[str, ...]]
    warnings: tuple[str, ...] = ()


def extract_opponent_observed_cards(snapshot: BattleSnapshot) -> OpponentObservedCards:
    sources: dict[int, set[str]] = defaultdict(set)
    warnings: list[str] = []
    _add_visible_uids(snapshot, snapshot.opponent.field, "field", sources, warnings)
    _add_visible_uids(snapshot, snapshot.opponent.hand, "visible_hand", sources, warnings)
    _add_counted_cards(snapshot.opponent.played_card_ids, "played", sources)
    _add_counted_cards(snapshot.opponent.destroyed_card_ids, "destroyed", sources)
    return OpponentObservedCards(
        card_ids=tuple(sorted(sources)),
        sources={card_id: tuple(sorted(items)) for card_id, items in sorted(sources.items())},
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _add_visible_uids(
    snapshot: BattleSnapshot,
    uids: tuple[int, ...],
    source: str,
    sources: dict[int, set[str]],
    warnings: list[str],
) -> None:
    for uid in uids:
        card = snapshot.cards_by_uid.get(uid)
        if card is None:
            warnings.append(f"opponent_observed_uid_missing:{source}:{uid}")
            continue
        if card.card_id <= 0:
            warnings.append(f"opponent_observed_card_id_missing:{source}:{uid}")
            continue
        sources[card.card_id].add(source)


def _add_counted_cards(items: tuple[CardIdCount, ...], source: str, sources: dict[int, set[str]]) -> None:
    for item in items:
        if item.card_id <= 0 or item.count <= 0:
            continue
        sources[item.card_id].add(source)
