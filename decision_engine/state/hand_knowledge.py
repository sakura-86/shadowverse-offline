# -*- coding: utf-8 -*-
"""手牌可见性与隐藏信息知识状态。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping


HandOwner = Literal["me", "opponent"]
HandVisibility = Literal["known", "unknown"]
HandOrigin = Literal[
    "deck_draw",
    "generated_by_effect",
    "copied_from_opponent_hand",
    "opponent_hidden_hand",
    "replay_bound",
]


@dataclass(frozen=True)
class HandCardKnowledge:
    uid: int
    owner: HandOwner
    visibility: HandVisibility
    origin: HandOrigin
    source_uid: int | None = None
    source_effect_id: str | None = None
    base_card_id: int | None = None
    card_id: int | None = None
    playable: bool = True
    usable_as_material: bool = True
    replay_bound_uid: int | None = None

    @property
    def is_known(self) -> bool:
        return self.visibility == "known"

    @property
    def is_unknown(self) -> bool:
        return self.visibility == "unknown"


@dataclass(frozen=True)
class HandKnowledgeState:
    cards: Mapping[int, HandCardKnowledge]

    @classmethod
    def empty(cls) -> "HandKnowledgeState":
        return cls(cards={})

    def get(self, uid: int) -> HandCardKnowledge | None:
        return self.cards.get(uid)

    def with_card(self, knowledge: HandCardKnowledge) -> "HandKnowledgeState":
        cards = dict(self.cards)
        cards[knowledge.uid] = knowledge
        return HandKnowledgeState(cards=cards)

    def without_card(self, uid: int) -> "HandKnowledgeState":
        if uid not in self.cards:
            return self
        cards = dict(self.cards)
        del cards[uid]
        return HandKnowledgeState(cards=cards)

    def bind_replay_uid(self, uid: int, replay_bound_uid: int, *, card_id: int | None = None, base_card_id: int | None = None) -> "HandKnowledgeState":
        knowledge = self.cards.get(uid)
        if knowledge is None:
            return self
        return self.with_card(replace(
            knowledge,
            origin="replay_bound",
            replay_bound_uid=replay_bound_uid,
            card_id=card_id if card_id is not None else knowledge.card_id,
            base_card_id=base_card_id if base_card_id is not None else knowledge.base_card_id,
        ))


def unknown_opponent_hand_copy(
    *,
    uid: int,
    owner: HandOwner,
    source_uid: int | None = None,
    source_effect_id: str | None = None,
) -> HandCardKnowledge:
    return HandCardKnowledge(
        uid=uid,
        owner=owner,
        visibility="unknown",
        origin="copied_from_opponent_hand",
        source_uid=source_uid,
        source_effect_id=source_effect_id,
        playable=False,
        usable_as_material=True,
    )


def known_generated_hand_card(
    *,
    uid: int,
    owner: HandOwner,
    base_card_id: int,
    card_id: int | None = None,
    source_uid: int | None = None,
    source_effect_id: str | None = None,
) -> HandCardKnowledge:
    return HandCardKnowledge(
        uid=uid,
        owner=owner,
        visibility="known",
        origin="generated_by_effect",
        source_uid=source_uid,
        source_effect_id=source_effect_id,
        base_card_id=base_card_id,
        card_id=card_id,
        playable=True,
        usable_as_material=True,
    )
