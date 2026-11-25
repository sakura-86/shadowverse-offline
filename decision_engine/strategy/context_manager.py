# -*- coding: utf-8 -*-
"""MatchContext 生命周期管理。

负责：
- 开局：从我方牌组 JSON + MetaSnapshot 构建 DeckProfile
- 开局：从对方 class_id 初始化 OpponentProfile
- 每帧：从 BattleSnapshot 提取对手新出牌 → 更新推断
- 输出：当前 MatchContext 供 Evaluator 消费
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from decision_engine.state.snapshot import BattleSnapshot
from decision_engine.strategy.match_context import (
    ArchetypeProfile,
    DeckProfile,
    MatchContext,
    OpponentProfile,
)
from decision_engine.strategy.meta_snapshot import MetaSnapshot, MetaSnapshotLoader
from decision_engine.strategy.opponent_identifier import OpponentIdentifier


class MatchContextManager:
    """MatchContext 的构建和实时更新。"""

    def __init__(
        self,
        meta: MetaSnapshot,
        my_profile: DeckProfile,
        opponent_class_id: int,
    ):
        self._meta = meta
        self._identifier = OpponentIdentifier(meta)
        self._my_profile = my_profile
        self._opponent = OpponentProfile(class_id=opponent_class_id)
        self._identifier.update_opponent(self._opponent)
        self._turn = 0
        self._seen_opponent_uids: set[int] = set()

    @property
    def context(self) -> MatchContext:
        return MatchContext(
            my_profile=self._my_profile,
            opponent_profile=self._opponent,
            turn=self._turn,
        )

    def update(self, snapshot: BattleSnapshot) -> None:
        """从 BattleSnapshot 更新对手观察和回合数。"""
        self._turn = max(snapshot.me.turn, snapshot.opponent.turn)

        new_observations = False
        for uid in snapshot.opponent.field:
            if uid in self._seen_opponent_uids:
                continue
            self._seen_opponent_uids.add(uid)
            card = snapshot.cards_by_uid.get(uid)
            if card and card.card_id:
                self._opponent.observe(card.card_id)
                new_observations = True

        if new_observations:
            self._identifier.update_opponent(self._opponent)

    @classmethod
    def from_deck_json(
        cls,
        deck_path: str | Path,
        opponent_class_id: int,
        meta_path: str | Path = "decision_engine/data/public/meta_snapshot.json",
    ) -> "MatchContextManager":
        """从牌组 JSON 文件 + meta_snapshot 构建 manager。"""
        meta = MetaSnapshotLoader.from_json(meta_path)
        deck_data = _load_deck_json(Path(deck_path))

        class_id = deck_data["class_id"]
        archetype_id = deck_data.get("archetype_id", "")
        deck_card_counts = {
            int(c["card_id"]): int(c.get("count", 1))
            for c in deck_data.get("cards", [])
        }
        deck_card_ids = frozenset(deck_card_counts)

        archetype = meta.get_archetype(archetype_id)
        if archetype is None:
            archetype = _infer_archetype(meta, class_id, deck_card_ids)

        my_profile = DeckProfile(
            class_id=class_id,
            archetype=archetype,
            deck_card_ids=deck_card_ids,
            deck_card_counts=deck_card_counts,
        )

        return cls(meta=meta, my_profile=my_profile, opponent_class_id=opponent_class_id)


def _load_deck_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_archetype(
    meta: MetaSnapshot,
    class_id: int,
    deck_card_ids: frozenset[int],
) -> ArchetypeProfile:
    """从 MetaSnapshot 中找最匹配的 archetype。"""
    candidates = meta.get_archetypes_for_class(class_id)
    if not candidates:
        return ArchetypeProfile(archetype_id="unknown", class_id=class_id)

    best: ArchetypeProfile | None = None
    best_score = -1.0
    for arch in candidates:
        if not arch.key_card_ids:
            continue
        overlap = len(deck_card_ids & arch.key_card_ids)
        score = overlap / len(arch.key_card_ids)
        if score > best_score:
            best_score = score
            best = arch

    return best if best is not None else candidates[0]
