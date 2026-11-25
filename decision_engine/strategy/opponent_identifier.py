# -*- coding: utf-8 -*-
"""对手体系识别。

根据对方职业 + 已观察到的出牌，与 MetaSnapshot 中各 archetype 的
key_card_ids 做交集比例匹配，输出概率排序的体系候选。
"""

from __future__ import annotations

from decision_engine.strategy.match_context import ArchetypeProfile, OpponentProfile
from decision_engine.strategy.meta_snapshot import MetaSnapshot


class OpponentIdentifier:
    """实时对手体系识别器。"""

    def __init__(self, meta: MetaSnapshot):
        self._meta = meta

    def identify(
        self,
        class_id: int,
        observed_card_ids: list[int],
    ) -> list[tuple[ArchetypeProfile, float]]:
        """返回按概率降序排列的 archetype 候选。

        匹配逻辑：
        - 若无观察数据，按环境频率返回该职业所有体系（均分概率）
        - 有观察数据时，用 observed 与 key_card_ids 交集比例作为匹配分数
        """
        candidates = self._meta.get_archetypes_for_class(class_id)
        if not candidates:
            return []

        if not observed_card_ids:
            prob = 1.0 / len(candidates)
            return [(arch, prob) for arch in candidates]

        observed_set = frozenset(observed_card_ids)
        scored: list[tuple[ArchetypeProfile, float]] = []

        for arch in candidates:
            if not arch.key_card_ids:
                scored.append((arch, 0.0))
                continue
            overlap = len(observed_set & arch.key_card_ids)
            score = overlap / len(arch.key_card_ids)
            scored.append((arch, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        total = sum(s for _, s in scored)
        if total > 0:
            scored = [(arch, s / total) for arch, s in scored]
        else:
            prob = 1.0 / len(scored)
            scored = [(arch, prob) for arch, s in scored]

        return scored

    def update_opponent(
        self,
        opponent: OpponentProfile,
    ) -> None:
        """根据当前观察更新对手的 likely_archetypes。"""
        opponent.likely_archetypes = self.identify(
            opponent.class_id,
            opponent.observed_card_ids,
        )
