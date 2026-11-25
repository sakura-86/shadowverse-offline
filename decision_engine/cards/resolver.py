# -*- coding: utf-8 -*-
"""卡牌知识查询入口。

CardResolver 是新架构获取卡牌知识的唯一入口。它不执行效果，
只返回 master data、结构化 patch 和目标提示等信息。
"""

from __future__ import annotations

from dataclasses import dataclass

from decision_engine.cards.effect_facts import CardEffectFact, EffectFactTiming, extract_effect_facts
from decision_engine.cards.master_data import BaseCardInfo, CardInfo, FaithAcquiredAbilityText, MasterDataProvider, SkillInfo
from decision_engine.cards.patches import CardPatch, EffectPatch, EffectTiming, TargetHint
from decision_engine.cards.tactical_tags import TacticalTags, extract_tactical_tags


@dataclass(frozen=True)
class CrestSummary:
    """纹章战术摘要。"""

    card_id: int
    name: str = ""
    tags: TacticalTags = TacticalTags()
    known: bool = False
    faith_related_base_card_ids: tuple[int, ...] = ()
    faith_acquired_ability_texts: tuple[FaithAcquiredAbilityText, ...] = ()


@dataclass(frozen=True)
class BoonSummary:
    """恩惠战术摘要。"""

    card_id: int
    name: str = ""
    tags: TacticalTags = TacticalTags()
    known: bool = False


def _knowledge_text_for_tags(base: BaseCardInfo) -> str:
    parts = [
        base.skill_desc,
        base.evolve_desc,
        base.super_evolve_desc,
        *(item.clean_text for item in base.faith_acquired_ability_texts),
    ]
    return "\n".join(part for part in parts if part)


def _default_effect_timing(base: BaseCardInfo) -> EffectFactTiming:
    if base.type_id == 4 or base.card_type in {"spell", "spell_evolved"}:
        return EffectFactTiming.ON_PLAY
    return EffectFactTiming.UNKNOWN


def _dedupe_effect_facts(facts: tuple[CardEffectFact, ...]) -> tuple[CardEffectFact, ...]:
    result: list[CardEffectFact] = []
    seen: set[tuple] = set()
    for fact in facts:
        key = (
            fact.timing,
            fact.kind,
            fact.amount,
            fact.count,
            fact.target_side,
            fact.target_zone,
            fact.target_type,
            fact.targeting,
            fact.value_source,
            fact.conditions,
            fact.raw_text,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return tuple(result)


class CardResolver:
    """集中式卡牌知识查询器。"""

    def __init__(
        self,
        master_data: MasterDataProvider | None = None,
        patches: dict[int, CardPatch] | None = None,
    ):
        self.master_data = master_data or MasterDataProvider()
        self._patches = patches or {}

    def get_base(self, base_card_id: int) -> BaseCardInfo | None:
        return self.master_data.get_base(base_card_id)

    def get_base_by_card_id(self, card_id: int) -> BaseCardInfo | None:
        return self.master_data.get_base_by_card_id(card_id)

    def get_tactical_tags(self, card_id: int) -> TacticalTags:
        base = self.get_base_by_card_id(card_id)
        if base is None:
            return TacticalTags()
        return extract_tactical_tags(_knowledge_text_for_tags(base))

    def get_effect_facts(self, card_id: int) -> tuple[CardEffectFact, ...]:
        base = self.get_base_by_card_id(card_id)
        if base is None:
            return ()
        facts = extract_effect_facts(
            _knowledge_text_for_tags(base),
            default_timing=_default_effect_timing(base),
        )
        patch = self.get_patch(base.base_card_id)
        if patch is None:
            return facts
        ignored = set(patch.ignored_auto_fact_raw_texts)
        if ignored:
            facts = tuple(fact for fact in facts if fact.raw_text not in ignored)
        return _dedupe_effect_facts((*facts, *patch.effect_facts))

    def summarize_crest(self, card_id: int) -> CrestSummary:
        base = self.get_base_by_card_id(card_id)
        if base is None:
            return CrestSummary(card_id=card_id)
        return CrestSummary(
            card_id=card_id,
            name=base.name,
            tags=extract_tactical_tags(_knowledge_text_for_tags(base)),
            known=True,
            faith_related_base_card_ids=base.faith_related_base_card_ids,
            faith_acquired_ability_texts=base.faith_acquired_ability_texts,
        )

    def summarize_crests(self, card_ids: tuple[int, ...] | list[int]) -> tuple[CrestSummary, ...]:
        return tuple(self.summarize_crest(cid) for cid in card_ids)

    def summarize_boon(self, card_id: int) -> BoonSummary:
        base = self.get_base_by_card_id(card_id)
        if base is None:
            return BoonSummary(card_id=card_id)
        return BoonSummary(
            card_id=card_id,
            name=base.name,
            tags=extract_tactical_tags(_knowledge_text_for_tags(base)),
            known=True,
        )

    def summarize_boons(self, card_ids: tuple[int, ...] | list[int]) -> tuple[BoonSummary, ...]:
        return tuple(self.summarize_boon(cid) for cid in card_ids)

    def get_card(self, card_id: int) -> CardInfo | None:
        return self.master_data.get_card(card_id)

    def get_skills(self, base_card_id: int) -> tuple[SkillInfo, ...]:
        return self.master_data.get_skills(base_card_id)

    def get_patch(self, base_card_id: int) -> CardPatch | None:
        return self._patches.get(base_card_id)

    def get_effects(self, base_card_id: int, timing: EffectTiming | None = None) -> tuple[EffectPatch, ...]:
        patch = self.get_patch(base_card_id)
        if patch is None:
            return ()
        if timing is None:
            return patch.effects
        return patch.effects_at(timing)

    def get_target_hints(self, base_card_id: int, timing: EffectTiming | None = None) -> tuple[TargetHint, ...]:
        hints = []
        for effect in self.get_effects(base_card_id, timing):
            if effect.target not in hints:
                hints.append(effect.target)
        return tuple(hints)

    def with_patch(self, patch: CardPatch) -> CardResolver:
        """返回附加指定补丁的新 resolver。"""
        return self.with_patches({patch.base_card_id: patch})

    def with_patches(self, patches_to_add: dict[int, CardPatch]) -> CardResolver:
        """返回附加多个补丁的新 resolver。"""
        patches = dict(self._patches)
        patches.update(patches_to_add)
        return CardResolver(self.master_data, patches)
