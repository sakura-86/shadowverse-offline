# -*- coding: utf-8 -*-
"""卡牌 master data 轻量视图。

该层只提供查询结构，不解析/执行卡牌效果。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FaithAcquiredAbilityText:
    """FaithAcquiredAbilityText 的轻量视图。"""

    related_base_card_id: int
    text_key: str = ""
    index: int = 0
    raw_text: str = ""
    clean_text: str = ""


@dataclass(frozen=True)
class BaseCardInfo:
    """BaseCardMaster 的轻量视图。"""

    base_card_id: int
    name: str = ""
    type_id: int = 0
    card_type: str = ""
    craft_class: str = ""
    class_id: int = 0
    cost: int = 0
    atk: int = 0
    life: int = 0
    rarity: str = ""
    tribes: tuple[int, ...] = ()
    related_base_card_ids: tuple[int, ...] = ()
    faith_related_base_card_ids: tuple[int, ...] = ()
    faith_acquired_ability_texts: tuple[FaithAcquiredAbilityText, ...] = ()
    skill_desc: str = ""
    evolve_desc: str = ""
    super_evolve_desc: str = ""


@dataclass(frozen=True)
class CardInfo:
    """CardMaster 的轻量视图。"""

    card_id: int
    base_card_id: int
    style_id: int = 0
    text_id: int = 0


@dataclass(frozen=True)
class SkillInfo:
    """SkillMaster 的轻量视图。"""

    skill_id: int
    base_card_id: int
    battle_log_type: str = ""
    side_log_type: str = ""
    option: str = ""


def _parse_faith_acquired_ability_texts(items: list[dict] | tuple[dict, ...]) -> tuple[FaithAcquiredAbilityText, ...]:
    parsed = []
    for item in items:
        if not isinstance(item, dict):
            continue
        parsed.append(
            FaithAcquiredAbilityText(
                related_base_card_id=int(item.get("related_base_card_id", 0)),
                text_key=str(item.get("text_key", "")),
                index=int(item.get("index", 0)),
                raw_text=str(item.get("raw_text", "")),
                clean_text=str(item.get("clean_text", "")),
            )
        )
    return tuple(parsed)


class MasterDataProvider:
    """Master data 查询接口。

    第一版 skeleton 使用内存字典；后续可替换为 JSON/MessagePack/MasterMemory 数据源。
    """

    def __init__(
        self,
        base_cards: dict[int, BaseCardInfo] | None = None,
        cards: dict[int, CardInfo] | None = None,
        skills_by_base_card: dict[int, tuple[SkillInfo, ...]] | None = None,
    ):
        self._base_cards = base_cards or {}
        self._cards = cards or {}
        self._skills_by_base_card = skills_by_base_card or {}

    def get_base(self, base_card_id: int) -> BaseCardInfo | None:
        return self._base_cards.get(base_card_id)

    def get_base_by_card_id(self, card_id: int) -> BaseCardInfo | None:
        card = self._cards.get(card_id)
        if card is None:
            return self._base_cards.get(card_id)
        return self._base_cards.get(card.base_card_id)

    def get_card(self, card_id: int) -> CardInfo | None:
        return self._cards.get(card_id)

    def get_skills(self, base_card_id: int) -> tuple[SkillInfo, ...]:
        return self._skills_by_base_card.get(base_card_id, ())

    def iter_base_cards(self) -> tuple[tuple[int, BaseCardInfo], ...]:
        """按 base_card_id 升序遍历全部基础卡（供卡牌检索 UI 使用）。"""
        return tuple(sorted(self._base_cards.items()))

    @classmethod
    def from_json(cls, path: str | Path) -> "MasterDataProvider":
        """从 card_database.json 加载 master data。

        兼容两种格式：
        - 旧格式：dict with "cards" key, entries have type_name/class_name/cards sub-entries
        - card_database.json 格式：flat list, entries have type/class, card_id 直接在顶层
        """
        database_path = Path(path)
        if not database_path.exists():
            return cls()

        with database_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        card_entries = raw.get("cards", raw) if isinstance(raw, dict) else raw
        base_cards: dict[int, BaseCardInfo] = {}
        cards: dict[int, CardInfo] = {}
        skills_by_base_card: dict[int, tuple[SkillInfo, ...]] = {}

        for entry in card_entries:
            base_card_id = int(entry.get("base_card_id", 0))
            if base_card_id == 0:
                continue

            base_cards[base_card_id] = BaseCardInfo(
                base_card_id=base_card_id,
                name=entry.get("name", ""),
                type_id=int(entry.get("type_id", 0)),
                card_type=entry.get("type_name", "") or entry.get("type", ""),
                craft_class=entry.get("class_name", "") or entry.get("class", ""),
                class_id=int(entry.get("class_id", 0) or 0),
                cost=int(entry.get("cost", 0)),
                atk=int(entry.get("atk", 0)),
                life=int(entry.get("life", 0)),
                rarity=entry.get("rarity", ""),
                tribes=tuple(int(tribe) if isinstance(tribe, (int, float)) else tribe for tribe in entry.get("tribes", [])),
                related_base_card_ids=tuple(int(cid) for cid in entry.get("related_base_card_ids", entry.get("related_cards", []))),
                faith_related_base_card_ids=tuple(int(cid) for cid in entry.get("faith_related_base_card_ids", entry.get("faith_related_cards", []))),
                faith_acquired_ability_texts=_parse_faith_acquired_ability_texts(entry.get("faith_acquired_ability_texts", [])),
                skill_desc=entry.get("clean_skill_desc", "") or entry.get("skill_desc_clean", "") or entry.get("skill_desc", ""),
                evolve_desc=entry.get("evolve_desc", ""),
                super_evolve_desc=entry.get("super_evolve_desc", ""),
            )

            # card_database.json: card_id 在顶层
            top_card_id = int(entry.get("card_id") or 0)
            if top_card_id:
                cards[top_card_id] = CardInfo(
                    card_id=top_card_id,
                    base_card_id=base_card_id,
                )
            evolved_card_id = int(entry.get("evolved_card_id") or 0)
            if evolved_card_id:
                cards[evolved_card_id] = CardInfo(
                    card_id=evolved_card_id,
                    base_card_id=base_card_id,
                )

            # 旧格式: cards 子条目
            for card_entry in entry.get("cards", []):
                card_id = int(card_entry.get("card_id", 0))
                if card_id == 0:
                    continue
                cards[card_id] = CardInfo(
                    card_id=card_id,
                    base_card_id=int(card_entry.get("base_card_id", base_card_id)),
                    style_id=int(card_entry.get("card_resource_id", 0)),
                    text_id=int(card_entry.get("card_text_id", 0)),
                )

            skills_by_base_card[base_card_id] = tuple(
                SkillInfo(
                    skill_id=int(skill.get("skill_id", 0)),
                    base_card_id=base_card_id,
                    battle_log_type=str(skill.get("battle_log_type", skill.get("type", ""))),
                    side_log_type=str(skill.get("side_log_type", "")),
                    option=skill.get("option", ""),
                )
                for skill in entry.get("skills", [])
            )

        return cls(base_cards=base_cards, cards=cards, skills_by_base_card=skills_by_base_card)
