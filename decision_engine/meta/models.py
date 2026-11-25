# -*- coding: utf-8 -*-
"""数据模型：环境分析的结构化输出。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CardSlot:
    cid: str
    count: int
    name: str
    cost: int
    type_name: str = ""
    card_set_id: Optional[int] = None


@dataclass
class DeckEntry:
    class_en: str
    win_streak: int
    player_name: str
    article_date: str
    comment: str
    format: str  # "standard" / "unlimited"
    deck_type: str
    cards: list[CardSlot] = field(default_factory=list)


@dataclass
class CardFreq:
    cid: str
    name: str
    cost: int
    frequency: int
    total_decks: int
    avg_count: float

    @property
    def rate(self) -> float:
        return self.frequency / self.total_decks if self.total_decks else 0.0


@dataclass
class ClassReport:
    class_en: str
    class_cn: str
    deck_count: int
    top_streak: int
    deck_types: dict[str, int] = field(default_factory=dict)
    core_cards: list[CardFreq] = field(default_factory=list)
    top_decks: list[DeckEntry] = field(default_factory=list)


@dataclass
class FormatReport:
    deck_count: int
    classes: list[ClassReport] = field(default_factory=list)
    cross_class_cards: list[CardFreq] = field(default_factory=list)


@dataclass
class MetaReport:
    generated_at: str
    data_range: tuple[str, str]
    total_decks: int
    standard: FormatReport = field(default_factory=lambda: FormatReport(deck_count=0))
    unlimited: FormatReport = field(default_factory=lambda: FormatReport(deck_count=0))


CLASS_MAP: dict[str, dict] = {
    "Elf": {"cn": "精灵", "id": 1},
    "Royal": {"cn": "皇家护卫", "id": 2},
    "Witch": {"cn": "巫师", "id": 3},
    "Dragon": {"cn": "龙族", "id": 4},
    "Nightmare": {"cn": "梦魇", "id": 5},
    "Bishop": {"cn": "主教", "id": 6},
    "Nemesis": {"cn": "超越者", "id": 7},
}


def class_cn(class_en: str) -> str:
    return CLASS_MAP.get(class_en, {}).get("cn", class_en)
