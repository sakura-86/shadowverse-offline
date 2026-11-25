# -*- coding: utf-8 -*-
"""分析引擎：从原始卡组数据提取结构化环境报告。"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

from .models import (
    CLASS_MAP,
    CardFreq,
    ClassReport,
    DeckEntry,
    FormatReport,
    MetaReport,
    class_cn,
)


def analyze(decks: list[DeckEntry]) -> MetaReport:
    if not decks:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        return MetaReport(
            generated_at=now, data_range=("", ""), total_decks=0
        )

    dates = [d.article_date for d in decks if d.article_date]
    data_range = (min(dates), max(dates)) if dates else ("", "")

    std_decks = [d for d in decks if d.format == "standard"]
    unl_decks = [d for d in decks if d.format == "unlimited"]

    return MetaReport(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        data_range=data_range,
        total_decks=len(decks),
        standard=_analyze_format(std_decks),
        unlimited=_analyze_format(unl_decks),
    )


def _analyze_format(decks: list[DeckEntry]) -> FormatReport:
    if not decks:
        return FormatReport(deck_count=0)

    cls_groups: dict[str, list[DeckEntry]] = defaultdict(list)
    for d in decks:
        cls_groups[d.class_en].append(d)

    classes = []
    for cls_en in sorted(cls_groups.keys(), key=lambda c: -len(cls_groups[c])):
        cls_decks = cls_groups[cls_en]
        classes.append(_analyze_class(cls_en, cls_decks))

    cross_cards = _find_cross_class_cards(cls_groups)

    return FormatReport(
        deck_count=len(decks),
        classes=classes,
        cross_class_cards=cross_cards,
    )


def _analyze_class(cls_en: str, decks: list[DeckEntry]) -> ClassReport:
    deck_types: Counter[str] = Counter()
    card_freq: Counter[str] = Counter()
    card_count: Counter[str] = Counter()
    card_names: dict[str, str] = {}
    card_costs: dict[str, int] = {}

    for d in decks:
        deck_types[d.deck_type] += 1
        seen_cids: set[str] = set()
        for c in d.cards:
            if c.cid not in seen_cids:
                card_freq[c.cid] += 1
                seen_cids.add(c.cid)
            card_count[c.cid] += c.count
            if c.cid not in card_names and c.name:
                card_names[c.cid] = c.name
                card_costs[c.cid] = c.cost

    total = len(decks)
    core_cards = [
        CardFreq(
            cid=cid,
            name=card_names.get(cid, f"?{cid}"),
            cost=card_costs.get(cid, 0),
            frequency=freq,
            total_decks=total,
            avg_count=card_count[cid] / freq,
        )
        for cid, freq in card_freq.most_common(40)
    ]

    top_streak = max((d.win_streak for d in decks), default=0)

    return ClassReport(
        class_en=cls_en,
        class_cn=class_cn(cls_en),
        deck_count=total,
        top_streak=top_streak,
        deck_types=dict(deck_types.most_common()),
        core_cards=core_cards,
        top_decks=sorted(decks, key=lambda d: -d.win_streak)[:3],
    )


def _find_cross_class_cards(
    cls_groups: dict[str, list[DeckEntry]],
) -> list[CardFreq]:
    card_class_freq: dict[str, Counter[str]] = defaultdict(Counter)
    card_names: dict[str, str] = {}
    card_costs: dict[str, int] = {}

    for cls_en, decks in cls_groups.items():
        seen_in_class: set[str] = set()
        for d in decks:
            for c in d.cards:
                if c.cid not in seen_in_class:
                    card_class_freq[c.cid][cls_en] += 1
                    seen_in_class.add(c.cid)
                if c.cid not in card_names and c.name:
                    card_names[c.cid] = c.name
                    card_costs[c.cid] = c.cost

    cross: list[CardFreq] = []
    total_all = sum(len(decks) for decks in cls_groups.values())
    for cid, cls_counter in card_class_freq.items():
        if len(cls_counter) >= 3:
            total_freq = sum(cls_counter.values())
            cross.append(CardFreq(
                cid=cid,
                name=card_names.get(cid, f"?{cid}"),
                cost=card_costs.get(cid, 0),
                frequency=total_freq,
                total_decks=total_all,
                avg_count=0.0,
            ))

    cross.sort(key=lambda c: -c.frequency)
    return cross[:20]
