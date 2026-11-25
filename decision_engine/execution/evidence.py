# -*- coding: utf-8 -*-
"""证据·护符（tribe 21）的判定。

「证据」是证据·护符组成的资源：`me.field_evidence_count`（战场）与
`me.hand_evidence_amulet_count`（手牌）两个变量、以及 11073310 的
「使自己的战场上的证据-N」都按「每张护符 = 1 点」计数。

帧里的实例 `CardView` 常常不带 `tribes`（协议只下发 `card_id`），因此判定必须
回退到 `card_database.json` 的 base 数据，否则场上有证据护符也数成 0。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decision_engine.state.snapshot import CardView

EVIDENCE_TRIBE = 21


@lru_cache(maxsize=1)
def base_card_tribes() -> dict[int, tuple[int, ...]]:
    """base_card_id -> tribes（卡牌数据库，不受实例字段缺失影响）。"""
    path = Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(card["base_card_id"]): tuple(card.get("tribes") or ())
        for card in payload.get("cards", [])
    }


def is_evidence_card(card: "CardView") -> bool:
    if EVIDENCE_TRIBE in tuple(card.tribes or ()):
        return True
    return EVIDENCE_TRIBE in base_card_tribes().get(int(card.base_card_id), ())
