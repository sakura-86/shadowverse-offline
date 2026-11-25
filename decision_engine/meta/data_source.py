# -*- coding: utf-8 -*-
"""数据源层：读取 yokidou_decks.db 和 sv_cards_cn.db。"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Optional

from .models import CardSlot, DeckEntry

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "meta"

_WIN_STREAK_RE = re.compile(r"(\d+)\s*[连連]\s*[胜勝]")
_DECK_TYPE_RE = re.compile(r"连胜\s+(.+)$")


def _detect_format(comment: str) -> str:
    if "无限" in comment or "無限" in comment:
        return "unlimited"
    return "standard"


def _extract_deck_type(comment: str) -> str:
    m = _DECK_TYPE_RE.search(comment)
    if m:
        return m.group(1).strip()
    parts = comment.split()
    return parts[-1] if parts else "?"


class MetaDataSource:
    def __init__(
        self,
        deck_db_path: Optional[str] = None,
        card_db_path: Optional[str] = None,
    ):
        self._deck_db = deck_db_path or str(_DEFAULT_DATA_DIR / "yokidou_decks.db")
        self._card_db = card_db_path or str(_DEFAULT_DATA_DIR / "sv_cards_cn.db")

        if not os.path.exists(self._deck_db):
            raise FileNotFoundError(
                f"卡组数据库不存在: {self._deck_db}\n"
                f"请通过 scp 从服务器拉取到 decision_engine/data/meta/"
            )

    def load_all_decks(self) -> list[DeckEntry]:
        conn = sqlite3.connect(self._deck_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM yokidou_decks ORDER BY class_en, win_streak DESC"
        ).fetchall()
        conn.close()

        decks: list[DeckEntry] = []
        for r in rows:
            comment = r["comment"] or ""
            cards = self._parse_deck_json(r["deck_json"])
            decks.append(DeckEntry(
                class_en=r["class_en"] or "",
                win_streak=r["win_streak"] or 0,
                player_name=r["player_name"] or "",
                article_date=r["article_date"] or "",
                comment=comment,
                format=_detect_format(comment),
                deck_type=_extract_deck_type(comment),
                cards=cards,
            ))
        return decks

    def get_latest_set_id(self) -> Optional[int]:
        if not os.path.exists(self._card_db):
            return None
        conn = sqlite3.connect(self._card_db)
        row = conn.execute(
            "SELECT MAX(card_set_id) FROM sv_cards WHERE card_set_id BETWEEN 10000 AND 19999"
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def get_set_name_map(self) -> dict[int, str]:
        if not os.path.exists(self._card_db):
            return {}
        conn = sqlite3.connect(self._card_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT set_id, name FROM sv_card_sets").fetchall()
        conn.close()
        return {r["set_id"]: r["name"] for r in rows}

    @staticmethod
    def _parse_deck_json(raw: Optional[str]) -> list[CardSlot]:
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if isinstance(payload, list):
            return [
                CardSlot(
                    cid=str(c.get("cid", "")),
                    count=int(c.get("count", 1)),
                    name=c.get("name", ""),
                    cost=int(c.get("cost", 0) or 0),
                    type_name=str(c.get("type", "")),
                    card_set_id=c.get("card_set_id"),
                )
                for c in payload
                if c.get("cid")
            ]
        elif isinstance(payload, dict):
            return [
                CardSlot(cid=cid, count=int(cnt), name="", cost=0)
                for cid, cnt in payload.items()
            ]
        return []
