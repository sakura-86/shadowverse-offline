# -*- coding: utf-8 -*-
"""Versioned append-only vocabulary for structured base-card embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CARD_VOCABULARY_SCHEMA = "BaseCardVocabulary"
CARD_VOCABULARY_VERSION = 1
PAD_INDEX = 0
UNKNOWN_INDEX = 1


@dataclass(frozen=True)
class CardVocabulary:
    """Stable base_card_id -> embedding-index mapping.

    Existing indices are immutable. Card-pool upgrades append new base ids only,
    so a checkpoint can reuse every previously trained embedding row.
    """

    base_card_id_to_index: Mapping[int, int]
    source_card_database_digest: str = ""
    schema: str = CARD_VOCABULARY_SCHEMA
    version: int = CARD_VOCABULARY_VERSION

    def __post_init__(self) -> None:
        normalized = {int(base_id): int(index) for base_id, index in self.base_card_id_to_index.items()}
        object.__setattr__(self, "base_card_id_to_index", normalized)
        self.validate()

    @property
    def vocab_size(self) -> int:
        return max(self.base_card_id_to_index.values(), default=UNKNOWN_INDEX) + 1

    @property
    def digest(self) -> str:
        payload = {
            "schema": self.schema,
            "version": self.version,
            "source_card_database_digest": self.source_card_database_digest,
            "base_card_id_to_index": {str(key): value for key, value in sorted(self.base_card_id_to_index.items())},
        }
        return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()

    def index_for_base_card_id(self, base_card_id: int | None) -> int:
        if base_card_id is None:
            return PAD_INDEX
        return self.base_card_id_to_index.get(int(base_card_id), UNKNOWN_INDEX)

    def extend(self, base_card_ids: list[int] | tuple[int, ...] | set[int]) -> "CardVocabulary":
        mapping = dict(self.base_card_id_to_index)
        next_index = self.vocab_size
        for base_card_id in sorted({int(value) for value in base_card_ids if int(value) > 0}):
            if base_card_id not in mapping:
                mapping[base_card_id] = next_index
                next_index += 1
        return CardVocabulary(mapping, source_card_database_digest=self.source_card_database_digest)

    def validate(self) -> None:
        if self.schema != CARD_VOCABULARY_SCHEMA:
            raise ValueError(f"card vocabulary schema must be {CARD_VOCABULARY_SCHEMA}")
        if self.version != CARD_VOCABULARY_VERSION:
            raise ValueError(f"card vocabulary version must be {CARD_VOCABULARY_VERSION}")
        indices = list(self.base_card_id_to_index.values())
        if any(base_id <= 0 for base_id in self.base_card_id_to_index):
            raise ValueError("card vocabulary base_card_id must be positive")
        if any(index < 2 for index in indices):
            raise ValueError("card vocabulary card indices must start at 2")
        if len(set(indices)) != len(indices):
            raise ValueError("card vocabulary indices must be unique")
        expected = list(range(2, self.vocab_size))
        if sorted(indices) != expected:
            raise ValueError("card vocabulary indices must be contiguous and append-only")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "pad_index": PAD_INDEX,
            "unknown_index": UNKNOWN_INDEX,
            "source_card_database_digest": self.source_card_database_digest,
            "base_card_id_to_index": {str(key): value for key, value in sorted(self.base_card_id_to_index.items())},
            "vocab_size": self.vocab_size,
            "digest": self.digest,
        }

    @classmethod
    def from_jsonable(cls, payload: Mapping[str, Any]) -> "CardVocabulary":
        vocabulary = cls(
            base_card_id_to_index={int(key): int(value) for key, value in _mapping(payload.get("base_card_id_to_index")).items()},
            source_card_database_digest=str(payload.get("source_card_database_digest") or ""),
            schema=str(payload.get("schema") or ""),
            version=int(payload.get("version") or 0),
        )
        declared_digest = str(payload.get("digest") or "")
        if declared_digest and declared_digest != vocabulary.digest:
            raise ValueError("card vocabulary digest mismatch")
        declared_size = payload.get("vocab_size")
        if declared_size is not None and int(declared_size) != vocabulary.vocab_size:
            raise ValueError("card vocabulary size mismatch")
        return vocabulary


def build_card_vocabulary(card_database_path: Path, *, previous: CardVocabulary | None = None) -> CardVocabulary:
    """Build an initial deterministic vocabulary or append a current card pool."""
    payload = json.loads(card_database_path.read_text(encoding="utf-8"))
    cards = payload.get("cards")
    if not isinstance(cards, list):
        raise ValueError("card database cards must be a list")
    base_ids = []
    for card in cards:
        if isinstance(card, Mapping) and card.get("base_card_id") is not None:
            base_ids.append(int(card["base_card_id"]))
    database_digest = "sha256:" + hashlib.sha256(card_database_path.read_bytes()).hexdigest()
    if previous is None:
        mapping = {base_id: index for index, base_id in enumerate(sorted(set(base_ids)), start=2)}
        return CardVocabulary(mapping, source_card_database_digest=database_digest)
    extended = previous.extend(base_ids)
    return CardVocabulary(dict(extended.base_card_id_to_index), source_card_database_digest=database_digest)


def build_legacy_card_vocabulary(card_database_path: Path) -> CardVocabulary:
    """Recreate the pre-versioned database-order base-card mapping for old checkpoints."""
    payload = json.loads(card_database_path.read_text(encoding="utf-8"))
    cards = payload.get("cards")
    if not isinstance(cards, list):
        raise ValueError("card database cards must be a list")
    mapping: dict[int, int] = {}
    for card in cards:
        if not isinstance(card, Mapping) or card.get("base_card_id") is None:
            continue
        base_card_id = int(card["base_card_id"])
        if base_card_id not in mapping:
            mapping[base_card_id] = len(mapping) + 2
    database_digest = "sha256:" + hashlib.sha256(card_database_path.read_bytes()).hexdigest()
    return CardVocabulary(mapping, source_card_database_digest=database_digest)


def save_card_vocabulary(path: Path, vocabulary: CardVocabulary) -> None:
    path.write_text(json.dumps(vocabulary.to_jsonable(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_card_vocabulary(path: Path) -> CardVocabulary:
    return CardVocabulary.from_jsonable(json.loads(path.read_text(encoding="utf-8")))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
