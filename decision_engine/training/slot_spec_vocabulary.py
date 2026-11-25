# -*- coding: utf-8 -*-
"""slot spec 词表：把 ``target_group_specs`` 的字符串标识编码成 embedding index。

``LegalAction.target_group_specs`` 的取值有两类：

- **effect-spec 路径**：``spec_id``（如 ``11064110_auto_001``），来自
  ``card_effect_specs.json``；每个 ``selected_*`` spec 是一个可能的槽语义。
- **非 effect-spec 路径**：固定来源标识（``attack_follower`` / ``mode_skill`` 等），
  由 ``LocalActionEnumerator._with_group_specs`` 写入。

词表**append-only**：已有 index 不可重排，否则旧 checkpoint 的
``slot_spec_embedding`` 行会整体错位。新增 spec 只能追加。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

SLOT_SPEC_VOCABULARY_SCHEMA = "SlotSpecVocabulary"
SLOT_SPEC_VOCABULARY_VERSION = 1
PAD_INDEX = 0
UNKNOWN_INDEX = 1

#: 非 effect-spec 路径的来源标识，必须与 ``LocalActionEnumerator._with_group_specs`` 的
#: 调用参数、以及 ``local_enumerator`` 攻击分支写入的值保持一致。
BUILTIN_SLOT_SPECS = (
    "attack_follower",
    "attack_leader",
    "protocol_select",
    "local_select",
    "mode_skill",
    "choice_skill",
)


@dataclass(frozen=True)
class SlotSpecVocabulary:
    """Stable ``spec_id -> index`` 映射（0=pad，1=unknown）。"""

    spec_to_index: Mapping[str, int]
    specs: tuple[str, ...] = ()
    source_effect_specs_digest: str = ""
    schema: str = SLOT_SPEC_VOCABULARY_SCHEMA
    version: int = SLOT_SPEC_VOCABULARY_VERSION

    def __post_init__(self) -> None:
        normalized = {str(spec): int(index) for spec, index in self.spec_to_index.items()}
        object.__setattr__(self, "spec_to_index", normalized)
        if not self.specs:
            ordered = sorted(normalized.items(), key=lambda item: item[1])
            object.__setattr__(self, "specs", tuple(spec for spec, _ in ordered))
        self.validate()

    @property
    def vocab_size(self) -> int:
        return max(self.spec_to_index.values(), default=UNKNOWN_INDEX) + 1

    @property
    def digest(self) -> str:
        payload = {
            "schema": self.schema,
            "version": self.version,
            "source_effect_specs_digest": self.source_effect_specs_digest,
            "spec_to_index": {spec: index for spec, index in sorted(self.spec_to_index.items())},
        }
        return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()

    def index_for_spec(self, spec_id: str | None) -> int:
        """空值 → PAD；未登录的 spec → UNKNOWN（不静默映射到某个已知 spec）。"""
        if not spec_id:
            return PAD_INDEX
        return self.spec_to_index.get(str(spec_id), UNKNOWN_INDEX)

    def spec_for_index(self, index: int) -> str | None:
        if index in (PAD_INDEX, UNKNOWN_INDEX):
            return None
        return self.specs[index] if 0 <= index < len(self.specs) else None

    def extend(self, spec_ids: list[str] | tuple[str, ...] | set[str]) -> "SlotSpecVocabulary":
        mapping = dict(self.spec_to_index)
        next_index = self.vocab_size
        for spec_id in sorted({str(value) for value in spec_ids if value}):
            if spec_id not in mapping:
                mapping[spec_id] = next_index
                next_index += 1
        return SlotSpecVocabulary(
            spec_to_index=mapping,
            source_effect_specs_digest=self.source_effect_specs_digest,
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "digest": self.digest,
            "source_effect_specs_digest": self.source_effect_specs_digest,
            "spec_to_index": {spec: index for spec, index in sorted(self.spec_to_index.items())},
        }

    @classmethod
    def from_jsonable(cls, data: Mapping[str, Any]) -> "SlotSpecVocabulary":
        raw = data.get("spec_to_index") or {}
        return cls(
            spec_to_index={str(spec): int(index) for spec, index in raw.items()},
            source_effect_specs_digest=str(data.get("source_effect_specs_digest") or ""),
            schema=str(data.get("schema") or SLOT_SPEC_VOCABULARY_SCHEMA),
            version=int(data.get("version") or SLOT_SPEC_VOCABULARY_VERSION),
        )

    def validate(self) -> None:
        if self.schema != SLOT_SPEC_VOCABULARY_SCHEMA:
            raise ValueError(f"slot spec 词表 schema 不匹配: {self.schema!r}")
        indices = sorted(self.spec_to_index.values())
        # 0=pad、1=unknown 保留，真实 spec 从 2 开始且必须连续
        expected = list(range(UNKNOWN_INDEX + 1, UNKNOWN_INDEX + 1 + len(indices)))
        if indices != expected:
            raise ValueError(f"slot spec 词表 index 必须是 {expected[:1]}..{expected[-1:]} 的连续整数，实际 {indices[:3]}...")
        if len(set(self.spec_to_index)) != len(self.spec_to_index):
            raise ValueError("slot spec 词表存在重复 spec_id")


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _effect_specs_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "public" / "card_effect_specs.json"


def _collect_effect_spec_ids(path: Path) -> tuple[tuple[str, ...], str]:
    """收集**会产生目标槽**的 spec_id。

    两类来源（与 ``LocalActionEnumerator`` 的槽生成路径一一对应）：

    1. spec 顶层 ``target.kind`` 以 ``selected_`` 开头且 ``count`` 非 None；
    2. spec 顶层 ``target`` 为 None，但 ``operations[].target_ref`` 是
       ``selected_*``（implicit selector，由 ``_synthesize_target_from_operations`` 合成，
       槽语义仍用 spec 自己的 spec_id）。

    ``count=None`` 的 ``selected_*`` 是内部引用、不产生槽，排除。
    """
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    rows = payload["cards"] if isinstance(payload, dict) else payload
    spec_ids: set[str] = set()
    for bundle in rows:
        for spec in bundle.get("specs") or []:
            spec_id = spec.get("spec_id")
            if not spec_id:
                continue
            target = spec.get("target")
            if target is not None:
                if not str(target.get("kind") or "").startswith("selected_"):
                    continue
                if target.get("count") is None:
                    continue
                spec_ids.add(str(spec_id))
                continue
            refs = [str(operation.get("target_ref") or "") for operation in (spec.get("operations") or [])]
            if any(ref.startswith("selected_") for ref in refs):
                spec_ids.add(str(spec_id))
    digest = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
    return tuple(sorted(spec_ids)), digest


@lru_cache(maxsize=1)
def default_slot_spec_vocabulary() -> SlotSpecVocabulary:
    """从 ``card_effect_specs.json`` 构建默认词表（内置标识优先，保证位置稳定）。"""
    effect_spec_ids, digest = _collect_effect_spec_ids(_effect_specs_path())
    mapping: dict[str, int] = {}
    next_index = UNKNOWN_INDEX + 1
    for spec_id in BUILTIN_SLOT_SPECS:
        mapping[spec_id] = next_index
        next_index += 1
    for spec_id in effect_spec_ids:
        if spec_id not in mapping:
            mapping[spec_id] = next_index
            next_index += 1
    return SlotSpecVocabulary(spec_to_index=mapping, source_effect_specs_digest=digest)
