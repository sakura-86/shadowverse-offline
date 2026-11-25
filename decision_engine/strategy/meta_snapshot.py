# -*- coding: utf-8 -*-
"""环境快照：离线预计算的体系数据。

MetaSnapshot 从 yokidou_decks.db 预计算，合并人工标注的体系定义。
运行时只需加载生成的 JSON，不依赖 sqlite。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from decision_engine.strategy.match_context import ArchetypeProfile


@dataclass(frozen=True)
class MetaSnapshot:
    """环境快照。"""

    archetypes: dict[str, ArchetypeProfile] = field(default_factory=dict)
    class_archetypes: dict[int, tuple[ArchetypeProfile, ...]] = field(default_factory=dict)

    def get_archetypes_for_class(self, class_id: int) -> tuple[ArchetypeProfile, ...]:
        return self.class_archetypes.get(class_id, ())

    def get_archetype(self, archetype_id: str) -> ArchetypeProfile | None:
        return self.archetypes.get(archetype_id)


class MetaSnapshotLoader:
    """加载预计算的 meta_snapshot.json。"""

    @classmethod
    def from_json(cls, path: str | Path) -> MetaSnapshot:
        p = Path(path)
        if not p.exists():
            return MetaSnapshot()
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls._parse(raw)

    @classmethod
    def _parse(cls, raw: dict) -> MetaSnapshot:
        archetypes: dict[str, ArchetypeProfile] = {}
        class_archetypes: dict[int, list[ArchetypeProfile]] = {}

        for entry in raw.get("archetypes", []):
            profile = ArchetypeProfile(
                archetype_id=entry["archetype_id"],
                class_id=int(entry["class_id"]),
                display_name=entry.get("display_name", ""),
                effective_resources=frozenset(entry.get("effective_resources", [])),
                resource_thresholds=entry.get("resource_thresholds", {}),
                key_card_ids=frozenset(int(cid) for cid in entry.get("key_card_ids", [])),
                card_roles={
                    role: frozenset(int(cid) for cid in card_ids)
                    for role, card_ids in entry.get("card_roles", {}).items()
                },
                win_condition=entry.get("win_condition", ""),
                aggro_level=float(entry.get("aggro_level", 0.5)),
                estimated_kill_turn=int(entry.get("estimated_kill_turn", 8)),
            )
            archetypes[profile.archetype_id] = profile
            class_archetypes.setdefault(profile.class_id, []).append(profile)

        return MetaSnapshot(
            archetypes=archetypes,
            class_archetypes={k: tuple(v) for k, v in class_archetypes.items()},
        )
