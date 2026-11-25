# -*- coding: utf-8 -*-
"""CardPatch JSON 加载器。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from decision_engine.cards.effect_facts import (
    CardEffectFact,
    EffectFactKind,
    EffectFactTiming,
    EffectTargeting,
    EffectValueSource,
)
from decision_engine.cards.patches import CardPatch, EffectKind, EffectPatch, EffectTiming, TargetHint
from decision_engine.cards.resolver import CardResolver


def load_card_patches(path: str | Path) -> dict[int, CardPatch]:
    """从 JSON 文件加载 CardPatch 字典。"""
    patch_path = Path(path)
    if not patch_path.exists():
        return {}

    with patch_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw.get("patches", raw) if isinstance(raw, dict) else raw
    patches: dict[int, CardPatch] = {}
    for item in items:
        patch = _parse_card_patch(item)
        patches[patch.base_card_id] = patch
    return patches


def resolver_from_patch_file(path: str | Path, base: CardResolver | None = None) -> CardResolver:
    """基于补丁文件创建 CardResolver。"""
    patches = load_card_patches(path)
    if base is None:
        return CardResolver(patches=patches)
    return base.with_patches(patches)


def _parse_card_patch(data: dict[str, Any]) -> CardPatch:
    effects = tuple(_parse_effect(effect) for effect in data.get("effects", []))
    effect_facts = tuple(_parse_effect_fact(fact) for fact in data.get("effect_facts", []))
    return CardPatch(
        base_card_id=int(data["base_card_id"]),
        name=data.get("name", ""),
        effects=effects,
        effect_facts=effect_facts,
        ignored_auto_fact_raw_texts=tuple(data.get("ignored_auto_fact_raw_texts", [])),
        tags=tuple(data.get("tags", [])),
        note=data.get("note", ""),
    )


def _parse_effect(data: dict[str, Any]) -> EffectPatch:
    return EffectPatch(
        timing=EffectTiming(data.get("timing", EffectTiming.SPELL.value)),
        kind=EffectKind(data.get("kind", EffectKind.UNKNOWN.value)),
        target=TargetHint(data.get("target", TargetHint.UNKNOWN.value)),
        amount=data.get("amount"),
        params=dict(data.get("params", {})),
        note=data.get("note", ""),
    )


def _parse_effect_fact(data: dict[str, Any]) -> CardEffectFact:
    return CardEffectFact(
        timing=EffectFactTiming(data.get("timing", EffectFactTiming.UNKNOWN.value)),
        kind=EffectFactKind(data.get("kind", EffectFactKind.UNKNOWN_COMPLEX.value)),
        amount=data.get("amount"),
        count=data.get("count"),
        target_side=data.get("target_side", ""),
        target_zone=data.get("target_zone", ""),
        target_type=data.get("target_type", ""),
        targeting=EffectTargeting(data.get("targeting", EffectTargeting.NONE.value)),
        value_source=EffectValueSource(data.get("value_source", EffectValueSource.NONE.value)),
        conditions=frozenset(data.get("conditions", [])),
        safe_to_apply=bool(data.get("safe_to_apply", False)),
        requires_oracle=bool(data.get("requires_oracle", True)),
        raw_text=data.get("raw_text", ""),
        unsupported_reason=data.get("unsupported_reason", ""),
    )
