# -*- coding: utf-8 -*-
"""候选 CardPatch JSON 加载与统计。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_engine.cards.candidates import CandidateCardPatch, CandidateConfidence, CandidateEffectPatch
from decision_engine.cards.patches import EffectKind, EffectTiming, TargetHint


@dataclass(frozen=True)
class CandidatePatchStats:
    """候选 patch 汇总统计。"""

    patch_count: int
    effect_count: int
    confidence: dict[str, int]
    kinds: dict[str, int]
    timings: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_count": self.patch_count,
            "effect_count": self.effect_count,
            "confidence": self.confidence,
            "kinds": self.kinds,
            "timings": self.timings,
        }


def load_candidate_patches(path: str | Path) -> dict[int, CandidateCardPatch]:
    """从 JSON 文件加载候选 CardPatch 字典。"""
    candidate_path = Path(path)
    if not candidate_path.exists():
        return {}

    with candidate_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw.get("patches", raw) if isinstance(raw, dict) else raw
    patches: dict[int, CandidateCardPatch] = {}
    for item in items:
        patch = _parse_candidate_card_patch(item)
        patches[patch.base_card_id] = patch
    return patches


def summarize_candidate_patches(patches: dict[int, CandidateCardPatch]) -> CandidatePatchStats:
    """统计候选 patch 的置信度、效果类别和时机分布。"""
    confidence: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    timings: Counter[str] = Counter()
    effect_count = 0

    for patch in patches.values():
        for effect in patch.effects:
            effect_count += 1
            confidence[effect.confidence.value] += 1
            kinds[effect.kind.value] += 1
            timings[effect.timing.value] += 1

    return CandidatePatchStats(
        patch_count=len(patches),
        effect_count=effect_count,
        confidence={level.value: confidence.get(level.value, 0) for level in CandidateConfidence},
        kinds=dict(sorted(kinds.items())),
        timings=dict(sorted(timings.items())),
    )


def validate_candidate_file(path: str | Path) -> CandidatePatchStats:
    """加载并校验候选文件，返回统计结果。"""
    patches = load_candidate_patches(path)
    return summarize_candidate_patches(patches)


def _parse_candidate_card_patch(data: dict[str, Any]) -> CandidateCardPatch:
    base_card_id = int(data["base_card_id"])
    if base_card_id == 0:
        raise ValueError("candidate patch must not target base_card_id=0")

    effects = tuple(_parse_candidate_effect(effect) for effect in data.get("effects", []))
    return CandidateCardPatch(
        base_card_id=base_card_id,
        name=data.get("name", ""),
        effects=effects,
        tags=tuple(data.get("tags", [])),
        note=data.get("note", ""),
    )


def _parse_candidate_effect(data: dict[str, Any]) -> CandidateEffectPatch:
    missing = [key for key in ("confidence", "source_text", "source_key", "parser_rule") if not data.get(key)]
    if missing:
        raise ValueError(f"candidate effect missing required fields: {', '.join(missing)}")

    return CandidateEffectPatch(
        timing=EffectTiming(data.get("timing", EffectTiming.SPELL.value)),
        kind=EffectKind(data.get("kind", EffectKind.UNKNOWN.value)),
        target=TargetHint(data.get("target", TargetHint.UNKNOWN.value)),
        amount=data.get("amount"),
        params=dict(data.get("params", {})),
        confidence=CandidateConfidence(data["confidence"]),
        source_text=data["source_text"],
        source_key=data["source_key"],
        parser_rule=data["parser_rule"],
        note=data.get("note", ""),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验并统计候选 CardPatch 文件")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("decision_engine/data/public/card_patch_candidates.json"),
        help="候选 patch JSON 路径",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出统计")
    return parser.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = _parse_args()
    stats = validate_candidate_file(args.path)
    if args.json:
        print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"candidate_patches={stats.patch_count} effects={stats.effect_count}")
        print("confidence=" + ", ".join(f"{key}:{value}" for key, value in stats.confidence.items()))
        print("kinds=" + ", ".join(f"{key}:{value}" for key, value in stats.kinds.items()))
        print("timings=" + ", ".join(f"{key}:{value}" for key, value in stats.timings.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
