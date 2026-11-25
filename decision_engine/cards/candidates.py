# -*- coding: utf-8 -*-
"""候选 CardPatch 数据模型。

候选 patch 来自自动文本模板提取，只用于审查与验证，不作为默认可执行补丁加载。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from decision_engine.cards.patches import EffectKind, EffectTiming, TargetHint


class CandidateConfidence(Enum):
    """候选效果置信度。"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class CandidateEffectPatch:
    """自动提取的候选效果补丁。"""

    timing: EffectTiming
    kind: EffectKind
    target: TargetHint = TargetHint.UNKNOWN
    amount: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    confidence: CandidateConfidence = CandidateConfidence.LOW
    source_text: str = ""
    source_key: str = ""
    parser_rule: str = ""
    note: str = ""


@dataclass(frozen=True)
class CandidateCardPatch:
    """一张 base card 的候选补丁集合。"""

    base_card_id: int
    name: str = ""
    effects: tuple[CandidateEffectPatch, ...] = ()
    tags: tuple[str, ...] = ()
    note: str = ""

    def effects_at(self, timing: EffectTiming) -> tuple[CandidateEffectPatch, ...]:
        return tuple(effect for effect in self.effects if effect.timing == timing)
