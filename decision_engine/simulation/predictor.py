# -*- coding: utf-8 -*-
"""预测器公共模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from decision_engine.state.snapshot import BattleSnapshot


class PredictionCertainty(Enum):
    """本地预测可靠度。"""

    EXACT = "exact"
    SPEC_ONLY = "spec_only"
    PARTIAL = "partial"
    APPROX = "approx"
    UNKNOWN = "unknown"
    PREDICT_ERROR = "predict_error"


@dataclass(frozen=True)
class PredictionResult:
    """一次动作预测结果。"""

    snapshot: BattleSnapshot
    certainty: PredictionCertainty
    description: str = ""
    assumptions: tuple[str, ...] = ()
    unresolved_effects: tuple[str, ...] = ()
