# -*- coding: utf-8 -*-
"""规划结果模型。"""

from __future__ import annotations

from dataclasses import dataclass

from decision_engine.actions.model import LegalAction
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot


@dataclass(frozen=True)
class PlanCandidate:
    """搜索得到的候选计划。"""

    actions: tuple[LegalAction, ...]
    snapshot: BattleSnapshot
    score: float
    certainty: PredictionCertainty
