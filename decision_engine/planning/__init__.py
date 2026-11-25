# -*- coding: utf-8 -*-
"""新架构规划层。

本包只导出**未被冻结**的规划基础设施。PUCT 搜索、proof search 与已证伪的
value target 见 :mod:`decision_engine.planning.frozen`：它们在 tests / tools 中
仍可 import（保留回归与诊断证据），但 ``decision_engine`` 主路径不得依赖。

当前唯一有数据支撑的验收锚点见 :func:`decision_engine.planning.turn_planner.plan_turn`
（严格配对镜像 12:0 完胜 PUCT(128 sims)）。
"""

from .result import PlanCandidate
from .search_prior import PriorResult, SearchPrior, StructuredNeuralPrior, UniformSearchPrior
from .search_value import (
    BoardAdvantageValue,
    NeuralValueFunction,
    SearchValue,
    StructuredNeuralValueFunction,
    TerminalValue,
    ValueResult,
)
from .trace import ActionScoreTrace, BeamDepthTrace, SearchTrace

__all__ = [
    "ActionScoreTrace",
    "BeamDepthTrace",
    "BoardAdvantageValue",
    "NeuralValueFunction",
    "PlanCandidate",
    "PriorResult",
    "SearchPrior",
    "SearchTrace",
    "SearchValue",
    "StructuredNeuralPrior",
    "StructuredNeuralValueFunction",
    "TerminalValue",
    "UniformSearchPrior",
    "ValueResult",
]
