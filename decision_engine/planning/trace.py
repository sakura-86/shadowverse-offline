# -*- coding: utf-8 -*-
"""搜索跟踪信息。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActionScoreTrace:
    """单个候选动作的评分变化摘要。"""

    depth: int
    action_repr: str
    action_kind: str
    source_uid: int
    baseline_score: float
    candidate_score: float
    score_delta: float
    baseline_detail: dict[str, float | str]
    candidate_detail: dict[str, float | str]
    detail_delta: dict[str, float]
    certainty: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class BeamDepthTrace:
    """单层 beam search 统计。"""

    depth: int
    beam_size: int
    actions_enumerated: int = 0
    predictions: int = 0
    unknown_skipped: int = 0
    duplicate_skipped: int = 0
    candidates: int = 0
    lethal_found: int = 0
    best_score: float | None = None
    top_actions: tuple[ActionScoreTrace, ...] = ()
    all_actions: tuple[ActionScoreTrace, ...] = ()


@dataclass
class SearchTrace:
    """一次搜索的跟踪信息。"""

    depths: list[BeamDepthTrace] = field(default_factory=list)
    terminal_reason: str = ""

    def add_depth(self, trace: BeamDepthTrace) -> None:
        self.depths.append(trace)
