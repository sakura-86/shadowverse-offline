# -*- coding: utf-8 -*-
"""Solution trace alignment scoring for puzzle search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from decision_engine.planning.trace_canonicalizer import CanonicalBlock, canonical_blocks
from decision_engine.planning.trace_footprint import ActionFootprint, footprint_distance, snapshot_distance
from decision_engine.state.snapshot import BattleSnapshot


@dataclass(frozen=True)
class CanonicalSolutionTrace:
    trace_id: str
    snapshots: tuple[BattleSnapshot, ...]
    footprints: tuple[ActionFootprint, ...]
    blocks: tuple[CanonicalBlock, ...]

    @property
    def final_snapshot(self) -> BattleSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "snapshot_count": len(self.snapshots),
            "footprint_count": len(self.footprints),
            "block_count": len(self.blocks),
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass(frozen=True)
class RouteAlignment:
    score: float = 0.0
    trace_id: str = ""
    matched_prefix: int = 0
    closest_step: int = 0
    state_distance: float | None = None
    block_bonus: float = 0.0
    state_bonus: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "trace_id": self.trace_id,
            "matched_prefix": self.matched_prefix,
            "closest_step": self.closest_step,
            "state_distance": self.state_distance,
            "block_bonus": self.block_bonus,
            "state_bonus": self.state_bonus,
        }


def make_solution_trace(trace_id: str, snapshots: tuple[BattleSnapshot, ...], footprints: tuple[ActionFootprint, ...]) -> CanonicalSolutionTrace:
    return CanonicalSolutionTrace(
        trace_id=trace_id,
        snapshots=snapshots,
        footprints=footprints,
        blocks=canonical_blocks(footprints),
    )


def best_route_alignment(snapshot: BattleSnapshot, route_footprints: tuple[ActionFootprint, ...], solution_traces: tuple[CanonicalSolutionTrace, ...]) -> RouteAlignment:
    best = RouteAlignment()
    for trace in solution_traces:
        alignment = align_route_to_solution(snapshot, route_footprints, trace)
        if alignment.score > best.score:
            best = alignment
    return best


def align_route_to_solution(snapshot: BattleSnapshot, route_footprints: tuple[ActionFootprint, ...], solution: CanonicalSolutionTrace) -> RouteAlignment:
    matched_prefix = _matched_prefix_blocks(route_footprints, solution.blocks)
    block_bonus = matched_prefix * 35.0
    closest_step, distance = _closest_solution_step(snapshot, solution.snapshots)
    state_bonus = max(0.0, 120.0 - distance * 4.0) if distance is not None else 0.0
    progress_bonus = _progress_bonus(route_footprints, solution.footprints)
    score = block_bonus + state_bonus + progress_bonus
    return RouteAlignment(
        score=score,
        trace_id=solution.trace_id,
        matched_prefix=matched_prefix,
        closest_step=closest_step,
        state_distance=distance,
        block_bonus=block_bonus,
        state_bonus=state_bonus,
    )


def _matched_prefix_blocks(route_footprints: tuple[ActionFootprint, ...], solution_blocks: tuple[CanonicalBlock, ...]) -> int:
    offset = 0
    matched = 0
    for block in solution_blocks:
        width = len(block.steps)
        if width <= 0 or offset + width > len(route_footprints):
            break
        route_slice = route_footprints[offset : offset + width]
        if _block_distance(route_slice, tuple(step.footprint for step in block.steps)) > 0.35:
            break
        matched += 1
        offset += width
    return matched


def _block_distance(left: tuple[ActionFootprint, ...], right: tuple[ActionFootprint, ...]) -> float:
    if len(left) != len(right):
        return 1.0
    unmatched = list(right)
    total = 0.0
    for item in left:
        best_index = min(range(len(unmatched)), key=lambda index: footprint_distance(item, unmatched[index]))
        total += footprint_distance(item, unmatched.pop(best_index))
    return total / max(1, len(left))


def _closest_solution_step(snapshot: BattleSnapshot, solution_snapshots: tuple[BattleSnapshot, ...]) -> tuple[int, float | None]:
    if not solution_snapshots:
        return (0, None)
    distances = [(index, snapshot_distance(snapshot, candidate)) for index, candidate in enumerate(solution_snapshots)]
    return min(distances, key=lambda item: item[1])


def _progress_bonus(route_footprints: tuple[ActionFootprint, ...], solution_footprints: tuple[ActionFootprint, ...]) -> float:
    if not route_footprints or not solution_footprints:
        return 0.0
    bonus = 0.0
    unused = list(solution_footprints)
    for route in route_footprints:
        best_index = min(range(len(unused)), key=lambda index: footprint_distance(route, unused[index])) if unused else None
        if best_index is None:
            break
        distance = footprint_distance(route, unused[best_index])
        if distance <= 0.25:
            bonus += 8.0
            unused.pop(best_index)
    return bonus
