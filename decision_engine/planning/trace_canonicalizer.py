# -*- coding: utf-8 -*-
"""Order-insensitive canonicalization for solution/search traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from decision_engine.planning.trace_footprint import ActionFootprint


@dataclass(frozen=True)
class CanonicalStep:
    index: int
    footprint: ActionFootprint

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "footprint": self.footprint.to_dict()}


@dataclass(frozen=True)
class CanonicalBlock:
    index: int
    steps: tuple[CanonicalStep, ...]

    @property
    def action_keys(self) -> tuple[str, ...]:
        return tuple(step.footprint.action_key for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action_keys": list(self.action_keys),
            "steps": [step.to_dict() for step in self.steps],
        }


def canonical_blocks(footprints: tuple[ActionFootprint, ...]) -> tuple[CanonicalBlock, ...]:
    blocks: list[CanonicalBlock] = []
    current: list[CanonicalStep] = []
    for index, footprint in enumerate(footprints):
        step = CanonicalStep(index=index, footprint=footprint)
        if current and any(step.footprint.conflicts_with(selected.footprint) for selected in current):
            blocks.append(CanonicalBlock(index=len(blocks), steps=tuple(sorted(current, key=lambda item: item.footprint.action_key))))
            current = []
        current.append(step)
    if current:
        blocks.append(CanonicalBlock(index=len(blocks), steps=tuple(sorted(current, key=lambda item: item.footprint.action_key))))
    return tuple(blocks)


def canonical_block_key(block: CanonicalBlock) -> tuple[str, ...]:
    return tuple(sorted(block.action_keys))


def blocks_to_dict(blocks: tuple[CanonicalBlock, ...]) -> list[dict[str, Any]]:
    return [block.to_dict() for block in blocks]
