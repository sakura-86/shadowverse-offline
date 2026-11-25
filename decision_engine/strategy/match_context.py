# -*- coding: utf-8 -*-
"""对局上下文数据结构。

MatchContext 让 Evaluator 知道双方的体系、有效资源和终结路线，
从而做差异化评分。不执行效果，不判断合法性。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArchetypeProfile:
    """环境中某个卡组体系的画像。离线预计算。"""

    archetype_id: str
    class_id: int
    display_name: str = ""
    effective_resources: frozenset[str] = frozenset()
    resource_thresholds: dict[str, int] = field(default_factory=dict)
    key_card_ids: frozenset[int] = frozenset()
    card_roles: dict[str, frozenset[int]] = field(default_factory=dict)
    win_condition: str = ""
    aggro_level: float = 0.5
    estimated_kill_turn: int = 8


@dataclass(frozen=True)
class DeckProfile:
    """我方卡组画像。赛前确定。"""

    class_id: int
    archetype: ArchetypeProfile
    deck_card_ids: frozenset[int] = frozenset()
    deck_card_counts: dict[int, int] = field(default_factory=dict)


@dataclass
class OpponentProfile:
    """对手画像。实时推断，可变。"""

    class_id: int
    likely_archetypes: list[tuple[ArchetypeProfile, float]] = field(default_factory=list)
    observed_card_ids: list[int] = field(default_factory=list)

    @property
    def primary_archetype(self) -> ArchetypeProfile | None:
        return self.likely_archetypes[0][0] if self.likely_archetypes else None

    @property
    def estimated_kill_turn(self) -> int:
        arch = self.primary_archetype
        return arch.estimated_kill_turn if arch else 8

    def observe(self, card_id: int) -> None:
        if card_id and card_id not in self.observed_card_ids:
            self.observed_card_ids.append(card_id)


@dataclass
class MatchContext:
    """对局上下文。注入 Evaluator。"""

    my_profile: DeckProfile
    opponent_profile: OpponentProfile
    turn: int = 0
