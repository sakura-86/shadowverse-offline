# -*- coding: utf-8 -*-
"""对局策略与体系识别。"""

from .context_manager import MatchContextManager
from .game_plan import (
    GamePlanAnalyzer,
    GamePlanAnalyzerFactory,
    NoopGamePlanAnalyzer,
    OrderingRule,
    ResourceReservation,
    SpellboostGamePlanAnalyzer,
    TacticalIntent,
    TacticalWindow,
    TacticalWindowKind,
    ThresholdObjective,
)
from .match_context import (
    ArchetypeProfile,
    DeckProfile,
    MatchContext,
    OpponentProfile,
)
from .meta_snapshot import MetaSnapshot, MetaSnapshotLoader
from .opponent_deck_identifier import DeckHypothesis, OpponentDeckIdentifier, OpponentDeckIdentifierConfig
from .opponent_identifier import OpponentIdentifier
from .opponent_observation import OpponentObservedCards, extract_opponent_observed_cards

__all__ = [
    "ArchetypeProfile",
    "DeckHypothesis",
    "DeckProfile",
    "MatchContext",
    "MatchContextManager",
    "GamePlanAnalyzer",
    "GamePlanAnalyzerFactory",
    "NoopGamePlanAnalyzer",
    "OrderingRule",
    "ResourceReservation",
    "SpellboostGamePlanAnalyzer",
    "TacticalIntent",
    "TacticalWindow",
    "TacticalWindowKind",
    "ThresholdObjective",
    "MetaSnapshot",
    "MetaSnapshotLoader",
    "OpponentDeckIdentifier",
    "OpponentDeckIdentifierConfig",
    "OpponentIdentifier",
    "OpponentObservedCards",
    "OpponentProfile",
    "extract_opponent_observed_cards",
]
