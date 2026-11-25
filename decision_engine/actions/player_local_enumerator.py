# -*- coding: utf-8 -*-
"""Player-aware wrapper for local action enumeration."""

from __future__ import annotations

from decision_engine.actions.local_enumerator import LocalActionEnumerator
from decision_engine.actions.model import LegalAction
from decision_engine.actions.player_perspective import opponent_as_me
from decision_engine.state.snapshot import BattleSnapshot


class PlayerLocalActionEnumerator:
    def __init__(self, base: LocalActionEnumerator | None = None):
        self.base = base or LocalActionEnumerator()

    def enumerate(self, snapshot: BattleSnapshot, *, player: str = "me") -> list[LegalAction]:
        if player == "me":
            return self.base.enumerate(snapshot)
        if player == "opponent":
            return self.base.enumerate(opponent_as_me(snapshot))
        raise ValueError(f"unsupported player: {player}")
