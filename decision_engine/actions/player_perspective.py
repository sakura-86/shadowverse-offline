# -*- coding: utf-8 -*-
"""Player perspective helpers for local opponent action modeling."""

from __future__ import annotations

from dataclasses import replace

from decision_engine.state.snapshot import BattleSnapshot


def opponent_as_me(snapshot: BattleSnapshot) -> BattleSnapshot:
    return replace(
        snapshot,
        me=snapshot.opponent,
        opponent=snapshot.me,
        is_ally_turn=True,
        is_turn_ended=False,
        legal=None,
    )


def restore_opponent_perspective(snapshot: BattleSnapshot, original: BattleSnapshot) -> BattleSnapshot:
    return replace(
        snapshot,
        me=snapshot.opponent,
        opponent=snapshot.me,
        is_ally_turn=False,
        legal=original.legal,
    )
