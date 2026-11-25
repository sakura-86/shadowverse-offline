# -*- coding: utf-8 -*-
"""本地动作空间与协议动作空间 diff。"""

from __future__ import annotations

from dataclasses import dataclass

from decision_engine.actions.local_enumerator import LocalActionEnumerator
from decision_engine.actions.model import LegalAction
from decision_engine.actions.protocol_enumerator import ProtocolActionEnumerator
from decision_engine.state.snapshot import BattleSnapshot


@dataclass(frozen=True)
class ActionSpaceDiffReport:
    local_count: int
    protocol_count: int
    missing: tuple[LegalAction, ...]
    extra: tuple[LegalAction, ...]
    matched: tuple[LegalAction, ...]
    mismatched: tuple[tuple[LegalAction, LegalAction], ...] = ()


def diff_local_protocol_actions(
    snapshot: BattleSnapshot,
    *,
    local_enumerator: LocalActionEnumerator | None = None,
    protocol_enumerator: ProtocolActionEnumerator | None = None,
) -> ActionSpaceDiffReport:
    local_actions = tuple((local_enumerator or LocalActionEnumerator()).enumerate(snapshot))
    protocol_actions = tuple((protocol_enumerator or ProtocolActionEnumerator()).enumerate(snapshot))
    local_by_key = {_action_key(action): action for action in local_actions}
    protocol_by_key = {_action_key(action): action for action in protocol_actions}

    missing_keys = tuple(sorted(set(protocol_by_key) - set(local_by_key)))
    extra_keys = tuple(sorted(set(local_by_key) - set(protocol_by_key)))
    matched_keys = tuple(sorted(set(local_by_key) & set(protocol_by_key)))

    return ActionSpaceDiffReport(
        local_count=len(local_actions),
        protocol_count=len(protocol_actions),
        missing=tuple(protocol_by_key[key] for key in missing_keys),
        extra=tuple(local_by_key[key] for key in extra_keys),
        matched=tuple(local_by_key[key] for key in matched_keys),
    )


def _action_key(action: LegalAction) -> tuple:
    return (
        action.kind.value,
        action.source_uid,
        action.play_kind.value if action.play_kind is not None else None,
        action.cost,
        action.target_groups,
        action.selected_indexes,
        action.material_uids,
        action.variant,
    )
