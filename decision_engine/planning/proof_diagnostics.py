"""Proof-search executor gap diagnostics.

从带 observation 的 proof result 中恢复每条 ``UNKNOWN_EXECUTOR_GAP`` parent edge，
保留 source card、动作、scheduler context 和 LocalProofGame 传递的具体阻断原因。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from decision_engine.planning.deterministic_proof_search import ProofSearchResult, ProofStatus
from decision_engine.planning.local_proof_game import LocalProofAction
from decision_engine.state.snapshot import BattleSnapshot


@dataclass(frozen=True)
class ProofExecutorGap:
    parent_fingerprint: str
    player: str
    turn_index: int
    action_index: int
    action_kind: str
    source_base_card_id: int | None
    source_name: str | None
    blocked_reason: str
    replay_key: Mapping[str, Any] | None

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProofExecutorGapReport:
    gaps: tuple[ProofExecutorGap, ...]
    by_base_card_id: Mapping[str, int]
    by_action_kind: Mapping[str, int]
    by_blocked_reason: Mapping[str, int]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "gap_count": len(self.gaps),
            "gaps": [item.to_jsonable() for item in self.gaps],
            "by_base_card_id": dict(self.by_base_card_id),
            "by_action_kind": dict(self.by_action_kind),
            "by_blocked_reason": dict(self.by_blocked_reason),
        }


def collect_proof_executor_gaps(result: ProofSearchResult) -> ProofExecutorGapReport:
    """收集所有由 parent action 触发的 strict executor gaps。

    对没有捕获 state 的旧 result，仍保留动作/原因；source card 标为 None，不猜测。
    """

    gaps: list[ProofExecutorGap] = []
    for parent_key, parent in result.nodes.items():
        snapshot = result.states.get(parent_key)
        for child in parent.children:
            if child.status is not ProofStatus.UNKNOWN_EXECUTOR_GAP:
                continue
            child_node = result.nodes.get(child.node_key)
            reason = child_node.blocked_reason if child_node is not None else "unknown_executor_gap"
            local_action = child.action if isinstance(child.action, LocalProofAction) else None
            raw_action = local_action.action if local_action is not None else child.action
            source_base: int | None = None
            source_name: str | None = None
            if isinstance(snapshot, BattleSnapshot) and local_action is not None:
                source = snapshot.cards_by_uid.get(local_action.action.source_uid)
                if source is not None:
                    source_base = int(source.base_card_id or source.card_id or 0) or None
                    source_name = source.name or None
            gaps.append(ProofExecutorGap(
                parent_fingerprint=parent_key.state_fingerprint,
                player=parent.active_player,
                turn_index=parent_key.scheduler_context.turn_index,
                action_index=parent_key.scheduler_context.action_index,
                action_kind=str(getattr(getattr(raw_action, "kind", None), "value", getattr(raw_action, "kind", "unknown"))),
                source_base_card_id=source_base,
                source_name=source_name,
                blocked_reason=reason or "unknown_executor_gap",
                replay_key=asdict(local_action.replay_key) if local_action is not None and local_action.replay_key is not None else None,
            ))
    gaps.sort(key=lambda item: (item.source_base_card_id or 0, item.action_kind, item.blocked_reason, item.parent_fingerprint))
    return ProofExecutorGapReport(
        gaps=tuple(gaps),
        by_base_card_id=_counter({str(item.source_base_card_id or "none"): 1 for item in gaps}, gaps, key=lambda item: str(item.source_base_card_id or "none")),
        by_action_kind=_counter({}, gaps, key=lambda item: item.action_kind),
        by_blocked_reason=_counter({}, gaps, key=lambda item: item.blocked_reason),
    )


def _counter(_unused: Mapping[str, int], gaps: list[ProofExecutorGap], *, key) -> Mapping[str, int]:
    return dict(sorted(Counter(key(item) for item in gaps).items()))
