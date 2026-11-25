"""Proof-Number Search（PNS）for deterministic AND/OR proof games.

``DeterministicProofSearch`` 保留为 DFS 正确性基线；本模块按 PNS 选择最证明叶：
root_player 的 OR 节点最小化 proof number，对手 AND 节点最小化 disproof number。
一旦根节点证明为 WIN/LOSS 即停止，不继续为无关 sibling 消耗预算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Generic, TypeVar

from decision_engine.planning.deterministic_proof_search import (
    DeterministicProofGame,
    ProofChild,
    ProofNode,
    ProofNodeKey,
    ProofSchedulerContext,
    ProofSearchConfig,
    ProofSearchResult,
    ProofSearchTrace,
    ProofStatus,
    SharedProofGraph,
)

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass
class _PnsNode(Generic[StateT, ActionT]):
    key: ProofNodeKey
    state: StateT
    active_player: str
    status: ProofStatus = ProofStatus.UNKNOWN_BUDGET
    proof_number: float = 1.0
    disproof_number: float = 1.0
    expanded: bool = False
    expandable: bool = True
    blocked_reason: str | None = None
    actions: tuple[ActionT, ...] = ()
    children: list[tuple[ActionT, "_PnsNode[StateT, ActionT]"]] = field(default_factory=list)


class ProofNumberSearch(Generic[StateT, ActionT]):
    """预算受限 PNS，支持 SharedProofGraph 的跨 root certificate reuse。"""

    def __init__(
        self,
        game: DeterministicProofGame[StateT, ActionT],
        *,
        config: ProofSearchConfig | None = None,
        shared_graph: SharedProofGraph[ActionT] | None = None,
    ) -> None:
        self.game = game
        self.config = config or ProofSearchConfig()
        self.shared_graph = shared_graph
        self._nodes: dict[ProofNodeKey, _PnsNode[StateT, ActionT]] = {}
        self._expanded_nodes = 0
        self._terminal_nodes = 0
        self._certificate_hits = 0
        self._cycle_cuts = 0
        self._budget_cuts = 0

    def search(
        self,
        initial_state: StateT,
        *,
        root_player: str,
        scheduler_context: ProofSchedulerContext,
    ) -> ProofSearchResult[ActionT]:
        root = self._get_node(initial_state, root_player=root_player, scheduler_context=scheduler_context, path_keys=())
        while root.status.is_proven is False and self._expanded_nodes < self.config.max_expansions:
            path = self._most_proving_path(root, root_player=root_player)
            leaf = path[-1]
            if not leaf.expandable:
                break
            self._expand(leaf, root_player=root_player, path_keys=tuple(node.key for node in path))
            for node in reversed(path):
                self._update(node, root_player=root_player)
            self._store_proven_path(path)

        if not root.status.is_proven and self._expanded_nodes >= self.config.max_expansions:
            self._budget_cuts += 1
        materialized = {key: self._materialize(node) for key, node in self._nodes.items()}
        states = dict(self._states()) if self.config.capture_observations else {}
        legal_actions = (
            {key: node.actions for key, node in self._nodes.items() if node.expanded}
            if self.config.capture_observations
            else {}
        )
        return ProofSearchResult(
            root_key=root.key,
            root=materialized[root.key],
            nodes=materialized,
            states=states,
            legal_actions=legal_actions,
            trace=ProofSearchTrace(
                expanded_nodes=self._expanded_nodes,
                terminal_nodes=self._terminal_nodes,
                cache_hits=0,
                cycle_cuts=self._cycle_cuts,
                budget_cuts=self._budget_cuts,
                certificate_hits=self._certificate_hits,
            ),
        )

    def _states(self):
        return ((key, node.state) for key, node in self._nodes.items())

    def _get_node(
        self,
        state: StateT,
        *,
        root_player: str,
        scheduler_context: ProofSchedulerContext,
        path_keys: tuple[ProofNodeKey, ...],
    ) -> _PnsNode[StateT, ActionT]:
        key = ProofNodeKey(self.game.state_fingerprint(state), scheduler_context, root_player)
        active_player = self.game.active_player(state, scheduler_context)
        if key in path_keys:
            self._cycle_cuts += 1
            return _PnsNode(
                key=key,
                state=state,
                active_player=active_player,
                status=ProofStatus.UNKNOWN_CYCLE,
                proof_number=inf,
                disproof_number=inf,
                expandable=False,
                blocked_reason=ProofStatus.UNKNOWN_CYCLE.value,
            )
        existing = self._nodes.get(key)
        if existing is not None:
            return existing
        node = _PnsNode(key=key, state=state, active_player=active_player)
        self._nodes[key] = node
        if self.shared_graph is not None:
            certificate = self.shared_graph.certificate(key)
            if certificate is not None:
                self._certificate_hits += 1
                node.status = certificate.status
                node.proof_number = certificate.proof_number
                node.disproof_number = certificate.disproof_number
                node.expandable = False
                node.blocked_reason = certificate.blocked_reason
                return node
        winner = self.game.terminal_winner(state, scheduler_context)
        if winner is not None:
            self._terminal_nodes += 1
            node.status = ProofStatus.PROVEN_WIN if winner == root_player else ProofStatus.PROVEN_LOSS
            node.proof_number, node.disproof_number = _numbers(node.status)
            node.expandable = False
        return node

    def _expand(
        self,
        node: _PnsNode[StateT, ActionT],
        *,
        root_player: str,
        path_keys: tuple[ProofNodeKey, ...],
    ) -> None:
        if node.expanded or not node.expandable:
            return
        self._expanded_nodes += 1
        actions = tuple(self.game.legal_actions(node.state, node.key.scheduler_context, player=node.active_player))
        node.actions = actions
        node.expanded = True
        if not actions:
            node.status = ProofStatus.UNKNOWN_NO_LEGAL_ACTIONS
            node.expandable = False
            node.proof_number = inf
            node.disproof_number = inf
            node.blocked_reason = node.status.value
            return
        for action in actions:
            transition = self.game.transition(
                node.state,
                action,
                node.key.scheduler_context,
                player=node.active_player,
            )
            child = self._get_node(
                transition.state,
                root_player=root_player,
                scheduler_context=transition.scheduler_context,
                path_keys=path_keys,
            )
            if transition.blocked_status is not None:
                child.status = transition.blocked_status
                child.expandable = False
                child.proof_number = inf
                child.disproof_number = inf
                child.blocked_reason = transition.blocked_reason or transition.blocked_status.value
            node.children.append((action, child))
        self._update(node, root_player=root_player)

    def _most_proving_path(self, root: _PnsNode[StateT, ActionT], *, root_player: str) -> list[_PnsNode[StateT, ActionT]]:
        path = [root]
        node = root
        while node.expanded and node.children and not node.status.is_proven:
            if node.active_player == root_player:
                _, child = min(node.children, key=lambda pair: (pair[1].proof_number, _action_order(pair[0])))
            else:
                _, child = min(node.children, key=lambda pair: (pair[1].disproof_number, _action_order(pair[0])))
            if child in path or not child.expandable or not child.expanded:
                path.append(child)
                break
            path.append(child)
            node = child
        return path

    def _update(self, node: _PnsNode[StateT, ActionT], *, root_player: str) -> None:
        if not node.expanded or not node.children or node.status.is_proven:
            return
        children = tuple(child for _, child in node.children)
        wins = tuple(child for child in children if child.status is ProofStatus.PROVEN_WIN)
        losses = tuple(child for child in children if child.status is ProofStatus.PROVEN_LOSS)
        if node.active_player == root_player:
            if wins:
                node.status = ProofStatus.PROVEN_WIN
                node.proof_number, node.disproof_number = _numbers(node.status)
                node.expandable = False
                return
            if len(losses) == len(children):
                node.status = ProofStatus.PROVEN_LOSS
                node.proof_number, node.disproof_number = _numbers(node.status)
                node.expandable = False
                return
            node.proof_number = min(child.proof_number for child in children)
            node.disproof_number = sum(child.disproof_number for child in children)
        else:
            if losses:
                node.status = ProofStatus.PROVEN_LOSS
                node.proof_number, node.disproof_number = _numbers(node.status)
                node.expandable = False
                return
            if len(wins) == len(children):
                node.status = ProofStatus.PROVEN_WIN
                node.proof_number, node.disproof_number = _numbers(node.status)
                node.expandable = False
                return
            node.proof_number = sum(child.proof_number for child in children)
            node.disproof_number = min(child.disproof_number for child in children)

    def _store_proven_path(self, path: list[_PnsNode[StateT, ActionT]]) -> None:
        if self.shared_graph is None:
            return
        for node in path:
            if not node.status.is_proven:
                continue
            self.shared_graph.store(
                self._materialize(node),
                state=node.state if self.config.capture_observations else None,
                legal_actions=node.actions if self.config.capture_observations else None,
            )

    def _materialize(self, node: _PnsNode[StateT, ActionT]) -> ProofNode[ActionT]:
        children = tuple(ProofChild(action=action, node_key=child.key, status=child.status) for action, child in node.children)
        if node.status is ProofStatus.PROVEN_WIN:
            proven = tuple(action for action, child in node.children if child.status is ProofStatus.PROVEN_WIN)
            refuted = tuple(action for action, child in node.children if child.status is ProofStatus.PROVEN_LOSS)
        elif node.status is ProofStatus.PROVEN_LOSS:
            proven = tuple(action for action, child in node.children if child.status is ProofStatus.PROVEN_WIN)
            refuted = tuple(action for action, child in node.children if child.status is ProofStatus.PROVEN_LOSS)
        else:
            proven = ()
            refuted = ()
        return ProofNode(
            key=node.key,
            active_player=node.active_player,
            status=node.status,
            proof_number=node.proof_number,
            disproof_number=node.disproof_number,
            children=children,
            proven_actions=proven,
            refuted_actions=refuted,
            blocked_reason=node.blocked_reason if not node.status.is_proven else None,
        )


def _numbers(status: ProofStatus) -> tuple[float, float]:
    if status is ProofStatus.PROVEN_WIN:
        return 0.0, inf
    if status is ProofStatus.PROVEN_LOSS:
        return inf, 0.0
    return 1.0, 1.0


def _action_order(action: object) -> str:
    return repr(action)
