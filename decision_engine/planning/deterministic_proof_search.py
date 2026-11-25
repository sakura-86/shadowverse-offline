"""通用、确定性的二值 AND/OR proof-search 内核。

本模块不依赖 Shadowverse 状态、动作或规则。调用方提供有限动作游戏的枚举、转移、
终局赢家和调度上下文；内核只证明 ``root_player`` 是否存在对抗任意合法回应的必胜
条件策略。M0 使用预算受限的完备 DFS 作为正确性基线，并保留 proof/disproof 数，
后续可替换为 PNS/DFPN 的最证明叶调度而不改变证明传播语义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import inf
from typing import Generic, Hashable, Mapping, Protocol, Sequence, TypeVar


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class ProofStatus(Enum):
    """从 root_player 视角的证明状态。"""

    PROVEN_WIN = "proven_win"
    PROVEN_LOSS = "proven_loss"
    UNKNOWN_BUDGET = "unknown_budget"
    UNKNOWN_CYCLE = "unknown_cycle"
    UNKNOWN_DEPTH = "unknown_depth"
    UNKNOWN_NO_LEGAL_ACTIONS = "unknown_no_legal_actions"
    UNKNOWN_EXECUTOR_GAP = "unknown_executor_gap"

    @property
    def is_proven(self) -> bool:
        return self in {ProofStatus.PROVEN_WIN, ProofStatus.PROVEN_LOSS}


@dataclass(frozen=True)
class ProofSchedulerContext:
    """影响未来确定性随机调度的搜索位置。

    M0 不解释其数值；真实适配层会将其对应到 turn/action 索引和 branch seed。它必须
    进入缓存 identity，避免同一表面 state 在不同未来随机调度下被错误合并。
    """

    turn_index: int
    action_index: int
    current_player: str


@dataclass(frozen=True)
class ProofTransition(Generic[StateT]):
    state: StateT
    scheduler_context: ProofSchedulerContext
    blocked_status: ProofStatus | None = None
    blocked_reason: str | None = None


class DeterministicProofGame(Protocol[StateT, ActionT]):
    """M0 搜索器消费的最小游戏适配协议。"""

    def state_fingerprint(self, state: StateT) -> str:
        ...

    def active_player(self, state: StateT, scheduler_context: ProofSchedulerContext) -> str:
        ...

    def terminal_winner(self, state: StateT, scheduler_context: ProofSchedulerContext) -> str | None:
        """返回终局赢家；非终局返回 None。M0 不把非终局解释为 draw。"""
        ...

    def legal_actions(
        self,
        state: StateT,
        scheduler_context: ProofSchedulerContext,
        *,
        player: str,
    ) -> Sequence[ActionT]:
        ...

    def transition(
        self,
        state: StateT,
        action: ActionT,
        scheduler_context: ProofSchedulerContext,
        *,
        player: str,
    ) -> ProofTransition[StateT]:
        ...


@dataclass(frozen=True)
class ProofSearchConfig:
    """M0 资源边界；耗尽预算只产生 UNKNOWN，不产生假终局。"""

    max_expansions: int = 10_000
    capture_observations: bool = False
    # DFS 递归深度硬上限：远低于 Python 递归极限，超过返回 UNKNOWN_DEPTH（可审计，不伪造终局）。
    # 防止在可长时间推进的局面上深度优先无限深钻导致栈溢出。
    max_depth: int = 500

    def __post_init__(self) -> None:
        if self.max_expansions < 0:
            raise ValueError("max_expansions must be non-negative")
        if self.max_depth < 1:
            raise ValueError("max_depth must be positive")


@dataclass(frozen=True)
class ProofNodeKey:
    state_fingerprint: str
    scheduler_context: ProofSchedulerContext
    root_player: str


@dataclass(frozen=True)
class ProofChild(Generic[ActionT]):
    action: ActionT
    node_key: ProofNodeKey
    status: ProofStatus


@dataclass(frozen=True)
class ProofNode(Generic[ActionT]):
    key: ProofNodeKey
    active_player: str
    status: ProofStatus
    proof_number: float
    disproof_number: float
    children: tuple[ProofChild[ActionT], ...] = ()
    proven_actions: tuple[ActionT, ...] = ()
    refuted_actions: tuple[ActionT, ...] = ()
    blocked_reason: str | None = None


@dataclass(frozen=True)
class ProofSearchTrace:
    expanded_nodes: int
    terminal_nodes: int
    cache_hits: int
    cycle_cuts: int
    budget_cuts: int
    certificate_hits: int = 0


@dataclass(frozen=True)
class ProofSearchResult(Generic[ActionT]):
    root_key: ProofNodeKey
    root: ProofNode[ActionT]
    nodes: Mapping[ProofNodeKey, ProofNode[ActionT]]
    states: Mapping[ProofNodeKey, object] = field(default_factory=dict)
    legal_actions: Mapping[ProofNodeKey, tuple[ActionT, ...]] = field(default_factory=dict)
    trace: ProofSearchTrace = field(default_factory=lambda: ProofSearchTrace(0, 0, 0, 0, 0))


class SharedProofGraph(Generic[ActionT]):
    """跨 proof root 复用的已证明 AND/OR certificate store。

    只持久化 ``PROVEN_WIN/PROVEN_LOSS``，绝不缓存 ``UNKNOWN_BUDGET`` 等暂态结论。
    ``ProofNodeKey`` 已包含 root player 和 scheduler context，因此复用不会跨越不同
    胜负视角或未来随机调度。节点自身的 children 即为可审计 proof graph 边。
    """

    def __init__(self) -> None:
        self._certificates: dict[ProofNodeKey, ProofNode[ActionT]] = {}
        self._states: dict[ProofNodeKey, object] = {}
        self._legal_actions: dict[ProofNodeKey, tuple[ActionT, ...]] = {}

    @property
    def certificate_count(self) -> int:
        return len(self._certificates)

    def certificate(self, key: ProofNodeKey) -> ProofNode[ActionT] | None:
        return self._certificates.get(key)

    def state(self, key: ProofNodeKey) -> object | None:
        return self._states.get(key)

    def legal_actions(self, key: ProofNodeKey) -> tuple[ActionT, ...] | None:
        return self._legal_actions.get(key)

    def store(
        self,
        node: ProofNode[ActionT],
        *,
        state: object | None = None,
        legal_actions: tuple[ActionT, ...] | None = None,
    ) -> None:
        if not node.status.is_proven:
            return
        self._certificates.setdefault(node.key, node)
        if state is not None:
            self._states.setdefault(node.key, state)
        if legal_actions is not None:
            self._legal_actions.setdefault(node.key, legal_actions)


class DeterministicProofSearch(Generic[StateT, ActionT]):
    """预算受限、缓存感知的二值 AND/OR minimax 正确性内核。"""

    def __init__(
        self,
        game: DeterministicProofGame[StateT, ActionT],
        *,
        config: ProofSearchConfig | None = None,
        shared_graph: SharedProofGraph[ActionT] | None = None,
    ):
        self.game = game
        self.config = config or ProofSearchConfig()
        self.shared_graph = shared_graph
        self._nodes: dict[ProofNodeKey, ProofNode[ActionT]] = {}
        self._states: dict[ProofNodeKey, StateT] = {}
        self._legal_actions: dict[ProofNodeKey, tuple[ActionT, ...]] = {}
        self._active: set[ProofNodeKey] = set()
        self._expanded_nodes = 0
        self._terminal_nodes = 0
        self._cache_hits = 0
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
        """求解或在预算内尽量证明根命题。搜索器实例仅可执行一次。"""

        if self._nodes or self._active:
            raise RuntimeError("DeterministicProofSearch instances are single-use; create a new instance")
        root = self._solve(initial_state, root_player=root_player, scheduler_context=scheduler_context, depth=0)
        return ProofSearchResult(
            root_key=root.key,
            root=root,
            nodes=dict(self._nodes),
            states=dict(self._states),
            legal_actions=dict(self._legal_actions),
            trace=ProofSearchTrace(
                expanded_nodes=self._expanded_nodes,
                terminal_nodes=self._terminal_nodes,
                cache_hits=self._cache_hits,
                cycle_cuts=self._cycle_cuts,
                budget_cuts=self._budget_cuts,
                certificate_hits=self._certificate_hits,
            ),
        )

    def _solve(
        self,
        state: StateT,
        *,
        root_player: str,
        scheduler_context: ProofSchedulerContext,
        depth: int,
    ) -> ProofNode[ActionT]:
        key = ProofNodeKey(
            state_fingerprint=self.game.state_fingerprint(state),
            scheduler_context=scheduler_context,
            root_player=root_player,
        )
        if self.config.capture_observations:
            self._states.setdefault(key, state)
        cached = self._nodes.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        if self.shared_graph is not None:
            certificate = self.shared_graph.certificate(key)
            if certificate is not None:
                self._certificate_hits += 1
                self._nodes[key] = certificate
                if self.config.capture_observations:
                    state = self.shared_graph.state(key)
                    actions = self.shared_graph.legal_actions(key)
                    if state is not None:
                        self._states.setdefault(key, state)  # type: ignore[arg-type]
                    if actions is not None:
                        self._legal_actions.setdefault(key, actions)
                return certificate

        active_player = self.game.active_player(state, scheduler_context)
        winner = self.game.terminal_winner(state, scheduler_context)
        if winner is not None:
            self._terminal_nodes += 1
            if self.config.capture_observations:
                self._legal_actions.setdefault(key, ())
            node = self._terminal_node(key, active_player=active_player, winner=winner, root_player=root_player)
            self._nodes[key] = node
            self._store_certificate(node, state=state, actions=())
            return node

        if key in self._active:
            self._cycle_cuts += 1
            return self._unknown_node(key, active_player, ProofStatus.UNKNOWN_CYCLE)
        if depth >= self.config.max_depth:
            # 深度硬上限：不伪造终局，返回可审计 UNKNOWN_DEPTH，避免栈溢出。
            self._budget_cuts += 1
            node = self._unknown_node(key, active_player, ProofStatus.UNKNOWN_DEPTH)
            self._nodes[key] = node
            return node
        if self._expanded_nodes >= self.config.max_expansions:
            self._budget_cuts += 1
            node = self._unknown_node(key, active_player, ProofStatus.UNKNOWN_BUDGET)
            self._nodes[key] = node
            return node

        self._expanded_nodes += 1
        actions = tuple(self.game.legal_actions(state, scheduler_context, player=active_player))
        if self.config.capture_observations:
            self._legal_actions[key] = actions
        if not actions:
            node = self._unknown_node(key, active_player, ProofStatus.UNKNOWN_NO_LEGAL_ACTIONS)
            self._nodes[key] = node
            return node

        self._active.add(key)
        try:
            children: list[ProofChild[ActionT]] = []
            for action in actions:
                transition = self.game.transition(
                    state,
                    action,
                    scheduler_context,
                    player=active_player,
                )
                if transition.blocked_status is not None:
                    child_key = ProofNodeKey(
                        state_fingerprint=self.game.state_fingerprint(transition.state),
                        scheduler_context=transition.scheduler_context,
                        root_player=root_player,
                    )
                    if self.config.capture_observations:
                        self._states.setdefault(child_key, transition.state)
                    child = self._unknown_node(
                        child_key,
                        self.game.active_player(transition.state, transition.scheduler_context),
                        transition.blocked_status,
                        blocked_reason=transition.blocked_reason,
                    )
                    self._nodes[child_key] = child
                else:
                    child = self._solve(
                        transition.state,
                        root_player=root_player,
                        scheduler_context=transition.scheduler_context,
                        depth=depth + 1,
                    )
                children.append(ProofChild(action=action, node_key=child.key, status=child.status))
        finally:
            self._active.remove(key)

        node = self._aggregate_node(
            key,
            active_player=active_player,
            root_player=root_player,
            children=tuple(children),
        )
        self._nodes[key] = node
        self._store_certificate(node, state=state, actions=actions)
        return node

    def _store_certificate(
        self,
        node: ProofNode[ActionT],
        *,
        state: StateT | None = None,
        actions: tuple[ActionT, ...] | None = None,
    ) -> None:
        if self.shared_graph is None or not node.status.is_proven:
            return
        self.shared_graph.store(node, state=state, legal_actions=actions)

    def _terminal_node(
        self,
        key: ProofNodeKey,
        *,
        active_player: str,
        winner: str,
        root_player: str,
    ) -> ProofNode[ActionT]:
        status = ProofStatus.PROVEN_WIN if winner == root_player else ProofStatus.PROVEN_LOSS
        proof, disproof = _numbers_for_status(status)
        return ProofNode(
            key=key,
            active_player=active_player,
            status=status,
            proof_number=proof,
            disproof_number=disproof,
        )

    def _unknown_node(
        self,
        key: ProofNodeKey,
        active_player: str,
        status: ProofStatus,
        *,
        blocked_reason: str | None = None,
    ) -> ProofNode[ActionT]:
        return ProofNode(
            key=key,
            active_player=active_player,
            status=status,
            proof_number=1.0,
            disproof_number=1.0,
            blocked_reason=blocked_reason or status.value,
        )

    def _aggregate_node(
        self,
        key: ProofNodeKey,
        *,
        active_player: str,
        root_player: str,
        children: tuple[ProofChild[ActionT], ...],
    ) -> ProofNode[ActionT]:
        child_nodes = tuple(self._nodes.get(child.node_key) for child in children)
        proofs = tuple(node.proof_number if node is not None else 1.0 for node in child_nodes)
        disproofs = tuple(node.disproof_number if node is not None else 1.0 for node in child_nodes)
        wins = tuple(child for child in children if child.status is ProofStatus.PROVEN_WIN)
        losses = tuple(child for child in children if child.status is ProofStatus.PROVEN_LOSS)
        all_win = len(wins) == len(children)
        all_loss = len(losses) == len(children)

        if active_player == root_player:
            status = ProofStatus.PROVEN_WIN if wins else ProofStatus.PROVEN_LOSS if all_loss else _first_unknown(children)
            proof_number = min(proofs)
            disproof_number = sum(disproofs)
            proven_actions = tuple(child.action for child in wins)
            refuted_actions = tuple(child.action for child in losses)
        else:
            status = ProofStatus.PROVEN_LOSS if losses else ProofStatus.PROVEN_WIN if all_win else _first_unknown(children)
            proof_number = sum(proofs)
            disproof_number = min(disproofs)
            proven_actions = tuple(child.action for child in wins)
            refuted_actions = tuple(child.action for child in losses)

        if status.is_proven:
            proof_number, disproof_number = _numbers_for_status(status)
        return ProofNode(
            key=key,
            active_player=active_player,
            status=status,
            proof_number=proof_number,
            disproof_number=disproof_number,
            children=children,
            proven_actions=proven_actions,
            refuted_actions=refuted_actions,
            blocked_reason=None if status.is_proven else status.value,
        )


def _numbers_for_status(status: ProofStatus) -> tuple[float, float]:
    if status is ProofStatus.PROVEN_WIN:
        return 0.0, inf
    if status is ProofStatus.PROVEN_LOSS:
        return inf, 0.0
    return 1.0, 1.0


def _first_unknown(children: tuple[ProofChild[ActionT], ...]) -> ProofStatus:
    for child in children:
        if not child.status.is_proven:
            return child.status
    raise AssertionError("expected at least one unknown child")
