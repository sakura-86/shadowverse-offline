"""Phase N1.10 M2：固定 seed 的确定性 proof job。

本模块把 M0 通用 proof-search 与 M1 严格本地适配编排为一个可审计、可恢复的作业：

- 固定 match seed / first_player / branch_seed_base / deck 摘要，写入 manifest；
- 用低成本 PUCT 生成一条 warm-start 主变（只作动作排序/审计，不作为证明真值）；
- 在预算内尽力求解，达到预算返回可审计 ``UNKNOWN_BUDGET``，不伪造终局；
- 对给定 root 和其"镜像 root"（交换先后手）分别求解，便于验证镜像对称性。

job 不写任何卡牌规则；合法性/转移/随机全部来自 M1 ``LocalProofGame``。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.execution.local_match_setup import (
    LocalDeckList,
    LocalMatchSetup,
    LocalMatchSetupConfig,
)
from decision_engine.execution.player_local_predictor import PlayerLocalPredictor
from decision_engine.planning.deterministic_proof_search import (
    DeterministicProofSearch,
    ProofSchedulerContext,
    ProofSearchConfig,
    ProofSearchResult,
    ProofStatus,
    SharedProofGraph,
)
from decision_engine.planning.local_proof_game import LocalProofGame, LocalProofGameConfig
from decision_engine.planning.proof_number_search import ProofNumberSearch
from decision_engine.state.snapshot import BattleSnapshot


@dataclass(frozen=True)
class ProofJobSpec:
    """固定确定性 root 的作业定义。"""

    job_id: str
    deck_a: LocalDeckList
    deck_b: LocalDeckList
    seed: int
    root_player: str
    first_player: str = "me"
    branch_seed_base: int = 0
    max_expansions: int = 5000
    capture_observations: bool = False
    proof_strategy: str = "pns"
    warm_start_simulations: int = 32
    warm_start_max_depth: int = 7

    def __post_init__(self) -> None:
        if self.proof_strategy not in {"dfs", "pns"}:
            raise ValueError("proof_strategy must be 'dfs' or 'pns'")

    def deterministic_contract(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "seed": self.seed,
            "first_player": self.first_player,
            "root_player": self.root_player,
            "branch_seed_base": self.branch_seed_base,
            "deck_a_digest": _deck_digest(self.deck_a),
            "deck_b_digest": _deck_digest(self.deck_b),
            "max_expansions": self.max_expansions,
            "capture_observations": self.capture_observations,
            "proof_strategy": self.proof_strategy,
        }


@dataclass(frozen=True)
class WarmStartLine:
    """PUCT warm-start 主变（仅排序/审计，不作证明真值）。"""

    action_keys: tuple[str, ...]
    reached_terminal: bool
    winner: str | None
    turns: int


@dataclass(frozen=True)
class ProofRootReport:
    label: str
    root_player: str
    status: str
    proven_action_count: int
    refuted_action_count: int
    proof_number: float
    disproof_number: float
    node_count: int
    expanded_nodes: int
    terminal_nodes: int
    executor_gap_nodes: int
    budget_cuts: int
    cycle_cuts: int
    blocked_reason: str | None


@dataclass(frozen=True)
class ProofJobManifest:
    job_id: str
    deterministic_contract: Mapping[str, Any]
    warm_start: Mapping[str, Any]
    primary: ProofRootReport
    mirror: ProofRootReport | None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "deterministic_contract": dict(self.deterministic_contract),
            "warm_start": dict(self.warm_start),
            "primary": _report_to_jsonable(self.primary),
            "mirror": _report_to_jsonable(self.mirror) if self.mirror is not None else None,
        }


@dataclass(frozen=True)
class ProofJobResult:
    manifest: ProofJobManifest
    primary_result: ProofSearchResult
    mirror_result: ProofSearchResult | None
    shared_graph: SharedProofGraph


class ProofJobRunner:
    """编排固定 seed 的 warm-start + 双根 proof-search。"""

    def __init__(self, *, predictor: PlayerLocalPredictor):
        self.predictor = predictor

    def run(
        self,
        spec: ProofJobSpec,
        *,
        solve_mirror: bool = True,
        shared_graph: SharedProofGraph | None = None,
    ) -> ProofJobResult:
        graph = shared_graph or SharedProofGraph()
        primary_start = self._start_state(spec, first_player=spec.first_player)
        warm_start = self._warm_start_line(spec, primary_start)
        primary_result = self._solve(spec, primary_start, label="primary", shared_graph=graph)

        mirror_result: ProofSearchResult | None = None
        mirror_report: ProofRootReport | None = None
        if solve_mirror:
            mirror_first = "opponent" if spec.first_player == "me" else "me"
            mirror_start = self._start_state(spec, first_player=mirror_first)
            mirror_result = self._solve(spec, mirror_start, label="mirror", shared_graph=graph)
            mirror_report = _root_report(mirror_result, label="mirror", root_player=spec.root_player)

        manifest = ProofJobManifest(
            job_id=spec.job_id,
            deterministic_contract=spec.deterministic_contract(),
            warm_start={
                "action_keys": list(warm_start.action_keys),
                "reached_terminal": warm_start.reached_terminal,
                "winner": warm_start.winner,
                "turns": warm_start.turns,
            },
            primary=_root_report(primary_result, label="primary", root_player=spec.root_player),
            mirror=mirror_report,
        )
        return ProofJobResult(
            manifest=manifest,
            primary_result=primary_result,
            mirror_result=mirror_result,
            shared_graph=graph,
        )

    # -- internals ---------------------------------------------------------

    def _game(self, spec: ProofJobSpec) -> LocalProofGame:
        return LocalProofGame(
            predictor=self.predictor,
            config=LocalProofGameConfig(branch_seed_base=spec.branch_seed_base),
        )

    def _start_state(self, spec: ProofJobSpec, *, first_player: str):
        setup = LocalMatchSetup().create(
            spec.deck_a,
            spec.deck_b,
            LocalMatchSetupConfig(seed=spec.seed, first_player=first_player),
        )
        if setup.snapshot is None:
            raise ValueError(f"proof job {spec.job_id} match setup failed: {setup.errors}")
        game = self._game(spec)
        start = game.start_turn(setup.snapshot, player=first_player, turn_index=1)
        return start

    def _solve(
        self,
        spec: ProofJobSpec,
        start,
        *,
        label: str,
        shared_graph: SharedProofGraph,
    ) -> ProofSearchResult:
        game = self._game(spec)
        search_type = ProofNumberSearch if spec.proof_strategy == "pns" else DeterministicProofSearch
        search = search_type(
            game,
            config=ProofSearchConfig(
                max_expansions=spec.max_expansions,
                capture_observations=spec.capture_observations,
            ),
            shared_graph=shared_graph,
        )
        return search.search(start.snapshot, root_player=spec.root_player, scheduler_context=start.scheduler_context)

    def _warm_start_line(self, spec: ProofJobSpec, start) -> WarmStartLine:
        """用低成本 PUCT 生成候选主变；仅记录动作 key 序列与是否触达终局。"""

        from decision_engine.planning.diff_value import DiffSearchValue
        from decision_engine.planning.puct import DefaultTurnTransitioner, PuctConfig, PuctSearch
        from decision_engine.planning.search_prior import UniformSearchPrior
        from decision_engine.actions.player_local_enumerator import PlayerLocalActionEnumerator
        from decision_engine.execution.execution_context import LocalExecutionContext
        from decision_engine.planning.local_proof_game import is_strict_prediction

        enumerator = PlayerLocalActionEnumerator()
        snapshot = start.snapshot
        context = start.scheduler_context
        action_keys: list[str] = []
        winner: str | None = None
        reached_terminal = False
        turns = 0
        game = self._game(spec)

        for _ in range(spec.warm_start_max_depth * 4):
            winner = game.terminal_winner(snapshot, context)
            if winner is not None:
                reached_terminal = True
                break
            player = context.current_player
            search = PuctSearch(
                enumerator=enumerator,
                child_enumerator=enumerator,
                predictor=self.predictor,
                prior=UniformSearchPrior(),
                value=DiffSearchValue(turn_start_snapshot=snapshot, player=player, predictor=self.predictor),
                turn_transitioner=DefaultTurnTransitioner(predictor=self.predictor),
                config=PuctConfig(
                    simulations=spec.warm_start_simulations,
                    max_depth=spec.warm_start_max_depth,
                    max_turns=1,
                    c_puct=1.5,
                    dirichlet_epsilon=0.0,
                ),
            )
            result = search.search(snapshot, player=player)
            if result.selected_action_index is None:
                break
            action = result.actions[result.selected_action_index]
            action_keys.append(_warm_action_key(snapshot, action))
            prediction = self.predictor.predict_with_replay_context(
                snapshot,
                action,
                player=player,
                execution_context=LocalExecutionContext(branch_seed=game.branch_seed(context)),
            )
            if not is_strict_prediction(prediction):
                break
            snapshot = prediction.snapshot
            if action.kind is ActionKind.TURN_END or snapshot.is_turn_ended:
                turns += 1
                started = game.start_turn(
                    snapshot,
                    player="opponent" if player == "me" else "me",
                    turn_index=context.turn_index + 1,
                )
                if started.blocked_status is not None:
                    break
                snapshot = started.snapshot
                context = started.scheduler_context
            else:
                context = ProofSchedulerContext(
                    turn_index=context.turn_index,
                    action_index=context.action_index + 1,
                    current_player=player,
                )

        return WarmStartLine(
            action_keys=tuple(action_keys),
            reached_terminal=reached_terminal,
            winner=winner,
            turns=turns,
        )


def _warm_action_key(snapshot: BattleSnapshot, action: LegalAction) -> str:
    card = snapshot.cards_by_uid.get(action.source_uid)
    base = card.base_card_id if card is not None else action.source_uid
    return f"{action.kind.value}:{base}"


def _root_report(result: ProofSearchResult, *, label: str, root_player: str) -> ProofRootReport:
    root = result.root
    executor_gap = sum(1 for node in result.nodes.values() if node.status is ProofStatus.UNKNOWN_EXECUTOR_GAP)
    return ProofRootReport(
        label=label,
        root_player=root_player,
        status=root.status.value,
        proven_action_count=len(root.proven_actions),
        refuted_action_count=len(root.refuted_actions),
        proof_number=root.proof_number,
        disproof_number=root.disproof_number,
        node_count=len(result.nodes),
        expanded_nodes=result.trace.expanded_nodes,
        terminal_nodes=result.trace.terminal_nodes,
        executor_gap_nodes=executor_gap,
        budget_cuts=result.trace.budget_cuts,
        cycle_cuts=result.trace.cycle_cuts,
        blocked_reason=root.blocked_reason,
    )


def _report_to_jsonable(report: ProofRootReport) -> dict[str, Any]:
    return {
        "label": report.label,
        "root_player": report.root_player,
        "status": report.status,
        "proven_action_count": report.proven_action_count,
        "refuted_action_count": report.refuted_action_count,
        "proof_number": _finite(report.proof_number),
        "disproof_number": _finite(report.disproof_number),
        "node_count": report.node_count,
        "expanded_nodes": report.expanded_nodes,
        "terminal_nodes": report.terminal_nodes,
        "executor_gap_nodes": report.executor_gap_nodes,
        "budget_cuts": report.budget_cuts,
        "cycle_cuts": report.cycle_cuts,
        "blocked_reason": report.blocked_reason,
    }


def _finite(value: float) -> float | str:
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return value


def _deck_digest(deck: LocalDeckList) -> str:
    payload = f"{deck.name}|{deck.class_name}|{','.join(str(cid) for cid in deck.card_ids)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
