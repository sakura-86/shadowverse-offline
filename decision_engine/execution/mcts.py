# -*- coding: utf-8 -*-
"""Phase L8 rollback search / MCTS 骨架。

当前实现是确定性 rollback search 骨架，不做随机 rollout；后续 MCTS 可复用节点、预算和分支策略。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.simulation.predictor import PredictionCertainty, PredictionResult
from decision_engine.state.snapshot import BattleSnapshot

from .cross_turn_transition import CrossTurnTransition, CrossTurnTransitionConfig, CrossTurnTransitionResult
from .execution_context import LocalExecutionContext
from .local_predictor import LocalPredictor


ActionProvider = Callable[[BattleSnapshot], Iterable[LegalAction]]
ScoreFunction = Callable[[BattleSnapshot, PredictionCertainty], float]


class PartialBranchPolicy(Enum):
    KEEP = "keep"
    PENALIZE = "penalize"
    SKIP = "skip"


class PredictErrorPolicy(Enum):
    SKIP = "skip"
    KEEP_WITH_PENALTY = "keep_with_penalty"
    STOP = "stop"


@dataclass(frozen=True)
class RollbackSearchConfig:
    max_depth: int = 2
    beam_width: int = 8
    partial_policy: PartialBranchPolicy = PartialBranchPolicy.PENALIZE
    predict_error_policy: PredictErrorPolicy = PredictErrorPolicy.SKIP
    partial_penalty: float = 25.0
    predict_error_penalty: float = 10_000.0
    cross_turn_transition: CrossTurnTransitionConfig | None = None
    branch_seed: int | None = 0


@dataclass(frozen=True)
class RollbackSearchNode:
    snapshot: BattleSnapshot
    actions: tuple[LegalAction, ...]
    score: float
    certainty: PredictionCertainty
    unresolved_effects: tuple[str, ...] = ()
    cross_turn_transitions: int = 0
    ally_turn_index: int = 0


@dataclass(frozen=True)
class RollbackSearchTrace:
    expanded_nodes: int
    predictions: int
    predict_error_skipped: int
    partial_skipped: int
    terminal_reason: str
    cross_turn_transitions: int = 0
    max_ally_turn_index_reached: int = 0
    cross_turn_budget_skipped: int = 0
    hidden_draw_branches: int = 0
    hidden_draw_residual_branches: int = 0
    hidden_draw_branch_errors: int = 0


@dataclass(frozen=True)
class RollbackSearchResult:
    candidates: tuple[RollbackSearchNode, ...]
    best: RollbackSearchNode | None
    trace: RollbackSearchTrace


@dataclass(frozen=True)
class CrossTurnExpansion:
    predictions: tuple[PredictionResult, ...]
    transition_applied: bool = False
    budget_skipped: bool = False
    hidden_draw_branches: int = 0
    hidden_draw_residual_branches: int = 0
    hidden_draw_branch_errors: int = 0


class LocalRollbackSearch:
    def __init__(
        self,
        action_provider: ActionProvider,
        score: ScoreFunction,
        *,
        predictor: LocalPredictor | None = None,
        config: RollbackSearchConfig | None = None,
        cross_turn_transition: CrossTurnTransition | None = None,
    ):
        self.action_provider = action_provider
        self.score = score
        self.predictor = predictor or LocalPredictor()
        self.config = config or RollbackSearchConfig()
        self.cross_turn_transition = cross_turn_transition or CrossTurnTransition()

    def search(self, snapshot: BattleSnapshot) -> RollbackSearchResult:
        initial = RollbackSearchNode(
            snapshot=snapshot,
            actions=(),
            score=self.score(snapshot, PredictionCertainty.EXACT),
            certainty=PredictionCertainty.EXACT,
        )
        beam = [initial]
        all_candidates = [initial]
        expanded_nodes = 0
        predictions = 0
        predict_error_skipped = 0
        partial_skipped = 0
        cross_turn_transitions = 0
        max_ally_turn_index_reached = 0
        cross_turn_budget_skipped = 0
        hidden_draw_branches = 0
        hidden_draw_residual_branches = 0
        hidden_draw_branch_errors = 0
        terminal_reason = "max_depth_reached"

        for _depth in range(self.config.max_depth):
            next_nodes: list[RollbackSearchNode] = []
            for node in beam:
                expanded_nodes += 1
                for action in self.action_provider(node.snapshot):
                    prediction = self._predict_with_seed(node.snapshot, action, depth=_depth, node_index=expanded_nodes)
                    predictions += 1
                    if prediction.certainty == PredictionCertainty.PREDICT_ERROR:
                        if self.config.predict_error_policy == PredictErrorPolicy.STOP:
                            return RollbackSearchResult(
                                candidates=tuple(all_candidates),
                                best=max(all_candidates, key=lambda item: item.score) if all_candidates else None,
                                trace=RollbackSearchTrace(
                                    expanded_nodes=expanded_nodes,
                                    predictions=predictions,
                                    predict_error_skipped=predict_error_skipped,
                                    partial_skipped=partial_skipped,
                                    terminal_reason="predict_error",
                                    cross_turn_transitions=cross_turn_transitions,
                                    max_ally_turn_index_reached=max_ally_turn_index_reached,
                                    cross_turn_budget_skipped=cross_turn_budget_skipped,
                                    hidden_draw_branches=hidden_draw_branches,
                                    hidden_draw_residual_branches=hidden_draw_residual_branches,
                                    hidden_draw_branch_errors=hidden_draw_branch_errors,
                                ),
                            )
                        if self.config.predict_error_policy == PredictErrorPolicy.SKIP:
                            predict_error_skipped += 1
                            continue
                    expansion = self._predictions_with_cross_turn_transition(node, action, prediction)
                    if expansion.transition_applied:
                        cross_turn_transitions += 1
                    if expansion.budget_skipped:
                        cross_turn_budget_skipped += 1
                    hidden_draw_branches += expansion.hidden_draw_branches
                    hidden_draw_residual_branches += expansion.hidden_draw_residual_branches
                    hidden_draw_branch_errors += expansion.hidden_draw_branch_errors
                    for item_prediction in expansion.predictions:
                        if item_prediction.certainty == PredictionCertainty.PREDICT_ERROR:
                            if self.config.predict_error_policy == PredictErrorPolicy.STOP:
                                return RollbackSearchResult(
                                    candidates=tuple(all_candidates),
                                    best=max(all_candidates, key=lambda item: item.score) if all_candidates else None,
                                    trace=RollbackSearchTrace(
                                        expanded_nodes=expanded_nodes,
                                        predictions=predictions,
                                        predict_error_skipped=predict_error_skipped,
                                        partial_skipped=partial_skipped,
                                        terminal_reason="predict_error",
                                        cross_turn_transitions=cross_turn_transitions,
                                        max_ally_turn_index_reached=max_ally_turn_index_reached,
                                        cross_turn_budget_skipped=cross_turn_budget_skipped,
                                        hidden_draw_branches=hidden_draw_branches,
                                        hidden_draw_residual_branches=hidden_draw_residual_branches,
                                        hidden_draw_branch_errors=hidden_draw_branch_errors,
                                    ),
                                )
                            if self.config.predict_error_policy == PredictErrorPolicy.SKIP:
                                predict_error_skipped += 1
                                continue
                        if item_prediction.certainty == PredictionCertainty.PARTIAL and self.config.partial_policy == PartialBranchPolicy.SKIP:
                            partial_skipped += 1
                            continue
                        next_node = self._make_node(node, action, item_prediction, cross_turn_transition_applied=expansion.transition_applied)
                        max_ally_turn_index_reached = max(max_ally_turn_index_reached, next_node.ally_turn_index)
                        next_nodes.append(next_node)
                        all_candidates.append(next_node)
            if not next_nodes:
                terminal_reason = "no_next_candidates"
                break
            next_nodes.sort(key=lambda item: item.score, reverse=True)
            beam = next_nodes[: self.config.beam_width]
        best = max(all_candidates, key=lambda item: item.score) if all_candidates else None
        return RollbackSearchResult(
            candidates=tuple(sorted(all_candidates, key=lambda item: item.score, reverse=True)),
            best=best,
            trace=RollbackSearchTrace(
                expanded_nodes=expanded_nodes,
                predictions=predictions,
                predict_error_skipped=predict_error_skipped,
                partial_skipped=partial_skipped,
                terminal_reason=terminal_reason,
                cross_turn_transitions=cross_turn_transitions,
                max_ally_turn_index_reached=max_ally_turn_index_reached,
                cross_turn_budget_skipped=cross_turn_budget_skipped,
                hidden_draw_branches=hidden_draw_branches,
                hidden_draw_residual_branches=hidden_draw_residual_branches,
                hidden_draw_branch_errors=hidden_draw_branch_errors,
            ),
        )

    def _predictions_with_cross_turn_transition(
        self,
        node: RollbackSearchNode,
        action: LegalAction,
        prediction: PredictionResult,
    ) -> CrossTurnExpansion:
        config = self.config.cross_turn_transition
        if config is None or not config.enabled:
            return CrossTurnExpansion(predictions=(prediction,))
        if action.kind is not ActionKind.TURN_END:
            return CrossTurnExpansion(predictions=(prediction,))
        if not self._can_apply_cross_turn_transition(node, config):
            return CrossTurnExpansion(predictions=(prediction,), budget_skipped=True)
        if prediction.certainty == PredictionCertainty.PREDICT_ERROR:
            return CrossTurnExpansion(predictions=(prediction,))

        transitions = self.cross_turn_transition.apply_after_ally_turn_end_branches(prediction.snapshot, config=config)
        merged = tuple(self._merge_transition_prediction(prediction, transition) for transition in transitions)
        return CrossTurnExpansion(
            predictions=merged,
            transition_applied=bool(transitions),
            hidden_draw_branches=sum(1 for transition in transitions if "hidden_draw_branch_from_deck_pool" in transition.assumptions),
            hidden_draw_residual_branches=sum(1 for transition in transitions if "hidden_draw_residual_branch" in transition.assumptions),
            hidden_draw_branch_errors=sum(1 for transition in transitions if transition.certainty == PredictionCertainty.PREDICT_ERROR and "deck_pool_required_for_hidden_draw_branch" in transition.description),
        )

    def _can_apply_cross_turn_transition(self, node: RollbackSearchNode, config: CrossTurnTransitionConfig) -> bool:
        if node.cross_turn_transitions >= config.max_transitions_per_search_path:
            return False
        if config.max_ally_turns is not None and node.ally_turn_index + 1 >= config.max_ally_turns:
            return False
        return True

    def _merge_transition_prediction(self, prediction: PredictionResult, transition: CrossTurnTransitionResult) -> PredictionResult:
        return PredictionResult(
            snapshot=transition.snapshot,
            certainty=self._combine_certainty(prediction.certainty, transition.certainty),
            description="; ".join(part for part in (prediction.description, transition.description) if part),
            assumptions=prediction.assumptions + transition.assumptions,
            unresolved_effects=prediction.unresolved_effects + transition.unresolved_effects,
        )

    def _make_node(self, parent: RollbackSearchNode, action: LegalAction, prediction: PredictionResult, *, cross_turn_transition_applied: bool = False) -> RollbackSearchNode:
        score = self.score(prediction.snapshot, prediction.certainty)
        if prediction.certainty == PredictionCertainty.PARTIAL and self.config.partial_policy == PartialBranchPolicy.PENALIZE:
            score -= self.config.partial_penalty
        if prediction.certainty == PredictionCertainty.PREDICT_ERROR and self.config.predict_error_policy == PredictErrorPolicy.KEEP_WITH_PENALTY:
            score -= self.config.predict_error_penalty
        return RollbackSearchNode(
            snapshot=prediction.snapshot,
            actions=parent.actions + (action,),
            score=score,
            certainty=self._combine_certainty(parent.certainty, prediction.certainty),
            unresolved_effects=parent.unresolved_effects + prediction.unresolved_effects,
            cross_turn_transitions=parent.cross_turn_transitions + (1 if cross_turn_transition_applied else 0),
            ally_turn_index=parent.ally_turn_index + (1 if cross_turn_transition_applied else 0),
        )

    def _combine_certainty(self, previous: PredictionCertainty, current: PredictionCertainty) -> PredictionCertainty:
        order = (
            PredictionCertainty.PREDICT_ERROR,
            PredictionCertainty.UNKNOWN,
            PredictionCertainty.APPROX,
            PredictionCertainty.PARTIAL,
            PredictionCertainty.SPEC_ONLY,
            PredictionCertainty.EXACT,
        )
        return min((previous, current), key=lambda item: order.index(item))

    def _predict_with_seed(self, snapshot: BattleSnapshot, action: LegalAction, *, depth: int, node_index: int) -> PredictionResult:
        if self.config.branch_seed is None:
            return self.predictor.predict(snapshot, action)
        seed = self.config.branch_seed + depth * 1000 + node_index
        context = LocalExecutionContext(branch_seed=seed)
        return self.predictor.predict_with_replay_context(snapshot, action, execution_context=context)
