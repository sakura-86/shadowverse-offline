# -*- coding: utf-8 -*-
"""Strict terminal self-play primitives for controlled offline experiments.

Unlike ``SelfPlaySearchRunner``, this module never exports a partial episode.
It is deliberately an experiment boundary: strict certainty is required for both
executed actions and every explored PUCT edge contributing to a policy target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.execution.execution_context import LocalExecutionContext
from decision_engine.execution.local_match_setup import LocalDeckList, LocalMatchSetup, LocalMatchSetupConfig, draw_next_from_deck_instances
from decision_engine.execution.rule_kernel import deckout_winner_from_results
from decision_engine.execution.player_local_predictor import PlayerLocalPredictor
from decision_engine.planning.diff_value import DiffSearchValue
from decision_engine.planning.puct import PuctConfig, PuctSearch
from decision_engine.planning.search_prior import UniformSearchPrior
from decision_engine.planning.search_value import SearchValue, TerminalValue
from decision_engine.simulation.predictor import PredictionCertainty, PredictionResult
from decision_engine.training.example_builder import build_training_example

STRICT_CERTAINTY_VALUES = frozenset({"exact", "spec_only"})
TURN_DIFF_VERSION = "turn_diff_v1"


@dataclass(frozen=True)
class StrictSelfPlayConfig:
    max_actions_per_turn: int = 12
    simulations: int = 64
    max_depth: int = 7
    c_puct: float = 1.5
    branch_seed: int = 0


@dataclass(frozen=True)
class StrictSelfPlayStep:
    snapshot: Any
    player: str
    turn_index: int
    action_index: int
    actions: tuple[LegalAction, ...]
    selected_index: int
    search_result: Any
    prediction: PredictionResult


@dataclass(frozen=True)
class StrictSelfPlayEpisode:
    game_id: str
    arm: str
    matchup_id: str
    seed: int
    first_player: str
    winner: str | None
    reason: str
    steps: tuple[StrictSelfPlayStep, ...]
    diagnostics: Mapping[str, Any]

    @property
    def accepted(self) -> bool:
        return self.winner in {"me", "opponent"} and self.reason.startswith("terminal:")


ValueFactory = Callable[[Any, str], SearchValue]


class StrictSelfPlayRunner:
    """Run one full game and either accept all its examples or reject all of them."""

    def __init__(
        self,
        *,
        predictor: PlayerLocalPredictor,
        value_factory: ValueFactory,
        arm: str,
        config: StrictSelfPlayConfig | None = None,
    ) -> None:
        self.predictor = predictor
        self.value_factory = value_factory
        self.arm = arm
        self.config = config or StrictSelfPlayConfig()
        if arm not in {"baseline", "diff"}:
            raise ValueError(f"unsupported strict self-play arm: {arm}")

    def run(
        self,
        deck_a: LocalDeckList,
        deck_b: LocalDeckList,
        *,
        game_id: str,
        matchup_id: str,
        seed: int,
        first_player: str,
    ) -> StrictSelfPlayEpisode:
        setup = LocalMatchSetup().create(deck_a, deck_b, LocalMatchSetupConfig(seed=seed, first_player=first_player))
        if setup.snapshot is None:
            return self._rejected(game_id, matchup_id, seed, first_player, "match_setup_error", ())

        snapshot = setup.snapshot
        current_player = "me" if snapshot.is_ally_turn else "opponent"
        steps: list[StrictSelfPlayStep] = []
        search_counts = {"partial": 0, "predict_error": 0, "rejected_edges": 0}

        # 不设回合上限：仅在本地终局判定或执行失败时结束。
        turn_index = 0
        while True:
            turn_index += 1
            winner = _winner(snapshot)
            if winner is not None:
                return self._accepted(game_id, matchup_id, seed, first_player, winner, steps, search_counts)

            started = self._start_turn(snapshot, current_player, seed=seed, turn_index=turn_index)
            if isinstance(started, str):
                return self._rejected(game_id, matchup_id, seed, first_player, started, steps, search_counts)
            snapshot = started
            winner = _winner(snapshot)
            if winner is not None:
                return self._accepted(game_id, matchup_id, seed, first_player, winner, steps, search_counts)

            value = self.value_factory(snapshot, current_player)
            if self.arm == "diff" and not isinstance(value, DiffSearchValue):
                raise ValueError("diff arm requires DiffSearchValue")
            if self.arm != "diff" and isinstance(value, DiffSearchValue):
                raise ValueError("baseline arm must not use DiffSearchValue")
            search = self._build_search(value, seed=seed, turn_index=turn_index)

            for action_index in range(1, self.config.max_actions_per_turn + 1):
                try:
                    result = search.search(snapshot, player=current_player)
                except Exception as exc:
                    return self._rejected(game_id, matchup_id, seed, first_player, f"search_error:{type(exc).__name__}", steps, search_counts)
                diagnostics = result.diagnostics
                partial_count = int(diagnostics.get("partial_prediction_count") or 0)
                error_count = int(diagnostics.get("prediction_error_count") or 0)
                rejected_count = int(diagnostics.get("rejected_action_count") or 0)
                search_counts["partial"] += partial_count
                search_counts["predict_error"] += error_count
                search_counts["rejected_edges"] += rejected_count
                if partial_count:
                    return self._rejected(
                        game_id,
                        matchup_id,
                        seed,
                        first_player,
                        "search_partial_prediction",
                        steps,
                        {**search_counts, "search_rejection": _search_rejection_details(diagnostics)},
                    )
                if error_count or rejected_count:
                    return self._rejected(
                        game_id,
                        matchup_id,
                        seed,
                        first_player,
                        "search_predict_error",
                        steps,
                        {**search_counts, "search_rejection": _search_rejection_details(diagnostics)},
                    )
                if result.selected_action_index is None:
                    return self._rejected(game_id, matchup_id, seed, first_player, "no_search_selection", steps, search_counts)
                if not _valid_policy(result.visit_counts, result.policy_target, result.trace):
                    return self._rejected(game_id, matchup_id, seed, first_player, "invalid_policy_target", steps, search_counts)

                selected_index = result.selected_action_index
                action = result.actions[selected_index]
                prediction = self.predictor.predict_with_replay_context(
                    snapshot,
                    action,
                    player=current_player,
                    execution_context=LocalExecutionContext(branch_seed=_action_seed(seed, turn_index, action_index)),
                )
                if not _is_strict_certainty(prediction.certainty) or prediction.unresolved_effects:
                    return self._rejected(
                        game_id,
                        matchup_id,
                        seed,
                        first_player,
                        f"executed_prediction_{_certainty_value(prediction.certainty)}",
                        steps,
                        {**search_counts, "executed_rejection": _prediction_rejection_details(prediction, action=action, snapshot=snapshot)},
                    )
                steps.append(StrictSelfPlayStep(
                    snapshot=snapshot,
                    player=current_player,
                    turn_index=turn_index,
                    action_index=action_index,
                    actions=result.actions,
                    selected_index=selected_index,
                    search_result=result,
                    prediction=prediction,
                ))
                snapshot = prediction.snapshot
                winner = _winner(snapshot)
                if winner is not None:
                    return self._accepted(game_id, matchup_id, seed, first_player, winner, steps, search_counts)
                if action.kind is ActionKind.TURN_END or snapshot.is_turn_ended:
                    break
            else:
                return self._rejected(game_id, matchup_id, seed, first_player, "max_actions_per_turn", steps, search_counts)
            current_player = "opponent" if current_player == "me" else "me"

    def examples(self, episode: StrictSelfPlayEpisode) -> list[dict[str, Any]]:
        if not episode.accepted or episode.winner is None:
            return []
        examples: list[dict[str, Any]] = []
        for step in episode.steps:
            if len(step.actions) <= 1:
                continue
            value_target = 1.0 if step.player == episode.winner else -1.0
            result = step.search_result
            example = build_training_example(
                snapshot=step.snapshot,
                legal_actions=step.actions,
                selected_action_index=step.selected_index,
                visit_counts=result.visit_counts,
                sample_id=f"{episode.game_id}:turn={step.turn_index}:action={step.action_index}:player={step.player}",
                source={
                    "kind": "strict_self_play_search",
                    "game_id": episode.game_id,
                    "episode_id": episode.game_id,
                    "seed": episode.seed,
                    "matchup_id": episode.matchup_id,
                },
                trust={
                    "level": "self_play_search",
                    "accepted": True,
                    "strict_terminal": True,
                    "arm": episode.arm,
                    "value_source": result.diagnostics.get("value_source"),
                },
                perspective=step.player,
                value_target=value_target,
                generation_config={
                    "kind": "self_play_search",
                    "strict_mode": True,
                    "arm": episode.arm,
                    "turn_diff_version": TURN_DIFF_VERSION if episode.arm == "diff" else None,
                    "simulations": self.config.simulations,
                    "max_depth": self.config.max_depth,
                    "max_search_turns": 1,
                    "c_puct": self.config.c_puct,
                    "seed": episode.seed,
                    "first_player": episode.first_player,
                    "matchup_id": episode.matchup_id,
                },
                metadata={
                    "game_id": episode.game_id,
                    "winner": episode.winner,
                    "terminal_reason": episode.reason,
                    "turn_index": step.turn_index,
                    "action_index": step.action_index,
                    "search_diagnostics": dict(result.diagnostics),
                    "trace": result.trace,
                },
            )
            examples.append(example.to_jsonable())
        return examples

    def _build_search(self, value: SearchValue, *, seed: int, turn_index: int) -> PuctSearch:
        return PuctSearch(
            predictor=self.predictor,
            prior=UniformSearchPrior(),
            value=value,
            config=PuctConfig(
                simulations=self.config.simulations,
                max_depth=self.config.max_depth,
                max_turns=1,
                c_puct=self.config.c_puct,
                reject_predict_error=True,
                branch_seed=_action_seed(seed, turn_index, 0),
            ),
        )

    def _start_turn(self, snapshot: Any, player: str, *, seed: int, turn_index: int) -> Any | str:
        drawn_card, draw_snapshot = draw_next_from_deck_instances(snapshot, player)
        action = LegalAction(kind=ActionKind.TURN_START, source_uid=0)
        result = self.predictor.predict_with_replay_context(
            draw_snapshot,
            action,
            player=player,
            generated_cards=(drawn_card,) if drawn_card is not None else (),
            execution_context=LocalExecutionContext(branch_seed=_action_seed(seed, turn_index, -1)),
        )
        if not _is_strict_certainty(result.certainty) or result.unresolved_effects:
            return f"turn_start_prediction_{result.certainty.value}"
        return result.snapshot

    def _accepted(self, game_id: str, matchup_id: str, seed: int, first_player: str, winner: str, steps: Sequence[StrictSelfPlayStep], diagnostics: Mapping[str, Any]) -> StrictSelfPlayEpisode:
        return StrictSelfPlayEpisode(game_id, self.arm, matchup_id, seed, first_player, winner, f"terminal:{winner}", tuple(steps), dict(diagnostics))

    def _rejected(self, game_id: str, matchup_id: str, seed: int, first_player: str, reason: str, steps: Sequence[StrictSelfPlayStep], diagnostics: Mapping[str, Any] | None = None) -> StrictSelfPlayEpisode:
        return StrictSelfPlayEpisode(game_id, self.arm, matchup_id, seed, first_player, None, reason, tuple(steps), dict(diagnostics or {}))


def terminal_value_factory(_: Any, __: str) -> SearchValue:
    return TerminalValue()


def diff_value_factory(turn_start: Any, player: str) -> SearchValue:
    return DiffSearchValue(turn_start, player=player)


def _certainty_value(certainty: Any) -> str:
    return str(getattr(certainty, "value", certainty))


def _is_strict_certainty(certainty: Any) -> bool:
    return _certainty_value(certainty) in STRICT_CERTAINTY_VALUES


def _prediction_rejection_details(prediction: PredictionResult, *, action: LegalAction, snapshot: Any) -> dict[str, Any]:
    source = snapshot.cards_by_uid.get(action.source_uid)
    return {
        "certainty": _certainty_value(prediction.certainty),
        "action_kind": action.kind.value,
        "source_uid": action.source_uid,
        "source_base_card_id": getattr(source, "base_card_id", None),
        "description": str(prediction.description)[:240],
        "unresolved_effects": list(prediction.unresolved_effects),
    }


def _search_rejection_details(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "partial_prediction_details": list(diagnostics.get("partial_prediction_details") or ()),
        "prediction_error_details": list(diagnostics.get("prediction_error_details") or ()),
        "rejection_reasons": list(diagnostics.get("rejection_reasons") or ()),
    }


def _winner(snapshot: Any) -> str | None:
    if snapshot.me.life <= 0 and snapshot.opponent.life <= 0:
        return None
    if snapshot.me.life <= 0:
        return "opponent"
    if snapshot.opponent.life <= 0:
        return "me"
    if snapshot.me.result or snapshot.opponent.result:
        me_result = int(snapshot.me.result or 0)
        opponent_result = int(snapshot.opponent.result or 0)
        winner = deckout_winner_from_results(me_result, opponent_result)
        if winner is not None:
            return winner
    return None


def _valid_policy(visits: Sequence[int], policy: Sequence[float], trace: Mapping[str, Any]) -> bool:
    root = trace.get("root") if isinstance(trace, Mapping) else None
    if not visits or len(visits) != len(policy) or not isinstance(root, list) or len(root) != len(visits):
        return False
    if any(int(v) < 0 for v in visits) or any(float(p) < 0.0 for p in policy):
        return False
    if sum(visits) <= 0 or abs(sum(float(p) for p in policy) - 1.0) > 1e-6:
        return False
    return all(not item.get("rejected") or int(visits[index]) == 0 for index, item in enumerate(root) if isinstance(item, Mapping))


def _action_seed(seed: int, turn_index: int, action_index: int) -> int:
    return seed * 100_000 + turn_index * 100 + action_index
