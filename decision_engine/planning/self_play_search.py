# -*- coding: utf-8 -*-
"""Self-play search sampling for Phase N1."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from decision_engine.actions.model import LegalAction
from decision_engine.cards import load_card_templates_by_base_card_id
from decision_engine.execution.local_match_setup import LocalDeckList, LocalMatchSetup, LocalMatchSetupConfig, draw_next_from_deck_instances, perform_mulligan
from decision_engine.execution.execution_context import LocalExecutionContext
from decision_engine.execution.local_predictor import LocalPredictor
from decision_engine.execution.rule_kernel import CoreRuleKernel, deckout_winner_from_results
from decision_engine.execution.player_local_predictor import PlayerLocalPredictor
from decision_engine.planning.puct import DefaultTurnTransitioner, PuctConfig, PuctSearch, PuctSearchResult
from decision_engine.planning.search_value import SearchValue
from decision_engine.state.snapshot import BattleSnapshot
from decision_engine.training.example_builder import build_training_example
from decision_engine.training.mulligan_example import MulliganTrainingExample, build_mulligan_training_example


@dataclass(frozen=True)
class SelfPlaySearchConfig:
    seed_start: int = 1
    episodes: int = 1
    max_actions_per_turn: int = 12
    first_player: str = "alternate"
    simulations: int = 64
    max_depth: int = 3
    max_search_turns: int = 3
    c_puct: float = 1.5
    #: 根节点 Dirichlet 噪声权重（透传给 ``PuctConfig.dirichlet_epsilon``）。
    #:
    #: **用于生成训练标签时需要留意**：噪声会混进 ``visit_counts``，而每个决策点
    #: 只采样一次，小规模数据（几十局）无法把它平均掉。0.25 是 AlphaZero 式的
    #: self-play 探索值；如果这份数据要当教师，建议降到 0 或很小。
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.3
    # 调试：保留前 N 次 UCB simulation 的深度/根动作/叶值（PuctConfig 同名字段）。
    # 0 = 关闭，不改变搜索语义。用于定位"effective_max_depth 恒为 1"这类问题。
    simulation_trace_limit: int = 0
    allow_terminal_only_value: bool = True


@dataclass(frozen=True)
class SelfPlaySearchStep:
    episode_index: int
    turn_index: int
    action_index: int
    player: str
    snapshot: BattleSnapshot
    legal_actions: tuple[LegalAction, ...]
    search_result: PuctSearchResult
    selected_action: LegalAction
    prediction: Any
    sample_id: str
    source: Mapping[str, Any]
    trust: Mapping[str, Any]
    generation_config: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SelfPlaySearchEpisodeResult:
    episode_index: int
    seed: int
    first_player: str
    winner: str | None
    reason: str
    steps: tuple[SelfPlaySearchStep, ...]


@dataclass(frozen=True)
class SelfPlaySearchRunResult:
    episodes: tuple[SelfPlaySearchEpisodeResult, ...]
    examples: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def _default_player_local_predictor() -> PlayerLocalPredictor:
    template_path = Path(__file__).resolve().parents[1] / "data" / "public" / "card_templates.json"
    templates = load_card_templates_by_base_card_id(template_path)
    return PlayerLocalPredictor(LocalPredictor(templates_by_base_card_id=templates))


class SelfPlaySearchRunner:
    def __init__(
        self,
        *,
        predictor: PlayerLocalPredictor | None = None,
        search: PuctSearch | None = None,
        config: SelfPlaySearchConfig | None = None,
        value: SearchValue | None = None,
        prior: object | None = None,
        mulligan_selector: object | None = None,
    ):
        self.predictor = predictor or _default_player_local_predictor()
        self.config = config or SelfPlaySearchConfig()
        self.kernel = CoreRuleKernel()
        self.mulligan_selector = mulligan_selector
        self.search = search or PuctSearch(
            predictor=self.predictor,
            value=value,
            prior=prior,
            turn_transitioner=DefaultTurnTransitioner(predictor=self.predictor) if self.config.max_search_turns > 1 else None,
            config=replace(
                PuctConfig(),
                simulations=self.config.simulations,
                max_depth=self.config.max_depth,
                max_turns=self.config.max_search_turns,
                c_puct=self.config.c_puct,
                dirichlet_epsilon=self.config.dirichlet_epsilon,
                dirichlet_alpha=self.config.dirichlet_alpha,
                simulation_trace_limit=self.config.simulation_trace_limit,
            ),
        )

    def run(self, deck_a: LocalDeckList, deck_b: LocalDeckList) -> SelfPlaySearchRunResult:
        episodes: list[SelfPlaySearchEpisodeResult] = []
        pending: list[SelfPlaySearchStep] = []
        mulligan_records_by_episode: dict[int, list[dict[str, Any]]] = {}
        me_class = deck_a.class_name
        opp_class = deck_b.class_name
        for episode_index in range(self.config.episodes):
            seed = self.config.seed_start + episode_index
            first_player = _first_player(self.config.first_player, episode_index)
            # 每个 episode 开始前显式重置势函数搜索器的 turn_start 基线。
            # IncrementalAwarePuctSearch / DiffAwarePuctSearch 的 leaf value 是
            # "本回合内 Φ 增量"；`search()` 只在 player 变化时重建基线，靠 me/opponent
            # 交替隐式成立，但跨 episode 的 player 序列相同，上一局最后的基线会被
            # 下一局第一次搜索沿用（跨局污染）。这里显式重置，不依赖交替顺序。
            _notify = getattr(self.search, "notify_turn_end", None)
            if callable(_notify):
                _notify()
            setup = LocalMatchSetup().create(deck_a, deck_b, LocalMatchSetupConfig(seed=seed, first_player=first_player))
            if setup.snapshot is None:
                episodes.append(SelfPlaySearchEpisodeResult(episode_index=episode_index, seed=seed, first_player=first_player, winner=None, reason="match_setup_error", steps=()))
                continue
            snapshot, mulligan_records = self._perform_mulligan(setup.snapshot, seed=seed)
            mulligan_records_by_episode[episode_index] = mulligan_records
            try:
                episode_steps, reason, winner = self._run_episode(snapshot, episode_index=episode_index, seed=seed, first_player=first_player)
            except Exception as exc:
                episodes.append(SelfPlaySearchEpisodeResult(episode_index=episode_index, seed=seed, first_player=first_player, winner=None, reason=f"episode_error:{type(exc).__name__}:{exc}", steps=()))
                continue
            pending.extend(episode_steps)
            episodes.append(SelfPlaySearchEpisodeResult(episode_index=episode_index, seed=seed, first_player=first_player, winner=winner, reason=reason, steps=tuple(episode_steps)))

        winner_by_episode = {episode.episode_index: episode.winner for episode in episodes}
        examples = list(_build_examples(pending, winner_by_episode=winner_by_episode))

        # 生成 mulligan training examples
        for episode_index, records in mulligan_records_by_episode.items():
            winner = winner_by_episode.get(episode_index)
            for record in records:
                player = record["player"]
                value_target = None if winner is None else (1.0 if winner == player else -1.0)
                my_cls = me_class if player == "me" else opp_class
                opp_cls = opp_class if player == "me" else me_class
                try:
                    mull_example = build_mulligan_training_example(
                        snapshot=record["snapshot"],
                        player=player,
                        cards_to_change=record["cards_to_change"],
                        my_class=my_cls,
                        opponent_class=opp_cls,
                        sample_id=f"mulligan-{episode_index}-{player}",
                        source={"kind": "self_play_search", "episode_id": episode_index},
                        trust={"level": "self_play_search", "accepted": True},
                        value_target=value_target,
                        generation_config={"kind": "mulligan_self_play", "episode_index": episode_index},
                        metadata={"original_hand": list(record["original_hand"]), "episode_index": episode_index, "player": player},
                    )
                    examples.append(mull_example.to_jsonable())
                except (ValueError, KeyError):
                    pass  # 跳过无法构建的 mulligan 样本

        report = _build_report(episodes, tuple(examples), baseline_config=self._baseline_config())
        return SelfPlaySearchRunResult(episodes=tuple(episodes), examples=tuple(examples), report=report)

    def _perform_mulligan(self, snapshot: BattleSnapshot, *, seed: int) -> tuple[BattleSnapshot, list[dict[str, Any]]]:
        """在 episode 开始前执行双方换牌并记录 trace。

        Returns:
            (post_mulligan_snapshot, mulligan_records)
            mulligan_records: 每方一条记录，包含 player, original_hand, cards_to_change, snapshot。
        """
        records: list[dict[str, Any]] = []
        if self.mulligan_selector is not None:
            me_original = tuple(snapshot.me.hand)
            me_changes = self.mulligan_selector.select_mulligan(snapshot, player="me")
            snapshot = perform_mulligan(snapshot, "me", me_changes, rng=random.Random(seed * 31 + 1))
            records.append({"player": "me", "original_hand": me_original, "cards_to_change": tuple(me_changes), "snapshot": snapshot})

            opp_original = tuple(snapshot.opponent.hand)
            opp_changes = self.mulligan_selector.select_mulligan(snapshot, player="opponent")
            snapshot = perform_mulligan(snapshot, "opponent", opp_changes, rng=random.Random(seed * 31 + 2))
            records.append({"player": "opponent", "original_hand": opp_original, "cards_to_change": tuple(opp_changes), "snapshot": snapshot})
        return snapshot, records

    def _baseline_config(self) -> dict[str, Any]:
        return {
            "baseline_name": "n1_two_player_puct",
            "baseline_version": 2,
            "search_name": "puct_two_player",
            "neural_network_used": False,
            "legal_action_source": "PlayerLocalActionEnumerator",
            "predictor_source": type(self.predictor).__name__,
            "prior_source": getattr(self.search.prior, "source", "uniform"),
            "value_source": getattr(self.search.value, "source", "terminal"),
            "policy_target_source": "visit_counts",
            "requested_simulations": self.search.config.simulations,
            "requested_max_depth": self.search.config.max_depth,
            "max_search_turns": self.search.config.max_turns,
        }

    def _run_episode(self, snapshot: BattleSnapshot, *, episode_index: int, seed: int, first_player: str) -> tuple[list[SelfPlaySearchStep], str, str | None]:
        steps: list[SelfPlaySearchStep] = []
        current_player = "me" if snapshot.is_ally_turn else "opponent"
        turn_index = 0
        winner: str | None = None
        # 不设回合上限：仅在本地终局判定或执行失败时结束。
        while True:
            turn_index += 1
            terminal = _terminal(snapshot)
            if terminal is not None:
                return steps, terminal, _winner(terminal)
            snapshot = self._start_turn(snapshot, current_player)
            terminal = _terminal(snapshot)
            if terminal is not None:
                return steps, terminal, _winner(terminal)
            for action_index in range(1, self.config.max_actions_per_turn + 1):
                terminal = _terminal(snapshot)
                if terminal is not None:
                    return steps, terminal, _winner(terminal)
                search_result = self.search.search(snapshot, player=current_player)
                selected_index = search_result.selected_action_index
                if selected_index is None:
                    return steps, "no_search_selection", None
                selected_action = search_result.actions[selected_index]
                prediction = self.predictor.predict_with_replay_context(snapshot, selected_action, player=current_player, execution_context=LocalExecutionContext(branch_seed=seed * 1000 + turn_index * 100 + action_index))
                diagnostics = dict(search_result.diagnostics)
                step = SelfPlaySearchStep(
                    episode_index=episode_index,
                    turn_index=turn_index,
                    action_index=action_index,
                    player=current_player,
                    snapshot=snapshot,
                    legal_actions=search_result.actions,
                    search_result=search_result,
                    selected_action=selected_action,
                    prediction=prediction,
                    sample_id=f"self-play-search-{episode_index}-turn-{turn_index}-action-{action_index}-{current_player}",
                    source={"kind": "self_play_search", "episode_id": episode_index, "seed": seed, "first_player": first_player},
                    trust={
                        "level": "self_play_search",
                        "accepted": True,
                        "baseline": diagnostics.get("baseline_name", "n1_no_nn_puct"),
                        "search": diagnostics.get("search_name", "puct"),
                        "neural_network_used": diagnostics.get("neural_network_used", False),
                        "prior_source": diagnostics.get("prior_source", "uniform"),
                        "value_source": diagnostics.get("value_source", "terminal"),
                        "after_state_source": diagnostics.get("after_state_source", "PredictionResult.snapshot"),
                    },
                    generation_config={
                        "kind": "self_play_search",
                        "baseline_name": diagnostics.get("baseline_name", "n1_no_nn_puct"),
                        "baseline_version": diagnostics.get("baseline_version", 1),
                        "search_name": diagnostics.get("search_name", "puct"),
                        "neural_network_used": diagnostics.get("neural_network_used", False),
                        "prior_source": diagnostics.get("prior_source", "uniform"),
                        "value_source": diagnostics.get("value_source", "terminal"),
                        "predictor_source": diagnostics.get("predictor_source", type(self.predictor).__name__),
                        "legal_action_source": diagnostics.get("legal_action_source", "PlayerLocalActionEnumerator"),
                        "policy_target_source": diagnostics.get("policy_target_source", "visit_counts"),
                        "requested_simulations": diagnostics.get("requested_simulations", self.config.simulations),
                        "executed_simulations": diagnostics.get("executed_simulations", self.config.simulations),
                        "requested_max_depth": diagnostics.get("requested_max_depth", self.config.max_depth),
                        "effective_max_depth": diagnostics.get("effective_max_depth", 1),
                        "seed": seed,
                        "episode_index": episode_index,
                        "first_player": first_player,
                        "simulations": self.config.simulations,
                        "max_depth": self.config.max_depth,
                        "c_puct": self.config.c_puct,
                        "max_search_turns": self.search.config.max_turns,
                        "max_actions_per_turn": self.config.max_actions_per_turn,
                    },
                    metadata={"turn_index": turn_index, "action_index": action_index, "player": current_player, "root_value": search_result.root_value, "trace": search_result.trace, "search_diagnostics": diagnostics, "visit_counts": list(search_result.visit_counts), "policy_target": list(search_result.policy_target), "selected_action_index": search_result.selected_action_index},
                )
                steps.append(step)
                snapshot = prediction.snapshot
                if prediction.certainty.value == "predict_error":
                    return steps, "predict_error", None
                if selected_action.kind.value == "turn_end" or snapshot.is_turn_ended:
                    break
            else:
                return steps, "max_actions_per_turn", None
            current_player = "opponent" if current_player == "me" else "me"

    def _start_turn(self, snapshot: BattleSnapshot, player: str) -> BattleSnapshot:
        if player not in {"me", "opponent"}:
            raise ValueError(f"unsupported player: {player}")
        drawn_card, draw_snapshot = draw_next_from_deck_instances(snapshot, player)
        # 通过 predictor 执行 turn_start（包含 event queue effects 如谢幕曲）
        from decision_engine.actions.model import ActionKind, LegalAction
        turn_start_action = LegalAction(kind=ActionKind.TURN_START, source_uid=0)
        generated = (drawn_card,) if drawn_card is not None else ()
        result = self.predictor.predict_with_generated_cards(
            draw_snapshot, turn_start_action, player=player, generated_cards=generated
        )
        return result.snapshot


def _build_examples(steps: Sequence[SelfPlaySearchStep], *, winner_by_episode: Mapping[int, str | None]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for step in steps:
        if step.prediction.certainty.value == "predict_error":
            continue
        # 只有 1 个合法动作的步骤没有训练价值（无选择余地）
        if len(step.legal_actions) <= 1:
            continue
        winner = winner_by_episode.get(step.episode_index)
        value_target = None if winner is None else (1.0 if winner == step.player else -1.0)
        example = build_training_example(
            snapshot=step.snapshot,
            legal_actions=step.legal_actions,
            selected_action_index=step.search_result.selected_action_index,
            visit_counts=step.search_result.visit_counts,
            sample_id=step.sample_id,
            source=step.source,
            trust=step.trust,
            perspective=step.player,
            value_target=value_target,
            generation_config=step.generation_config,
            metadata=step.metadata,
        )
        examples.append(example.to_jsonable())
    return examples


def _build_report(episodes: Sequence[SelfPlaySearchEpisodeResult], examples: Sequence[dict[str, Any]], *, baseline_config: Mapping[str, Any]) -> dict[str, Any]:
    terminal_count = sum(1 for episode in episodes if episode.winner is not None)
    predict_error_count = sum(1 for episode in episodes if episode.reason == "predict_error")
    contract_summary = _contract_summary(examples)

    # 报告里的 effective_max_depth 必须是**实测**深度。
    #
    # 旧实现取 `baseline_config.get("effective_max_depth", 1)`，而
    # `_baseline_config()` 根本不产出这个键，于是恒返回默认值 1——
    # 任何 `--max-depth 10` 的实验都会在报告里显示成"实际深度 1"，
    # 把一个正常工作的深搜误读成"搜索只走一步"。
    # 实测值本来就在每个 step 的 `search_result.diagnostics` 里（puct.py:554）。
    _observed_depths: list[int] = []
    _cross_turn_total = 0
    for _ep in episodes:
        for _st in _ep.steps:
            _diag = getattr(getattr(_st, "search_result", None), "diagnostics", None) or {}
            _d = _diag.get("effective_max_depth")
            if isinstance(_d, int):
                _observed_depths.append(_d)
            _c = _diag.get("cross_turn_count")
            if isinstance(_c, int):
                _cross_turn_total += _c

    return {
        "schema": "SelfPlaySearchSamplingReport",
        "version": 1,
        "episodes_total": len(episodes),
        "episodes_terminal": terminal_count,
        "episodes_completed": len(episodes),
        "episodes_failed": sum(1 for episode in episodes if episode.winner is None and episode.reason not in {"max_actions_per_turn"}),
        "training_examples_written": len(examples),
        "predict_error_count": predict_error_count,
        "baseline_name": baseline_config.get("baseline_name", "n1_no_nn_puct"),
        "baseline_version": baseline_config.get("baseline_version", 1),
        "search_name": baseline_config.get("search_name", "puct"),
        "neural_network_used": baseline_config.get("neural_network_used", False),
        "legal_action_source": baseline_config.get("legal_action_source", "PlayerLocalActionEnumerator"),
        "predictor_source": baseline_config.get("predictor_source", "PlayerLocalPredictor"),
        "prior_source": baseline_config.get("prior_source", "uniform"),
        "value_source": baseline_config.get("value_source", "terminal"),
        "policy_target_source": baseline_config.get("policy_target_source", "visit_counts"),
        "requested_simulations": baseline_config.get("requested_simulations", 0),
        "requested_max_depth": baseline_config.get("requested_max_depth", 1),
        "effective_max_depth": max(_observed_depths) if _observed_depths else 0,
        "observed_cross_turn_count": _cross_turn_total,
        "contract_summary": contract_summary,
        "non_goals": ["not_runtime_selector", "not_neural_model", "not_heuristic_scorer", "not_protocol_fallback", "not_structured_encoder"],
    }


def _first_player(mode: str, index: int) -> str:
    if mode == "alternate":
        return "me" if index % 2 == 0 else "opponent"
    return mode


def _terminal(snapshot: BattleSnapshot) -> str | None:
    if snapshot.me.life <= 0 and snapshot.opponent.life <= 0:
        return "simultaneous_leader_death_unresolved"
    if snapshot.me.life <= 0:
        return "opponent_win_leader_damage"
    if snapshot.opponent.life <= 0:
        return "me_win_leader_damage"
    if snapshot.me.result or snapshot.opponent.result:
        return f"battle_result:me={snapshot.me.result}:opponent={snapshot.opponent.result}"
    return None


def _winner(reason: str) -> str | None:
    if reason.startswith("me_win"):
        return "me"
    if reason.startswith("opponent_win"):
        return "opponent"
    if reason.startswith("battle_result:"):
        parts = dict(part.split("=", 1) for part in reason.split(":")[1:] if "=" in part)
        me_result = int(parts.get("me", "0"))
        opponent_result = int(parts.get("opponent", "0"))
        winner = deckout_winner_from_results(me_result, opponent_result)
        if winner is not None:
            return winner
    return None


def _contract_summary(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    example_schema_counts: dict[str, int] = {}
    state_schema_counts: dict[str, int] = {}
    action_schema_counts: dict[str, int] = {}
    for example in examples:
        example_schema = str(example.get("schema") or "unknown")
        example_schema_counts[example_schema] = example_schema_counts.get(example_schema, 0) + 1
        state_schema = str(_mapping(example.get("state_features")).get("schema") or "unknown")
        state_schema_counts[state_schema] = state_schema_counts.get(state_schema, 0) + 1
        for action in _list(example.get("legal_action_features")):
            action_schema = str(_mapping(action).get("schema") or "unknown")
            action_schema_counts[action_schema] = action_schema_counts.get(action_schema, 0) + 1
    return {
        "example_schema_counts": dict(sorted(example_schema_counts.items())),
        "state_schema_counts": dict(sorted(state_schema_counts.items())),
        "action_schema_counts": dict(sorted(action_schema_counts.items())),
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []
