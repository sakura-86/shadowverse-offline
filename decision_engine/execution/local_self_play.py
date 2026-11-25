# -*- coding: utf-8 -*-
"""Local two-player self-play runner."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.actions.player_local_enumerator import PlayerLocalActionEnumerator
from decision_engine.actions.player_perspective import opponent_as_me
from decision_engine.cards.templates import ExecutableEffect, OperationSpec
from decision_engine.simulation.predictor import PredictionCertainty, PredictionResult
from decision_engine.state.snapshot import BattleSnapshot

from .execution_context import LocalExecutionContext, RandomBranchBinding
from .generated_card_materializer import operation_generated_card_count, source_pool_cards
from .local_match_setup import draw_next_from_deck_instances, perform_mulligan
from .rule_kernel import deckout_winner_from_results
from .player_local_predictor import PlayerLocalPredictor


@dataclass(frozen=True)
class LocalSelfPlayConfig:
    max_actions_per_turn: int = 30
    stop_on_predict_error: bool = True
    stop_on_partial: bool = False
    allow_selector_none_end_turn: bool = True
    advance_pp_by_turn: bool = True
    expand_random_source_pool_branches: bool = True
    branch_seed: int | None = None


@dataclass(frozen=True)
class LocalSelfPlaySummary:
    source: str
    me_life: int
    opponent_life: int
    me_pp: int
    opponent_pp: int
    me_max_pp: int
    opponent_max_pp: int
    me_hand_count: int
    opponent_hand_count: int
    me_field_count: int
    opponent_field_count: int
    is_ally_turn: bool
    is_turn_ended: bool


@dataclass(frozen=True)
class LocalSelfPlayStepTrace:
    turn_index: int
    action_index: int
    player: str
    phase: str
    before: LocalSelfPlaySummary
    legal_action_count: int = 0
    selected_action: LegalAction | None = None
    selected_card_id: int | None = None
    selected_base_card_id: int | None = None
    selected_card_name: str = ""
    prediction_certainty: str | None = None
    prediction_description: str = ""
    assumptions: tuple[str, ...] = ()
    unresolved_effects: tuple[str, ...] = ()
    after: LocalSelfPlaySummary | None = None
    reason: str = ""
    selected_action_score_trace: object | None = None
    planner_top_action_traces: tuple[object, ...] = ()
    planner_all_action_traces: tuple[object, ...] = ()
    selected_action_rank: int | None = None
    selection_reason: str = ""


@dataclass(frozen=True)
class LocalSelfPlayTrace:
    steps: tuple[LocalSelfPlayStepTrace, ...] = ()
    terminal_reason: str = ""
    unresolved_counts: dict[str, int] = field(default_factory=dict)
    certainty_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MulliganTrace:
    """记录单方 mulligan 决策，用于对局结束后生成训练样本。"""
    player: str
    original_hand: tuple[int, ...]  # 换牌前的手牌 uid
    cards_to_change: tuple[int, ...]  # 被换掉的 uid
    post_mulligan_snapshot: BattleSnapshot | None = None  # 换牌后的 snapshot


@dataclass(frozen=True)
class LocalSelfPlayResult:
    final_snapshot: BattleSnapshot
    trace: LocalSelfPlayTrace
    winner: str | None = None
    reason: str = ""
    mulligan_traces: tuple[MulliganTrace, ...] = ()


class LocalActionSelector(Protocol):
    def select(self, snapshot: BattleSnapshot, actions: tuple[LegalAction, ...], *, player: str) -> LegalAction | None:
        ...


class MulliganSelector(Protocol):
    """换牌选择器：给定初始手牌的 snapshot，返回要换掉的卡牌 uid 列表。"""
    def select_mulligan(self, snapshot: BattleSnapshot, *, player: str) -> tuple[int, ...]:
        ...


class NoMulliganSelector:
    """默认不换牌：保留全部初始手牌。"""
    def select_mulligan(self, snapshot: BattleSnapshot, *, player: str) -> tuple[int, ...]:
        return ()


class StrategyMulliganSelector:
    """将 MulliganStrategy（如 GuardHavenMulliganStrategy）适配为 MulliganSelector 接口。

    strategy 需要有 decide(hand: Sequence[MulliganCard]) -> MulliganDecision 方法。
    """

    def __init__(self, strategy: object):
        if not hasattr(strategy, 'decide'):
            raise TypeError(f"{type(strategy).__name__} has no 'decide' method")
        self.strategy = strategy

    def select_mulligan(self, snapshot: BattleSnapshot, *, player: str) -> tuple[int, ...]:
        from decision_engine.strategy.mulligan import MulliganCard

        view = snapshot.me if player == "me" else snapshot.opponent
        hand: list[MulliganCard] = []
        for uid in view.hand:
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            hand.append(MulliganCard(
                unique_id=uid,
                card_id=card.card_id,
                base_card_id=card.base_card_id,
                name=card.name,
            ))
        decision = self.strategy.decide(hand)
        return tuple(decision.cards_to_change)


class NeuralMulliganSelector:
    """基于 MulliganNet 模型的换牌选择器。

    使用训练好的神经网络预测每张手牌的 keep/change 概率。
    policy logit > 0 表示该卡应该被换掉。
    """

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu", threshold: float = 0.0):
        import torch
        import sys
        from pathlib import Path as _Path

        self._device = device
        self._threshold = threshold

        # Import MulliganNet from tools
        tools_dir = str(_Path(__file__).resolve().parents[2] / "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from train_mulligan_model import MulliganNet, INPUT_DIM, encode_card_features, encode_class_onehot

        self._encode_card_features = encode_card_features
        self._encode_class_onehot = encode_class_onehot
        self._input_dim = INPUT_DIM

        self._model = MulliganNet(input_dim=INPUT_DIM)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self._model.load_state_dict(ckpt["model_state_dict"])
        else:
            self._model.load_state_dict(ckpt)
        self._model.to(device)
        self._model.eval()

    def select_mulligan(self, snapshot: BattleSnapshot, *, player: str) -> tuple[int, ...]:
        import torch

        view = snapshot.me if player == "me" else snapshot.opponent
        opp_view = snapshot.opponent if player == "me" else snapshot.me

        # Global features
        is_first = [1.0 if view.is_first_side else 0.0]
        my_class = self._encode_class_onehot("")  # class_name not available in PlayerView; use empty
        opp_class = self._encode_class_onehot("")

        # Hand card features
        hand_uids = list(view.hand)
        card_features: list[float] = []
        for i in range(4):
            if i < len(hand_uids):
                uid = hand_uids[i]
                card = snapshot.cards_by_uid.get(uid)
                if card is not None:
                    card_dict = {
                        "known": True,
                        "cost": card.cost,
                        "atk": card.atk,
                        "life": card.life,
                        "card_type": card.card_type,
                        "base_card_id": card.base_card_id,
                        "has_guard": card.has_guard if hasattr(card, 'has_guard') else False,
                        "has_storm": card.has_storm if hasattr(card, 'has_storm') else False,
                        "has_rush": card.has_rush if hasattr(card, 'has_rush') else False,
                        "has_drain": card.has_drain if hasattr(card, 'has_drain') else False,
                        "has_bane": card.has_bane if hasattr(card, 'has_bane') else False,
                        "has_ambush": card.has_ambush if hasattr(card, 'has_ambush') else False,
                        "has_activation": card.has_activation if hasattr(card, 'has_activation') else False,
                        "has_last_words": card.has_last_words if hasattr(card, 'has_last_words') else False,
                        "has_spell_boost": card.has_spell_boost if hasattr(card, 'has_spell_boost') else False,
                    }
                    card_features.extend(self._encode_card_features(card_dict))
                else:
                    card_features.extend([0.0] * 18)
            else:
                card_features.extend([0.0] * 18)

        input_vec = is_first + my_class + opp_class + card_features
        x = torch.tensor([input_vec], dtype=torch.float32, device=self._device)

        with torch.no_grad():
            policy_logits, _ = self._model(x)

        # logit > threshold → change
        changes: list[int] = []
        for i in range(min(4, len(hand_uids))):
            if policy_logits[0, i].item() > self._threshold:
                changes.append(hand_uids[i])

        return tuple(changes)


@dataclass(frozen=True)
class _SelectorTraceSnapshot:
    selected_action_score_trace: object | None = None
    planner_top_action_traces: tuple[object, ...] = ()
    planner_all_action_traces: tuple[object, ...] = ()
    selected_action_rank: int | None = None
    selection_reason: str = ""


class EndTurnSelector:
    def select(self, _snapshot: BattleSnapshot, actions: tuple[LegalAction, ...], *, player: str) -> LegalAction | None:
        for action in actions:
            if action.kind is ActionKind.TURN_END:
                return action
        return None


class LocalSelfPlayRunner:
    def __init__(
        self,
        *,
        enumerator: PlayerLocalActionEnumerator | None = None,
        predictor: PlayerLocalPredictor | None = None,
        config: LocalSelfPlayConfig | None = None,
    ):
        self.enumerator = enumerator or PlayerLocalActionEnumerator()
        if predictor is None:
            raise ValueError("predictor is required — use PlayerLocalPredictor(LocalPredictor(templates_by_base_card_id=...))") 
        self.predictor = predictor
        self.config = config or LocalSelfPlayConfig()

    def run(
        self,
        initial_snapshot: BattleSnapshot,
        *,
        me_selector: LocalActionSelector,
        opponent_selector: LocalActionSelector,
        me_mulligan: MulliganSelector | None = None,
        opponent_mulligan: MulliganSelector | None = None,
    ) -> LocalSelfPlayResult:
        snapshot = initial_snapshot

        # --- Mulligan 阶段 ---
        me_mull = me_mulligan or NoMulliganSelector()
        opp_mull = opponent_mulligan or NoMulliganSelector()

        # 记录换牌前的原始手牌
        me_original_hand = tuple(snapshot.me.hand)
        me_changes = me_mull.select_mulligan(snapshot, player="me")
        snapshot = perform_mulligan(snapshot, "me", me_changes, rng=random.Random(id(snapshot) ^ 0xDEAD))
        me_mulligan_trace = MulliganTrace(
            player="me",
            original_hand=me_original_hand,
            cards_to_change=tuple(me_changes),
            post_mulligan_snapshot=snapshot,
        )

        opp_original_hand = tuple(snapshot.opponent.hand)
        opp_changes = opp_mull.select_mulligan(snapshot, player="opponent")
        snapshot = perform_mulligan(snapshot, "opponent", opp_changes, rng=random.Random(id(snapshot) ^ 0xBEEF))
        opp_mulligan_trace = MulliganTrace(
            player="opponent",
            original_hand=opp_original_hand,
            cards_to_change=tuple(opp_changes),
            post_mulligan_snapshot=snapshot,
        )
        mulligan_traces = (me_mulligan_trace, opp_mulligan_trace)

        current_player = "me" if snapshot.is_ally_turn else "opponent"
        traces: list[LocalSelfPlayStepTrace] = []
        unresolved_counts: dict[str, int] = {}
        certainty_counts: dict[str, int] = {}
        completed_turns = 0

        # 不设回合上限：对局只在本地终局判定（主战者生命归零 / deckout 结果码）
        # 或显式执行失败时结束。牌库抽空由 CoreRuleKernel 写入结果码后自然终止。
        while True:
            turn_index = completed_turns + 1
            start_prediction = self._start_turn(snapshot, current_player, turn_index=turn_index)
            traces.append(self._trace_prediction(
                snapshot,
                start_prediction,
                turn_index=turn_index,
                action_index=0,
                player=current_player,
                phase="turn_start",
                action=LegalAction(kind=ActionKind.TURN_START, source_uid=0),
                legal_action_count=0,
            ))

            self._accumulate(start_prediction, unresolved_counts, certainty_counts)
            snapshot = start_prediction.snapshot
            terminal = self._terminal(snapshot)
            if terminal is not None:
                return self._result(snapshot, traces, unresolved_counts, certainty_counts, terminal, mulligan_traces)
            if self._should_stop_for_prediction(start_prediction):
                return self._result(snapshot, traces, unresolved_counts, certainty_counts, start_prediction.description or start_prediction.certainty.value, mulligan_traces)

            for action_index in range(1, self.config.max_actions_per_turn + 1):
                terminal = self._terminal(snapshot)
                if terminal is not None:
                    return self._result(snapshot, traces, unresolved_counts, certainty_counts, terminal, mulligan_traces)
                actions = tuple(self.enumerator.enumerate(snapshot, player=current_player))
                if not actions:
                    traces.append(LocalSelfPlayStepTrace(
                        turn_index=turn_index,
                        action_index=action_index,
                        player=current_player,
                        phase="action",
                        before=summarize_local_self_play(snapshot),
                        legal_action_count=0,
                        reason="no_local_legal_actions",
                    ))
                    return self._result(snapshot, traces, unresolved_counts, certainty_counts, "no_local_legal_actions", mulligan_traces)

                selector = me_selector if current_player == "me" else opponent_selector
                action = selector.select(snapshot, actions, player=current_player)
                if action is None:
                    if not self.config.allow_selector_none_end_turn:
                        return self._result(snapshot, traces, unresolved_counts, certainty_counts, "selector_returned_none", mulligan_traces)
                    action = self._end_turn_action(actions)
                    if action is None:
                        return self._result(snapshot, traces, unresolved_counts, certainty_counts, "selector_none_without_end_turn", mulligan_traces)

                prediction = self.predictor.predict_with_replay_context(snapshot, action, player=current_player, execution_context=LocalExecutionContext(branch_seed=self._branch_seed(turn_index, action_index)))
                prediction = self._retry_with_random_source_pool_branch(snapshot, action, prediction, player=current_player, turn_index=turn_index, action_index=action_index)
                traces.append(self._trace_prediction(
                    snapshot,
                    prediction,
                    turn_index=turn_index,
                    action_index=action_index,
                    player=current_player,
                    phase="action",
                    action=action,
                    legal_action_count=len(actions),
                    selector=selector,
                ))
                self._accumulate(prediction, unresolved_counts, certainty_counts)
                snapshot = prediction.snapshot
                if self._should_stop_for_prediction(prediction):
                    return self._result(snapshot, traces, unresolved_counts, certainty_counts, prediction.description or prediction.certainty.value, mulligan_traces)
                if action.kind is ActionKind.TURN_END or snapshot.is_turn_ended:
                    break
            else:
                return self._result(snapshot, traces, unresolved_counts, certainty_counts, "max_actions_per_turn", mulligan_traces)

            completed_turns += 1
            current_player = "opponent" if current_player == "me" else "me"

    def _start_turn(self, snapshot: BattleSnapshot, player: str, *, turn_index: int = 0) -> PredictionResult:
        """回合开始：统一走 GameStepExecutor.execute_turn_transition 的等价路径。

        直接用 predictor 处理 TURN_START action（内部包含 kernel.start_turn + event queue）。
        """
        drawn_card, with_deck_state = draw_next_from_deck_instances(snapshot, player)
        action = LegalAction(kind=ActionKind.TURN_START, source_uid=0)
        generated_cards = (drawn_card,) if drawn_card is not None else ()
        return self.predictor.predict_with_replay_context(with_deck_state, action, player=player, generated_cards=generated_cards, execution_context=LocalExecutionContext(branch_seed=self._branch_seed(turn_index, 0)))

    def _should_stop_for_prediction(self, prediction: PredictionResult) -> bool:
        if prediction.certainty is PredictionCertainty.PREDICT_ERROR and self.config.stop_on_predict_error:
            return True
        return prediction.certainty is PredictionCertainty.PARTIAL and self.config.stop_on_partial

    def _retry_with_random_source_pool_branch(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        prediction: PredictionResult,
        *,
        player: str,
        turn_index: int,
        action_index: int,
    ) -> PredictionResult:
        if not self.config.expand_random_source_pool_branches:
            return prediction
        if prediction.certainty is not PredictionCertainty.PARTIAL:
            return prediction
        if not any(item == "generated_cards_not_materializable:source_pool_random_ambiguous" for item in prediction.unresolved_effects):
            return prediction
        binding = self._random_source_pool_branch_binding(snapshot, action, player=player, turn_index=turn_index, action_index=action_index)
        if binding is None:
            # 显式上报能力缺失：无法构造 source_pool 随机分支（牌组实例不可用或模板缺失）
            return PredictionResult(
                snapshot=prediction.snapshot,
                certainty=PredictionCertainty.PREDICT_ERROR,
                description="source_pool_random_branch_binding_failed: deck_instances unavailable or template missing",
                assumptions=prediction.assumptions,
                unresolved_effects=(*prediction.unresolved_effects, "source_pool_branch_binding_failed"),
            )
        context = LocalExecutionContext(random_branch_bindings=(binding,), branch_seed=self._branch_seed(turn_index, action_index))
        retried = self.predictor.predict_with_replay_context(snapshot, action, player=player, execution_context=context)
        if retried.certainty is PredictionCertainty.PARTIAL and retried.unresolved_effects == prediction.unresolved_effects:
            return prediction
        return retried

    def _random_source_pool_branch_binding(self, snapshot: BattleSnapshot, action: LegalAction, *, player: str, turn_index: int, action_index: int) -> RandomBranchBinding | None:
        predictor = getattr(self.predictor, "predictor", None)
        templates = getattr(predictor, "templates_by_base_card_id", {}) if predictor is not None else {}
        perspective_snapshot = opponent_as_me(snapshot) if player == "opponent" else snapshot
        card = perspective_snapshot.cards_by_uid.get(action.source_uid)
        if card is None:
            return None
        template = templates.get(card.base_card_id)
        if template is None:
            return None
        for effect in template.executable_effects:
            if effect.trigger.event != "fanfare":
                continue
            for operation_index, operation in enumerate(effect.operations):
                binding = self._branch_binding_for_operation(perspective_snapshot, effect, operation, operation_index, turn_index=turn_index, action_index=action_index)
                if binding is not None:
                    return binding
        return None

    def _branch_binding_for_operation(
        self,
        snapshot: BattleSnapshot,
        effect: ExecutableEffect,
        operation: OperationSpec,
        operation_index: int,
        *,
        turn_index: int,
        action_index: int,
    ) -> RandomBranchBinding | None:
        source_pool = operation.params.get("source_pool")
        if not isinstance(source_pool, dict) or source_pool.get("selection") != "random":
            return None
        count = operation_generated_card_count(operation)
        if count <= 0:
            return None
        candidates = source_pool_cards(snapshot, source_pool)
        if len(candidates) < count:
            return None
        seed = self._branch_seed(turn_index, action_index)
        cards = tuple(random.Random(seed).sample(list(candidates), count))
        assumption = f"random_branch:source_pool:seed={seed}:effect={effect.effect_id}:operation={operation_index}"
        return RandomBranchBinding(source="source_pool", effect_id=effect.effect_id, operation_index=operation_index, cards=cards, assumption=assumption)

    def _branch_seed(self, turn_index: int, action_index: int) -> int:
        base = self.config.branch_seed if self.config.branch_seed is not None else 0
        return base + turn_index * 1000 + action_index

    def _trace_prediction(
        self,
        before: BattleSnapshot,
        prediction: PredictionResult,
        *,
        turn_index: int,
        action_index: int,
        player: str,
        phase: str,
        action: LegalAction,
        legal_action_count: int,
        selector: object | None = None,
    ) -> LocalSelfPlayStepTrace:
        selected_card = before.cards_by_uid.get(action.source_uid)
        selector_trace = self._selector_trace(selector)
        return LocalSelfPlayStepTrace(
            turn_index=turn_index,
            action_index=action_index,
            player=player,
            phase=phase,
            before=summarize_local_self_play(before),
            legal_action_count=legal_action_count,
            selected_action=action,
            selected_card_id=selected_card.card_id if selected_card is not None else None,
            selected_base_card_id=selected_card.base_card_id if selected_card is not None else None,
            selected_card_name=selected_card.name if selected_card is not None else "",
            prediction_certainty=prediction.certainty.value,
            prediction_description=prediction.description,
            assumptions=prediction.assumptions,
            unresolved_effects=prediction.unresolved_effects,
            after=summarize_local_self_play(prediction.snapshot),
            selected_action_score_trace=selector_trace.selected_action_score_trace,
            planner_top_action_traces=selector_trace.planner_top_action_traces,
            planner_all_action_traces=selector_trace.planner_all_action_traces,
            selected_action_rank=selector_trace.selected_action_rank,
            selection_reason=selector_trace.selection_reason,
        )

    def _selector_trace(self, selector: object | None) -> _SelectorTraceSnapshot:
        source = getattr(selector, "selector", selector)
        selected = getattr(source, "last_selected_action_trace", None)
        plan = getattr(source, "last_plan_trace", None)
        top_actions: tuple[object, ...] = ()
        all_actions: tuple[object, ...] = ()
        depths = getattr(plan, "depths", ()) if plan is not None else ()
        if depths:
            depth = depths[0]
            top_actions = tuple(getattr(depth, "top_actions", ()) or ())
            all_actions = tuple(getattr(depth, "all_actions", ()) or ())
        return _SelectorTraceSnapshot(
            selected_action_score_trace=selected,
            planner_top_action_traces=top_actions,
            planner_all_action_traces=all_actions,
            selected_action_rank=getattr(selected, "selection_rank", None),
            selection_reason=str(getattr(source, "last_selection_reason", "") or ""),
        )

    def _accumulate(self, prediction: PredictionResult, unresolved_counts: dict[str, int], certainty_counts: dict[str, int]) -> None:
        certainty_counts[prediction.certainty.value] = certainty_counts.get(prediction.certainty.value, 0) + 1
        for item in prediction.unresolved_effects:
            unresolved_counts[item] = unresolved_counts.get(item, 0) + 1

    def _terminal(self, snapshot: BattleSnapshot) -> str | None:
        if snapshot.me.life <= 0 and snapshot.opponent.life <= 0:
            return "simultaneous_leader_death_unresolved"
        if snapshot.me.life <= 0:
            return "opponent_win_leader_damage"
        if snapshot.opponent.life <= 0:
            return "me_win_leader_damage"
        if snapshot.me.result or snapshot.opponent.result:
            return f"battle_result:me={snapshot.me.result}:opponent={snapshot.opponent.result}"
        return None

    def _winner(self, reason: str) -> str | None:
        if reason.startswith("me_win"):
            return "me"
        if reason.startswith("opponent_win"):
            return "opponent"
        if reason.startswith("battle_result:"):
            parts = dict(part.split("=", 1) for part in reason.split(":")[1:] if "=" in part)
            return deckout_winner_from_results(int(parts.get("me", "0")), int(parts.get("opponent", "0")))
        return None

    def _result(
        self,
        snapshot: BattleSnapshot,
        traces: list[LocalSelfPlayStepTrace],
        unresolved_counts: dict[str, int],
        certainty_counts: dict[str, int],
        reason: str,
        mulligan_traces: tuple[MulliganTrace, ...] = (),
    ) -> LocalSelfPlayResult:
        trace = LocalSelfPlayTrace(
            steps=tuple(traces),
            terminal_reason=reason,
            unresolved_counts=dict(sorted(unresolved_counts.items())),
            certainty_counts=dict(sorted(certainty_counts.items())),
        )
        return LocalSelfPlayResult(final_snapshot=snapshot, trace=trace, winner=self._winner(reason), reason=reason, mulligan_traces=mulligan_traces)

    def _end_turn_action(self, actions: tuple[LegalAction, ...]) -> LegalAction | None:
        for action in actions:
            if action.kind is ActionKind.TURN_END:
                return action
        return None


def summarize_local_self_play(snapshot: BattleSnapshot) -> LocalSelfPlaySummary:
    return LocalSelfPlaySummary(
        source=snapshot.source,
        me_life=snapshot.me.life,
        opponent_life=snapshot.opponent.life,
        me_pp=snapshot.me.pp,
        opponent_pp=snapshot.opponent.pp,
        me_max_pp=snapshot.me.max_pp,
        opponent_max_pp=snapshot.opponent.max_pp,
        me_hand_count=len(snapshot.me.hand),
        opponent_hand_count=len(snapshot.opponent.hand),
        me_field_count=len(snapshot.me.field),
        opponent_field_count=len(snapshot.opponent.field),
        is_ally_turn=snapshot.is_ally_turn,
        is_turn_ended=snapshot.is_turn_ended,
    )
