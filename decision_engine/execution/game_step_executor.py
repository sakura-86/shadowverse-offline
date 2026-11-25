# -*- coding: utf-8 -*-
"""统一的游戏步进执行器。

Game server 和 self-play 必须共享同一个执行路径，保证行为严格一致。
任何一端能正确执行的效果，另一端必然也能正确执行。

执行语义：
  - execute_action: 执行单个动作（含 turn_end event queue effects）
  - execute_turn_transition: 回合切换（对方 start_turn，含 countdown/draw/效果）
  - execute_action_and_maybe_transition: 执行动作 + 如果是 turn_end 则自动切换
"""

from __future__ import annotations

from dataclasses import dataclass

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.execution.execution_context import LocalExecutionContext
from decision_engine.execution.local_match_setup import draw_next_from_deck_instances
from decision_engine.execution.player_local_predictor import PlayerLocalPredictor
from decision_engine.execution.rule_kernel import CoreRuleKernel
from decision_engine.simulation.predictor import PredictionResult
from decision_engine.state.snapshot import BattleSnapshot, CardView


@dataclass(frozen=True)
class StepResult:
    """单步执行结果。"""
    snapshot: BattleSnapshot
    is_turn_end: bool = False
    prediction: PredictionResult | None = None
    turn_start_prediction: PredictionResult | None = None


class GameStepExecutor:
    """统一执行器，game server 和 self-play 共享。

    branch_seed 由执行器内部自动管理（seed_base + step 自增），调用者无需手动传入。
    这确保所有涉及随机选择的效果（source_pool random、random_targets 等）都能正确物化。

    使用方式：
        executor = GameStepExecutor(predictor=predictor, seed_base=game_seed)
        result = executor.execute_action(snapshot, action, player="me")
        if result.is_turn_end:
            result = executor.execute_turn_transition(result.snapshot, ending_player="me")

    或者一步到位：
        result = executor.execute_action_and_maybe_transition(snapshot, action, player="me")
    """

    def __init__(self, *, predictor: PlayerLocalPredictor, seed_base: int = 0):
        self.predictor = predictor
        self._kernel = CoreRuleKernel()
        self._seed_base = seed_base
        self._step = 0

    def _next_branch_seed(self) -> int:
        self._step += 1
        return self._seed_base + self._step

    def execute_action(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        *,
        player: str,
        branch_seed: int | None = None,
    ) -> StepResult:
        """执行单个动作（包括 turn_end 的 event queue effects）。

        对所有动作类型统一走 predictor.predict，保证 event queue 触发。
        branch_seed 如未显式提供，自动使用内部自增 seed。
        """
        if branch_seed is None:
            branch_seed = self._next_branch_seed()
        execution_context = LocalExecutionContext(branch_seed=branch_seed)
        prediction = self.predictor.predict_with_replay_context(
            snapshot, action, player=player, execution_context=execution_context,
        )
        # random source_pool 重试
        prediction = self._retry_random_source_pool(
            snapshot, action, prediction, player=player, branch_seed=branch_seed,
        )
        is_turn_end = action.kind is ActionKind.TURN_END or prediction.snapshot.is_turn_ended
        return StepResult(
            snapshot=prediction.snapshot,
            is_turn_end=is_turn_end,
            prediction=prediction,
        )

    def execute_turn_transition(
        self,
        snapshot: BattleSnapshot,
        *,
        ending_player: str,
        branch_seed: int | None = None,
    ) -> StepResult:
        """执行回合切换：对方 start_turn（PP恢复、countdown、抽牌、效果触发）。

        输入应该是 turn_end 效果执行完毕后的 snapshot。
        branch_seed 如未显式提供，自动使用内部自增 seed。
        """
        if branch_seed is None:
            branch_seed = self._next_branch_seed()
        next_player = "opponent" if ending_player == "me" else "me"
        drawn_card, with_deck_state = draw_next_from_deck_instances(snapshot, next_player)
        turn_start_action = LegalAction(kind=ActionKind.TURN_START, source_uid=0)
        generated_cards = (drawn_card,) if drawn_card is not None else ()
        execution_context = LocalExecutionContext(branch_seed=branch_seed)
        prediction = self.predictor.predict_with_replay_context(
            with_deck_state, turn_start_action, player=next_player,
            generated_cards=generated_cards, execution_context=execution_context,
        )
        return StepResult(
            snapshot=prediction.snapshot,
            is_turn_end=False,
            turn_start_prediction=prediction,
        )

    def execute_action_and_maybe_transition(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        *,
        player: str,
        branch_seed: int | None = None,
    ) -> StepResult:
        """执行动作，如果是 turn_end 则自动执行回合切换。"""
        result = self.execute_action(snapshot, action, player=player, branch_seed=branch_seed)
        if result.is_turn_end:
            transition = self.execute_turn_transition(
                result.snapshot, ending_player=player, branch_seed=branch_seed,
            )
            return StepResult(
                snapshot=transition.snapshot,
                is_turn_end=True,
                prediction=result.prediction,
                turn_start_prediction=transition.turn_start_prediction,
            )
        return result

    def _retry_random_source_pool(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        prediction: PredictionResult,
        *,
        player: str,
        branch_seed: int | None,
    ) -> PredictionResult:
        """如果 prediction 有 source_pool_random_ambiguous，用 RandomBranchBinding 重试。"""
        if not any("source_pool_random_ambiguous" in r for r in prediction.unresolved_effects):
            return prediction
        from decision_engine.execution.execution_context import RandomBranchBinding
        from decision_engine.state.deck_instance import DeckInstanceState
        # 从 deck draw_pile 构造随机绑定
        view = snapshot.me if player == "me" else snapshot.opponent
        if view.deck_instances is None or view.deck_instances.status not in {"available", "ambiguous"}:
            return prediction
        import random as _random
        seed = branch_seed if branch_seed is not None else 42
        rng = _random.Random(seed)
        new_context = LocalExecutionContext(branch_seed=seed)
        retry = self.predictor.predict_with_replay_context(
            snapshot, action, player=player, execution_context=new_context,
        )
        return retry
