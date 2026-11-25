# -*- coding: utf-8 -*-
"""Player-aware wrapper for LocalPredictor."""

from __future__ import annotations

from dataclasses import replace

from decision_engine.actions.model import LegalAction
from decision_engine.actions.player_perspective import opponent_as_me, restore_opponent_perspective
from decision_engine.simulation.predictor import PredictionResult
from decision_engine.state.snapshot import BattleSnapshot, CardView

from .execution_context import LocalExecutionContext
from .local_predictor import LocalPredictor


class PlayerLocalPredictor:
    def __init__(self, predictor: LocalPredictor):
        self.predictor = predictor

    def predict(self, snapshot: BattleSnapshot, action: LegalAction, *, player: str = "me") -> PredictionResult:
        return self.predict_with_generated_cards(snapshot, action, player=player)

    def predict_with_generated_cards(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        *,
        player: str = "me",
        generated_cards: tuple[CardView, ...] = (),
    ) -> PredictionResult:
        return self.predict_with_replay_context(snapshot, action, player=player, generated_cards=generated_cards)

    def predict_with_replay_context(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        *,
        player: str = "me",
        generated_cards: tuple[CardView, ...] = (),
        execution_context: LocalExecutionContext | None = None,
    ) -> PredictionResult:
        # Extra PP 自动前置：variant="extra_pp" 的动作需要先激活 extra_pp
        if action.variant == "extra_pp":
            snapshot, action = self._apply_extra_pp_prefix(snapshot, action, player=player)

        if player == "me":
            return self.predictor.predict_with_replay_context(snapshot, action, generated_cards=generated_cards, execution_context=execution_context)
        if player != "opponent":
            raise ValueError(f"unsupported player: {player}")

        swapped = opponent_as_me(snapshot)
        prediction = self.predictor.predict_with_replay_context(swapped, action, generated_cards=generated_cards, execution_context=execution_context)
        return replace(
            prediction,
            snapshot=restore_opponent_perspective(prediction.snapshot, snapshot),
            assumptions=prediction.assumptions + ("player_view_swapped_for_opponent_prediction",),
        )

    def _apply_extra_pp_prefix(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        *,
        player: str,
    ) -> tuple[BattleSnapshot, LegalAction]:
        """对 variant='extra_pp' 的动作，先激活 extra_pp 再执行实际动作。"""
        from decision_engine.actions.model import ActionKind

        # 构造 EXTRA_PP 动作并预测
        extra_pp_action = LegalAction(kind=ActionKind.EXTRA_PP, source_uid=0)
        if player == "me":
            activated = self.predictor.predict(snapshot, extra_pp_action)
        else:
            swapped = opponent_as_me(snapshot)
            activated = self.predictor.predict(swapped, extra_pp_action)
            activated = replace(
                activated,
                snapshot=restore_opponent_perspective(activated.snapshot, snapshot),
            )

        # 去掉 variant 标记，在激活后的 snapshot 上执行原始动作
        clean_action = replace(action, variant=None)
        return activated.snapshot, clean_action
