# -*- coding: utf-8 -*-
"""卡牌「打出效果条件」进度的物化。

设计目标：条件进度是**状态转移的产物**，不是特征编码时的计算。每次 `LocalPredictor`
完成一次 action 预测后刷新一次，特征编码只读取 `BattleSnapshot.condition_progress_by_uid`。
因此 self-play 不需要为特征额外模拟或搜索，成本是 O(己方手牌+场上 × 条件数) 的整数比较。

三态规则（禁止把未知当 False）：

- `coverage == "unknown"`（模板缺失）→ `UNKNOWN`
- `coverage == "modeled"` 且无条件 → `NONE`（已确认的平卡）
- 全部条件可判定且为真 → `SATISFIED`
- 任一条件可判定为假 → `UNSATISFIED`（即使同时存在不可判定条件）
- 其余（存在不可判定条件）→ `UNKNOWN`

视角边界：只刷新己方手牌与场上。条件 subject 的 `me.` / `opponent.` 是绝对视角，
对对手卡求值会用到错误的玩家计数，因此对手卡保持无进度（特征层输出 unknown）。
"""

from __future__ import annotations

from dataclasses import replace

from decision_engine.cards.condition_plan import ConditionPlanCache
from decision_engine.execution.condition_evaluator import ConditionEvaluator
from decision_engine.state.snapshot import (
    BattleSnapshot,
    ConditionProgress,
    ConditionState,
)


class ConditionProgressProjector:
    """把条件计划物化为 snapshot 上的条件进度。"""

    def __init__(
        self,
        plans: ConditionPlanCache,
        evaluator: ConditionEvaluator | None = None,
    ):
        self._plans = plans
        self._evaluator = evaluator or ConditionEvaluator()

    def refresh(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        if not self._plans.enabled:
            return snapshot
        progress = self._compute(snapshot)
        if progress == snapshot.condition_progress_by_uid:
            return snapshot
        return replace(snapshot, condition_progress_by_uid=progress)

    def _compute(self, snapshot: BattleSnapshot) -> dict[int, ConditionProgress]:
        cards = snapshot.cards_by_uid
        result: dict[int, ConditionProgress] = {}
        for uid in (*snapshot.me.hand, *snapshot.me.field):
            card = cards.get(uid)
            if card is None:
                # snapshot 不一致：给 UNKNOWN 而不是静默跳过，避免特征层把缺卡当无条件。
                result[uid] = ConditionProgress(state=ConditionState.UNKNOWN, unknown=1, total=1)
                continue
            plan = self._plans.get(card.base_card_id)
            value_flags = {
                "has_discard_payoff": plan.has_discard_payoff,
                "has_hand_cost_reduction": plan.has_hand_cost_reduction,
            }
            if plan.coverage != "modeled":
                result[uid] = ConditionProgress(state=ConditionState.UNKNOWN, unknown=1, total=1, **value_flags)
                continue
            if not plan.keys:
                result[uid] = ConditionProgress(state=ConditionState.NONE, **value_flags)
                continue
            satisfied_groups = unsatisfied_groups = unknown_groups = 0
            for group in plan.groups:
                values = [self._evaluator.evaluate_static(snapshot, condition, card) for condition in group]
                if any(value is False for value in values):
                    unsatisfied_groups += 1
                elif any(value is None for value in values):
                    unknown_groups += 1
                else:
                    satisfied_groups += 1
            # 组间 OR：任一组全部成立就说明至少一条强化分支已可用。
            if satisfied_groups:
                state = ConditionState.SATISFIED
            elif unknown_groups:
                state = ConditionState.UNKNOWN
            else:
                state = ConditionState.UNSATISFIED
            result[uid] = ConditionProgress(
                state=state,
                satisfied=satisfied_groups,
                unsatisfied=unsatisfied_groups,
                unknown=unknown_groups,
                total=len(plan.groups),
                has_conditional_effect=True,
                **value_flags,
            )
        return result
