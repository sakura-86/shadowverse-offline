"""control_ops：模式选择/多目标绑定/重复执行等控制类 operation 执行器。

覆盖 op：mode_choice_effect / modify_mode_choice_count / mode_option_effect /
bind_multi_target_selection / repeat
拆分方案见 docs/EFFECT_INTERPRETER_REFACTOR_PLAN.md。
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from decision_engine.cards.templates import OperationSpec

from ..events import ExecutionEvent
from ..transactions import RollbackTransactionLog
from .base import EffectInterpreterError, OpContext, register_op

if TYPE_CHECKING:
    from decision_engine.state.snapshot import BattleSnapshot

    from ..effect_interpreter import EffectInterpretationContext, EffectInterpretationResult


@register_op("mode_choice_effect")
def op_mode_choice_effect(snapshot, operation, context, ctx: OpContext):
    from ..effect_interpreter import EffectInterpretationResult
    selection_count = int(operation.params.get("selection_count", 0) or 0)
    if selection_count <= 0:
        raise EffectInterpreterError("mode_choice_effect requires selection_count")
    if operation.params.get("ability_resolution_mode") == "all_abilities":
        if len(context.selected_mode_options) <= 0:
            raise EffectInterpreterError("mode_choice_effect selected_mode_options count mismatch")
        return EffectInterpretationResult(snapshot=snapshot, transaction_log=RollbackTransactionLog(), events=())
    # 普通单选门控: 至少选够 selection_count 即可。爆能强化"改为发动所有能力"
    # (all_abilities, 由 is_enhance 条件 gate) 时上层传入多个 selected_mode_options
    # (如(1,2)), 此单选 gate 不应因数量超出报错。少于才是真选择不足。
    if len(context.selected_mode_options) < selection_count:
        raise EffectInterpreterError("mode_choice_effect selected_mode_options count mismatch")
    return EffectInterpretationResult(snapshot=snapshot, transaction_log=RollbackTransactionLog(), events=())


@register_op("modify_mode_choice_count")
def op_modify_mode_choice_count(snapshot, operation, context, ctx: OpContext):
    from ..effect_interpreter import EffectInterpretationResult
    if operation.target_ref not in {"self_player", "ally_player", "opponent_player", "enemy_player"}:
        raise EffectInterpreterError("modify_mode_choice_count requires player target")
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    player = ctx.operation_player(operation, context)
    event = ExecutionEvent(
        "mode_choice_rule_changed",
        source_uid=context.source_uid,
        player=player,
        payload={"amount": amount, "source": operation.params.get("source")},
    )
    legal = snapshot.legal
    if legal is None or not legal.mode_skill_map:
        return EffectInterpretationResult(snapshot=snapshot, transaction_log=RollbackTransactionLog(), events=(event,))
    next_mode_skill_map = {
        uid: replace(info, select_count=max(0, min(info.mode_count, info.select_count + amount)))
        for uid, info in legal.mode_skill_map.items()
    }
    next_legal = replace(legal, mode_skill_map=next_mode_skill_map)
    return EffectInterpretationResult(snapshot=replace(snapshot, legal=next_legal), transaction_log=RollbackTransactionLog(), events=(event,))


@register_op("mode_option_effect")
def op_mode_option_effect(snapshot, operation, context, ctx: OpContext):
    raise EffectInterpreterError("mode_option_effect requires concrete operations")


@register_op("bind_multi_target_selection")
def op_bind_multi_target_selection(snapshot, operation, context, ctx: OpContext):
    from ..effect_interpreter import EffectInterpretationResult
    if not context.selected_targets:
        raise EffectInterpreterError("bind_multi_target_selection requires selected_targets")
    return EffectInterpretationResult(snapshot=snapshot, transaction_log=RollbackTransactionLog(), events=(ExecutionEvent("targets_selected", source_uid=context.source_uid, payload={"target_uids": context.selected_targets}),))


@register_op("repeat")
def op_repeat(snapshot, operation, context, ctx: OpContext):
    from ..effect_interpreter import EffectInterpretationResult
    nested = operation.params.get("operation")
    if not isinstance(nested, dict):
        raise EffectInterpreterError("repeat requires operation")
    nested_operation = OperationSpec(
        op=str(nested.get("op", "")),
        target_ref=nested.get("target_ref"),
        amount=nested.get("amount"),
        params=dict(nested.get("params", {})) if isinstance(nested.get("params", {}), dict) else {},
        emits=tuple(nested.get("emits", ())) if isinstance(nested.get("emits", ()), (list, tuple)) else (),
    )
    count = int(operation.params.get("count", 1) or 1)
    current = snapshot
    log = RollbackTransactionLog()
    events: list[ExecutionEvent] = []
    random_offset = 0
    for _ in range(count):
        repeat_context = context
        if context.random_targets:
            repeat_context = replace(context, random_targets=context.random_targets[random_offset:])
        elif (nested_operation.target_ref or "").startswith("random_") and context.execution_context.branch_seed is not None:
            import random as _random
            rng = _random.Random(context.execution_context.branch_seed + random_offset)
            candidates = ctx.random_target_candidates(
                current, nested_operation.target_ref,
                player=context.player or "me",
                opponent=context.opponent or "opponent",
                source_uid=context.source_uid,
            )
            if candidates:
                pick = rng.choice(candidates)
                repeat_context = replace(context, random_targets=(pick,))
            else:
                break
        result = ctx.dispatch(current, nested_operation, repeat_context)
        if result.transaction_log.applied:
            current = log.apply_all(current, tuple(applied.forward for applied in result.transaction_log.applied))
        else:
            current = result.snapshot
        events.extend(result.events)
        if context.random_targets and (nested_operation.target_ref or "").startswith("random_"):
            random_offset += 1
        elif not context.random_targets and (nested_operation.target_ref or "").startswith("random_"):
            random_offset += 1
    return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))
