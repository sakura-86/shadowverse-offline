"""evolve_ops：进化/超进化 operation 执行器。

覆盖 op：evolve / super_evolve
拆分方案见 docs/EFFECT_INTERPRETER_REFACTOR_PLAN.md。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..events import EventKind, ExecutionEvent
from ..state_ops import SetCardField
from ..transactions import RollbackTransactionLog
from .base import EffectInterpreterError, OpContext, register_op

if TYPE_CHECKING:
    from decision_engine.cards.templates import OperationSpec
    from decision_engine.state.snapshot import BattleSnapshot

    from ..effect_interpreter import EffectInterpretationContext, EffectInterpretationResult


@register_op("evolve")
def op_evolve(snapshot, operation, context, ctx: OpContext):
    from ..effect_interpreter import EffectInterpretationResult
    current = snapshot
    log = RollbackTransactionLog()
    events: list[ExecutionEvent] = []
    for uid in ctx.target_uids(snapshot, operation, context):
        card = current.cards_by_uid.get(uid)
        if card is not None and card.evolve_state > 0:
            # 「本随从进化」对已经进化的目标是无操作，不是执行错误。
            continue
        owner = ctx.owner_of_field_uid(current, uid)
        if owner is None:
            raise EffectInterpreterError("evolve target must be on field")
        # 效果进化是否解除召唤失调取决于触发时点：
        # - 「进入战场瞬间 / 使用随从时 / 模式选择」触发的进化：录像中登场即可攻击；
        # - 「入场曲（fanfare）」结算中的进化：录像中 attacks_left 仍为 0。
        trigger_event = context.trigger_event
        if trigger_event is None:
            event_names = {event.name for event in context.execution_context_events}
            trigger_event = "play_card" if "play_card" in event_names else None
        grant_attack_window = trigger_event in {"play_card", "on_self_entered_field", "mode_option_resolution"}
        result = ctx.kernel.auto_evolve(current, owner, uid, grant_attack_window=grant_attack_window)
        current = log.apply_all(current, tuple(applied.forward for applied in result.transaction_log.applied)) if result.transaction_log.applied else result.snapshot
        events.extend(result.events)
    return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))


@register_op("super_evolve")
def op_super_evolve(snapshot, operation, context, ctx: OpContext):
    from ..effect_interpreter import EffectInterpretationResult
    current = snapshot
    log = RollbackTransactionLog()
    events: list[ExecutionEvent] = []
    for uid in ctx.target_uids(snapshot, operation, context):
        owner = ctx.owner_of_field_uid(current, uid)
        if owner is None:
            raise EffectInterpreterError("super_evolve target must be on field")
        card = current.cards_by_uid[uid]
        if context.execution_context.replay_mode and card.evolve_state >= 2:
            events.append(ExecutionEvent(EventKind.SOURCE_BUFFED, source_uid=uid, target_uid=uid, player=owner, payload={"attack_delta": 3, "life_delta": 3, "replay_recorded": True}))
            continue
        if uid in {generated.uid for generated in context.generated_cards} and card.attacks_left <= 0:
            attack_limit = card.attack_limit or (card.buff.attack_limit if card.buff is not None else 0) or 1
            current = log.apply_all(current, (SetCardField(uid, "attacks_left", attack_limit),))
        result = ctx.kernel.auto_super_evolve(current, owner, uid)
        current = log.apply_all(current, tuple(applied.forward for applied in result.transaction_log.applied)) if result.transaction_log.applied else result.snapshot
        events.extend(result.events)
    return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))
