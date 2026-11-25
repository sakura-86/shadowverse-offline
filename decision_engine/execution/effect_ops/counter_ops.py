"""counter_ops：计数器/融合记录类 operation 执行器。

覆盖 op：modify_card_counter / modify_source_variable / modify_play_count / record_fusion
拆分方案见 docs/EFFECT_INTERPRETER_REFACTOR_PLAN.md。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from decision_engine.cards.templates import OperationSpec

from ..events import EventKind, ExecutionEvent
from ..state_ops import (
    ModifyPlayerCounter,
    RemoveCard,
    RemoveZoneUidAt,
    SetCardField,
)
from ..transactions import SnapshotTransaction
from .base import EffectInterpreterError, OpContext, register_op

if TYPE_CHECKING:
    from decision_engine.state.snapshot import BattleSnapshot

    from ..effect_interpreter import EffectInterpretationContext, EffectInterpretationResult


@register_op("modify_play_count")
def op_modify_play_count(
    snapshot: "BattleSnapshot",
    operation: OperationSpec,
    context: "EffectInterpretationContext",
    ctx: OpContext,
) -> "EffectInterpretationResult":
    amount = ctx.interpreter._amount(operation, snapshot=snapshot, context=context)
    return ctx.run(snapshot, (ModifyPlayerCounter(context.player, "play_count", amount),), ())


@register_op("modify_card_counter")
def op_modify_card_counter(
    snapshot: "BattleSnapshot",
    operation: OperationSpec,
    context: "EffectInterpretationContext",
    ctx: OpContext,
) -> "EffectInterpretationResult":
    field = str(operation.params.get("field", ""))
    if field not in {"variable_x", "spell_boost_count", "stack", "countdown", "attack_limit"}:
        raise EffectInterpreterError(f"unsupported card counter field: {field}")
    amount = ctx.interpreter._amount(operation, snapshot=snapshot, context=context)
    if int(operation.params.get("delta_sign", 1)) < 0:
        amount = -amount
    transactions: list[SnapshotTransaction] = []
    target_uids = ctx.interpreter._target_uids(snapshot, operation, context)
    if operation.params.get("requires_target_variable_x"):
        target_uids = tuple(uid for uid in target_uids if snapshot.cards_by_uid[uid].variable_x > 0)
    for uid in target_uids:
        card = snapshot.cards_by_uid[uid]
        value = int(getattr(card, field)) + amount
        transactions.append(SetCardField(uid, field, value))
        if field == "variable_x":
            transactions.append(SetCardField(uid, "supplement_info", {**card.supplement_info, "variable_x": value}))
    return ctx.run(snapshot, tuple(transactions), ())


@register_op("modify_source_variable")
def op_modify_source_variable(
    snapshot: "BattleSnapshot",
    operation: OperationSpec,
    context: "EffectInterpretationContext",
    ctx: OpContext,
) -> "EffectInterpretationResult":
    params = {**operation.params, "field": "variable_x"}
    converted = OperationSpec(
        op="modify_card_counter",
        target_ref=operation.target_ref or "source_card",
        amount=operation.amount,
        params=params,
    )
    return op_modify_card_counter(snapshot, converted, context, ctx)


@register_op("record_fusion")
def op_record_fusion(
    snapshot: "BattleSnapshot",
    operation: OperationSpec,
    context: "EffectInterpretationContext",
    ctx: OpContext,
) -> "EffectInterpretationResult":
    if context.source_uid is None:
        raise EffectInterpreterError("record_fusion requires source_uid")
    player_view = snapshot.me if context.player == "me" else snapshot.opponent
    if context.source_uid not in player_view.hand:
        raise EffectInterpreterError("fusion source must be in hand")
    material_uids = tuple(int(uid) for uid in operation.params.get("material_uids", context.selected_targets))
    if not material_uids:
        raise EffectInterpreterError("record_fusion requires material_uids")
    hand_positions = {uid: index for index, uid in enumerate(player_view.hand)}
    transactions: list[SnapshotTransaction] = []
    for uid in sorted(material_uids, key=lambda item: hand_positions.get(item, -1), reverse=True):
        if uid == context.source_uid:
            raise EffectInterpreterError("fusion source cannot be a material")
        index = hand_positions.get(uid)
        if index is None:
            raise EffectInterpreterError("fusion material must be in hand")
        transactions.extend((RemoveZoneUidAt(context.player, "hand", index, uid), RemoveCard(uid, snapshot.cards_by_uid[uid])))
    source = snapshot.cards_by_uid[context.source_uid]
    fusion_count = source.supplement_info.get("fusion_count", 0) + len(material_uids)
    transactions.append(SetCardField(context.source_uid, "supplement_info", {**source.supplement_info, "fusion_count": fusion_count}))
    return ctx.run(snapshot, tuple(transactions), (ExecutionEvent(EventKind.FUSION, source_uid=context.source_uid, payload={"material_uids": material_uids}),))
