"""resource_ops：PP/EP/SEP/墓地/大地印记等资源类 operation 执行器。

覆盖 op：recover_pp / recover_ep / recover_sep / consume_necromancy /
consume_earth_sigil / modify_earth_sigil / modify_max_pp / pay_pp / modify_graveyard
拆分方案见 docs/EFFECT_INTERPRETER_REFACTOR_PLAN.md。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..events import EventKind, ExecutionEvent
from ..state_ops import ModifyPlayerCounter, RemoveCard, RemoveZoneUid, SetCardField
from .base import EffectInterpreterError, OpContext, register_op

if TYPE_CHECKING:
    from decision_engine.cards.templates import OperationSpec
    from decision_engine.state.snapshot import BattleSnapshot

    from ..effect_interpreter import EffectInterpretationContext, EffectInterpretationResult


@register_op("recover_pp")
def op_recover_pp(snapshot, operation, context, ctx: OpContext):
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    player_view = snapshot.me if context.player == "me" else snapshot.opponent
    delta = min(amount, player_view.max_pp - player_view.pp)
    return ctx.run(snapshot, (ModifyPlayerCounter(context.player, "pp", delta),), ())


@register_op("recover_ep")
def op_recover_ep(snapshot, operation, context, ctx: OpContext):
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    player_view = snapshot.me if context.player == "me" else snapshot.opponent
    delta = min(amount, player_view.max_ep - player_view.ep)
    return ctx.run(snapshot, (ModifyPlayerCounter(context.player, "ep", delta),), ())


@register_op("recover_sep")
def op_recover_sep(snapshot, operation, context, ctx: OpContext):
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    player_view = snapshot.me if context.player == "me" else snapshot.opponent
    delta = min(amount, player_view.max_sep - player_view.sep)
    return ctx.run(snapshot, (ModifyPlayerCounter(context.player, "sep", delta),), ())


@register_op("consume_necromancy")
def op_consume_necromancy(snapshot, operation, context, ctx: OpContext):
    from ..effect_interpreter import EffectInterpretationResult
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    result = ctx.kernel.consume_necromancy(snapshot, context.player, amount)
    return EffectInterpretationResult(snapshot=result.snapshot, transaction_log=result.transaction_log, events=result.events)


@register_op("consume_earth_sigil")
def op_consume_earth_sigil(snapshot, operation, context, ctx: OpContext):
    from ..effect_interpreter import EffectInterpretationResult
    amount = ctx.amount(operation, default=1, snapshot=snapshot, context=context)
    result = ctx.kernel.consume_earth_sigil(snapshot, context.player, amount)
    return EffectInterpretationResult(snapshot=result.snapshot, transaction_log=result.transaction_log, events=result.events)


@register_op("consume_evidence")
def op_consume_evidence(snapshot, operation, context, ctx: OpContext):
    """「使自己的战场上的证据-N」：破坏 N 个证据·护符（每个算 1 点证据）。"""
    from ..effect_interpreter import EffectInterpretationResult
    amount = ctx.amount(operation, default=1, snapshot=snapshot, context=context)
    result = ctx.kernel.consume_evidence(snapshot, context.player, amount)
    return EffectInterpretationResult(snapshot=result.snapshot, transaction_log=result.transaction_log, events=result.events)


@register_op("modify_earth_sigil")
def op_modify_earth_sigil(snapshot, operation, context, ctx: OpContext):
    from ..effect_interpreter import EffectInterpretationResult
    from ..transactions import RollbackTransactionLog

    amount = ctx.amount(operation, default=1, snapshot=snapshot, context=context)
    player_view = snapshot.me if context.player == "me" else snapshot.opponent
    earth_sigil_uids = tuple(uid for uid in player_view.field if snapshot.cards_by_uid[uid].is_earth_sigil)
    if not earth_sigil_uids:
        # 「使自己的战场上的土之印+N」在场上没有土之印时**生成**一个土之印护符
        # （大地之魔片 90031210，stack=N），不是空过。
        # 证据：replay_...1279432 st11 / ...1145981 st13 / ...1392654 st10 ——
        # 10731120 的入场曲「使自己的土之印+1」之后，录像里场上多了一张卡。
        from decision_engine.execution.generated_card_materializer import default_card_resolver
        from decision_engine.simulation.deck_pool import next_synthetic_uid
        from decision_engine.state.snapshot import CardType, CardView

        if amount <= 0:
            return EffectInterpretationResult(snapshot=snapshot, transaction_log=RollbackTransactionLog(), events=())
        resolver = default_card_resolver()
        base = resolver.get_base_by_card_id(90031210)
        if base is None:
            return EffectInterpretationResult(snapshot=snapshot, transaction_log=RollbackTransactionLog(), events=())
        # replay 模式下优先用 after 帧新增的真实卡（录像里这张土之印有具体 uid，
        # 用 synthetic uid 会与录像对不上）；self-play 没有绑定，仍走 synthetic。
        new_card = None
        if context.execution_context.replay_mode:
            for candidate in context.generated_cards:
                if candidate.base_card_id == 90031210 and candidate.uid not in snapshot.cards_by_uid:
                    new_card = candidate
                    break
        if new_card is None:
            new_uid = next_synthetic_uid(snapshot)
            new_card = CardView(
                uid=new_uid,
                card_id=90031210,
                base_card_id=90031210,
                name=base.name,
                cost=int(getattr(base, "cost", 1) or 1),
                card_type=CardType.AMULET,
                is_earth_sigil=True,
                stack=int(amount),
                tribes=tuple(int(t) for t in (getattr(base, "tribes", ()) or ())),
            )
        return ctx.kernel.summon_card(snapshot, context.player, new_card)
    uid = earth_sigil_uids[0]
    card = snapshot.cards_by_uid[uid]
    next_stack = max(card.stack, 1) + amount
    if next_stack <= 0:
        return ctx.run(snapshot, (RemoveZoneUid(context.player, "field", uid), RemoveCard(uid, card)), (ExecutionEvent(EventKind.EARTH_RITE, player=context.player, source_uid=uid, amount=-amount),))
    return ctx.run(snapshot, (SetCardField(uid, "stack", next_stack),), ())


@register_op("modify_max_pp")
def op_modify_max_pp(snapshot, operation, context, ctx: OpContext):
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    # PP 最大值上限为 10 (shadowverse-rules.md: PP 上限 10)。增量后封顶, 避免
    # 溢出到 11+ 导致"若能量点最大值为10"类条件误判。
    player_view = snapshot.me if context.player == "me" else snapshot.opponent
    capped_amount = min(amount, max(0, 10 - player_view.max_pp)) if amount > 0 else amount
    return ctx.run(snapshot, (ModifyPlayerCounter(context.player, "max_pp", capped_amount),), ())


@register_op("pay_pp")
def op_pay_pp(snapshot, operation, context, ctx: OpContext):
    from ..effect_interpreter import EffectInterpretationResult
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    result = ctx.kernel.consume_pp(snapshot, context.player, amount)
    return EffectInterpretationResult(snapshot=result.snapshot, transaction_log=result.transaction_log, events=result.events)


@register_op("modify_graveyard")
def op_modify_graveyard(snapshot, operation, context, ctx: OpContext):
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    return ctx.run(snapshot, (ModifyPlayerCounter(context.player, "graveyard_count", amount),), ())
