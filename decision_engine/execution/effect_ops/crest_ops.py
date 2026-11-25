"""crest_ops：纹章/主战者数值类 operation 执行器。

覆盖 op：add_crest / remove_crest / modify_crest_value / modify_crest_countdown /
set_leader_max_life / modify_hand_arts_gauge
拆分方案见 docs/EFFECT_INTERPRETER_REFACTOR_PLAN.md。
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from decision_engine.cards.templates import OperationSpec
from decision_engine.state.snapshot import CrestView

from ..state_ops import SetCardField, SetPlayerField
from .base import EffectInterpreterError, OpContext, register_op

if TYPE_CHECKING:
    from decision_engine.state.snapshot import BattleSnapshot

    from ..effect_interpreter import EffectInterpretationContext, EffectInterpretationResult


@register_op("add_crest")
def op_add_crest(snapshot, operation, context, ctx: OpContext):
    card_ids = tuple(int(item) for item in operation.params.get("crest_card_ids", ()))
    if not card_ids:
        raise EffectInterpreterError("add_crest requires crest_card_ids")
    player = ctx.operation_player(operation, context)
    player_view = snapshot.me if player == "me" else snapshot.opponent
    existing_card_ids = frozenset(crest.card_id for crest in player_view.crests)
    card_ids = tuple(cid for cid in card_ids if cid not in existing_card_ids)
    if not card_ids:
        return ctx.run(snapshot, (), ())
    if len(player_view.crests) + len(card_ids) > 5:
        raise EffectInterpreterError("add_crest exceeds crest zone limit")
    start_unique_id = max((crest.unique_id for crest in player_view.crests), default=0) + 1
    additions: list[CrestView] = []
    from decision_engine.simulation.deck_pool import _default_templates_by_base_card_id
    tmpl_map = _default_templates_by_base_card_id()
    for index, card_id in enumerate(card_ids):
        countdown = 0
        if tmpl_map and card_id in tmpl_map:
            tmpl = tmpl_map[card_id]
            if tmpl.printed.countdown:
                countdown = tmpl.printed.countdown
        additions.append(CrestView(card_id=card_id, unique_id=start_unique_id + index, countdown=countdown))
    return ctx.run(snapshot, (SetPlayerField(player, "crests", (*player_view.crests, *tuple(additions))),), ())


@register_op("remove_crest")
def op_remove_crest(snapshot, operation, context, ctx: OpContext):
    player = ctx.operation_player(operation, context)
    player_view = snapshot.me if player == "me" else snapshot.opponent
    card_ids = tuple(int(item) for item in operation.params.get("crest_card_ids", ()))
    if card_ids:
        remaining = tuple(crest for crest in player_view.crests if crest.card_id not in card_ids)
    else:
        remaining = ()
    if remaining == player_view.crests:
        # 无纹章可移除 ("使所有纹章消失"但场上本无纹章, 或指定 card_id 无匹配):
        # 效果自然不生效, noop 返回 (不报错)。纹章消失是"有则移除"语义。
        return ctx.run(snapshot, (), ())
    return ctx.run(snapshot, (SetPlayerField(player, "crests", remaining),), ())


@register_op("modify_crest_value")
def op_modify_crest_value(snapshot, operation, context, ctx: OpContext):
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    player = ctx.operation_player(operation, context)
    player_view = snapshot.me if player == "me" else snapshot.opponent
    card_ids = tuple(int(item) for item in operation.params.get("crest_card_ids", ()))
    # `faith_card_id` 是信仰类 spec 的既有写法（10624122/10614122 用它指定
    # 「改哪张信仰的信仰值」）。此前 op 不认这个参数，只靠「纹章区唯一」的
    # 侥幸路径生效；一旦同时存在其他纹章就会抛
    # "modify_crest_value requires unique crest target"。
    faith_card_id = operation.params.get("faith_card_id")
    if faith_card_id is not None:
        card_ids = (*card_ids, int(faith_card_id))
    # 信仰值(faith_value)是非负累积量, 任何修改后都 clamp 到 >=0。
    # 这保证即便门控条件被绕过或多源扣减叠加, faith_value 也不会出现负数
    # (历史 bug: 古旧天书·莲妥丝回合结束无条件 -10 曾把信仰砸成 -10/-20/...)。
    def _apply(crest: CrestView) -> CrestView:
        return replace(crest, faith_value=max(0, crest.faith_value + amount))
    if card_ids:
        crests = tuple(_apply(crest) if crest.card_id in card_ids else crest for crest in player_view.crests)
    elif len(player_view.crests) == 1:
        crests = (_apply(player_view.crests[0]),)
    else:
        raise EffectInterpreterError("modify_crest_value requires unique crest target")
    return ctx.run(snapshot, (SetPlayerField(player, "crests", crests),), ())


@register_op("modify_crest_countdown")
def op_modify_crest_countdown(snapshot, operation, context, ctx: OpContext):
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    player = ctx.operation_player(operation, context)
    player_view = snapshot.me if player == "me" else snapshot.opponent
    if not player_view.crests:
        return ctx.run(snapshot, (), ())
    card_ids = tuple(int(item) for item in operation.params.get("crest_card_ids", ()))
    if card_ids:
        crests = tuple(replace(crest, countdown=crest.countdown + amount) if crest.card_id in card_ids and crest.countdown > 0 else crest for crest in player_view.crests)
    else:
        crests = tuple(replace(crest, countdown=crest.countdown + amount) if crest.countdown > 0 else crest for crest in player_view.crests)
    return ctx.run(snapshot, (SetPlayerField(player, "crests", crests),), ())


@register_op("set_leader_max_life")
def op_set_leader_max_life(snapshot, operation, context, ctx: OpContext):
    player = ctx.operation_player(operation, context)
    view = snapshot.me if player == "me" else snapshot.opponent
    raw_delta = operation.params.get("delta")
    if raw_delta is not None:
        # 「使对手的主战者的生命值的最大值-2」：相对当前上限做增量，不能写死绝对值
        # （不同局的上限不同，且同一局可能被多次触发）。
        new_max = max(1, view.max_life + int(raw_delta))
    elif operation.params.get("half_of_current"):
        # 「使对手的主战者的生命值的最大值变为一半」：运行时按目标当前上限取半，
        # 不能写死绝对值（不同局的上限不同）。取证为**向上取整**：录像里
        # 5 -> 3（replay_...751115031 st158），向下取整会得到 2。
        new_max = max(1, (view.max_life + 1) // 2)
    else:
        new_max = ctx.amount(operation, snapshot=snapshot, context=context)
    return ctx.run(snapshot, (SetPlayerField(player, "max_life", new_max), SetPlayerField(player, "life", min(view.life, new_max))), ())


@register_op("modify_hand_arts_gauge")
def op_modify_hand_arts_gauge(snapshot, operation, context, ctx: OpContext):
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    transactions = []
    for uid in ctx.target_uids(snapshot, operation, context):
        card = snapshot.cards_by_uid[uid]
        transactions.append(SetCardField(uid, "supplement_info", {**card.supplement_info, "arts_gauge": card.supplement_info.get("arts_gauge", 0) + amount}))
    return ctx.run(snapshot, tuple(transactions), ())
