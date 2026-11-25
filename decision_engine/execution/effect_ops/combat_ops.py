"""combat_ops：伤害/治疗/破坏/放逐/减伤类 operation 执行器。

覆盖 op：damage / distributed_damage / heal / destroy / banish /
destruction_immunity / damage_limit
拆分方案见 docs/EFFECT_INTERPRETER_REFACTOR_PLAN.md。
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..damage_reduction import effective_damage_to_follower
from ..destruction import destroyed_card_transactions, vanish_on_leave_event, vanishes_on_leave
from ..events import EventKind, ExecutionEvent
from decision_engine.state.snapshot import CardIdCount
from ..state_ops import (
    ModifyCardCounter,
    ModifyPlayerCounter,
    RemoveCard,
    RemoveZoneUid,
    SetCardField,
    SetPlayerField,
)
from ..transactions import RollbackTransactionLog, SnapshotTransaction
from .base import OpContext, register_op

if TYPE_CHECKING:
    from decision_engine.state.snapshot import CardView


def _leaving_payload(card: "CardView") -> dict[str, object]:
    """伤害致死时的离场卡 payload，与 CoreRuleKernel._leaving_card_payload 同口径。"""
    payload: dict[str, object] = {
        "base_card_id": card.base_card_id,
        "card_id": card.card_id,
        "name": card.name,
        "has_guard": card.has_guard,
        "card_type": card.card_type.value,
    }
    if card.supplement_info:
        payload["supplement_info"] = dict(card.supplement_info)
    return payload


def _destroyed_by_damage_events(card: "CardView", context, owner: str) -> tuple[ExecutionEvent, ...]:
    """伤害致死产出的破坏事件组。

    与 CoreRuleKernel.destroy_field_card 保持一致：破坏 → 进墓地 → 谢幕曲。
    谢幕曲在产生端显式发出，使事件队列成为唯一消费点；不能留给
    LocalPredictor 事后按场地差分反推（dispatch 内部致死时 before 帧已看不到该卡）。
    """
    events = []
    if vanishes_on_leave(card):
        # 「离场时，使本随从消失」：伤害致死同样退场为消失（BANISH），
        # 由 on_self_card_left_field 订阅者响应，而不是 destroy + 谢幕曲。
        return (vanish_on_leave_event(owner, card, _leaving_payload(card)),)
    events.append(
        ExecutionEvent(
            EventKind.DESTROYED_BY_DAMAGE,
            source_uid=context.source_uid,
            target_uid=card.uid,
            player=owner,
            payload={"leaving_card": _leaving_payload(card)},
        )
    )
    if card.has_last_words:
        events.append(
            ExecutionEvent(
                EventKind.LAST_WORDS,
                player=owner,
                source_uid=card.uid,
                payload={"leaving_card": _leaving_payload(card)},
            )
        )
    return tuple(events)


def _leader_damage_transactions(snapshot, side: str, amount: int) -> tuple[SnapshotTransaction, ...]:
    """对主战者造成伤害，并在生命归零时判定胜负。

    真实规则：任一方生命归零即对局结束，同一步里排队的后续效果（另一侧随从的
    谢幕曲、纹章触发）都不再发生。此前只有战斗打脸（`CoreRuleKernel.attack_leader`）
    写 `result`，效果伤害致死不写，导致对局结束后仍在结算剩余效果
    （replay_2882303761750358211 step78：致命谢幕曲之后另一侧谢幕曲还多打 2 点）。
    """
    view = snapshot.me if side == "me" else snapshot.opponent
    transactions: list[SnapshotTransaction] = [ModifyPlayerCounter(side, "life", -amount)]
    if view.life - amount <= 0 and int(getattr(view, "result", 0) or 0) == 0:
        winner = "opponent" if side == "me" else "me"
        transactions.extend((SetPlayerField(winner, "result", 101), SetPlayerField(side, "result", 102)))
    return tuple(transactions)


@register_op("damage")
def op_damage(snapshot, operation, context, ctx: OpContext):
    from ..effect_interpreter import EffectInterpretationResult
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    target_ref = operation.target_ref or ""
    if target_ref in {"enemy_leader", "opponent_leader", "all_enemy_leaders"}:
        return ctx.run(snapshot, _leader_damage_transactions(snapshot, context.opponent, amount), (ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, player=context.opponent, payload={"amount": amount}),))
    if target_ref in {"self_leader", "ally_leader", "all_self_leaders"}:
        return ctx.run(snapshot, _leader_damage_transactions(snapshot, context.player, amount), (ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, player=context.player, payload={"amount": amount}),))
    if target_ref == "all_leaders":
        return ctx.run(
            snapshot,
            _leader_damage_transactions(snapshot, context.player, amount)
            + _leader_damage_transactions(snapshot, context.opponent, amount),
            (
                ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, player=context.player, payload={"amount": amount}),
                ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, player=context.opponent, payload={"amount": amount}),
            ),
        )
    if target_ref == "selected_enemy_follower_or_leader" and context.selected_targets in {(0,), (1,)}:
        # 「选择对手的 1 个随从**或对手的主战者**」选了主战者：协议用 0/1 表示主战者
        # （随从 uid 不会这么小）。90021310 黄金短剑…2993409 st169：录像打主战者 1 点，
        # 本地把 (1,) 当随从去场地里找、找不到就整条 noop。
        return ctx.run(
            snapshot,
            _leader_damage_transactions(snapshot, context.opponent, amount),
            (ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, player=context.opponent, payload={"amount": amount}),),
        )
    current = snapshot
    log = RollbackTransactionLog()
    events: list[ExecutionEvent] = []
    _initial_card_uids = frozenset(snapshot.cards_by_uid) | frozenset(snapshot.me.field) | frozenset(snapshot.opponent.field)
    if operation.params.get("distribution"):
        # 「对对手的战场上的所有随从和对手的主战者**分配** X 点伤害」
        # （10364122 纹章：绝望的显现·玛温）：target_ref="target" 在 turn_end 这类
        # 无 action 目标的场景下解析为空，整个效果会被静默丢弃。
        #
        # 分配**不是随机**，而是按入场顺序依次装满：
        #   1. 先把 X 点全给最先入场的随从（打到生命归零为止）
        #   2. 剩下的依次给后面入场的随从
        #   3. 全场随从都吃完还有剩余，才溢出到主战者
        # `snapshot.opponent.field` 就是入场顺序（槽位顺序），直接顺序消费即可。
        remaining = amount
        # 破坏记账与事件：分配伤害同样会造成「破坏」，必须与 `op_distributed_damage`
        # 同口径——移除场地 / 墓地 +1 / 追加 destroyed_card_ids / 发
        # destroyed_by_damage 事件。少任何一项都会让谢幕曲与「本局破坏数」类条件
        # 看不到这次破坏（replay_...0006 st65：本地杀掉 4 只但一条都没记账）。
        destroyed_cards: dict[str, list[CardView]] = {"me": [], "opponent": []}
        for uid in snapshot.opponent.field:
            if remaining <= 0:
                break
            card = current.cards_by_uid.get(uid)
            if card is None or card.life <= 0:
                continue
            damage = min(remaining, card.life)
            remaining -= damage
            if card.has_temp_shield:
                current = log.apply_all(current, (SetCardField(uid, "has_temp_shield", False),))
                events.append(ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, target_uid=uid, payload={"amount": 0, "blocked_amount": damage}))
                continue
            current = log.apply_all(current, (ModifyCardCounter(uid, "life", -damage),))
            events.append(ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, target_uid=uid, payload={"amount": damage}))
            if card.life - damage <= 0:
                owner = ctx.owner_of_field_uid(current, uid)
                if owner is not None:
                    destroyed_card = replace(card, life=card.life - damage)
                    destroyed_cards[owner].append(destroyed_card)
                    current = log.apply_all(current, (
                        RemoveZoneUid(owner, "field", uid),
                        ModifyPlayerCounter(owner, "graveyard_count", 1),
                        RemoveCard(uid, destroyed_card),
                    ))
                    events.extend(_destroyed_by_damage_events(destroyed_card, context, owner))
        for owner, cards in destroyed_cards.items():
            if not cards:
                continue
            view = snapshot.me if owner == "me" else snapshot.opponent
            current = log.apply_all(current, (
                SetPlayerField(
                    owner,
                    "destroyed_card_ids",
                    (*view.destroyed_card_ids, *(CardIdCount(card_id=card.card_id, count=card.style_id) for card in cards)),
                ),
            ))
        if remaining > 0:
            current = log.apply_all(current, _leader_damage_transactions(current, context.opponent, remaining))
            events.append(ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, player=context.opponent, payload={"amount": remaining}))
        return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))
    for uid in ctx.target_uids(snapshot, operation, context):
        transactions: list[SnapshotTransaction] = []
        if target_ref in {"selected_target", "selected_enemy_follower_or_leader", "random_enemy_follower_or_leader", "random_enemy_leader"}:
            if uid not in _initial_card_uids:
                step_events = (ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, player=context.opponent, payload={"amount": amount}),)
                # uid 0 = 主战者（`random_enemy_follower_or_leader` 的 leader 目标）。
                # 主战者生命值下限是 0：协议 after 帧不会给出负数 life。
                # 其它 leader 伤害路径仍用增量 ModifyPlayerCounter（改变它对
                # 全部 replay 生效，会引入一批 turn_end 回归），这里只在
                # 「随机目标打到主战者」这条路径 clamp。
                leader_view = current.me if context.opponent == "me" else current.opponent
                life_after = int(leader_view.life) - amount
                if life_after <= 0:
                    transactions_leader: list[SnapshotTransaction] = [SetPlayerField(context.opponent, "life", 0)]
                    if int(getattr(leader_view, "result", 0) or 0) == 0:
                        winner = "me" if context.opponent == "opponent" else "opponent"
                        transactions_leader.extend((SetPlayerField(winner, "result", 101), SetPlayerField(context.opponent, "result", 102)))
                    current = log.apply_all(current, tuple(transactions_leader))
                else:
                    current = log.apply_all(current, (ModifyPlayerCounter(context.opponent, "life", -amount),))
                events.extend(step_events)
                continue
        card = current.cards_by_uid.get(uid)
        if card is None or uid not in current.me.field and uid not in current.opponent.field:
            continue
        owner = ctx.owner_of_field_uid(current, uid)
        effective_amount = effective_damage_to_follower(current, card, owner or "", amount)
        if card.has_temp_shield:
            transactions.append(SetCardField(uid, "has_temp_shield", False))
            step_events = [ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, target_uid=uid, payload={"amount": 0, "blocked_amount": amount})]
        elif effective_amount == 0:
            step_events = [ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, target_uid=uid, payload={"amount": 0})]
        else:
            transactions.append(ModifyCardCounter(uid, "life", -effective_amount))
            step_events = [ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, target_uid=uid, payload={"amount": effective_amount})]
            if card.life - effective_amount <= 0 and owner is not None:
                destroyed_card = replace(card, life=card.life - effective_amount)
                transactions.extend(destroyed_card_transactions(current, owner, destroyed_card))
                step_events.extend(_destroyed_by_damage_events(destroyed_card, context, owner))
        current = log.apply_all(current, transactions)
        events.extend(step_events)
    return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))


@register_op("distributed_damage")
def op_distributed_damage(snapshot, operation, context, ctx: OpContext):
    remaining = ctx.amount(operation, snapshot=snapshot, context=context)
    transactions: list[SnapshotTransaction] = []
    events: list[ExecutionEvent] = []
    destroyed_cards: dict[str, list["CardView"]] = {"me": [], "opponent": []}
    for uid in ctx.target_uids(snapshot, operation, context):
        if remaining <= 0:
            break
        card = snapshot.cards_by_uid.get(uid)
        if card is None or card.life <= 0:
            continue
        damage = min(remaining, card.life)
        remaining -= damage
        if card.has_temp_shield:
            transactions.append(SetCardField(uid, "has_temp_shield", False))
            events.append(ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, target_uid=uid, payload={"amount": 0, "blocked_amount": damage}))
            continue
        transactions.append(ModifyCardCounter(uid, "life", -damage))
        events.append(ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, target_uid=uid, payload={"amount": damage}))
        if card.life - damage <= 0:
            owner = ctx.owner_of_field_uid(snapshot, uid)
            if owner is not None:
                destroyed_card = replace(card, life=card.life - damage)
                destroyed_cards[owner].append(destroyed_card)
                transactions.extend((RemoveZoneUid(owner, "field", uid), ModifyPlayerCounter(owner, "graveyard_count", 1), RemoveCard(uid, destroyed_card)))
                events.extend(_destroyed_by_damage_events(destroyed_card, context, owner))
    if remaining > 0 and operation.params.get("overflow_to_leader"):
        leader_player = context.opponent
        transactions.append(ModifyPlayerCounter(leader_player, "life", -remaining))
        events.append(ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, target_uid=None, payload={"amount": remaining, "target": "leader"}))
    for owner, cards in destroyed_cards.items():
        if not cards:
            continue
        view = snapshot.me if owner == "me" else snapshot.opponent
        transactions.append(SetPlayerField(owner, "destroyed_card_ids", (*view.destroyed_card_ids, *(CardIdCount(card_id=card.card_id, count=card.style_id) for card in cards))))
    return ctx.run(snapshot, tuple(transactions), tuple(events))


@register_op("heal")
def op_heal(snapshot, operation, context, ctx: OpContext):
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    target_ref = operation.target_ref or ""
    if target_ref in {"self_leader", "ally_leader"}:
        cur_life = snapshot.me.life if context.player == "me" else snapshot.opponent.life
        max_life = snapshot.me.max_life if context.player == "me" else snapshot.opponent.max_life
        new_life = min(max_life, cur_life + amount)
        delta = new_life - cur_life
        events = ()
        if delta > 0:
            # 主战者实际回复时发 HEAL 事件（player=被回复方, target_uid=None 表示回复玩家/leader），
            # 供"主战者回复时"(on_leader_healed) 持续效果响应（条规成员/誓言干部/纹章）。
            events = (ExecutionEvent(EventKind.HEAL, source_uid=context.source_uid, target_uid=None, player=context.player, payload={"amount": delta}),)
        return ctx.run(snapshot, (ModifyPlayerCounter(context.player, "life", delta),), events)
    transactions = []
    for uid in ctx.target_uids(snapshot, operation, context):
        card = snapshot.cards_by_uid.get(uid)
        if card is None:
            continue
        # 随从回血同样要按 max_life 封顶：满血随从的「回复1点生命值」是 no-op，
        # 不能把 life 抬到 max_life 之上。此前这里直接 ModifyCardCounter(+amount)，
        # 导致「回复自己战场上所有随从1点生命值」把满血随从算成 +1
        # （replay_...5110789 st91：录像 life=4/4 不变，本地预测 5）。
        # max_life 缺失（<=0，协议未下发）时保持原行为，避免把有效回复也吞掉。
        delta = amount
        if card.max_life > 0:
            delta = min(card.max_life, card.life + amount) - card.life
        if delta:
            transactions.append(ModifyCardCounter(uid, "life", delta))
    return ctx.run(snapshot, tuple(transactions), ())


@register_op("destroy")
def op_destroy(snapshot, operation, context, ctx: OpContext):
    return ctx.apply_field_card_operation(snapshot, operation, context, "destroy")


@register_op("banish")
def op_banish(snapshot, operation, context, ctx: OpContext):
    return ctx.apply_field_card_operation(snapshot, operation, context, "banish")


@register_op("destruction_immunity")
def op_destruction_immunity(snapshot, operation, context, ctx: OpContext):
    transactions = tuple(SetCardField(uid, "has_cant_destroy", True) for uid in ctx.target_uids(snapshot, operation, context))
    return ctx.run(snapshot, transactions, ())


@register_op("damage_limit")
def op_damage_limit(snapshot, operation, context, ctx: OpContext):
    set_to = int(operation.params.get("set_to", 0))
    transactions = []
    for uid in ctx.target_uids(snapshot, operation, context):
        card = snapshot.cards_by_uid[uid]
        transactions.extend((
            SetCardField(uid, "has_damage_cut", True),
            SetCardField(uid, "has_reduce_damage", True),
            SetCardField(uid, "supplement_info", {**card.supplement_info, "damage_limit": set_to}),
        ))
    return ctx.run(snapshot, tuple(transactions), ())
