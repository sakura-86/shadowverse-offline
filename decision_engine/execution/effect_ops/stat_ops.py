"""stat_ops：卡牌关键字/能力/费用/倒计时/攻击次数类 operation 执行器。

覆盖 op：gain_keyword / remove_keyword / remove_all_abilities / change_cost /
set_countdown / modify_countdown / set_attack_limit / modify_attack_count
拆分方案见 docs/EFFECT_INTERPRETER_REFACTOR_PLAN.md。
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..events import EventKind, ExecutionEvent
from ..state_ops import InsertZoneUid, ModifyCardCounter, ModifyPlayerCounter, RemoveCard, RemoveZoneUidAt, SetCardField
from ..transactions import RollbackTransactionLog, SnapshotTransaction
from .base import EffectInterpreterError, OpContext, register_op

if TYPE_CHECKING:
    from decision_engine.cards.templates import OperationSpec
    from decision_engine.state.snapshot import BattleSnapshot

    from ..effect_interpreter import EffectInterpretationContext, EffectInterpretationResult


def _attack_limit_is_permanent(operation: "OperationSpec") -> bool:
    """这次「攻击次数」改动是不是随从的**关键字**（直到离场都有效）。

    spec 里只有显式写 `duration="current_turn"` 的才是「本回合中」的临时加成；
    缺省（`set_attack_limit` 的既有写法）以及 `while_on_field` 都是永久。
    两类在协议帧里都只表现为 `attack_limit=N` + `buff.attack_limit=N`，
    运行期无法区分，所以只能在赋予的这一刻按 spec 语义留痕。
    """
    params = operation.params or {}
    duration = str(params.get("duration") or "").strip()
    return duration not in {"current_turn", "this_turn", "until_turn_end"}


@register_op("gain_keyword")
def op_gain_keyword(snapshot, operation, context, ctx: OpContext):
    keyword = str(operation.params.get("keyword", ""))
    field_name = {
        "storm": "has_storm",
        "rush": "has_rush",
        "guard": "has_guard",
        "bane": "has_bane",
        "drain": "has_drain",
        "intimidate": "has_cant_be_attacked",
        "ambush": "has_ambush",
        "barrier": "has_temp_shield",
        "temp_shield": "has_temp_shield",
        "last_words": "has_last_words",
        "cant_select": "has_cant_select",
        "cant_attack": "has_cant_attack",
    }.get(keyword)
    if keyword == "double_attack":
        transactions: list[SnapshotTransaction] = []
        for uid in ctx.target_uids(snapshot, operation, context):
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            if card.attack_limit < 2:
                bonus = 2 - card.attack_limit
                transactions.append(SetCardField(uid, "attack_limit", 2))
                transactions.append(SetCardField(uid, "attacks_left", card.attacks_left + bonus))
        return ctx.run(snapshot, tuple(transactions), ())
    if field_name is None:
        raise EffectInterpreterError(f"unsupported keyword: {keyword}")
    transactions: list[SnapshotTransaction] = []
    for uid in ctx.target_uids(snapshot, operation, context):
        card = snapshot.cards_by_uid.get(uid)
        transactions.append(SetCardField(uid, field_name, True))
        # 攻击窗口只在战场上成立：手牌/牌组里的卡没有攻击次数概念，协议手牌帧也不下发
        # attack_limit / attacks_left（恒 0）。给手牌卡写这组字段会让它在打出前就带
        # 「可攻击」状态，replay after 帧会与录制不一致（10722310 给手牌铁甲骑士突进）。
        on_field = card is not None and (uid in snapshot.me.field or uid in snapshot.opponent.field)
        if on_field and card.card_type.is_follower and keyword in {"storm", "rush"}:
            if card.attack_limit == 0:
                transactions.append(SetCardField(uid, "attack_limit", 1))
            if card.attacks_left <= 0:
                transactions.append(SetCardField(uid, "attacks_left", 1))
            transactions.append(SetCardField(uid, "can_attack", True))
            if keyword == "storm":
                transactions.append(SetCardField(uid, "can_attack_leader", True))
        elif card is not None and card.card_type.is_follower and keyword == "cant_attack":
            transactions.append(SetCardField(uid, "can_attack", False))
            transactions.append(SetCardField(uid, "can_attack_leader", False))
            transactions.append(SetCardField(uid, "attacks_left", 0))
    return ctx.run(snapshot, tuple(transactions), ())


@register_op("remove_keyword")
def op_remove_keyword(snapshot, operation, context, ctx: OpContext):
    keyword = str(operation.params.get("keyword", ""))
    field_name = {
        "storm": "has_storm",
        "rush": "has_rush",
        "guard": "has_guard",
        "bane": "has_bane",
        "drain": "has_drain",
        "last_words": "has_last_words",
        "cant_attack": "has_cant_attack",
    }.get(keyword)
    if field_name is None:
        if not keyword:
            return ctx.run(snapshot, (), ())
        raise EffectInterpreterError(f"unsupported keyword removal: {keyword}")
    target_uids = ctx.target_uids(snapshot, operation, context)
    transactions = tuple(SetCardField(uid, field_name, False) for uid in target_uids)
    events = tuple(
        ExecutionEvent(EventKind.KEYWORD_REMOVED, source_uid=context.source_uid, target_uid=uid, payload={"keyword": keyword})
        for uid in target_uids
    )
    return ctx.run(snapshot, transactions, events)


@register_op("remove_all_abilities")
def op_remove_all_abilities(snapshot, operation, context, ctx: OpContext):
    fields = (
        "has_guard",
        "has_storm",
        "has_rush",
        "has_bane",
        "has_drain",
        "has_cant_attack",
        "has_cant_be_attacked",
        "has_cant_select",
        "has_last_words",
        "has_activation",
        "has_ambush",
        "has_damage_cut",
        "has_induction",
        "has_reduce_damage",
        "has_cant_destroy",
        "has_super_evolve_buff",
        "has_temp_shield",
    )
    transactions: list[SnapshotTransaction] = []
    targets = ctx.target_uids(snapshot, operation, context)
    for uid in targets:
        transactions.extend(SetCardField(uid, field, False) for field in fields)
        transactions.append(SetCardField(uid, "attack_limit", 0))
    # 事件必须发：同一 effect 里「先失去所有能力、再致死伤害」时，谢幕曲不发动。
    # 消费端（last words 结算）靠事件时序判断，回看 step 前的 CardView 会误触发。
    events = tuple(
        ExecutionEvent(EventKind.ABILITY_REMOVED, source_uid=context.source_uid, target_uid=uid)
        for uid in targets
    )
    return ctx.run(snapshot, tuple(transactions), events)


@register_op("change_cost")
def op_change_cost(snapshot, operation, context, ctx: OpContext):
    if operation.target_ref == "last_drawn_cards" and context.execution_context.replay_mode:
        # replay 的抽牌绑定来自 after 帧，抽到的卡已带最终费用；再按 delta 改一次
        # 会重复扣减（10504110「抽2张，使其费用-8」predicted 0 / actual 2）。
        return ctx.run(snapshot, (), ())
    targets = ctx.target_uids(snapshot, operation, context)
    # 目标可能在同批次前序操作中已离场 (如飞燕入场伤害打死随从后再改其费用),
    # 与其它 stat/combat op 一致地跳过不存在的 uid, 而非 KeyError 崩溃
    # (对齐 resolution_policy.target_missing_policy=skip_operation)。
    targets = tuple(uid for uid in targets if uid in snapshot.cards_by_uid)
    if "set_to" in operation.params:
        transactions = tuple(SetCardField(uid, "cost", max(0, int(operation.params["set_to"]))) for uid in targets)
        return ctx.run(snapshot, transactions, ())
    if "formula" in operation.params:
        raise EffectInterpreterError(f"change_cost formula requires runtime execution: {operation.params['formula']}")
    if "delta" not in operation.params:
        raise EffectInterpreterError("change_cost requires set_to, delta, or explicit runtime formula")
    delta = int(operation.params["delta"])
    if operation.params.get("requires_target_has_spell_boost"):
        targets = tuple(uid for uid in targets if snapshot.cards_by_uid[uid].has_spell_boost)
    if operation.params.get("skip_target_variable_x"):
        targets = tuple(uid for uid in targets if snapshot.cards_by_uid[uid].variable_x <= 0)
    transactions = tuple(SetCardField(uid, "cost", max(0, snapshot.cards_by_uid[uid].cost + delta)) for uid in targets)
    return ctx.run(snapshot, transactions, ())


@register_op("set_countdown")
def op_set_countdown(snapshot, operation, context, ctx: OpContext):
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    transactions = tuple(SetCardField(uid, "countdown", amount) for uid in ctx.target_uids(snapshot, operation, context))
    return ctx.run(snapshot, transactions, ())


@register_op("modify_countdown")
def op_modify_countdown(snapshot, operation, context, ctx: OpContext):
    if operation.amount is not None:
        amount = ctx.amount(operation, snapshot=snapshot, context=context)
    else:
        # 自动解析产物常把增量写在 params.delta（如 90021210「使用法术时倒计数-1」），
        # 与 amount 字段等价，这里统一支持。
        delta = operation.params.get("delta")
        if delta is None:
            raise EffectInterpreterError("modify_countdown requires amount or params.delta")
        amount = int(delta)
    if int(operation.params.get("delta_sign", 1)) < 0:
        amount = -amount
    current = snapshot
    log = RollbackTransactionLog()
    events: list[ExecutionEvent] = []
    # replay 模式下本 step 前序 effect 新召唤/生成的卡用 after 帧状态，countdown
    # 已经包含本次修正的结果，不能再按 delta 减一遍。
    recorded_generated_uids = context.step_generated_uids if context.execution_context.replay_mode else frozenset()
    for uid in ctx.target_uids(snapshot, operation, context):
        if uid in recorded_generated_uids:
            continue
        card = current.cards_by_uid.get(uid)
        if card is None:
            continue
        next_countdown = card.countdown + amount
        owner = ctx.owner_of_field_uid(current, uid)
        if next_countdown <= 0 and owner is not None:
            result = ctx.kernel.destroy_field_card(current, owner, uid)
            current = log.apply_all(current, tuple(applied.forward for applied in result.transaction_log.applied)) if result.transaction_log.applied else result.snapshot
            events.extend(result.events)
            continue
        current = log.apply_all(current, (SetCardField(uid, "countdown", next_countdown),))
    from ..effect_interpreter import EffectInterpretationResult
    return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))


@register_op("set_attack_limit")
def op_set_attack_limit(snapshot, operation, context, ctx: OpContext):
    amount = ctx.amount(operation, snapshot=snapshot, context=context)
    transactions = ctx.attack_limit_transactions(
        snapshot,
        ctx.target_uids(snapshot, operation, context),
        amount,
        permanent=_attack_limit_is_permanent(operation),
    )
    return ctx.run(snapshot, transactions, ())


@register_op("modify_attack_count")
def op_modify_attack_count(snapshot, operation, context, ctx: OpContext):
    attacks_per_turn = operation.params.get("attacks_per_turn", operation.params.get("attack_limit"))
    if attacks_per_turn is None:
        raise EffectInterpreterError("modify_attack_count requires attacks_per_turn")
    amount = ctx.resolve_amount_value(attacks_per_turn, snapshot=snapshot, context=context)
    transactions = ctx.attack_limit_transactions(
        snapshot,
        ctx.target_uids(snapshot, operation, context),
        amount,
        restore_attacks_left=bool(operation.params.get("restore_attacks_left", False)),
        sync_attacks_left_to_limit_increase=True,
        can_attack_leader=bool(operation.params.get("can_attack_leader", False)),
        permanent=_attack_limit_is_permanent(operation),
    )
    return ctx.run(snapshot, transactions, ())



@register_op("banish_hand_card")
def op_banish_hand_card(snapshot, operation, context, ctx: OpContext):
    """使手牌中的目标卡「消失」：不进墓地，也不触发舍弃事件。"""
    player = context.player
    view = snapshot.me if player == "me" else snapshot.opponent
    positions = {uid: index for index, uid in enumerate(view.hand)}
    transactions: list[SnapshotTransaction] = []
    for uid in ctx.target_uids(snapshot, operation, context):
        index = positions.get(uid)
        card = snapshot.cards_by_uid.get(uid)
        if index is None or card is None:
            continue
        transactions.extend((RemoveZoneUidAt(player, "hand", index, uid), RemoveCard(uid, card)))
    return ctx.run(snapshot, tuple(transactions), ())


@register_op("summon_drawn_elf_followers")
def op_summon_drawn_elf_followers(snapshot, operation, context, ctx: OpContext):
    """把本效果刚抽到的精灵（elf）随从从手牌移到战场。

    用于「抽取1张卡牌，召唤其中的所有精灵·随从」。只处理本 effect 的 draw
    事件（context.last_drawn_uids），不扫描整手牌。
    """
    player = context.player
    view = snapshot.me if player == "me" else snapshot.opponent
    from ..generated_card_materializer import default_card_resolver

    resolver = default_card_resolver()
    hand_positions = {uid: index for index, uid in enumerate(view.hand)}
    drawn_uids = context.last_drawn_uids or tuple(
        (event.target_uid if event.target_uid is not None else event.source_uid)
        for event in context.execution_context_events
        if event.name == "draw" and (event.target_uid is not None or event.source_uid is not None)
    )
    transactions: list[SnapshotTransaction] = []
    events: list[ExecutionEvent] = []
    selected: list[tuple[int, int]] = []
    for uid in drawn_uids:
        card = snapshot.cards_by_uid.get(uid)
        if card is None or uid not in hand_positions:
            continue
        base = resolver.get_base_by_card_id(card.card_id)
        if base is None or int(getattr(base, "type_id", 0) or 0) != 1:
            continue
        is_elf = int(getattr(base, "class_id", -1)) == 1 or str(getattr(base, "craft_class", "")) == "elf"
        if not is_elf:
            continue
        selected.append((hand_positions[uid], uid))
    # 先按手牌索引降序移除，避免前面的移除导致后面的索引偏移。
    for index, uid in sorted(selected, key=lambda item: item[0], reverse=True):
        transactions.append(RemoveZoneUidAt(player, "hand", index, uid))
    for _, uid in selected:
        transactions.extend((
            InsertZoneUid(player, "field", uid),
            SetCardField(uid, "can_attack", False),
            SetCardField(uid, "can_attack_leader", False),
            SetCardField(uid, "attacks_left", 0),
            ModifyPlayerCounter(player, "rally", 1),
        ))
        events.append(ExecutionEvent(EventKind.SUMMON, source_uid=context.source_uid, target_uid=uid, player=player))
    return ctx.run(snapshot, tuple(transactions), tuple(events))


@register_op("grant_last_words_summon_copy")
def op_grant_last_words_summon_copy(snapshot, operation, context, ctx: OpContext):
    """使目标随从获得「谢幕曲：召唤1个与其同名的随从」。

    只写入 supplement_info 标记并置 has_last_words；实际召唤在同名卡离场时由
    TemplateEventHandler 的 last_words 分支物化并执行。
    """
    transactions: list[SnapshotTransaction] = []
    for uid in ctx.target_uids(snapshot, operation, context):
        card = snapshot.cards_by_uid.get(uid)
        if card is None or not card.card_type.is_follower:
            # 卡面是「自己使用**随从**时」：护符/法术不该获得谢幕曲
            # （replay_...1054398 st112：打出神域的物证时被多赋了 has_last_words）。
            continue
        transactions.append(SetCardField(uid, "supplement_info", {**card.supplement_info, "last_words_summon_copy": 1}))
        transactions.append(SetCardField(uid, "has_last_words", True))
    return ctx.run(snapshot, tuple(transactions), ())


@register_op("restrict_attack_this_turn")
def op_restrict_attack_this_turn(snapshot, operation, context, ctx: OpContext):
    """回合结束前无法攻击随从或主战者；标记在 turn_start 清理并恢复。"""
    transactions: list[SnapshotTransaction] = []
    for uid in ctx.target_uids(snapshot, operation, context):
        card = snapshot.cards_by_uid.get(uid)
        if card is None or not card.card_type.is_follower:
            continue
        transactions.extend((
            SetCardField(uid, "has_cant_attack", True),
            SetCardField(uid, "can_attack", False),
            SetCardField(uid, "can_attack_leader", False),
            SetCardField(uid, "supplement_info", {**card.supplement_info, "cant_attack_until_turn_end": 1}),
        ))
    return ctx.run(snapshot, tuple(transactions), ())


@register_op("swap_stats")
def op_swap_stats(snapshot, operation, context, ctx: OpContext):
    """交换目标的攻击力与生命值（生命上限同步）。

    用于「攻击力与生命值交换」类文本。只交换当前 atk/life，不推断 buff 来源；
    目标可以是场上卡或手牌卡（如「在手牌中发动」的持续效果）。
    """
    transactions: list[SnapshotTransaction] = []
    for uid in ctx.target_uids(snapshot, operation, context):
        card = snapshot.cards_by_uid.get(uid)
        if card is None:
            continue
        transactions.extend((
            SetCardField(uid, "atk", card.life),
            SetCardField(uid, "life", card.atk),
            SetCardField(uid, "max_life", card.atk),
        ))
    return ctx.run(snapshot, tuple(transactions), ())


@register_op("set_stat")
def op_set_stat(snapshot, operation, context, ctx: OpContext):
    transactions: list[SnapshotTransaction] = []
    attack = operation.params.get("attack", operation.params.get("atk"))
    life = operation.params.get("life", operation.params.get("defense"))
    if attack is None and life is None:
        raise EffectInterpreterError("set_stat requires attack or life")
    damaged: list[tuple[int, int]] = []
    for uid in ctx.target_uids(snapshot, operation, context):
        if attack is not None:
            transactions.append(SetCardField(uid, "atk", ctx.resolve_amount_value(attack, snapshot=snapshot, context=context)))
        if life is not None:
            value = ctx.resolve_amount_value(life, snapshot=snapshot, context=context)
            transactions.append(SetCardField(uid, "life", value))
            transactions.append(SetCardField(uid, "max_life", value))
            damaged.append((uid, value))
    transactions.extend(ctx.field_death_transactions(snapshot, damaged))
    events = ctx.field_destroy_events(snapshot, damaged, source_uid=context.source_uid)
    return ctx.run(snapshot, tuple(transactions), events)


@register_op("buff_stat", "stat_buff", "stat_debuff")
def op_buff_stat(snapshot, operation, context, ctx: OpContext):
    atk = ctx.resolve_amount_value(operation.params.get("atk", operation.params.get("attack", 0)), snapshot=snapshot, context=context)
    life = ctx.resolve_amount_value(operation.params.get("life", operation.params.get("defense", 0)), snapshot=snapshot, context=context)
    duration = str(operation.params.get("duration", "") or "")
    transactions: list[SnapshotTransaction] = []
    events: list[ExecutionEvent] = []
    damaged: list[tuple[int, int]] = []
    # 「本次 buff 是否已由录像帧体现」必须按 uid 判断：此前用「任一事件带
    # replay_recorded」的全局布尔，再配合 `card.buff.atk >= atk`，会把**累积 buff**
    # 的卡全部误判成「已记录」。典型场景：手牌里的 10434110（buff.atk 已累积到 5）
    # 每次魔力增幅再 +1/+1 时，5 >= 1 成立 → 直接跳过，导致 atk 一直不涨
    # （replay_...6900318 st21：录像 atk 5->6，本地不动）。
    replay_recorded_uids = {
        event.source_uid
        for event in context.execution_context_events
        if event.payload.get("replay_recorded") and event.source_uid is not None
    }
    generated_uids = {card.uid for card in context.generated_cards}
    if context.execution_context.replay_mode:
        # 本 step 前序 effect 已分配掉的生成卡也是「录像取证」：它们来自 after 帧，
        # 数值已含本次结算结果。stat_buff / summon 是各自独立的 effect，
        # 后一个 effect 的 context.generated_cards 是空的（只有需要生成卡的 op
        # 才会分到池），单看 generated_cards 会把「召唤 X 再使其+1/+0」
        # 判成未记录而双算（10932310 人性的爱：after 帧 atk=3，本地变 4）。
        generated_uids |= set(context.step_generated_uids)
    for uid in ctx.target_uids(snapshot, operation, context):
        card = snapshot.cards_by_uid.get(uid)
        if card is None:
            continue
        already_recorded = (uid in replay_recorded_uids or uid in generated_uids) and card.buff is not None and card.buff.atk >= atk and card.buff.max_life_diff >= life
        if not already_recorded:
            if atk:
                transactions.append(ModifyCardCounter(uid, "atk", atk))
            if life:
                transactions.append(ModifyCardCounter(uid, "life", life))
                if uid in snapshot.me.hand or uid in snapshot.opponent.hand:
                    # 手牌里的随从卡尚未上场，max_life 保持 0；「+Y 生命」只体现在
                    # buff.max_life_diff，上场时再由它决定实际上限。
                    # 录像证据（replay_...6900318 st21 的手牌 uid=25）：
                    #   BEFORE life=6 / max_life=0 / buff.max_life_diff=5
                    #   AFTER  life=7 / max_life=0 / buff.max_life_diff=6
                    if card.buff is not None:
                        transactions.append(
                            SetCardField(
                                uid,
                                "buff",
                                replace(card.buff, max_life_diff=card.buff.max_life_diff + life),
                            )
                        )
                else:
                    transactions.append(ModifyCardCounter(uid, "max_life", life))
                    damaged.append((uid, card.life + life))
        if card.card_type.is_follower and (atk > 0 or life > 0):
            events.append(ExecutionEvent(EventKind.SOURCE_BUFFED, source_uid=uid, target_uid=uid, player=ctx.owner_of_field_uid(snapshot, uid), payload={"attack_delta": atk, "life_delta": life, "replay_recorded": already_recorded}))
        if duration == "turn_end" and (atk or life):
            # 「回合结束前 +X/+Y」：记录增量标记，由 end_turn 回滚，
            # 不能把临时身材当成永久 buff（否则回合结束后 atk/life 会偏高）。
            info = dict(card.supplement_info)
            if atk:
                info["temp_atk_bonus_until_turn_end"] = int(info.get("temp_atk_bonus_until_turn_end", 0) or 0) + atk
            if life:
                info["temp_life_bonus_until_turn_end"] = int(info.get("temp_life_bonus_until_turn_end", 0) or 0) + life
            transactions.append(SetCardField(uid, "supplement_info", info))
    transactions.extend(ctx.field_death_transactions(snapshot, damaged))
    events.extend(ctx.field_destroy_events(snapshot, damaged, source_uid=context.source_uid))
    return ctx.run(snapshot, tuple(transactions), tuple(events))


@register_op("stat_debuff_random_repeated")
def op_stat_debuff_random_repeated(snapshot, operation, context, ctx: OpContext):
    targets = ctx.validate_field_targets(snapshot, context.random_targets, context.opponent, operation.target_ref or "random_enemy_followers") if context.random_targets else ctx.target_uids(snapshot, operation, context)
    if not targets:
        from ..effect_interpreter import EffectInterpretationResult
        from ..transactions import RollbackTransactionLog
        return EffectInterpretationResult(snapshot=snapshot, transaction_log=RollbackTransactionLog(), events=())
    atk = ctx.resolve_amount_value(operation.params.get("atk", operation.params.get("attack", 0)), snapshot=snapshot, context=context)
    life = ctx.resolve_amount_value(operation.params.get("life", operation.params.get("defense", 0)), snapshot=snapshot, context=context)
    current_life = {uid: snapshot.cards_by_uid[uid].life for uid in targets if uid in snapshot.cards_by_uid}
    transactions: list[SnapshotTransaction] = []
    events: list[ExecutionEvent] = []
    for uid in targets:
        card = snapshot.cards_by_uid.get(uid)
        if card is None:
            continue
        if atk:
            transactions.append(ModifyCardCounter(uid, "atk", atk))
        if life:
            transactions.append(ModifyCardCounter(uid, "life", life))
            transactions.append(ModifyCardCounter(uid, "max_life", life))
            current_life[uid] = current_life.get(uid, card.life) + life
        events.append(ExecutionEvent(EventKind.DAMAGE, source_uid=context.source_uid, target_uid=uid, payload={"atk": atk, "life": life}))
    transactions.extend(ctx.field_death_transactions(snapshot, list(current_life.items())))
    return ctx.run(snapshot, tuple(transactions), tuple(events))
