# -*- coding: utf-8 -*-
"""Phase L6 通用效果操作解释器。

解释器只消费结构化 operation，不按卡名写规则，也不做协议 fallback。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace, replace
from typing import Any

from decision_engine.cards.effect_specs import EffectSpec
from decision_engine.cards.resolver import CardResolver
from decision_engine.cards.templates import AmountFormula, OperationSpec
from decision_engine.state.snapshot import BattleSnapshot, CardIdCount, CardType, CardView, CrestView

from .damage_reduction import effective_damage_to_follower, is_super_evolved_on_own_turn
from .events import EventKind, ExecutionEvent
from .execution_context import DrawBinding, LocalExecutionContext
from .hand_materializer import copy_random_opponent_hand_to_self, mark_known_generated_hand_card
from .rule_kernel import CoreRuleKernel, RuleExecutionResult
from .state_ops import (
    AddCard,
    InsertZoneUid,
    ModifyCardCounter,
    ModifyPlayerCounter,
    RemoveCard,
    RemoveZoneUid,
    RemoveZoneUidAt,
    SetCardField,
    SetCardFull,
    SetPlayerField,
)
from .transactions import RollbackTransactionLog, SnapshotTransaction
from .effect_ops.base import EffectInterpreterError, OP_REGISTRY, OpContext
from .effect_ops import counter_ops as _counter_ops  # noqa: F401  触发 op 注册
from .effect_ops import crest_ops as _crest_ops  # noqa: F401  触发 op 注册
from .effect_ops import stat_ops as _stat_ops  # noqa: F401  触发 op 注册
from .effect_ops import evolve_ops as _evolve_ops  # noqa: F401  触发 op 注册
from .effect_ops import resource_ops as _resource_ops  # noqa: F401  触发 op 注册
from .effect_ops import control_ops as _control_ops  # noqa: F401  触发 op 注册
from .effect_ops import combat_ops as _combat_ops  # noqa: F401  触发 op 注册
from .effect_ops import legacy_ops as _legacy_ops  # noqa: F401  触发 op 注册
from .effect_ops.resolution import (
    compare_value as _compare_value,
    effective_tribes as _effective_tribes,
    explicit_random_target_repeat as _explicit_random_target_repeat,
    filter_value as _filter_value,
    keyword_values as _keyword_values,
    unsupported_operation_message as _unsupported_operation_message,
)


@dataclass(frozen=True)
class EffectInterpretationContext:
    source_uid: int | None = None
    source_card: CardView | None = None
    player: str = "me"
    opponent: str = "opponent"
    selected_targets: tuple[int, ...] = ()
    random_targets: tuple[int, ...] = ()
    generated_cards: tuple[CardView, ...] = ()
    draw_bindings: tuple[DrawBinding, ...] = ()
    execution_context: LocalExecutionContext = LocalExecutionContext()
    # 仅 TemplateEventHandler 的多独立随机池事件使用：每个 operation 可按自身
    # 候选池确定性物化；普通 replay/直接调用仍必须提供记录 random_targets。
    allow_unrecorded_random_targets: bool = False
    selected_mode_options: tuple[int, ...] = ()
    last_summoned_uid: int | None = None
    last_summoned_uids: tuple[int, ...] = ()
    last_entered_self_follower_uid: int | None = None
    previous_summoned_uid: int | None = None
    last_added_card_uid: int | None = None
    last_drawn_uids: tuple[int, ...] = ()
    allow_missing_selected_field_targets: bool = False
    execution_context_events: tuple[ExecutionEvent, ...] = ()
    # 当前执行的是哪个 trigger 的效果（fanfare / on_self_entered_field / play_card / …）。
    # 用于区分「入场曲内进化」与「进入战场瞬间进化」的攻击窗口差异。
    trigger_event: str | None = None
    # 本 step 内已经被前序 effect 消耗掉的生成卡 uid（跨 effect 共享）。
    # replay 的 after 帧给的是这些卡「最终状态」，后续 op 若再按 delta 修改会重复
    # 结算（10924110 巴巴洛丝：先召唤海盗旗，再对所有旗子倒计数-5，新旗子的
    # countdown 已经是 2，再 -5 会被打死）。
    step_generated_uids: frozenset[int] = frozenset()


@dataclass(frozen=True)
class EffectInterpretationResult:
    snapshot: BattleSnapshot
    transaction_log: RollbackTransactionLog
    events: tuple[ExecutionEvent, ...] = ()


class EffectInterpreter:
    def __init__(self, kernel: CoreRuleKernel | None = None, card_resolver: CardResolver | None = None):
        self.kernel = kernel or CoreRuleKernel()
        self.card_resolver = card_resolver or CardResolver()
        self._op_context = OpContext(interpreter=self)

    def apply_spec(self, snapshot: BattleSnapshot, spec: EffectSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        raise EffectInterpreterError("EffectSpec must be bound to OperationSpec before execution")

    def apply_operations(
        self,
        snapshot: BattleSnapshot,
        operations: tuple[OperationSpec, ...],
        context: EffectInterpretationContext,
    ) -> EffectInterpretationResult:
        current = snapshot
        log = RollbackTransactionLog()
        all_events: list[ExecutionEvent] = []
        # 「从以下 N 个能力里随机发动 K 个」：任一 op 的 params 声明
        # random_ability_subset=K 时，从带 mode_option.option_index 的能力里按
        # branch_seed 确定性抽取 K 个，其余视为未选中（_mode_option_selected 返回 False）。
        # 随机性统一由 branch_seed 消解，保证同一 (snapshot, action, branch_seed)
        # 产出相同预测（与 source_pool / random_enemy_follower 同一约定）。
        subset_k = 0
        for operation in operations:
            raw_k = operation.params.get("random_ability_subset")
            if isinstance(raw_k, int) and raw_k > 0:
                subset_k = min(raw_k, sum(
                    1 for o in operations
                    if isinstance(o.params.get("mode_option"), dict)
                ))
                break
        if subset_k > 0:
            option_indexes = [
                o.params["mode_option"].get("option_index")
                for o in operations
                if isinstance(o.params.get("mode_option"), dict)
                and isinstance(o.params["mode_option"].get("option_index"), int)
            ]
            if subset_k < len(option_indexes):
                import random as _random
                seed = getattr(context.execution_context, "branch_seed", None) or 0
                chosen = set(_random.Random(int(seed)).sample(option_indexes, subset_k))
                context = replace(context, selected_mode_options=tuple(sorted(chosen)))
        try:
            generated_offset = 0
            for index, operation in enumerate(operations):
                if index == 0:
                    self._prevalidate_targets(snapshot, operation, context)
                if not self._mode_option_selected(operation, context):
                    continue
                threshold = operation.params.get("if_counter_at_least")
                condition = operation.params.get("condition")
                if isinstance(threshold, dict) and threshold.get("subject") == "source" and not self._operation_condition_met(current, operation, context):
                    continue
                if isinstance(condition, dict) and str(condition.get("subject", "")).startswith("source.") and not self._operation_condition_met(current, operation, context):
                    continue
                if operation.target_ref in {"last_summoned", "last_summoned_cards"} and context.last_summoned_uid is None:
                    continue
                operation_context, generated_consumed = self._context_for_operation(operation, context, generated_offset)
                operation_context = replace(operation_context, execution_context_events=(*context.execution_context_events, *all_events))
                generated_offset += generated_consumed
                # 【土之秘术_N】的代价：`resource_cost` 挂在 operation.params 上，而
                # spec 的 `conditions` 只检查「够不够」。如果只检查不扣，一个只有 2 层
                # 的土之印也能连续支付 `土之秘术_1` 和 `土之秘术_3`，搜索会找到真实
                # 打不出来的路线（CASE 02 就出现了 5 步的假解）。
                earth_rite_cost = operation.params.get("resource_cost")
                if isinstance(earth_rite_cost, Mapping) and earth_rite_cost.get("kind") == "earth_rite":
                    paid = self.kernel.consume_earth_sigil(
                        current, operation_context.player, int(earth_rite_cost.get("amount") or 1)
                    )
                    if paid.transaction_log.applied:
                        current = log.apply_all(current, tuple(applied.forward for applied in paid.transaction_log.applied))
                    else:
                        current = paid.snapshot
                    all_events.extend(paid.events)
                result = self.apply_operation(current, operation, operation_context)
                if result.transaction_log.applied:
                    current = log.apply_all(current, tuple(applied.forward for applied in result.transaction_log.applied))
                else:
                    current = result.snapshot
                last_added = next((event.target_uid for event in reversed(result.events) if event.kind in {EventKind.DRAW, EventKind.COPY_CARD} and event.target_uid is not None), None)
                if last_added is not None:
                    context = replace(context, last_added_card_uid=last_added)
                summoned_uids = tuple(
                    event.target_uid if event.target_uid is not None else event.source_uid
                    for event in result.events
                    if event.name in {EventKind.SUMMON.value, EventKind.TRANSFORM.value} and (event.target_uid is not None or event.source_uid is not None)
                )
                if summoned_uids:
                    last_summoned = summoned_uids[-1]
                    context = replace(
                        context,
                        previous_summoned_uid=context.last_summoned_uid,
                        last_summoned_uid=last_summoned,
                        last_summoned_uids=summoned_uids,
                        last_entered_self_follower_uid=last_summoned,
                    )
                drawn_uids = tuple(
                    (event.target_uid if event.target_uid is not None else event.source_uid)
                    for event in result.events
                    if event.name == EventKind.DRAW.value and (event.target_uid is not None or event.source_uid is not None)
                )
                if drawn_uids:
                    context = replace(context, last_drawn_uids=drawn_uids)
                all_events.extend(result.events)
                if context.selected_targets:
                    context = replace(context, allow_missing_selected_field_targets=True)
        except Exception as exc:
            try:
                if log.current is not None:
                    log.rollback()
            finally:
                from .rule_kernel import InsufficientResourceError
                if isinstance(exc, InsufficientResourceError):
                    # 资源不足导致效果不发动：整条效果已回滚，属正常游戏语义。
                    # 抛带标记的异常，上层降级为 PARTIAL（效果未发动），不报 predict_error。
                    raise EffectInterpreterError(f"effect not activated (insufficient resource): {exc}") from exc
                raise EffectInterpreterError(str(exc)) from exc
        return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(all_events))

    def _context_for_operation(self, operation: OperationSpec, context: EffectInterpretationContext, generated_offset: int) -> tuple[EffectInterpretationContext, int]:
        if operation.op not in {"draw", "summon", "add_to_hand", "put_token_to_hand", "transform", "replace_hand_card", "copy_card", "copy_from_opponent_hand", "reanimate", "history_same_name_pool", "deck_operation"}:
            return context, 0
        if not context.generated_cards:
            return context, 0
        # draw 从牌库抽牌，不消耗 generated_cards
        if operation.op == "draw":
            return context, 0
        count = 0
        if "generated_card_count" in operation.params:
            try:
                count = int(operation.params.get("generated_card_count") or 0)
            except (TypeError, ValueError):
                count = 0
        elif operation.amount is not None and operation.amount.kind == "fixed" and operation.amount.value is not None:
            try:
                count = int(operation.amount.value)
            except (TypeError, ValueError):
                count = 0
        elif "count" in operation.params and not isinstance(operation.params.get("count"), dict):
            try:
                count = int(operation.params.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
        if count <= 0:
            return context, 0
        pool = context.generated_cards
        if context.step_generated_uids:
            # 不能复用已经被本 step 前序 effect 分配掉的生成卡：同一个 step 里两张卡
            # 都召唤同一种 token 时池里有多张，先执行者拿走第一张，后执行者若还拿到
            # 同一个 uid，`op_summon` 会因为它已在场而静默 noop，表现为「少召唤 1 张」
            # （10964120 evolve 链）。用 `step_generated_uids`（只含已分配生成卡），
            # 不用「快照里已存在的卡」——后者会把「重复路由导致的重复执行」也过滤掉，
            # 而那种情况本来就该 noop 去重。
            remaining = tuple(card for card in pool if card.uid not in context.step_generated_uids)
            if remaining:
                pool = remaining
        return replace(context, generated_cards=pool[generated_offset:generated_offset + count]), count

    def apply_operation(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        if not self._mode_option_selected(operation, context):
            return EffectInterpretationResult(snapshot=snapshot, transaction_log=RollbackTransactionLog(), events=())
        op = operation.op
        handler = OP_REGISTRY.get(op)
        if handler is not None:
            return handler(snapshot, operation, context, self._op_context)
        raise EffectInterpreterError(_unsupported_operation_message(operation))

    def _run(self, snapshot: BattleSnapshot, transactions: tuple[SnapshotTransaction, ...], events: tuple[ExecutionEvent, ...]) -> EffectInterpretationResult:
        result = self.kernel.apply_transactions(snapshot, transactions, events)
        return EffectInterpretationResult(snapshot=result.snapshot, transaction_log=result.transaction_log, events=result.events)

    def _mode_option_selected(self, operation: OperationSpec, context: EffectInterpretationContext) -> bool:
        mode_option = operation.params.get("mode_option")
        if mode_option is None:
            return True
        if not isinstance(mode_option, dict):
            raise EffectInterpreterError("mode_option must be an object")
        option_index = mode_option.get("option_index")
        if option_index is None:
            raise EffectInterpreterError("mode_option requires option_index")
        if not context.selected_mode_options:
            raise EffectInterpreterError("mode_option requires selected_mode_options")
        return int(option_index) in context.selected_mode_options

    def _prevalidate_targets(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> None:
        supported_ops = {"mode_choice_effect", "mode_option_effect", "damage", "distributed_damage", "heal", "draw", "spellboost", "spellboost_hand", "invoke_self", "recover_pp", "recover_ep", "consume_necromancy", "consume_earth_sigil", "modify_earth_sigil", "summon", "destroy", "banish", "bounce", "return_to_deck", "transform", "gain_keyword", "remove_keyword", "remove_all_abilities", "change_cost", "set_countdown", "modify_countdown", "set_attack_limit", "modify_attack_count", "put_token_to_hand", "add_to_hand", "record_fusion", "replace_hand_card", "copy_card", "copy_from_opponent_hand", "discard", "discard_hand_card", "reanimate", "history_same_name_pool", "set_stat", "buff_stat", "stat_buff", "stat_debuff", "stat_debuff_random_repeated", "swap_stats", "modify_max_pp", "modify_graveyard", "modify_card_counter", "modify_source_variable", "evolve", "super_evolve", "add_crest", "remove_crest", "modify_crest_value", "modify_crest_countdown", "set_leader_max_life", "destruction_immunity", "damage_limit", "modify_hand_arts_gauge"}
        if operation.op not in supported_ops:
            return
        ref = operation.target_ref or ""
        if ref in {"enemy_follower", "selected_enemy_follower"}:
            if not context.selected_targets:
                raise EffectInterpreterError(f"{ref} requires selected_targets")
            if not context.allow_missing_selected_field_targets:
                self._validate_field_targets(snapshot, context.selected_targets, context.opponent, ref, allow_missing=False)
        if ref in {"selected_self_hand_card", "selected_self_follower", "selected_target"}:
            if not context.selected_targets:
                # 手牌为空时「选择自己的 1 张手牌」无事可做（卡面语义为「尽可能选择」），
                # 校验阶段同样放行，避免整段效果被中断（10643110）。
                if ref == "selected_self_hand_card":
                    player_view = snapshot.me if context.player == "me" else snapshot.opponent
                    if not player_view.hand:
                        return
                raise EffectInterpreterError(f"{ref} requires selected_targets")
        if ref in {"last_summoned", "last_entered_self_follower", "previous_summoned", "last_added_card"}:
            return

    def _amount(self, operation: OperationSpec, default: int = 0, snapshot: BattleSnapshot | None = None, context: EffectInterpretationContext | None = None) -> int:
        if operation.amount is not None:
            return self._resolve_amount_value(operation.amount, snapshot=snapshot, context=context)
        raw = operation.params.get("amount", default)
        return self._resolve_amount_value(raw, snapshot=snapshot, context=context)

    def _resolve_amount_value(self, value: Any, *, snapshot: BattleSnapshot | None, context: EffectInterpretationContext | None) -> int:
        result = self._resolve_amount_value_inner(value, snapshot=snapshot, context=context)
        # 通用减益: formula.params.negate=True 时对结果取负 (如 -0/-X debuff 的 X)。
        if isinstance(value, AmountFormula):
            if value.params.get("negate"):
                return -result
        elif isinstance(value, dict):
            p = value.get("params", {})
            if isinstance(p, dict) and p.get("negate"):
                return -result
        return result

    def _resolve_amount_value_inner(self, value: Any, *, snapshot: BattleSnapshot | None, context: EffectInterpretationContext | None) -> int:
        if isinstance(value, AmountFormula):
            formula = value
        elif isinstance(value, dict):
            formula = AmountFormula(
                kind=str(value.get("kind", "fixed")),
                value=value.get("value"),
                variable=value.get("variable"),
                params=dict(value.get("params", {})) if isinstance(value.get("params", {}), dict) else {},
            )
        else:
            return int(value)
        if formula.value is not None:
            return int(formula.value)
        if formula.kind == "snapshot_count_diff":
            if snapshot is None:
                raise EffectInterpreterError("snapshot_count_diff requires snapshot")
            minuend = str(formula.params.get("minuend", ""))
            subtrahend = str(formula.params.get("subtrahend", ""))
            return max(0, self._resolve_amount_value(AmountFormula(kind="snapshot_count", variable=minuend), snapshot=snapshot, context=context) - self._resolve_amount_value(AmountFormula(kind="snapshot_count", variable=subtrahend), snapshot=snapshot, context=context))
        if formula.kind == "derived_value":
            return self._derived_amount(formula, snapshot=snapshot, context=context)
        if formula.kind == "operation_result_count" and formula.variable == "previous_destroyed_cards":
            if context is None:
                raise EffectInterpreterError("previous_destroyed_cards requires context")
            return sum(1 for event in context.execution_context_events if event.name == "destroy")
        if formula.kind == "operation_result_count" and formula.variable == "previous_returned_to_deck":
            if context is None:
                raise EffectInterpreterError("previous_returned_to_deck requires context")
            return sum(1 for event in context.execution_context_events if event.name == "return_to_deck")
        if formula.kind not in {"variable", "snapshot_value", "snapshot_count", "snapshot_history_total"} or not formula.variable:
            raise EffectInterpreterError(f"unsupported amount formula: {formula.kind}:{formula.variable or ''}")
        variable = formula.variable
        if variable == "filtered_cards":
            if snapshot is None:
                raise EffectInterpreterError("filtered_cards requires snapshot")
            return len(self._resolve_filtered_targets(snapshot, formula.params.get("filters", ()), context))
        if variable in {"source.variable_x", "variable_x"}:
            source = self._source_card(snapshot, context, "variable amount")
            return int(source.variable_x)
        if variable == "source.spellboost":
            source = self._source_card(snapshot, context, "source.spellboost")
            return int(source.spell_boost_count or source.variable_x)
        if variable == "source.atk":
            source = self._source_card(snapshot, context, "source.atk")
            return int(source.atk)
        if variable == "source.life":
            source = self._source_card(snapshot, context, "source.life")
            return int(source.life)
        if variable == "source.cost":
            source = self._source_card(snapshot, context, "source.cost")
            return int(source.cost)
        if variable == "source.fused_kinds":
            source = self._source_card(snapshot, context, "source.fused_kinds")
            return int(source.supplement_info.get("fused_kinds", source.supplement_info.get("fusion_count", 0)) or 0)
        if variable == "source.distinct_types":
            raise EffectInterpreterError("source.distinct_types requires structured history query")
        if snapshot is None:
            raise EffectInterpreterError(f"snapshot amount requires snapshot: {variable}")
        if variable == "me.play_count":
            return int(snapshot.me.play_count)
        if variable == "me.rally":
            return int(snapshot.me.rally)
        if variable == "me.pp_missing":
            return max(0, snapshot.me.max_pp - snapshot.me.pp)
        if variable == "me.earth_sigil":
            # 「土之印数」= 土之印的 stack 总量，不是场上张数：
            # 一张 stack=13 的土之印，卡面写「X 为土之印数」时 X=13。
            # 与 condition_evaluator 的同名 subject 保持同一口径（那里用
            # sum(max(card.stack, 1))）。此前这里返回张数（恒 1），导致
            # 11034120 劫焰奇美拉的 distributed_damage 只按 1 分配
            # （replay_...1145981 st88：录像 opponent.life 19->8，本地少打）。
            return sum(
                max(int(card.stack), 1)
                for uid in snapshot.me.field
                if (card := snapshot.cards_by_uid.get(uid)) is not None and card.is_earth_sigil
            )
        if variable == "me.field_evidence_count":
            return sum(1 for uid in snapshot.me.field if (card := snapshot.cards_by_uid.get(uid)) is not None and 21 in (card.tribes or ()))
        if variable == "me.hand_evidence_amulet_count":
            return sum(1 for uid in snapshot.me.hand if (card := snapshot.cards_by_uid.get(uid)) is not None and 21 in (card.tribes or ()))
        if variable == "opponent.life_missing":
            return max(0, snapshot.opponent.max_life - snapshot.opponent.life)
        if variable == "me.same_name_entered_count":
            source = self._source_card(snapshot, context, "same_name_entered_count")
            total = 0
            for item in snapshot.me.entered_field_card_ids:
                if item.card_id == source.base_card_id:
                    total = item.count
                    break
            return max(0, total - 1)
        if variable == "all_field_count":
            # 场上所有随从的张数（双方合计），供「对战场上的所有随从造成 X 点伤害，
            # X 为战场上的随从的张数」（10542310 日珥咆哮）使用。
            if snapshot is None:
                raise EffectInterpreterError("all_field_count requires snapshot")
            total = 0
            for side_view in (snapshot.me, snapshot.opponent):
                for uid in side_view.field:
                    card = snapshot.cards_by_uid.get(uid)
                    if card is not None and card.card_type is CardType.FOLLOWER:
                        total += 1
            return total
        if variable in {"me.faith_value", "opponent.faith_value"}:
            # 纹章区信仰值：取该侧所有纹章 faith_value 的最大值
            # （与 condition_evaluator 的同名 subject 同一口径）。
            # 90034330 天晶深渊的 X/Y/Z 合计等于信仰值。
            if snapshot is None:
                raise EffectInterpreterError(f"{variable} requires snapshot")
            view = snapshot.me if variable.startswith("me.") else snapshot.opponent
            return max((int(crest.faith_value) for crest in view.crests), default=0)
        if variable in {"me.artifact_types_entered", "opponent.artifact_types_entered"}:
            # 本局进入战场的创造物（tribe=14）·随从的**种类数**。
            # 与 condition_evaluator 的同名 subject 保持同一口径
            # （10774110 虚刻的安纳提玛的入场曲 AOE 伤害量 X）。
            if snapshot is None:
                raise EffectInterpreterError(f"{variable} requires snapshot")
            from .generated_card_materializer import default_card_resolver

            view = snapshot.me if variable.startswith("me.") else snapshot.opponent
            resolver = default_card_resolver()
            seen: set[int] = set()
            for item in view.entered_field_card_ids:
                base = resolver.get_base_by_card_id(item.card_id)
                if base is None:
                    continue
                tribes = tuple(getattr(base, "tribes", ()) or ())
                type_id = int(getattr(base, "type_id", getattr(base, "card_type", 0)) or 0)
                if 14 in tribes and type_id == 1:
                    seen.add(int(base.base_card_id))
            return len(seen)
        if variable == "me.graveyard_count":
            return int(snapshot.me.graveyard_count)
        if variable == "me.hand":
            return len(snapshot.me.hand)
        if variable == "me.field":
            return len(snapshot.me.field)
        if variable == "opponent.field":
            return len(snapshot.opponent.field)
        if variable == "me.crests":
            return len(snapshot.me.crests)
        if variable == "crest_count":
            player = context.player if context else "me"
            view = snapshot.me if player == "me" else snapshot.opponent
            return len(view.crests)
        if variable == "opponent.crests":
            return len(snapshot.opponent.crests)
        if variable == "me.played_card_ids":
            return sum(item.count for item in snapshot.me.played_card_ids)
        if variable == "me.destroyed_card_ids":
            return sum(item.count for item in snapshot.me.destroyed_card_ids)
        if variable in {"destroyed_ally_shikigami_atk_sum_this_turn", "destroyed_ally_shikigami_life_sum_this_turn"}:
            # 统计本回合被破坏的己方式神随从的原始攻击力/生命值合计
            # tribe 13 = 式神
            from .generated_card_materializer import default_card_resolver
            stat = "atk" if "atk" in variable else "life"
            total = 0
            player = context.player if context else "me"
            view = snapshot.me if player == "me" else snapshot.opponent
            resolver = default_card_resolver()
            for item in view.destroyed_card_ids:
                base_info = resolver.get_base_by_card_id(item.card_id)
                if base_info is None:
                    continue
                if 13 not in base_info.tribes:
                    continue
                total += base_info.atk if stat == "atk" else base_info.life
            return total
        raise EffectInterpreterError(f"unsupported amount variable: {variable}")

    def _derived_amount(self, formula: AmountFormula, *, snapshot: BattleSnapshot | None, context: EffectInterpretationContext | None) -> int:
        params = formula.params
        scale_variable = params.get("scale_variable")
        if scale_variable not in {"source.spellboost_count", "source.spellboost"}:
            raise EffectInterpreterError(f"unsupported derived amount scale_variable: {scale_variable}")
        source = self._source_card(snapshot, context, "derived amount")
        base = int(params.get("base", 0) or 0)
        scale = int(params.get("scale", 1) or 1)
        return base + int(source.spell_boost_count or source.variable_x) * scale

    def _source_card(self, snapshot: BattleSnapshot | None, context: EffectInterpretationContext | None, label: str) -> CardView:
        if context is None:
            raise EffectInterpreterError(f"{label} requires source context")
        if snapshot is not None and context.source_uid is not None and context.source_uid in snapshot.cards_by_uid:
            return snapshot.cards_by_uid[context.source_uid]
        if context.source_card is not None:
            return context.source_card
        raise EffectInterpreterError(f"{label} requires source_uid")

    def _event_context_target_uid(self, target_ref: str, context: EffectInterpretationContext) -> int | None:
        # event_source: 从事件中获取触发源（如攻击者）
        if target_ref == "event_source":
            for event in context.execution_context_events:
                if event.source_uid is not None:
                    return int(event.source_uid)
            # fallback 到 context.source_uid
            if context.source_uid is not None:
                return int(context.source_uid)
            return None
        # event_target: 从事件中获取目标
        if target_ref == "event_target":
            for event in context.execution_context_events:
                if event.target_uid is not None:
                    return int(event.target_uid)
                # 攻击主战者时不存在「交战随从」目标：event_target 指交战对手，
                # 不能退回 event.source_uid——那是攻击者自己。退回会把
                # 「对交战对手造成 N 点伤害」打到自己身上（10823110 打脸时
                # KeyError / 自残），而卡面限定「若攻击随从」时本就不该触发。
                if str(event.name).startswith("attack"):
                    return None
                if event.source_uid is not None:
                    return int(event.source_uid)
            return None
        field_by_ref = {
            "last_summoned": "last_summoned_uid",
            "last_entered_self_follower": "last_entered_self_follower_uid",
            "previous_summoned": "previous_summoned_uid",
            "last_added_card": "last_added_card_uid",
        }
        field = field_by_ref.get(target_ref)
        if field is None:
            return None
        uid = getattr(context, field)
        if uid is None:
            raise EffectInterpreterError(f"{target_ref} requires {field}")
        return int(uid)

    def _hand_is_observable(self, context: EffectInterpretationContext) -> bool:
        """当前视角能否观测到该玩家手牌。

        可见性由 ``LocalExecutionContext.hidden_hand_players`` 声明：默认对手手牌
        不可观测；replay 视角互换后当前视角的 me 才是录像对手，由 replay 上下文
        改成 {"me"}，此时 opponent（录像我方）的手牌反而可见。
        """
        return context.player not in context.execution_context.hidden_hand_players

    def _target_uids(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> tuple[int, ...]:
        threshold = operation.params.get("if_counter_at_least")
        if isinstance(threshold, dict) and threshold.get("subject") == "source" and not self._operation_condition_met(snapshot, operation, context):
            return ()
        ref = operation.target_ref or ""
        if ref == "leftmost_self_field_followers":
            # 「使自己的战场上的从左起的N个随从…」：按场地顺序取前 N 个随从。
            view = snapshot.me if context.player == "me" else snapshot.opponent
            count = int(operation.params.get("count", 2) or 2)
            followers = [
                uid for uid in view.field
                if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower
            ]
            return tuple(followers[:count])
        if ref in {"all_filtered_cards", "random_filtered_cards", "selected_filtered_cards"}:
            targets = self._resolve_filtered_targets(snapshot, operation.params.get("filters", ()), context)
            if ref == "random_filtered_cards":
                if not targets:
                    # filtered pool 为空 (无满足条件的卡): 无目标可选, noop
                    return ()
                cnt = int(operation.params.get("count", 1) or 1)
                exec_ctx = context.execution_context
                seed = getattr(exec_ctx, "branch_seed", None) if exec_ctx is not None else None
                if context.random_targets:
                    random_targets = tuple(uid for uid in context.random_targets if uid in targets)
                    if random_targets:
                        return random_targets[:cnt]
                    # 录制/共享 random_targets 与本 pool 无交集: 有 branch_seed 时按其
                    # 独立采样 (self-play 多 random op 各有独立池); 否则报错 (replay
                    # 契约: 录制目标必须在 pool 内)。
                    if seed is not None:
                        import random as _random
                        return tuple(_random.Random(int(seed) + (context.source_uid or 0)).sample(list(targets), min(cnt, len(targets))))
                    raise EffectInterpreterError("random_filtered_cards requires random target in filtered pool")
                # 无 random_targets: 有 branch_seed 时随机采样, 否则取前 count 个
                if seed is not None and len(targets) > cnt:
                    import random as _random
                    return tuple(_random.Random(int(seed) + (context.source_uid or 0)).sample(list(targets), cnt))
                return targets[:cnt]
            if ref == "selected_filtered_cards":
                if not context.selected_targets:
                    return ()
                selected = tuple(uid for uid in context.selected_targets if uid in targets)
                if len(selected) != len(context.selected_targets):
                    raise EffectInterpreterError("selected target outside filtered pool")
                return selected
            return targets
        if ref in {"source_card", "source_follower", "self_follower"}:
            if context.source_uid is None:
                raise EffectInterpreterError(f"{ref} target requires source_uid")
            return (context.source_uid,)
        if ref == "generated_cards":
            return tuple(card.uid for card in context.generated_cards if card.uid in snapshot.cards_by_uid)
        if ref == "last_summoned_cards":
            if not context.last_summoned_uids:
                raise EffectInterpreterError("last_summoned_cards requires last_summoned_uids")
            return tuple(uid for uid in context.last_summoned_uids if uid in snapshot.cards_by_uid)
        if ref == "last_added_group":
            # 对最后一批 add_to_hand/summon 生成的卡生效。
            # 只保留已物化进 cards_by_uid 的 uid: add_to_hand 生成的手牌卡可能是未物化
            # sentinel (uid<0, 只在 hand 列表不在 cards_by_uid), 下游 SetCardField/stat op
            # 对其 apply 会 KeyError 崩溃。未物化卡无处作用, 过滤掉 (noop)。
            if context.last_summoned_uids:
                return tuple(uid for uid in context.last_summoned_uids if uid in snapshot.cards_by_uid)
            if context.generated_cards:
                return tuple(card.uid for card in context.generated_cards if card.uid in snapshot.cards_by_uid)
            return ()
        if ref == "last_added_card":
            # 「刚加入手牌的卡」：加入动作在手牌不可观测（对手手牌 / replay 视角互换）
            # 或 replay 中 after 手牌没有新卡证据时会被按 no-op 跳过，此时该目标
            # 不存在，按空集跳过而不是抛错。
            if context.last_added_card_uid is None:
                return ()
            return (int(context.last_added_card_uid),)
        event_target_uid = self._event_context_target_uid(ref, context)
        if event_target_uid is not None:
            # 与 last_added_group / last_summoned_cards / generated_cards 一致做存在性过滤:
            # add_to_hand 生成的手牌卡可能是未物化 sentinel (uid<0, 不在 cards_by_uid),
            # 若不在 cards_by_uid, 下游 SetCardField/stat op 对其 apply 会 KeyError 崩溃,
            # 此时该 uid 无处作用, 返回 () noop。
            if event_target_uid in snapshot.cards_by_uid:
                return (event_target_uid,)
            return ()
        if ref == "selected_self_hand_card":
            if not context.selected_targets:
                # 手牌为空时该选择无事可做（discard 等 op 对空目标 no-op），不中断整段效果。
                player_view = snapshot.me if context.player == "me" else snapshot.opponent
                if not player_view.hand:
                    return ()
                raise EffectInterpreterError("selected_self_hand_card requires selected_targets")
            player_view = snapshot.me if context.player == "me" else snapshot.opponent
            for uid in context.selected_targets:
                if uid not in player_view.hand:
                    raise EffectInterpreterError("selected_self_hand_card target must be in hand")
            return context.selected_targets
        if ref == "selected_opponent_hand_card":
            if not context.selected_targets:
                raise EffectInterpreterError("selected_opponent_hand_card requires selected_targets")
            player_view = snapshot.opponent if context.player == "me" else snapshot.me
            for uid in context.selected_targets:
                if uid not in player_view.hand:
                    raise EffectInterpreterError("selected_opponent_hand_card target must be in opponent hand")
            return context.selected_targets
        if ref in {"selected_self_card", "selected_any_card", "selected_any_follower", "selected_target"}:
            if not context.selected_targets:
                raise EffectInterpreterError(f"{ref} requires selected_targets")
            return context.selected_targets
        if ref in {"enemy_follower", "selected_enemy_follower", "combat_opponent"}:
            if not context.selected_targets:
                raise EffectInterpreterError(f"{ref} requires selected_targets")
            return self._validate_field_targets(snapshot, context.selected_targets, context.opponent, ref, allow_missing=context.allow_missing_selected_field_targets)
        if ref == "selected_enemy_follower_or_leader":
            if not context.selected_targets:
                raise EffectInterpreterError(f"{ref} requires selected_targets")
            if context.selected_targets == (0,):
                return context.selected_targets
            return self._validate_field_targets(snapshot, context.selected_targets, context.opponent, ref, allow_missing=context.allow_missing_selected_field_targets)
        if ref == "random_enemy_max_atk_follower":
            # 「破坏对手的攻击力最大的随机1个随从」(10743110 龙人先驱者)：
            # 候选池只取当前攻击力最大的对手随从；replay 录制目标优先。
            view = snapshot.opponent if context.player == "me" else snapshot.me
            followers = [
                uid for uid in view.field
                if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower
            ]
            if not followers:
                return ()
            max_atk = max(snapshot.cards_by_uid[uid].atk for uid in followers)
            candidates = tuple(uid for uid in followers if snapshot.cards_by_uid[uid].atk == max_atk)
            if context.random_targets:
                recorded = tuple(uid for uid in context.random_targets if uid in candidates)
                if recorded:
                    return recorded[:1]
            return candidates[:1]
        if ref in {"random_enemy_follower", "random_enemy_followers", "random_self_card", "random_self_follower", "random_enemy_follower_or_leader", "random_enemy_leader", "random_enemy_guard_followers"}:
            if ref in {"random_enemy_followers", "random_enemy_guard_followers"} and operation.params.get("repeat_source") in {"runtime_x", "me.play_count"}:
                repeat = int(snapshot.me.play_count)
            else:
                # 目标数量优先取 params 里显式声明的 repeat / count / random_count；
                # 没写时只有「复数形式 + 非 fixed 的 amount」才用它表达数量
                # （`destroy random_enemy_followers` 的 `snapshot_count_diff`）。
                # `damage` 的 fixed amount 是伤害值，`random_enemy_follower` 单数形式
                # 的 amount 同样是伤害值，两者都不能当目标数。
                explicit_repeat = _explicit_random_target_repeat(operation)
                if explicit_repeat is not None:
                    repeat = explicit_repeat
                elif (
                    ref in {"random_enemy_followers", "random_enemy_guard_followers"}
                    and operation.amount is not None
                    and operation.amount.kind != "fixed"
                ):
                    repeat = self._amount(operation, default=1, snapshot=snapshot, context=context)
                else:
                    repeat = 1
            if not context.random_targets:
                candidates = self.random_target_candidates(snapshot, ref, player=context.player, opponent=context.opponent)
                if not candidates:
                    return ()
                if len(candidates) == 1:
                    return candidates * repeat
                if len(candidates) == repeat:
                    return candidates
                # 候选多于所需: 用 branch_seed 确定性采样 repeat 个 (统一随机消解, 与
                # source_pool/self-play 一致; 无 branch_seed 时才上报缺录制目标)。
                exec_ctx = context.execution_context
                seed = getattr(exec_ctx, "branch_seed", None) if exec_ctx is not None else None
                if seed is not None:
                    import random as _random
                    picked = _random.Random(int(seed) + (context.source_uid or 0)).sample(list(candidates), min(repeat, len(candidates)))
                    return tuple(picked)
                if context.allow_unrecorded_random_targets:
                    return tuple(candidates[:min(repeat, len(candidates))])
                raise EffectInterpreterError(f"{ref} requires random_targets")
            if context.random_targets:
                # 多随机池 effect 会一次性传入所有组的目标；本 op 只认领属于自己目标侧的那些
                # （random_*_self 取 player 侧，其余取 opponent 侧），其余留给其它 operation。
                own_side = context.player if ref in {"random_self_card", "random_self_follower"} else context.opponent
                own_field = snapshot.me.field if own_side == "me" else snapshot.opponent.field
                # `random_enemy_follower_or_leader` 里 0 表示「主战者」目标，它不在
                # field 列表里。只按 field 过滤会把这类混合池的 leader 目标整块丢掉，
                # 令 10404110 的「5 次随机随从或主战者」被误判成录制目标数量不足。
                claimed = tuple(
                    uid for uid in context.random_targets
                    if uid in own_field or (uid == 0 and ref == "random_enemy_follower_or_leader")
                )
                if len(claimed) >= repeat:
                    # 目标组数量足够时才按本 op 的目标侧认领；不足说明这是
                    # 「同一目标重复 N 次」（如 random_targets=(4,4,0)）的录制写法，
                    # 交给下面的原始逻辑按顺序消费（已死亡的目标会被跳过）。
                    targets = claimed[:repeat]
                    if ref in {"random_enemy_follower_or_leader", "random_enemy_leader"} and 0 in targets:
                        # 0 是主战者，不能进 _validate_field_targets（它要求 uid 在 field 中）。
                        field_targets = tuple(uid for uid in targets if uid != 0)
                        if field_targets:
                            self._validate_field_targets(snapshot, field_targets, own_side, ref, allow_missing=False)
                        return targets
                    return self._validate_field_targets(snapshot, targets, own_side, ref, allow_missing=False)
                if context.allow_unrecorded_random_targets:
                    # 多随机池 effect：录制目标全部属于别的池（本 op 的候选池为空），
                    # 本 op 无事可做。不能让上一个 op 的随机目标被本 op 复用——那会因
                    # 「目标已离场」抛错并回滚整条 effect（10954112 纹章回合结束：
                    # op1 破坏己方随从后，op2 的敌方池为空却复用同一个 uid）。
                    return ()
            if len(context.random_targets) < repeat:
                raise EffectInterpreterError(f"{ref} random_targets count mismatch")
            targets = context.random_targets[:repeat]
            if ref in {"random_enemy_follower_or_leader", "random_enemy_leader"} and 0 in targets:
                field_targets = tuple(uid for uid in targets if uid != 0)
                if field_targets:
                    self._validate_field_targets(snapshot, field_targets, context.opponent, ref, allow_missing=False)
                return targets
            owner = context.player if ref in {"random_self_card", "random_self_follower"} else context.opponent
            return self._validate_field_targets(snapshot, targets, owner, ref, allow_missing=False)
        if ref == "last_drawn_cards":
            # 「抽取N张，使其费用-8」类效果：目标就是本次效果刚抽到的牌。
            if not context.last_drawn_uids:
                raise EffectInterpreterError("last_drawn_cards requires last_drawn_uids")
            return tuple(uid for uid in context.last_drawn_uids if uid in snapshot.cards_by_uid)
        if ref == "all_self_hand_cards":
            return snapshot.me.hand if context.player == "me" else snapshot.opponent.hand
        if ref == "random_self_hand_card":
            # 「使自己的随机N张手牌返回牌组」：随机性统一由 branch_seed 消解
            # （与 source_pool / random_enemy_follower 同一约定），保证同一
            # (snapshot, action, branch_seed) 产出相同结果。
            player_view = snapshot.me if context.player == "me" else snapshot.opponent
            hand = tuple(uid for uid in player_view.hand if uid in snapshot.cards_by_uid)
            repeat = int(operation.params.get("count", 1) or 1)
            if not hand:
                return ()
            if context.random_targets:
                # 录制目标优先，与其它 random_* 分支同一约定：replay 不记录随机分支时，
                # 验收侧用 after 差分反推候选并把它们作为录制目标传入；这里若不认领，
                # 整个 step 会因为「没有 branch_seed」被判成 predict_error
                # （replay_...1185752 st43：10561310 随机回 1 张手牌 + 抽 3）。
                recorded = tuple(uid for uid in context.random_targets if uid in hand)
                if recorded:
                    return recorded[:repeat]
            if repeat >= len(hand):
                return hand
            exec_ctx = context.execution_context
            seed = getattr(exec_ctx, "branch_seed", None) if exec_ctx is not None else None
            if seed is None:
                # 没有 branch_seed 时无法唯一确定随机结果：显式失败，
                # 不静默取前 N 张。
                raise EffectInterpreterError("random_self_hand_card requires branch_seed")
            import random as _random
            ordered = list(hand)
            _random.Random(int(seed)).shuffle(ordered)
            return tuple(ordered[:repeat])
        if ref == "all_self_amulets":
            field = snapshot.me.field if context.player == "me" else snapshot.opponent.field
            return tuple(uid for uid in field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_amulet_like)
        if ref in {"all_self_cards", "self_field", "ally_field"}:
            return snapshot.me.field if context.player == "me" else snapshot.opponent.field
        if ref == "all_self_followers":
            field = snapshot.me.field if context.player == "me" else snapshot.opponent.field
            return tuple(uid for uid in field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower)
        if ref == "all_self_unevolved_followers":
            field = snapshot.me.field if context.player == "me" else snapshot.opponent.field
            return tuple(uid for uid in field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower and card.evolve_state == 0)
        if ref == "all_self_shikigami_followers":
            field = snapshot.me.field if context.player == "me" else snapshot.opponent.field
            return tuple(uid for uid in field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower and self._is_shikigami(card))
        if ref == "other_self_followers":
            field = snapshot.me.field if context.player == "me" else snapshot.opponent.field
            return tuple(uid for uid in field if uid != context.source_uid and (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower)
        if ref == "all_other_self_sword_followers":
            field = snapshot.me.field if context.player == "me" else snapshot.opponent.field
            result = tuple(uid for uid in field if uid != context.source_uid and (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower and self._class_name(card) == "royal")
            if not result:
                import logging
                logging.getLogger(__name__).debug("all_other_self_sword_followers: empty result, field=%s source=%s", field, context.source_uid)
            return result
        if ref == "all_other_followers":
            return tuple(uid for uid in (*snapshot.me.field, *snapshot.opponent.field) if uid != context.source_uid and (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower)
        if ref in {"random_self_guard_unevolved_other_follower", "random_self_unevolved_other_follower", "random_self_unevolved_follower"}:
            repeat = int(operation.params.get("repeat", 1) or 1)
            if not context.random_targets:
                candidates = self.random_target_candidates(snapshot, ref, player=context.player, opponent=context.opponent, source_uid=context.source_uid)
                if not candidates:
                    return ()  # 无满足条件的目标，效果不执行
                if len(candidates) == repeat:
                    return candidates
                if len(candidates) < repeat:
                    return candidates  # 候选不足时使用全部可用候选
                raise EffectInterpreterError(f"{ref} requires random_targets")
            if len(context.random_targets) < repeat:
                raise EffectInterpreterError(f"{ref} random_targets count mismatch")
            targets = context.random_targets[:repeat]
            owner = context.player
            return self._validate_field_targets(snapshot, targets, owner, ref, allow_missing=False)
        if ref in {"enemy_field", "opponent_field"}:
            return snapshot.opponent.field if context.opponent == "opponent" else snapshot.me.field
        if ref == "all_enemy_followers":
            field = snapshot.opponent.field if context.opponent == "opponent" else snapshot.me.field
            return tuple(uid for uid in field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower)
        if ref in {"all_enemy_leaders", "enemy_leader", "opponent_leader", "enemy_player", "opponent_player"}:
            return (0,)
        if ref in {"self_leader", "ally_leader", "self_player", "ally_player"}:
            return (0,)
        if ref == "all_any_followers":
            return tuple(uid for uid in (*snapshot.me.field, *snapshot.opponent.field) if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower)
        if ref == "all_any_amulets":
            return tuple(uid for uid in (*snapshot.me.field, *snapshot.opponent.field) if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_amulet_like)
        if ref == "all_any_cards":
            return tuple(uid for uid in (*snapshot.me.field, *snapshot.opponent.field) if snapshot.cards_by_uid.get(uid) is not None)
        target_uid = operation.params.get("target_uid")
        if target_uid is not None:
            return (int(target_uid),)
        target_uids = operation.params.get("target_uids")
        if isinstance(target_uids, (list, tuple)):
            return tuple(int(uid) for uid in target_uids)
        if context.selected_targets:
            return context.selected_targets
        raise EffectInterpreterError(f"cannot resolve target_ref: {operation.target_ref}")

    def _is_shikigami(self, card: CardView) -> bool:
        return card.name.startswith("式神") or "式神" in card.name

    def random_target_candidates(self, snapshot: BattleSnapshot, target_ref: str, *, player: str = "me", opponent: str = "opponent", source_uid: int | None = None) -> tuple[int, ...]:
        if target_ref == "random_self_card":
            return snapshot.me.field if player == "me" else snapshot.opponent.field
        if target_ref == "random_self_follower":
            field = snapshot.me.field if player == "me" else snapshot.opponent.field
            return tuple(uid for uid in field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower)
        if target_ref == "random_self_guard_unevolved_other_follower":
            field = snapshot.me.field if player == "me" else snapshot.opponent.field
            return tuple(
                uid for uid in field
                if uid != source_uid
                and (card := snapshot.cards_by_uid.get(uid)) is not None
                and card.card_type.is_follower
                and card.has_guard
                and card.evolve_state == 0
            )
        if target_ref == "random_self_unevolved_follower":
            field = snapshot.me.field if player == "me" else snapshot.opponent.field
            return tuple(
                uid for uid in field
                if (card := snapshot.cards_by_uid.get(uid)) is not None
                and card.card_type.is_follower
                and card.evolve_state == 0
            )
        if target_ref == "random_self_unevolved_other_follower":
            field = snapshot.me.field if player == "me" else snapshot.opponent.field
            return tuple(
                uid for uid in field
                if uid != source_uid
                and (card := snapshot.cards_by_uid.get(uid)) is not None
                and card.card_type.is_follower
                and card.evolve_state == 0
            )
        if target_ref == "random_enemy_leader":
            return (0,)
        if target_ref == "random_enemy_guard_followers":
            # 对手战场上拥有【守护】的随从 (亚修雷&莉缇雅进化时破坏对手随机2个守护随从)
            field = snapshot.opponent.field if opponent == "opponent" else snapshot.me.field
            return tuple(
                uid for uid in field
                if (card := snapshot.cards_by_uid.get(uid)) is not None
                and card.card_type.is_follower
                and card.has_guard
            )
        if target_ref not in {"random_enemy_follower", "random_enemy_followers", "random_enemy_follower_or_leader"}:
            return ()
        field = snapshot.opponent.field if opponent == "opponent" else snapshot.me.field
        candidates = tuple(uid for uid in field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower)
        if target_ref == "random_enemy_follower_or_leader":
            return (*candidates, 0)
        return candidates

    def random_target_requires_recorded(self, snapshot: BattleSnapshot, target_ref: str, *, repeat: int = 1, player: str = "me", opponent: str = "opponent", source_uid: int | None = None) -> bool:
        candidates = self.random_target_candidates(snapshot, target_ref, player=player, opponent=opponent, source_uid=source_uid)
        if not candidates:
            return False
        return len(candidates) != repeat

    def _validate_field_targets(self, snapshot: BattleSnapshot, target_uids: tuple[int, ...], player: str, target_ref: str, *, allow_missing: bool = True) -> tuple[int, ...]:
        field = snapshot.me.field if player == "me" else snapshot.opponent.field
        valid: list[int] = []
        for uid in target_uids:
            if uid not in snapshot.cards_by_uid:
                if allow_missing:
                    continue
                raise EffectInterpreterError(f"{target_ref} target card is missing")
            if uid not in field:
                if allow_missing:
                    continue
                raise EffectInterpreterError(f"{target_ref} target must be on {player} field")
            valid.append(uid)
        return tuple(valid)

    def _resolve_filtered_targets(self, snapshot: BattleSnapshot, filters: Any, context: EffectInterpretationContext | None) -> tuple[int, ...]:
        normalized = tuple(dict(item) for item in filters or ())
        if not normalized:
            raise EffectInterpreterError("filtered target requires filters")
        zone = _filter_value(normalized, "zone") or "field"
        if zone not in {"field", "hand"}:
            raise EffectInterpreterError(f"unsupported filtered zone: {zone}")
        side = _filter_value(normalized, "side") or "self"
        players = ("me", "opponent") if side == "both" else (context.opponent if side in {"enemy", "opponent"} and context is not None else "opponent",) if side in {"enemy", "opponent"} else (context.player if context is not None else "me",)
        uids: list[int] = []
        for player in players:
            player_view = snapshot.me if player == "me" else snapshot.opponent
            uids.extend(getattr(player_view, zone))
        cards = [snapshot.cards_by_uid[uid] for uid in uids if uid in snapshot.cards_by_uid]
        for item in normalized:
            if item.get("exclude") == "source_card":
                if context is None or context.source_uid is None:
                    raise EffectInterpreterError("exclude source_card requires source_uid")
                cards = [card for card in cards if card.uid != context.source_uid]
                continue
            field = item.get("field")
            operator = item.get("operator", "==")
            value = item.get("value")
            if field in {"zone", "side"}:
                continue
            if field == "card_type":
                cards = [card for card in cards if _compare_value(card.card_type.value, operator, value)]
            elif field in {"cost", "original_cost"}:
                cards = [card for card in cards if _compare_value(card.cost, operator, value)]
            elif field == "base_card_id":
                cards = [card for card in cards if _compare_value(card.base_card_id, operator, value)]
            elif field == "atk":
                if isinstance(value, dict) and value.get("kind") == "pool_extreme":
                    if cards:
                        extreme = max(card.atk for card in cards) if value.get("mode") == "max" else min(card.atk for card in cards)
                        cards = [card for card in cards if card.atk == extreme]
                else:
                    cards = [card for card in cards if _compare_value(card.atk, operator, value)]
            elif field == "life":
                if isinstance(value, dict) and value.get("kind") == "pool_extreme":
                    if cards:
                        extreme = max(card.life for card in cards) if value.get("mode") == "max" else min(card.life for card in cards)
                        cards = [card for card in cards if card.life == extreme]
                else:
                    cards = [card for card in cards if _compare_value(card.life, operator, value)]
            elif field == "tribes":
                cards = [card for card in cards if _compare_value(_effective_tribes(card), operator, value)]
            elif field == "keyword":
                cards = [card for card in cards if _compare_value(_keyword_values(card), operator, value)]
            elif field == "class_name":
                cards = [card for card in cards if _compare_value(self._class_name(card), operator, value)]
            elif field == "evolved":
                # evolved filter: check if card's evolve_state > 0
                # value is typically true/false boolean
                cards = [card for card in cards if _compare_value(card.evolve_state > 0, operator, value)]
            else:
                raise EffectInterpreterError(f"unsupported filtered field: {field}")
        return tuple(card.uid for card in cards)

    def _filter_generated_cards(self, cards: tuple[CardView, ...], filters: tuple[Any, ...]) -> tuple[int, ...]:
        result = list(cards)
        for item in filters:
            if not isinstance(item, dict):
                raise EffectInterpreterError("history_same_name_pool filters must be objects")
            field = item.get("field")
            operator = item.get("operator", "==")
            value = item.get("value")
            if field == "tribes":
                result = [card for card in result if _compare_value(_effective_tribes(card), operator, value)]
            elif field == "card_type":
                if str(operator) in {"==", "="} and str(value) == "amulet":
                    result = [card for card in result if card.card_type.is_amulet_like]
                else:
                    result = [card for card in result if _compare_value(card.card_type.value, operator, value)]
            elif field in {"cost", "original_cost"}:
                result = [card for card in result if _compare_value(card.cost, operator, value)]
            elif field == "keyword":
                result = [card for card in result if _compare_value(_keyword_values(card), operator, value)]
            elif field == "has_last_words":
                result = [card for card in result if _compare_value(1 if card.has_last_words else 0, operator, 1 if value else 0)]
            else:
                raise EffectInterpreterError(f"unsupported history_same_name_pool filter field: {field}")
        return tuple(card.uid for card in result)

    def _draw(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        count = self._amount(operation, default=1, snapshot=snapshot, context=context)
        current = snapshot
        log = RollbackTransactionLog()
        events: list[ExecutionEvent] = []
        # 抽卡玩家由 operation.target_ref 决定（如「对手抽取1张」target_ref=opponent_player），
        # 不能一律抽给效果发动者。默认（无 player target_ref）抽给 context.player。
        draw_player = self._operation_player(operation, context)
        # source_pool filter: 从牌库中筛选满足条件的卡来抽
        source_pool = operation.params.get("source_pool") if operation.params else None
        use_source_pool_filter = isinstance(source_pool, dict) and source_pool.get("zone") == "deck"
        # draw 从牌库抽牌，不消耗 generated_cards（generated_cards 留给 summon/add_to_hand 用）
        # draw_bindings 是为效果发动者（context.player）预置的已知抽牌序列，
        # 抽给对手时不适用（那是对手牌库的牌），此时 card 走对手 deck_instances 自动取牌。
        use_bindings = draw_player == context.player
        if use_bindings:
            # 绑定池按「效果持有者侧」选：视角互换后 context.player 是 opponent，
            # 此时它的新增卡在 opponent_draw_bindings 里，不能读 me 侧那组。
            # 取**完整**绑定序列而不是前 count 条：一次效果抽多张时必须依次使用
            # 多条绑定。只取前 count 条时，第一张抽完进入手牌，第二张的候选又被
            # visible_uids 过滤成空，于是退化成 hidden draw——deck 减 1 但手牌
            # 不变（replay_...1735553820 st77：两个『同悦之钟』各抽 1 张，
            # 第二张没进手牌，deck 21->19 但 hand 只 +1）。已用过的绑定由
            # visible_uids 排除，循环内按 index 取第 index 条。
            if context.player == "me":
                raw_bindings = context.draw_bindings or context.execution_context.draw_bindings
            else:
                raw_bindings = context.execution_context.opponent_draw_bindings
            # 只排除已经可见（手牌/场上）的实例；deck 中的卡在 replay snapshot 里
            # 也可能有 uid，不能因为 uid 在 cards_by_uid 就丢弃 draw binding。
            visible_uids = set(current.me.hand) | set(current.me.field) | set(current.opponent.hand) | set(current.opponent.field)
            bindings = [binding for binding in raw_bindings if binding.card.uid not in visible_uids]
        else:
            bindings = []
        drawn_uids_so_far: set[int] = set()
        source_pool_filters = tuple(source_pool.get("filters", ()) or ()) if isinstance(source_pool, dict) else ()
        for index in range(count):
            binding = bindings[index] if index < len(bindings) else None
            if binding is not None and use_source_pool_filter:
                # 抽牌绑定只是「本 step 新增的手牌卡」，不等于「本效果抽到的那张」：
                # 绑定必须满足本效果的 source_pool 条件，否则不能用它（用 after 帧的
                # 结果绕过卡面条件）。replay_...749786874 st72：10974110 的进化抽
                # 「1 张拥有【毁灭】的超越者·随从」绑到了普通随从 10804110，
                # 本该「牌库没有符合条件的卡 → 不抽」，结果多抽 1 张、手牌溢出爆牌，
                # 墓地比录像多 1。
                if not self._card_matches_source_pool_filters(binding.card, list(source_pool_filters)):
                    binding = None
            if binding is not None:
                card = binding.card
            elif use_source_pool_filter:
                card = self._find_filtered_deck_card(current, draw_player, source_pool, exclude_uids=drawn_uids_so_far)
            else:
                card = None
            # source_pool 检索（如「抽1张拥有【毁灭】的超越者·随从」）在牌库中找不到
            # 匹配卡时，卡面语义是「没有符合条件的卡」——不能回落成抽牌库顶牌。
            # 曾因此多抽 1 张：replay_...749786874 step72 的 10974110 进化抽取本应失败，
            # 却把牌库顶牌抽了出来（deck 19->16 三张，录制只 2 张）。
            if use_source_pool_filter and card is None:
                break
            # card=None 时 kernel.draw_card 会自动从 deck_instances 取牌
            result = self.kernel.draw_card(current, draw_player, card)
            current = log.apply_all(current, tuple(applied.forward for applied in result.transaction_log.applied))
            drawn_uid = next(((event.target_uid if event.target_uid is not None else event.source_uid) for event in reversed(result.events) if event.kind is EventKind.DRAW and (event.target_uid is not None or event.source_uid is not None)), None)
            if drawn_uid is not None:
                drawn_uids_so_far.add(drawn_uid)
            if drawn_uid is not None and drawn_uid in current.cards_by_uid:
                current = log.apply_all(current, (mark_known_generated_hand_card(current.cards_by_uid[drawn_uid], player=draw_player, source_uid=context.source_uid),))
            if binding is not None and binding.instance_id is not None:
                current = self._apply_deck_instance_draw_binding(current, draw_player, binding)
            elif card is not None and (binding is None or binding.instance_id is None):
                # 指定具体 card 抽牌(source_pool 检索 / binding 无 instance_id):
                # kernel.draw_card(card≠None) 已减 deck_count,但 deck_instances 的 move 分支仅在
                # card is None 时执行,未把该 instance 从 draw_pile 移除 → deck_count 与 deck_instances
                # 失配(深搜 remaining_deck 校验报 DeckFeatureError)。补 move,且必须经 log.apply_all
                # 让该 transaction 进入 transaction_log,否则上层 apply_operations 重放 log 时会丢弃此 move。
                sync_tx = self._deck_instance_move_tx_after_specified_draw(current, draw_player, card)
                if sync_tx is not None:
                    current = log.apply_all(current, (sync_tx,))
            events.extend(result.events)
        return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))

    def _deck_instance_move_tx_after_specified_draw(self, snapshot: BattleSnapshot, player: str, card: CardView):
        """返回把 card 对应 instance 从 draw_pile move 到 hand 的 SetPlayerField transaction(供 log 记录)。
        仅当该 instance 仍在 draw_pile 时返回,否则返回 None(不误动已离库的牌)。"""
        view = snapshot.me if player == "me" else snapshot.opponent
        deck_instances = view.deck_instances
        if deck_instances is None:
            return None
        instance = deck_instances.by_uid().get(card.uid)
        if instance is None or instance.instance_id not in deck_instances.draw_pile:
            return None
        moved = deck_instances.move(instance.instance_id, "hand", uid=card.uid, card=card, visibility="visible")
        return SetPlayerField(player, "deck_instances", moved)

    def _find_filtered_deck_card(self, snapshot: BattleSnapshot, player: str, source_pool: dict, *, exclude_uids: set[int] | None = None) -> CardView | None:
        """从牌库中找到满足 source_pool filters 条件的第一张卡。"""
        view = snapshot.me if player == "me" else snapshot.opponent
        deck_instances = view.deck_instances
        if deck_instances is None:
            return None
        filters = source_pool.get("filters", [])
        by_id = deck_instances.by_instance_id()
        for instance_id in deck_instances.draw_pile:
            instance = by_id.get(instance_id)
            if instance is None or instance.zone != "deck":
                continue
            card = instance.card
            uid = instance.uid if instance.uid is not None else -instance_id
            if exclude_uids and uid in exclude_uids:
                continue
            if self._card_matches_source_pool_filters(card, filters):
                return replace(card, uid=uid) if card.uid != uid else card
        return None

    def _card_matches_source_pool_filters(self, card: CardView, filters: list) -> bool:
        """检查卡是否满足所有 source_pool filters。"""
        for f in filters:
            if not isinstance(f, dict):
                continue
            field = f.get("field", "")
            operator = f.get("operator", "==")
            value = f.get("value")
            if field in ("side", "zone"):
                continue  # 调用方已确定
            if field == "card_type":
                card_type_str = card.card_type.value if hasattr(card.card_type, "value") else str(card.card_type)
                if operator == "==" and card_type_str != value:
                    return False
            elif field == "cost":
                card_cost = card.cost or 0
                if operator == ">=" and card_cost < (value or 0):
                    return False
                elif operator == "<=" and card_cost > (value or 0):
                    return False
                elif operator == "==" and card_cost != (value or 0):
                    return False
            elif field == "atk":
                if operator == ">=" and (card.atk or 0) < (value or 0):
                    return False
            elif field == "tribes":
                if operator == "contains" and value not in (card.tribes or ()):
                    return False
            elif field == "has_bane":
                # 「拥有【毁灭】的随从」这类检索条件。协议只在部分卡上下发 has_bane，
                # adapter 会用模板静态关键字补全，因此这里读 CardView 的最终值。
                if bool(card.has_bane) != bool(value):
                    return False
        return True

    def _apply_deck_instance_draw_binding(self, snapshot: BattleSnapshot, player: str, binding: DrawBinding) -> BattleSnapshot:
        view = snapshot.me if player == "me" else snapshot.opponent
        deck_instances = view.deck_instances
        if deck_instances is None or binding.instance_id is None:
            return snapshot
        updated = deck_instances.move(binding.instance_id, "hand", uid=binding.card.uid, card=binding.card, visibility="visible")
        return self.kernel.apply_transactions(snapshot, (SetPlayerField(player, "deck_instances", updated),)).snapshot

    def _deck_summon_transactions(self, snapshot: BattleSnapshot, player: str, generated_cards: tuple[CardView, ...]) -> tuple[SnapshotTransaction, ...]:
        """从牌组召唤时，返回将匹配 deck instance 从 draw_pile 转移到 field 的事务。"""
        view = snapshot.me if player == "me" else snapshot.opponent
        deck_instances = view.deck_instances
        if deck_instances is None:
            return ()
        by_id = deck_instances.by_instance_id()
        # 构建 draw_pile 中按 base_card_id 索引的 instance_id 列表（支持重复卡）
        available: dict[int, list[int]] = {}
        for instance_id in deck_instances.draw_pile:
            instance = by_id.get(instance_id)
            if instance is None or instance.zone != "deck":
                continue
            available.setdefault(instance.card.base_card_id, []).append(instance_id)
        updated = deck_instances
        moved_count = 0
        for card in generated_cards:
            candidates = available.get(card.base_card_id)
            if not candidates:
                # 也尝试按 card_id 匹配
                for instance_id in list(updated.draw_pile):
                    inst = by_id.get(instance_id)
                    if inst is not None and inst.zone == "deck" and inst.card.card_id == card.card_id:
                        candidates = [instance_id]
                        break
            if not candidates:
                continue
            instance_id = candidates.pop(0)
            updated = updated.move(instance_id, "field", uid=card.uid, card=card, visibility="visible")
            moved_count += 1
        if moved_count == 0:
            return ()
        return (SetPlayerField(player, "deck_instances", updated), ModifyPlayerCounter(player, "deck_count", -moved_count))

    def _spellboost(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        amount = self._amount(operation, default=1, snapshot=snapshot, context=context)
        target_ref = operation.target_ref or ""
        if target_ref == "all_self_hand_cards" or operation.op == "spellboost_hand":
            hand_uids = snapshot.me.hand if context.player == "me" else snapshot.opponent.hand
            target_uids = tuple(
                uid for uid in hand_uids
                if snapshot.cards_by_uid[uid].has_spell_boost or snapshot.cards_by_uid[uid].variable_x > 0
            )
        else:
            target_uids = self._target_uids(snapshot, operation, context)
        transactions: list[SnapshotTransaction] = []
        for uid in target_uids:
            card = snapshot.cards_by_uid[uid]
            transactions.append(ModifyCardCounter(uid, "spell_boost_count", amount))
            if card.spellboost_reduces_cost:
                transactions.append(SetCardField(uid, "cost", max(0, card.cost - amount)))
            elif card.variable_x > 0 or (card.has_spell_boost and not card.spellboost_reduces_cost):
                variable_x = card.variable_x + amount
                transactions.append(SetCardField(uid, "variable_x", variable_x))
                transactions.append(SetCardField(uid, "supplement_info", {**card.supplement_info, "variable_x": variable_x}))
        # 「发动 N 次魔力增幅」要发 N 个事件：spellboost 反应效果（【魔力增幅时】
        # 使本卡牌+1/+1、费用-1 等）的 spec 都按「每次 +1」建模，只发一个
        # amount=N 的事件会让它们只结算 1 次。
        # 录像证据：replay_...992331 st51 的 10832310（发动 2 次）——
        # uid 60 水之法则·瓦姆杜斯 sb 11->13（+2）、atk/life 11/12->13/14（+2），
        # uid 79 破式的变貌 sb 7->9、cost 3->1；本地按 1 次结算，atk/life 只 +1。
        events = tuple(
            ExecutionEvent(EventKind.SPELLBOOST, source_uid=context.source_uid, target_uid=uid, payload={"amount": 1})
            for uid in target_uids
            for _ in range(max(int(amount), 0))
        )
        return self._run(snapshot, tuple(transactions), events)

    def _summon(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        # 「在对手的战场上召唤」类文本：target_ref 明确指定对手半场时，召唤归属
        # 是 context.player 的对手，而不是效果发动者。
        summon_player = context.player
        if operation.target_ref in {"opponent_field", "enemy_field"}:
            summon_player = "opponent" if context.player == "me" else "me"
        if operation.params.get("copy_source") in {"selected_source_pool_card", "selected_self_hand_card"}:
            if not context.generated_cards:
                raise EffectInterpreterError("summon copy_source selected hand card requires generated_cards")
            targets = self._target_uids(snapshot, OperationSpec(op="copy_card", target_ref="selected_self_hand_card", params={}), context)
            if len(targets) != len(context.generated_cards):
                raise EffectInterpreterError("summon copy_source selected hand card count mismatch")
            current = snapshot
            log = RollbackTransactionLog()
            events: list[ExecutionEvent] = []
            for target_uid, generated in zip(targets, context.generated_cards, strict=True):
                if generated.uid in current.cards_by_uid:
                    raise EffectInterpreterError("summon copy_source selected hand card generated uid already exists")
                copied = replace(snapshot.cards_by_uid[target_uid], uid=generated.uid)
                result = self.kernel.summon_card(current, summon_player, copied)
                current = log.apply_all(current, tuple(applied.forward for applied in result.transaction_log.applied)) if result.transaction_log.applied else result.snapshot
                events.extend((ExecutionEvent(EventKind.COPY_CARD, source_uid=target_uid, target_uid=copied.uid, player=summon_player), *result.events))
            return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))
        current = snapshot
        log = RollbackTransactionLog()
        events: list[ExecutionEvent] = []
        remove_kws = operation.params.get("remove_keywords") or ()
        kw_field_map = {
            "storm": "has_storm", "rush": "has_rush", "guard": "has_guard",
            "bane": "has_bane", "drain": "has_drain", "last_words": "has_last_words",
            "cant_attack": "has_cant_attack", "ambush": "has_ambush",
        }
        summoned_uids: list[int] = []
        for card in context.generated_cards:
            if card.uid in current.cards_by_uid:
                continue
            # 【证据】护符在场上唯一：召唤新证据时，移除同侧已有的证据护符。
            # replay 证据：召唤 90001210 后旧物证被 RemoveCard，并伴随
            # BattleResponseEvidence；旧证据不进入墓地。
            from .generated_card_materializer import default_card_resolver

            resolver = default_card_resolver()

            def _has_evidence_tribe(target: CardView) -> bool:
                tribes = tuple(target.tribes or ())
                if 21 in tribes:
                    return True
                base = resolver.get_base_by_card_id(target.card_id)
                return base is not None and 21 in tuple(base.tribes or ())

            if _has_evidence_tribe(card):
                view = current.me if summon_player == "me" else current.opponent
                for old_uid in view.field:
                    old = current.cards_by_uid.get(old_uid)
                    if old is None or old_uid == card.uid or not _has_evidence_tribe(old):
                        continue
                    current = log.apply_all(current, (
                        RemoveZoneUid(summon_player, "field", old_uid),
                        RemoveCard(old_uid, old),
                    ))
                    # 「离场」必须作为事件发出：否则 11003210 这类
                    # 「本卡牌在战场上消失时，回复自己的主战者2点生命值」的
                    # on_self_card_left_field 效果永远不会触发。
                    # 证据被替换是「消失」而非「破坏」：不进墓地、不触发谢幕曲，
                    # 因此发 BANISH（registry 的 banish alias 只含 on_self_card_left_field）。
                    events.append(ExecutionEvent(EventKind.BANISH, source_uid=old_uid, target_uid=old_uid, player=summon_player))
            result = self.kernel.summon_card(current, summon_player, card)
            current = log.apply_all(current, tuple(applied.forward for applied in result.transaction_log.applied)) if result.transaction_log.applied else result.snapshot
            events.extend(result.events)
            summoned_uids.append(card.uid)
        # 召唤后应用 remove_keywords（在 sync 之后也能覆盖）
        if remove_kws and summoned_uids:
            remove_transactions: list[SnapshotTransaction] = []
            for uid in summoned_uids:
                for kw in remove_kws:
                    field_name = kw_field_map.get(kw)
                    if field_name:
                        remove_transactions.append(SetCardField(uid, field_name, False))
            if remove_transactions:
                current = log.apply_all(current, tuple(remove_transactions))
        # 召唤后应用 stat_buff（「召唤1个X，使其+X/+Y」）：
        # 与 remove_keywords 同理，必须在 summon 之后对 summoned_uids 应用。
        # 10633310 魔恋的天晶 / 90034330 天晶深渊都依赖它。
        stat_buff = operation.params.get("stat_buff")
        if isinstance(stat_buff, dict) and summoned_uids:
            # atk/life 允许写成 AmountFormula 形态（如 {kind: snapshot_value,
            # variable: me.faith_value}），供「使其 +X/+Y，X 为信仰值」这类文本使用。
            buff_atk = self._resolve_amount_value(stat_buff.get("atk", 0) or 0, snapshot=current, context=context)
            buff_life = self._resolve_amount_value(stat_buff.get("life", 0) or 0, snapshot=current, context=context)
            if buff_atk or buff_life:
                buff_transactions: list[SnapshotTransaction] = []
                for uid in summoned_uids:
                    buffed = current.cards_by_uid.get(uid)
                    if buffed is None:
                        continue
                    if buff_atk:
                        buff_transactions.append(SetCardField(uid, "atk", int(buffed.atk) + buff_atk))
                    if buff_life:
                        buff_transactions.append(SetCardField(uid, "life", int(buffed.life) + buff_life))
                        buff_transactions.append(SetCardField(uid, "max_life", int(buffed.max_life) + buff_life))
                if buff_transactions:
                    current = log.apply_all(current, tuple(buff_transactions))
        # 从牌组召唤时，将对应实例从 draw_pile 转移到 field
        source_pool = operation.params.get("source_pool")
        if isinstance(source_pool, dict) and source_pool.get("zone") == "deck" and context.generated_cards:
            deck_transactions = self._deck_summon_transactions(current, context.player, context.generated_cards)
            if deck_transactions:
                current = log.apply_all(current, tuple(deck_transactions))
        return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))

    def _deck_ability_activation(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        if operation.target_ref != "self_deck":
            raise EffectInterpreterError("deck_ability_activation requires self_deck")
        return EffectInterpretationResult(
            snapshot=snapshot,
            transaction_log=RollbackTransactionLog(),
            events=(ExecutionEvent("deck_ability_activated", source_uid=context.source_uid, player=context.player),),
        )

    def _invoke_self(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        if operation.params.get("source") != "deck":
            raise EffectInterpreterError("invoke_self requires source=deck")
        if operation.params.get("trigger_fanfare") is not False:
            raise EffectInterpreterError("invoke_self requires explicit trigger_fanfare=false")
        view = snapshot.me if context.player == "me" else snapshot.opponent
        deck_instances = view.deck_instances
        if deck_instances is None or deck_instances.status != "available":
            raise EffectInterpreterError("invoke_self requires available deck_instances")
        target_ids = [
            instance_id for instance_id in deck_instances.draw_pile
            if (instance := deck_instances.by_instance_id().get(instance_id)) is not None
            and instance.zone == "deck"
            and instance.card.card_type.is_follower
            and (context.source_card is None or instance.card.base_card_id == context.source_card.base_card_id)
        ]
        if not target_ids:
            raise EffectInterpreterError("invoke_self requires matching follower in deck")
        instance_id = target_ids[0]
        instance = deck_instances.by_instance_id()[instance_id]
        uid = instance.uid if instance.uid is not None else -instance_id
        card = replace(instance.card, uid=uid)
        if len(view.field) >= 5:
            raise EffectInterpreterError("invoke_self field is full")
        updated = deck_instances.move(instance_id, "field", uid=uid, card=card, visibility="visible")
        # 如果 uid 已存在于 cards_by_uid（LocalMatchSetup 预先注册了所有 deck 卡），
        # 不需要 AddCard，只更新卡数据即可
        if uid in snapshot.cards_by_uid:
            transactions: list[SnapshotTransaction] = [SetCardFull(uid, card)]
        else:
            transactions: list[SnapshotTransaction] = [AddCard(card)]
        transactions.append(InsertZoneUid(context.player, "field", uid))
        if card.card_type.is_follower:
            transactions.extend((ModifyPlayerCounter(context.player, "rally", 1), ModifyPlayerCounter(context.player, "followers_entered_battlefield", 1)))
            entered_tx = self.kernel._record_entered_field_transaction(snapshot, context.player, card.base_card_id)
            if entered_tx is not None:
                transactions.append(entered_tx)
        transactions.extend((SetPlayerField(context.player, "deck_instances", updated), ModifyPlayerCounter(context.player, "deck_count", -1)))
        events = (
            ExecutionEvent(EventKind.SUMMON, player=context.player, source_uid=uid),
            ExecutionEvent(EventKind.ENTER_FIELD, player=context.player, source_uid=uid),
        )
        return self._run(snapshot, tuple(transactions), events)

    def _bounce(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        targets = self._target_uids(snapshot, operation, context)
        if not targets:
            raise EffectInterpreterError("bounce requires target cards")
        current = snapshot
        log = RollbackTransactionLog()
        events: list[ExecutionEvent] = []
        for uid in targets:
            owner = self._owner_of_field_uid(current, uid)
            if owner is None:
                # invoke_self 可能改变了 uid（从 deck instance 取了新 uid）
                # 尝试用 source_card.base_card_id 在 field 上找到对应的卡
                if context.source_card is not None and uid == context.source_uid:
                    for field_uid in (*current.me.field, *current.opponent.field):
                        card = current.cards_by_uid.get(field_uid)
                        if card is not None and card.base_card_id == context.source_card.base_card_id:
                            uid = field_uid
                            owner = self._owner_of_field_uid(current, uid)
                            break
                if owner is None:
                    continue  # 目标不在场上，跳过（而非报错）
            result = self.kernel.bounce_field_card(current, owner, uid)
            current = log.apply_all(current, tuple(applied.forward for applied in result.transaction_log.applied)) if result.transaction_log.applied else result.snapshot
            view = current.me if owner == "me" else current.opponent
            deck_instances = view.deck_instances
            if deck_instances is not None:
                instance = deck_instances.by_uid().get(uid)
                if instance is not None:
                    hand_card = current.cards_by_uid.get(uid)
                    if hand_card is None:
                        raise EffectInterpreterError("bounce deck instance card is missing after kernel move")
                    updated = deck_instances.move(instance.instance_id, "hand", uid=uid, card=hand_card, visibility="visible")
                    current = log.apply_all(current, (SetPlayerField(owner, "deck_instances", updated),))
            events.extend(result.events)
        return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))

    def _return_to_deck(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        is_copy = bool(operation.params.get("copy"))
        player_view = snapshot.me if context.player == "me" else snapshot.opponent

        # copy 模式：直接将指定 card_ids 的副本加入牌组，不从手牌移除
        if is_copy:
            card_ids = operation.params.get("card_ids", ())
            if not card_ids:
                # fallback: 使用 source_card 的 card_id
                if context.source_card is not None:
                    card_ids = [context.source_card.base_card_id]
            count = int(operation.params.get("count", 1) or 1)
            transactions: list[SnapshotTransaction] = []
            deck_instances = player_view.deck_instances
            if deck_instances is not None:
                import random as _random
                rng = _random.Random(context.execution_context.branch_seed if context.execution_context else 0)
                for _ in range(count):
                    for card_id in card_ids:
                        # 创建新 instance 加入卡组
                        next_id = max((inst.instance_id for inst in deck_instances.instances), default=0) + 1
                        from decision_engine.state.deck_instance import DeckInstance
                        base = self.card_resolver.get_base(card_id) if self.card_resolver else None
                        from decision_engine.state.snapshot import CardView, CardType
                        _TYPE_MAP = {1: CardType.FOLLOWER, 2: CardType.AMULET, 3: CardType.COUNTDOWN_AMULET, 4: CardType.SPELL, 5: CardType.CREST, 6: CardType.COUNTDOWN_CREST, 7: CardType.FAITH}
                        new_card = CardView(
                            uid=-next_id,
                            card_id=card_id,
                            base_card_id=card_id,
                            name=base.name if base else "",
                            cost=base.cost if base else 0,
                            atk=base.atk if base else 0,
                            life=base.life if base else 0,
                            card_type=_TYPE_MAP.get(base.type_id, CardType.FOLLOWER) if base else CardType.FOLLOWER,
                        )
                        new_instance = DeckInstance(
                            instance_id=next_id,
                            card=new_card,
                            owner=context.player,
                            zone="deck",
                            uid=-next_id,
                            origin="return_to_deck_copy",
                            order_hint=next_id,
                            visibility="hidden",
                        )
                        instances = (*deck_instances.instances, new_instance)
                        # 插入 draw_pile 随机位置
                        draw_pile = list(deck_instances.draw_pile)
                        insert_pos = rng.randint(0, len(draw_pile)) if draw_pile else 0
                        draw_pile.insert(insert_pos, next_id)
                        deck_instances = replace(deck_instances, instances=instances, draw_pile=tuple(draw_pile))
                transactions.append(SetPlayerField(context.player, "deck_instances", deck_instances))
                transactions.append(ModifyPlayerCounter(context.player, "deck_count", count * len(card_ids)))
            else:
                transactions.append(ModifyPlayerCounter(context.player, "deck_count", count * len(card_ids) if card_ids else count))
            return self._run(snapshot, tuple(transactions), ())

        # 非 copy 模式：从手牌移除并放回卡组
        targets = self._target_uids(snapshot, operation, context)
        if not targets:
            # “自己的所有手牌返回牌组”在打出最后一张手牌后可能为空集。
            # 空集全体操作应保持合法 no-op；仅需要显式目标的操作才报错。
            if operation.target_ref == "all_self_hand_cards":
                return self._run(snapshot, (), ())
            raise EffectInterpreterError("return_to_deck requires target cards")
        hand_positions = {uid: index for index, uid in enumerate(player_view.hand)}
        transactions: list[SnapshotTransaction] = []
        deck_instances = player_view.deck_instances
        if deck_instances is not None:
            by_uid = deck_instances.by_uid()
            for uid in sorted(targets, key=lambda item: hand_positions.get(item, -1)):
                instance = by_uid.get(uid)
                if instance is not None:
                    deck_instances = deck_instances.move(instance.instance_id, "deck", visibility="hidden")
                else:
                    # 手牌中存在但 deck_instances 未追踪的牌(如效果生成的卡)返回牌库时，
                    # 必须为其创建新的 deck instance，否则 deck_instances 不增加而下方 deck_count+1，
                    # 造成 deck_count 与 deck_instances 失配(深搜 state_features remaining_deck 校验报 DeckFeatureError)。
                    card = snapshot.cards_by_uid.get(uid)
                    if card is not None:
                        from decision_engine.state.deck_instance import DeckInstance
                        next_id = max((inst.instance_id for inst in deck_instances.instances), default=0) + 1
                        new_instance = DeckInstance(
                            instance_id=next_id,
                            card=replace(card, uid=-next_id),
                            owner=context.player,
                            zone="deck",
                            uid=-next_id,
                            origin="return_to_deck_untracked",
                            order_hint=next_id,
                            visibility="hidden",
                        )
                        instances = (*deck_instances.instances, new_instance)
                        draw_pile = (*deck_instances.draw_pile, next_id)
                        deck_instances = replace(deck_instances, instances=instances, draw_pile=draw_pile)
            transactions.append(SetPlayerField(context.player, "deck_instances", deck_instances))
        return_events: list[ExecutionEvent] = []
        for uid in sorted(targets, key=lambda item: hand_positions.get(item, -1), reverse=True):
            index = hand_positions.get(uid)
            if index is None:
                raise EffectInterpreterError("return_to_deck target must be in hand")
            transactions.append(RemoveZoneUidAt(context.player, "hand", index, uid))
            transactions.append(ModifyPlayerCounter(context.player, "deck_count", 1))
            transactions.append(RemoveCard(uid))
            return_events.append(ExecutionEvent(EventKind.RETURN_TO_DECK, player=context.player, source_uid=uid))
        return self._run(snapshot, tuple(transactions), tuple(return_events))

    def _transform(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        targets = self._target_uids(snapshot, operation, context)
        if not targets:
            # 「使手牌中所有同名卡变身」在没有同名卡时是合法 noop。
            return EffectInterpretationResult(snapshot=snapshot, transaction_log=RollbackTransactionLog(), events=())
        if not context.generated_cards:
            raise EffectInterpreterError("transform requires generated_cards")
        if len(targets) == 1:
            return self._transform_one(snapshot, operation, context, targets[0], context.generated_cards[0], None)
        # 多目标变身（如 10741310「使自己的手牌中的所有『百无聊赖的睥睨』变身」）：
        # 逐个目标替换，replacement 按 generated_cards 顺序取（materializer 按
        # generated_card_count 物化足量副本）。单个目标失败不影响其余目标。
        current = snapshot
        log = RollbackTransactionLog()
        events: list[ExecutionEvent] = []
        for index, target_uid in enumerate(targets):
            if index >= len(context.generated_cards):
                break
            try:
                result = self._transform_one(current, operation, context, target_uid, context.generated_cards[index], log)
            except EffectInterpreterError:
                continue
            current = result.snapshot
            events.extend(result.events)
        return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))

    def _transform_one(
        self,
        snapshot: BattleSnapshot,
        operation: OperationSpec,
        context: EffectInterpretationContext,
        target_uid: int,
        replacement: "CardView",
        shared_log,
    ) -> EffectInterpretationResult:
        if not self._operation_condition_met(snapshot, operation, context, target_uid):
            return EffectInterpretationResult(snapshot=snapshot, transaction_log=RollbackTransactionLog(), events=())
        owner = self._owner_of_field_uid(snapshot, target_uid)
        if owner is not None:
            result = self.kernel.transform_field_card(snapshot, owner, target_uid, replacement)
            if shared_log is not None and result.transaction_log.applied:
                current = shared_log.apply_all(snapshot, tuple(applied.forward for applied in result.transaction_log.applied))
                return EffectInterpretationResult(snapshot=current, transaction_log=shared_log, events=result.events)
            return EffectInterpretationResult(snapshot=result.snapshot, transaction_log=result.transaction_log, events=result.events)
        hand_owner = self._owner_of_hand_uid(snapshot, target_uid)
        if hand_owner is None:
            raise EffectInterpreterError("transform target must be on field or in hand")
        player_view = snapshot.me if hand_owner == "me" else snapshot.opponent
        hand_index = player_view.hand.index(target_uid)
        old_card = snapshot.cards_by_uid[target_uid]
        if replacement.uid == target_uid:
            raise EffectInterpreterError("transform replacement must use a new uid")
        if replacement.uid in snapshot.cards_by_uid:
            raise EffectInterpreterError("transform replacement uid already exists")
        transactions = (
            RemoveZoneUidAt(hand_owner, "hand", hand_index, target_uid),
            RemoveCard(target_uid, old_card),
            AddCard(replacement),
            InsertZoneUid(hand_owner, "hand", replacement.uid, hand_index),
        )
        log = shared_log if shared_log is not None else RollbackTransactionLog()
        current = log.apply_all(snapshot, transactions)
        return EffectInterpretationResult(
            snapshot=current,
            transaction_log=log,
            events=(ExecutionEvent(EventKind.TRANSFORM, source_uid=target_uid, target_uid=replacement.uid, player=hand_owner),),
        )

    def _apply_field_card_operation(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext, kind: str) -> EffectInterpretationResult:
        current = snapshot
        log = RollbackTransactionLog()
        events: list[ExecutionEvent] = []
        for uid in self._target_uids(current, operation, context):
            if uid not in current.cards_by_uid:
                continue
            owner = self._owner_of_field_uid(current, uid)
            if owner is None:
                continue
            if kind == "destroy":
                if current.cards_by_uid[uid].has_cant_destroy:
                    continue
                # 超进化随从在**自己的回合**不会被能力破坏
                # （shadowverse-rules「超进化」：不会被能力破坏，且不会受到战斗伤害）。
                # 伤害致死那条路径已由 effective_damage_to_follower 归零，
                # 但 destroy 是「直接破坏」、不经过伤害，必须单独挡一次。
                if is_super_evolved_on_own_turn(current, current.cards_by_uid[uid], owner):
                    continue
                result = self.kernel.destroy_field_card(current, owner, uid)
            elif kind == "banish":
                result = self.kernel.banish_field_card(current, owner, uid)
            elif kind == "bounce":
                result = self.kernel.bounce_field_card(current, owner, uid)
            else:
                raise EffectInterpreterError(f"unsupported field operation: {kind}")
            if result.transaction_log.applied:
                current = log.apply_all(current, tuple(applied.forward for applied in result.transaction_log.applied))
            else:
                current = result.snapshot
            events.extend(result.events)
        return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))

    def _operation_player(self, operation: OperationSpec, context: EffectInterpretationContext) -> str:
        if operation.target_ref in {"opponent_player", "opponent_leader", "enemy_leader"}:
            return context.opponent
        return context.player

    def _add_to_hand(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        if not self._hand_is_observable(context):
            # 对手手牌是隐藏信息；本地不能把生成的卡物化进可见
            # opponent.hand，否则会与录像的隐藏手牌产生伪 diff。
            # 同理，replay 视角互换后当前视角的 me 就是录像对手，其手牌
            # 同样不可观测（hidden_hand_players 由 replay 上下文注入）。
            # 必须在 generated_cards 检查**之前**返回：手牌不可观测时既不需要
            # 录制生成物、也不该因为缺少它而报错
            # （replay_...1283403 st64 的 11024122_option_1）。
            return self._run(snapshot, (), ())
        if not context.generated_cards:
            raise EffectInterpreterError("add_to_hand requires generated_cards")
        player_view = snapshot.me if context.player == "me" else snapshot.opponent
        transactions: list[SnapshotTransaction] = []
        events: list[ExecutionEvent] = []
        hand_size = len(player_view.hand)
        count = self._amount(operation, default=int(operation.params.get("generated_card_count", len(context.generated_cards)) or len(context.generated_cards)), snapshot=snapshot, context=context)
        # replay 模式下 synthetic uid（负数，由 card_factory 为「临时生成的卡」分配）
        # 不代表录像里真实存在的卡：录像只暴露 after 帧手牌里的实际卡（正 uid）。
        # 把它当证据塞进手牌会留下幽灵卡（replay_...0025 st84：hand 末尾多 -102，
        # 来源是 10544122 纹章「每回合 1 次加 1 张大海虎鲸」被重复触发后，第二张
        # 没有录像绑定，于是用了 synthetic uid）。
        # 过滤后 len(cards) != count，下面会走「录像证据表明本次未加入」的 no-op 分支。
        cards = tuple(
            card
            for card in context.generated_cards
            if card.uid not in snapshot.cards_by_uid
            and (card.uid >= 0 or not context.execution_context.replay_mode)
        )[:count]
        overflow_count = 0
        if len(cards) != count:
            if context.execution_context.replay_mode and cards:
                # replay 的 after 手牌只提供了实际加入的卡（例如证据/同名卡上限
                # 导致只加入 1 张）；以录像证据为准，不静默伪造缺失卡。
                missing = count - len(cards)
                if hand_size + count > 9:
                    # 手牌空间不足以放进全部生成卡：录像 after 手牌少掉的张数
                    # 就是被 overdraw 直接送进墓地的张数（11073110 类）。
                    overflow_count = missing
                count = len(cards)
            elif context.execution_context.replay_mode:
                # after 手牌没有任何新卡：录像证据表明这次「加入手牌」没有发生
                # （手牌已满等），按已知 no-op 处理，不伪造卡牌。
                return self._run(snapshot, (), ())
            else:
                raise EffectInterpreterError("add_to_hand generated_cards count mismatch")
        # set_cost_to: 加入手牌时把生成卡的费用覆盖为指定值（如燃尽之缘费用变1、
        # 波涛副船长超进化黄金短剑/项链费用变0）。
        set_cost_to = operation.params.get("set_cost_to")
        if set_cost_to is not None:
            cards = tuple(replace(card, cost=int(set_cost_to)) for card in cards)
        # stat_buff: 加入手牌时对生成卡施加固定攻击/生命增量(如人性的爱: 手牌那张
        # 沉溺的实验体 +1/0)。max_life 不跟随增量——手牌卡 max_life 恒为 0。
        buff_atk = int(operation.params.get("stat_buff_attack", 0) or 0)
        buff_life = int(operation.params.get("stat_buff_life", 0) or 0)
        if buff_atk or buff_life:
            # replay 模式下生成卡取自录像 after 帧，数值已含本次效果的结果；
            # 若卡上的 buff 已经记录了同等增量，再叠加就是双算
            # （10932310：after 帧 atk=3 + buff.atk=1，再 +1 会变 4）。
            def _apply_buff(card: CardView) -> CardView:
                if (
                    context.execution_context.replay_mode
                    and card.buff is not None
                    and card.buff.atk >= buff_atk
                    and card.buff.max_life_diff >= buff_life
                ):
                    return card
                return replace(card, atk=card.atk + buff_atk, life=card.life + buff_life)

            cards = tuple(_apply_buff(card) for card in cards)
        # 手牌区没有「最大生命」概念：录像里手牌卡的 max_life 恒为 0（life 仍有值）。
        # 生成卡 materialize 时按 base.life 设了 max_life，加手牌前必须归零，
        # 否则 diff 报 cards_by_uid[n].max_life 2 -> 0（10932310 人性的爱：
        # 手牌那张沉溺的实验体是 life=2/0）。
        cards = tuple(replace(card, max_life=0) for card in cards)
        # gain_keyword: 加入手牌时给生成卡赋予关键字(如根延潜伏者连击3: 妖精获毁灭)。
        _kw = operation.params.get("gain_keyword")
        if _kw:
            _kw_field = {
                "storm": "has_storm", "rush": "has_rush", "guard": "has_guard",
                "bane": "has_bane", "drain": "has_drain", "ambush": "has_ambush",
                "last_words": "has_last_words", "cant_select": "has_cant_select",
            }.get(str(_kw))
            if _kw_field is not None:
                cards = tuple(replace(card, **{_kw_field: True}) for card in cards)
        for card in cards:
            if hand_size >= 9:
                transactions.extend((AddCard(card), ModifyPlayerCounter(context.player, "graveyard_count", 1), RemoveCard(card.uid, card)))
                events.append(ExecutionEvent(EventKind.OVERDRAW, source_uid=context.source_uid, target_uid=card.uid))
                continue
            transactions.extend((AddCard(card), InsertZoneUid(context.player, "hand", card.uid), mark_known_generated_hand_card(card, player=context.player, source_uid=context.source_uid)))
            hand_size += 1
            events.append(ExecutionEvent(EventKind.DRAW, source_uid=context.source_uid, target_uid=card.uid))
        if overflow_count:
            transactions.append(ModifyPlayerCounter(context.player, "graveyard_count", overflow_count))
            events.append(ExecutionEvent(EventKind.OVERDRAW, source_uid=context.source_uid, payload={"count": overflow_count}))
        return self._run(snapshot, tuple(transactions), tuple(events))

    def _put_token_to_hand(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        if not context.generated_cards:
            raise EffectInterpreterError("put_token_to_hand requires generated_cards")
        return self._add_to_hand(snapshot, operation, context)

    def _replace_hand_card(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        targets = self._target_uids(snapshot, operation, context)
        if len(targets) != 1:
            raise EffectInterpreterError("replace_hand_card requires exactly one target")
        if not context.generated_cards:
            raise EffectInterpreterError("replace_hand_card requires generated_cards")
        player_view = snapshot.me if context.player == "me" else snapshot.opponent
        uid = targets[0]
        if uid not in player_view.hand:
            raise EffectInterpreterError("replace_hand_card target must be in hand")
        index = player_view.hand.index(uid)
        replacement = context.generated_cards[0]
        transactions = (
            RemoveZoneUidAt(context.player, "hand", index, uid),
            RemoveCard(uid, snapshot.cards_by_uid[uid]),
            AddCard(replacement),
            InsertZoneUid(context.player, "hand", replacement.uid, index),
        )
        return self._run(snapshot, transactions, (ExecutionEvent(EventKind.REPLACE_HAND_CARD, source_uid=uid, target_uid=replacement.uid),))

    def _copy_card(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        targets = self._target_uids(snapshot, operation, context)
        if len(targets) != 1:
            raise EffectInterpreterError("copy_card requires exactly one target")
        if not context.generated_cards:
            raise EffectInterpreterError("copy_card requires generated_cards")
        target = snapshot.cards_by_uid[targets[0]]
        generated = context.generated_cards[0]
        if generated.uid in snapshot.cards_by_uid:
            raise EffectInterpreterError("copy_card generated uid already exists")
        copied = replace(target, uid=generated.uid)
        if operation.params.get("use_base_stats"):
            # 「将1张与其同名的卡牌加入手牌」：加入的是卡牌定义本身，不是场上
            # 实例（不继承进化/受伤/buff 后的身材）。手牌卡协议 max_life 为 0。
            base = self.card_resolver.get_base_by_card_id(target.base_card_id)
            if base is not None:
                from decision_engine.state.snapshot import CardType as _CardType

                type_map = {
                    1: _CardType.FOLLOWER,
                    2: _CardType.AMULET,
                    3: _CardType.COUNTDOWN_AMULET,
                    4: _CardType.SPELL,
                    5: _CardType.CREST,
                    6: _CardType.COUNTDOWN_CREST,
                    7: _CardType.FAITH,
                }
                copied = CardView(
                    uid=generated.uid,
                    card_id=base.base_card_id,
                    base_card_id=base.base_card_id,
                    # 协议手牌卡不带 name；保持空串以与 after 快照一致。
                    name="",
                    cost=base.cost,
                    atk=base.atk,
                    life=base.life,
                    max_life=0,
                    card_type=type_map.get(int(getattr(base, "type_id", 1) or 1), _CardType.FOLLOWER),
                    tribes=tuple(getattr(base, "tribes", ()) or ()),
                )
        cost_delta = int(operation.params.get("cost_delta", 0) or 0)
        if cost_delta:
            # 「将1张与其同名的卡牌加入手牌，使其费用-3」类效果：复制出的卡在
            # 原费用基础上调整，不能沿用原费用。
            copied = replace(copied, cost=max(0, copied.cost + cost_delta))
        target_ref = operation.params.get("destination", operation.target_ref or "")
        if target_ref in {"self_hand", "ally_hand", "hand"}:
            return self._run(snapshot, (AddCard(copied), InsertZoneUid(context.player, "hand", copied.uid)), (ExecutionEvent(EventKind.COPY_CARD, source_uid=targets[0], target_uid=copied.uid, player=context.player),))
        if target_ref in {"self_field", "ally_field", "field"}:
            result = self.kernel.summon_card(snapshot, context.player, copied)
            return EffectInterpretationResult(snapshot=result.snapshot, transaction_log=result.transaction_log, events=(ExecutionEvent(EventKind.COPY_CARD, source_uid=targets[0], target_uid=copied.uid, player=context.player), *result.events))
        raise EffectInterpreterError("copy_card destination must be self_hand or self_field")

    def _copy_from_opponent_hand(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        if not context.generated_cards:
            raise EffectInterpreterError("copy_from_opponent_hand requires generated_cards")
        if operation.target_ref:
            targets = self._target_uids(snapshot, operation, context)
            if len(targets) != 1:
                raise EffectInterpreterError("copy_from_opponent_hand requires exactly one target")
            copied_uid = targets[0]
        else:
            copied_uid = None
        generated = context.generated_cards[0]
        try:
            transactions, events = copy_random_opponent_hand_to_self(
                snapshot,
                player=context.player,
                copied_uid=copied_uid,
                placeholder_uid=generated.uid,
                source_uid=context.source_uid,
                cost_delta=int(operation.params.get("cost_delta", 0) or 0),
            )
        except ValueError as exc:
            raise EffectInterpreterError(str(exc)) from exc
        return self._run(snapshot, transactions, events)

    def _discard_hand_card(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        targets = self._target_uids(snapshot, operation, context)
        if not targets:
            # 手牌为空时「选择自己的 1 张手牌舍弃」无事可做（卡面语义为「尽可能选择」），
            # 返回 no-op 而不是中断整段效果（10643110 舍弃后仍要结算后续伤害）。
            player_view = snapshot.me if context.player == "me" else snapshot.opponent
            if not player_view.hand:
                return self._run(snapshot, (), ())
            raise EffectInterpreterError("discard requires target hand cards")
        player_view = snapshot.me if context.player == "me" else snapshot.opponent
        hand_positions = {uid: index for index, uid in enumerate(player_view.hand)}
        transactions: list[SnapshotTransaction] = []
        events: list[ExecutionEvent] = []
        for uid in sorted(targets, key=lambda item: hand_positions.get(item, -1), reverse=True):
            index = hand_positions.get(uid)
            if index is None:
                raise EffectInterpreterError("discard target must be in hand")
            transactions.extend((RemoveZoneUidAt(context.player, "hand", index, uid), ModifyPlayerCounter(context.player, "graveyard_count", 1), RemoveCard(uid, snapshot.cards_by_uid[uid])))
            discarded_card = snapshot.cards_by_uid[uid]
            events.append(ExecutionEvent(EventKind.DISCARD, source_uid=context.source_uid, target_uid=uid, player=context.player, payload={"discarded_card": {"base_card_id": discarded_card.base_card_id, "card_id": discarded_card.card_id}}))
        return self._run(snapshot, tuple(transactions), tuple(events))

    def _destroyed_history_ids(self, view) -> tuple[set[int], set[int]]:
        """返回 (destroyed card_id 集合, 对应 base_card_id 集合)。

        `destroyed_card_ids` 记录的是实战实例的 card_id（可能是变体/进化 id），
        而历史池物化出的生成卡常是 base card_id。两者按 base 归一后比较，
        否则同名历史会被误判为「不在被破坏历史中」。
        """
        card_ids = {int(item.card_id) for item in view.destroyed_card_ids}
        base_ids: set[int] = set()
        for card_id in card_ids:
            base = self.card_resolver.get_base_by_card_id(card_id)
            base_ids.add(int(base.base_card_id) if base is not None else card_id)
        return card_ids, base_ids

    def _reanimate(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        if not context.generated_cards:
            raise EffectInterpreterError("reanimate requires generated_cards")
        max_cost = self._amount(operation, snapshot=snapshot, context=context)
        destroyed, destroyed_bases = self._destroyed_history_ids(
            snapshot.me if context.player == "me" else snapshot.opponent
        )
        current = snapshot
        log = RollbackTransactionLog()
        events: list[ExecutionEvent] = []
        for card in context.generated_cards:
            # replay 证据里生成卡可能是变体 card_id（如 90074151），而 destroyed 历史
            # 记录的是同名 base（90074150）：只在这种情况下按 base 归一放行，非
            # replay 仍要求 card_id 精确出现在被破坏历史中。
            if card.card_id not in destroyed and not (
                context.execution_context.replay_mode and card.base_card_id in destroyed_bases
            ):
                raise EffectInterpreterError("reanimate generated card is not in destroyed history")
            base = self.card_resolver.get_base_by_card_id(card.card_id)
            if base is not None and base.cost > max_cost:
                raise EffectInterpreterError("reanimate generated card exceeds cost limit")
            result = self.kernel.summon_card(current, context.player, card)
            current = log.apply_all(current, tuple(applied.forward for applied in result.transaction_log.applied)) if result.transaction_log.applied else result.snapshot
            events.append(ExecutionEvent(EventKind.REANIMATE, source_uid=context.source_uid, target_uid=card.uid, player=context.player, payload={"max_cost": max_cost}))
            events.extend(result.events)
        return EffectInterpretationResult(snapshot=current, transaction_log=log, events=tuple(events))

    def _history_same_name_pool(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        if not context.generated_cards:
            # 无可用候选（墓地为空或无匹配），效果不生效
            return EffectInterpretationResult(snapshot=snapshot, transaction_log=RollbackTransactionLog(), events=())
        if operation.params.get("source_event") != "destroyed":
            raise EffectInterpreterError("history_same_name_pool only supports destroyed history")
        expected_count = self._amount(operation, default=int(operation.params.get("count", 1) or 1), snapshot=snapshot, context=context)
        # 部分满足：候选不足时用全部可用的
        player_view = snapshot.me if context.player == "me" else snapshot.opponent
        destroyed, destroyed_bases = self._destroyed_history_ids(player_view)
        card_type = operation.params.get("card_type")
        filters = tuple(operation.params.get("filters", ()) or ())
        for card in context.generated_cards:
            # 同 _reanimate：replay 证据允许变体 card_id 与历史 base 同名匹配。
            if card.card_id not in destroyed and not (
                context.execution_context.replay_mode and card.base_card_id in destroyed_bases
            ):
                raise EffectInterpreterError("history_same_name_pool generated card is not in destroyed history")
            if card_type is not None:
                # "amulet" 涵盖倒计数护符(countdown_amulet)。
                if str(card_type) == "amulet":
                    if not card.card_type.is_amulet_like:
                        raise EffectInterpreterError("history_same_name_pool generated card type mismatch")
                elif card.card_type.value != str(card_type):
                    raise EffectInterpreterError("history_same_name_pool generated card type mismatch")
            if filters and card.uid not in self._filter_generated_cards((card,), filters):
                raise EffectInterpreterError("history_same_name_pool generated card does not match filters")
        destination = operation.params.get("destination") or ("hand" if operation.target_ref in {"self_hand", "ally_hand"} else "field")
        if destination == "field":
            return self._summon(snapshot, OperationSpec(op="summon", target_ref="self_field"), context)
        if destination == "hand":
            return self._add_to_hand(snapshot, OperationSpec(op="add_to_hand", target_ref="self_hand"), context)
        if destination == "deck":
            return self._add_generated_cards_to_deck(snapshot, context)
        raise EffectInterpreterError(f"unsupported history_same_name_pool destination: {destination}")

    def _add_generated_cards_to_deck(self, snapshot: BattleSnapshot, context: EffectInterpretationContext) -> EffectInterpretationResult:
        """将 context.generated_cards 以非公开形式作为新实例加入牌组 draw_pile。

        用于"以非公开形式加入牌组"类效果（如轮回转冲）。复用 _return_to_deck copy 模式
        的 DeckInstance 插入逻辑：为每张生成卡创建新 instance，随机插入 draw_pile。
        """
        if not context.generated_cards:
            return EffectInterpretationResult(snapshot=snapshot, transaction_log=RollbackTransactionLog(), events=())
        player_view = snapshot.me if context.player == "me" else snapshot.opponent
        deck_instances = player_view.deck_instances
        transactions: list[SnapshotTransaction] = []
        events: list[ExecutionEvent] = []
        if deck_instances is not None:
            import random as _random
            from decision_engine.state.deck_instance import DeckInstance
            rng = _random.Random(context.execution_context.branch_seed if context.execution_context else 0)
            for card in context.generated_cards:
                next_id = max((inst.instance_id for inst in deck_instances.instances), default=0) + 1
                new_instance = DeckInstance(
                    instance_id=next_id,
                    card=card,
                    owner=context.player,
                    zone="deck",
                    uid=card.uid,
                    origin="history_same_name_pool_deck",
                    order_hint=next_id,
                    visibility="hidden",
                )
                instances = (*deck_instances.instances, new_instance)
                draw_pile = list(deck_instances.draw_pile)
                insert_pos = rng.randint(0, len(draw_pile)) if draw_pile else 0
                draw_pile.insert(insert_pos, next_id)
                deck_instances = replace(deck_instances, instances=instances, draw_pile=tuple(draw_pile))
                events.append(ExecutionEvent("deck_changed", source_uid=context.source_uid, target_uid=card.uid, player=context.player, payload={"kind": "add_to_deck"}))
            transactions.append(SetPlayerField(context.player, "deck_instances", deck_instances))
            transactions.append(ModifyPlayerCounter(context.player, "deck_count", len(context.generated_cards)))
        else:
            transactions.append(ModifyPlayerCounter(context.player, "deck_count", len(context.generated_cards)))
        return self._run(snapshot, tuple(transactions), tuple(events))

    def _deck_operation(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext) -> EffectInterpretationResult:
        kind = operation.params.get("kind")
        if kind == "banish_from_deck":
            from .generated_card_materializer import default_card_resolver

            view = snapshot.me if context.player == "me" else snapshot.opponent
            deck = view.deck_instances
            if deck is None:
                raise EffectInterpreterError("banish_from_deck requires deck_instances")
            resolver = default_card_resolver()
            filters = operation.params.get("filters", ())
            removed: list = []
            kept: list = []
            for instance in deck.instances:
                card = instance.card
                base = resolver.get_base_by_card_id(card.card_id)
                cost = int(base.cost) if base is not None else int(card.cost)
                matched = True
                for item in filters:
                    if not isinstance(item, dict):
                        continue
                    field = item.get("field")
                    operator = item.get("operator")
                    value = item.get("value")
                    if field in {"cost", "original_cost"}:
                        if operator == "<=" and not (cost <= int(value)):
                            matched = False
                        elif operator == "==" and not (cost == int(value)):
                            matched = False
                        elif operator == ">=" and not (cost >= int(value)):
                            matched = False
                if matched:
                    removed.append(instance)
                else:
                    kept.append(instance)
            if not removed:
                return self._run(snapshot, (), ())
            removed_ids = {instance.instance_id for instance in removed}
            new_deck = replace(
                deck,
                instances=tuple(kept),
                draw_pile=tuple(item for item in deck.draw_pile if item not in removed_ids),
                banished=(*deck.banished, *sorted(removed_ids)),
            )
            transactions = (
                SetPlayerField(context.player, "deck_instances", new_deck),
                ModifyPlayerCounter(context.player, "deck_count", -len(removed)),
            )
            events = (ExecutionEvent("deck_changed", source_uid=context.source_uid, player=context.player, payload={"kind": "banish_from_deck", "count": len(removed)}),)
            return self._run(snapshot, transactions, events)
        if kind == "stat_buff":
            # 「使自己的牌组中的所有随从 +X/+Y」：牌组内的卡只存在于
            # deck_instances（没有 uid，不在 cards_by_uid），必须直接改实例的 card。
            # 只作用于仍在 draw_pile 的实例：「牌组中」不含已抽到手上/已打出的卡。
            # 抽牌走 draw_next_from_deck_instances 时直接用 instance.card，
            # 所以这里写入的 atk/life 会随抽牌带进手牌。
            view = snapshot.me if context.player == "me" else snapshot.opponent
            deck = view.deck_instances
            if deck is None or deck.status not in {"available", "ambiguous"}:
                if context.execution_context.replay_mode:
                    # replay 协议只下发 deck_count、不下发牌组构成，无法定位
                    # 「牌组中的所有随从」。牌组内的 buff 本身不改变任何**可见**状态
                    # （牌组不可见），因此按 no-op 结算；effect 仍由调用方计入
                    # state.consumed，报告的 assumptions 里会出现
                    # consumed_event_effects:<effect_id>，可追溯。
                    # 已知局限：若之后抽到本该被 buff 的牌，手牌数值会与录像不一致，
                    # 会以 mismatch 暴露出来，不会静默丢失。
                    return self._run(snapshot, (), ())
                raise EffectInterpreterError(
                    "deck_operation stat_buff requires available deck_instances"
                )
            attack = int(operation.params.get("attack", operation.params.get("atk", 0)) or 0)
            life = int(operation.params.get("life", operation.params.get("defense", 0)) or 0)
            filters = tuple(operation.params.get("filters") or ())
            from .generated_card_materializer import _generated_card_matches_filters

            in_deck = set(deck.draw_pile)
            updated_instances = []
            changed = 0
            for instance in deck.instances:
                card = instance.card
                if instance.instance_id not in in_deck or not _generated_card_matches_filters(card, filters):
                    updated_instances.append(instance)
                    continue
                updated_instances.append(
                    replace(
                        instance,
                        card=replace(
                            card,
                            atk=max(0, card.atk + attack) if attack else card.atk,
                            life=card.life + life if life else card.life,
                            max_life=card.max_life + life if life else card.max_life,
                        ),
                    )
                )
                changed += 1
            if changed == 0:
                return self._run(snapshot, (), ())
            events = (ExecutionEvent("deck_changed", source_uid=context.source_uid, player=context.player, payload={"kind": "stat_buff", "count": changed}),)
            return self._run(
                snapshot,
                (SetPlayerField(context.player, "deck_instances", replace(deck, instances=tuple(updated_instances))),),
                events,
            )
        if kind != "add_to_deck":
            raise EffectInterpreterError(f"deck_operation kind requires runtime execution: {kind}")
        if not context.generated_cards:
            raise EffectInterpreterError("deck_operation add_to_deck requires generated_cards")
        count = int(operation.params.get("count", len(context.generated_cards)) or len(context.generated_cards))
        if len(context.generated_cards) != count:
            raise EffectInterpreterError("deck_operation add_to_deck generated_cards count mismatch")
        transactions: list[SnapshotTransaction] = []
        events: list[ExecutionEvent] = []
        for card in context.generated_cards:
            transactions.extend((AddCard(card), ModifyPlayerCounter(context.player, "deck_count", 1)))
            events.append(ExecutionEvent("deck_changed", source_uid=context.source_uid, target_uid=card.uid, player=context.player, payload={"kind": "add_to_deck"}))
        return self._run(snapshot, tuple(transactions), tuple(events))

    def _attack_limit_transactions(self, snapshot: BattleSnapshot, target_uids: tuple[int, ...], value: int, *, restore_attacks_left: bool = False, sync_attacks_left_to_limit_increase: bool = False, can_attack_leader: bool = False, permanent: bool = False) -> tuple[SnapshotTransaction, ...]:
        transactions: list[SnapshotTransaction] = []
        for uid in target_uids:
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            old_limit = card.attack_limit or 1
            transactions.append(SetCardField(uid, "attack_limit", value))
            if permanent and value > 1:
                # 「使其获得『1回合可以攻击 N 次』」是**随从的关键字**（直到离场都
                # 有效），不是「本回合」的临时加成。两者在协议帧里都只表现为
                # `attack_limit=N` + `buff.attack_limit=N`（其余字段完全相同），无法
                # 事后区分，必须在赋予的这一刻留下来源标记，供
                # `CoreRuleKernel.end_turn` 决定是否重置。
                transactions.append(SetCardField(
                    uid,
                    "supplement_info",
                    {**(card.supplement_info or {}), "attack_limit_permanent": value},
                ))
            if card.card_type.is_follower and value > 0:
                transactions.append(SetCardField(uid, "can_attack", True))
                if sync_attacks_left_to_limit_increase and value > old_limit:
                    # 「使随从获得『1回合可以攻击N次』」：本回合总攻击次数提升到 N，
                    # 保留已用次数，把新增的攻击机会加到 attacks_left（与 gain_keyword double_attack 一致）。
                    transactions.append(SetCardField(uid, "attacks_left", card.attacks_left + (value - old_limit)))
                elif restore_attacks_left and card.attacks_left <= 0 and old_limit < value:
                    # 只有当 attack_limit 实际增加时才恢复攻击次数（+1次机会）
                    transactions.append(SetCardField(uid, "attacks_left", 1))
                if can_attack_leader:
                    transactions.append(SetCardField(uid, "can_attack_leader", True))
        return tuple(transactions)

    def _field_death_transactions(self, snapshot: BattleSnapshot, damaged: list[tuple[int, int]]) -> list[SnapshotTransaction]:
        transactions: list[SnapshotTransaction] = []
        destroyed_cards: dict[str, list[CardView]] = {"me": [], "opponent": []}
        for uid, life_after in damaged:
            if life_after > 0:
                continue
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            owner = self._owner_of_field_uid(snapshot, uid)
            if owner is None:
                continue
            destroyed_cards[owner].append(card)
            transactions.extend((RemoveZoneUid(owner, "field", uid), ModifyPlayerCounter(owner, "graveyard_count", 1), RemoveCard(uid)))
        for owner, cards in destroyed_cards.items():
            if not cards:
                continue
            view = snapshot.me if owner == "me" else snapshot.opponent
            transactions.append(SetPlayerField(owner, "destroyed_card_ids", (*view.destroyed_card_ids, *(CardIdCount(card_id=card.card_id, count=card.style_id) for card in cards))))
        return transactions

    def _owner_of_field_uid(self, snapshot: BattleSnapshot, uid: int) -> str | None:
        if uid in snapshot.me.field:
            return "me"
        if uid in snapshot.opponent.field:
            return "opponent"
        return None

    def _owner_of_hand_uid(self, snapshot: BattleSnapshot, uid: int) -> str | None:
        if uid in snapshot.me.hand:
            return "me"
        if uid in snapshot.opponent.hand:
            return "opponent"
        return None

    def _class_name(self, card: CardView) -> str:
        base = self.card_resolver.get_base(card.base_card_id)
        return base.craft_class if base is not None else ""

    def _operation_condition_met(self, snapshot: BattleSnapshot, operation: OperationSpec, context: EffectInterpretationContext, target_uid: int | None = None) -> bool:
        threshold = operation.params.get("if_counter_at_least")
        if threshold is None:
            condition = operation.params.get("condition")
            if isinstance(condition, dict) and condition.get("subject") in {"source.variable_x", "source.spellboost_count"}:
                subject = str(condition.get("subject", ""))
                threshold = {
                    "subject": "source",
                    "field": "spell_boost_count" if subject == "source.spellboost_count" else "variable_x",
                    "value": condition.get("value", 0),
                }
            else:
                return True
        if not isinstance(threshold, dict):
            raise EffectInterpreterError("if_counter_at_least must be an object")
        field = str(threshold.get("field", ""))
        if field not in {"variable_x", "spell_boost_count", "stack", "countdown", "attack_limit", "evolve_state", "supplement_info"}:
            raise EffectInterpreterError(f"unsupported counter condition field: {field}")
        subject = str(threshold.get("subject", "target"))
        if subject == "source":
            if context.source_uid is None:
                raise EffectInterpreterError("source counter condition requires source_uid")
            card = snapshot.cards_by_uid.get(context.source_uid)
        else:
            if target_uid is None:
                raise EffectInterpreterError("target counter condition requires target_uid")
            card = snapshot.cards_by_uid.get(target_uid)
        if card is None:
            raise EffectInterpreterError("counter condition card missing")
        if field == "supplement_info":
            key = str(threshold.get("key", ""))
            if not key:
                raise EffectInterpreterError("if_counter_at_least supplement_info requires key")
            value = int(card.supplement_info.get(key, 0) or 0)
            reference_key = threshold.get("reference_key")
            if reference_key is None:
                required = int(threshold.get("value", 0))
            else:
                reference = card.supplement_info.get(str(reference_key))
                if reference is None:
                    # 引用一个当前卡上不存在的计数器时默认成 0 会让条件恒真
                    # （薇拉解放奥义曾因此无条件超进化），必须显式失败。
                    raise EffectInterpreterError(
                        f"if_counter_at_least reference_key missing in supplement_info: {reference_key}"
                    )
                required = int(reference)
            return value >= required
        return int(getattr(card, field)) >= int(threshold.get("value", 0))

