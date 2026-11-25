# -*- coding: utf-8 -*-
"""EventQueue 模板效果处理器。"""

from __future__ import annotations

import os

_TRACE_EVENTS = os.environ.get("DSH_TRACE_EVENTS") == "1"
_TRACE_DUP_PATTERN = os.environ.get("DSH_TRACE_DUP", "")
_TRACE_DUP = bool(_TRACE_DUP_PATTERN)

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from decision_engine.actions.model import LegalAction
from decision_engine.cards.templates import ExecutableEffect, UnifiedCardTemplate
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView

from .effect_interpreter import EffectInterpretationContext, EffectInterpreter, EffectInterpreterError
from .execution_context import LocalExecutionContext
from .effect_trigger_resolver import resolve_template_effects_for_event
from .effect_ops.resolution import effective_tribes as _effective_tribes
from .generated_card_materializer import _compare_value, default_card_resolver, effect_candidate_generated_base_ids, effect_has_explicit_generated_card_source, materialize_effect_generated_cards
from .persistent_effect_registry import PersistentEffectRegistry, PersistentSubscription, advance_persistent_registration_order, trigger_count_key
from .target_selector_materializer import materialize_effect_target_groups
from .events import EventHandlerResult, ExecutionEvent


@dataclass
class EventEffectState:
    consumed: list[str]
    unresolved: list[str]


class TemplateEventHandler:
    def __init__(
        self,
        *,
        templates_by_base_card_id: Mapping[int, UnifiedCardTemplate],
        effect_interpreter: EffectInterpreter,
        action: LegalAction,
        generated_cards: tuple[CardView, ...],
        state: EventEffectState,
        execution_context: LocalExecutionContext = LocalExecutionContext(),
        source_snapshot: BattleSnapshot | None = None,
        effect_unresolved_reasons: Callable[[BattleSnapshot, ExecutableEffect, LegalAction, tuple[CardView, ...], CardView | None], tuple[str, ...]],
        registry: PersistentEffectRegistry | None = None,
        skip_effect_ids: frozenset[str] = frozenset(),
        used_generated_uids: set[int] | None = None,
    ):
        self.templates_by_base_card_id = templates_by_base_card_id
        self.effect_interpreter = effect_interpreter
        self.action = action
        self.generated_cards = generated_cards
        self.execution_context = execution_context
        self.source_snapshot = source_snapshot
        self.state = state
        self.effect_unresolved_reasons = effect_unresolved_reasons
        self._per_turn_counters: dict[str, int] = {}
        self._registry = registry
        # 已在动作主路径预结算过的效果（如攻击时自身强化）：本 action 内不再重复执行。
        self._skip_effect_ids = frozenset(skip_effect_ids)
        # 本 step 内已经被分配掉的生成卡 uid。生成卡池是「本 step 所有新入场卡」的并集，
        # 没有归属信息；同一个 step 里两张卡都召唤同一种 token 时，先执行的 effect 必须
        # 拿走第一张，后执行的那次才能拿到第二张（10964120 evolve 链：10062210 的谢幕曲
        # 与 10964120_auto_007 各召唤 1 个神圣猎鹰）。
        # 这个集合必须**跨 handler 实例共享**：一个 step 会构造 entry / pre / main /
        # followup 等多个 handler，各自独立记账的话后执行者仍会拿到已用过的卡。
        # 由调用方（LocalPredictor._apply_template_effects）创建并逐层传入；
        # 不传时退化为实例内局部集合。
        self._used_generated_uids: set[int] = used_generated_uids if used_generated_uids is not None else set()

    def handle(self, snapshot: BattleSnapshot, event: ExecutionEvent) -> EventHandlerResult:
        trace = _TRACE_EVENTS
        if trace:
            print(f"[EV] enter name={event.name} source={event.source_uid} player={event.player}")
        if event.name == "turn_start":
            # CoreRuleKernel uses this as a phase marker. Template effects are
            # dispatched only through the ordered second-stage window events.
            return EventHandlerResult(snapshot=snapshot)
        if int(getattr(snapshot.me, "result", 0) or 0) != 0 or int(getattr(snapshot.opponent, "result", 0) or 0) != 0:
            # 对局已判定（任一方主战者生命归零）：队列里剩余的效果不再结算。
            # 真实规则里生命归零即结束对局，同一步里排队的另一方谢幕曲/纹章触发都不会发生。
            return EventHandlerResult(snapshot=snapshot)
        if event.name == "turn_start_deck":
            field_last_words = event.payload.get("field_last_words", ())
            crest_last_words = event.payload.get("crest_last_words", ())
            if not isinstance(field_last_words, tuple) or not isinstance(crest_last_words, tuple) or not all(isinstance(item, ExecutionEvent) for item in (*field_last_words, *crest_last_words)):
                self.state.unresolved.append("invalid_turn_start_deck_payload")
                return EventHandlerResult(snapshot=snapshot)
            return self._handle_turn_start_deck(snapshot, event, field_last_words, crest_last_words)
        if event.name == "turn_end_deck":
            # 对称 turn_start_deck：处理"在牌组中发动，回合结束时瞬念召唤"类效果（如泽勒尔）。
            return self._handle_deck_invoke(snapshot, event, priority_events=())
        if event.name in {"turn_start_field_last_words", "turn_start_crest_last_words"}:
            pending = event.payload.get("pending_events", ())
            if not isinstance(pending, tuple) or not all(isinstance(item, ExecutionEvent) for item in pending):
                self.state.unresolved.append(f"invalid_turn_start_last_words_payload:{event.name}")
                return EventHandlerResult(snapshot=snapshot)
            return EventHandlerResult(snapshot=snapshot, priority_emitted_events=pending)
        if event.name == "turn_start_invoke_effect":
            deferred_effect = event.payload.get("deferred_effect")
            source_card = snapshot.cards_by_uid.get(event.source_uid or 0)
            if not isinstance(deferred_effect, ExecutableEffect) or source_card is None or event.player not in {"me", "opponent"}:
                self.state.unresolved.append("invalid_turn_start_invoke_effect_payload")
                return EventHandlerResult(snapshot=snapshot)
            context = EffectInterpretationContext(
                source_uid=event.source_uid,
                source_card=source_card,
                player=event.player,
                opponent="opponent" if event.player == "me" else "me",
                execution_context=self.execution_context,
                execution_context_events=(event,),
            )
            try:
                result = self.effect_interpreter.apply_operations(snapshot, deferred_effect.operations, context)
            except EffectInterpreterError as exc:
                self.state.unresolved.append(f"event_effect_execution_failed:{deferred_effect.effect_id}:{exc}")
                return EventHandlerResult(snapshot=snapshot)
            return EventHandlerResult(snapshot=result.snapshot, emitted_events=result.events)
        source_cards = self._source_cards(snapshot, event)
        if event.name == "last_words" and event.payload.get("turn_start_phase"):
            source_cards = tuple(
                (uid, card) for uid, card in source_cards
                if self._template_has_last_words_effect(card.base_card_id)
            )
        if not source_cards:
            # 没有场上/纹章 source 时仍要保留 registry 订阅派发：
            # 「在手牌中发动」类持续效果（source_zone="hand"，如 10913310 的
            # 回合结束时减费）不需要任何场上 source 卡。
            if self._registry is None:
                return EventHandlerResult(snapshot=snapshot)
        current = snapshot
        # turn_start / turn_end：同一时点触发的多个裁定按卡牌登记顺序结算。
        # direct 段（template executable）与 registry 段（persistent）必须交错，
        # 否则两段固定先后会把「场上随从」恒定排在「纹章」之前。
        interleave = event.name in {"turn_start", "turn_end"}
        interleaved_handled: frozenset[int] = frozenset()
        if event.name == "last_words":
            leaving = event.payload.get("leaving_card") or {}
            supplement = leaving.get("supplement_info") if isinstance(leaving, dict) else None
            if isinstance(supplement, dict) and supplement.get("last_words_summon_copy"):
                current = self._summon_last_words_copy(current, event)
        emitted: list[ExecutionEvent] = []
        _consumed_before_direct = len(self.state.consumed)
        for source_uid, source_card in source_cards:
            if event.name == "activate" and source_uid == event.source_uid:
                # The activated card's own effects are resolved by the action path;
                # this event is only for other subscribers such as activation listeners.
                continue
            if event.name == "turn_start_deck" and source_uid is not None and source_uid in snapshot.cards_by_uid:
                # A deck-source effect can move itself to field while resolving this
                # same event. Its second routing pass must not re-run it as a field card.
                continue
            template = self.templates_by_base_card_id.get(source_card.base_card_id)
            if template is None:
                continue
            # discard event 映射到 on_discard_self trigger
            resolve_event = event
            if event.name == "discard" and source_uid == event.target_uid:
                resolve_event = replace(event, kind="on_discard_self")
            if event.name in {"draw", "draw_hidden"}:
                resolve_event = replace(event, kind="on_draw_card")
            # 直接路由段只能执行「从该卡当前所在区域发动」的 persistent 效果。
            # 场地卡在 turn_start / turn_end 时只允许 source_zone="field"
            # （否则「在手牌中发动」的效果会跟着场上的卡一起跑）；
            # 纹章/额外纹章/恩惠卡不在此限制内（部分纹章效果被解析成
            # source_zone="field"，收紧会漏掉它们）。
            allowed_persistent_zones = self._allowed_persistent_zones(snapshot, event, source_uid)
            resolution = resolve_template_effects_for_event(
                template, resolve_event, allowed_persistent_zones=allowed_persistent_zones
            )
            self.state.unresolved.extend(resolution.unresolved_reasons)
            # "按顺序发动以下1个能力" (ordered_abilities): 若 matched_effects 含 ordered
            # 门控标记, 只保留 cursor 选定的那一个能力项, 推进并持久化 cursor。
            matched_effects, current = self._reduce_ordered_abilities(
                current, source_uid, source_card, event, resolution.matched_effects
            )
            # 「本卡牌在战场上消失时」由离场卡自身触发：registry 只注册场上的卡，
            # 卡一旦离场就不再是订阅者，因此必须在直接路由段把这张卡自己的
            # on_self_card_left_field persistent 补进来
            # （11003210「理想国·涅格尔赫亚」离场回复主战者 2 点）。
            if (
                event.name in {"destroy", "destroyed_by_damage", "banish", "bounce", "return_to_deck"}
                and event.target_uid == event.source_uid
                and source_uid == event.source_uid
            ):
                self_exit_effects = tuple(
                    executable_effect
                    for persistent in template.persistent_effects
                    if persistent.trigger is not None and persistent.trigger.event == "on_self_card_left_field"
                    for executable_effect in persistent.executable_effects
                )
                if self_exit_effects:
                    matched_effects = (*matched_effects, *self_exit_effects)
            executable = []
            for effect in matched_effects:
                if effect.effect_id in self._skip_effect_ids:
                    continue
                # activation/activate trigger 只能由自身启动触发，不响应其他卡的 activate 事件
                if event.name == "activate" and source_uid != event.source_uid and effect.trigger.event in {"activation", "activate"}:
                    continue
                # 「本卡牌在战场上消失时」(on_self_card_left_field) 是**自身**语义。
                # 离场类事件的 source_cards 会把「同侧场上剩余的卡」一并带上（供
                # on_self_amulet_destroyed 这类同侧持续效果响应），但那张卡自己并没有
                # 离场，不能执行它的离场效果
                # （replay_...1484307 st85：11054110 还在场上，却因同侧 90051140
                # 被打死而召唤了一个骸骨士兵）。
                if effect.trigger.event == "on_self_card_left_field" and source_uid not in {event.source_uid, event.target_uid}:
                    continue
                # per_turn_limit 检查（按订阅实例计数：同名卡各自独立）
                if effect.effect_id in resolution.per_turn_limits:
                    limit = resolution.per_turn_limits[effect.effect_id]
                    count_key = trigger_count_key(effect.effect_id, source_uid)
                    if self._per_turn_counters.get(count_key, 0) >= limit:
                        continue
                if not self._event_target_conditions_match(current, event, effect):
                    continue
                if not self._event_trigger_scope_matches(current, event, effect, source_uid):
                    continue
                if not self._snapshot_conditions_match(current, effect, source_card=source_card, event_player=event.player, event=event):
                    continue
                # on_self_entered_field: 只有自己入场时才触发。
                # summon 事件可能用 source_uid=召唤者、target_uid=被召唤卡；
                # 两者任一等于订阅者都算「自己入场」。
                if effect.trigger.event == "on_self_entered_field" and source_uid not in {event.source_uid, event.target_uid}:
                    continue
                # 泛化 enter_field/summon 效果（如「己方其他随从入场时…」）不响应自己入场；
                # 「自己入场时」应使用 on_self_entered_field trigger。
                # on_self_follower_entered_field 同一语义：卡面「自己的**其他**随从进入
                # 战场时」明确排除本体（10834120 灾难言灵·洋荷：入场曲召唤 2 个，录像里
                # 手牌只 +2 次魔力增幅，本地按「本体 + 2 个召唤物」算了 3 次）。
                # summon 事件的 source_uid 是召唤者、target_uid 是被召唤卡，两者任一
                # 等于订阅者都算「自己入场」。
                if (
                    event.name in {"enter_field", "summon"}
                    and effect.trigger.event in {"enter_field", "summon", "on_self_follower_entered_field"}
                    and source_uid is not None
                    and source_uid in {event.source_uid, event.target_uid}
                ):
                    continue
                # entered_filters: 检查入场卡是否满足 operations 中的过滤条件
                if not self._entered_filters_match(current, event, effect, source_uid):
                    continue
                effect_for_unresolved = self._without_event_target_conditions(effect)
                effect_player = self._card_owner_side(current, source_uid) or event.player or "me"
                effect_opponent = "opponent" if effect_player == "me" else "me"
                event_targets = self._selected_targets(event)
                event_action = replace(self.action, target_groups=(event_targets,)) if event_targets else self.action
                # 反应式事件触发的 mode_option effect (如"按顺序发动1个能力"的能力项):
                # 其 option_index 本身即"本次要发动的能力", 非玩家选择。注入 selected_indexes
                # 使 mode-selection 相关的 unresolved 检查与 gate 把它视为已选中。
                _effect_option_index = self._option_index_of(effect)
                if _effect_option_index is not None and not event_action.selected_indexes:
                    event_action = replace(event_action, selected_indexes=(_effect_option_index,))
                if not event_targets:
                    if effect.target_selector is None and effect.operations and all(
                        (operation.target_ref or "") in {"source_card", "self"} for operation in effect.operations
                    ):
                        # 效果只作用于效果持有者自身（如护符「本护符的倒计数-1」），
                        # 不需要通用目标物化。
                        if source_uid is not None:
                            event_targets = (source_uid,)
                            event_action = replace(event_action, target_groups=((source_uid,),))
                    else:
                        materialized_targets = materialize_effect_target_groups(
                            current,
                            event_action,
                            effect,
                            source_card=source_card,
                            player=effect_player,
                            opponent=effect_opponent,
                        )
                        if materialized_targets.unresolved_reason is None and materialized_targets.target_groups and materialized_targets.target_groups != ((),):
                            if len(materialized_targets.target_groups) == 1:
                                event_targets = tuple(uid for group in materialized_targets.target_groups for uid in group)
                                event_action = replace(event_action, target_groups=materialized_targets.target_groups)
                            else:
                                # 「发动N次『对随机1个目标…』」需要 N 个命中目标：
                                # 多组候选合并；否则退化为选第一个分支。
                                repeat_count = max(
                                    (int(operation.params.get("repeat", 1) or 1) for operation in effect.operations),
                                    default=1,
                                )
                                if repeat_count <= 1 and effect.target_selector is not None and effect.target_selector.kind == "random_filtered_cards":
                                    # 「对随机N个随从造成…」用 params.count 表达数量（10544110）。
                                    repeat_count = max(
                                        (int(operation.params.get("count", 1) or 1) for operation in effect.operations),
                                        default=1,
                                    )
                                if repeat_count > 1:
                                    merged = tuple(
                                        uid for group in materialized_targets.target_groups for uid in group
                                    )[:repeat_count]
                                    event_targets = merged
                                    event_action = replace(
                                        event_action,
                                        random_target_groups=materialized_targets.target_groups,
                                    )
                                else:
                                    first_group = materialized_targets.target_groups[0]
                                    event_targets = first_group
                                    event_action = replace(event_action, random_target_groups=(first_group,))
                        elif materialized_targets.unresolved_reason is None and materialized_targets.target_groups == ((),):
                            # 目标池为空 → 效果不触发（safe noop）
                            self.state.consumed.append(effect.effect_id)
                            continue
                event_generated_cards = self._generated_cards_for_effect(effect, snapshot=current, player=event.player)
                if event.name in {"last_words", "destroy", "destroyed_by_damage", "banish"} and event.player:
                    # 离场类事件的生成卡属于「事件源卡所在侧」。录制池是全局的，
                    # 同一 step 里双方的谢幕曲会共用它；不过滤会把对手的召唤卡也算到本方
                    # （replay_...50156640 frame31）。只接受同侧或未标注归属的卡。
                    event_generated_cards = tuple(
                        card for card in event_generated_cards
                        if str(card.supplement_info.get("replay_owner_side") or "") in {"", event.player}
                    )
                if not event_generated_cards and effect_has_explicit_generated_card_source(effect):
                    materialized = materialize_effect_generated_cards(
                        current,
                        event_action,
                        effect,
                        source_card=source_card,
                        resolver=default_card_resolver(),
                        player=event.player or "me",
                    )
                    if materialized.unresolved_reason is not None:
                        self.state.unresolved.append(f"{materialized.unresolved_reason}:{effect.effect_id}")
                        continue
                    event_generated_cards = materialized.cards
                effect_unresolved = self.effect_unresolved_reasons(
                    current,
                    effect_for_unresolved,
                    event_action,
                    event_generated_cards,
                    source_card,
                    execution_context=self.execution_context,
                    player=effect_player,
                    opponent=effect_opponent,
                )
                if effect_unresolved:
                    self.state.unresolved.extend(effect_unresolved)
                    continue
                executable_effect = self._without_core_spellboost_operations(effect) if event.name == "spellboost" else effect
                if event.name == "turn_start_deck" and any(operation.op == "invoke_self" for operation in executable_effect.operations):
                    executable_effect = replace(
                        executable_effect,
                        operations=tuple(operation for operation in executable_effect.operations if operation.op == "invoke_self"),
                    )
                if not executable_effect.operations:
                    self.state.consumed.append(effect.effect_id)
                    continue
                executable.append((executable_effect, event_generated_cards, event_targets, event_action, resolution.per_turn_limits, effect))
            for effect, event_generated_cards, effect_event_targets, effect_event_action, effect_per_turn_limits, original_effect in executable:
                player = event.player or "me"
                opponent = "opponent" if player == "me" else "me"
                # Cross-side turn_end: source_card 在对方场上时，player 用 source_card 的 owner side
                if event.name == "turn_end" and source_uid is not None:
                    source_owner = self._card_owner_side(current, source_uid)
                    if source_owner is not None and source_owner != player:
                        player = source_owner
                        opponent = "opponent" if player == "me" else "me"
                # 一条 event effect 若含多个独立 random pool，单个 action 的记录目标
                # 不能同时绑定到所有 operation；保留单池 replay 目标，多池则让每个
                # operation 按 branch seed/自身候选池独立物化。
                # 单池 effect 直接用录制目标；多池 effect（如 10954110 的回合结束
                # 「随机破坏本方 + 随机破坏对方」）也需要把各组传下去，由每个 operation
                # 按自己的目标侧认领一组。此前多池一律置空，导致两个 destroy 都退回
                # 「按候选顺序取首个」，破坏的实例与录制的随机结果不同。
                random_targets = tuple(
                    uid for group in effect_event_action.random_target_groups for uid in group
                )
                # 反应式效果携带的 mode_option 即"本次发动的那一个能力"
                # (如 ordered_abilities cursor 选定的能力)。其 option_index 本身就是
                # 被选中项, 注入 selected_mode_options 使 interpreter 的 mode gate 通过。
                effect_option_index = self._option_index_of(effect)
                selected_mode_options = (
                    (effect_option_index,) if effect_option_index is not None
                    else tuple(self.action.selected_indexes)
                )
                if _TRACE_DUP and _TRACE_DUP_PATTERN in effect.effect_id:
                    print(f"[OPT] effect={effect.effect_id} idx={effect_option_index} "
                          f"sel={selected_mode_options} ops={[tuple((o.params or {}).keys()) for o in effect.operations]}")
                context = EffectInterpretationContext(
                    source_uid=source_uid,
                    source_card=source_card,
                    player=player,
                    opponent=opponent,
                    selected_targets=effect_event_targets,
                    random_targets=random_targets,
                    generated_cards=event_generated_cards,
                    execution_context=self.execution_context,
                    allow_unrecorded_random_targets=_random_target_ref_count(effect.operations) > 1,
                    selected_mode_options=selected_mode_options,
                    execution_context_events=(event,),
                    trigger_event=effect.trigger.event,
                    last_entered_self_follower_uid=event.source_uid if event.name in {"enter_field", "summon", "on_self_follower_entered_field"} else None,
                    step_generated_uids=frozenset(self._used_generated_uids),
                )
                try:
                    if _TRACE_DUP and _TRACE_DUP_PATTERN in effect.effect_id:
                        print(f"[DUP] segment=registry event={event.name} source={event.source_uid} effect={effect.effect_id}")
                    result = self.effect_interpreter.apply_operations(current, effect.operations, context)
                except EffectInterpreterError as exc:
                    exc_msg = str(exc)
                    if "requires random_targets" in exc_msg or "random_target" in exc_msg:
                        # 随机目标多候选：effect 可执行但 replay 缺录制目标。
                        # 标记为 target_selector_not_supported 让 validator 的
                        # random branch matching 逻辑尝试所有候选。
                        self.state.unresolved.append(f"target_selector_not_supported:{effect.effect_id}")
                        self.state.consumed.append(effect.effect_id)
                    elif "target card is missing" in exc_msg or "target must be on" in exc_msg:
                        # 目标已离场（先前效果杀/弹/归零），效果 safe noop
                        self.state.consumed.append(effect.effect_id)
                    else:
                        self.state.unresolved.append(f"event_effect_execution_failed:{effect.effect_id}:{exc}")
                    continue
                current = result.snapshot
                emitted.extend(result.events)
                self.state.consumed.append(effect.effect_id)
                self._record_used_generated_uids(result.events)
                if effect.effect_id in effect_per_turn_limits:
                    count_key = trigger_count_key(effect.effect_id, source_uid)
                    self._per_turn_counters[count_key] = self._per_turn_counters.get(count_key, 0) + 1
            if interleave and self._registry is not None:
                # 该卡自己的 persistent 订阅紧跟它自己的 executable 效果执行。
                current, _, handled = self._dispatch_registry_subscriptions(
                    event, current, emitted,
                    direct_consumed_effects=frozenset(self.state.consumed[_consumed_before_direct:]),
                    only_uids=frozenset({source_uid}),
                    exclude_subscription_ids=interleaved_handled,
                )
                interleaved_handled = interleaved_handled | frozenset(handled)
        # === Registry 订阅段（类型 B）===
        # 只阻止当前 event 中直接路由段已消费的 effect（避免同一 event 内重复执行），
        # 不阻止不同 event 触发同一 persistent effect（如多个 enter_field 各触发一次）。
        direct_consumed_this_event = frozenset(self.state.consumed[_consumed_before_direct:])
        if interleave:
            # turn_start / turn_end：direct 段已按登记顺序逐张卡交错执行过各自的
            # 订阅，剩余订阅（手牌中发动、跨侧等不在登记顺序里的）在末尾补执行。
            current, _, handled = self._dispatch_registry_subscriptions(
                event, current, emitted,
                direct_consumed_effects=direct_consumed_this_event,
                exclude_subscription_ids=interleaved_handled,
            )
            interleaved_handled = interleaved_handled | frozenset(handled)
        else:
            current, _, _ = self._dispatch_registry_subscriptions(
                event, current, emitted,
                direct_consumed_effects=direct_consumed_this_event,
            )
        return EventHandlerResult(snapshot=current, emitted_events=tuple(emitted))

    def _dispatch_registry_subscriptions(
        self,
        event: ExecutionEvent,
        current: BattleSnapshot,
        emitted: list[ExecutionEvent],
        *,
        direct_consumed_effects: frozenset[str] = frozenset(),
        only_uids: frozenset[int | None] | None = None,
        exclude_subscription_ids: frozenset[int] = frozenset(),
    ) -> tuple[BattleSnapshot, list[ExecutionEvent], tuple[int, ...]]:
        """执行 registry 订阅的 persistent 效果（类型 B），返回已检查的订阅 id。

        从 ``handle`` 的「Registry 订阅段」抽出，供两种调用方式复用：

        * 一次性派发（``only_uids=None``）：原有行为，direct 段全部执行完之后
          再执行所有订阅；
        * 按 uid 切片（``only_uids={uid}``）：与 direct 段**交错**执行，用于
          ``turn_start`` / ``turn_end`` —— 同一时点触发的多个裁定必须按卡牌
          登记顺序结算（场地与纹章合并），两段式会让场上随从的效果恒定排在
          纹章之前（replay_...0006 st65：10564110 的 debuff 先清场，纹章
          10364122 的 3 点分配只能打主战者，录像里主战者不动）。

        ``handled`` 记录的是「本次已检查过」的订阅：条件不满足的也计入，
        保证交错阶段与剩余阶段合起来每个订阅恰好检查一次（与一次性派发等价）。
        """
        if self._registry is None:
            return current, emitted, ()
        handled: list[int] = []
        direct_consumed_this_event = direct_consumed_effects
        subscriptions = [
            sub
            for sub in self._registry.get_subscribers(event, current)
            if (only_uids is None or sub.subscriber_uid in only_uids)
            and id(sub) not in exclude_subscription_ids
        ]
        if _TRACE_DUP:
            _hits = [
                (sub.subscriber_uid, sub.effect.effect_id, sub.scope, sub.player)
                for sub in subscriptions
                if _TRACE_DUP_PATTERN in sub.effect.effect_id
            ]
            if _hits:
                print(f"[DUPSUB] event={event.name} source={event.source_uid} target={event.target_uid} hits={_hits}")
        if any(self._is_ordered_abilities_gate(sub.effect) for sub in subscriptions):
            _by_uid: dict = {}
            for sub in subscriptions:
                if sub.effect.effect_id in direct_consumed_this_event:
                    # 直接路由段已经执行过这张卡的 ordered 能力（门控与能力项都记进
                    # consumed），这里不能再次调用 _reduce_ordered_abilities —— 否则
                    # cursor 会被推进第二次，同一个回合结束会发动两个能力项。
                    continue
                _by_uid.setdefault(sub.subscriber_uid, []).append(sub)
            _reduced: list = []
            for _uid, _group in _by_uid.items():
                if not any(self._is_ordered_abilities_gate(sub.effect) for sub in _group):
                    _reduced.extend(_group)
                    continue
                _kept, current = self._reduce_ordered_abilities(
                    current, _uid, _group[0].subscriber_card, event,
                    tuple(sub.effect for sub in _group),
                )
                _kept_ids = {e.effect_id for e in _kept}
                _reduced.extend(sub for sub in _group if sub.effect.effect_id in _kept_ids)
            subscriptions = tuple(_reduced)
        for sub in subscriptions:
            handled.append(id(sub))
            effect = sub.effect
            # 跳过本次 event 中已被直接路由段消费的效果
            if effect.effect_id in direct_consumed_this_event:
                continue
            # 跳过已由 LocalPredictor 在动作前预结算的效果（「攻击时」自身强化
            # 必须在伤害结算前生效，不能在这里二次结算）。
            if effect.effect_id in self._skip_effect_ids:
                continue
            # draft 检查
            # event_target conditions 检查
            if not self._event_target_conditions_match(current, event, effect):
                continue
            # trigger scope 检查（如「自己的随从进入战场时」必须事件源是 follower）
            if not self._event_trigger_scope_matches(current, event, effect, sub.subscriber_uid):
                continue
            # snapshot 级条件检查（如 self_unevolved、self_evolved、turn_check 等）
            if not self._snapshot_conditions_match(current, effect, source_card=sub.subscriber_card, event_player=sub.player, event=event):
                continue
            if not self._entered_filters_match(current, event, effect, sub.subscriber_uid):
                continue
            # generated_cards 处理
            event_generated_cards = self._generated_cards_for_effect(effect, snapshot=current, player=sub.player)
            if not event_generated_cards and effect_has_explicit_generated_card_source(effect):
                materialized = materialize_effect_generated_cards(
                    current,
                    self.action,
                    effect,
                    source_card=sub.subscriber_card,
                    resolver=default_card_resolver(),
                    player=sub.player,
                )
                if materialized.unresolved_reason is not None:
                    self.state.unresolved.append(f"{materialized.unresolved_reason}:{effect.effect_id}")
                    continue
                event_generated_cards = materialized.cards
            event_action = self.action
            # 入场事件触发的 persistent（如纹章"自己的随从进入战场时使其获得X"）
            # 目标是本次入场的随从（last_entered_self_follower / last_summoned），
            # 这是 event 上下文依赖目标，通用 materializer 无法解析。
            # 与直接路由段一致：用 event.source_uid 直接定位，并在 context 中提供
            # last_entered_self_follower_uid 供 interpreter 解析 operation.target_ref。
            event_entered_uid: int | None = None
            selector_kind = effect.target_selector.kind if effect.target_selector is not None else ""
            references_entered = selector_kind in {"last_entered_self_follower", "last_summoned"} or any(
                (operation.target_ref or "") in {"last_entered_self_follower", "last_summoned"}
                for operation in effect.operations
            )
            if references_entered and event.name in {"enter_field", "summon", "on_self_follower_entered_field", "play_card"} and event.source_uid is not None:
                event_entered_uid = event.source_uid
            if event_entered_uid is not None:
                target_groups = ((event_entered_uid,),)
            elif effect.target_selector is None and effect.operations and all(
                (operation.target_ref or "") in {"source_card", "self"} for operation in effect.operations
            ):
                # 效果只作用于订阅者自身（如护符「本护符的倒计数-1」）：
                # 不需要通用目标物化，直接以订阅者为唯一目标。
                target_groups = ((sub.subscriber_uid,),) if sub.subscriber_uid is not None else ((),)
            else:
                materialized_targets = materialize_effect_target_groups(
                    current,
                    event_action,
                    effect,
                    source_card=sub.subscriber_card,
                    # 订阅者所在侧决定 self/enemy 的解析方向。纹章/持续效果可能
                    # 属于任一方；不传时 materializer 默认按 "me" 解释 side="self"，
                    # 对方侧订阅者会取到错误的一边（10954112 对方纹章回合结束）。
                    player=sub.player,
                    opponent="opponent" if sub.player == "me" else "me",
                )
                if materialized_targets.unresolved_reason is not None:
                    self.state.unresolved.append(f"{materialized_targets.unresolved_reason}:{effect.effect_id}")
                    continue
                target_groups = materialized_targets.target_groups
            if target_groups == ((),):
                self.state.consumed.append(effect.effect_id)
                self._registry.increment_counter(sub)
                continue
            random_targets: tuple[int, ...] = ()
            selected_targets: tuple[int, ...] = ()
            multi_random_pools = _random_target_ref_count(effect.operations) > 1
            # 「发动N次『对随机1个目标…』」（repeat>1）需要 N 个命中目标，
            # 多组候选应合并而不是只取第一组。
            repeat_count = max(
                (
                    int(operation.params.get("repeat", 1) or 1)
                    for operation in effect.operations
                ),
                default=1,
            )
            if repeat_count <= 1 and effect.target_selector is not None and effect.target_selector.kind == "random_filtered_cards":
                # 「对随机N个随从造成…」用 params.count 表达数量（10544110）。
                repeat_count = max(
                    (
                        int(operation.params.get("count", 1) or 1)
                        for operation in effect.operations
                    ),
                    default=1,
                )
            if target_groups:
                if multi_random_pools:
                    # 多随机池 effect（如 10954112 纹章回合结束「随机破坏己方谢幕曲
                    # 随从 + 随机破坏敌方随从」）：物化结果只用来判定「至少有一个池
                    # 非空」，不能把合并后的 uid 集合当成任一个 op 的目标——两池的
                    # uid 混在一起会让 op 的字段校验失败（uid 不在自己那一侧的场上）。
                    # 每个 op 在 interpreter 里按自己的候选池独立求解。
                    pass
                elif effect.target_selector is not None and (
                    effect.target_selector.scope == "random" or effect.target_selector.kind.startswith("random_")
                ):
                    if repeat_count > 1 and len(target_groups) > 1:
                        random_targets = tuple(uid for group in target_groups for uid in group)[:repeat_count]
                    else:
                        random_targets = tuple(target_groups[0])
                elif any((operation.target_ref or "").startswith("random_") for operation in effect.operations):
                    if repeat_count > 1 and len(target_groups) > 1:
                        random_targets = tuple(uid for group in target_groups for uid in group)[:repeat_count]
                    else:
                        random_targets = tuple(target_groups[0])
                else:
                    selected_targets = tuple(target_groups[0])
            player = sub.player
            opponent = "opponent" if player == "me" else "me"
            context = EffectInterpretationContext(
                source_uid=sub.subscriber_uid,
                source_card=sub.subscriber_card,
                player=player,
                opponent=opponent,
                selected_targets=selected_targets,
                random_targets=random_targets,
                generated_cards=event_generated_cards,
                execution_context=self.execution_context,
                allow_unrecorded_random_targets=multi_random_pools,
                # effect 自带 mode_option.option_index 时（ordered_abilities 的
                # 能力项），它本身就是「被选中的能力」，必须注入
                # selected_mode_options，否则 interpreter 的 mode gate 会报
                # 「mode_option requires selected_mode_options」
                # （replay_...1283403 st64 的 11024122_option_1）。
                selected_mode_options=(
                    (self._option_index_of(effect),)
                    if self._option_index_of(effect) is not None
                    else ()
                ),
                execution_context_events=(event,),
                trigger_event=effect.trigger.event,
                last_entered_self_follower_uid=event_entered_uid,
                step_generated_uids=frozenset(self._used_generated_uids),
            )
            try:
                result = self.effect_interpreter.apply_operations(current, effect.operations, context)
            except EffectInterpreterError as exc:
                exc_msg = str(exc)
                if "requires random_targets" in exc_msg or "random_target" in exc_msg:
                    self.state.unresolved.append(f"target_selector_not_supported:{effect.effect_id}")
                    self.state.consumed.append(effect.effect_id)
                elif "target card is missing" in exc_msg or "target must be on" in exc_msg:
                    # 目标已离场，效果 safe noop
                    self.state.consumed.append(effect.effect_id)
                else:
                    self.state.unresolved.append(f"event_effect_execution_failed:{effect.effect_id}:{exc}")
                continue
            current = result.snapshot
            emitted.extend(result.events)
            self.state.consumed.append(effect.effect_id)
            self._record_used_generated_uids(result.events)
            self._registry.increment_counter(sub)
            # 将 per_turn 计数器持久化到 snapshot，供后续 action 的 registry 恢复。
            # key 必须带订阅者 uid：同名卡同时在场时各自有独立的「每回合 N 次」。
            if sub.per_turn_limit is not None:
                view = current.me if sub.player == "me" else current.opponent
                count_key = trigger_count_key(effect.effect_id, sub.subscriber_uid)
                updated_counts = {**view.effect_trigger_counts, count_key: sub.counter}
                if sub.player == "me":
                    current = replace(current, me=replace(current.me, effect_trigger_counts=updated_counts))
                else:
                    current = replace(current, opponent=replace(current.opponent, effect_trigger_counts=updated_counts))
        return current, emitted, tuple(handled)

    def _handle_turn_start_deck(
        self,
        snapshot: BattleSnapshot,
        event: ExecutionEvent,
        field_last_words: tuple[ExecutionEvent, ...],
        crest_last_words: tuple[ExecutionEvent, ...],
    ) -> EventHandlerResult:
        # turn_start 特有的第二阶段窗口事件（场上/纹章谢幕曲 + turn_start 效果）
        priority_events = (
            ExecutionEvent("turn_start_field_last_words", player=event.player, payload={"pending_events": field_last_words}),
            ExecutionEvent("turn_start_field_effects", player=event.player),
            ExecutionEvent("turn_start_crest_last_words", player=event.player, payload={"pending_events": crest_last_words}),
            ExecutionEvent("turn_start_crest_effects", player=event.player),
        )
        return self._handle_deck_invoke(snapshot, event, priority_events=priority_events)

    def _handle_deck_invoke(
        self,
        snapshot: BattleSnapshot,
        event: ExecutionEvent,
        *,
        priority_events: tuple[ExecutionEvent, ...],
    ) -> EventHandlerResult:
        """扫描牌组内卡的 invoke_self 效果（瞬念召唤）。turn_start_deck / turn_end_deck 共用。

        priority_events 为该时点特有的第二阶段窗口事件（turn_start 有场上/纹章谢幕曲窗口，
        turn_end 目前无额外窗口）。invoke 产生的剩余 operations 作为 deferred 事件追加在末尾。
        """
        invoke_effect_event_name = "turn_start_invoke_effect"
        sources = self._source_cards(snapshot, event)
        current = snapshot
        deferred: list[ExecutionEvent] = []
        invoked_base_card_ids: set[int] = set()  # 同名瞬念召唤每回合只触发一次
        for source_uid, source_card in sources:
            # 如果该 uid 已经在场上（被之前的 invoke_self 移动过），跳过
            if source_uid is not None and (source_uid in current.me.field or source_uid in current.opponent.field):
                continue
            template = self.templates_by_base_card_id.get(source_card.base_card_id)
            if template is None:
                continue
            resolution = resolve_template_effects_for_event(template, event)
            self.state.unresolved.extend(resolution.unresolved_reasons)
            for effect in resolution.matched_effects:
                if not self._snapshot_conditions_match(current, effect, source_card=source_card, event_player=event.player, event=event):
                    continue
                invoke_operations = tuple(operation for operation in effect.operations if operation.op == "invoke_self")
                deck_activation_operations = tuple(operation for operation in effect.operations if operation.op == "deck_ability_activation")
                if not invoke_operations:
                    if deck_activation_operations:
                        self.state.consumed.append(effect.effect_id)
                    continue
                # 同名瞬念召唤每回合只触发一次
                if source_card.base_card_id in invoked_base_card_ids:
                    continue
                player = event.player or "me"
                context = EffectInterpretationContext(
                    source_uid=source_uid,
                    source_card=source_card,
                    player=player,
                    opponent="opponent" if player == "me" else "me",
                    execution_context=self.execution_context,
                    execution_context_events=(event,),
                )
                try:
                    result = self.effect_interpreter.apply_operations(current, invoke_operations, context)
                except EffectInterpreterError as exc:
                    self.state.unresolved.append(f"event_effect_execution_failed:{effect.effect_id}:{exc}")
                    continue
                current = result.snapshot
                remaining_operations = tuple(operation for operation in effect.operations if operation.op not in {"invoke_self", "deck_ability_activation"})
                if remaining_operations:
                    deferred.append(ExecutionEvent(
                        invoke_effect_event_name,
                        source_uid=source_uid,
                        player=player,
                        payload={"deferred_effect": replace(effect, operations=remaining_operations)},
                    ))
                self.state.consumed.append(effect.effect_id)
                invoked_base_card_ids.add(source_card.base_card_id)
        return EventHandlerResult(
            snapshot=current,
            priority_emitted_events=(*priority_events, *deferred),
        )

    def _template_has_last_words_effect(self, base_card_id: int) -> bool:
        template = self.templates_by_base_card_id.get(base_card_id)
        return template is not None and any(effect.trigger.event == "last_words" for effect in template.executable_effects)

    def _entered_filters_match(self, snapshot: BattleSnapshot, event: ExecutionEvent, effect: ExecutableEffect, source_uid: int | None = None) -> bool:
        """检查入场卡是否满足 effect operations 中的 entered_filters 条件。"""
        if event.name not in {"enter_field", "summon", "on_self_follower_entered_field"}:
            return True
        # 收集所有 operations 的 entered_filters
        all_filters = []
        for op in effect.operations:
            filters = op.params.get("entered_filters") if op.params else None
            if filters:
                all_filters.extend(filters)
        if not all_filters:
            return True
        # 获取入场卡
        entered_uid = event.source_uid if event.source_uid is not None else event.target_uid
        entered_card = snapshot.cards_by_uid.get(entered_uid or 0)
        if entered_card is None:
            return False
        for f in all_filters:
            if not isinstance(f, dict):
                continue
            field = f.get("field", "")
            operator = f.get("operator", "")
            value = f.get("value")
            if field == "tribes" and operator == "contains":
                if value not in self._entered_card_tribes(entered_card):
                    return False
            elif field == "class_name":
                if not _compare_value(self._entered_card_class_name(entered_card), operator or "==", value):
                    return False
            elif field == "card_type":
                if not _compare_value(entered_card.card_type.value, operator or "==", value):
                    return False
            elif field in {"cost", "original_cost"}:
                if not _compare_value(entered_card.cost, operator or "==", value):
                    return False
            elif field == "name":
                # 「自己的『天晶魔手』进入战场时…」：按入场卡的卡名过滤。
                # 候选生成器产出 {"field": "name", "operator": "in"|"==", ...}；
                # 执行端此前没有 name 分支，该过滤条件被静默忽略。
                if not _match_name_filter(self._entered_card_name(entered_card), operator or "==", value):
                    return False
            elif field == "base_card_id":
                if not _match_base_card_id_filter(entered_card.base_card_id, operator or "==", value):
                    return False
            elif f.get("exclude") == "source_card":
                # exclude source_card: 入场卡不能是持有该效果的卡本身
                # （如「自己的*其他*梦魇随从进入战场时…」，本随从自身入场不触发）。
                if source_uid is not None and entered_uid == source_uid:
                    return False
        return True

    def _entered_card_tribes(self, card: CardView) -> tuple[int, ...]:
        """入场卡的 tribe 口径：协议快照的 CardView.tribes 可能为空，需回退 CardResolver。

        直接复用 effect_ops.resolution.effective_tribes（同一口径，也处理
        buff.add_tribe 与 0 占位）；否则 `entered_filters` 的 `tribes contains N`
        恒不匹配，效果被静默丢弃。
        """
        return _effective_tribes(card)

    def _entered_card_name(self, card: CardView) -> str:
        if card.name:
            return str(card.name)
        base = default_card_resolver().get_base(card.base_card_id)
        return base.name if base is not None and base.name else ""

    def _entered_card_class_name(self, card: CardView) -> str:
        value = card.supplement_info.get("class_name", "") if card.supplement_info else ""
        if value:
            return str(value)
        base = default_card_resolver().get_base(card.base_card_id)
        return base.craft_class if base is not None else ""

    def _event_trigger_scope_matches(self, snapshot: BattleSnapshot, event: ExecutionEvent, effect: ExecutableEffect, source_uid: int | None) -> bool:
        if effect.trigger.event in {"on_self_follower_entered_field", "on_enemy_follower_entered_field"}:
            # 「自己的随从进入战场时」只对 follower 生效：护符/法术/纹章入场不应触发。
            entered = snapshot.cards_by_uid.get(event.source_uid or 0)
            if event.player not in {"me", "opponent"} or entered is None or not entered.card_type.is_follower:
                return False
            # 「自己的/对手的」决定触发侧：入场随从的所属方必须与 trigger 指向一致。
            # 此前只校验「入场的是 follower」，于是对手的 10834120「自己的随从入场时
            # 使所有手牌魔力增幅」会在我方随从入场时触发
            # （replay_...1739027299 st73：本 step 手牌 spell_boost 多 1）。
            subscriber_side = self._card_owner_side(snapshot, source_uid)
            if subscriber_side is None:
                return False
            if effect.trigger.event == "on_self_follower_entered_field":
                return event.player == subscriber_side
            return event.player != subscriber_side
        if effect.trigger.event == "on_self_amulet_activated":
            activated = snapshot.cards_by_uid.get(event.source_uid or 0)
            source_type = event.payload.get("source_card_type")
            is_amulet = activated is not None and activated.card_type.is_amulet_like
            if source_type is not None:
                is_amulet = source_type in {"amulet", "countdown_amulet"}
            # 「自己【启动】证据·护符时」：证据护符由 tribe 21 标识。条件只约束
            # 触发源的类型，不能变成对订阅者自身的要求。
            for condition in effect.conditions:
                if condition.kind == "event_source_tribe":
                    if activated is None or int(condition.value) not in self._entered_card_tribes(activated):
                        return False
            return (
                event.player in {"me", "opponent"}
                and is_amulet
                and source_uid is not None
                and source_uid != event.source_uid
            )
        if effect.trigger.event == "on_self_follower_super_evolved":
            evolved = snapshot.cards_by_uid.get(event.source_uid or 0)
            return event.player in {"me", "opponent"} and evolved is not None and evolved.card_type.is_follower
        if effect.trigger.event == "on_self_card_left_field":
            left_uid = event.target_uid if event.target_uid is not None else event.source_uid
            if left_uid is None or source_uid is None:
                return False
            allow_self = any(condition.kind == "event_source_is_self" for condition in effect.conditions)
            if source_uid == left_uid and not allow_self:
                return False
            if allow_self and source_uid != left_uid:
                # `event_source_is_self` 的语义是「本卡牌（订阅者自身）离场时」：
                # 必须由订阅者自身离场触发，其他同侧卡离场不能命中。
                return False
            subscriber_side = self._card_owner_side(snapshot, source_uid)
            if subscriber_side is None:
                # 订阅者自身已离场（如「本卡牌在战场上消失时」）；registry 已按
                # side 过滤同侧订阅，这里不能再因卡不在 snapshot 而丢弃自身触发。
                subscriber_side = event.player
            if subscriber_side is None or event.player not in {None, subscriber_side}:
                return False
            # 可选：限定离场卡的 card_type / tribe（如「自己的创造物·随从消失时」）。
            leaving = snapshot.cards_by_uid.get(left_uid)
            for condition in effect.conditions:
                if condition.kind == "event_source_card_type":
                    if leaving is None or leaving.card_type.value != str(condition.value):
                        return False
                if condition.kind == "event_source_tribe":
                    if leaving is None or int(condition.value) not in self._entered_card_tribes(leaving):
                        return False
            return True
        if effect.trigger.event == "on_self_amulet_destroyed":
            # "自己的护符被破坏时": 被破坏的卡必须是与订阅者同侧的护符类卡。
            # 被破坏卡通常已从 cards_by_uid 移除, 从 leaving_card payload 判定 card_type;
            # subscriber (source_uid) 必须是被破坏卡的同侧场上卡 (且不是被破坏卡自身)。
            destroyed_uid = event.target_uid if event.target_uid is not None else event.source_uid
            if destroyed_uid is None or source_uid is None or source_uid == destroyed_uid:
                return False
            subscriber_side = self._card_owner_side(snapshot, source_uid)
            if subscriber_side is None or subscriber_side != event.player:
                return False
            destroyed_is_amulet = self._destroyed_card_is_amulet(snapshot, event, destroyed_uid)
            return destroyed_is_amulet
        return True

    def _destroyed_card_is_amulet(self, snapshot: BattleSnapshot, event: ExecutionEvent, destroyed_uid: int) -> bool:
        """判定被破坏卡是否为护符类卡。优先用 leaving_card payload 的 card_type,
        回退到 snapshot 中仍存在的卡 (destroy 前的 pre-snapshot)。"""
        lc = event.payload.get("leaving_card")
        if isinstance(lc, dict) and lc.get("card_type") is not None:
            try:
                return CardType(lc["card_type"]).is_amulet_like
            except ValueError:
                return False
        existing = snapshot.cards_by_uid.get(destroyed_uid)
        if existing is not None:
            return existing.card_type.is_amulet_like
        return False

    @staticmethod
    def _card_view_from_leaving_payload(
        uid: int | None,
        payload: dict[str, object],
        *,
        with_card_type: bool,
        has_guard: bool = False,
    ) -> CardView:
        return card_view_from_leaving_payload(uid, payload, with_card_type=with_card_type, has_guard=has_guard)

    def _event_target_conditions_match(self, snapshot: BattleSnapshot, event: ExecutionEvent, effect: ExecutableEffect) -> bool:
        return event_target_conditions_match(snapshot, event, effect, self.state.unresolved)

    def _snapshot_conditions_match(self, snapshot: BattleSnapshot, effect: ExecutableEffect, source_card: CardView | None = None, *, event_player: str | None = None, event: ExecutionEvent | None = None) -> bool:
        """评估非 event_source/event_target 前缀的 snapshot 级条件。"""
        from .condition_evaluator import ConditionEvaluator, ConditionEvaluationContext
        snapshot_conditions = tuple(
            c for c in effect.conditions
            if not c.subject.startswith(("event_source.", "event_target."))
        )
        if not snapshot_conditions:
            return True
        evaluator = ConditionEvaluator()
        event_source_card = snapshot.cards_by_uid.get(event.source_uid or 0) if event is not None else None
        # 法术打出后立即进墓地，snapshot 里已无该卡：用事件 payload 携带的本体 id 回退，
        # 供「自己使用其他卡牌时」类条件取原始费用（10503210 大游戏世界）。
        raw_base_id = event.payload.get("source_base_card_id") if event is not None else None
        try:
            source_base_card_id = int(raw_base_id) if raw_base_id is not None else None
        except (TypeError, ValueError):
            source_base_card_id = None
        raw_field_ids = event.payload.get("field_card_ids") if event is not None else None
        try:
            field_card_ids = tuple(int(item) for item in raw_field_ids) if raw_field_ids else ()
        except (TypeError, ValueError):
            field_card_ids = ()
        # `entered_card.*` 条件只对入场事件有意义：enter_field / summon 的 source_uid
        # 就是刚进场的那张卡。其它事件的 source_uid 语义不同，不能当入场卡使用。
        entered_uid = event.source_uid if (event is not None and event.name in {"enter_field", "summon", "on_self_follower_entered_field"}) else None
        entered_card = snapshot.cards_by_uid.get(entered_uid) if entered_uid is not None else None
        ctx = ConditionEvaluationContext(
            action=self.action,
            source_card=source_card,
            event_player=event_player,
            turn_event_player=event.player if event is not None else None,
            event_source_card=event_source_card,
            event_source_base_card_id=source_base_card_id,
            event_field_card_ids=field_card_ids,
            entered_card=entered_card,
            # 「打出前」基线：奥义进度等补充信息只由协议在手牌卡上下发，卡进入
            # 场地后同一 uid 的 CardView 就没有这份信息了。用 source_snapshot
            # 才不会让 fanfare 的「奥义_10 / 解放奥义_15」条件恒为不成立。
            # 只对 fanfare 生效：其余事件（攻击时等）用打出前快照求墓地/手牌数
            # 会让条件看到过期的基线（attack 回归）。
            resource_snapshot=(
                self.source_snapshot
                if effect.trigger is not None and effect.trigger.event == "fanfare"
                else None
            ),
        )
        result = evaluator.evaluate_all(snapshot, effect, snapshot_conditions, ctx)
        return result.matched

    def _without_event_target_conditions(self, effect: ExecutableEffect) -> ExecutableEffect:
        conditions = tuple(condition for condition in effect.conditions if not condition.subject.startswith(("event_source.", "event_target.")))
        return effect if len(conditions) == len(effect.conditions) else replace(effect, conditions=conditions)

    def _without_core_spellboost_operations(self, effect: ExecutableEffect) -> ExecutableEffect:
        operations = tuple(operation for operation in effect.operations if operation.op not in {"change_cost", "modify_source_variable"})
        return effect if len(operations) == len(effect.operations) else replace(effect, operations=operations)

    def _record_used_generated_uids(self, events: tuple[ExecutionEvent, ...]) -> None:
        """登记本 effect 实际用掉的生成卡 uid（跨 handler 实例共享）。

        `op_summon` 发出的 summon / enter_field 事件把被召唤卡放在 **source_uid**、
        `target_uid` 为 None；draw / copy_card 则放在 target_uid。两者都要读。
        """

        for event in events:
            if event.name not in {"summon", "enter_field", "draw", "copy_card"}:
                continue
            uid = event.target_uid if event.target_uid is not None else event.source_uid
            if uid is not None:
                self._used_generated_uids.add(uid)

    def _generated_cards_for_effect(
        self,
        effect: ExecutableEffect,
        *,
        snapshot: BattleSnapshot | None = None,
        player: str | None = None,
    ) -> tuple[CardView, ...]:
        # history_same_name_pool / reanimate 效果必须从 destroyed history 物化，
        # 不能把本步其它召唤物混进来。但 replay 的 after 帧**就是**「被抽中的那几张」
        # 的证据（10903210 混沌监狱「随机 4 种与本次对战中被破坏的自己的随从同名」），
        # 不利用它就只能按历史顺序取前 N 个（录像实际是随机抽的 4 种）。
        # 这里按「base 在 destroyed history 里」过滤出候选，op 内部还会再校验一次。
        if any(op.op in {"history_same_name_pool", "reanimate"} for op in effect.operations):
            if snapshot is None:
                return ()
            return self._history_pool_generated_cards(effect, snapshot=snapshot, player=player)
        candidate_ids = effect_candidate_generated_base_ids(effect)
        if not candidate_ids:
            # spec 没写候选（如「召唤刚消失那张卡的复制随从」，base 只能从事件里拿到）：
            # 池子里混着本 step 双方各自生成的新卡，按 `replay_owner_side` 优先取同侧的，
            # 否则会把对手那张抓过来（与 LocalPredictor._generated_cards_for_effect 同口径）。
            if player in {"me", "opponent"}:
                same_side = tuple(
                    card
                    for card in self.generated_cards
                    if str(card.supplement_info.get("replay_owner_side") or "") in {"", player}
                )
                if same_side:
                    return same_side
            return self.generated_cards
        return tuple(card for card in self.generated_cards if card.base_card_id in candidate_ids or card.card_id in candidate_ids)

    def _history_pool_generated_cards(
        self,
        effect: ExecutableEffect,
        *,
        snapshot: BattleSnapshot,
        player: str | None,
    ) -> tuple[CardView, ...]:
        """history 池效果在本步可用的录制候选（按 destroyed history 过滤 + 截断）。"""
        resolver = default_card_resolver()
        sides = (player,) if player in {"me", "opponent"} else ("me", "opponent")
        history_card_ids: set[int] = set()
        for side in sides:
            view = snapshot.me if side == "me" else snapshot.opponent
            history_card_ids.update(int(item.card_id) for item in view.destroyed_card_ids)
        if not history_card_ids:
            return ()
        history_bases: set[int] = set()
        for card_id in history_card_ids:
            base = resolver.get_base_by_card_id(card_id)
            history_bases.add(int(base.base_card_id) if base is not None else card_id)
        count = 0
        params: dict[str, Any] = {}
        for operation in effect.operations:
            if operation.op in {"history_same_name_pool", "reanimate"}:
                params = dict(operation.params or {})
                if operation.amount is not None and getattr(operation.amount, "value", None):
                    count = int(operation.amount.value)
                break
        if count <= 0:
            count = int(params.get("count", 1) or 1)
        selected: list[CardView] = []
        for card in self.generated_cards:
            base = resolver.get_base_by_card_id(card.card_id)
            matches = card.card_id in history_card_ids or (
                base is not None and int(base.base_card_id) in history_bases
            )
            if not matches:
                continue
            selected.append(card)
            if len(selected) >= count:
                break
        return tuple(selected)

    def _summon_last_words_copy(self, snapshot: BattleSnapshot, event: ExecutionEvent) -> BattleSnapshot:
        """「谢幕曲：召唤1个与本随从同名的随从」的通用执行。"""
        leaving = event.payload.get("leaving_card") or {}
        if not isinstance(leaving, dict):
            return snapshot
        try:
            base_card_id = int(leaving.get("base_card_id") or 0)
        except (TypeError, ValueError):
            return snapshot
        if base_card_id <= 0 or event.player not in {"me", "opponent"}:
            return snapshot
        return summon_same_name_copy(
            snapshot,
            player=event.player,
            base_card_id=base_card_id,
            execution_context=self.execution_context,
            unresolved=self.state.unresolved,
        )

    def _allowed_persistent_zones(
        self,
        snapshot: BattleSnapshot,
        event: ExecutionEvent,
        source_uid: int | None,
    ) -> frozenset[str] | None:
        """直接路由段对该 source 卡允许执行的 persistent ``source_zone``。

        只有 ``turn_start`` / ``turn_end`` 这类「登记顺序」时点需要收紧：此时
        ``_source_cards`` 返回的是场地/纹章区域的卡，而同一张卡的模板里可能还有
        ``source_zone="hand"`` 的持续效果（「在手牌中发动」）。那些效果属于手牌
        订阅（registry hand scope），不能由场上的同一张卡执行。

        纹章 / 额外纹章 / 恩惠不收紧：纹章卡的部分效果被解析成
        ``source_zone="field"``，按区域过滤会漏掉它们。
        """
        if event.name not in {"turn_start", "turn_end"}:
            return None
        if source_uid is None or event.player not in {"me", "opponent"}:
            return None
        view = snapshot.me if event.player == "me" else snapshot.opponent
        if source_uid in view.field:
            return frozenset({"field"})
        return None

    def _source_cards(self, snapshot: BattleSnapshot, event: ExecutionEvent) -> tuple[tuple[int | None, CardView], ...]:
        if event.name in {"spellboost", "on_source_buffed"} and event.target_uid is not None and event.target_uid in snapshot.cards_by_uid:
            return ((event.target_uid, snapshot.cards_by_uid[event.target_uid]),)
        if event.name == "becomes_super_evolved" and event.player in {"me", "opponent"}:
            view = snapshot.me if event.player == "me" else snapshot.opponent
            return tuple((uid, snapshot.cards_by_uid[uid]) for uid in view.hand if uid in snapshot.cards_by_uid)
        if event.name in {"enter_field", "summon", "activate"} and event.player in {"me", "opponent"}:
            view = snapshot.me if event.player == "me" else snapshot.opponent
            source_cards = list((uid, snapshot.cards_by_uid[uid]) for uid in view.field if uid in snapshot.cards_by_uid)
            for crest in view.crests:
                source_cards.append((None, CardView(uid=0, card_id=crest.card_id, base_card_id=crest.card_id, name="")))
            # 随从入场时, 对侧场上的卡也可能有"对手随从进入战场时"(on_enemy_follower_entered_field)
            # 效果响应 (如森林陷阱)。返回对侧场上卡; 不匹配该 trigger 的效果自然 skip。
            if event.name in {"enter_field", "summon"}:
                other = snapshot.opponent if event.player == "me" else snapshot.me
                for uid in other.field:
                    if uid in snapshot.cards_by_uid:
                        source_cards.append((uid, snapshot.cards_by_uid[uid]))
            if source_cards:
                return tuple(source_cards)
        if event.name in {"attack_leader", "attack_follower"} and event.player in {"me", "opponent"}:
            view = snapshot.me if event.player == "me" else snapshot.opponent
            source_cards = []
            if event.source_uid is not None and event.source_uid in snapshot.cards_by_uid:
                source_cards.append((event.source_uid, snapshot.cards_by_uid[event.source_uid]))
            for crest in view.crests:
                source_cards.append((event.source_uid, CardView(uid=0, card_id=crest.card_id, base_card_id=crest.card_id, name="")))
            if source_cards:
                return tuple(source_cards)
        if event.name in {"draw", "draw_hidden"} and event.player in {"me", "opponent"}:
            view = snapshot.me if event.player == "me" else snapshot.opponent
            return tuple(
                (uid, snapshot.cards_by_uid[uid])
                for uid in view.field
                if uid in snapshot.cards_by_uid
            )
        if event.name in {"destroy", "destroyed_by_damage"} and event.player in {"me", "opponent"}:
            # 返回：1) 被破坏的卡本身（last_words source）；2) 同侧场上剩余的卡（持续效果）
            view = snapshot.me if event.player == "me" else snapshot.opponent
            source_cards: list[tuple[int | None, CardView]] = []
            # 被破坏的卡（可能已从 cards_by_uid 移除，从 payload 恢复）
            destroyed_uid = event.target_uid
            if destroyed_uid is not None:
                destroyed_card = snapshot.cards_by_uid.get(destroyed_uid)
                if destroyed_card is not None:
                    source_cards.append((destroyed_uid, destroyed_card))
                elif event.payload.get("leaving_card"):
                    lc = event.payload["leaving_card"]
                    source_cards.append((destroyed_uid, CardView(
                        uid=destroyed_uid,
                        card_id=int(lc.get("card_id", 0)),
                        base_card_id=int(lc.get("base_card_id", 0)),
                        name=str(lc.get("name", "")),
                        has_guard=bool(lc.get("has_guard", False)),
                        has_last_words=True,
                    )))
            # 同侧场上剩余的卡
            for uid in view.field:
                if uid in snapshot.cards_by_uid and uid != destroyed_uid:
                    source_cards.append((uid, snapshot.cards_by_uid[uid]))
            # 同侧纹章 (crest): 用 unique_id 作 source_uid, 使其能作为
            # on_self_amulet_destroyed 等持续效果订阅者被遍历并经 _card_owner_side
            # 正确归属。例: 信仰纹章 10664122 "自己的护符被破坏时信仰+1"。
            for crest in view.crests:
                source_cards.append((
                                    crest.unique_id,
                                    CardView(uid=crest.unique_id, card_id=crest.card_id,
                                             base_card_id=crest.card_id, name=""),
                                ))
            if source_cards:
                return tuple(source_cards)
        if event.name == "heal" and event.player in {"me", "opponent"} and event.target_uid is None:
            # 主战者回复时（target_uid is None 表示回复的是玩家/leader 而非具体随从）:
            # 检查同侧场上随从与纹章的 on_leader_healed 持续效果
            # (条规成员进化 / 誓言干部破坏敌随从 / 纹章等)。
            view = snapshot.me if event.player == "me" else snapshot.opponent
            source_cards: list[tuple[int | None, CardView]] = []
            for uid in view.field:
                if uid in snapshot.cards_by_uid:
                    source_cards.append((uid, snapshot.cards_by_uid[uid]))
            for crest in view.crests:
                source_cards.append((None, CardView(uid=0, card_id=crest.card_id, base_card_id=crest.card_id, name="")))
            if source_cards:
                return tuple(source_cards)
        if event.name == "damage" and event.target_uid is not None:
            # 随从受到伤害时，检查：
            # 1. 受伤卡本身是否有 on_self_damaged 的 persistent 效果
            # 2. 受伤方的纹章区域是否有持续效果触发
            target_card = snapshot.cards_by_uid.get(event.target_uid)
            if target_card is not None:
                if event.target_uid in snapshot.me.field:
                    view = snapshot.me
                elif event.target_uid in snapshot.opponent.field:
                    view = snapshot.opponent
                else:
                    view = None
                if view is not None:
                    source_cards: list[tuple[int | None, CardView]] = []
                    # 受伤卡本身
                    source_cards.append((event.target_uid, target_card))
                    # 同侧纹章
                    for crest in view.crests:
                        source_cards.append((None, CardView(uid=0, card_id=crest.card_id, base_card_id=crest.card_id, name="")))
                    return tuple(source_cards)
        if event.name == "discard" and event.target_uid is not None:
            # 被舍弃卡的 on_discard_self 效果
            discarded_info = event.payload.get("discarded_card")
            if discarded_info:
                base_card_id = int(discarded_info.get("base_card_id", 0))
                if base_card_id in self.templates_by_base_card_id:
                    discarded_card = CardView(
                        uid=event.target_uid,
                        card_id=int(discarded_info.get("card_id", 0)),
                        base_card_id=base_card_id,
                    )
                    return ((event.target_uid, discarded_card),)
        if event.name in {"last_words", "countdown"}:
            # 这些事件的 source 可能已从 cards_by_uid 中移除（归零/被破坏），
            # 必须从 payload 恢复以避免 uid 冲突
            sc = event.payload.get("source_card") or event.payload.get("leaving_card")
            if sc:
                card = CardView(
                    uid=event.source_uid or 0,
                    card_id=int(sc.get("card_id", 0)),
                    base_card_id=int(sc.get("base_card_id", 0)),
                    name=str(sc.get("name", "")),
                )
                return ((event.source_uid, card),)
        # 其他事件（becomes_evolved, discard 等）：
        # source_uid 指向 cards_by_uid 中现存的卡
        if event.source_uid is not None and event.source_uid in snapshot.cards_by_uid:
            return ((event.source_uid, snapshot.cards_by_uid[event.source_uid]),)
        if event.name in {"turn_start", "turn_end", "turn_start_field_effects", "turn_start_crest_effects", "turn_start_deck", "turn_end_deck"} and event.player in {"me", "opponent"}:
            view = snapshot.me if event.player == "me" else snapshot.opponent
            source_cards: list[tuple[int | None, CardView]] = []
            if event.name in {"turn_start", "turn_end"}:
                # 同一个时点会触发多个裁定时，场地区域与纹章区域**合并**结算，
                # 顺序按卡牌的登记顺序（进入持久效果区域的先后），而不是
                # 「先场地后纹章」——后者与登记顺序无关：
                # replay_...0006 st65 的纹章 10364122（uid 85，frame 30 登记）
                # 早于场上随从 10564110（uid 62，frame 61 才进手牌），却被排在其后，
                # 导致 3 点分配先让 10564110 的 debuff 清场、再全打主战者。
                # `persistent_registration_order` 由 `LocalPredictor` 逐 step 推进、
                # replay 工具按帧历史注入；缺失时退化为「场地序 → 纹章序」。
                field_cards = {
                    uid: snapshot.cards_by_uid[uid] for uid in view.field if uid in snapshot.cards_by_uid
                }
                crest_cards = {
                    crest.unique_id: CardView(uid=crest.unique_id, card_id=crest.card_id, base_card_id=crest.card_id, name="")
                    for crest in view.crests
                }
                # 额外纹章与恩惠同属持久效果区域，也参与登记顺序（§29.2：
                # 纹章「含额外纹章、恩惠」出现即登记）。不并入时它们只能在
                # registry 的剩余阶段执行，恒定排在所有场地/纹章之后。
                extra_cards = {
                    extra.unique_id: CardView(uid=extra.unique_id, card_id=extra.card_id, base_card_id=extra.card_id, name="")
                    for extra in view.extra_crests
                    if extra.unique_id
                }
                boon_cards = {
                    boon.unique_id: CardView(uid=boon.unique_id, card_id=boon.card_id, base_card_id=boon.card_id, name="")
                    for boon in view.boons
                    if boon.unique_id
                }
                for uid in advance_persistent_registration_order(view.persistent_registration_order, view):
                    field_card = field_cards.get(uid)
                    if field_card is not None:
                        source_cards.append((uid, field_card))
                        continue
                    crest_card = crest_cards.get(uid)
                    if crest_card is not None:
                        source_cards.append((uid, crest_card))
                        continue
                    extra_card = extra_cards.get(uid)
                    if extra_card is not None:
                        source_cards.append((uid, extra_card))
                        continue
                    boon_card = boon_cards.get(uid)
                    if boon_card is not None:
                        source_cards.append((uid, boon_card))
            elif event.name == "turn_start_field_effects":
                source_cards.extend((uid, snapshot.cards_by_uid[uid]) for uid in view.field if uid in snapshot.cards_by_uid)
            elif event.name == "turn_start_crest_effects":
                source_cards.extend((crest.unique_id, CardView(uid=crest.unique_id, card_id=crest.card_id, base_card_id=crest.card_id, name="")) for crest in view.crests)
            if event.name == "turn_start_deck" or event.name == "turn_end_deck":
                deck_source_view = self.source_snapshot.me if self.source_snapshot is not None and event.player == "me" else self.source_snapshot.opponent if self.source_snapshot is not None else view
                if deck_source_view.deck_instances is not None and deck_source_view.deck_instances.status == "available":
                    by_id = deck_source_view.deck_instances.by_instance_id()
                    for instance_id in deck_source_view.deck_instances.draw_pile:
                        instance = by_id.get(instance_id)
                        if instance is None or instance.zone != "deck":
                            continue
                        uid = instance.uid if instance.uid is not None else -instance_id
                        source_cards.append((uid, replace(instance.card, uid=uid)))
            # Cross-side turn_end: 另一侧场上有 turn_player condition 的 turn_end 效果卡
            # 例：安的巨大英灵 — 对手回合结束时破坏本卡牌
            if event.name == "turn_end":
                cross_view = snapshot.opponent if event.player == "me" else snapshot.me
                for uid in cross_view.field:
                    card = snapshot.cards_by_uid.get(uid)
                    if card is None:
                        continue
                    template = self.templates_by_base_card_id.get(card.base_card_id)
                    if template is None:
                        continue
                    if self._has_cross_side_turn_end_effect(template):
                        source_cards.append((uid, card))
            return tuple(source_cards)
        return ()

    @staticmethod
    def _has_cross_side_turn_end_effect(template: UnifiedCardTemplate) -> bool:
        """检查模板是否有跨侧 turn_end 效果（含 turn_player condition）。"""
        for effect in template.executable_effects:
            if effect.trigger.event != "turn_end":
                continue
            for condition in effect.conditions:
                if condition.subject == "turn_player":
                    return True
        return False

    @staticmethod
    def _card_owner_side(snapshot: BattleSnapshot, uid: int) -> str | None:
        """判断 uid 属于哪一侧。返回 'me' / 'opponent' / None。

        纹章 (crest) 用 unique_id 标识, 不在 field/hand 的 uid 集合里,
        故需额外检查 me/opponent 的 crests, 否则纹章作为持续效果订阅者
        (如信仰纹章的 on_self_amulet_destroyed) 归属侧判定会误判为 None 而被过滤。
        """
        if uid in snapshot.me.field or uid in snapshot.me.hand:
            return "me"
        if uid in snapshot.opponent.field or uid in snapshot.opponent.hand:
            return "opponent"
        if any(crest.unique_id == uid for crest in snapshot.me.crests):
            return "me"
        if any(crest.unique_id == uid for crest in snapshot.opponent.crests):
            return "opponent"
        if any(crest.unique_id == uid for crest in snapshot.me.extra_crests):
            return "me"
        if any(crest.unique_id == uid for crest in snapshot.opponent.extra_crests):
            return "opponent"
        if any(boon.unique_id == uid for boon in snapshot.me.boons):
            return "me"
        if any(boon.unique_id == uid for boon in snapshot.opponent.boons):
            return "opponent"
        return None

    @staticmethod
    def _option_index_of(effect: ExecutableEffect) -> int | None:
        """返回 effect 声明的 mode option 序号 (option_index), 无则 None。"""
        for operation in effect.operations:
            mode_option = operation.params.get("mode_option")
            if isinstance(mode_option, Mapping):
                idx = mode_option.get("option_index")
                if isinstance(idx, int):
                    return idx
        return None

    @staticmethod
    def _is_ordered_abilities_gate(effect: ExecutableEffect) -> bool:
        """effect 是否为 ordered_abilities 门控标记 (声明"按顺序发动1个能力")。"""
        return any(
            operation.params.get("ability_resolution_mode") == "ordered_abilities"
            for operation in effect.operations
        )

    def _reduce_ordered_abilities(
        self,
        snapshot: BattleSnapshot,
        source_uid: int | None,
        source_card: CardView,
        event: ExecutionEvent,
        matched_effects: tuple[ExecutableEffect, ...],
    ) -> tuple[tuple[ExecutableEffect, ...], BattleSnapshot]:
        """通用 "按顺序发动以下1个能力" (ordered_abilities) cursor 机制。

        若 matched_effects 中含 ordered_abilities 门控标记 (某 spec op 带
        ability_resolution_mode=ordered_abilities), 则本次事件只保留 mode option 序列中
        cursor 指向的那一个能力项, 并把 cursor 推进 1
        (持久化到 source_card.supplement_info["ordered_ability_cursor"], 跨事件/跨 action 递进)。

        非 ordered_abilities 卡 (无门控标记) 原样返回, 不影响既有多效果发动。
        """
        gate_effects = [e for e in matched_effects if self._is_ordered_abilities_gate(e)]
        if not gate_effects:
            return matched_effects, snapshot
        gate = gate_effects[0]
        option_effects = [
            e for e in matched_effects
            if not self._is_ordered_abilities_gate(e) and self._option_index_of(e) is not None
        ]
        # ordered 能力项的命名约定是 `{base}_option_N`，与门控 `{base}_ordered_effect`
        # 同前缀。同一张卡可能还有另一套顶层 `{base}_mode_N` 效果（同一段文本的
        # 「模式选择」解析产物），它们的 option_index 与 ordered 能力重叠，混进来
        # 会让 cursor 指向错误的能力（11024122：cursor=1 命中 mode_1 而不是 option_2，
        # 本地加『黄金短剑』、录像是『黄金之杯』）。
        option_prefix = gate.effect_id.split("_ordered_effect")[0] + "_option_"
        prefixed_options = [e for e in option_effects if e.effect_id.startswith(option_prefix)]
        if prefixed_options:
            option_effects = prefixed_options
        # 非门控、非 option 的普通 effect 原样保留 (同卡若还有其它独立触发效果),
        # 由后续 per-effect 循环各自做 scope/condition 过滤。
        other_effects = [
            e for e in matched_effects
            if not self._is_ordered_abilities_gate(e) and self._option_index_of(e) is None
        ]
        # 只有当门控 effect 的触发范围/条件匹配本事件时, 才推进 cursor 并发动能力。
        # (如"被破坏的不是己方护符"则整组能力不触发, cursor 不动; 门控与 option 都不发动。)
        if not self._event_trigger_scope_matches(snapshot, event, gate, source_uid):
            return tuple(other_effects), snapshot
        # 门控标记本身不执行 (占位 mode_choice_effect op), 从序列剔除并标记 consumed。
        for e in gate_effects:
            self.state.consumed.append(e.effect_id)
        if not option_effects:
            self.state.unresolved.append(f"ordered_abilities_no_executable_option:{gate.effect_id}")
            return tuple(other_effects), snapshot
        option_effects.sort(key=lambda e: self._option_index_of(e))
        cursor = 0
        is_crest_subscriber = source_uid is None or source_uid == 0 or source_uid not in snapshot.cards_by_uid
        live = snapshot.cards_by_uid.get(source_uid) if source_uid is not None else None
        # use_protocol_cursor：spec 显式声明「本卡的按顺序发动游标来自协议的
        # current_run_in_order_count」。不能无条件用该字段 —— 有些卡的该字段
        # 语义不是能力索引（无条件使用会让它们的回合结束结算错位，
        # replay_...1431478 st43 / ...1302905 st73）。
        use_protocol_cursor = any(
            (op.params or {}).get("use_protocol_cursor") for op in gate.operations
        )
        if live is not None:
            raw_cursor = live.supplement_info.get("ordered_ability_cursor")
            if raw_cursor is None and use_protocol_cursor:
                # 协议字段是**结算后**的值：after 帧的 current_run_in_order_count 已经
                # 推进到本 step 结束的位置（10964120：BEFORE=0 / AFTER=3）。拿它当起点
                # 会让第一次触发直接从能力 3 之后开始，后续事件顺序整体错位。
                # 起点必须取 before 帧；before 帧没有该卡（本 step 刚入场）时
                # 退回当前 snapshot 的字段，那是唯一可用来源。
                before_live = (
                    self.source_snapshot.cards_by_uid.get(source_uid)
                    if self.source_snapshot is not None
                    else None
                )
                raw_cursor = getattr(before_live or live, "current_run_in_order_count", 0)
            cursor = max(0, int(raw_cursor or 0))
        elif is_crest_subscriber:
            # 纹章没有实体 uid（source_uid=None/0）：cursor 持久化在 CrestView.supplement_info，
            # 否则每个事件都会从 option 1 重新开始。
            for side in ("me", "opponent"):
                view = getattr(snapshot, side)
                for crest in view.crests:
                    if crest.card_id == source_card.base_card_id:
                        raw_cursor = crest.supplement_info.get("ordered_ability_cursor")
                        if raw_cursor is None:
                            # 录像协议字段 current_run_in_order_count：-1 表示尚未开始，
                            # 0/1…表示已经发动到第几个能力，可直接作为 cursor。
                            raw_cursor = getattr(crest, "current_run_in_order_count", -1)
                        cursor = max(0, int(raw_cursor or 0))
                        break
        else:
            cursor = int(source_card.supplement_info.get("ordered_ability_cursor", 0) or 0)
        selected = option_effects[cursor % len(option_effects)]
        for e in option_effects:
            if e is not selected:
                self.state.consumed.append(e.effect_id)
        # 推进并持久化 cursor 到 snapshot.cards_by_uid, 供后续 destroy 事件递进。
        updated_snapshot = snapshot
        if source_uid is not None and source_uid in snapshot.cards_by_uid:
            live = snapshot.cards_by_uid[source_uid]
            new_supplement = {**live.supplement_info, "ordered_ability_cursor": cursor + 1}
            updated_card = replace(live, supplement_info=new_supplement)
            updated_snapshot = replace(
                snapshot,
                cards_by_uid={**snapshot.cards_by_uid, source_uid: updated_card},
            )
        elif is_crest_subscriber:
            for side in ("me", "opponent"):
                view = getattr(snapshot, side)
                updated_crests = tuple(
                    replace(crest, supplement_info={**crest.supplement_info, "ordered_ability_cursor": cursor + 1})
                    if crest.card_id == source_card.base_card_id
                    else crest
                    for crest in view.crests
                )
                if updated_crests != view.crests:
                    updated_snapshot = replace(updated_snapshot, **{side: replace(view, crests=updated_crests)})
                    break
        return (*other_effects, selected), updated_snapshot


    def _sample_random_targets_for_effect(self, snapshot: BattleSnapshot, effect: ExecutableEffect, *, player: str, opponent: str, source_uid: int | None) -> tuple[int, ...]:
        """为含 random target 的效果自动采样随机目标。"""
        import random as _random
        targets: list[int] = []
        rng = _random.Random(self.execution_context.branch_seed)
        for operation in effect.operations:
            target_ref = operation.target_ref or ""
            if not target_ref.startswith("random_"):
                # repeat 内嵌 operation
                nested = operation.params.get("operation")
                if isinstance(nested, dict):
                    target_ref = nested.get("target_ref", "")
                if not target_ref.startswith("random_"):
                    continue
            candidates = self.effect_interpreter.random_target_candidates(
                snapshot, target_ref, player=player, opponent=opponent, source_uid=source_uid
            )
            if not candidates:
                continue
            # repeat count
            count = int(operation.params.get("count", 1) or 1) if operation.op == "repeat" else 1
            for _ in range(count):
                if not candidates:
                    break
                pick = rng.choice(candidates)
                targets.append(pick)
        return tuple(targets)

    def _selected_targets(self, event: ExecutionEvent) -> tuple[int, ...]:
        if event.target_uid is not None:
            return (event.target_uid,)
        if event.name in {"enter_field", "summon"} and event.source_uid is not None:
            return (event.source_uid,)
        if event.source_uid == self.action.source_uid:
            return tuple(uid for group in self.action.target_groups for uid in group)
        return ()


def _random_target_ref_count(operations) -> int:
    """统计一组 operation（含 repeat 嵌套）中的独立 random target pool 数。"""

    refs: set[str] = set()
    for operation in operations:
        target_ref = operation.target_ref or ""
        if target_ref.startswith("random_"):
            refs.add(target_ref)
        nested = operation.params.get("operation") if isinstance(operation.params, dict) else None
        if isinstance(nested, dict):
            nested_ref = str(nested.get("target_ref", "") or "")
            if nested_ref.startswith("random_"):
                refs.add(nested_ref)
    return len(refs)

def summon_same_name_copy(
    snapshot: BattleSnapshot,
    *,
    player: str,
    base_card_id: int,
    execution_context: object,
    unresolved: list[str],
) -> BattleSnapshot:
    """「谢幕曲：召唤1个与本随从同名的随从」的通用执行。

    优先沿用 replay 绑定（after 帧新增的同名卡，uid/card_id/关键字全部对齐），
    否则用牌库物化生成新实例。handler 的事件路径与 LocalPredictor 的补处理路径
    共用这一份实现，避免两条路径对动态谢幕曲给出不同结果。
    """
    from decision_engine.simulation.deck_pool import materialize_draw_card

    from .generated_card_materializer import default_card_resolver
    from .rule_kernel import CoreRuleKernel

    used = set(snapshot.cards_by_uid)
    bindings = tuple(getattr(execution_context, "summon_copy_bindings", ()) or ())
    bound = next(
        (item for item in bindings if item.base_card_id == base_card_id and item.uid not in used),
        None,
    )
    if bound is not None:
        return CoreRuleKernel().summon_card(snapshot, player, bound).snapshot
    uid = -1
    while uid in used:
        uid -= 1
    card = materialize_draw_card(base_card_id, snapshot, uid=uid, resolver=default_card_resolver())
    if card is None:
        unresolved.append(f"last_words_summon_copy_materialize_failed:{base_card_id}")
        return snapshot
    return CoreRuleKernel().summon_card(snapshot, player, card).snapshot


def card_view_from_leaving_payload(
    uid: int | None,
    payload: dict[str, object],
    *,
    with_card_type: bool,
    has_guard: bool = False,
) -> CardView:
    """从事件 payload 的离场前快照恢复 CardView（用于已离开 snapshot 的事件对象）。"""
    card_type = CardType.FOLLOWER
    if with_card_type and payload.get("card_type") is not None:
        try:
            card_type = CardType(payload["card_type"])
        except ValueError:
            card_type = CardType.FOLLOWER
    return CardView(
        uid=uid or 0,
        card_id=int(payload.get("card_id", 0)),
        base_card_id=int(payload.get("base_card_id", 0)),
        name=str(payload.get("name", "")),
        card_type=card_type,
        has_guard=bool(payload.get("has_guard", has_guard)),
    )


def event_target_conditions_match(
    snapshot: BattleSnapshot,
    event: ExecutionEvent,
    effect: ExecutableEffect,
    unresolved: list[str],
) -> bool:
    """评估 event_source.* / event_target.* 条件。

    模块级函数：事件队列消费阶段（TemplateEventHandler）与动作前预结算阶段
    （LocalPredictor 的「攻击时」自身强化）必须共用同一套判定语义，否则打脸/
    攻击随从的 event_target 条件会在预结算里失效。
    """
    event_source_card = snapshot.cards_by_uid.get(event.source_uid or 0)
    # 如果 source_uid 已不在 cards_by_uid 中（如被破坏的卡），从 payload 恢复关键属性。
    # leaving_card 的语义是「事件源自身离场前状态」（destroy / last_words / countdown）；
    # attack_follower 的攻击者可能被这次攻击反击致死，用它专属的 leaving_source_card。
    if event_source_card is None:
        source_payload = event.payload.get("leaving_source_card") or event.payload.get("leaving_card")
        if isinstance(source_payload, dict):
            event_source_card = card_view_from_leaving_payload(
                event.source_uid,
                source_payload,
                with_card_type=False,
                has_guard=bool(event.payload.get("has_guard", False)),
            )
    if event_source_card is None:
        # 法术使用后立即离开手牌，snapshot.cards_by_uid 里已经没有它，
        # play_spell 事件也不带 leaving_card。此时用 payload 里的 source_base_card_id
        # 造一个仅含 card_id 的 CardView —— effective_tribes 会回退 resolver 查卡面
        # 模板的 tribes，让「自己使用玛纳利亚·法术时」这类条件能命中
        # （10833112 纹章：事件源为 90031310 玛纳利亚魔弹）。
        raw_base_id = event.payload.get("source_base_card_id")
        try:
            base_id = int(raw_base_id) if raw_base_id is not None else None
        except (TypeError, ValueError):
            base_id = None
        if base_id is not None:
            event_source_card = CardView(uid=event.source_uid or 0, card_id=base_id, base_card_id=base_id)
    # attack 事件的 event_target 严格来自 target_uid：
    # attack_follower 带防御随从 uid；attack_leader 攻击主战者时 target_uid 为 None，
    # 表示"没有随从目标"，不能 fallback 到攻击者自身（否则"攻击随从时"条件会在打脸时误命中）。
    if event.name in {"attack_leader", "attack_follower"}:
        event_target_card = snapshot.cards_by_uid.get(event.target_uid) if event.target_uid is not None else None
    else:
        event_target_card = snapshot.cards_by_uid.get((event.target_uid if event.target_uid is not None else event.source_uid) or 0)
    event_cards = {
        "event_source": event_source_card,
        "event_target": event_target_card,
    }
    # 如果 event_target 已从 snapshot 中移除（如被这次攻击击杀的防御随从），
    # 从 payload 恢复，并带上 card_type，
    # 使「攻击随从时」等 event_target.card_type 条件在目标已死时仍能正确判定。
    if event_cards["event_target"] is None:
        target_payload = event.payload.get("leaving_target_card")
        if not isinstance(target_payload, dict) and event.target_uid in (None, event.source_uid):
            # destroy / last_words 等自引用事件：目标就是事件源自身。
            target_payload = event.payload.get("leaving_card")
        if isinstance(target_payload, dict):
            event_cards["event_target"] = card_view_from_leaving_payload(
                event.target_uid if event.target_uid is not None else event.source_uid,
                target_payload,
                with_card_type=True,
                has_guard=bool(event.payload.get("has_guard", False)),
            )
    for condition in effect.conditions:
        prefix = ""
        if condition.subject.startswith("event_source."):
            prefix = "event_source"
        elif condition.subject.startswith("event_target."):
            prefix = "event_target"
        else:
            continue
        target = event_cards[prefix]
        if target is None:
            return False
        subject = condition.subject.removeprefix(f"{prefix}.")
        if subject == "has_guard":
            value = 1 if target.has_guard else 0
        elif subject == "card_type":
            # "攻击随从时"等语义：要求 event_target 是指定卡类型（如 follower）。
            # 攻击主战者时 event_target 为 None，已在上方 target is None 分支返回 False。
            expected_type = str(condition.value)
            actual_type = getattr(target.card_type, "value", None)
            matched = actual_type == expected_type
            if condition.operator == "==" and not matched:
                return False
            if condition.operator == "!=" and matched:
                return False
            continue
        elif subject == "is_other":
            # "入场的是其他随从（不是持有效果的卡本身）"：由外层 source_uid 判定。
            continue
        elif subject in {"evolve_state", "atk", "attack", "life"}:
            value = int(getattr(target, "atk" if subject == "attack" else subject))
        elif subject in {"cost", "original_cost"}:
            # 「抽到/使用费用为N的卡牌时」类条件：与 selector / interpreter 的
            # cost 口径一致，统一读 CardView.cost（含本回合减费修正后的当前费用）。
            value = int(target.cost)
        elif subject in {"base_card_id", "card_id"}:
            # 「自己的『X』攻击时，使其…」类条件：按事件源/目标的卡牌 id 门控。
            # 变体 id 只影响卡面展示，效果口径以 base_card_id 为准，故 base_card_id
            # 直接读 CardView.base_card_id（协议已归一），card_id 才读具体变体。
            if subject == "base_card_id":
                actual_id = int(target.base_card_id or target.card_id)
            else:
                actual_id = int(target.card_id)
            expected_id = int(condition.value)
            if condition.operator in {"==", "=", "eq"} and actual_id != expected_id:
                return False
            if condition.operator in {"!=", "ne"} and actual_id == expected_id:
                return False
            continue
        elif subject == "tribes":
            # tribes 使用 "contains" operator。必须走 effective_tribes：协议帧不下发
            # tribes，直读 CardView.tribes 会让「玛纳利亚·法术」「海洋·随从」这类
            # 事件源种族条件恒不命中。
            if condition.operator == "contains":
                if int(condition.value) not in _effective_tribes(target):
                    return False
                continue
            unresolved.append(f"event_condition_not_supported:{effect.effect_id}:{condition.subject}:{condition.operator}")
            return False
        else:
            unresolved.append(f"event_condition_not_supported:{effect.effect_id}:{condition.subject}")
            return False
        if condition.operator == "in":
            # 枚举式条件（「费用为1、3、5的卡牌」）：value 是离散集合。
            candidates = condition.value
            if not isinstance(candidates, (list, tuple, set, frozenset)):
                unresolved.append(f"event_condition_not_supported:{effect.effect_id}:{condition.subject}:in")
                return False
            if value not in {int(item) for item in candidates}:
                return False
            continue
        expected = int(condition.value)
        if condition.operator == ">=" and not value >= expected:
            return False
        if condition.operator == "==" and not value == expected:
            return False
    return True


def _match_name_filter(actual_name: str, operator: str, value: object) -> bool:
    """entered_filters 的 name 字段匹配（「自己的『天晶魔手』进入战场时…」）。

    `in` / `any_of` 接受离散名称集合，`contains` 做子串匹配，其余交给 `_compare_value`。
    """
    actual = str(actual_name or "")
    if operator in {"in", "any_of"}:
        candidates = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
        return actual in {str(item) for item in candidates}
    if operator == "contains":
        return str(value) in actual
    return _compare_value(actual, operator, str(value))


def _match_base_card_id_filter(actual_base_card_id: int, operator: str, value: object) -> bool:
    """entered_filters 的 base_card_id 字段匹配；`in` 接受离散 id 集合。"""
    if operator in {"in", "any_of"}:
        candidates = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
        try:
            expected = {int(item) for item in candidates}
        except (TypeError, ValueError):
            return False
        return int(actual_base_card_id) in expected
    try:
        expected_id = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return _compare_value(int(actual_base_card_id), operator, expected_id)
