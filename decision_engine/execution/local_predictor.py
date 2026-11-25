# -*- coding: utf-8 -*-
"""Phase L7 本地预测器。

LocalPredictor 只消费本地规则内核和通用效果解释器；不调用旧 heuristic predictor，不 fallback 到协议动作空间。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from decision_engine.actions.model import ActionKind, LegalAction, PlayKind
from decision_engine.cards.condition_plan import ConditionPlanCache
from decision_engine.cards.templates import ExecutableEffect, TargetSelector, UnifiedCardTemplate
from decision_engine.simulation.predictor import PredictionCertainty, PredictionResult
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView

from .condition_evaluator import ConditionEvaluationContext, ConditionEvaluationResult, ConditionEvaluator
from .condition_progress import ConditionProgressProjector
from .effect_interpreter import EffectInterpretationContext, EffectInterpreter, EffectInterpreterError
from .effect_ops.resolution import explicit_random_target_repeat
from .execution_context import LocalExecutionContext
from .effect_trigger_resolver import resolve_template_effects_for_action
from .events import EventKind, EventQueue, EventQueueError, ExecutionEvent
from .generated_card_materializer import (
    GeneratedCardMaterialization,
    default_card_resolver,
    generated_card_requirement_for_effect,
    materialize_effect_generated_cards,
    operation_generated_card_count,
    select_recorded_generated_cards_for_effect,
)
from .rule_kernel import CoreRuleKernel
from .target_selector_materializer import materialize_effect_target_groups
from .state_ops import (
    EXTERNAL_STATIC_KEYWORD_CARD_FIELDS,
    SetCardField,
    static_keyword_card_updates,
)
from .template_event_handler import EventEffectState, TemplateEventHandler, event_target_conditions_match, summon_same_name_copy
from .persistent_effect_registry import PersistentEffectRegistry, advance_persistent_registration_order


_CERTAINTY_RANK = {
    PredictionCertainty.EXACT: 5,
    PredictionCertainty.SPEC_ONLY: 4,
    PredictionCertainty.PARTIAL: 3,
    PredictionCertainty.APPROX: 2,
    PredictionCertainty.UNKNOWN: 1,
    PredictionCertainty.PREDICT_ERROR: 0,
}

# effect_interpreter 对 InsufficientResourceError 的统一标记。资源不足时整条效果
# 已回滚，结果是确定的「效果未发动」，不是预测缺口。
_INSUFFICIENT_RESOURCE_MARKER = "effect not activated (insufficient resource)"


def _certainty_rank(certainty: PredictionCertainty) -> int:
    return _CERTAINTY_RANK.get(certainty, 0)


class LocalPredictor:
    def __init__(
        self,
        *,
        kernel: CoreRuleKernel | None = None,
        effect_interpreter: EffectInterpreter | None = None,
        templates_by_base_card_id: Mapping[int, UnifiedCardTemplate] | None = None,
        enable_event_queue_effects: bool = True,
        condition_evaluator: ConditionEvaluator | None = None,
    ):
        self.kernel = kernel or CoreRuleKernel()
        self.effect_interpreter = effect_interpreter or EffectInterpreter(self.kernel, card_resolver=default_card_resolver())
        self.templates_by_base_card_id = dict(templates_by_base_card_id or {})
        self.enable_event_queue_effects = enable_event_queue_effects
        self.condition_evaluator = condition_evaluator or ConditionEvaluator()
        # 条件计划在实例生命周期内缓存；条件进度随每次 action 转移物化（见 condition_progress.py）。
        self.condition_plans = ConditionPlanCache(self.templates_by_base_card_id)
        self.condition_progress = ConditionProgressProjector(self.condition_plans, self.condition_evaluator)

    def predict(self, snapshot: BattleSnapshot, action: LegalAction) -> PredictionResult:
        return self._predict_and_refresh_local_action_flags(snapshot, action, generated_cards=(), execution_context=LocalExecutionContext())

    def predict_with_generated_cards(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        *,
        generated_cards: tuple[CardView, ...] = (),
    ) -> PredictionResult:
        return self._predict_and_refresh_local_action_flags(snapshot, action, generated_cards=generated_cards, execution_context=LocalExecutionContext())

    def predict_with_replay_context(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        *,
        generated_cards: tuple[CardView, ...] = (),
        execution_context: LocalExecutionContext | None = None,
    ) -> PredictionResult:
        return self._predict_and_refresh_local_action_flags(snapshot, action, generated_cards=generated_cards, execution_context=execution_context or LocalExecutionContext(replay_mode=True))

    def _predict_and_refresh_local_action_flags(self, snapshot: BattleSnapshot, action: LegalAction, *, generated_cards: tuple[CardView, ...], execution_context: LocalExecutionContext) -> PredictionResult:
        external_root = snapshot.source == "battle_root_mpo"
        prepared_snapshot = self._hydrate_external_static_template_fields(snapshot) if external_root else snapshot
        result = self._predict(prepared_snapshot, action, generated_cards=generated_cards, execution_context=execution_context)
        if result.certainty is PredictionCertainty.PREDICT_ERROR:
            return result
        # zone/cards_by_uid 一致性修复：防止上游效果执行留下孤立 uid
        sanitized = self._sanitize_zone_consistency(result.snapshot)
        if sanitized is not result.snapshot:
            result = replace(result, snapshot=sanitized)
        refreshed = self._refresh_local_action_flags(result.snapshot)
        refreshed = self._advance_persistent_registration_order(refreshed)
        refreshed = self.condition_progress.refresh(refreshed)
        refreshed = self._check_lethal(refreshed)
        if refreshed is result.snapshot:
            return result
        return replace(result, snapshot=refreshed)

    def _hydrate_external_static_template_fields(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        """补齐外部协议root未显式传输的模板固有能力。

        该水合只设置缺失的正向静态字段，正常分支已具备字段时为幂等操作；
        不改写 snapshot.source，保持 replay after-state 可验收。能力移除后的反向静态
        状态需由显式状态模型表达，不能在这里猜测。"""
        if not self.templates_by_base_card_id:
            return snapshot
        keyword_fields = EXTERNAL_STATIC_KEYWORD_CARD_FIELDS
        cards = dict(snapshot.cards_by_uid)
        changed = False
        for uid in (*snapshot.me.field, *snapshot.opponent.field):
            card = cards.get(uid)
            if card is None:
                continue
            template = self.templates_by_base_card_id.get(card.base_card_id)
            if template is None:
                continue
            explicit = frozenset(getattr(card, "protocol_keyword_fields", ()) or ())
            updates = {
                field_name: True
                for keyword in template.static_keywords
                if (field_name := keyword_fields.get(keyword.keyword)) is not None
                and not getattr(card, field_name)
                # 协议显式下发过该字段（包括显式 False）时不得用模板补全：
                # 那是「失去关键字」的明确状态，不是「缺字段」。
                and field_name not in explicit
            }
            if updates:
                cards[uid] = replace(card, **updates)
                changed = True
        return replace(snapshot, cards_by_uid=cards) if changed else snapshot


    def _refresh_local_action_flags(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        if not snapshot.is_ally_turn:
            return snapshot
        can_evolve = snapshot.me.ep > 0 and not snapshot.me.is_evolved_this_turn and snapshot.me.evolve_turn > 0 and snapshot.me.turn >= snapshot.me.evolve_turn
        can_super_evolve = snapshot.me.sep > 0 and not snapshot.me.is_evolved_this_turn and snapshot.me.super_evolve_turn > 0 and snapshot.me.turn >= snapshot.me.super_evolve_turn
        changed = False
        cards = dict(snapshot.cards_by_uid)
        for uid in snapshot.me.field:
            card = cards.get(uid)
            if card is None or not card.card_type.is_follower:
                continue
            next_can_evolve = can_evolve and card.evolve_state == 0
            next_can_super_evolve = can_super_evolve and card.evolve_state == 0
            if card.can_evolve != next_can_evolve or card.can_super_evolve != next_can_super_evolve:
                cards[uid] = replace(card, can_evolve=next_can_evolve, can_super_evolve=next_can_super_evolve)
                changed = True
        if not changed:
            return snapshot
        return replace(snapshot, cards_by_uid=cards, legal=None)

    def _advance_persistent_registration_order(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        """把本 step 的入场/离场固化进持久效果登记顺序（供后续 step 排序用）。

        角色与 ``PersistentEffectRegistry.from_snapshot`` 的分工：registry 只消费
        顺序，不产生顺序；顺序必须在每个 step 的终态上推进一次，离场项才会被真正
        移除，重新入场才有「重新登记」的新位置。self-play 与 replay 走同一条路径。
        """
        me_order = advance_persistent_registration_order(snapshot.me.persistent_registration_order, snapshot.me)
        opponent_order = advance_persistent_registration_order(snapshot.opponent.persistent_registration_order, snapshot.opponent)
        if me_order == snapshot.me.persistent_registration_order and opponent_order == snapshot.opponent.persistent_registration_order:
            return snapshot
        return replace(
            snapshot,
            me=replace(snapshot.me, persistent_registration_order=me_order),
            opponent=replace(snapshot.opponent, persistent_registration_order=opponent_order),
        )

    def _check_lethal(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        """如果任一方 life <= 0 且 result 未设置，设置 battle result。"""
        if snapshot.me.result != 0 or snapshot.opponent.result != 0:
            return snapshot
        if snapshot.opponent.life <= 0:
            return replace(snapshot,
                me=replace(snapshot.me, result=101),
                opponent=replace(snapshot.opponent, result=102))
        if snapshot.me.life <= 0:
            return replace(snapshot,
                me=replace(snapshot.me, result=102),
                opponent=replace(snapshot.opponent, result=101))
        return snapshot

    def _predict(self, snapshot: BattleSnapshot, action: LegalAction, *, generated_cards: tuple[CardView, ...], execution_context: LocalExecutionContext) -> PredictionResult:
        try:
            if action.kind == ActionKind.PLAY:
                return self._play(snapshot, action, generated_cards=generated_cards, execution_context=execution_context)
            if action.kind == ActionKind.ATTACK_LEADER:
                return self._attack_leader(snapshot, action, generated_cards=generated_cards, execution_context=execution_context)
            if action.kind == ActionKind.ATTACK_FOLLOWER:
                return self._attack_follower(snapshot, action, generated_cards=generated_cards, execution_context=execution_context)
            if action.kind == ActionKind.EVOLVE:
                return self._evolve(snapshot, action, generated_cards=generated_cards, execution_context=execution_context)
            if action.kind == ActionKind.SUPER_EVOLVE:
                return self._super_evolve(snapshot, action, generated_cards=generated_cards, execution_context=execution_context)
            if action.kind == ActionKind.ACTIVATE:
                return self._activate(snapshot, action, generated_cards=generated_cards, execution_context=execution_context)
            if action.kind == ActionKind.SPECIAL_ACTION:
                return self._special_action(snapshot, action, generated_cards=generated_cards, execution_context=execution_context)
            if action.kind == ActionKind.FUSION:
                return self._fusion(snapshot, action, generated_cards=generated_cards, execution_context=execution_context)
            if action.kind == ActionKind.EXTRA_PP:
                return self._extra_pp(snapshot, action)
            if action.kind == ActionKind.TURN_START:
                return self._turn_start(snapshot, action, generated_cards=generated_cards, execution_context=execution_context)
            if action.kind == ActionKind.TURN_END:
                return self._turn_end(snapshot, action, generated_cards=generated_cards, execution_context=execution_context)
            return PredictionResult(
                snapshot=snapshot,
                certainty=PredictionCertainty.PREDICT_ERROR,
                description=f"unsupported local action kind: {action.kind.value}",
                unresolved_effects=(action.kind.value,),
            )
        except Exception as exc:
            import traceback as _tb
            tb_short = _tb.format_exception_only(type(exc), exc)[-1].strip()
            # 状态不一致类错误（zone/cards_by_uid 不同步导致的 card mismatch / KeyError）
            # 降级为 PARTIAL 让 self-play 继续
            from .rule_kernel import RuleKernelError
            exc_msg = str(exc)
            is_state_inconsistency = (
                (isinstance(exc, RuleKernelError) and "card mismatch" in exc_msg)
                or (isinstance(exc, KeyError) and action.kind in (ActionKind.TURN_START, ActionKind.TURN_END))
            )
            if is_state_inconsistency:
                return PredictionResult(
                    snapshot=snapshot,
                    certainty=PredictionCertainty.PARTIAL,
                    description=f"{tb_short} | action={action.kind.value} src={action.source_uid} targets={action.target_groups}",
                    unresolved_effects=("kernel_state_inconsistency",),
                )
            return PredictionResult(
                snapshot=snapshot,
                certainty=PredictionCertainty.PREDICT_ERROR,
                description=f"{tb_short} | action={action.kind.value} src={action.source_uid} targets={action.target_groups}",
                unresolved_effects=(action.kind.value,),
            )

    def _play(self, snapshot: BattleSnapshot, action: LegalAction, *, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        card = snapshot.cards_by_uid.get(action.source_uid)
        if card is None:
            return self._error(snapshot, "card not found", action)
        cost = action.cost if action.cost is not None else card.cost
        # Enhance mode 推断：action.play_kind 未标记时，从 template play_modes 推断
        # 规则：PP >= enhance_cost 时强制优先以爆能费用打出；多个时选最高可负担的
        effective_play_kind = action.play_kind
        if effective_play_kind is None and card.base_card_id in self.templates_by_base_card_id:
            tmpl = self.templates_by_base_card_id[card.base_card_id]
            enhance_cost = self._resolve_enhance_cost(tmpl, snapshot.me.pp)
            if enhance_cost is not None:
                effective_play_kind = PlayKind.ENHANCE
                cost = enhance_cost
            elif snapshot.me.pp < card.cost:
                # PP 不够原始费用 → 检查激奏/结晶
                accel_cost = self._resolve_accelerate_cost(tmpl, snapshot.me.pp)
                if accel_cost is not None:
                    effective_play_kind = PlayKind.ACCELERATE
                    cost = accel_cost
        if effective_play_kind is PlayKind.ENHANCE:
            return self._play_enhanced(snapshot, action, cost=cost, generated_cards=generated_cards, execution_context=execution_context)
        if effective_play_kind is PlayKind.ACCELERATE:
            return self._play_accelerated(snapshot, action, cost=cost, generated_cards=generated_cards, execution_context=execution_context)
        if effective_play_kind is PlayKind.CRYSTAL:
            return self._play_crystallized(snapshot, action, cost=cost, generated_cards=generated_cards, execution_context=execution_context)
        if card.card_type.is_spell:
            result = self.kernel.play_spell_from_hand(snapshot, "me", action.source_uid, cost)
            after_core = self._apply_hand_spellboost_effects(snapshot, result.snapshot, generated_cards=generated_cards)
            after_core = self._sync_spellboost_hand_costs(snapshot, after_core)
            return self._apply_template_effects(
                before=snapshot,
                after_core=after_core,
                action=action,
                events=result.events,
                default_certainty=PredictionCertainty.SPEC_ONLY,
                default_description="local play spell core rules only",
                default_assumptions=("no_spell_effects_applied",),
                default_unresolved=("play_spell_effects",) if self._template_has_trigger_event(card.base_card_id, "on_spell_play") else (),
                generated_cards=generated_cards,
                execution_context=execution_context,
            )
        result = self.kernel.play_card_from_hand(snapshot, "me", action.source_uid, cost)
        after_core = self._sync_played_field_card_template_stats(snapshot, result.snapshot, action.source_uid, card)
        # 证据护符在场上唯一：从手牌打出的证据护符也要替换同侧旧证据。
        played_template = self.templates_by_base_card_id.get(card.base_card_id)
        extra_events: list[ExecutionEvent] = []
        if played_template is not None and 21 in tuple(played_template.identity.tribes or ()):
            for old_uid in list(after_core.me.field):
                if old_uid == action.source_uid:
                    continue
                old_card = after_core.cards_by_uid.get(old_uid)
                if old_card is None:
                    continue
                old_template = self.templates_by_base_card_id.get(old_card.base_card_id)
                if old_template is None or 21 not in tuple(old_template.identity.tribes or ()):
                    continue
                banish = self.kernel.banish_field_card(after_core, "me", old_uid)
                after_core = banish.snapshot
                # 旧证据的离场必须作为事件进入队列：11003210「本卡牌在战场上消失时，
                # 回复自己的主战者2点生命值」只能靠 BANISH 触发（之前丢弃了
                # banish.events，录像 life 13->15、本地不动）。
                extra_events.extend(banish.events)
        # 已融合的卡从手牌打出时，重放 fusion trigger 的效果（replay 中材料在
        # before 快照已消耗，不能再走 record_fusion 扣材料）。
        if played_template is not None and (getattr(card, "fusion_list", ()) or card.supplement_info.get("fusion_count")):
            fusion_effects = [effect for effect in played_template.executable_effects if effect.trigger.event == "fusion"]
            if fusion_effects:
                from .effect_interpreter import EffectInterpretationContext

                fusion_context = EffectInterpretationContext(
                    source_uid=action.source_uid,
                    source_card=card,
                    player="me",
                    opponent="opponent",
                    generated_cards=generated_cards,
                    execution_context=execution_context,
                )
                for fusion_effect in fusion_effects:
                    fusion_result = self.effect_interpreter.apply_operations(after_core, fusion_effect.operations, fusion_context)
                    after_core = fusion_result.snapshot
        unresolved = ("fanfare",) if card.card_type.is_follower and self._template_has_trigger_event(card.base_card_id, "fanfare") else (("on_play_effects",) if card.card_type.is_amulet_like and self._template_has_trigger_event(card.base_card_id, "on_play") else ())
        return self._apply_template_effects(
            before=snapshot,
            after_core=after_core,
            action=action,
            events=(*result.events, *extra_events),
            default_certainty=PredictionCertainty.PARTIAL,
            default_description="local play follower core rules only",
            default_assumptions=("core_rules_only",),
            default_unresolved=unresolved,
            generated_cards=generated_cards,
            execution_context=execution_context,
        )

    def _play_enhanced(self, snapshot: BattleSnapshot, action: LegalAction, *, cost: int, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        card = snapshot.cards_by_uid.get(action.source_uid)
        # 条件求值需要「本次爆能强化的实际费用」来区分同一张卡的多个并列
        # 【爆能强化_N】段落（10624110 的 _7 / _8 各召唤一个随从）：
        # `action.cost` 可能为空（由模板推断补上），这里把推断结果写回动作。
        action = replace(action, play_kind=PlayKind.ENHANCE, cost=cost)
        result = self.kernel.play_enhanced_from_hand(snapshot, "me", action.source_uid, cost)
        after_core = self._sync_played_field_card_template_stats(snapshot, result.snapshot, action.source_uid, card) if card else result.snapshot
        return self._apply_template_effects(
            before=snapshot,
            after_core=after_core,
            action=action,
            events=result.events,
            default_certainty=PredictionCertainty.PARTIAL,
            default_description="local enhance core rules only",
            default_assumptions=("enhance_cost_paid",),
            default_unresolved=("enhance_effects",),
            generated_cards=generated_cards,
            execution_context=execution_context,
        )

    def _play_accelerated(self, snapshot: BattleSnapshot, action: LegalAction, *, cost: int, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        # 找 accelerate 法术卡的 base_card_id（从本体 template 的 related_cards 中查找）
        source_card = snapshot.cards_by_uid.get(action.source_uid)
        accelerate_card_id: int | None = None
        if source_card is not None:
            tmpl = self.templates_by_base_card_id.get(source_card.base_card_id)
            if tmpl is not None and tmpl.related_cards:
                for rc in tmpl.related_cards:
                    rc_tmpl = self.templates_by_base_card_id.get(rc.base_card_id)
                    if rc_tmpl is not None and rc_tmpl.identity.type_name == "spell":
                        accelerate_card_id = rc.base_card_id
                        break
        result = self.kernel.play_accelerated_from_hand(snapshot, "me", action.source_uid, cost, recorded_card_id=accelerate_card_id)
        return self._apply_template_effects(
            before=snapshot,
            after_core=result.snapshot,
            action=action,
            events=result.events,
            default_certainty=PredictionCertainty.SPEC_ONLY,
            default_description="local accelerate core rules only",
            default_assumptions=("accelerate_as_spell",),
            default_unresolved=("accelerate_effects",),
            generated_cards=generated_cards,
            execution_context=execution_context,
            override_base_card_id=accelerate_card_id,
        )

    def _play_crystallized(self, snapshot: BattleSnapshot, action: LegalAction, *, cost: int, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        # 结晶打出的语义与激奏一致：打出的卡本身就是关联的结晶护符（如 10962122），
        # 花结晶费用直接进场，并非本体随从进场后变身。这里从本体 template 的
        # related_cards 找到结晶护符 base_card_id，并组装其形态字段让进场实例真正
        # 变为结晶护符（base_card_id/card_id/name/countdown），使留场后的谢幕曲/
        # 吟唱倒计数/启动效果按结晶护符模板结算。
        source_card = snapshot.cards_by_uid.get(action.source_uid)
        crystal_card_id: int | None = None
        crystal_form_fields: dict[str, object] = {}
        if source_card is not None:
            tmpl = self.templates_by_base_card_id.get(source_card.base_card_id)
            if tmpl is not None and tmpl.related_cards:
                for rc in tmpl.related_cards:
                    rc_tmpl = self.templates_by_base_card_id.get(rc.base_card_id)
                    if rc_tmpl is not None and rc_tmpl.identity.type_name in {"countdown_amulet", "amulet"}:
                        crystal_card_id = rc.base_card_id
                        crystal_form_fields = self._crystal_form_fields(rc_tmpl)
                        break
        # replay 里协议会给结晶出的护符分配新的实例 uid：取录制帧里同一张
        # 结晶护符卡作为进场实例，否则本地沿用原手牌 uid，after 快照对不上
        # （replay_...1129510 st5：手牌 uid=60 -> 场上 uid=85 base=10962122）。
        recorded_card: CardView | None = None
        if execution_context.replay_mode and crystal_card_id is not None:
            for generated in generated_cards:
                if generated.base_card_id != crystal_card_id:
                    continue
                if generated.supplement_info.get("replay_generated_zone") != "field":
                    continue
                recorded_card = generated
                break
        result = self.kernel.play_crystallized_from_hand(
            snapshot, "me", action.source_uid, cost,
            recorded_card_id=crystal_card_id,
            crystal_form_fields=crystal_form_fields or None,
            recorded_card=recorded_card,
        )
        return self._apply_template_effects(
            before=snapshot,
            after_core=result.snapshot,
            action=action,
            events=result.events,
            default_certainty=PredictionCertainty.SPEC_ONLY,
            default_description="local crystal core rules only",
            default_assumptions=("crystal_as_amulet",),
            default_unresolved=("crystal_effects",),
            generated_cards=generated_cards,
            execution_context=execution_context,
            override_base_card_id=crystal_card_id,
        )

    def _crystal_form_fields(self, crystal_template: UnifiedCardTemplate) -> dict[str, object]:
        """从结晶护符 template 组装进场实例需要变身的字段。"""
        identity = crystal_template.identity
        fields: dict[str, object] = {
            "base_card_id": identity.base_card_id,
            "name": identity.name,
        }
        normal_card_id = getattr(identity, "normal_card_id", None) or getattr(identity, "base_card_id", None)
        if normal_card_id:
            fields["card_id"] = normal_card_id
        printed = crystal_template.printed
        if printed is not None and printed.countdown is not None:
            fields["countdown"] = int(printed.countdown)
        # 谢幕曲标志: 结晶护符常带 last_words(如 10962122 谢幕曲召唤本体)。
        # 与 local_match_setup 口径一致: static_keyword 或 executable last_words effect。
        has_last_words = any(kw.keyword == "last_words" for kw in crystal_template.static_keywords) or any(
            e.trigger.event == "last_words" for e in crystal_template.executable_effects
        )
        if has_last_words:
            fields["has_last_words"] = True
        return fields

    def _apply_hand_spellboost_effects(self, before: BattleSnapshot, after: BattleSnapshot, *, generated_cards: tuple[CardView, ...] = ()) -> BattleSnapshot:
        current = after
        generated_offset = 0
        for uid in self._spellboosted_hand_uids(before, current):
            card = current.cards_by_uid.get(uid)
            if card is None:
                continue
            template = self.templates_by_base_card_id.get(card.base_card_id)
            if template is None:
                continue
            for effect in template.executable_effects:
                if effect.trigger.event != "spellboost" or effect.trigger.source_zone != "hand":
                    continue
                action = LegalAction(kind=ActionKind.SPECIAL_ACTION, source_uid=uid)
                operations = tuple(operation for operation in effect.operations if operation.op not in {"modify_source_variable", "change_cost"})
                if not operations:
                    continue
                replay_generated = generated_cards[generated_offset:] if generated_cards else ()
                generated = self._generated_cards_for_effect(current, action, effect, generated_cards=replay_generated, source_card=card)
                if generated.unresolved_reason is not None:
                    continue
                generated_offset += len(generated.cards)
                context = EffectInterpretationContext(
                    source_uid=uid,
                    source_card=card,
                    generated_cards=generated.cards,
                )
                current = self.effect_interpreter.apply_operations(current, operations, context).snapshot
        return current

    def _spellboosted_hand_uids(self, before: BattleSnapshot, after: BattleSnapshot) -> tuple[int, ...]:
        boosted = []
        for uid in after.me.hand:
            before_card = before.cards_by_uid.get(uid)
            after_card = after.cards_by_uid.get(uid)
            if before_card is None or after_card is None:
                continue
            if after_card.spell_boost_count > before_card.spell_boost_count:
                boosted.append(uid)
        return tuple(boosted)

    def _sync_spellboost_hand_costs(self, before: BattleSnapshot, after: BattleSnapshot) -> BattleSnapshot:
        transactions = []
        for uid in after.me.hand:
            before_card = before.cards_by_uid.get(uid)
            after_card = after.cards_by_uid.get(uid)
            if before_card is None or after_card is None:
                continue
            if not self._template_uses_count_only_spellboost(before_card.base_card_id):
                continue
            if after_card.cost != before_card.cost:
                transactions.append(SetCardField(uid, "cost", before_card.cost))
            variable_x = after_card.variable_x or after_card.spell_boost_count
            if after_card.variable_x != variable_x:
                transactions.append(SetCardField(uid, "variable_x", variable_x))
            if after_card.supplement_info.get("variable_x") != variable_x:
                transactions.append(SetCardField(uid, "supplement_info", {**after_card.supplement_info, "variable_x": variable_x}))
        if not transactions:
            return after
        return self.kernel.apply_transactions(after, tuple(transactions)).snapshot

    def _template_uses_count_only_spellboost(self, base_card_id: int) -> bool:
        template = self.templates_by_base_card_id.get(base_card_id)
        if template is None:
            return False
        has_count_only = False
        for effect in template.executable_effects:
            if effect.trigger.event != "spellboost":
                continue
            if any(operation.op == "change_cost" and operation.target_ref == "source_card" for operation in effect.operations):
                return False
            if any(operation.op in {"modify_source_variable", "modify_card_counter"} for operation in effect.operations):
                has_count_only = True
        return has_count_only

    def _apply_last_words_effects(self, before: BattleSnapshot, after: BattleSnapshot, action: LegalAction, *, events: tuple[ExecutionEvent, ...] = (), generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext(), reserved_generated_uids: frozenset[int] = frozenset(), already_dispatched_uids: frozenset[int] = frozenset(), step_used_generated_uids: set[int] | None = None) -> tuple[BattleSnapshot, list[str], LocalExecutionContext, tuple[ExecutionEvent, ...]]:
        step_used = step_used_generated_uids if step_used_generated_uids is not None else set()
        current = after
        consumed: list[str] = []
        emitted_events: list[ExecutionEvent] = []
        last_words_uids = {event.source_uid for event in events if event.name == "last_words" and event.source_uid is not None}
        # 「破坏」类事件：伤害(damage)、伤害致死(destroyed_by_damage)、属性归零直接破坏(destroy)。
        # 只有被这些事件指向的 uid 才触发谢幕曲；消失(banish)/弹回(bounce)/变身(transform)
        # 不属于破坏，不能因为"离场"就触发。
        destroying_uids = {
            event.target_uid
            for event in events
            if event.name in {"damage", "destroy", "destroyed_by_damage"} and event.target_uid is not None
        }
        if not last_words_uids and not destroying_uids:
            return current, consumed, execution_context, ()
        before_field = set(before.me.field) | set(before.opponent.field)
        after_field = set(after.me.field) | set(after.opponent.field)
        # 「失去所有能力」之后再被破坏的随从不再拥有谢幕曲：必须按事件时序判定，
        # 不能回看 step 前的 CardView（那时能力还在）。10474120 的入场曲是
        # 「使 2 个敌方随从失去所有能力，再造成 9 点伤害」，本地曾因此在对手
        # 手牌 9 张时多执行一次谢幕曲抽牌并判成爆牌（…37784992 st132）。
        ability_removed: set[int] = set()
        last_words_suppressed: set[int] = set()
        for event in events:
            uid = event.target_uid
            if uid is None:
                continue
            if event.name == "ability_removed":
                ability_removed.add(uid)
            elif event.name in {"damage", "destroy", "destroyed_by_damage"} and uid in ability_removed:
                last_words_suppressed.add(uid)
        # 已由 EventQueue 消费过 LAST_WORDS 的卡不能再执行一次：事件队列是唯一
        # 消费点，本函数只补处理「离场但没有对应事件」的卡。
        leaving_uids = ((before_field - after_field) & (last_words_uids | destroying_uids)) - set(already_dispatched_uids) - last_words_suppressed
        # 谢幕曲结算顺序必须跟「死亡顺序」一致，不能用 set 的迭代顺序。
        # 同一 op 里多张卡一起被破坏时（`damage random_enemy_followers repeat=2`），
        # 召唤物是先死后进场的：谁先死，谁的谢幕曲召唤物先占空位。
        # set 对小整数的迭代顺序接近 uid 升序，会把后死的排到前面
        # （replay_...1294768 st32：怨灵/白骨士兵的实际槽位是 (86, 87)，本地 (87, 86)）。
        ordered_leaving: list[int] = []
        for event in events:
            uid = event.target_uid
            if uid is not None and uid in leaving_uids and uid not in ordered_leaving:
                ordered_leaving.append(uid)
        ordered_leaving.extend(sorted(leaving_uids - set(ordered_leaving)))
        generated_offset = 0
        reserved_generated_uids: set[int] = set(reserved_generated_uids)
        for uid in ordered_leaving:
            source_card = before.cards_by_uid.get(uid)
            if source_card is None:
                continue
            # 只有卡实例标记了 has_last_words 或被事件队列显式触发时才执行
            if not source_card.has_last_words and uid not in last_words_uids:
                continue
            # 动态赋予的「谢幕曲：召唤1个与本随从同名的随从」（纹章 11004112）。
            # 这类效果不属于卡模板的 last_words，必须在这里单独执行；事件队列路径
            # （handler.handle 的 last_words 分支）与这条补处理路径共用同一实现。
            if source_card.supplement_info.get("last_words_summon_copy"):
                owner_side = "me" if uid in before.me.field else "opponent"
                current = summon_same_name_copy(
                    current,
                    player=owner_side,
                    base_card_id=source_card.base_card_id,
                    execution_context=execution_context,
                    unresolved=self._effect_unresolved_reasons,
                )
            template = self.templates_by_base_card_id.get(source_card.base_card_id)
            if template is None:
                continue
            if template is None:
                continue
            for effect in template.executable_effects:
                if effect.trigger.event != "last_words":
                    continue
                owner = "me" if uid in before.me.field else "opponent"
                replay_generated_cards = generated_cards
                if owner == "opponent" and self._effect_generated_cards_are_draw_only(effect):
                    replay_generated_cards = ()
                # 用 `current`（逐步更新的盘面）而不是 `before` 判断「这张生成卡是否
                # 已被本 step 的效果用过」：同一 step 里多张卡各召唤 1 张同种 token 时，
                # `before` 一直不含前一次召唤出来的那张，于是第二次又拿到同一个 uid，
                # `summon` 因「uid 已存在」静默跳过，表现为少召唤一个
                # （replay_...1246399 st55：10954110 的入场曲把对手 97/98 一起打死，
                # 两张 10751120 的谢幕曲各召唤 1 个白骨士兵，第二次又拿到同一张 107）。
                effect_generated = self._generated_cards_for_effect(current, action, effect, generated_cards=replay_generated_cards, source_card=source_card, offset=generated_offset, execution_context=execution_context, reserved_uids=frozenset(reserved_generated_uids), player=owner)
                generated_offset += len(effect_generated.cards)
                reserved_generated_uids.update(card.uid for card in effect_generated.cards)
                if effect_generated.unresolved_reason is not None:
                    continue
                context = EffectInterpretationContext(
                    source_uid=uid,
                    source_card=source_card,
                    player=owner,
                    opponent="opponent" if owner == "me" else "me",
                    generated_cards=effect_generated.cards,
                    draw_bindings=execution_context.draw_bindings,
                    execution_context=execution_context,
                    step_generated_uids=frozenset(step_used),
                )
                try:
                    result = self.effect_interpreter.apply_operations(current, effect.operations, context)
                except EffectInterpreterError as exc:
                    if "insufficient resource" not in str(exc):
                        raise
                    # 谢幕曲资源不足（如墓地不够的【死灵术】）只让**这一条**不发动，
                    # 不能让整步回滚：否则同一步已结算的入场曲/其它效果都会丢失
                    # （…2230352 st20：10763210 的 4 点入场曲伤害被同侧另一张卡的
                    # 死灵术谢幕曲带崩，整步判成 insufficient resource）。
                    continue
                for event in result.events:
                    if event.name not in {"summon", "enter_field", "draw", "copy_card"}:
                        continue
                    # summon / enter_field 把被召唤卡放在 source_uid（target_uid 为 None）。
                    used_uid = event.target_uid if event.target_uid is not None else event.source_uid
                    if used_uid is not None:
                        step_used.add(used_uid)
                drawn_count = sum(1 for event in result.events if event.name == "draw")
                if drawn_count:
                    execution_context = execution_context.drop_draw_bindings(drawn_count)
                current = self._sync_entered_field_card_template_stats(current, result.snapshot, result.events)
                consumed.append(effect.effect_id)
                emitted_events.extend(result.events)
        return current, consumed, execution_context, tuple(emitted_events)

    def _effect_generated_cards_are_draw_only(self, effect: ExecutableEffect) -> bool:
        return any(operation.op == "draw" for operation in effect.operations) and not any(operation.op != "draw" and operation_generated_card_count(operation) > 0 for operation in effect.operations)

    def _sync_played_field_card_template_stats(self, before: BattleSnapshot, after: BattleSnapshot, uid: int, source_card: CardView, *, require_played_from_hand: bool = True, skip_last_words_uids: frozenset[int] = frozenset()) -> BattleSnapshot:
        if not source_card.card_type.is_field_card:
            return after
        if uid not in after.me.field and uid not in after.opponent.field:
            return after
        if require_played_from_hand and uid not in before.me.hand:
            return after
        template = self.templates_by_base_card_id.get(source_card.base_card_id)
        if template is None:
            return after
        transactions = []
        card = after.cards_by_uid.get(uid)
        if card is None:
            return after
        # 手牌卡的 max_life 在协议上恒为 0，入场后的生命上限必须取「印刷生命 + buff」。
        # 若退回印刷生命值，牌组/手牌 buff 的随从入场后会丢掉生命上限增量
        # （replay 中「牌组中所有随从+1/+1」类效果即表现为该随从 max_life 差）。
        buff_life_diff = int(getattr(source_card.buff, "max_life_diff", 0) or 0) if source_card.buff is not None else 0
        entered_life = card.life if card.life else template.printed.life + buff_life_diff
        fills_life = card.life == 0 and bool(template.printed.life)
        fills_max_life = card.max_life == 0 and bool(template.printed.life)
        if fills_life:
            transactions.append(SetCardField(uid, "life", entered_life))
        if fills_max_life:
            transactions.append(SetCardField(uid, "max_life", entered_life))
        is_played_field_card = uid in after.me.field or uid in after.opponent.field
        if card.card_type.is_follower and (fills_life or fills_max_life) and card.attack_limit == 0:
            transactions.append(SetCardField(uid, "attack_limit", 1))
        if card.card_type.is_amulet_like and is_played_field_card and card.attack_limit == 0:
            transactions.append(SetCardField(uid, "attack_limit", 1))
        if is_played_field_card and card.card_type.is_amulet_like and card.countdown == 0 and template.printed.countdown:
            transactions.append(SetCardField(uid, "countdown", template.printed.countdown))
        # attack_limit / attacks_left 由多个关键字共同决定（storm/rush 保证至少 1；
        # attack_limit_N 指定 N）。统一计算目标最大值，避免 static_keywords 遍历顺序
        # 导致后者覆盖前者；映射与沙盒构造共用 state_ops.static_keyword_card_updates。
        # 召唤/生成卡的 tribes 在 replay 或效果物化时可能为空；从模板补齐，
        # 否则「自己的创造物·随从进入战场时」（tribe 14）等事件条件永远不成立。
        if is_played_field_card and not card.tribes and template.identity.tribes:
            transactions.append(SetCardField(uid, "tribes", tuple(template.identity.tribes)))
        for field_name, value in static_keyword_card_updates(card, template).items():
            transactions.append(SetCardField(uid, field_name, value))
        # 召唤失调：无【疾驰】/【突进】的随从进入战场当回合不能攻击，
        # 下一次回合开始时由 CoreRuleKernel.start_turn 恢复。
        # 有疾驰/突进的随从已由 static_keyword_card_updates 授予攻击窗口。
        template_keywords = {
            str(getattr(keyword, "keyword", "") or "")
            for keyword in getattr(template, "static_keywords", ()) or ()
        }
        # buff 赋予的突进/疾驰（协议 buff.rush / buff.quick）不体现在模板静态关键词里，
        # 但等价于入场即拥有该关键词：必须同时授予攻击窗口，否则会被下面的
        # 「召唤失调」分支清零。replay 中 10722310 给手牌铁甲骑士突进后打出即能攻击，
        # 协议帧把它放进 can_attack_field_cards，本地却判 attacks_left=0。
        buff_grants_attack = (
            require_played_from_hand
            and is_played_field_card
            and card.card_type.is_follower
            and (card.has_rush or card.has_storm)
        )
        if buff_grants_attack:
            if card.attack_limit == 0:
                transactions.append(SetCardField(uid, "attack_limit", 1))
            if card.attacks_left <= 0:
                transactions.append(SetCardField(uid, "attacks_left", 1))
            transactions.append(SetCardField(uid, "can_attack", True))
            if card.has_storm:
                transactions.append(SetCardField(uid, "can_attack_leader", True))
        # 召唤失调：无【疾驰】/【突进】的随从「从手牌打出」当回合不能攻击，
        # 下一次回合开始时由 CoreRuleKernel.start_turn 恢复。
        # 只对从手牌打出的卡生效：效果召唤的 token 常由效果赋予突进/疾驰，
        # 其模板 static_keywords 未必完整，不能据此清零。
        if (
            require_played_from_hand
            and is_played_field_card
            and card.card_type.is_follower
            and not buff_grants_attack
            and not ({"storm", "rush"} & template_keywords)
        ):
            if card.attacks_left or card.can_attack or card.can_attack_leader:
                transactions.extend((
                    SetCardField(uid, "attacks_left", 0),
                    SetCardField(uid, "can_attack", False),
                    SetCardField(uid, "can_attack_leader", False),
                ))
        if fills_life or fills_max_life:
            if template.printed.cost is not None and card.cost != template.printed.cost:
                transactions.append(SetCardField(uid, "cost", template.printed.cost))
            if card.spell_boost_count:
                transactions.append(SetCardField(uid, "spell_boost_count", 0))
        if not card.has_last_words and uid not in skip_last_words_uids and any(effect.trigger.event == "last_words" for effect in template.executable_effects):
            transactions.append(SetCardField(uid, "has_last_words", True))
        if not transactions:
            return after
        return self.kernel.apply_transactions(after, tuple(transactions)).snapshot

    def _sync_entered_field_card_template_stats(self, before: BattleSnapshot, after: BattleSnapshot, events: tuple[ExecutionEvent, ...] = ()) -> BattleSnapshot:
        current = after
        before_field = set(before.me.field) | set(before.opponent.field)
        after_field = tuple(uid for uid in (*after.me.field, *after.opponent.field) if uid not in before_field)
        # 收集本次效果中被显式移除 last_words 的 uid（如幽冥中尉"使召唤物失去谢幕曲"）,
        # sync 补模板静态能力时不得覆盖此显式移除。
        removed_last_words_uids = frozenset(
            event.target_uid for event in events
            if event.name == "keyword_removed"
            and event.payload.get("keyword") == "last_words"
            and event.target_uid is not None
        )
        for uid in after_field:
            card = current.cards_by_uid.get(uid)
            if card is None:
                continue
            current = self._sync_played_field_card_template_stats(before, current, uid, card, require_played_from_hand=False, skip_last_words_uids=removed_last_words_uids)
        return current

    def _apply_pre_attack_effects(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        *,
        execution_context: LocalExecutionContext,
        defender_uid: int | None = None,
    ) -> tuple[BattleSnapshot, tuple[str, ...]]:
        """在伤害结算前应用「攻击时」效果。

        真实规则：「攻击时」效果先结算，再计算这次攻击的伤害。因此**全部** on_attack
        效果都必须在这里执行，不能只处理「仅修改攻击者自身」的那一类。
        攻击者可能在这次战斗里被反击致死（例：1/1 的 10851120 攻击 2/2），
        交给伤害之后的事件队列就再也执行不了了——10851120「对所有主战者造成1点伤害」
        就这样一直没结算（7 条 replay 都是双方 leader 各差 1 点）。

        这里同时收集攻击者自身模板与同侧纹章/场上持续效果的「攻击时」效果
        （例：纹章 10634112「自己的『天晶魔手』攻击时，使其+1/+0」）。
        预结算过的 effect_id 通过 handler 的 skip 集合跳过，只作用于本次 action
        的 handler 实例，不影响后续事件对同一效果的独立触发。
        """
        card = snapshot.cards_by_uid.get(action.source_uid)
        if card is None or not self.templates_by_base_card_id:
            return snapshot, ()
        effects: list[ExecutableEffect] = []
        template = self.templates_by_base_card_id.get(card.base_card_id)
        if template is not None:
            effects.extend(template.executable_effects)
        attacker_side = (
            "me" if action.source_uid in snapshot.me.field
            else "opponent" if action.source_uid in snapshot.opponent.field
            else None
        )
        # 本次攻击事件既是 registry 订阅的匹配依据，也是操作里 `event_target`
        # 的解析来源（「对交战对手造成 N 点伤害」）。不传进 context 时
        # `_target_uids("event_target")` 拿不到目标，op 直接跳过，攻击时伤害
        # 永远不会结算（10823110 冲田总司：录像里进化后 3 次 3 伤打死 8/10
        # 的对手随从，本地一点伤害都没有）。
        attack_event = ExecutionEvent(
            kind=EventKind.ATTACK_FOLLOWER if defender_uid is not None else EventKind.ATTACK_LEADER,
            source_uid=action.source_uid,
            target_uid=defender_uid,
            player=attacker_side or "me",
        )
        if attacker_side is not None:
            registry = PersistentEffectRegistry.from_snapshot(snapshot, self.templates_by_base_card_id)
            for sub in registry.get_subscribers(attack_event, snapshot):
                # event_target 条件（如「攻击随从时」）必须按本次攻击的真实目标判定，
                # 否则打脸时也会误触发只应作用于攻击随从的纹章效果。
                if not event_target_conditions_match(snapshot, attack_event, sub.effect, self._effect_unresolved_reasons):
                    continue
                effects.append(sub.effect)
        current = snapshot
        consumed: list[str] = []
        for effect in effects:
            if effect.trigger.event not in {"on_attack", "attack"}:
                continue
            if not effect.operations:
                continue
            if not self._effect_play_modes_match_action(effect, action):
                continue
            if not self._evaluate_effect_conditions(
                current, effect, action, source_card=card, event_source_card=card
            ).matched:
                continue
            context = EffectInterpretationContext(
                source_uid=action.source_uid,
                source_card=card,
                player=attacker_side or "me",
                opponent="opponent" if (attacker_side or "me") == "me" else "me",
                execution_context=execution_context,
                trigger_event=effect.trigger.event,
                execution_context_events=(attack_event,),
            )
            try:
                result = self.effect_interpreter.apply_operations(current, effect.operations, context)
            except EffectInterpreterError:
                continue
            current = result.snapshot
            consumed.append(effect.effect_id)
        return current, tuple(consumed)

    def _attack_leader(self, snapshot: BattleSnapshot, action: LegalAction, *, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        pre_snapshot, pre_consumed = self._apply_pre_attack_effects(
            snapshot, action, execution_context=execution_context
        )
        result = self.kernel.attack_leader(pre_snapshot, action.source_uid, "opponent")
        unresolved = self._attack_leader_unresolved_candidates(pre_snapshot, action)
        assumptions = ("no_attack_effect_candidates",) if not unresolved else ("attack_effect_candidates_present",)
        return self._apply_event_queue_effects(
            after_core=result.snapshot,
            action=action,
            events=result.events,
            default_certainty=PredictionCertainty.SPEC_ONLY if not unresolved else PredictionCertainty.PARTIAL,
            default_description="local attack leader core rules only",
            default_assumptions=assumptions,
            default_unresolved=unresolved,
            generated_cards=generated_cards,
            execution_context=execution_context,
            skip_effect_ids=frozenset(pre_consumed),
        )

    @staticmethod
    def _sanitize_zone_consistency(snapshot: BattleSnapshot) -> BattleSnapshot:
        """移除 zone 列表中 cards_by_uid 不存在的孤立 uid，防止下游 KeyError。"""
        valid = snapshot.cards_by_uid
        me_field = tuple(uid for uid in snapshot.me.field if uid in valid)
        me_hand = tuple(uid for uid in snapshot.me.hand if uid in valid)
        opp_field = tuple(uid for uid in snapshot.opponent.field if uid in valid)
        opp_hand = tuple(uid for uid in snapshot.opponent.hand if uid in valid)
        if (me_field == snapshot.me.field and me_hand == snapshot.me.hand
                and opp_field == snapshot.opponent.field and opp_hand == snapshot.opponent.hand):
            return snapshot
        return replace(
            snapshot,
            me=replace(snapshot.me, field=me_field, hand=me_hand),
            opponent=replace(snapshot.opponent, field=opp_field, hand=opp_hand),
        )

    def _attack_follower(self, snapshot: BattleSnapshot, action: LegalAction, *, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        if not action.target_groups or not action.target_groups[0]:
            return self._error(snapshot, "attack follower requires target", action)
        defender_uid = action.target_groups[0][0]
        # 「攻击时」效果先结算，再由增强后的攻击力计算战斗伤害。
        pre_snapshot, pre_consumed = self._apply_pre_attack_effects(
            snapshot, action, execution_context=execution_context, defender_uid=defender_uid
        )
        result = self.kernel.attack_follower(pre_snapshot, action.source_uid, defender_uid)

        unresolved = self._attack_follower_unresolved_candidates(pre_snapshot, action, defender_uid, result.events)
        assumptions = ("no_attack_effect_candidates",) if not unresolved else ("attack_effect_candidates_present",)
        default_certainty = PredictionCertainty.SPEC_ONLY if not unresolved else PredictionCertainty.PARTIAL

        return self._apply_event_queue_effects(
            after_core=result.snapshot,
            action=action,
            events=result.events,
            default_certainty=default_certainty,
            default_description="local attack follower core rules only",
            default_assumptions=assumptions,
            default_unresolved=unresolved,
            generated_cards=generated_cards,
            execution_context=execution_context,
            source_snapshot=snapshot,
            skip_effect_ids=frozenset(pre_consumed),
        )

    def _attack_leader_unresolved_candidates(self, snapshot: BattleSnapshot, action: LegalAction) -> tuple[str, ...]:
        unresolved: list[str] = []
        attacker = snapshot.cards_by_uid.get(action.source_uid)
        if attacker is not None:
            unresolved.extend(self._attack_trigger_candidates(attacker, "on_attack", uid=action.source_uid, snapshot=snapshot, action=action))
            unresolved.extend(self._damage_modifier_candidates(snapshot, attacker=attacker, defender=None, defending_player="opponent"))
        return tuple(dict.fromkeys(unresolved))

    def _attack_follower_unresolved_candidates(self, snapshot: BattleSnapshot, action: LegalAction, defender_uid: int, events: tuple[ExecutionEvent, ...]) -> tuple[str, ...]:
        unresolved: list[str] = []
        attacker = snapshot.cards_by_uid.get(action.source_uid)
        defender = snapshot.cards_by_uid.get(defender_uid)
        if attacker is not None:
            unresolved.extend(self._attack_trigger_candidates(attacker, "on_attack", uid=action.source_uid, snapshot=snapshot, action=action))
            unresolved.extend(self._attack_trigger_candidates(attacker, "on_clash", uid=action.source_uid, snapshot=snapshot, action=action))
        if defender is not None:
            unresolved.extend(self._attack_trigger_candidates(defender, "on_clash", uid=defender_uid, snapshot=snapshot, action=action))
        if attacker is not None or defender is not None:
            unresolved.extend(self._combat_keyword_candidates(attacker, defender))
            unresolved.extend(self._damage_modifier_candidates(snapshot, attacker=attacker, defender=defender, defending_player=None))
        unresolved.extend(self._last_words_unresolved_from_events(snapshot, events))
        return tuple(dict.fromkeys(unresolved))

    def _attack_trigger_candidates(self, card: CardView, event: str, *, uid: int, snapshot: BattleSnapshot | None = None, action: LegalAction | None = None) -> tuple[str, ...]:
        template = self.templates_by_base_card_id.get(card.base_card_id)
        if template is None:
            return ()
        aliases = {
            "on_attack": {"on_attack", "attack", "attack_leader", "attack_follower"},
            "on_clash": {"on_clash", "clash"},
        }.get(event, {event})
        for effect in template.executable_effects:
            if effect.trigger.event not in aliases:
                continue
            # 条件不满足时效果本来就不会发动，不应记为 unresolved 候选。
            if snapshot is not None and action is not None and not self._evaluate_effect_conditions(
                snapshot, effect, action, source_card=card
            ).matched:
                continue
            return (f"{event}:uid={uid}:card_id={card.card_id}:base_card_id={card.base_card_id}",)
        return ()

    def _combat_keyword_candidates(self, attacker: CardView | None, defender: CardView | None) -> tuple[str, ...]:
        """检测 rule_kernel 未建模的战斗关键词候选。

        rule_kernel.attack_follower 已建模：
        - bane (致命一击)
        - drain (吸血)
        - temp_shield (屏障)
        - damage_cut (4点以上伤害变为3点)

        因此只有 has_cant_destroy 和 has_reduce_damage 才需要标记为 unresolved。
        """
        unresolved: list[str] = []
        for role, card in (("attacker", attacker), ("defender", defender)):
            if card is None:
                continue
            keywords = []
            # has_cant_destroy 未被 rule_kernel 建模
            if card.has_cant_destroy:
                keywords.append("cant_destroy")
            # has_reduce_damage 未被 rule_kernel 建模（不同于 has_damage_cut）
            if card.has_reduce_damage:
                keywords.append("damage_reduction")
            if keywords:
                unresolved.append(f"keywords:{role}:uid={card.uid}:card_id={card.card_id}:base_card_id={card.base_card_id}:{','.join(keywords)}")
        return tuple(unresolved)

    def _damage_modifier_candidates(self, snapshot: BattleSnapshot, *, attacker: CardView | None, defender: CardView | None, defending_player: str | None) -> tuple[str, ...]:
        """检测 rule_kernel 未建模的 damage modifier 候选。

        rule_kernel 已建模：
        - 卡牌级: has_damage_cut, has_temp_shield (attack_leader + attack_follower)
        - 主战者级: player_buff.damage_cut, player_buff.temp_shield, player_buff.increase_damage
          (attack_leader)

        因此只有 has_reduce_damage（非标准减伤）和 damage_modifier crest 才需要标记为
        unresolved。
        """
        unresolved: list[str] = []
        # has_reduce_damage 未被 rule_kernel 建模，需要标记；has_damage_cut/has_temp_shield 已建模，跳过
        if attacker is not None and attacker.has_reduce_damage:
            unresolved.append(f"damage_modifiers:attacker:uid={attacker.uid}:card_id={attacker.card_id}:base_card_id={attacker.base_card_id}")
        if defender is not None and defender.has_reduce_damage:
            unresolved.append(f"damage_modifiers:defender:uid={defender.uid}:card_id={defender.card_id}:base_card_id={defender.base_card_id}")
        if self._has_damage_modifier_crest(snapshot):
            unresolved.append("damage_modifiers:crest_candidates_present")
        return tuple(unresolved)

    def _has_damage_modifier_crest(self, snapshot: BattleSnapshot) -> bool:
        for crest in (*snapshot.me.crests, *snapshot.me.extra_crests, *snapshot.opponent.crests, *snapshot.opponent.extra_crests):
            values = {**crest.supplement_info, **crest.buff}
            if any(_damage_modifier_key(key) for key in values):
                return True
        return False

    def _last_words_unresolved_from_events(self, snapshot: BattleSnapshot, events: tuple[ExecutionEvent, ...]) -> tuple[str, ...]:
        unresolved: list[str] = []
        for event in events:
            if event.name != "last_words" or event.source_uid is None:
                continue
            card = snapshot.cards_by_uid.get(event.source_uid)
            if card is None:
                unresolved.append("last_words:event_without_candidate_card")
                continue
            template = self.templates_by_base_card_id.get(card.base_card_id)
            has_template_last_words = template is not None and any(effect.trigger.event == "last_words" for effect in template.executable_effects)
            has_dynamic_copy = bool(card.supplement_info.get("last_words_summon_copy"))
            if card.has_last_words and not has_template_last_words and not has_dynamic_copy:
                # 有 last_words 但无 template 处理 → 确定无法消费
                unresolved.append(f"last_words:uid={card.uid}:card_id={card.card_id}:base_card_id={card.base_card_id}")
            elif has_template_last_words:
                # 模板存在 last_words：条件不满足（如【唤灵_2】墓地不足）是合法的
                # 「效果不发动」，不标记；条件满足才作为候选，让 default_certainty
                # 反映"可能尚未被事件队列消费"。
                effect = next(
                    effect for effect in template.executable_effects if effect.trigger.event == "last_words"
                )
                if effect.conditions and snapshot is not None:
                    condition_action = LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=event.source_uid or 0)
                    if not self._evaluate_effect_conditions(snapshot, effect, condition_action, source_card=card).matched:
                        continue
                unresolved.append(f"last_words:uid={card.uid}:card_id={card.card_id}:base_card_id={card.base_card_id}")
        return tuple(dict.fromkeys(unresolved))

    def _evolve(self, snapshot: BattleSnapshot, action: LegalAction, *, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        card = snapshot.cards_by_uid.get(action.source_uid)
        result = self.kernel.manual_evolve(snapshot, "me", action.source_uid)
        unresolved = ()
        if card is not None and self._template_has_trigger_event(card.base_card_id, "manual_evolve"):
            unresolved = ("manual_evolve_effects", "becomes_evolved_effects")
        # 无模板进化 trigger 时，进化只是 +2/+2 身材变化，确定性为 SPEC_ONLY
        default_certainty = PredictionCertainty.PARTIAL if unresolved else PredictionCertainty.SPEC_ONLY
        return self._apply_template_effects(
            before=snapshot,
            after_core=result.snapshot,
            action=action,
            events=result.events,
            default_certainty=default_certainty,
            default_description="local manual evolve core rules only",
            default_assumptions=("no_evolve_effects",),
            default_unresolved=unresolved,
            generated_cards=generated_cards,
            execution_context=execution_context,
        )

    def _super_evolve(self, snapshot: BattleSnapshot, action: LegalAction, *, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        card = snapshot.cards_by_uid.get(action.source_uid)
        result = self.kernel.manual_super_evolve(snapshot, "me", action.source_uid)
        unresolved = ()
        if card is not None and self._template_has_trigger_event(card.base_card_id, "manual_super_evolve"):
            unresolved = ("manual_evolve_effects", "manual_super_evolve_effects", "becomes_evolved_effects")
        # 无模板超进化 trigger 时，超进化只是 +3/+3 身材变化，确定性为 SPEC_ONLY
        default_certainty = PredictionCertainty.PARTIAL if unresolved else PredictionCertainty.SPEC_ONLY
        return self._apply_template_effects(
            before=snapshot,
            after_core=result.snapshot,
            action=action,
            events=result.events,
            default_certainty=default_certainty,
            default_description="local manual super evolve core rules only",
            default_assumptions=("no_super_evolve_effects",),
            default_unresolved=unresolved,
            generated_cards=generated_cards,
            execution_context=execution_context,
        )

    def _activate(self, snapshot: BattleSnapshot, action: LegalAction, *, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        activation_snapshot = snapshot
        card = snapshot.cards_by_uid.get(action.source_uid)
        if card is not None and not card.has_activation and self._template_has_trigger_event(card.base_card_id, "activation"):
            activation_snapshot = replace(snapshot, cards_by_uid={**snapshot.cards_by_uid, action.source_uid: replace(card, has_activation=True)})
        # PP 消耗由模板效果的 pay_pp operation 处理，不在此预扣。
        # （之前此处预扣 activation_cost 导致与模板 pay_pp 双重扣费）
        result = self.kernel.activate_card(activation_snapshot, "me", action.source_uid)
        return self._apply_template_effects(
            before=activation_snapshot,
            after_core=result.snapshot,
            action=action,
            events=result.events,
            default_certainty=PredictionCertainty.PARTIAL,
            default_description="local activation core rules only",
            default_assumptions=("no_activation_effects",),
            default_unresolved=("activation_effects",) if self._action_source_has_trigger_event(activation_snapshot, action, "activate") else (),
            generated_cards=generated_cards,
            execution_context=execution_context,
        )

    def _special_action(self, snapshot: BattleSnapshot, action: LegalAction, *, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        result = self.kernel.perform_special_action(snapshot, "me", action.source_uid)
        return self._apply_template_effects(
            before=snapshot,
            after_core=result.snapshot,
            action=action,
            events=result.events,
            default_certainty=PredictionCertainty.PARTIAL,
            default_description="local special action core rules only",
            default_assumptions=("no_special_action_effects",),
            default_unresolved=("special_action_effects",) if self._action_source_has_trigger_event(snapshot, action, "special_action") else (),
            generated_cards=generated_cards,
            execution_context=execution_context,
        )

    def _fusion(self, snapshot: BattleSnapshot, action: LegalAction, *, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        result = self.kernel.record_fusion(snapshot, "me", action.source_uid, action.material_uids)
        has_fusion_effect = self._action_source_has_trigger_event(snapshot, action, "fusion")
        # 融合本体只记录素材/已融合状态时，CoreRuleKernel 已完整建模，不能因模板
        # 中“未来才会读取融合状态”的超进化条件而降级为 PARTIAL。仅在当前 fusion
        # 事件确有待执行的模板效果时，才保留 strict gap 边界。
        return self._apply_template_effects(
            before=snapshot,
            after_core=result.snapshot,
            action=action,
            events=result.events,
            default_certainty=PredictionCertainty.PARTIAL if has_fusion_effect else PredictionCertainty.SPEC_ONLY,
            default_description="local fusion core rules only",
            default_assumptions=("fusion_materials_recorded",),
            default_unresolved=("fusion_effects",) if has_fusion_effect else (),
            generated_cards=generated_cards,
            execution_context=execution_context,
        )

    def _turn_start(self, snapshot: BattleSnapshot, action: LegalAction, *, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        drawn_card = generated_cards[0] if generated_cards else None
        result = self.kernel.start_turn(snapshot, "me", drawn_card, defer_draw=True)
        unresolved = self._turn_start_unresolved_candidates(snapshot)
        assumptions = ("no_turn_start_effect_candidates",) if not unresolved else ("turn_start_effect_candidates_present",)
        phase_result = self._apply_event_queue_effects(
            after_core=result.snapshot,
            action=action,
            events=result.events,
            source_snapshot=snapshot,
            default_certainty=PredictionCertainty.SPEC_ONLY if not unresolved else PredictionCertainty.PARTIAL,
            default_description="local turn start phase effects only",
            default_assumptions=assumptions,
            default_unresolved=unresolved,
            generated_cards=generated_cards,
            execution_context=execution_context,
        )
        draw_result = self.kernel.start_turn_draw(phase_result.snapshot, "me", drawn_card)
        return PredictionResult(
            snapshot=draw_result.snapshot,
            certainty=phase_result.certainty,
            description=phase_result.description,
            assumptions=phase_result.assumptions,
            unresolved_effects=phase_result.unresolved_effects,
        )

    def _turn_end(self, snapshot: BattleSnapshot, action: LegalAction, *, generated_cards: tuple[CardView, ...] = (), execution_context: LocalExecutionContext = LocalExecutionContext()) -> PredictionResult:
        # 结束回合的一方由 is_ally_turn 决定，不能硬编码成 me：
        # replay 里双方都会结束回合，若一律按 me 结算，「对手回合结束时」的
        # 纹章/持续效果会全部错侧（replay_...1484307 st51：回合结束方是 opponent，
        # 属于 opponent 的纹章 10954112 因此没有发动）。
        ending_player = "me" if snapshot.is_ally_turn else "opponent"
        result = self.kernel.end_turn(snapshot, ending_player, defer_removal_uids=self._turn_end_self_destroy_uids(snapshot, ending_player=ending_player, action=action))
        unresolved = self._turn_end_unresolved_candidates(snapshot, result.events, include_template_turn_end=not self.enable_event_queue_effects)
        assumptions = ("no_turn_end_effect_candidates",) if not unresolved else ("turn_end_effect_candidates_present",)
        turn_end_result = self._apply_event_queue_effects(
            after_core=result.snapshot,
            action=action,
            events=result.events,
            default_certainty=PredictionCertainty.SPEC_ONLY if not unresolved else PredictionCertainty.PARTIAL,
            default_description="local turn end core rules only",
            default_assumptions=assumptions,
            default_unresolved=unresolved,
            generated_cards=generated_cards,
            execution_context=execution_context,
            source_snapshot=snapshot,
        )
        return turn_end_result

    def _turn_start_unresolved_candidates(self, snapshot: BattleSnapshot) -> tuple[str, ...]:
        unresolved: list[str] = []
        for uid in (*snapshot.me.field, *snapshot.opponent.field):
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            template = self.templates_by_base_card_id.get(card.base_card_id)
            if template is not None and any(effect.trigger.event in {"turn_start", "on_turn_start"} for effect in template.executable_effects):
                unresolved.append(f"turn_start_effects:uid={uid}:card_id={card.card_id}:base_card_id={card.base_card_id}")
        return tuple(dict.fromkeys(unresolved))

    def _turn_end_self_destroy_uids(self, snapshot: BattleSnapshot, *, ending_player: str, action: LegalAction) -> frozenset[int]:
        """模板里带「回合结束时破坏本卡牌」且**本步条件成立**的 uid。

        这些卡的离场是**它自己的 turn_end 模板效果**，必须由效果队列按登记顺序结算，
        不能被 `kernel.end_turn` 依据 `is_remove_field_at_turn_change` 提前移除
        （10903210 混沌监狱：纹章 10964112 的条件要先看到它还在场上，
        …692618 st156 少 1 点伤害 + 1 点回复）。

        条件不成立的卡（如 90071120「**对手**的回合结束时破坏本卡牌」在我方回合结束）
        不能延迟：协议仍会在该回合结束把它带走（…1393093 st118）。
        """
        evaluator = self.condition_evaluator
        context = ConditionEvaluationContext(action=action, event_player=ending_player)
        uids: set[int] = set()
        for side in ("me", "opponent"):
            for uid in getattr(snapshot, side).field:
                card = snapshot.cards_by_uid.get(uid)
                if card is None:
                    continue
                template = self.templates_by_base_card_id.get(card.base_card_id)
                if template is None:
                    continue
                effects = list(template.executable_effects)
                for persistent in template.persistent_effects:
                    effects.extend(persistent.executable_effects or ())
                for effect in effects:
                    if effect.trigger is None or effect.trigger.event != "turn_end":
                        continue
                    if not any(
                        operation.op == "destroy" and (operation.target_ref or "") in {"source_card", "self"}
                        for operation in effect.operations
                    ):
                        continue
                    matched = evaluator.evaluate_all(snapshot, effect, effect.conditions or (), context)
                    if matched.matched and not matched.unresolved_reasons:
                        uids.add(uid)
                    break
        return frozenset(uids)

    def _turn_end_unresolved_candidates(self, snapshot: BattleSnapshot, events: tuple[ExecutionEvent, ...], *, include_template_turn_end: bool = True) -> tuple[str, ...]:
        unresolved: list[str] = []
        for uid in (*snapshot.me.field, *snapshot.opponent.field):
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            template = self.templates_by_base_card_id.get(card.base_card_id)
            if include_template_turn_end and template is not None and any(effect.trigger.event in {"turn_end", "on_turn_end"} for effect in template.executable_effects):
                unresolved.append(f"turn_end_effects:uid={uid}:card_id={card.card_id}:base_card_id={card.base_card_id}")
            if card.has_last_words and (card.is_remove_field_at_turn_change or any(event.name == "last_words" and event.source_uid == uid for event in events)):
                template_can_handle = template is not None and any(effect.trigger.event == "last_words" for effect in template.executable_effects)
                if not template_can_handle and not card.supplement_info.get("last_words_summon_copy"):
                    unresolved.append(f"last_words:uid={uid}:card_id={card.card_id}:base_card_id={card.base_card_id}")
            elif template is not None and any(effect.trigger.event == "last_words" for effect in template.executable_effects) and any(event.name == "last_words" and event.source_uid == uid for event in events):
                # template 有 last_words 且 event 已触发 → EventQueue 会消费，不标 unresolved
                pass
        if any(event.name == "last_words" for event in events) and not any(item.startswith("last_words:") for item in unresolved):
            # 动态赋予的「谢幕曲：召唤同名」已由 TemplateEventHandler 处理，不再报通用缺口。
            dynamic_handled = all(
                isinstance((event.payload or {}).get("leaving_card"), dict)
                and (event.payload["leaving_card"].get("supplement_info") or {}).get("last_words_summon_copy")
                for event in events
                if event.name == "last_words"
            )
            if not dynamic_handled:
                unresolved.append("last_words:event_without_candidate_card")
        return tuple(dict.fromkeys(unresolved))

    def _extra_pp(self, snapshot: BattleSnapshot, action: LegalAction) -> PredictionResult:
        result = self.kernel.activate_extra_pp(snapshot, "me")
        return PredictionResult(
            snapshot=result.snapshot,
            certainty=PredictionCertainty.SPEC_ONLY,
            description="local extra pp activation",
            assumptions=("extra_pp_activated",),
            unresolved_effects=(),
        )

    def _apply_event_queue_effects(
        self,
        *,
        after_core: BattleSnapshot,
        action: LegalAction,
        events: tuple[ExecutionEvent, ...],
        default_certainty: PredictionCertainty,
        default_description: str,
        default_assumptions: tuple[str, ...],
        default_unresolved: tuple[str, ...],
        generated_cards: tuple[CardView, ...] = (),
        execution_context: LocalExecutionContext = LocalExecutionContext(),
        source_snapshot: BattleSnapshot | None = None,
        skip_effect_ids: frozenset[str] = frozenset(),
        prior_consumed: tuple[str, ...] = (),
    ) -> PredictionResult:
        event_queue_effects_enabled = self.enable_event_queue_effects
        # 胜负已判定（任一方 result 非 0）后不再结算剩余效果：
        # 真实规则中生命归零即结束对局，谢幕曲/离场/入场等后续效果都不再发生。
        if after_core.me.result or after_core.opponent.result:
            # 后续效果确定不会发生 → 没有未解析部分；不能保留 PARTIAL（否则会出现
            # unresolved_effects=() 却标 partial 的自相矛盾结果）。
            certainty = (
                default_certainty
                if _certainty_rank(default_certainty) >= _certainty_rank(PredictionCertainty.SPEC_ONLY)
                else PredictionCertainty.SPEC_ONLY
            )
            return PredictionResult(
                snapshot=after_core,
                certainty=certainty,
                description=default_description,
                assumptions=(*default_assumptions, "battle_already_decided"),
                unresolved_effects=(),
            )
        if not self.templates_by_base_card_id or not event_queue_effects_enabled:
            return PredictionResult(
                snapshot=after_core,
                certainty=default_certainty,
                description=default_description,
                assumptions=default_assumptions,
                unresolved_effects=default_unresolved,
            )
        state = EventEffectState(consumed=[], unresolved=[])
        # 动作前已预结算的效果（「攻击时」自身强化）算作已消费：它们不会在事件队列
        # 里重复执行，但仍应出现在 consumed_event_effects 里以保持可追溯。
        if skip_effect_ids:
            state.consumed.extend(sorted(skip_effect_ids))
        # registry 以执行后的快照为准；但**本步离场**的卡仍要保留它自己的
        # 「离场时」订阅：效果执行可能在事件发出前就把卡移出战场（11023310 打出后
        # 先移除旧证据护符再发 BANISH，旧 11003210 的回复要结算）。
        event_source_snapshot = source_snapshot or after_core
        registry = PersistentEffectRegistry.from_snapshot(after_core, self.templates_by_base_card_id)
        if event_source_snapshot is not after_core:
            registry.register_departed(event_source_snapshot, after_core, self.templates_by_base_card_id)
        handler = TemplateEventHandler(
            templates_by_base_card_id=self.templates_by_base_card_id,
            effect_interpreter=self.effect_interpreter,
            action=action,
            generated_cards=generated_cards,
            state=state,
            execution_context=execution_context,
            source_snapshot=event_source_snapshot,
            effect_unresolved_reasons=self._effect_unresolved_reasons,
            registry=registry,
            skip_effect_ids=skip_effect_ids,
        )
        try:
            dispatch = EventQueue(events).dispatch(after_core, (handler,))
        except EffectInterpreterError as exc:
            if _INSUFFICIENT_RESOURCE_MARKER in str(exc):
                # 资源不足导致该效果不发动：整条效果已回滚，盘面就是 after_core，
                # 结果是确定的（效果未发动），不是预测缺口，因此记 SPEC_ONLY。
                return PredictionResult(
                    snapshot=after_core,
                    certainty=PredictionCertainty.SPEC_ONLY,
                    description="event queue template effect not activated (insufficient resource)",
                    assumptions=("template_effects_enabled", "event_queue_effects_enabled"),
                    unresolved_effects=(),
                )
            return PredictionResult(
                snapshot=after_core,
                certainty=PredictionCertainty.PREDICT_ERROR,
                description=f"event queue template effect execution failed: {exc}",
                assumptions=("template_effects_enabled", "event_queue_effects_enabled"),
                unresolved_effects=("event_queue_template_effect_execution_failed",),
            )
        assumptions = ["template_effects_enabled", "event_queue_effects_enabled"]
        # prior_consumed：调用方在进入本函数前已经结算掉的持续效果（如
        # `_apply_template_effects` 的「本卡入场」预结算段）。它们是本 step 真实发生的
        # 消费，必须一并体现在 assumptions 里，否则「效果执行了但没有消费记录」，
        # certainty 也会被判回 default，与 prediction 的结果自相矛盾。
        consumed_effects = tuple(dict.fromkeys((*prior_consumed, *state.consumed)))
        if consumed_effects:
            assumptions.append(f"consumed_event_effects:{','.join(consumed_effects)}")
        # default_unresolved（如 "fanfare"）只由**本段自己**的消费证据清除：
        # prior_consumed 属于入场窗口里的持续效果，不能证明本卡的入场曲已执行。
        unresolved = tuple(dict.fromkeys((*state.unresolved, *(() if (state.consumed or skip_effect_ids) else default_unresolved))))
        dispatch_snapshot = self._sync_entered_field_card_template_stats(after_core, dispatch.snapshot, dispatch.processed_events)
        # dispatch 过程中可能有卡被 damage 打死（如纹章 turn_end AOE）但没有通过
        # event queue 的 LAST_WORDS 事件处理谢幕曲。需要补充检查。
        # 排除已被 dispatch 通过 LAST_WORDS 事件处理过的 uid。
        after_last_words = dispatch_snapshot
        lw_consumed: list[str] = []
        dispatched_last_words_uids = {e.source_uid for e in dispatch.processed_events if e.name == "last_words" and e.source_uid is not None}
        # 检查是否有卡在 dispatch 中被 damage 打死但没通过 LAST_WORDS 事件处理
        before_field = set(after_core.me.field) | set(after_core.opponent.field)
        after_field = set(dispatch_snapshot.me.field) | set(dispatch_snapshot.opponent.field)
        unhandled_leaving = (before_field - after_field) - dispatched_last_words_uids
        if unhandled_leaving:
            # 只对未被 dispatch 处理的离场卡执行 last_words
            destruction_events = tuple(
                e for e in dispatch.processed_events
                if e.name in {"damage", "destroy", "destroyed_by_damage"}
            )
            after_last_words, lw_consumed_tuple, execution_context, lw_emitted = self._apply_last_words_effects(
                after_core, dispatch_snapshot, action,
                events=destruction_events,
                generated_cards=generated_cards,
                execution_context=execution_context,
                already_dispatched_uids=frozenset(dispatched_last_words_uids),
            )
            lw_consumed = list(lw_consumed_tuple)
            # 谢幕曲自身产生的事件（召唤 / enter_field 等）仍需入队，否则纹章与
            # 其它持续效果不会响应新入场对象（例：维尔伯特谢幕曲召唤的圣骑兵
            # 必须触发「守护随从入场 +1/+2」纹章）。
            followup_events = tuple(event for event in lw_emitted if event.name != "last_words")
            if followup_events:
                followup_registry = PersistentEffectRegistry.from_snapshot(after_last_words, self.templates_by_base_card_id)
                followup_handler = TemplateEventHandler(
                    templates_by_base_card_id=self.templates_by_base_card_id,
                    effect_interpreter=self.effect_interpreter,
                    action=action,
                    generated_cards=generated_cards,
                    state=state,
                    execution_context=execution_context,
                    source_snapshot=event_source_snapshot,
                    effect_unresolved_reasons=self._effect_unresolved_reasons,
                    registry=followup_registry,
                )
                followup_dispatch = EventQueue(followup_events).dispatch(after_last_words, (followup_handler,))
                after_last_words = followup_dispatch.snapshot
        if lw_consumed:
            state.consumed.extend(lw_consumed)
        return PredictionResult(
            snapshot=after_last_words,
            certainty=PredictionCertainty.SPEC_ONLY if consumed_effects else default_certainty,
            description="local core rules plus event queue template effects" if consumed_effects else default_description,
            assumptions=tuple(assumptions if consumed_effects or state.unresolved else default_assumptions),
            unresolved_effects=unresolved,
        )

    def _apply_template_effects(
        self,
        *,
        before: BattleSnapshot,
        after_core: BattleSnapshot,
        action: LegalAction,
        events: tuple[ExecutionEvent, ...],
        default_certainty: PredictionCertainty,
        default_description: str,
        default_assumptions: tuple[str, ...],
        default_unresolved: tuple[str, ...],
        generated_cards: tuple[CardView, ...] = (),
        execution_context: LocalExecutionContext = LocalExecutionContext(),
        override_base_card_id: int | None = None,
    ) -> PredictionResult:
        source_card = before.cards_by_uid.get(action.source_uid)
        if source_card is None:
            return self._error(before, "card not found", action)
        lookup_id = override_base_card_id if override_base_card_id is not None else source_card.base_card_id
        template = self.templates_by_base_card_id.get(lookup_id)
        if template is None:
            # 模板缺失时统一使用调用者提供的 default_* 参数返回。
            # 调用者已通过 _template_has_trigger_event 判断是否期望模板效果，
            # 并据此设置了 default_unresolved（如 "enhance_effects"）。
            return PredictionResult(
                snapshot=after_core,
                certainty=default_certainty,
                description=default_description,
                assumptions=default_assumptions,
                unresolved_effects=default_unresolved,
            )
        resolution = resolve_template_effects_for_action(template, action, events)
        unresolved = list(resolution.unresolved_reasons)
        # 本 step 的「已分配生成卡」载体：一个 step 会构造 entry / pre / main / followup
        # 多个 handler 实例，它们必须共享同一份记账，否则后执行者会重复领取前序 effect
        # 已经用掉的生成卡（10964120 evolve 链的两种召唤）。
        step_used_generated_uids: set[int] = set()
        # 「本卡入场」预结算段消费掉的持续效果 id（见下方 entry_dispatch）。
        entry_consumed: list[str] = []
        # 真实规则顺序：随从入场 → 「其他随从入场时」触发的持续效果 → 入场曲。
        # 原实现把 enter_field 留在 fanfare 之后的 EventQueue 里 dispatch，持续效果
        # 看到的是入场曲结算后的快照。典型失败：10834120 灾难言灵·洋荷 在场时打出
        # 10832110 快乐绽花·萨米&玛莉（入场曲抽 2 张），洋荷的「使所有手牌魔力增幅
        # 1 次」把本 step 才抽到的卡也 +1 了（replay_...766621 st101：
        # 新卡 sb 录像 1 / 本地 2）。
        # 这里先把本卡自身的 enter_field 结算掉，再从 events 里移除，避免后续重复。
        if self.enable_event_queue_effects:
            entry_events = tuple(event for event in events if event.name == "enter_field")
            if entry_events:
                entry_state = EventEffectState(consumed=[], unresolved=[])
                entry_registry = PersistentEffectRegistry.from_snapshot(after_core, self.templates_by_base_card_id)
                entry_handler = TemplateEventHandler(
                    templates_by_base_card_id=self.templates_by_base_card_id,
                    effect_interpreter=self.effect_interpreter,
                    action=action,
                    generated_cards=generated_cards,
                    state=entry_state,
                    execution_context=execution_context,
                    effect_unresolved_reasons=self._effect_unresolved_reasons,
                    registry=entry_registry,
                    used_generated_uids=step_used_generated_uids,
                )
                entry_dispatch = EventQueue(entry_events).dispatch(after_core, (entry_handler,))
                after_core = entry_dispatch.snapshot
                unresolved.extend(entry_state.unresolved)
                # 「本卡入场」窗口里被消费的持续效果也要进入本 step 的消费账本：
                # 它们已经改了盘面，若只丢在 entry_state 里，assumptions 会显示
                # 「什么都没发生」，certainty 也会退回 default。
                entry_consumed.extend(entry_state.consumed)
                events = tuple(event for event in events if event.name != "enter_field")
        executable = []
        target_group_index = 0
        generated_offset = 0
        reserved_generated_uids: set[int] = set()
        has_executable_summon = False
        last_selected_target_groups: tuple[tuple[int, ...], ...] = ()
        # "改为发动所有能力"预扫描: 若存在满足条件的 all_abilities spec
        # (ability_resolution_mode=all_abilities), 则无视玩家单选, 所有 mode option
        # 效果都发动 (铭刻之约: 场上有原始费用>=6卡时发动全部)。
        all_abilities_active = False
        for effect in resolution.matched_effects:
            if any(op.params.get("ability_resolution_mode") == "all_abilities" for op in effect.operations):
                if self._evaluate_effect_conditions(after_core, effect, action, source_card=source_card).matched:
                    all_abilities_active = True
                    break
        for effect in resolution.matched_effects:
            current_target_index = target_group_index
            if effect.target_selector is not None:
                target_group_index += 1
            if not self._effect_mode_option_matches_action(effect, action, all_abilities_active=all_abilities_active):
                continue
            if not self._effect_play_modes_match_action(effect, action):
                continue
            if self._requires_previous_summoned(effect) and not has_executable_summon and not self._summons_generated_cards(effect):
                # 同一效果内「召唤X，然后使X获得…」：last_summoned 引用的就是
                # 本效果刚召唤的卡，不能因为此前没有 summon 效果而跳过。
                continue  # 前提条件不满足（无 summon），效果自然不触发
            # 资源类条件（墓地/手牌数）在 fanfare 上必须按**打出前**求值：
            # 法术打出后会先进墓地，用 after_core 判「唤灵_6」会多算这张法术本身
            # （墓地 5 -> 6 就越过门槛，replay_...1359740 st40）。
            resource_snapshot = before if (effect.trigger is not None and effect.trigger.event == "fanfare") else None
            condition_result = self._evaluate_effect_conditions(after_core, effect, action, source_card=source_card, resource_snapshot=resource_snapshot)
            if not condition_result.matched:
                continue
            unresolved.extend(condition_result.unresolved_reasons)
            materialized_target_groups: tuple[tuple[int, ...], ...] = ()
            # evolve/super_evolve actions may omit an optional selected target;
            # materialize candidate groups through the common selector boundary.
            # Other action kinds retain their recorded-target contract below.
            if (
                effect.target_selector is not None
                and not action.target_groups
                and action.kind in (ActionKind.EVOLVE, ActionKind.SUPER_EVOLVE)
            ):
                materialized_targets = materialize_effect_target_groups(before, action, effect, source_card=source_card)
                if materialized_targets.unresolved_reason is not None:
                    unresolved.append(self._target_selector_unresolved_reason(before, action, effect, source_card=source_card))
                    continue
                if materialized_targets.target_groups == ((),):
                    continue
                materialized_target_groups = materialized_targets.target_groups
            effect_target_groups = materialized_target_groups or self._target_groups_for_effect(action, effect, current_target_index, before)
            if not effect_target_groups and any((operation.target_ref or "") == "selected_target" for operation in effect.operations):
                effect_target_groups = last_selected_target_groups
            effect_action = replace(action, target_groups=effect_target_groups)
            # "改为发动所有能力"激活时, effect_action 的 selected_indexes 也扩展为全部,
            # 使 generated_card_materializer 的 mode 门控物化所有 mode option 的生成卡
            # (绿伞会 combo3: option2 add_to_hand 森林的奥秘+妖精)。
            if all_abilities_active:
                effect_action = replace(effect_action, selected_indexes=tuple(range(1, 10)))
            requires_random_targets = self._requires_random_targets(before, effect, effect_action, execution_context=execution_context)
            has_selected_targets = self._has_concrete_target_groups(effect_action.target_groups)
            if not requires_random_targets and self._requires_recorded_targets(before, effect, effect_action, execution_context=execution_context) and not has_selected_targets:
                # optional selector + evolve/super_evolve 无目标：效果安全跳过（不标记 unresolved）
                if effect.target_selector and effect.target_selector.optional and action.kind in (ActionKind.EVOLVE, ActionKind.SUPER_EVOLVE):
                    continue
                # 目标池为空时效果不触发（noop），不标记 unresolved。
                # spec.target 为 null 但 operation 用 selected_enemy_follower 等
                # 隐式选择器时同样适用（materializer 会合成 implicit selector）。
                mat_check = materialize_effect_target_groups(before, effect_action, effect, source_card=source_card)
                if mat_check.unresolved_reason is None and mat_check.target_groups == ((),):
                    continue
                unresolved.append(self._target_selector_unresolved_reason(before, action, effect, source_card=source_card))
                continue
            effect_generated = self._generated_cards_for_effect(after_core, effect_action, effect, generated_cards=generated_cards, source_card=source_card, offset=generated_offset, execution_context=execution_context, reserved_uids=frozenset(reserved_generated_uids))
            generated_offset += len(effect_generated.cards)
            reserved_generated_uids.update(card.uid for card in effect_generated.cards)
            if effect_generated.unresolved_reason is not None:
                unresolved.append(effect_generated.unresolved_reason)
                continue
            # 需要物化但结果为空（如墓地为空、history 无匹配）→ 效果不生效，跳过
            if not effect_generated.cards and self._effect_requires_generated_cards(effect):
                continue
            effect_unresolved = self._effect_unresolved_reasons(before, effect, effect_action, effect_generated.cards, source_card=source_card, execution_context=execution_context)
            if effect_unresolved:
                unresolved.extend(effect_unresolved)
                continue
            if effect_target_groups == ((),) and any((operation.target_ref or "") in {"selected_target", "selected_filtered_cards"} for operation in effect.operations):
                continue
            executable.append((effect, effect_target_groups, effect_generated.cards, effect_generated.assumptions))
            if effect_target_groups:
                last_selected_target_groups = effect_target_groups
            if self._summons_generated_cards(effect):
                has_executable_summon = True
        if not executable:
            # known_noop: 模板存在但效果确认不触发（而非缺失/不确定）
            # - resolution.matched_effects 非空但全被条件过滤 → 效果已知不满足触发条件
            # - 模板无当前 action 对应的 trigger 且文本也不含相关描述 → 效果确认不存在
            # - 两者都要求 unresolved 为空（无未解决的效果需求）
            known_noop = (bool(resolution.matched_effects) or self._template_has_no_action_trigger(template, action)) and not unresolved
            if not known_noop:
                unresolved.extend(default_unresolved)
            event_queue_unresolved = tuple(dict.fromkeys(unresolved))
            return self._apply_event_queue_effects(
                after_core=after_core,
                action=action,
                events=events,
                default_certainty=PredictionCertainty.SPEC_ONLY if known_noop else PredictionCertainty.PARTIAL,
                default_description="local core rules plus inactive template effects" if known_noop else "template effects enabled but no executable effects matched",
                default_assumptions=("template_effects_enabled",),
                default_unresolved=tuple(dict.fromkeys(unresolved)),
                generated_cards=generated_cards,
                execution_context=execution_context,
                # 事件消费阶段需要 before 帧：ordered_abilities 的起点 cursor
                # 必须取「结算前」的值，after 帧的 current_run_in_order_count 已经
                # 推进到本 step 结束位置（10964120：BEFORE=0 / AFTER=3）。
                source_snapshot=before,
                prior_consumed=tuple(entry_consumed),
            )
        current = after_core
        consumed: list[str] = []
        emitted: list[ExecutionEvent] = []
        assumptions = ["template_effects_enabled"]
        if generated_cards:
            assumptions.append(f"generated_cards_materialized:{len(generated_cards)}")
        use_after_hand_for_next_draw = False
        effect_context = EffectInterpretationContext(source_uid=action.source_uid, source_card=before.cards_by_uid.get(action.source_uid))
        prior_effect_events: tuple[ExecutionEvent, ...] = ()
        reserved_effect_generated_uids = frozenset(card.uid for _, _, effect_generated_cards, _ in executable for card in effect_generated_cards)
        try:
            step_generated_uids: set[int] = set()
            for effect, effect_target_groups, effect_generated_cards, effect_assumptions in executable:
                assumptions.extend(effect_assumptions)
                draw_bindings = self._returned_draw_bindings(effect, execution_context, executable, consumed_count=len(consumed)) if use_after_hand_for_next_draw and any(operation.op == "draw" for operation in effect.operations) else execution_context.draw_bindings
                context_random_targets = tuple(uid for group in action.random_target_groups for uid in group)
                if not context_random_targets:
                    context_random_targets = self._deterministic_random_targets(current, effect, action, execution_context=execution_context)
                    if context_random_targets and execution_context.branch_seed is not None:
                        assumptions.append(f"random_target_sampled:seed={execution_context.branch_seed}:effect={effect.effect_id}")
                context = replace(
                    effect_context,
                    selected_targets=tuple(uid for group in effect_target_groups for uid in group),
                    random_targets=context_random_targets,
                    generated_cards=effect_generated_cards,
                    draw_bindings=draw_bindings,
                    execution_context=execution_context,
                    # "改为发动所有能力"激活时覆盖全部 option, 使 all_abilities gate
                    # (auto ability_resolution_mode) 通过、各 mode option 实际 op 都发动;
                    # 但普通 mode_choice_effect (selection_count 单选, 无 ability_
                    # resolution_mode) 是数量校验门控, 必须用玩家实际 selected_indexes,
                    # 否则 range(1,10) 触发 count mismatch。
                    selected_mode_options=(
                        tuple(action.selected_indexes)
                        if any(op.op == "mode_choice_effect" and op.params.get("ability_resolution_mode") != "all_abilities" for op in effect.operations)
                        else (tuple(range(1, 10)) if all_abilities_active else tuple(action.selected_indexes))
                    ),
                    execution_context_events=prior_effect_events,
                    step_generated_uids=frozenset(step_generated_uids),
                )
                result = self.effect_interpreter.apply_operations(current, effect.operations, context)
                drawn_count = sum(1 for event in result.events if event.name == "draw")
                if drawn_count:
                    execution_context = execution_context.drop_draw_bindings(drawn_count)
                synced_result_snapshot = self._sync_entered_field_card_template_stats(current, result.snapshot)
                after_effect = synced_result_snapshot if self.enable_event_queue_effects else self._apply_hand_spellboost_effects(current, synced_result_snapshot, generated_cards=generated_cards)
                after_last_words, last_words_consumed, execution_context, last_words_emitted = self._apply_last_words_effects(current, after_effect, action, events=(*context.execution_context_events, *result.events), generated_cards=generated_cards, execution_context=execution_context, reserved_generated_uids=reserved_effect_generated_uids, step_used_generated_uids=step_used_generated_uids)
                current = self._sync_spellboost_hand_costs(current, after_last_words)
                emitted.extend(result.events)
                emitted.extend(last_words_emitted)
                prior_effect_events = (*prior_effect_events, *result.events)
                effect_context = replace(
                    self._context_after_events(context, result.events),
                    allow_missing_selected_field_targets=bool(context.selected_targets),
                )
                step_generated_uids.update(card.uid for card in effect_generated_cards)
                consumed.append(effect.effect_id)
                consumed.extend(last_words_consumed)
                use_after_hand_for_next_draw = any(operation.op == "return_to_deck" for operation in effect.operations)
            event_consumed: list[str] = []
            # earth_rite 必须进队列：「自己发动【土之秘术】时」这类手牌中发动的
            # 持续效果（10733310 饕餮魔咒费用-1）靠它触发。回合结束阶段的土之秘术
            # 由 kernel.end_turn 发出，若不列入白名单会被直接丢弃。
            # spellboost 也在白名单里：使用法术后的「使手牌发动 1 次魔力增幅」由
            # kernel 发出 SPELLBOOST 事件，【魔力增幅时】反应效果（10434110 的
            # +1/+1、10934120 的费用-1）订阅的是事件，之前被过滤掉就完全不触发
            # （replay_...992331 st26：sb 涨了、atk/life 没涨）。
            passive_event_names = {"activate", "becomes_super_evolved", "enter_field", "summon", "on_source_buffed", "damage", "turn_start", "play_card", "play_spell", "earth_rite", "spellboost"}
            queued_events = (
                *tuple(
                    event for event in emitted
                    if event.name != "last_words" or action.kind in {ActionKind.TURN_START, ActionKind.TURN_END}
                ),
                *tuple(
                    event for event in events
                    if (event.name in passive_event_names or (event.name == "last_words" and action.kind in {ActionKind.TURN_START, ActionKind.TURN_END}))
                    # 出牌卡自身的 enter_field 事件仍需进入 event queue：
                    # 场上其他持续效果（如 10724110「己方其他随从入场时赋予突进」）需要响应它。
                    # 「入场卡不应响应自己入场」由消费端（TemplateEventHandler / PersistentEffectRegistry）
                    # 按订阅者过滤，而不是在源头删除事件（否则会连累其他订阅者）。
                ),
            )
            queued_events = tuple(
                replace(
                    event,
                    payload={
                        **event.payload,
                        "source_card_type": before.cards_by_uid[event.source_uid].card_type.value,
                    },
                )
                if event.name == "activate" and event.source_uid in before.cards_by_uid
                else event
                for event in queued_events
            )
            if self.enable_event_queue_effects and queued_events:
                state = EventEffectState(consumed=[], unresolved=[])
                # 「使用卡牌瞬间」事件（play_spell / play_card）只应被**使用那一刻**
                # 已在场的卡响应。fanfare 结算后才重建 registry，会把本 step 刚召唤的
                # 卡也算进来：10923310 黄金时代打出后召唤的『令人战栗的海盗旗』
                # （【吟唱_7】+「自己使用法术时倒计时-1」）会对自己这次入场的那次使用
                # 触发，countdown 7->6（replay_...5315092 st27 等 5 条）。
                # 这里把这两类事件单独用 after_core（fanfare 前）的 registry 派发。
                use_events = tuple(
                    event for event in queued_events if event.name in {"play_spell", "play_card"}
                )
                rest_events = tuple(
                    event for event in queued_events if event.name not in {"play_spell", "play_card"}
                )
                if use_events:
                    pre_registry = PersistentEffectRegistry.from_snapshot(after_core, self.templates_by_base_card_id)
                    pre_handler = TemplateEventHandler(
                        templates_by_base_card_id=self.templates_by_base_card_id,
                        effect_interpreter=self.effect_interpreter,
                        action=action,
                        generated_cards=generated_cards,
                        state=state,
                        execution_context=execution_context,
                        effect_unresolved_reasons=self._effect_unresolved_reasons,
                        registry=pre_registry,
                        used_generated_uids=step_used_generated_uids,
                    )
                    pre_dispatch = EventQueue(use_events).dispatch(current, (pre_handler,))
                    current = pre_dispatch.snapshot
                # registry 以结算后的盘面为准，另外只为**本步离场**的卡补注册它自己的
                # 「离场时」触发（11023310 打出后旧证据护符在事件发出前就被移除，
                # 它的 11003210 回复仍要结算）。
                registry = PersistentEffectRegistry.from_snapshot(current, self.templates_by_base_card_id)
                if before is not current:
                    registry.register_departed(before, current, self.templates_by_base_card_id)
                handler = TemplateEventHandler(
                    templates_by_base_card_id=self.templates_by_base_card_id,
                    effect_interpreter=self.effect_interpreter,
                    action=action,
                    generated_cards=generated_cards,
                    state=state,
                            execution_context=execution_context,
                            effect_unresolved_reasons=self._effect_unresolved_reasons,
                    registry=registry,
                    used_generated_uids=step_used_generated_uids,
                )
                dispatch = EventQueue(rest_events).dispatch(current, (handler,))
                current = dispatch.snapshot
                event_consumed = [*entry_consumed, *state.consumed]
                unresolved.extend(state.unresolved)
        except (EffectInterpreterError, EventQueueError) as exc:
            exc_msg = str(exc)
            if _INSUFFICIENT_RESOURCE_MARKER in exc_msg:
                # 资源不足导致效果不发动：整条效果已回滚，盘面是合法的「效果未发动」，
                # 结果是确定的，记 SPEC_ONLY 而不是 PARTIAL/predict_error。
                return PredictionResult(
                    snapshot=current if current is not after_core else after_core,
                    certainty=PredictionCertainty.SPEC_ONLY,
                    description="template effect not activated (insufficient resource)",
                    assumptions=("template_effects_enabled",),
                    unresolved_effects=(),
                )
            # event_target/event_source 解析失败、随机目标已离场等不是致命错误
            # 降级为 PARTIAL 而非 PREDICT_ERROR，避免中断 self-play
            non_fatal_patterns = (
                "cannot resolve target_ref: event_target",
                "cannot resolve target_ref: event_source",
                "target card is missing",
                "target must be in hand",
                "target must be on me field",
            )
            if any(p in exc_msg for p in non_fatal_patterns):
                return PredictionResult(
                    snapshot=current if current is not after_core else after_core,
                    certainty=PredictionCertainty.PARTIAL,
                    description=f"template effect partially failed: {exc}",
                    assumptions=("template_effects_enabled",),
                    unresolved_effects=("event_target_resolution_failed",),
                )
            return PredictionResult(
                snapshot=after_core,
                certainty=PredictionCertainty.PREDICT_ERROR,
                description=f"template effect execution failed: {exc}",
                assumptions=("template_effects_enabled",),
                unresolved_effects=("template_effect_execution_failed",),
            )
        assumptions.append(f"consumed_effects:{','.join(consumed)}")
        if event_consumed:
            assumptions.append(f"consumed_event_effects:{','.join(event_consumed)}")
        return PredictionResult(
            snapshot=current,
            certainty=PredictionCertainty.SPEC_ONLY,
            description="local core rules plus template effects",
            assumptions=tuple(dict.fromkeys(assumptions)),
            unresolved_effects=tuple(dict.fromkeys(unresolved)),
        )

    def _context_after_events(self, context: EffectInterpretationContext, events: tuple[ExecutionEvent, ...]) -> EffectInterpretationContext:
        last_added = next((event.target_uid for event in reversed(events) if event.name in {"draw", "copy_card"} and event.target_uid is not None), None)
        if last_added is not None:
            context = replace(context, last_added_card_uid=last_added)
        summoned_uids = tuple(
            event.target_uid if event.target_uid is not None else event.source_uid
            for event in events
            if event.name == "summon" and (event.target_uid is not None or event.source_uid is not None)
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
        return context

    def _deterministic_random_targets(self, snapshot: BattleSnapshot, effect: ExecutableEffect, action: LegalAction, *, execution_context: LocalExecutionContext | None = None) -> tuple[int, ...]:
        # repeat 每次独立重掷时，不能在 effect 开始时预采样并复用 uid：第一次
        # 伤害可能已使该目标离场。交给 EffectInterpreter._repeat 在每轮以当前
        # snapshot 和 branch_seed 重新取候选。
        if effect.random_policy is not None and effect.random_policy.reroll_each_repeat:
            return ()
        random_ops = tuple(self._random_target_operations(effect, action))
        refs = tuple(ref for ref, _ in random_ops)
        if not refs:
            return ()
        first = refs[0]
        if any(ref != first for ref in refs):
            return ()
        repeat = max((repeat for _, repeat in random_ops), default=1)
        candidates = self.effect_interpreter.random_target_candidates(snapshot, first, source_uid=action.source_uid)
        if not candidates:
            return ()
        if len(candidates) == repeat:
            return candidates
        # 候选数 != repeat：用 branch_seed 采样（允许重复命中同一目标）
        if execution_context is not None and execution_context.branch_seed is not None:
            import random as _random
            sampled = _random.Random(execution_context.branch_seed).choices(candidates, k=repeat)
            return tuple(sampled)
        return ()

    def _random_target_operations(self, effect: ExecutableEffect, action: LegalAction) -> tuple[tuple[str, int], ...]:
        result: list[tuple[str, int]] = []
        if effect.target_selector is not None and effect.target_selector.kind.startswith("random_"):
            result.append((effect.target_selector.kind, 1))
        for operation in effect.operations:
            if not self._operation_mode_selected(operation, action):
                continue
            ref = operation.target_ref or ""
            if ref.startswith("random_"):
                result.append((ref, self._operation_random_repeat(None, operation)))
                continue
            if operation.op == "repeat":
                nested = operation.params.get("operation")
                if not isinstance(nested, dict):
                    continue
                nested_ref = str(nested.get("target_ref", "") or "")
                if nested_ref.startswith("random_"):
                    result.append((nested_ref, int(operation.params.get("count", 1) or 1)))
        return tuple(result)

    def _target_groups_for_effect(self, action: LegalAction, effect: ExecutableEffect, effect_index: int, snapshot: BattleSnapshot) -> tuple[tuple[int, ...], ...]:
        if effect.target_selector is None:
            if any((operation.target_ref or "") == "selected_self_hand_card" for operation in effect.operations):
                return tuple(group for group in action.target_groups if all(uid in snapshot.me.hand and not (action.kind is ActionKind.PLAY and uid == action.source_uid) for uid in group))
            if any((operation.target_ref or "") == "selected_opponent_hand_card" for operation in effect.operations):
                return tuple(group for group in action.target_groups if all(uid in snapshot.opponent.hand for uid in group))
            return action.target_groups
        if not action.target_groups:
            return ()
        matching = tuple(group for group in action.target_groups if self._target_group_matches_selector(snapshot, group, effect.target_selector) and not (action.kind is ActionKind.PLAY and effect.target_selector.kind == "selected_self_hand_card" and action.source_uid in group))
        if matching:
            return (matching[min(effect_index, len(matching) - 1)],)
        if effect.target_selector.zone == "hand" or effect.target_selector.kind.endswith("hand_card"):
            return ()
        if len(action.target_groups) > effect_index:
            return (action.target_groups[effect_index],)
        return action.target_groups

    def _target_group_matches_selector(self, snapshot: BattleSnapshot, group: tuple[int, ...], selector: TargetSelector) -> bool:
        if selector.zone == "hand" or selector.kind.endswith("hand_card"):
            zones = (snapshot.me.hand,) if selector.side in {None, "self"} else (snapshot.opponent.hand,)
        elif selector.side in {"enemy", "opponent"} or selector.kind.startswith("selected_enemy_"):
            zones = (snapshot.opponent.field,)
            if selector.kind == "selected_enemy_follower_or_leader":
                return all(uid == 0 or uid in snapshot.opponent.field for uid in group)
        elif selector.side == "both":
            zones = (snapshot.me.field, snapshot.opponent.field)
        else:
            zones = (snapshot.me.field,)
        allowed = set().union(*(set(zone) for zone in zones))
        return all(uid in allowed for uid in group)

    def _returned_draw_bindings(self, effect: ExecutableEffect, execution_context: LocalExecutionContext, executable: list, *, consumed_count: int) -> tuple:
        if not execution_context.after_hand_bindings:
            return ()
        boost_amount = 0
        for next_effect, _, _, _ in executable[consumed_count + 1:]:
            for operation in next_effect.operations:
                if operation.op != "spellboost_hand":
                    continue
                if operation.amount is not None and operation.amount.kind == "fixed" and operation.amount.value is not None:
                    boost_amount += int(operation.amount.value)
        if boost_amount <= 0:
            return execution_context.after_hand_bindings
        from dataclasses import replace
        from .execution_context import DrawBinding
        return tuple(
            DrawBinding(
                card=replace(
                    binding.card,
                    spell_boost_count=max(0, binding.card.spell_boost_count - boost_amount),
                    cost=binding.card.cost + boost_amount if binding.card.spellboost_reduces_cost else binding.card.cost,
                ),
                source=binding.source,
                instance_id=binding.instance_id,
            )
            for binding in execution_context.after_hand_bindings
        )

    def _target_selector_reason_or_empty(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        effect: ExecutableEffect,
        *,
        source_card: CardView | None,
        player: str = "me",
        opponent: str = "opponent",
    ) -> str | None:
        """目标选择器的 unresolved 原因；合法空池（无满足条件的卡）返回 None。"""
        materialized = materialize_effect_target_groups(
            snapshot,
            action,
            effect,
            source_card=source_card,
            player=player,
            opponent=opponent,
        )
        if materialized.unresolved_reason is None and materialized.target_groups == ((),):
            return None
        return self._target_selector_unresolved_reason(
            snapshot, action, effect, source_card=source_card, player=player, opponent=opponent
        )

    def _target_selector_unresolved_reason(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        effect: ExecutableEffect,
        *,
        source_card: CardView | None,
        player: str = "me",
        opponent: str = "opponent",
    ) -> str:
        materialized = materialize_effect_target_groups(
            snapshot,
            action,
            effect,
            source_card=source_card,
            player=player,
            opponent=opponent,
        )
        if materialized.unresolved_reason == f"target_selector_not_materializable:{effect.effect_id}":
            return f"target_selector_insufficient_candidates:{effect.effect_id}"
        return f"target_selector_not_supported:{effect.effect_id}"

    def _effect_requires_generated_cards(self, effect: ExecutableEffect) -> bool:
        """判断效果的操作是否需要 generated_cards 才能执行（如 summon/reanimate/history_same_name_pool）。"""
        return any(op.op in {"summon", "reanimate", "history_same_name_pool", "put_token_to_hand", "add_to_hand", "transform"} for op in effect.operations)

    def _effect_unresolved_reasons(
        self,
        snapshot: BattleSnapshot,
        effect: ExecutableEffect,
        action: LegalAction,
        generated_cards: tuple[CardView, ...],
        source_card: CardView | None = None,
        *,
        execution_context: LocalExecutionContext | None = None,
        player: str = "me",
        opponent: str = "opponent",
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        condition_result = self._evaluate_effect_conditions(snapshot, effect, action, source_card=source_card)
        reasons.extend(condition_result.unresolved_reasons)
        if effect.random_policy is not None and effect.random_policy.target_pool == "source_pool" and not self._random_policy_satisfied_by_recorded_generated_cards(effect, generated_cards):
            reasons.append(f"random_policy_requires_recorded_targets:{effect.effect_id}")
        if not action.selected_indexes and (self._requires_mode_choice_selection(effect) or self._requires_selected_mode_options(effect)):
            # evolve/super_evolve 动作不需要 mode 选择（mode_choice 通常只在 play 时有效）
            # 不标记为 unresolved，让 _effect_mode_option_matches_action 直接跳过效果
            if action.kind not in (ActionKind.EVOLVE, ActionKind.SUPER_EVOLVE):
                reasons.append(f"mode_option_requires_selected_indexes:{effect.effect_id}")
        requires_random_targets = self._requires_random_targets(snapshot, effect, action, execution_context=execution_context)
        if requires_random_targets and not action.random_target_groups:
            reason = self._target_selector_reason_or_empty(snapshot, action, effect, source_card=source_card, player=player, opponent=opponent)
            if reason is not None:
                reasons.append(reason)
        has_selected_targets = any(group for group in action.target_groups)
        if not requires_random_targets and self._requires_recorded_targets(snapshot, effect, action, execution_context=execution_context) and not has_selected_targets:
            # 「选择自己的 1 张手牌」在**打出后**手牌为空时无事可做（卡面语义为「尽可能选择」），
            # 游戏不会因此中止整段效果：10643110 隔断的龙斗士在手牌为空时仍对对手随机 1 个
            # 随从造成 6 点伤害（replay_2882303761750463579 frame114：墓地不变、对手随从被杀）。
            self_hand_noop = (
                not snapshot.me.hand
                and any((operation.target_ref or "") == "selected_self_hand_card" for operation in effect.operations)
            )
            if not self_hand_noop:
                reason = self._target_selector_reason_or_empty(snapshot, action, effect, source_card=source_card, player=player, opponent=opponent)
                if reason is not None:
                    reasons.append(reason)
        if self._requires_source_on_field(effect) and action.kind == ActionKind.PLAY and source_card is not None and source_card.uid == action.source_uid and source_card.card_type.is_spell:
            reasons.append(f"source_target_not_available_after_play:{effect.effect_id}")
        if self._requires_generated_cards(effect, action) and not generated_cards and self._requires_recorded_generated_cards(effect, action):
            reasons.append(f"generated_cards_not_available:{effect.effect_id}")
        return tuple(reasons)

    def _generated_cards_for_effect(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        effect: ExecutableEffect,
        *,
        generated_cards: tuple[CardView, ...],
        source_card: CardView | None = None,
        offset: int = 0,
        execution_context: LocalExecutionContext = LocalExecutionContext(),
        reserved_uids: frozenset[int] = frozenset(),
        player: str = "me",
    ) -> GeneratedCardMaterialization:
        resolver = self._generated_card_resolver()
        requirement = generated_card_requirement_for_effect(effect, action)
        if generated_cards:
            if requirement.count <= 0:
                return GeneratedCardMaterialization()
            available_generated_cards = tuple(card for card in generated_cards[offset:] if card.uid not in snapshot.cards_by_uid)
            # 同一步里双方各自生成的卡共用同一个池（`_generated_cards` 按 after 帧的
            # hand→field 顺序收集），同一个 effect 若两侧都能生成同类 token，会拿到
            # 对方那一张。`replay_owner_side` 带着取证时的归属侧，这里优先取同侧的卡。
            # 只在同侧候选数量足够时收窄，否则保留原池：有些效果是「使对手召唤」，
            # 生成物本来就落在对方侧。
            same_side_cards = tuple(
                card
                for card in available_generated_cards
                if str(card.supplement_info.get("replay_owner_side") or "") in {"", player}
            )
            if len(same_side_cards) >= requirement.count:
                available_generated_cards = same_side_cards
            cards = select_recorded_generated_cards_for_effect(effect, available_generated_cards, requirement.count)
            if len(cards) != requirement.count:
                # 录制生成物顺序不一定与 spec 中 effects 的执行顺序一致
                # （如 10922110 同一入场曲先召唤海盗旗、再加入黄金之杯，而
                # after 帧按 hand→field 排列）。offset 切片匹配失败时，回退到
                # 全部未消费生成物按候选 id 精确匹配，避免误报 not_available。
                fallback_pool = tuple(card for card in generated_cards if card.uid not in snapshot.cards_by_uid)
                fallback = select_recorded_generated_cards_for_effect(effect, fallback_pool, requirement.count)
                if len(fallback) == requirement.count:
                    cards = fallback
            if len(cards) != requirement.count:
                if requirement.allows_partial_recorded_cards:
                    cards = available_generated_cards
                if len(cards) == 0:
                    # 「使土之印+N」只有场上没有土之印时才会生成新卡，属于条件性
                    # 隐式造卡：录像里场上已有土之印时就没有新卡可绑定，此时应该
                    # 让 op 走「stack+N」分支，而不是把整个 effect 判成不可物化
                    # （replay_...170798 st26：多算一次 generated_cards_not_available，
                    # 结果 stack 5->6 整段没执行）。
                    if any(operation.op == "modify_earth_sigil" for operation in effect.operations):
                        return GeneratedCardMaterialization()
                    return GeneratedCardMaterialization(unresolved_reason=f"generated_cards_not_available:{effect.effect_id}")
            return GeneratedCardMaterialization(cards=cards, assumptions=(f"generated_cards_materialized:{len(cards)}",))
        if not requirement.requires_materialization:
            return GeneratedCardMaterialization()
        return materialize_effect_generated_cards(
            snapshot,
            action,
            effect,
            source_card=source_card,
            resolver=resolver,
            branch_offset=offset,
            reserved_uids=frozenset(binding.card.uid for binding in execution_context.draw_bindings) | reserved_uids | frozenset({action.source_uid}),
            execution_context=execution_context,
            player=player,
        )

    def _random_policy_satisfied_by_recorded_generated_cards(self, effect: ExecutableEffect, generated_cards: tuple[CardView, ...]) -> bool:
        random_policy = effect.random_policy
        if random_policy is None or random_policy.target_pool != "source_pool":
            return False
        expected_count = generated_card_requirement_for_effect(effect, LegalAction(kind=ActionKind.SPECIAL_ACTION, source_uid=0)).count
        if expected_count <= 0 or len(generated_cards) < expected_count:
            return False
        matched = select_recorded_generated_cards_for_effect(effect, generated_cards, expected_count)
        return len(matched) >= expected_count

    def _action_source_has_trigger_event(self, snapshot: BattleSnapshot, action: LegalAction, event: str) -> bool:
        card = snapshot.cards_by_uid.get(action.source_uid)
        return card is not None and self._template_has_trigger_event(card.base_card_id, event)

    def _template_has_trigger_event(self, base_card_id: int, event: str) -> bool:
        template = self.templates_by_base_card_id.get(base_card_id)
        if template is None:
            return False
        events = self._trigger_event_aliases(event)
        return any(effect.trigger.event in events for effect in template.executable_effects)

    def _trigger_event_aliases(self, event: str) -> frozenset[str]:
        if event == "activate":
            return frozenset({"activate", "activation"})
        if event == "activation":
            return frozenset({"activate", "activation"})
        return frozenset({event})

    def _resolve_accelerate_cost(self, template: UnifiedCardTemplate, available_pp: int) -> int | None:
        """从 template play_modes 中获取可负担的 accelerate 费用。"""
        for mode in template.play_modes:
            kind_value = getattr(mode.kind, "value", mode.kind)
            if kind_value != "accelerate":
                continue
            if mode.cost is None or mode.cost.kind != "fixed" or mode.cost.value is None:
                continue
            accel_cost = int(mode.cost.value)
            if available_pp >= accel_cost:
                return accel_cost
        return None

    def _resolve_enhance_cost(self, template: UnifiedCardTemplate, available_pp: int) -> int | None:
        """从 template play_modes 中推断 enhance 费用。

        规则：PP >= enhance_cost 时强制以爆能费用打出；多个 enhance 时选最高可负担的。
        返回 None 表示不走 enhance。
        """
        best: int | None = None
        for mode in template.play_modes:
            kind_value = getattr(mode.kind, "value", mode.kind)
            if kind_value != "enhance":
                continue
            if mode.cost is None or mode.cost.kind != "fixed" or mode.cost.value is None:
                continue
            enhance_cost = int(mode.cost.value)
            if available_pp >= enhance_cost:
                if best is None or enhance_cost > best:
                    best = enhance_cost
        return best

    def _activation_cost_from_template(self, base_card_id: int) -> int | None:
        """从 template play_modes 中获取 activation 的 PP 消耗。"""
        template = self.templates_by_base_card_id.get(base_card_id)
        if template is None:
            return None
        for mode in template.play_modes:
            kind_value = getattr(mode.kind, "value", mode.kind)
            if kind_value != "activation":
                continue
            if mode.cost is None:
                return None
            if mode.cost.kind == "fixed" and mode.cost.value is not None:
                return int(mode.cost.value)
        return None

    def _generated_card_resolver(self):
        resolver = getattr(self.effect_interpreter, "card_resolver", None)
        master_data = getattr(resolver, "master_data", None)
        base_cards = getattr(master_data, "_base_cards", None)
        if isinstance(base_cards, dict) and base_cards:
            return resolver
        return default_card_resolver()

    def _evaluate_effect_conditions(self, snapshot: BattleSnapshot, effect: ExecutableEffect, action: LegalAction, *, source_card: CardView | None = None, event_source_card: CardView | None = None, resource_snapshot: BattleSnapshot | None = None):
        # event_target.* 条件由 template_event_handler 的事件匹配逻辑处理，这里跳过。
        # event_source.* 只在调用方显式提供事件源卡时求值（攻击前预结算等场景），
        # 否则沿用「不在此处判定」的既有语义。
        skipped_prefixes = ("event_source.", "event_target.") if event_source_card is None else ("event_target.",)
        conditions = tuple(c for c in effect.conditions if not c.subject.startswith(skipped_prefixes))
        if not conditions:
            return ConditionEvaluationResult(matched=True, unresolved_reasons=())
        return self.condition_evaluator.evaluate_all(
            snapshot,
            effect,
            conditions,
            ConditionEvaluationContext(action=action, source_card=source_card, event_source_card=event_source_card, resource_snapshot=resource_snapshot),
        )

    def _requires_mode_choice_selection(self, effect: ExecutableEffect) -> bool:
        return any(operation.op == "mode_choice_effect" for operation in effect.operations)

    def _requires_selected_mode_options(self, effect: ExecutableEffect) -> bool:
        for operation in effect.operations:
            mode_option = operation.params.get("mode_option")
            if isinstance(mode_option, dict) and mode_option.get("option_index") is not None:
                return True
        return False

    def _effect_mode_option_matches_action(self, effect: ExecutableEffect, action: LegalAction, *, all_abilities_active: bool = False) -> bool:
        needs_mode = self._requires_selected_mode_options(effect) or self._requires_mode_choice_selection(effect)
        if not needs_mode:
            return True
        # "改为发动所有能力"激活时, 所有具体 mode option 效果都发动 (无视玩家单选)。
        # all_abilities 控制 spec 自身(ability_resolution_mode)仍按其条件正常匹配。
        if all_abilities_active and any(
            op.params.get("ability_resolution_mode") != "all_abilities"
            and isinstance(op.params.get("mode_option"), dict)
            for op in effect.operations
        ):
            return True
        # evolve/super_evolve 动作不触发 mode_choice 效果（mode_choice 仅在 play 时有效）
        if action.kind in (ActionKind.EVOLVE, ActionKind.SUPER_EVOLVE) and not action.selected_indexes:
            return False
        if not action.selected_indexes:
            return True
        return any(self._operation_mode_selected(operation, action) for operation in effect.operations)

    @staticmethod
    def _effect_play_modes_match_action(effect: ExecutableEffect, action: LegalAction) -> bool:
        """检查效果的 trigger.play_modes 是否与当前 action 的 play_kind 匹配。

        规则：
        - play_modes 为空 → 不做限制，始终匹配
        - action.play_kind == ENHANCE → 匹配 "enhance_*" 模式
        - action.play_kind == NORMAL 或 None → 匹配 "normal_play"
        - action.play_kind == ACCELERATE → 匹配 "accelerate_*"
        - action.play_kind == CRYSTAL → 匹配 "crystal_*"
        """
        play_modes = effect.trigger.play_modes
        if not play_modes:
            return True
        play_kind = action.play_kind
        if play_kind is PlayKind.ENHANCE:
            return any(m.startswith("enhance") for m in play_modes)
        if play_kind is PlayKind.ACCELERATE:
            return any(m.startswith("accelerate") for m in play_modes)
        if play_kind is PlayKind.CRYSTAL:
            return any(m.startswith("crystal") for m in play_modes)
        # normal_play 或 None
        return "normal_play" in play_modes

    def _requires_generated_cards(self, effect: ExecutableEffect, action: LegalAction) -> bool:
        from .generated_card_materializer import operation_needs_generated_cards

        for operation in effect.operations:
            if not self._operation_mode_selected(operation, action):
                continue
            if operation.op in {
                "summon",
                "add_to_hand",
                "put_token_to_hand",
                "transform",
                "replace_hand_card",
                "copy_card",
                "copy_from_opponent_hand",
                "reanimate",
                "history_same_name_pool",
            }:
                return True
            # deck_operation 只有 add_to_deck / replace_deck 需要生成卡；
            # stat_buff 等只改牌组内已有实例的 kind 不需要。
            if operation.op == "deck_operation" and operation_needs_generated_cards(operation):
                return True
        return False

    def _requires_recorded_generated_cards(self, effect: ExecutableEffect, action: LegalAction) -> bool:
        requirement = generated_card_requirement_for_effect(effect, action)
        return requirement.count > 0 and not requirement.requires_materialization and not requirement.candidate_base_ids

    def _operation_mode_selected(self, operation, action: LegalAction) -> bool:
        mode_option = operation.params.get("mode_option")
        if not isinstance(mode_option, dict) or mode_option.get("option_index") is None:
            return True
        try:
            return int(mode_option["option_index"]) in action.selected_indexes
        except (TypeError, ValueError):
            return False

    def _requires_source_on_field(self, effect: ExecutableEffect) -> bool:
        return any(
            operation.target_ref in {"source_card", "source_follower", "self_follower"}
            and operation.op not in {"return_to_deck", "bounce"}
            for operation in effect.operations
        )

    def _requires_previous_summoned(self, effect: ExecutableEffect) -> bool:
        return any(operation.target_ref in {"previous_summoned", "last_summoned", "last_summoned_cards"} for operation in effect.operations)

    def _summons_generated_cards(self, effect: ExecutableEffect) -> bool:
        return any(operation.op == "summon" for operation in effect.operations)

    def _requires_random_targets(self, snapshot: BattleSnapshot, effect: ExecutableEffect, action: LegalAction, *, execution_context: LocalExecutionContext | None = None) -> bool:
        if effect.random_policy is not None and effect.random_policy.target_pool == "source_pool":
            return False
        # 有 branch_seed 时预测器可自行采样，不需要外部提供 random_target_groups
        if execution_context is not None and execution_context.branch_seed is not None:
            return False
        return any(
            self._random_target_requires_recorded(snapshot, target_ref, repeat=repeat, source_uid=action.source_uid)
            for target_ref, repeat in self._random_target_operations(effect, action)
        )

    def _operation_random_repeat(self, snapshot: BattleSnapshot | None, operation) -> int:
        # 与 EffectInterpreter 的 explicit_random_target_repeat 同一口径：spec 里
        # repeat / count / random_count 三者等价。此前只认 `repeat`，于是
        # 10401110_auto_001（写的是 count=2 + random_count=2）在这里被算成 1 个目标，
        # 预测器只预采样 1 个 uid，interpreter 侧需要 2 个，抛
        # requires random_targets 后整条「随机 2 个随从 5 点伤害」被跳过
        # （replay_...1748781012 st79）。
        explicit = explicit_random_target_repeat(operation)
        if explicit is not None:
            return explicit
        if operation.target_ref != "random_enemy_followers":
            return 1
        # `random_enemy_followers` 还有一套写法用 amount 表达数量
        # （destroy random_enemy_followers + snapshot_count_diff）；amount 是固定值时
        # 才是目标数量。
        amount = operation.amount
        if amount is not None and amount.kind == "fixed" and amount.value is not None:
            return int(amount.value)
        return 1

    def _random_target_requires_recorded(self, snapshot: BattleSnapshot, target_ref: str, *, repeat: int, source_uid: int | None = None) -> bool:
        candidates = self.effect_interpreter.random_target_candidates(snapshot, target_ref, source_uid=source_uid)
        if not candidates:
            return False
        if len(candidates) == 1:
            return False
        return True

    def _has_concrete_target_groups(self, target_groups: tuple[tuple[int, ...], ...]) -> bool:
        return any(group for group in target_groups)

    def _effect_requires_selected_targets(self, effect: ExecutableEffect, action: LegalAction) -> bool:
        return any(
            (
                operation.op == "bind_multi_target_selection"
                or (operation.target_ref or "").startswith("selected_")
                or (operation.target_ref or "") in {"selected_target", "enemy_follower", "combat_opponent"}
            )
            and not (operation.op == "copy_from_opponent_hand" and not (operation.target_ref or ""))
            and self._operation_mode_selected(operation, action)
            for operation in effect.operations
        )

    def _requires_recorded_targets(self, snapshot: BattleSnapshot, effect: ExecutableEffect, action: LegalAction, *, execution_context: LocalExecutionContext | None = None) -> bool:
        has_seed = execution_context is not None and execution_context.branch_seed is not None
        selector = effect.target_selector
        if selector is not None:
            if selector.optional and action.target_groups == ((),):
                return False
            if selector.optional and (selector.scope == "selected" or selector.kind.startswith("selected_")):
                materialized = materialize_effect_target_groups(snapshot, action, effect, source_card=snapshot.cards_by_uid.get(action.source_uid))
                if materialized.unresolved_reason is None and materialized.target_groups == ((),):
                    return False
            if selector.scope == "selected":
                return True
            if selector.kind.startswith("selected_"):
                return True
            if selector.kind.startswith("random_"):
                if has_seed:
                    return False
                return self.effect_interpreter.random_target_requires_recorded(snapshot, selector.kind)
        if self._effect_requires_selected_targets(effect, action):
            return not self._target_groups_for_effect(action, effect, 0, snapshot)
        if has_seed:
            return False
        return any(
            self._random_target_requires_recorded(snapshot, ref, repeat=repeat, source_uid=action.source_uid)
            for ref, repeat in self._random_target_operations(effect, action)
        )

    def _template_has_no_play_trigger(self, template: UnifiedCardTemplate, action: LegalAction) -> bool:
        if action.kind is not ActionKind.PLAY:
            return False
        play_triggers = {"fanfare", "on_play", "on_play_from_hand", "on_spell_play"}
        if any(effect.trigger.event in play_triggers for effect in template.executable_effects):
            return False
        text = "\n".join((template.source_text.normal_desc, template.source_text.clean_skill_desc, template.source_text.raw_skill_desc)).lower()
        play_markers = ("入场曲", "出牌", "打出", "fanfare", "on play", "play this", "when this card is played")
        return not any(marker in text for marker in play_markers)

    def _template_has_no_action_trigger(self, template: UnifiedCardTemplate, action: LegalAction) -> bool:
        """检查模板是否确认不包含当前 action 对应的触发效果。

        对于 PLAY 动作委托给 _template_has_no_play_trigger。
        对于 EVOLVE/SUPER_EVOLVE，检查模板和文本是否都不包含进化相关 trigger/描述。
        """
        if action.kind is ActionKind.PLAY:
            return self._template_has_no_play_trigger(template, action)
        if action.kind is ActionKind.FUSION:
            # 「融合」既可能即时触发模板效果，也可能只写入已融合/素材状态，供未来
            # 超进化等条件读取。没有 fusion trigger 就是已知无即时模板效果。
            return not any(effect.trigger.event == "fusion" for effect in template.executable_effects)
        if action.kind is ActionKind.EVOLVE:
            evolve_triggers = {"manual_evolve", "becomes_evolved", "on_evolve"}
            if any(effect.trigger.event in evolve_triggers for effect in template.executable_effects):
                return False
            text = "\n".join((template.source_text.normal_desc, template.source_text.clean_skill_desc, template.source_text.raw_skill_desc))
            # 排除"超进化时"和条件语境"为进化后"的误匹配
            text_for_evolve = text.replace("超进化时", "").replace("超进化后", "").replace("为进化后", "").replace("是进化后", "")
            evolve_markers = ("【进化时】", "本随从进化时", "on evolve", "when this follower evolves")
            return not any(marker in text_for_evolve for marker in evolve_markers)
        if action.kind is ActionKind.SUPER_EVOLVE:
            super_triggers = {"manual_super_evolve", "becomes_super_evolved", "on_super_evolve", "manual_evolve", "becomes_evolved", "on_evolve"}
            if any(effect.trigger.event in super_triggers for effect in template.executable_effects):
                return False
            text = "\n".join((template.source_text.normal_desc, template.source_text.clean_skill_desc, template.source_text.raw_skill_desc))
            # 只匹配明确的"本卡超进化/进化时"触发格式，排除被动监听语境
            super_markers = ("【超进化时】", "【进化时】", "on super evolve", "on evolve")
            return not any(marker in text for marker in super_markers)
        return False

    def _error(self, snapshot: BattleSnapshot, message: str, action: LegalAction, *, unresolved: tuple[str, ...] | None = None) -> PredictionResult:
        return PredictionResult(
            snapshot=snapshot,
            certainty=PredictionCertainty.PREDICT_ERROR,
            description=message,
            unresolved_effects=unresolved or (action.kind.value,),
        )


def _damage_modifier_key(key: object) -> bool:
    normalized = str(key).lower()
    return any(token in normalized for token in ("damage", "shield", "barrier", "cut", "reduce"))