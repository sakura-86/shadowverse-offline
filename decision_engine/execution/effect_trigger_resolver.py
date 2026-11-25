# -*- coding: utf-8 -*-
"""模板效果触发解析。"""

from __future__ import annotations

from dataclasses import dataclass, field

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.cards.templates import ExecutableEffect, PersistentEffect, UnifiedCardTemplate

from .events import ExecutionEvent


@dataclass(frozen=True)
class EffectTriggerResolution:
    matched_effects: tuple[ExecutableEffect, ...]
    skipped_effect_ids: tuple[str, ...] = ()
    unresolved_reasons: tuple[str, ...] = ()
    per_turn_limits: dict[str, int] = field(default_factory=dict)  # effect_id -> limit


def resolve_template_effects_for_action(
    template: UnifiedCardTemplate,
    action: LegalAction,
    events: tuple[ExecutionEvent, ...] = (),
) -> EffectTriggerResolution:
    trigger_events = _trigger_events_for_action(action.kind, events)
    matched: list[ExecutableEffect] = []
    skipped: list[str] = []
    unresolved: list[str] = []
    for effect in template.executable_effects:
        if effect.trigger.event in trigger_events:
            matched.append(effect)
        else:
            skipped.append(effect.effect_id)
    # "改为N张"替换语义: SUPER_EVOLVE 同时触发 manual_evolve + manual_super_evolve
    # (超进化触发进化时)。若某 manual_super_evolve effect 标记 supersedes_evolve=True
    # (卡文本"超进化时改为N张"是替换而非叠加), 则抑制同卡的 manual_evolve effects,
    # 避免进化时(+1)与超进化时(+N)叠加成 N+1。
    if action.kind is ActionKind.SUPER_EVOLVE and any(
        e.trigger.event == "manual_super_evolve"
        and any(op.params.get("supersedes_evolve") for op in e.operations)
        for e in matched
    ):
        matched = [e for e in matched if e.trigger.event != "manual_evolve"]
    persistent_matched, persistent_unresolved, persistent_limits = _resolve_persistent_effects(template.persistent_effects, trigger_events)
    matched.extend(persistent_matched)
    unresolved.extend(persistent_unresolved)
    return EffectTriggerResolution(
        matched_effects=tuple(matched),
        skipped_effect_ids=tuple(skipped),
        unresolved_reasons=tuple(unresolved),
        per_turn_limits=persistent_limits,
    )


def resolve_template_effects_for_event(
    template: UnifiedCardTemplate,
    event: ExecutionEvent,
    *,
    allowed_persistent_zones: frozenset[str] | None = None,
) -> EffectTriggerResolution:
    """把一张卡作为事件 source 解析它在该时点要执行的效果。

    ``allowed_persistent_zones`` 限制**直接路由段**可以执行哪些 ``source_zone``
    的 persistent 效果。为 ``None`` 时不过滤（保持既有行为）；场地卡在
    ``turn_start`` / ``turn_end`` 时必须传 ``{"field"}``：否则「在手牌中发动」
    （``source_zone="hand"``）的持续效果会在卡出场后继续按场地 source 执行
    （replay_...1246399 st125：场上的 10954120 每个回合结束都被扣 1 费，
    录像里只有它在手牌时才减费）。
    """
    matched: list[ExecutableEffect] = []
    skipped: list[str] = []
    unresolved: list[str] = []
    for effect in template.executable_effects:
        if _event_matches_trigger(event, effect.trigger.event):
            matched.append(effect)
        else:
            skipped.append(effect.effect_id)
    persistent_trigger_events = _EVENT_TO_TRIGGER_ALIASES.get(event.name, (event.name,))
    persistent_matched, persistent_unresolved, persistent_limits = _resolve_persistent_effects(
        template.persistent_effects,
        persistent_trigger_events,
        allowed_persistent_zones=allowed_persistent_zones,
    )
    matched.extend(persistent_matched)
    unresolved.extend(persistent_unresolved)
    return EffectTriggerResolution(
        matched_effects=tuple(matched),
        skipped_effect_ids=tuple(skipped),
        unresolved_reasons=tuple(unresolved),
        per_turn_limits=persistent_limits,
    )


def _resolve_persistent_effects(
    persistent_effects: tuple[PersistentEffect, ...],
    trigger_events: tuple[str, ...],
    *,
    allowed_persistent_zones: frozenset[str] | None = None,
) -> tuple[list[ExecutableEffect], list[str], dict[str, int]]:
    matched: list[ExecutableEffect] = []
    unresolved: list[str] = []
    per_turn_limits: dict[str, int] = {}
    # persistent 效果中的"响应式"触发（如 enter_field, summon, damage 等）
    # 只能通过 EventQueue 正确执行（需要事件上下文如 event_target），
    # 不应参与 action-level 的 template effects 直接执行。
    event_queue_only_triggers = {
        "enter_field", "on_enter_field", "summon", "on_self_entered_field", "on_self_follower_entered_field",
        "damage", "on_source_buffed", "source_buffed", "destroy", "destroyed_by_damage",
        "activate", "on_self_amulet_activated", "becomes_super_evolved",
        "on_self_follower_super_evolved", "discard", "on_discard_self",
        # 「自己使用随从时」类持续效果（如纹章 10974112）需要 event 上下文
        # （本次使用的随从 uid），只能通过 EventQueue 执行。
        "play_card",
    }
    for persistent in persistent_effects:
        if persistent.trigger is None:
            unresolved.append(f"persistent_trigger_not_supported:{persistent.effect_id}")
            continue
        if persistent.trigger.event not in trigger_events:
            continue
        if allowed_persistent_zones is not None and persistent.source_zone not in allowed_persistent_zones:
            # 该持续效果从别的区域发动（如手牌），不能由场上的同一张卡执行。
            continue
        # 响应式 persistent 只在 EventQueue 中处理，不参与 action resolution
        if persistent.trigger.event in event_queue_only_triggers:
            continue
        if persistent.countdown is not None:
            unresolved.append(f"persistent_countdown_not_supported:{persistent.effect_id}")
            continue
        if not persistent.executable_effects:
            unresolved.append(f"persistent_executable_effects_missing:{persistent.effect_id}")
            continue
        matched.extend(persistent.executable_effects)
        if persistent.per_turn_limit is not None:
            for exe in persistent.executable_effects:
                per_turn_limits[exe.effect_id] = persistent.per_turn_limit
    return matched, unresolved, per_turn_limits


def _trigger_events_for_action(kind: ActionKind, events: tuple[ExecutionEvent, ...]) -> tuple[str, ...]:
    names = {event.name for event in events if event.name not in {"on_source_buffed", "source_buffed", "spellboost"}}
    result: list[str] = list(names)
    if kind is ActionKind.PLAY:
        if "accelerate" in names or "crystal" in names:
            # 激奏/结晶模式：作为法术/护符使用，不触发原卡 fanfare
            result.extend(("mode_choice_resolution", "ability_resolution", "on_play", "on_play_from_hand"))
            if "play_spell" in names:
                result.append("on_spell_play")
            if "accelerate" in names:
                result.append("accelerate")
            if "crystal" in names:
                result.append("crystal")
        else:
            result.extend(("mode_choice_resolution", "ability_resolution", "fanfare", "on_play", "on_play_from_hand"))
            if "play_spell" in names:
                result.append("on_spell_play")
    elif kind is ActionKind.EVOLVE:
        result.extend(("mode_choice_resolution", "ability_resolution", "manual_evolve", "becomes_evolved", "on_evolve"))
    elif kind is ActionKind.SUPER_EVOLVE:
        result.extend(("mode_choice_resolution", "ability_resolution", "manual_super_evolve", "manual_evolve", "becomes_super_evolved", "becomes_evolved", "on_super_evolve", "on_evolve"))
    elif kind is ActionKind.TURN_START:
        # 回合开始时，牌组区被动与“回合开始时瞬念召唤”均在抽牌前检查。
        result.extend(("turn_start", "deck_zone_passive", "turn_start_invoke"))
    elif kind in (ActionKind.ATTACK_LEADER, ActionKind.ATTACK_FOLLOWER):
        result.extend(("on_attack", "attack"))
        if "clash" in names:
            result.append("on_clash")
    elif kind is ActionKind.ACTIVATE:
        result.extend(("activate", "activation"))
    else:
        result.append(kind.value)
    return tuple(dict.fromkeys(result))


# ── event-to-trigger 匹配映射 ──────────────────────────────────────
# enter_field 事件可以匹配 on_self_entered_field trigger（但 handler 需要额外的 uid 过滤）
_EVENT_TO_TRIGGER_ALIASES: dict[str, tuple[str, ...]] = {
    "enter_field": ("enter_field", "on_enter_field", "on_self_entered_field", "on_self_follower_entered_field", "on_enemy_follower_entered_field"),
    "activate": ("activate", "activation", "on_self_amulet_activated"),
    # 护符被破坏事件匹配"自己的护符被破坏时"(on_self_amulet_destroyed) 反应式效果。
    # handler 侧再过滤"被破坏的是己方护符"(见 _event_trigger_scope_matches)。
    # 「…在战场上**消失**时」(on_self_card_left_field) 只对应 banish：卡面「消失」
    # 就是 banish 这个动作，普通破坏/回手不算。
    "destroy": ("destroy", "on_self_amulet_destroyed"),
    "destroyed_by_damage": ("destroyed_by_damage", "on_self_amulet_destroyed"),
    "banish": ("banish", "on_self_card_left_field"),
    "bounce": ("bounce",),
    "return_to_deck": ("return_to_deck",),
    "becomes_super_evolved": ("becomes_super_evolved", "on_self_follower_super_evolved"),
    # 效果造成的自动进化（纹章「使用随从时使其进化」等）**不触发【进化时】**：
    # 卡面的「进化时」只对手动进化生效。手动进化会同时发 manual_evolve 与
    # becomes_evolved 两个事件，`manual_evolve` 事件本身就能匹配 trigger=manual_evolve，
    # 不需要 becomes_evolved 再别名过去。此前把 manual_evolve 放进这里的别名，
    # 导致 replay_...749786874 st72：纹章让 10974110 自动进化时，
    # 【进化时】「抽 1 张拥有【毁灭】的超越者·随从」被错误触发，多抽 1 张。
    "becomes_evolved": ("becomes_evolved", "on_evolve"),
    "play_spell": ("fanfare", "on_spell_play"),
    "turn_start_field_last_words": ("turn_start_field_last_words",),
    "turn_start_crest_last_words": ("turn_start_crest_last_words",),
    "turn_start_deck": ("deck_zone_passive", "turn_start_invoke"),
    "turn_end_deck": ("deck_zone_passive", "turn_end_invoke"),
    "turn_start_field_effects": ("turn_start",),
    "turn_start_crest_effects": ("turn_start",),
    "turn_start_invoke_effect": ("turn_start_invoke_effect",),
    # 回合结束：spec 里既有 trigger=turn_end（core 事件名），也有 trigger=on_turn_end
    # （持久效果命名）。persistent_effect_registry 的别名表两者都认，直接路由段原先
    # 只认 turn_end，导致 10742120「自己的回合结束时回复主战者1点」这类
    # trigger=on_turn_end 的直接 spec 既不进候选也不被路由（B_turn_end_blind）。
    "turn_end": ("turn_end", "on_turn_end"),
    "turn_start": ("turn_start", "on_turn_start"),
    # 攻击事件匹配【攻击时】(on_attack) 效果
    "attack_leader": ("on_attack", "attack"),
    "attack_follower": ("on_attack", "attack"),
    "clash": ("on_clash",),
    # 主战者回复事件匹配"主战者回复时"(on_leader_healed) 效果
    "heal": ("on_leader_healed",),
}


def _event_matches_trigger(event: ExecutionEvent, trigger_event: str) -> bool:
    """事件是否匹配 trigger event。支持别名映射。"""
    if trigger_event == event.name:
        return True
    aliases = _EVENT_TO_TRIGGER_ALIASES.get(event.name, ())
    return trigger_event in aliases
