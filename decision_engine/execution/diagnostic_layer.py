# -*- coding: utf-8 -*-
"""Replay/search 诊断分层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiagnosticBlocker:
    layer: str = "unknown"
    reason: str = ""
    effect_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_blocker_layer": self.layer,
            "primary_blocker_reason": self.reason,
            "primary_blocker_effect_id": self.effect_id,
        }


MATERIALIZE_PREFIXES = (
    "target_selector_",
    "generated_cards_not_available",
    "generated_cards_not_materializable",
    "previous_summoned_not_available",
    "source_pool_",
    "selected_filtered_cards_",
    "random_policy_requires_recorded_targets",
    "source_target_not_available_after_play",
)

RUNTIME_PREFIXES = (
    "state_adapter_gap",
    "protocol_legal_gap",
    "runtime_hidden_state",
    "hidden_",
    "persistent_per_turn_limit_requires_runtime_counter",
    "load_error",
    "adapter_error",
)

ACTION_PREFIXES = (
    "recorded_action_parse_error",
    "unsupported recorded action kind",
    "unsupported_action_kind",
    "unsupported_action",
)

SPEC_PREFIXES = (
    "condition_not_supported",
    "mode_option_requires_selected_indexes",
    "draft_effect_blocked",
    "template_effect_execution_failed",
)

EVENT_REASONS = {
    "play_spell_effects",
    "on_play_effects",
    "activation_effects",
    "fusion_effects",
    "turn_start_effects",
    "turn_end_effects",
    "manual_evolve_effects",
    "manual_super_evolve_effects",
    "becomes_evolved_effects",
    "becomes_super_evolved_effects",
    "enhance_effects",
    "accelerate_effects",
    "crystal_effects",
    "on_attack",
    "attack",
    "on_clash",
    "last_words",
    "spellboost",
    "on_spell_play",
    "on_source_buffed",
    "manual_super_evolve",
    "manual_evolve",
    "turn_end",
    "turn_start",
}

TEMPLATE_PREFIXES = (
    "missing_template",
    "no_executable_template_effects:no_template",
    "no_executable_template_effects:template_without_executable_effects",
    "no_executable_template_effects:trigger_only:",
    "no_executable_template_effects:missing_executable_template_effects",
    "no_executable_template_effects:no_play_effect_expected",
    "no_executable_template_effects:matched_but_no_operations",
    "no_executable_template_effects:matched_but_no_effect_executed",
)

SPEC_REASONS = {
    "no_executable_template_effects:matched_but_condition_unresolved",
    "no_executable_template_effects:matched_but_mode_option_not_selected",
}

ACTION_REASONS = {
    "no_executable_template_effects:source_missing",
    "no_executable_template_effects:unplayable_card_enumerated",
    "unsupported_goal",
}


STATUS_LAYER = {
    "action_parse_error": "action",
    "adapter_gap": "runtime",
    "source_missing": "action",
    "missing_candidate": "template",
    "card_has_specs_but_no_trigger_match": "template",
    "empty_bundle": "template",
    "candidate_available": "unknown",
}


def classify_blockers(reasons: tuple[str, ...] | list[str], *, description: str = "") -> DiagnosticBlocker:
    ordered = tuple(str(reason) for reason in reasons if str(reason))
    if not ordered:
        return _from_description(description)

    materialize = _first_matching(ordered, _is_materialize_reason)
    if materialize is not None:
        return DiagnosticBlocker("materialize", materialize, _effect_id(materialize))

    runtime = _first_matching(ordered, _is_runtime_reason)
    if runtime is not None:
        return DiagnosticBlocker("runtime", runtime, _effect_id(runtime))

    action = _first_matching(ordered, _is_action_reason)
    if action is not None:
        return DiagnosticBlocker("action", action, _effect_id(action))

    spec = _first_matching(ordered, _is_spec_reason)
    if spec is not None:
        return DiagnosticBlocker("spec", spec, _effect_id(spec))

    event = _first_matching(ordered, _is_event_reason)
    if event is not None:
        return DiagnosticBlocker("event", event, _effect_id(event))

    template = _first_matching(ordered, _is_template_reason)
    if template is not None:
        return DiagnosticBlocker("template", template, _effect_id(template))

    return DiagnosticBlocker("unknown", ordered[0], _effect_id(ordered[0]))


def classify_coverage_gap(status: str, reason: str = "") -> DiagnosticBlocker:
    layer = STATUS_LAYER.get(status, "unknown")
    primary_reason = reason or status
    return DiagnosticBlocker(layer, primary_reason, _effect_id(primary_reason))


def _from_description(description: str) -> DiagnosticBlocker:
    if not description:
        return DiagnosticBlocker()
    reason = str(description)
    if _is_action_reason(reason):
        return DiagnosticBlocker("action", reason, _effect_id(reason))
    if _is_runtime_reason(reason):
        return DiagnosticBlocker("runtime", reason, _effect_id(reason))
    return DiagnosticBlocker("unknown", reason, _effect_id(reason))


def _first_matching(reasons: tuple[str, ...], predicate) -> str | None:
    for reason in reasons:
        if predicate(reason):
            return reason
    return None


def _is_materialize_reason(reason: str) -> bool:
    if reason == "no_executable_template_effects:matched_but_blocked":
        return False
    return reason.startswith(MATERIALIZE_PREFIXES)


def _is_runtime_reason(reason: str) -> bool:
    return reason.startswith(RUNTIME_PREFIXES) or "hidden-state" in reason


def _is_action_reason(reason: str) -> bool:
    return reason in ACTION_REASONS or reason.startswith(ACTION_PREFIXES) or reason.startswith("no_executable_template_effects:trigger_not_matched:")


def _is_spec_reason(reason: str) -> bool:
    return reason in SPEC_REASONS or reason.startswith(SPEC_PREFIXES)


def _is_event_reason(reason: str) -> bool:
    if reason.startswith("no_executable_template_effects:"):
        return False
    return reason in EVENT_REASONS or reason.endswith("_effects")


def _is_template_reason(reason: str) -> bool:
    if reason == "no_executable_template_effects:matched_but_blocked":
        return False
    return reason.startswith(TEMPLATE_PREFIXES)


def _effect_id(reason: str) -> str | None:
    if ":" not in reason:
        return None
    suffix = reason.rsplit(":", 1)[1]
    if not suffix or suffix in {"play", "attack", "evolve", "activate", "fusion", "special_action", "turn_start", "turn_end", "fanfare", "last_words", "enhance", "activation"}:
        return None
    return suffix



def check_zone_cards_consistency(snapshot) -> list[str]:
    """检查 zone lists 与 cards_by_uid 的一致性，返回不一致描述列表。"""
    issues: list[str] = []
    for player_name in ("me", "opponent"):
        view = snapshot.me if player_name == "me" else snapshot.opponent
        for zone_name in ("field", "hand"):
            zone = getattr(view, zone_name, ())
            for uid in zone:
                if uid not in snapshot.cards_by_uid:
                    issues.append(f"{player_name}.{zone_name} contains uid {uid} missing from cards_by_uid")
    return issues
