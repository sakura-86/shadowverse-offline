# -*- coding: utf-8 -*-
"""EffectSpec 到 UnifiedCardTemplate 可执行效果的绑定层。

本模块只做结构映射，不执行效果、不访问 battle snapshot。
EffectSpec 的正确性由可复现的快照与下一帧验收维护，不由静态审核状态控制。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .effect_specs import (
    CardEffectSpecBundle,
    EffectSpec,
    EffectSpecCondition,
    EffectSpecEventPolicy,
    EffectSpecOperation,
    EffectSpecPersistentEffect,
    EffectSpecRandomPolicy,
    EffectSpecResolutionPolicy,
    EffectSpecRollbackRequirement,
    EffectSpecSource,
    EffectSpecTarget,
    EffectSpecTrigger,
)
from .templates import (
    ConditionSpec,
    EffectSource,
    EventPolicy,
    ExecutableEffect,
    OperationSpec,
    PersistentEffect,
    RandomPolicy,
    ResolutionPolicy,
    RollbackPolicy,
    TargetSelector,
    TriggerSpec,
    UnifiedCardTemplate,
)


class EffectBindingError(ValueError):
    pass


def _status_value(status: object) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _compile_trigger(trigger: EffectSpecTrigger) -> TriggerSpec:
    return TriggerSpec(
        event=trigger.event,
        source_zone=trigger.source_zone,
        play_origin=trigger.play_origin,
        allowed_sources=trigger.allowed_sources,
        play_modes=trigger.play_modes,
        manual_policy=trigger.manual_policy,
    )


def _compile_condition(condition: EffectSpecCondition) -> ConditionSpec:
    return ConditionSpec(
        kind=condition.kind,
        subject=condition.subject,
        operator=condition.operator,
        value=condition.value,
        ref=condition.ref,
    )


def _compile_target(target: EffectSpecTarget | None) -> TargetSelector | None:
    if target is None:
        return None
    return TargetSelector(
        kind=target.kind,
        filters=target.filters,
        count=target.count,
        optional=target.optional,
        side=target.side,
        zone=target.zone,
        card_type=target.card_type,
        selector_refs=target.selector_refs,
        relation=target.relation,
        scope=target.scope,
    )


def _compile_random_policy(policy: EffectSpecRandomPolicy | None) -> RandomPolicy | None:
    if policy is None:
        return None
    return RandomPolicy(
        target_pool=policy.target_pool,
        count=policy.count,
        replacement=policy.replacement,
        reroll_each_repeat=policy.reroll_each_repeat,
    )


def _compile_event_policy(policy: EffectSpecEventPolicy) -> EventPolicy:
    return EventPolicy(emits=policy.emits, ordering_group=policy.ordering_group)


def _compile_resolution_policy(policy: EffectSpecResolutionPolicy) -> ResolutionPolicy:
    return ResolutionPolicy(order=policy.order, partial_failure=policy.partial_failure)


def _compile_rollback_policy(policy: EffectSpecRollbackRequirement) -> RollbackPolicy:
    return RollbackPolicy(required=policy.required, transaction_kinds=policy.transaction_kinds)


def _compile_operation(operation: EffectSpecOperation) -> OperationSpec:
    return OperationSpec(
        op=operation.op,
        target_ref=operation.target_ref,
        amount=operation.amount,
        params=operation.params,
        emits=operation.emits,
        summon_policy=operation.summon_policy,
        transform_policy=operation.transform_policy,
        copy_policy=operation.copy_policy,
    )


def _compile_source(source: EffectSpecSource) -> EffectSource:
    return EffectSource(
        kind=_status_value(source.kind),
        raw_text=source.raw_text,
        clean_text=source.clean_text,
        text_key=source.text_key,
    )


def compile_effect_spec(spec: EffectSpec) -> ExecutableEffect:
    return ExecutableEffect(
        effect_id=spec.spec_id,
        trigger=_compile_trigger(spec.trigger),
        conditions=tuple(_compile_condition(condition) for condition in spec.conditions),
        target_selector=_compile_target(spec.target),
        operations=tuple(_compile_operation(operation) for operation in spec.operations),
        random_policy=_compile_random_policy(spec.random_policy),
        event_policy=_compile_event_policy(spec.event_policy),
        resolution_policy=_compile_resolution_policy(spec.resolution_policy),
        rollback_policy=_compile_rollback_policy(spec.rollback),
        source=_compile_source(spec.source),
    )


def compile_effect_spec_bundle(bundle: CardEffectSpecBundle) -> tuple[ExecutableEffect, ...]:
    return tuple(compile_effect_spec(spec) for spec in bundle.specs)


def compile_persistent_effect_spec(spec: EffectSpecPersistentEffect) -> PersistentEffect:
    return PersistentEffect(
        effect_id=spec.effect_id,
        owner=spec.owner,
        source_zone=spec.source_zone,
        source_card_ref=spec.source_card_ref,
        trigger=_compile_trigger(spec.trigger) if spec.trigger is not None else None,
        conditions=tuple(_compile_condition(condition) for condition in spec.conditions),
        per_turn_limit=spec.per_turn_limit,
        countdown=spec.countdown,
        duration=spec.duration,
        executable_effects=tuple(compile_effect_spec(effect) for effect in spec.executable_effects),
        source=_compile_source(spec.source),
    )


def compile_persistent_effect_specs(bundle: CardEffectSpecBundle) -> tuple[PersistentEffect, ...]:
    return tuple(compile_persistent_effect_spec(spec) for spec in bundle.persistent_specs)


def bind_effect_specs_to_template(
    template: UnifiedCardTemplate,
    bundle: CardEffectSpecBundle,
) -> UnifiedCardTemplate:
    if template.identity.base_card_id != bundle.base_card_id:
        raise EffectBindingError(
            f"effect spec bundle base_card_id mismatch: template={template.identity.base_card_id} "
            f"bundle={bundle.base_card_id}"
        )
    for spec in bundle.specs:
        if spec.base_card_id != template.identity.base_card_id:
            raise EffectBindingError(
                f"effect spec base_card_id mismatch: template={template.identity.base_card_id} "
                f"spec={spec.base_card_id} spec_id={spec.spec_id}"
            )
    for persistent_spec in bundle.persistent_specs:
        for nested_spec in persistent_spec.executable_effects:
            if nested_spec.base_card_id != template.identity.base_card_id:
                raise EffectBindingError(
                    f"persistent effect nested spec base_card_id mismatch: template={template.identity.base_card_id} "
                    f"spec={nested_spec.base_card_id} spec_id={nested_spec.spec_id}"
                )
    return replace(
        template,
        executable_effects=compile_effect_spec_bundle(bundle),
        persistent_effects=compile_persistent_effect_specs(bundle),
    )
