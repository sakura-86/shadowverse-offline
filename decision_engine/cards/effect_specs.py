# -*- coding: utf-8 -*-
"""Phase L2 可执行效果规格 schema。

EffectSpec 是本地执行器未来消费的正式规格；本模块只定义结构与序列化，不执行效果。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

from .templates import AmountFormula, CopyPolicy, SummonPolicy, TransformPolicy


class EffectSpecSourceKind(Enum):
    AUTO_TEXT_PARSE = "auto_text_parse"
    MANUAL_RULE_SPEC = "manual_rule_spec"
    RUNTIME_CONFIRMED = "runtime_confirmed"
    REPLAY_CONFIRMED = "replay_confirmed"
    MASTER_DATA = "master_data"


@dataclass(frozen=True)
class EffectSpecTrigger:
    event: str
    source_zone: str | None = None
    play_origin: str | None = None
    allowed_sources: tuple[str, ...] = ()
    play_modes: tuple[str, ...] = ()
    manual_policy: str | None = None


@dataclass(frozen=True)
class EffectSpecCondition:
    kind: str
    subject: str = ""
    operator: str = ""
    value: Any = None
    ref: str | None = None


@dataclass(frozen=True)
class EffectSpecTarget:
    kind: str
    side: str | None = None
    zone: str | None = None
    card_type: str | None = None
    count: int | None = None
    filters: tuple[dict[str, Any], ...] = ()
    optional: bool = False
    selector_refs: tuple[str, ...] = ()
    relation: str | None = None
    scope: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EffectSpecTarget":
        return _from_dict(cls, data)


@dataclass(frozen=True)
class EffectSpecRandomPolicy:
    target_pool: str
    count: int = 1
    replacement: bool = False
    reroll_each_repeat: bool = True
    seed_source: str = "runtime_or_search_branch"


@dataclass(frozen=True)
class EffectSpecEventPolicy:
    ordering_group: str = "text_order"
    emits: tuple[str, ...] = ()
    consumes_events: tuple[str, ...] = ()


@dataclass(frozen=True)
class EffectSpecResolutionPolicy:
    order: str = "text_order"
    partial_failure: str = "continue"
    board_full_policy: str | None = None
    target_missing_policy: str = "skip_operation"


@dataclass(frozen=True)
class EffectSpecRollbackRequirement:
    required: bool = True
    transaction_kinds: tuple[str, ...] = ()
    random_choices_must_be_logged: bool = True


@dataclass(frozen=True)
class EffectSpecOperation:
    op: str
    target_ref: str | None = None
    amount: AmountFormula | None = None
    params: dict[str, Any] = field(default_factory=dict)
    emits: tuple[str, ...] = ()
    summon_policy: SummonPolicy | None = None
    transform_policy: TransformPolicy | None = None
    copy_policy: CopyPolicy | None = None


@dataclass(frozen=True)
class EffectSpecSource:
    kind: EffectSpecSourceKind | str
    raw_text: str = ""
    clean_text: str = ""
    text_key: str = ""
    rule_ref: str = ""


@dataclass(frozen=True)
class EffectSpec:
    spec_id: str
    base_card_id: int
    name: str
    trigger: EffectSpecTrigger
    conditions: tuple[EffectSpecCondition, ...] = ()
    target: EffectSpecTarget | None = None
    operations: tuple[EffectSpecOperation, ...] = ()
    random_policy: EffectSpecRandomPolicy | None = None
    event_policy: EffectSpecEventPolicy = field(default_factory=EffectSpecEventPolicy)
    resolution_policy: EffectSpecResolutionPolicy = field(default_factory=EffectSpecResolutionPolicy)
    rollback: EffectSpecRollbackRequirement = field(default_factory=EffectSpecRollbackRequirement)
    source: EffectSpecSource = field(default_factory=lambda: EffectSpecSource(kind=EffectSpecSourceKind.MANUAL_RULE_SPEC))

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EffectSpec":
        return _from_dict(cls, data)


@dataclass(frozen=True)
class EffectSpecPersistentEffect:
    effect_id: str
    owner: str
    source_zone: str
    source_card_ref: int | None = None
    trigger: EffectSpecTrigger | None = None
    conditions: tuple[EffectSpecCondition, ...] = ()
    per_turn_limit: int | None = None
    countdown: int | None = None
    duration: str = "while_active"
    executable_effects: tuple[EffectSpec, ...] = ()
    source: EffectSpecSource = field(default_factory=lambda: EffectSpecSource(kind=EffectSpecSourceKind.MANUAL_RULE_SPEC))

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EffectSpecPersistentEffect":
        return _from_dict(cls, data)


@dataclass(frozen=True)
class CardEffectSpecBundle:
    base_card_id: int
    name: str
    specs: tuple[EffectSpec, ...] = ()
    persistent_specs: tuple[EffectSpecPersistentEffect, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CardEffectSpecBundle":
        return _from_dict(cls, data)


T = TypeVar("T")


def load_effect_spec_bundles(path: str | Path) -> tuple[CardEffectSpecBundle, ...]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_bundles = payload.get("cards", payload) if isinstance(payload, dict) else payload
    return tuple(CardEffectSpecBundle.from_dict(item) for item in raw_bundles)


def load_effect_spec_payload(path: str | Path) -> dict[str, Any] | list[dict[str, Any]]:
    """读取原始 EffectSpec 聚合文件，供构建与审计链路校验。"""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise ValueError(f"unsupported effect spec format: {path}")


def validate_effect_spec_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> tuple[CardEffectSpecBundle, ...]:
    """校验 EffectSpec 聚合的结构、归属和生成物引用一致性。

    此校验只保护规格数据与编译输入的完整性；卡面语义仍必须走
    snapshot/next-frame/replay/runtime 动态验收。
    """
    errors: list[str] = []
    if isinstance(payload, dict):
        raw_bundles = payload.get("cards")
        if not isinstance(raw_bundles, list):
            errors.append("payload.cards must be a list")
            raw_bundles = []
        declared_count = payload.get("count")
        if declared_count is not None:
            if not isinstance(declared_count, int):
                errors.append("payload.count must be an integer when present")
            elif declared_count != len(raw_bundles):
                errors.append(f"payload.count={declared_count} does not match cards={len(raw_bundles)}")
    else:
        raw_bundles = payload
    if not all(isinstance(item, dict) for item in raw_bundles):
        errors.append("payload.cards contains a non-object bundle")

    bundles: list[CardEffectSpecBundle] = []
    for index, raw_bundle in enumerate(raw_bundles):
        if not isinstance(raw_bundle, dict):
            continue
        try:
            bundles.append(CardEffectSpecBundle.from_dict(raw_bundle))
        except (TypeError, ValueError) as exc:
            errors.append(f"cards[{index}] cannot be decoded: {exc}")
    errors.extend(_effect_spec_bundle_errors(tuple(bundles)))
    if errors:
        raise ValueError("invalid effect specs:\n- " + "\n- ".join(errors))
    return tuple(bundles)


def validate_effect_spec_bundles(bundles: tuple[CardEffectSpecBundle, ...]) -> None:
    """校验内存中的 EffectSpec bundle，供构建工具与测试复用。"""
    errors = _effect_spec_bundle_errors(bundles)
    if errors:
        raise ValueError("invalid effect specs:\n- " + "\n- ".join(errors))


def effect_spec_counts(bundles: tuple[CardEffectSpecBundle, ...]) -> dict[str, int]:
    """返回用于 provenance 的稳定规格数量。"""
    return {
        "bundle_count": len(bundles),
        "effect_spec_count": sum(len(bundle.specs) for bundle in bundles),
        "persistent_spec_count": sum(len(bundle.persistent_specs) for bundle in bundles),
        "persistent_executable_effect_count": sum(
            len(effect.executable_effects)
            for bundle in bundles
            for effect in bundle.persistent_specs
        ),
    }


def _effect_spec_bundle_errors(bundles: tuple[CardEffectSpecBundle, ...]) -> list[str]:
    errors: list[str] = []
    seen_bundle_ids: set[int] = set()
    seen_spec_ids: set[str] = set()
    for bundle in bundles:
        base_card_id = bundle.base_card_id
        if base_card_id <= 0:
            errors.append(f"bundle has invalid base_card_id={base_card_id}")
        if base_card_id in seen_bundle_ids:
            errors.append(f"duplicate bundle base_card_id={base_card_id}")
        seen_bundle_ids.add(base_card_id)
        for spec in bundle.specs:
            _validate_effect_spec(spec, base_card_id, seen_spec_ids, errors, context=f"bundle {base_card_id}")
        for persistent in bundle.persistent_specs:
            if persistent.source_card_ref is not None and persistent.source_card_ref != base_card_id:
                errors.append(
                    f"bundle {base_card_id} persistent {persistent.effect_id} source_card_ref="
                    f"{persistent.source_card_ref} does not match bundle"
                )
            if not persistent.effect_id:
                errors.append(f"bundle {base_card_id} has persistent effect with empty effect_id")
            for spec in persistent.executable_effects:
                _validate_effect_spec(
                    spec,
                    base_card_id,
                    seen_spec_ids,
                    errors,
                    context=f"bundle {base_card_id} persistent {persistent.effect_id}",
                )
    return errors


def _validate_effect_spec(
    spec: EffectSpec,
    bundle_base_card_id: int,
    seen_spec_ids: set[str],
    errors: list[str],
    *,
    context: str,
) -> None:
    if not spec.spec_id:
        errors.append(f"{context} has effect with empty spec_id")
    elif spec.spec_id in seen_spec_ids:
        errors.append(f"duplicate spec_id={spec.spec_id}")
    seen_spec_ids.add(spec.spec_id)
    if spec.base_card_id != bundle_base_card_id:
        errors.append(
            f"{context} spec {spec.spec_id} base_card_id={spec.base_card_id} "
            f"does not match bundle"
        )
    if not spec.trigger.event:
        errors.append(f"{context} spec {spec.spec_id} has empty trigger.event")
    for operation_index, operation in enumerate(spec.operations):
        _validate_generated_card_params(
            operation.params,
            errors,
            context=f"{context} spec {spec.spec_id} operation[{operation_index}]",
        )


def _validate_generated_card_params(params: dict[str, Any], errors: list[str], *, context: str) -> None:
    candidate_ids = params.get("candidate_base_card_ids")
    generated_refs = params.get("generated_card_refs")
    if candidate_ids is not None:
        if not isinstance(candidate_ids, (list, tuple)) or not candidate_ids:
            errors.append(f"{context} candidate_base_card_ids must be a non-empty list")
            candidate_set: set[int] = set()
        elif not all(isinstance(card_id, int) and card_id > 0 for card_id in candidate_ids):
            errors.append(f"{context} candidate_base_card_ids must contain positive integers")
            candidate_set = set()
        elif len(set(candidate_ids)) != len(candidate_ids):
            errors.append(f"{context} candidate_base_card_ids contains duplicates")
            candidate_set = set(candidate_ids)
        else:
            candidate_set = set(candidate_ids)
    else:
        candidate_set = set()
    if generated_refs is None:
        return
    if not isinstance(generated_refs, (list, tuple)) or not generated_refs:
        errors.append(f"{context} generated_card_refs must be a non-empty list")
        return
    for ref_index, ref in enumerate(generated_refs):
        ref_context = f"{context} generated_card_refs[{ref_index}]"
        if not isinstance(ref, dict):
            errors.append(f"{ref_context} must be an object")
            continue
        base_card_ids = ref.get("base_card_ids")
        if not isinstance(base_card_ids, (list, tuple)) or not base_card_ids:
            errors.append(f"{ref_context}.base_card_ids must be a non-empty list")
            continue
        if not all(isinstance(card_id, int) and card_id > 0 for card_id in base_card_ids):
            errors.append(f"{ref_context}.base_card_ids must contain positive integers")
            continue
        if len(set(base_card_ids)) != len(base_card_ids):
            errors.append(f"{ref_context}.base_card_ids contains duplicates")
        if candidate_ids is not None and not set(base_card_ids).issubset(candidate_set):
            errors.append(f"{ref_context}.base_card_ids must be included in candidate_base_card_ids")
        preferred_base_card_ids = ref.get("preferred_base_card_ids")
        if preferred_base_card_ids is None:
            continue
        if not isinstance(preferred_base_card_ids, (list, tuple)) or not preferred_base_card_ids:
            errors.append(f"{ref_context}.preferred_base_card_ids must be a non-empty list when present")
            continue
        if not all(isinstance(card_id, int) and card_id > 0 for card_id in preferred_base_card_ids):
            errors.append(f"{ref_context}.preferred_base_card_ids must contain positive integers")
            continue
        if not set(preferred_base_card_ids).issubset(set(base_card_ids)):
            errors.append(f"{ref_context}.preferred_base_card_ids must be included in base_card_ids")


def _to_plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    return value


def _from_dict(cls: type[T], data: Any) -> T:
    if data is None:
        return data
    origin = get_origin(cls)
    if origin in (tuple, list):
        args = get_args(cls)
        item_type = args[0] if args else Any
        return tuple(_from_dict(item_type, item) for item in data)  # type: ignore[return-value]
    if origin is dict:
        return dict(data)  # type: ignore[return-value]
    if origin is None and isinstance(cls, type) and issubclass(cls, Enum):
        return cls(data)  # type: ignore[return-value]
    if origin is not None:
        args = [arg for arg in get_args(cls) if arg is not type(None)]
        for arg in args:
            if isinstance(arg, type) and issubclass(arg, Enum):
                try:
                    return _from_dict(arg, data)
                except ValueError:
                    continue
        if len(args) == 1:
            return _from_dict(args[0], data)
        return data
    if not is_dataclass(cls):
        return data

    kwargs: dict[str, Any] = {}
    annotations = get_type_hints(cls)
    for name, field_type in annotations.items():
        if isinstance(data, dict) and name in data:
            kwargs[name] = _from_dict(field_type, data[name])
    return cls(**kwargs)  # type: ignore[return-value]
