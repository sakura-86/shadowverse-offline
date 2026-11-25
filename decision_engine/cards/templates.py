# -*- coding: utf-8 -*-
"""统一卡牌模板 schema。

该模块只定义 Phase L 的模板数据结构，不执行卡牌效果。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, TypeVar, get_args, get_origin, get_type_hints


class PlayModeKind(Enum):
    NORMAL_PLAY = "normal_play"
    ENHANCE = "enhance"
    ACCELERATE = "accelerate"
    CRYSTALLIZE = "crystallize"
    MODE_CHOICE = "mode_choice"
    CHOICE_SKILL = "choice_skill"
    FUSION = "fusion"
    ACTIVATION = "activation"
    SPECIAL_ACTION = "special_action"


@dataclass(frozen=True)
class CardIdentity:
    base_card_id: int
    name: str
    type_id: int
    type_name: str
    class_id: int
    class_name: str
    rarity: int | str
    rarity_name: str
    tribes: tuple[int | str, ...] = ()
    card_ids: tuple[int, ...] = ()
    normal_card_id: int | None = None
    evolved_card_id: int | None = None
    super_evolved_card_id: int | None = None


@dataclass(frozen=True)
class PrintedStats:
    cost: int
    atk: int
    life: int
    evolved_atk: int | None = None
    evolved_life: int | None = None
    super_evolved_atk: int | None = None
    super_evolved_life: int | None = None
    countdown: int | None = None
    faith_value: int | None = None


@dataclass(frozen=True)
class CardFormSpec:
    form: str
    card_id: int
    atk: int | None = None
    life: int | None = None
    cost: int | None = None


@dataclass(frozen=True)
class CostRule:
    kind: str
    value: int | None = None
    variable: str | None = None


@dataclass(frozen=True)
class ModeChoiceOption:
    choice_index: int
    raw_text: str = ""
    clean_text: str = ""
    text_key: str = ""
    skill_index: int | None = None


@dataclass(frozen=True)
class FusionSpec:
    """静态融合材料限制；具体 uid 由当前局面枚举。"""

    min_material_count: int = 1
    max_material_count: int | None = None
    allowed_tribes: tuple[int, ...] = ()
    allowed_class_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlayModeSpec:
    mode_id: str
    kind: PlayModeKind | str
    cost: CostRule | None = None
    priority: int = 0
    selection_count: int | None = None
    resolve_policy: str | None = None
    required_targets: tuple[str, ...] = ()
    effect_refs: tuple[str, ...] = ()
    condition_refs: tuple[str, ...] = ()
    options: tuple[ModeChoiceOption, ...] = ()
    overrides: tuple[dict[str, Any], ...] = ()
    protocol_mapping_hint: str = ""
    fusion: FusionSpec | None = None


@dataclass(frozen=True)
class KeywordSpec:
    keyword: str
    source: str = "text"
    duration: str = "while_on_field"


@dataclass(frozen=True)
class RelatedCardRef:
    base_card_id: int
    relation: str
    source: str = "master_data"


@dataclass(frozen=True)
class TriggerSpec:
    event: str
    source_zone: str | None = None
    play_origin: str | None = None
    allowed_sources: tuple[str, ...] = ()
    play_modes: tuple[str, ...] = ()
    manual_policy: str | None = None


@dataclass(frozen=True)
class ConditionSpec:
    kind: str
    subject: str = ""
    operator: str = ""
    value: Any = None
    ref: str | None = None


@dataclass(frozen=True)
class TargetSelector:
    kind: str
    filters: tuple[dict[str, Any], ...] = ()
    count: int | None = None
    optional: bool = False
    side: str | None = None
    zone: str | None = None
    card_type: str | None = None
    selector_refs: tuple[str, ...] = ()
    relation: str | None = None
    scope: str | None = None


@dataclass(frozen=True)
class AmountFormula:
    kind: str
    value: int | None = None
    variable: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RandomPolicy:
    target_pool: str
    count: int = 1
    replacement: bool = False
    reroll_each_repeat: bool = True


@dataclass(frozen=True)
class EventPolicy:
    emits: tuple[str, ...] = ()
    ordering_group: str = "text_order"


@dataclass(frozen=True)
class ResolutionPolicy:
    order: str = "text_order"
    partial_failure: str = "continue"


@dataclass(frozen=True)
class RollbackPolicy:
    required: bool = True
    transaction_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class SummonPolicy:
    order: str = "listed_order"
    field_full_policy: str = "fail_to_enter_and_vanish"
    overflow_destination: str | None = None
    overflow_counts_as_destroyed: bool = False
    overflow_triggers_last_words: bool = False
    overflow_enters_graveyard: bool = False
    overflow_emits_enter_events: bool = False


@dataclass(frozen=True)
class TransformPolicy:
    preserve_controller: bool = True
    preserve_zone: bool = True
    preserve_position: bool = True
    preserve_unique_id: bool = True
    preserve_damage: bool = False
    preserve_buffs: bool = False
    preserve_keywords: bool = False
    preserve_evolved_state: bool = False
    preserve_attack_used: bool = False
    counts_as_new_entered: bool = True
    emits_fanfare: bool = False
    can_trigger_last_words_later: bool = True


@dataclass(frozen=True)
class CopyPolicy:
    copy_kind: str = "current_instance_snapshot"
    copy_current_stats: bool = True
    copy_damage: bool = True
    copy_buffs: bool = True
    copy_keywords: bool = True
    copy_cost_modifications: bool = True
    copy_spellboost: bool = True
    copy_evolved_state: bool = True
    copy_countdown: bool = True
    copy_runtime_counters: bool = True


@dataclass(frozen=True)
class OperationSpec:
    op: str
    target_ref: str | None = None
    amount: AmountFormula | None = None
    params: dict[str, Any] = field(default_factory=dict)
    emits: tuple[str, ...] = ()
    summon_policy: SummonPolicy | None = None
    transform_policy: TransformPolicy | None = None
    copy_policy: CopyPolicy | None = None


@dataclass(frozen=True)
class EffectSource:
    kind: str
    raw_text: str = ""
    clean_text: str = ""
    text_key: str = ""


@dataclass(frozen=True)
class ExecutableEffect:
    effect_id: str
    trigger: TriggerSpec
    conditions: tuple[ConditionSpec, ...] = ()
    target_selector: TargetSelector | None = None
    operations: tuple[OperationSpec, ...] = ()
    random_policy: RandomPolicy | None = None
    event_policy: EventPolicy = field(default_factory=EventPolicy)
    resolution_policy: ResolutionPolicy = field(default_factory=ResolutionPolicy)
    rollback_policy: RollbackPolicy = field(default_factory=RollbackPolicy)
    source: EffectSource = field(default_factory=lambda: EffectSource(kind="unknown"))


@dataclass(frozen=True)
class PersistentEffect:
    effect_id: str
    owner: str
    source_zone: str
    source_card_ref: int | None = None
    trigger: TriggerSpec | None = None
    conditions: tuple[ConditionSpec, ...] = ()
    per_turn_limit: int | None = None
    countdown: int | None = None
    duration: str = "while_active"
    executable_effects: tuple[ExecutableEffect, ...] = ()
    source: EffectSource = field(default_factory=lambda: EffectSource(kind="unknown"))


@dataclass(frozen=True)
class CardSourceText:
    raw_skill_desc: str = ""
    clean_skill_desc: str = ""
    normal_desc: str = ""
    evolve_desc: str = ""
    super_evolve_desc: str = ""
    supplement_texts: tuple[dict[str, Any], ...] = ()
    text_keys: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnifiedCardTemplate:
    template_version: int
    identity: CardIdentity
    printed: PrintedStats
    forms: tuple[CardFormSpec, ...] = ()
    play_modes: tuple[PlayModeSpec, ...] = ()
    static_keywords: tuple[KeywordSpec, ...] = ()
    related_cards: tuple[RelatedCardRef, ...] = ()
    executable_effects: tuple[ExecutableEffect, ...] = ()
    persistent_effects: tuple[PersistentEffect, ...] = ()
    source_text: CardSourceText = field(default_factory=CardSourceText)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedCardTemplate":
        return _from_dict(cls, data)


T = TypeVar("T")


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
