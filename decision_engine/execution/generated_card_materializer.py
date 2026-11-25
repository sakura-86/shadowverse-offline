# -*- coding: utf-8 -*-
"""Template effect generated card materialization helpers.

This module only materializes explicit generated-card inputs from effect specs.
It does not guess hidden draws or invent unsupported card identities.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from decision_engine.actions.model import LegalAction
from decision_engine.cards import CardResolver, MasterDataProvider
from decision_engine.cards.templates import ExecutableEffect, OperationSpec
from decision_engine.state.deck_instance import DeckInstanceState
from decision_engine.state.snapshot import BattleSnapshot, CardView

from .execution_context import LocalExecutionContext, RandomBranchBinding
from ..simulation.deck_pool import materialize_draw_card, next_synthetic_uid

_DEFAULT_CARD_DATABASE = Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"


@dataclass(frozen=True)
class GeneratedCardMaterialization:
    cards: tuple[CardView, ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class GeneratedCardRequirement:
    count: int
    requires_materialization: bool
    allows_partial_recorded_cards: bool
    candidate_base_ids: frozenset[int]


GENERATED_CARD_OPS = frozenset({
    "draw",
    "summon",
    "add_to_hand",
    "put_token_to_hand",
    "transform",
    "replace_hand_card",
    "copy_card",
    "copy_from_opponent_hand",
    "reanimate",
    "history_same_name_pool",
    "deck_operation",
    # 「使自己的土之印+N」在场上没有土之印时会生成一张 90031210 大地之魔片。
    # 这是 op 内部的隐式造卡：replay 的 after 帧会给出真实 uid，必须在
    # 需求表里声明，否则 op 只能造 synthetic uid（replay_...392654 st10 的 -81）。
    "modify_earth_sigil",
})

# `deck_operation` 有多个 kind，只有这两类需要携带 generated_cards：
#   add_to_deck   —— 往牌组加入指定卡
#   replace_deck  —— 把牌组整体换成指定卡
# 其余 kind（stat_buff / modify_deck_card_cost / dedupe_deck / banish_from_deck）
# 只改牌组内已有实例，不需要生成卡。按 op 名一刀切会把这些 kind 全判成
# `generated_cards_not_materializable`。
DECK_OPERATION_GENERATED_KINDS = frozenset({"add_to_deck", "replace_deck"})


def operation_needs_generated_cards(operation) -> bool:
    """该 operation 是否必须携带 generated_cards。"""
    if operation.op == "deck_operation":
        return str(operation.params.get("kind") or "") in DECK_OPERATION_GENERATED_KINDS
    return operation.op in GENERATED_CARD_OPS


@lru_cache(maxsize=1)
def default_card_resolver() -> CardResolver:
    return CardResolver(MasterDataProvider.from_json(_DEFAULT_CARD_DATABASE))


def generated_card_requirement_for_effect(effect: ExecutableEffect, action: LegalAction) -> GeneratedCardRequirement:
    count = 0
    for operation in effect.operations:
        if not _operation_mode_selected(operation, action):
            continue
        count += operation_generated_card_count(operation)
    return GeneratedCardRequirement(
        count=count,
        requires_materialization=effect_has_explicit_generated_card_source(effect),
        allows_partial_recorded_cards=effect_allows_partial_recorded_generated_cards(effect),
        candidate_base_ids=effect_candidate_generated_base_ids(effect),
    )


def operation_generated_card_count(operation: OperationSpec) -> int:
    if operation.op not in GENERATED_CARD_OPS:
        return 0
    count = _generated_card_count(operation)
    if count > 0:
        return count
    return 1


def effect_has_explicit_generated_card_source(effect: ExecutableEffect) -> bool:
    for operation in effect.operations:
        params = operation.params
        if any(key in params for key in ("generated_card_names", "generated_card_refs", "candidate_base_card_ids", "source_pool")):
            return True
        if operation.op == "summon" and params.get("copy_source") in {"selected_source_pool_card", "selected_self_hand_card"}:
            return True
        if operation.op in {"copy_card", "copy_from_opponent_hand", "put_token_to_hand", "replace_hand_card", "reanimate", "history_same_name_pool"}:
            return True
        if operation.op == "deck_operation" and operation_needs_generated_cards(operation):
            return True
        if operation.op == "modify_earth_sigil":
            # 场上没有土之印时隐式生成一张；有绑定时按录像的真实卡物化。
            return True
    return False


def effect_allows_partial_recorded_generated_cards(effect: ExecutableEffect) -> bool:
    return any(operation.op in {"reanimate", "history_same_name_pool"} for operation in effect.operations)


def effect_candidate_generated_base_ids(effect: ExecutableEffect) -> frozenset[int]:
    result: set[int] = set()
    for operation in effect.operations:
        if operation.op == "modify_earth_sigil":
            # 场上没有土之印时 op 会生成「大地之魔片」。
            result.add(90031210)
        for item in operation.params.get("candidate_base_card_ids", ()):
            try:
                result.add(int(item))
            except (TypeError, ValueError):
                continue
        source_pool = operation.params.get("source_pool")
        if isinstance(source_pool, dict):
            result.update(source_pool_candidate_ids(source_pool))
        refs = operation.params.get("generated_card_refs", ())
        if isinstance(refs, list):
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                for key in ("preferred_base_card_ids", "base_card_ids", "card_ids"):
                    for item in ref.get(key, ()):
                        try:
                            result.add(int(item))
                        except (TypeError, ValueError):
                            continue
    return frozenset(result)


def source_pool_candidate_ids(source_pool: dict[str, Any]) -> frozenset[int]:
    result: set[int] = set()
    for key in ("candidate_base_card_ids", "preferred_base_card_ids", "base_card_ids", "card_ids"):
        values = source_pool.get(key, ())
        if isinstance(values, int):
            values = (values,)
        for item in values or ():
            try:
                result.add(int(item))
            except (TypeError, ValueError):
                continue
    return frozenset(result)


def operation_candidate_base_ids(operation: Any) -> frozenset[int]:
    """单个 operation 声明的候选生成卡 base id 集合。"""
    result: set[int] = set()
    for item in (operation.params.get("candidate_base_card_ids") or ()):
        try:
            result.add(int(item))
        except (TypeError, ValueError):
            continue
    source_pool = operation.params.get("source_pool")
    if isinstance(source_pool, dict):
        result.update(source_pool_candidate_ids(source_pool))
    refs = operation.params.get("generated_card_refs", ())
    if isinstance(refs, list):
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            for key in ("preferred_base_card_ids", "base_card_ids", "card_ids"):
                for item in ref.get(key, ()):
                    try:
                        result.add(int(item))
                    except (TypeError, ValueError):
                        continue
    return frozenset(result)


def _effect_generated_zone(effect: ExecutableEffect) -> str | None:
    """该 effect 的生成卡应该落在哪个区域（replay 取证池按区域筛选用）。

    `draw` 不参与：它取牌靠 draw_bindings，生成卡池只用于「占用」避免后续
    effect 重复取同一张，按混合池原有顺序取前 N 张即可。
    """
    ops = {operation.op for operation in effect.operations}
    wants_field = "summon" in ops
    wants_hand = bool(ops & {"add_to_hand", "put_token_to_hand", "replace_hand_card"})
    if wants_field and wants_hand:
        # 同一个 effect 内先召唤、再加入手牌（10721120：召唤『骑士』90021110，
        # 再把『铁甲骑士』90021120 加入手牌）。两块卡分属不同区域，按 effect 级
        # 过滤只剩一半，add_to_hand 会拿不到 generated_cards。
        # 这类交给下面按 operation 顺序对齐的逻辑处理。
        return None
    if wants_field:
        return "field"
    if wants_hand:
        return "hand"
    # transform 不参与：变换对象可能在手牌也可能在场上，按 op 名无法判断区域，
    # 一刀切会让「把手牌变成另一张」的卡去场上池取卡并撞上
    # `transform replacement uid already exists`（10133310 / 10833310 st47）。
    return None


def select_recorded_generated_cards_for_effect(effect: ExecutableEffect, generated_cards: tuple[CardView, ...], count: int) -> tuple[CardView, ...]:
    # replay 的 after 帧取证池同时包含「抽到/加入手牌」和「召唤上场」的新卡，
    # 两者的 uid 不可互换：draw 若拿走场上那张 token，summon 就只剩一张已经在
    # 手牌里的卡，op_summon 看到 uid 已存在直接 noop（10933110 陶醉的才女：
    # 入场曲「抽2张 + 召唤1个沉溺的实验体」，本地少召唤 1 张）。
    # 本地物化的卡没有该标注（None），不参与过滤。
    wanted_zone = _effect_generated_zone(effect)
    if wanted_zone is not None:
        zone_pool = tuple(
            card for card in generated_cards
            if card.supplement_info.get("replay_generated_zone") in (None, "", wanted_zone)
        )
        if len(zone_pool) < len(generated_cards):
            generated_cards = zone_pool
    candidate_ids = effect_candidate_generated_base_ids(effect)
    if not candidate_ids:
        matched = tuple(card for card in generated_cards if _matches_effect_source_pool(effect, card))
        if matched:
            return matched[:count]
        return generated_cards[:count]
    # 录制生成物在 after 帧里按 hand→field 排列，与 spec 的 operation 执行顺序无关。
    # interpreter 按 operations 顺序消费生成卡，所以这里必须按 **operation 顺序** 对齐：
    # 每个 op 用自己声明的候选 id 从池中取第一张未使用的卡（10721120：先召唤『骑士』
    # 90021110、再把『铁甲骑士』90021120 加入手牌；不排序会把两者配反）。
    ordered: list[CardView] = []
    used: set[int] = set()
    for operation in effect.operations:
        op_ids = operation_candidate_base_ids(operation)
        if not op_ids:
            continue
        for card in generated_cards:
            if card.uid in used:
                continue
            if card.base_card_id in op_ids or card.card_id in op_ids:
                ordered.append(card)
                used.add(card.uid)
                break
    for card in generated_cards:
        if len(ordered) >= count:
            break
        if card.uid in used:
            continue
        if card.base_card_id in candidate_ids or card.card_id in candidate_ids:
            ordered.append(card)
            used.add(card.uid)
    if ordered:
        return tuple(ordered[:count])
    matched = tuple(card for card in generated_cards if card.base_card_id in candidate_ids or card.card_id in candidate_ids)
    return matched[:count]


def materialize_effect_generated_cards(
    snapshot: BattleSnapshot,
    action: LegalAction,
    effect: ExecutableEffect,
    *,
    source_card: CardView | None,
    resolver: CardResolver,
    branch_offset: int = 0,
    reserved_uids: frozenset[int] = frozenset(),
    execution_context: LocalExecutionContext = LocalExecutionContext(),
    player: str = "me",
) -> GeneratedCardMaterialization:
    cards: list[CardView] = []
    assumptions: list[str] = []
    effective_reserved = set(reserved_uids)
    for operation_index, operation in enumerate(effect.operations):
        if not _operation_mode_selected(operation, action):
            continue
        materialized = _materialize_operation_generated_cards(
            snapshot,
            action,
            operation,
            effect_id=effect.effect_id,
            operation_index=operation_index,
            source_card=source_card,
            resolver=resolver,
            branch_index=branch_offset + len(cards),
            reserved_uids=frozenset(effective_reserved),
            execution_context=execution_context,
            player=player,
        )
        if materialized.unresolved_reason is not None:
            return GeneratedCardMaterialization(
                assumptions=tuple(dict.fromkeys((*assumptions, *materialized.assumptions))),
                unresolved_reason=materialized.unresolved_reason,
            )
        cards.extend(materialized.cards)
        effective_reserved.update(card.uid for card in materialized.cards)
        assumptions.extend(materialized.assumptions)
    return GeneratedCardMaterialization(
        cards=tuple(cards),
        assumptions=tuple(dict.fromkeys(assumptions)),
    )


def _operation_mode_selected(operation: OperationSpec, action: LegalAction) -> bool:
    mode_option = operation.params.get("mode_option")
    if mode_option is None:
        return True
    if not isinstance(mode_option, dict):
        return True
    option_index = mode_option.get("option_index")
    if option_index is None:
        return True
    try:
        return int(option_index) in action.selected_indexes
    except (TypeError, ValueError):
        return False


def _materialize_operation_generated_cards(
    snapshot: BattleSnapshot,
    action: LegalAction,
    operation: OperationSpec,
    *,
    effect_id: str,
    operation_index: int,
    source_card: CardView | None,
    resolver: CardResolver,
    branch_index: int,
    reserved_uids: frozenset[int],
    execution_context: LocalExecutionContext,
    player: str = "me",
) -> GeneratedCardMaterialization:
    op = operation.op
    if op == "draw":
        return GeneratedCardMaterialization()
    if op == "copy_from_opponent_hand" and not operation.target_ref:
        uid = _next_available_synthetic_uid(snapshot, branch_index, set(snapshot.cards_by_uid) | set(reserved_uids))
        return GeneratedCardMaterialization(cards=(CardView(uid=uid),), assumptions=("generated_cards_materialized:copy_from_opponent_hand_unknown",))
    if op in {"copy_card", "copy_from_opponent_hand"}:
        copied = _copy_target_card(snapshot, action, operation, source_card=source_card, branch_index=branch_index, reserved_uids=reserved_uids)
        if copied is None:
            return GeneratedCardMaterialization(unresolved_reason=f"generated_cards_not_materializable:{op}")
        return GeneratedCardMaterialization(cards=(copied,), assumptions=(f"generated_cards_materialized:{op}",))
    if op == "summon" and operation.params.get("copy_source") in {"selected_source_pool_card", "selected_self_hand_card"}:
        copied_cards = _copy_target_cards(snapshot, action, operation, source_card=source_card, count=_generated_card_count(operation), branch_index=branch_index, reserved_uids=reserved_uids)
        if copied_cards is None:
            return GeneratedCardMaterialization(unresolved_reason="generated_cards_not_materializable:summon_copy_source")
        return GeneratedCardMaterialization(cards=copied_cards, assumptions=("generated_cards_materialized:summon_copy_source",))

    count = _generated_card_count(operation)
    if count <= 0:
        count = 1
    explicit = _materialize_explicit_generated_cards(snapshot, action, operation, source_card=source_card, resolver=resolver, count=count, branch_index=branch_index, reserved_uids=reserved_uids)
    if explicit is not None:
        return explicit

    pool_materialization = _materialize_from_source_pool(snapshot, action, operation, effect_id=effect_id, operation_index=operation_index, source_card=source_card, resolver=resolver, count=count, branch_index=branch_index, reserved_uids=reserved_uids, execution_context=execution_context)
    if pool_materialization is not None:
        return pool_materialization

    history_materialization = _materialize_from_history(snapshot, operation, resolver=resolver, count=count, branch_index=branch_index, reserved_uids=reserved_uids, player=player)
    if history_materialization is not None:
        return history_materialization

    if op in {"add_to_hand", "put_token_to_hand", "summon", "transform", "replace_hand_card", "reanimate", "history_same_name_pool"} or (
        op == "deck_operation" and operation_needs_generated_cards(operation)
    ):
        return GeneratedCardMaterialization(unresolved_reason=f"generated_cards_not_materializable:{op}")
    return GeneratedCardMaterialization()


def _generated_card_count(operation: OperationSpec) -> int:
    params = operation.params
    if "generated_card_count" in params:
        try:
            return int(params.get("generated_card_count") or 0)
        except (TypeError, ValueError):
            return 0
    # reanimate/history_same_name_pool 的 amount.value 表示费用/阈值上限(max_cost),
    # 不是召唤数量。这类 op 默认召唤 1 个,只有显式 generated_card_count/count 才多召。
    # 参见 docs/reference/local-rule-spec.md §13.10 亡者召还: "选择费用最高的候选;
    # 同费用内随机抽取"——单数召唤。
    if operation.op in {"reanimate", "history_same_name_pool"}:
        for key in ("count", "selection_count"):
            value = params.get(key)
            if value is not None and not isinstance(value, dict):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return 1
    if operation.amount is not None and operation.amount.kind == "fixed" and operation.amount.value is not None:
        try:
            return int(operation.amount.value)
        except (TypeError, ValueError):
            return 0
    for key in ("count", "selection_count"):
        value = params.get(key)
        if value is not None and not isinstance(value, dict):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    source_pool = params.get("source_pool")
    if isinstance(source_pool, dict):
        value = source_pool.get("count")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    names = params.get("generated_card_names")
    if isinstance(names, list) and names:
        return len(names)
    refs = params.get("generated_card_refs")
    if isinstance(refs, list) and refs:
        return len(refs)
    return 0


def _materialize_explicit_generated_cards(
    snapshot: BattleSnapshot,
    action: LegalAction,
    operation: OperationSpec,
    *,
    source_card: CardView | None,
    resolver: CardResolver,
    count: int,
    branch_index: int,
    reserved_uids: frozenset[int],
) -> GeneratedCardMaterialization | None:
    params = operation.params
    refs = params.get("generated_card_refs")
    if isinstance(refs, list) and refs:
        ref_cards = _materialize_cards_from_refs(snapshot, resolver, refs, count=count, branch_index=branch_index, reserved_uids=reserved_uids, source_card=source_card)
        if ref_cards is not None:
            return ref_cards

    names = params.get("generated_card_names")
    if isinstance(names, list) and names:
        resolved_card_id = _resolve_card_id_from_names(names, params.get("candidate_base_card_ids"), source_card=source_card, resolver=resolver)
        if resolved_card_id is None:
            return GeneratedCardMaterialization(unresolved_reason="generated_cards_not_materializable:ambiguous_card_names")
        return _materialize_repeated_card(snapshot, resolver, resolved_card_id, count=count, branch_index=branch_index, reserved_uids=reserved_uids, assumption_key="generated_cards_materialized:card_names")

    candidate_ids = params.get("candidate_base_card_ids")
    if isinstance(candidate_ids, list) and candidate_ids:
        resolved_card_id = _resolve_card_id_from_ids(candidate_ids, source_card=source_card)
        if resolved_card_id is None:
            return GeneratedCardMaterialization(unresolved_reason="generated_cards_not_materializable:ambiguous_candidate_base_card_ids")
        return _materialize_repeated_card(snapshot, resolver, resolved_card_id, count=count, branch_index=branch_index, reserved_uids=reserved_uids, assumption_key="generated_cards_materialized:candidate_base_card_ids")

    return None


def _materialize_from_source_pool(
    snapshot: BattleSnapshot,
    action: LegalAction,
    operation: OperationSpec,
    *,
    effect_id: str,
    operation_index: int,
    source_card: CardView | None,
    resolver: CardResolver,
    count: int,
    branch_index: int,
    reserved_uids: frozenset[int],
    execution_context: LocalExecutionContext,
) -> GeneratedCardMaterialization | None:
    source_pool = operation.params.get("source_pool")
    if not isinstance(source_pool, dict):
        return None
    selection = str(source_pool.get("selection", "") or "")
    if selection not in {"selected", "all", "random"}:
        return None
    selected_targets = tuple(uid for group in action.target_groups for uid in group)
    if selection == "selected":
        pool_cards = tuple(card for uid in selected_targets if (card := snapshot.cards_by_uid.get(uid)) is not None)
    else:
        pool_cards = _source_pool_cards(snapshot, source_pool)
        if selection == "random":
            if selected_targets:
                selected_set = set(selected_targets)
                pool_cards = tuple(card for card in pool_cards if card.uid in selected_set)
            elif branch := _matching_random_branch_binding(execution_context.random_branch_bindings, effect_id, operation_index, pool_cards, count):
                pool_cards = branch.cards
            elif len(pool_cards) != count:
                # 用 branch_seed 直接采样（与 _deterministic_random_targets 对称）
                if execution_context.branch_seed is not None and pool_cards:
                    import random as _random
                    sampled = _random.Random(execution_context.branch_seed + operation_index).sample(list(pool_cards), min(count, len(pool_cards)))
                    pool_cards = tuple(sampled)
                    # 构造等价的 RandomBranchBinding 以统一 assumption 格式
                    execution_context = replace(execution_context, random_branch_bindings=(
                        *execution_context.random_branch_bindings,
                        RandomBranchBinding(
                            source="source_pool",
                            effect_id=effect_id,
                            operation_index=operation_index,
                            cards=pool_cards,
                            assumption=f"random_branch:source_pool:seed={execution_context.branch_seed}:effect={effect_id}:operation={operation_index}",
                        ),
                    ))
                else:
                    # 无 branch_seed 时用确定性近似：取前 count 个
                    pool_cards = pool_cards[:count]
    if not pool_cards:
        # source_pool 解析成功但候选为空（如牌组里没有满足条件的卡）
        # 这是合法的 noop：效果不触发，不是物化失败
        return GeneratedCardMaterialization(cards=(), assumptions=(f"generated_cards_materialized:source_pool:empty_pool:{selection}",))
    copied_cards: list[CardView] = []
    used_uids = set(snapshot.cards_by_uid) | set(reserved_uids)
    assumptions = ["generated_cards_materialized:source_pool", f"generated_cards_materialized:source_pool:{selection}"]
    if selection == "random":
        branch = _matching_random_branch_binding(execution_context.random_branch_bindings, effect_id, operation_index, pool_cards, count)
        if branch is not None:
            assumptions.append(branch.assumption)
    for index, card in enumerate(pool_cards[:count]):
        copied_cards.append(replace(card, uid=_next_available_synthetic_uid(snapshot, branch_index + index, used_uids)))
        used_uids.add(copied_cards[-1].uid)
    # 当 pool 不足 count 时（如 distinct 过滤后候选不够），使用所有可用的
    return GeneratedCardMaterialization(cards=tuple(copied_cards), assumptions=tuple(assumptions))


def source_pool_cards(snapshot: BattleSnapshot, source_pool: dict[str, Any]) -> tuple[CardView, ...]:
    return _source_pool_cards(snapshot, source_pool)


def _matching_random_branch_binding(
    bindings: tuple[RandomBranchBinding, ...],
    effect_id: str,
    operation_index: int,
    pool_cards: tuple[CardView, ...],
    count: int,
) -> RandomBranchBinding | None:
    pool_ids = {(card.card_id, card.base_card_id) for card in pool_cards}
    pool_uids = {card.uid for card in pool_cards}
    for binding in bindings:
        if binding.source != "source_pool" or binding.effect_id != effect_id or binding.operation_index != operation_index:
            continue
        if len(binding.cards) != count:
            continue
        if len({card.base_card_id for card in binding.cards}) != len(binding.cards):
            continue
        if all(card.uid in pool_uids or (card.card_id, card.base_card_id) in pool_ids for card in binding.cards):
            return binding
    return None


def _matches_effect_source_pool(effect: ExecutableEffect, card: CardView) -> bool:
    pools = tuple(operation.params.get("source_pool") for operation in effect.operations if isinstance(operation.params.get("source_pool"), dict))
    if not pools:
        return False
    return any(_card_matches_source_pool(card, pool) for pool in pools)


def _card_matches_source_pool(card: CardView, source_pool: dict[str, Any]) -> bool:
    card_type = source_pool.get("card_type")
    if card_type is not None and card.card_type.value != str(card_type):
        return False
    max_cost = source_pool.get("max_cost")
    if max_cost is not None:
        try:
            if card.cost > int(max_cost):
                return False
        except (TypeError, ValueError):
            return False
    candidate_ids = source_pool_candidate_ids(source_pool)
    if candidate_ids and card.base_card_id not in candidate_ids and card.card_id not in candidate_ids:
        return False
    filters = tuple(source_pool.get("filters", ()) or ())
    return all(_card_matches_generated_filter(card, item) for item in filters)


def _card_matches_generated_filter(card: CardView, item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    field = item.get("field")
    operator = str(item.get("operator", "==") or "==")
    value = item.get("value")
    if field == "card_type":
        return _compare_filter_value(card.card_type.value, operator, value)
    if field in {"cost", "original_cost"}:
        return _compare_filter_value(card.cost, operator, value)
    if field == "base_card_id":
        return _compare_filter_value(card.base_card_id, operator, value)
    if field == "card_id":
        return _compare_filter_value(card.card_id, operator, value)
    if field == "has_bane":
        # 「拥有【毁灭】」检索条件；协议缺省时由模板静态关键字补全（见 mpo_adapter）。
        return bool(card.has_bane) == bool(value)
    return False


def _compare_filter_value(current: Any, operator: str, expected: Any) -> bool:
    if operator == "==":
        return current == expected
    if operator == "!=":
        return current != expected
    if operator == "<=":
        return current <= expected
    if operator == ">=":
        return current >= expected
    if operator == "<":
        return current < expected
    if operator == ">":
        return current > expected
    if operator == "in":
        return current in expected
    return False


def _source_pool_cards(snapshot: BattleSnapshot, source_pool: dict[str, Any]) -> tuple[CardView, ...]:
    side = str(source_pool.get("side", "self") or "self")
    zone = str(source_pool.get("zone", "field") or "field")
    if zone not in {"field", "hand", "deck"}:
        return ()
    players: tuple[str, ...]
    if side in {"self", "me", "ally"}:
        players = ("me",)
    elif side in {"enemy", "opponent"}:
        players = ("opponent",)
    elif side == "both":
        players = ("me", "opponent")
    else:
        return ()
    cards: list[CardView] = []
    for player in players:
        view = snapshot.me if player == "me" else snapshot.opponent
        if zone == "deck":
            cards.extend(_deck_source_pool_cards(view.deck_instances))
            continue
        cards.extend(snapshot.cards_by_uid[uid] for uid in getattr(view, zone) if uid in snapshot.cards_by_uid)
    card_type = source_pool.get("card_type")
    if card_type is not None:
        cards = [card for card in cards if card.card_type.value == str(card_type)]
    max_cost = source_pool.get("max_cost")
    if max_cost is not None:
        try:
            max_cost_value = int(max_cost)
        except (TypeError, ValueError):
            return ()
        cards = [card for card in cards if card.cost <= max_cost_value]
    filters = tuple(source_pool.get("filters", ()) or ())
    if filters:
        cards = [card for card in cards if _generated_card_matches_filters(card, filters)]
    if source_pool.get("distinct"):
        by_base_id: dict[int, CardView] = {}
        for card in cards:
            by_base_id.setdefault(card.base_card_id, card)
        cards = list(by_base_id.values())
    return tuple(cards)


def _deck_source_pool_cards(deck_instances: DeckInstanceState | None) -> tuple[CardView, ...]:
    if deck_instances is None or deck_instances.status not in {"available", "ambiguous"}:
        return ()
    by_id = deck_instances.by_instance_id()
    cards: list[CardView] = []
    for instance_id in deck_instances.draw_pile:
        instance = by_id.get(instance_id)
        if instance is None or instance.zone != "deck":
            continue
        uid = instance.uid if instance.uid is not None else -instance.instance_id
        cards.append(replace(instance.card, uid=uid))
    return tuple(cards)


def _materialize_from_history(
    snapshot: BattleSnapshot,
    operation: OperationSpec,
    *,
    resolver: CardResolver,
    count: int,
    branch_index: int,
    reserved_uids: frozenset[int],
    player: str = "me",
) -> GeneratedCardMaterialization | None:
    if operation.op not in {"history_same_name_pool", "reanimate"}:
        return None
    source_event = operation.params.get("source_event", "destroyed")
    if source_event != "destroyed":
        return None
    # destroyed history 属于效果归属方(player): reanimate / history_same_name_pool 从
    # "自己被破坏的历史"物化候选。interpreter 的执行期校验(_history_same_name_pool /
    # _reanimate)同样按 context.player 读取该方 destroyed_card_ids, 两处必须使用同一方,
    # 否则对手谢幕曲触发时物化(me)与校验(opponent)错配, 抛
    # "generated card is not in destroyed history"。
    player_view = snapshot.me if player == "me" else snapshot.opponent
    destroyed_items = player_view.destroyed_card_ids
    if not destroyed_items:
        return GeneratedCardMaterialization(cards=(), assumptions=(f"generated_cards_materialized:{operation.op}_history_empty",))
    max_cost = None
    if operation.op == "reanimate" and operation.amount is not None and operation.amount.kind == "fixed" and operation.amount.value is not None:
        max_cost = int(operation.amount.value)
    card_type = operation.params.get("card_type")
    filters = tuple(operation.params.get("filters", ()) or ())
    distinct = bool(operation.params.get("distinct", False))
    candidates: list[CardView] = []
    seen_base_ids: set[int] = set()
    used_uids = set(snapshot.cards_by_uid) | set(reserved_uids)
    for item in destroyed_items:
        card = materialize_draw_card(int(item.card_id), snapshot, uid=_next_available_synthetic_uid(snapshot, branch_index + len(candidates), used_uids), resolver=resolver)
        if card is None:
            continue
        if max_cost is not None:
            base = resolver.get_base_by_card_id(card.card_id)
            if base is not None and base.cost > max_cost:
                continue
        if card_type is not None:
            # "amulet" 涵盖 countdown_amulet（吟唱护符也是护符）。
            if str(card_type) == "amulet":
                if not card.card_type.is_amulet_like:
                    continue
            elif card.card_type.value != str(card_type):
                continue
        if filters and not _generated_card_matches_filters(card, filters):
            continue
        if distinct and card.base_card_id in seen_base_ids:
            # 「随机N种……同名各1个」：按同名去重，每种只召唤 1 个。
            continue
        candidates.append(card)
        seen_base_ids.add(card.base_card_id)
        if len(candidates) == count:
            return GeneratedCardMaterialization(cards=tuple(candidates), assumptions=(f"generated_cards_materialized:{operation.op}_history",))
    if candidates:
        # 部分满足：pool 不够 count 时返回所有可用的
        return GeneratedCardMaterialization(cards=tuple(candidates), assumptions=(f"generated_cards_materialized:{operation.op}_history_partial",))
    return GeneratedCardMaterialization(cards=(), assumptions=(f"generated_cards_materialized:{operation.op}_history_no_match",))


def _generated_card_matches_filters(card: CardView, filters: tuple[Any, ...]) -> bool:
    for item in filters:
        if not isinstance(item, dict):
            return False
        field = item.get("field")
        operator = str(item.get("operator", "==") or "==")
        value = item.get("value")
        if field == "card_type":
            # "amulet" 语义应涵盖倒计数护符(countdown_amulet): 吟唱护符也是"护符"。
            # 直接字符串比较会漏掉 countdown_amulet，导致护符教 history pool 筛选失败。
            if operator in {"==", "="} and str(value) == "amulet":
                if not card.card_type.is_amulet_like:
                    return False
            elif not _compare_value(card.card_type.value, operator, value):
                return False
        if field == "tribes" and not _compare_value(card.tribes, operator, value):
            return False
        if field in {"cost", "original_cost"} and not _compare_value(card.cost, operator, value):
            return False
        if field == "class_name" and not _compare_value(_card_class_name(card), operator, value):
            return False
        if field == "has_last_words" and not _compare_value(1 if card.has_last_words else 0, operator, 1 if value else 0):
            return False
    return True


def _card_class_name(card: CardView) -> str:
    """取卡牌职业名（filtered_pool 的 class_name filter 用）。

    优先 supplement_info.class_name，回退到 CardResolver 的 craft_class；
    与 target_selector_materializer / effect_interpreter 的 class_name 解析保持一致。
    """
    value = card.supplement_info.get("class_name", "") if card.supplement_info else ""
    if value:
        return str(value)
    base = default_card_resolver().get_base(card.base_card_id)
    return base.craft_class if base is not None else ""


def _copy_target_card(
    snapshot: BattleSnapshot,
    action: LegalAction,
    operation: OperationSpec,
    *,
    source_card: CardView | None,
    branch_index: int,
    reserved_uids: frozenset[int],
) -> CardView | None:
    copied = _copy_target_cards(snapshot, action, operation, source_card=source_card, count=1, branch_index=branch_index, reserved_uids=reserved_uids)
    return copied[0] if copied else None


def _copy_target_cards(
    snapshot: BattleSnapshot,
    action: LegalAction,
    operation: OperationSpec,
    *,
    source_card: CardView | None,
    count: int,
    branch_index: int,
    reserved_uids: frozenset[int],
) -> tuple[CardView, ...] | None:
    target_uids = tuple(uid for group in action.target_groups for uid in group)
    if not target_uids and source_card is not None:
        target_uids = (source_card.uid,)
    if not target_uids:
        target_uid = _int_param(operation.params.get("target_uid"))
        if target_uid is not None:
            target_uids = (target_uid,)
    if not target_uids:
        raw_target_uids = operation.params.get("target_uids")
        if isinstance(raw_target_uids, (list, tuple)):
            target_uids = tuple(uid for item in raw_target_uids if (uid := _int_param(item)) is not None)
    if count <= 0:
        count = len(target_uids)
    if len(target_uids) < count:
        return None
    cards: list[CardView] = []
    used_uids = set(snapshot.cards_by_uid) | set(reserved_uids)
    for index, target_uid in enumerate(target_uids[:count]):
        target = snapshot.cards_by_uid.get(target_uid)
        if target is None:
            return None
        cards.append(replace(target, uid=_next_available_synthetic_uid(snapshot, branch_index + index, used_uids)))
    return tuple(cards)


def _materialize_cards_from_refs(
    snapshot: BattleSnapshot,
    resolver: CardResolver,
    refs: list[dict[str, Any]],
    *,
    count: int,
    branch_index: int,
    reserved_uids: frozenset[int],
    source_card: CardView | None,
) -> GeneratedCardMaterialization | None:
    refs_to_resolve = refs * count if len(refs) == 1 else refs[:count]
    resolved_ids: list[int] = []
    for ref in refs_to_resolve:
        if not isinstance(ref, dict):
            return None
        resolved = _resolve_card_id_from_refs([ref], source_card=source_card)
        if resolved is None:
            return None
        resolved_ids.append(resolved)
    if len(resolved_ids) != count:
        return None
    cards: list[CardView] = []
    used_uids = set(snapshot.cards_by_uid) | set(reserved_uids)
    for index, base_card_id in enumerate(resolved_ids):
        uid = _next_available_synthetic_uid(snapshot, branch_index + index, used_uids)
        card = materialize_draw_card(base_card_id, snapshot, uid=uid, resolver=resolver)
        if card is None:
            return GeneratedCardMaterialization(unresolved_reason=f"generated_cards_not_materializable:{base_card_id}")
        cards.append(card)
    return GeneratedCardMaterialization(cards=tuple(cards), assumptions=("generated_cards_materialized:card_refs",))


def _resolve_card_id_from_refs(refs: list[dict[str, Any]], *, source_card: CardView | None) -> int | None:
    preferred: list[int] = []
    candidates: list[int] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        base_ids = ref.get("preferred_base_card_ids") or ref.get("base_card_ids") or ()
        if not base_ids:
            # 兼容单数 base_card_id 写法
            single = ref.get("base_card_id")
            if single is not None:
                base_ids = [single]
        ids = [_int_param(card_id) for card_id in base_ids if _int_param(card_id) is not None]
        if not ids:
            continue
        if source_card is not None and source_card.base_card_id in ids:
            return source_card.base_card_id
        if len(ids) == 1:
            preferred.append(ids[0])
        candidates.extend(ids)
    unique_preferred = sorted(set(preferred))
    if len(unique_preferred) == 1:
        return unique_preferred[0]
    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    return None


def _resolve_card_id_from_names(
    names: list[str],
    candidate_base_card_ids: Any,
    *,
    source_card: CardView | None,
    resolver: CardResolver,
) -> int | None:
    candidate_ids = [_int_param(card_id) for card_id in candidate_base_card_ids or () if _int_param(card_id) is not None]
    if source_card is not None and source_card.base_card_id in candidate_ids:
        return source_card.base_card_id
    if len(candidate_ids) == 1:
        return candidate_ids[0]
    if len(names) == 1:
        base = _match_base_card_by_name(names[0], resolver, candidate_ids=tuple(candidate_ids))
        if base is not None:
            return base.base_card_id
    if candidate_ids:
        return candidate_ids[0]
    return None


def _resolve_card_id_from_ids(candidate_base_card_ids: list[Any], *, source_card: CardView | None) -> int | None:
    ids = [_int_param(card_id) for card_id in candidate_base_card_ids if _int_param(card_id) is not None]
    if source_card is not None and source_card.base_card_id in ids:
        return source_card.base_card_id
    unique_ids = sorted(set(ids))
    if len(unique_ids) == 1:
        return unique_ids[0]
    if unique_ids:
        return unique_ids[0]
    return None


def _materialize_repeated_card(
    snapshot: BattleSnapshot,
    resolver: CardResolver,
    base_card_id: int,
    *,
    count: int,
    branch_index: int,
    reserved_uids: frozenset[int],
    assumption_key: str,
) -> GeneratedCardMaterialization:
    base = resolver.get_base(base_card_id)
    if base is None:
        return GeneratedCardMaterialization(unresolved_reason=f"generated_cards_not_materializable:{base_card_id}")
    card_id = base.base_card_id
    cards: list[CardView] = []
    used_uids = set(snapshot.cards_by_uid) | set(reserved_uids)
    for index in range(count):
        uid = _next_available_synthetic_uid(snapshot, branch_index + index, used_uids)
        card = materialize_draw_card(card_id, snapshot, uid=uid, resolver=resolver)
        if card is None:
            return GeneratedCardMaterialization(unresolved_reason=f"generated_cards_not_materializable:{base_card_id}")
        cards.append(card)
    return GeneratedCardMaterialization(cards=tuple(cards), assumptions=(assumption_key,))


def _match_base_card_by_name(name: str, resolver: CardResolver, *, candidate_ids: tuple[int, ...] = ()):
    normalized = _normalize_name(name)
    master_data = getattr(resolver, "master_data", None)
    base_cards = getattr(master_data, "_base_cards", None)
    if not isinstance(base_cards, dict):
        return None
    pool = (base_cards.get(card_id) for card_id in candidate_ids) if candidate_ids else base_cards.values()
    matches = [base for base in pool if base is not None and getattr(base, "name", "") and _normalize_name(getattr(base, "name", "")) == normalized]
    if len(matches) == 1:
        return matches[0]
    return None


def _first_selected_target(action: LegalAction) -> int | None:
    for group in action.target_groups:
        for uid in group:
            return uid
    return None


def _compare_value(left: Any, operator: str, right: Any) -> bool:
    if operator in {"==", "=", "eq"}:
        return left == right
    if operator in {"!=", "ne"}:
        return left != right
    if operator in {"<=", "<", ">=", ">"}:
        # 数值范围比较（如 cost <= 2）。仅在两侧可比较为数值时生效，避免 tuple/None 抛错。
        left_num = _int_param(left)
        right_num = _int_param(right)
        if left_num is None or right_num is None:
            return False
        if operator == "<=":
            return left_num <= right_num
        if operator == "<":
            return left_num < right_num
        if operator == ">=":
            return left_num >= right_num
        return left_num > right_num
    if operator == "contains":
        if isinstance(left, tuple):
            try:
                return int(right) in tuple(int(item) for item in left)
            except (TypeError, ValueError):
                return right in left
        return False
    if operator == "not_contains":
        return not _compare_value(left, "contains", right)
    return False


def _int_param(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _next_available_synthetic_uid(snapshot: BattleSnapshot, branch_index: int, used_uids: set[int]) -> int:
    uid = next_synthetic_uid(snapshot, branch_index)
    while uid in used_uids:
        uid -= 1
    used_uids.add(uid)
    return uid


def _normalize_name(value: str) -> str:
    return value.replace("纹章：", "").replace("信仰：", "").strip()
