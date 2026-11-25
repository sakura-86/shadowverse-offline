# -*- coding: utf-8 -*-
"""Template effect target-selector materialization helpers.

This module only expands explicit selector metadata into concrete target groups.
It does not guess hidden targets or invent random evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.cards.effect_specs import EffectSpec, EffectSpecTarget
from decision_engine.execution.generated_card_materializer import default_card_resolver
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView


@dataclass(frozen=True)
class TargetGroupMaterialization:
    target_groups: tuple[tuple[int, ...], ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved_reason: str | None = None


def materialize_effect_target_groups(
    snapshot: BattleSnapshot,
    action: LegalAction,
    effect: EffectSpec,
    *,
    source_card: CardView | None = None,
    player: str = "me",
    opponent: str = "opponent",
) -> TargetGroupMaterialization:
    selector = effect.target_selector
    if selector is None:
        operation_groups = _operation_random_target_groups(snapshot, action, effect, player=player, opponent=opponent)
        if operation_groups is not None:
            return TargetGroupMaterialization(target_groups=operation_groups, assumptions=(f"target_selector_materialized:{effect.effect_id}",))
        if _has_selected_random_target_operation(effect, action):
            non_empty_groups = tuple(group for group in action.target_groups if group)
            if non_empty_groups:
                return TargetGroupMaterialization(target_groups=non_empty_groups, assumptions=("recorded_targets_preserved",))
            return TargetGroupMaterialization(unresolved_reason=f"target_selector_not_materializable:{effect.effect_id}")
        # operations 中有 selected_* target_ref 但 spec.target 为 None 时，
        # 合成 implicit selector 进行物化（常见于 evolve/super_evolve spec）
        implicit_selector = _synthesize_implicit_selector(effect, action)
        if implicit_selector is not None:
            if action.target_groups:
                pools = _candidate_uids(snapshot, implicit_selector, source_card=source_card, player=player, opponent=opponent)
                if pools is not None:
                    allowed = set(pools)
                    preserved = tuple(group for group in action.target_groups if group and all(uid in allowed for uid in group))
                    if preserved:
                        return TargetGroupMaterialization(target_groups=preserved, assumptions=("recorded_targets_preserved",))
            pools = _candidate_uids(snapshot, implicit_selector, source_card=source_card, player=player, opponent=opponent)
            if pools is None:
                return TargetGroupMaterialization(unresolved_reason=f"target_selector_not_materializable:{effect.effect_id}")
            if not pools:
                return TargetGroupMaterialization(target_groups=((),), assumptions=(f"target_selector_materialized:{effect.effect_id}:empty_pool",))
            count = implicit_selector.count or 1
            if count <= 1:
                groups = tuple((uid,) for uid in pools)
            else:
                if len(pools) < count:
                    return TargetGroupMaterialization(unresolved_reason=f"target_selector_not_materializable:{effect.effect_id}")
                groups = tuple(tuple(group) for group in combinations(pools, count))
            return TargetGroupMaterialization(target_groups=_dedupe_groups(groups), assumptions=(f"target_selector_materialized:{effect.effect_id}",))
        return TargetGroupMaterialization()
    if effect.random_policy is not None:
        return TargetGroupMaterialization(unresolved_reason=f"target_selector_not_materializable:{effect.effect_id}")
    is_random_selector = selector.scope == "random" or selector.kind.startswith("random_") or selector.kind == "random"
    is_all_selector = selector.kind.startswith("all_") or selector.kind in {
        "all_filtered_cards",
        "enemy_followers_all",
        "enemy_board_and_leader",
    }
    if action.target_groups and not is_random_selector:
        pools = _candidate_uids(snapshot, selector, source_card=source_card, player=player, opponent=opponent)
        if pools is not None:
            if action.kind is ActionKind.PLAY and selector.kind == "selected_self_hand_card":
                pools = tuple(uid for uid in pools if uid != action.source_uid)
            allowed = set(pools)
            preserved = tuple(group for group in action.target_groups if group and all(uid in allowed for uid in group))
            if preserved:
                return TargetGroupMaterialization(target_groups=preserved, assumptions=("recorded_targets_preserved",))
        if selector is not None and selector.kind in {"selected_enemy_follower_or_leader", "selected_follower_or_leader"}:
            return TargetGroupMaterialization(target_groups=action.target_groups, assumptions=("recorded_targets_preserved",))
    if is_random_selector:
        # 录制目标优先：replay 给出的随机目标（action.random_target_groups）若都在
        # 合法候选池内，直接沿用；否则多候选会被"取第一分支"选中错误目标。
        # 必须先于 _operation_random_target_groups：后者按候选池自行组组，会把
        # 录像里真实命中的目标丢掉（10544110 回合结束随机 8 点伤害即因此空转）。
        if action.random_target_groups:
            recorded_pools = _candidate_uids(snapshot, selector, source_card=source_card, player=player, opponent=opponent)
            if recorded_pools is not None:
                allowed = set(recorded_pools)
                recorded = tuple(
                    group for group in action.random_target_groups
                    if group and all(uid in allowed for uid in group)
                )
                if recorded:
                    return TargetGroupMaterialization(target_groups=recorded, assumptions=("recorded_random_targets_preserved",))
        operation_groups = _operation_random_target_groups(snapshot, action, effect, player=player, opponent=opponent)
        # 多随机池 effect（如 10954112 纹章回合结束「随机破坏己方谢幕曲随从 + 随机破坏
        # 敌方随从」）里，某个 op 的候选池为空不应否决整条 effect：op 级空池说明
        # 那一个 op 无事可做，其它 op 仍按 selector 描述的池正常物化。
        # 只有 selector 自身的池也为空时才落到下面的空池分支（effect 不触发）。
        if operation_groups is not None and operation_groups != ((),):
            return TargetGroupMaterialization(target_groups=operation_groups, assumptions=(f"target_selector_materialized:{effect.effect_id}",))

    pools = _candidate_uids(snapshot, selector, source_card=source_card, player=player, opponent=opponent)
    if pools is None:
        return TargetGroupMaterialization(unresolved_reason=f"target_selector_not_materializable:{effect.effect_id}")
    if action.kind is ActionKind.PLAY and selector.kind == "selected_self_hand_card":
        pools = tuple(uid for uid in pools if uid != action.source_uid)
    if not pools:
        if selector.optional or is_all_selector or selector.scope in {"selected", "random"} or selector.kind.startswith(("selected_", "random_", "all_")):
            return TargetGroupMaterialization(target_groups=((),), assumptions=(f"target_selector_materialized:{effect.effect_id}:empty_pool",))
        return TargetGroupMaterialization(unresolved_reason=f"target_selector_not_materializable:{effect.effect_id}")

    if is_all_selector:
        return TargetGroupMaterialization(target_groups=(tuple(pools),), assumptions=(f"target_selector_materialized:{effect.effect_id}",))

    # 注意：这里不能用 `count is None` 判断「内部引用」。TargetSelector.count 的默认值
    # 就是 None，spec 顶层 target 与测试/模板构造的 selector 都走这条通用路径，
    # 语义是「默认选 1 个」。spec 顶层 target 的 count=None（内部引用）由
    # LocalActionEnumerator._expand_effect_target 拦截，不会走到这里生成动作。
    count = selector.count or 1
    if count <= 1:
        groups = tuple((uid,) for uid in pools)
    else:
        if len(pools) < count:
            return TargetGroupMaterialization(unresolved_reason=f"target_selector_not_materializable:{effect.effect_id}")
        groups = tuple(tuple(group) for group in combinations(pools, count))

    if selector.optional:
        groups = (*groups, ())
    if not groups:
        return TargetGroupMaterialization(unresolved_reason=f"target_selector_not_materializable:{effect.effect_id}")
    return TargetGroupMaterialization(target_groups=_dedupe_groups(groups), assumptions=(f"target_selector_materialized:{effect.effect_id}",))


def _candidate_uids(
    snapshot: BattleSnapshot,
    selector: EffectSpecTarget,
    *,
    source_card: CardView | None,
    player: str,
    opponent: str,
) -> tuple[int, ...] | None:
    side = selector.side
    zone = selector.zone
    card_type = selector.card_type
    # Player/leader selectors: 效果作用于玩家本身而非具体卡牌，返回 sentinel (0,)
    if selector.kind in {"self_player", "self_leader", "enemy_leader", "enemy_player", "opponent_leader", "opponent_player"}:
        return (0,)
    # all_leaders: 作用于双方主战者，实际由 combat_ops 的 all_leaders 分支处理，
    # 选择器只需返回非空 sentinel 避免"空目标=效果不触发"被 skip。
    if selector.kind == "all_leaders":
        return (0, 0)
    if side is None:
        if selector.kind.startswith("selected_enemy_") or selector.kind in {"enemy_follower", "selected_opponent_hand_card", "combat_opponent"}:
            side = "enemy"
        elif selector.kind.startswith("selected_self_") or selector.kind in {"selected_target", "selected_filtered_cards"}:
            side = "self"
        elif selector.kind in {"selected_any_card", "selected_any_follower", "selected_follower_or_leader"}:
            side = "both"
    include_leader = selector.kind in {"selected_enemy_follower_or_leader", "selected_follower_or_leader"}
    if zone is None:
        if selector.kind.endswith("hand_card") or selector.kind in {"all_self_hand_cards", "all_enemy_hand_cards"}:
            zone = "hand"
        elif selector.kind == "all_crests":
            zone = "crest"
        else:
            zone = "field"
    if side not in {None, "self", "enemy", "opponent", "both"}:
        return None
    if zone not in {"field", "hand", "crest"}:
        return None

    # Crest zone: add_crest/remove_crest/modify_crest operations are self-contained
    # (they use operation.params, not target selection). Return a sentinel (0,) so
    # the 'all_*' path produces a non-empty target_groups, preventing the
    # "empty target = effect doesn't fire" skip in _apply_template_effects.
    if zone == "crest":
        return (0,)

    players: tuple[str, ...]
    if side in {None, "self"}:
        players = (player,)
    elif side in {"enemy", "opponent"}:
        players = (opponent,)
    else:
        players = (player, opponent)

    uids: list[int] = []
    for owner in players:
        view = snapshot.me if owner == "me" else snapshot.opponent
        uids.extend(getattr(view, zone))
    cards = [snapshot.cards_by_uid[uid] for uid in uids if uid in snapshot.cards_by_uid]

    if source_card is not None:
        cards = [card for card in cards if card.uid != source_card.uid or selector.relation not in {"other_than_source", "other"}]

    # 与 LocalActionEnumerator 的 target pool 保持一致：不可被对方能力选择
    # (cant_select) 的随从不能作为敌方目标。any/both selector 仍可选择己方的
    # cant_select 随从，只排除对方一侧，避免把合法己方选择一并过滤掉。
    # 潜伏（ambush）同样「不能被对手的能力指定为目标」，必须一起排除：
    # 否则「选择对手 1 个随从」在对手只剩潜伏随从时仍会选中它
    # （replay_...1219490 st36：11002110 进化时对潜伏中的 0/2 打了 2 点，录像没有伤害）。
    # 随机类 selector 不在此列——随机效果可以命中潜伏随从。
    enemy_field_uids = set(snapshot.opponent.field)
    targets_enemy_only = side in {"enemy", "opponent"} or selector.kind.startswith("selected_enemy_")
    if targets_enemy_only:
        cards = [card for card in cards if not card.has_cant_select and not card.has_ambush]
    elif side == "both":
        cards = [
            card
            for card in cards
            if not ((card.has_cant_select or card.has_ambush) and card.uid in enemy_field_uids)
        ]

    for item in selector.filters:
        if not isinstance(item, dict):
            return None
        field = item.get("field")
        operator = str(item.get("operator", "==") or "==")
        value = item.get("value")
        if field is None and item.get("exclude") is not None:
            field = "exclude"
            value = item.get("exclude")
        if field in {"zone", "side"}:
            continue
        if field == "card_type":
            cards = [card for card in cards if _compare_value(card.card_type.value, operator, value)]
        elif field in {"cost", "original_cost"}:
            cards = [card for card in cards if _compare_value(card.cost, operator, value)]
        elif field == "base_card_id":
            cards = [card for card in cards if _compare_value(card.base_card_id, operator, value)]
        elif field == "atk":
            cards = [card for card in cards if _compare_value(card.atk, operator, value)]
        elif field == "life":
            cards = [card for card in cards if _compare_value(card.life, operator, value)]
        elif field in {"is_evolved", "evolved"}:
            cards = [card for card in cards if _compare_value(card.evolve_state > 0, operator, value)]
        elif field == "attacked_this_turn":
            cards = [card for card in cards if _compare_value(card.attacks_left <= 0, operator, value)]
        elif field == "tribes":
            cards = [card for card in cards if _compare_value(_effective_tribes(card), operator, value)]
        elif field == "keyword":
            cards = [card for card in cards if _compare_value(_keyword_values(card), operator, value)]
        elif field == "class_name":
            cards = [card for card in cards if _compare_value(_class_name(card), operator, value)]
        elif field == "exclude" and value == "source_card":
            if source_card is None:
                return None
            cards = [card for card in cards if card.uid != source_card.uid]
        elif field == "uid" and value == "source_uid":
            # 「其他随从 / 本随从」类过滤器：uid != / == source_uid。
            # 与既有 relation="other_than_source" / exclude="source_card" 归一，
            # 不作为不可物化，候选为空时由上层 selected_* 空池分支合法落空。
            if source_card is None:
                return None
            cards = [card for card in cards if _compare_value(card.uid, operator, source_card.uid)]
        else:
            return None

    if card_type is not None:
        cards = [card for card in cards if _matches_card_type(card, card_type)]

    result = tuple(card.uid for card in cards)
    if include_leader:
        result = (*result, 0)
    if selector.kind in {"selected_self_card", "selected_any_card", "selected_enemy_card", "selected_self_follower", "selected_enemy_follower", "selected_any_follower", "selected_enemy_follower_or_leader", "selected_target", "selected_self_hand_card", "selected_opponent_hand_card", "selected_filtered_cards", "enemy_follower", "combat_opponent", "random", "random_self_follower", "random_enemy_follower", "random_enemy_followers", "random_filtered_cards", "enemy_followers_all"}:
        return result
    if selector.kind in {"all_filtered_cards", "all_self_followers", "all_enemy_followers", "all_self_hand_cards", "all_enemy_hand_cards", "all_self_unevolved_followers", "all_self_shikigami_followers", "all_enemy_leaders", "all_crests", "self_field", "ally_field", "enemy_field", "opponent_field", "enemy_board_and_leader"}:
        if selector.kind == "all_self_unevolved_followers":
            return tuple(card.uid for card in cards if card.card_type.is_follower and card.evolve_state == 0)
        if selector.kind == "all_self_shikigami_followers":
            return tuple(card.uid for card in cards if card.card_type.is_follower and _is_shikigami(card))
        if selector.kind == "all_enemy_leaders":
            return (0,)
        if selector.kind == "enemy_board_and_leader":
            return (*tuple(card.uid for card in cards), 0)
        return tuple(card.uid for card in cards)
    if selector.scope == "selected":
        return tuple(card.uid for card in cards)
    if selector.kind.startswith("selected_"):
        return tuple(card.uid for card in cards)
    return None


def _synthesize_implicit_selector(effect: EffectSpec, action: LegalAction) -> EffectSpecTarget | None:
    """从 operations 的 target_ref 合成一个 implicit EffectSpecTarget。

    当 spec.target 为 None 但 operations 中有 selected_* target_ref 时使用。
    常见于 evolve/super_evolve spec。
    """
    selected_kinds = {
        "selected_enemy_follower", "selected_self_follower", "selected_any_follower",
        "selected_target", "selected_any_card", "selected_self_card",
        "selected_self_hand_card", "selected_opponent_hand_card",
        "selected_filtered_cards", "selected_enemy_follower_or_leader",
        "selected_enemy_card",
    }
    for operation in effect.operations:
        if not _operation_mode_selected(operation, action):
            continue
        ref = operation.target_ref or ""
        if ref in selected_kinds:
            # 合成的 implicit 选择一定是「要选 1 个」：显式写 count=1，与 spec
            # 顶层 target 的 count=None（内部引用、不产生用户选择）区分开。
            return EffectSpecTarget(kind=ref, count=1)
    return None


def _has_selected_random_target_operation(effect: EffectSpec, action: LegalAction) -> bool:
    return any((operation.target_ref or "").startswith("random_") and _operation_mode_selected(operation, action) for operation in effect.operations)


def _operation_random_target_groups(
    snapshot: BattleSnapshot,
    action: LegalAction,
    effect: EffectSpec,
    *,
    player: str,
    opponent: str,
) -> tuple[tuple[int, ...], ...] | None:
    self_view = snapshot.me if player == "me" else snapshot.opponent
    opponent_view = snapshot.me if opponent == "me" else snapshot.opponent
    if action.random_target_groups:
        # 录制随机目标优先（replay 由 after 快照反推的真实命中目标）。`random_filtered_cards`
        # 这类只由 op 声明的随机效果没有 target_selector，若不在这里优先采用录制目标，
        # 物化会退化成「无目标」空转（10544110 回合结束随机 8 点伤害）。
        recorded_groups = tuple(group for group in action.random_target_groups if group)
        if recorded_groups:
            return recorded_groups
    random_refs = _distinct_random_refs(effect.operations)
    if len(random_refs) > 1:
        # 多随机池 effect（如 10954112 纹章回合结束「随机破坏己方谢幕曲随从 + 随机
        # 破坏敌方随从」）：每个 op 有独立候选池，某个池为空只说明该 op 无事可做，
        # 不能让整条 effect 空转。这里把所有非空池合并成一组统一下发，各 op 在
        # interpreter 里按自己的目标侧/过滤器认领属于自己那份（同
        # effect_interpreter 的多池认领逻辑）。全部池为空才是真正的空目标。
        union: list[int] = []
        for operation in effect.operations:
            if not _operation_mode_selected(operation, action):
                continue
            if not (operation.target_ref or "").startswith("random_"):
                continue
            pool = _single_random_ref_pool(
                snapshot, action, effect, operation,
                self_view=self_view, opponent_view=opponent_view,
                player=player, opponent=opponent,
            )
            if pool is None:
                return None
            union.extend(pool)
        if not union:
            return ((),)
        return (tuple(dict.fromkeys(union)),)
    for operation in effect.operations:
        if not _operation_mode_selected(operation, action):
            continue
        if operation.target_ref == "random_self_follower":
            candidates = tuple(uid for uid in self_view.field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower)
            if len(candidates) == 1:
                return (candidates,)
            return tuple((uid,) for uid in candidates) if candidates else None
        if operation.target_ref == "random_self_guard_unevolved_other_follower":
            candidates = tuple(
                uid
                for uid in self_view.field
                if uid != action.source_uid
                and (card := snapshot.cards_by_uid.get(uid)) is not None
                and card.card_type.is_follower
                and card.has_guard
                and card.evolve_state == 0
            )
            if len(candidates) == 1:
                return (candidates,)
            return tuple((uid,) for uid in candidates) if candidates else None
        if operation.target_ref == "random_self_unevolved_follower":
            candidates = tuple(
                uid
                for uid in self_view.field
                if (card := snapshot.cards_by_uid.get(uid)) is not None
                and card.card_type.is_follower
                and card.evolve_state == 0
            )
            if len(candidates) == 1:
                return (candidates,)
            return tuple((uid,) for uid in candidates) if candidates else None
        if operation.target_ref == "random_enemy_leader":
            repeat = int(operation.params.get("repeat", 1) or 1)
            if repeat <= 0:
                return ((),)
            return ((0,) * repeat,)
        if operation.target_ref == "random_enemy_follower":
            candidates = tuple(uid for uid in opponent_view.field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower)
            repeat = int(operation.params.get("repeat", 1) or 1)
            if repeat <= 0:
                return ((),)
            if not candidates:
                return ((),)
            if len(candidates) == 1:
                return (candidates * repeat,)
            return tuple(tuple(group) for group in product(candidates, repeat=repeat))
        if operation.target_ref == "random_enemy_follower_or_leader":
            candidates = tuple(uid for uid in opponent_view.field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower)
            if operation.params.get("target_pool") == "enemy_followers_or_leader":
                candidates = (*candidates, 0)
            repeat = int(operation.params.get("repeat", 1) or 1)
            if repeat <= 0:
                return ((),)
            if not candidates:
                return ((),)
            if len(candidates) == 1:
                return (candidates * repeat,)
            return tuple(tuple(group) for group in product(candidates, repeat=repeat))
        if operation.target_ref != "random_enemy_followers":
            continue
        # 「对对手战场上的随机 N 个随从」用 params.count 表达（10964120 的
        # 「随机2个随从造成3点伤害」）。原来只支持 repeat_source（对同一目标重复），
        # 没有 repeat_source 时直接返回 None → target_selector_not_supported。
        fixed_count = int(operation.params.get("count", 0) or operation.params.get("random_count", 0) or 0)
        if fixed_count > 0:
            followers = tuple(uid for uid in opponent_view.field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower)
            if not followers:
                return ((),)
            take = min(fixed_count, len(followers))
            return tuple(tuple(group) for group in combinations(followers, take))
        repeat_source = operation.params.get("repeat_source")
        if repeat_source not in {"runtime_x", "me.play_count"}:
            return None
        repeat = max(0, int(snapshot.me.play_count) + (1 if action.kind is ActionKind.PLAY else 0))
        candidates = tuple(uid for uid in opponent_view.field if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower)
        if repeat <= 0:
            return ((),)
        if not candidates:
            return ((),)
        return tuple(tuple(group) for group in product(candidates, repeat=repeat))
    return None


def _operation_mode_selected(operation: Any, action: LegalAction) -> bool:
    mode_option = operation.params.get("mode_option")
    if not isinstance(mode_option, dict) or mode_option.get("option_index") is None:
        return True
    try:
        return int(mode_option["option_index"]) in action.selected_indexes
    except (TypeError, ValueError):
        return False


def _distinct_random_refs(operations) -> set[str]:
    """一组 operation（含 repeat 嵌套）中独立 random target pool 的 ref 集合。"""

    refs: set[str] = set()
    for operation in operations:
        target_ref = operation.target_ref or ""
        if target_ref.startswith("random_"):
            refs.add(target_ref)
        params = operation.params if isinstance(operation.params, dict) else None
        nested = params.get("operation") if params else None
        if isinstance(nested, dict):
            nested_ref = str(nested.get("target_ref", "") or "")
            if nested_ref.startswith("random_"):
                refs.add(nested_ref)
    return refs


def _single_random_ref_pool(
    snapshot: BattleSnapshot,
    action: LegalAction,
    effect: EffectSpec,
    operation: Any,
    *,
    self_view: Any,
    opponent_view: Any,
    player: str,
    opponent: str,
) -> tuple[int, ...] | None:
    """单个 random target_ref 的完整候选池（供多池 effect 合并下发）。

    返回 ``None`` 表示该 ref 无法在这里确定候选池（交给调用方回退到原有逻辑）。
    与 ``_operation_random_target_groups`` 的单池分支不同，这里返回「池本身」而不是
    按 repeat 组好的目标组合：多池时各 op 由 interpreter 自行认领。
    """

    def followers(view: Any) -> tuple[int, ...]:
        return tuple(
            uid for uid in view.field
            if (card := snapshot.cards_by_uid.get(uid)) is not None and card.card_type.is_follower
        )

    ref = operation.target_ref or ""
    if ref in {"random_filtered_cards", "selected_filtered_cards", "all_filtered_cards"}:
        selector = getattr(effect, "target_selector", None)
        if selector is None:
            return None
        return _candidate_uids(snapshot, selector, source_card=None, player=player, opponent=opponent)
    if ref == "random_self_card":
        return tuple(self_view.field)
    if ref == "random_self_follower":
        return followers(self_view)
    if ref == "random_self_unevolved_follower":
        return tuple(uid for uid in followers(self_view) if snapshot.cards_by_uid[uid].evolve_state == 0)
    if ref == "random_self_unevolved_other_follower":
        return tuple(
            uid for uid in followers(self_view)
            if uid != action.source_uid and snapshot.cards_by_uid[uid].evolve_state == 0
        )
    if ref == "random_self_guard_unevolved_other_follower":
        return tuple(
            uid for uid in followers(self_view)
            if uid != action.source_uid
            and snapshot.cards_by_uid[uid].has_guard
            and snapshot.cards_by_uid[uid].evolve_state == 0
        )
    if ref in {"random_enemy_follower", "random_enemy_followers"}:
        return followers(opponent_view)
    if ref == "random_enemy_guard_followers":
        return tuple(uid for uid in followers(opponent_view) if snapshot.cards_by_uid[uid].has_guard)
    if ref == "random_enemy_max_atk_follower":
        pool = followers(opponent_view)
        if not pool:
            return ()
        max_atk = max(snapshot.cards_by_uid[uid].atk for uid in pool)
        return tuple(uid for uid in pool if snapshot.cards_by_uid[uid].atk == max_atk)
    if ref == "random_enemy_follower_or_leader":
        pool = followers(opponent_view)
        if operation.params.get("target_pool") == "enemy_followers_or_leader":
            pool = (*pool, 0)
        return pool
    if ref == "random_enemy_leader":
        return (0,)
    if ref == "random_self_hand_card":
        return tuple(self_view.hand)
    return None


def _dedupe_groups(groups: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    seen: set[tuple[int, ...]] = set()
    result: list[tuple[int, ...]] = []
    for group in groups:
        if group in seen:
            continue
        seen.add(group)
        result.append(group)
    return tuple(result)


def _matches_card_type(card: CardView, card_type: str) -> bool:
    """委托到 ``CardType.matches``，保证枚举器与物化器的 card_type 语义一致。"""
    return card.card_type.matches(card_type)


def _is_shikigami(card: CardView) -> bool:
    return card.name.startswith("式神") or "式神" in card.name


def _compare_value(left: Any, operator: str, right: Any) -> bool:
    if operator in {"==", "=", "eq"}:
        return left == right
    if operator == "!=":
        return left != right
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == "<":
        return left < right
    if operator == "contains":
        if isinstance(left, tuple):
            try:
                return int(right) in tuple(int(item) for item in left)
            except (TypeError, ValueError):
                return right in left
        if isinstance(left, str):
            return str(right) in left
        return False
    if operator == "in":
        return left in right if isinstance(right, (list, tuple, set, frozenset)) else False
    return False


def _effective_tribes(card: CardView) -> tuple[int, ...]:
    if card.tribes:
        return card.tribes
    base = default_card_resolver().get_base(card.base_card_id)
    return tuple(base.tribes) if base is not None else ()


def _keyword_values(card: CardView) -> tuple[str, ...]:
    values: list[str] = []
    if card.has_guard:
        values.append("guard")
    if card.has_storm:
        values.append("storm")
    if card.has_rush:
        values.append("rush")
    if card.has_drain:
        values.append("drain")
    if card.has_bane:
        values.append("bane")
    if card.has_ambush:
        values.append("ambush")
    if card.has_cant_select:
        values.append("cant_select")
    if card.has_last_words:
        values.append("last_words")
    return tuple(values)


def _class_name(card: CardView) -> str:
    value = card.supplement_info.get("class_name", "")
    if value:
        return str(value)
    base = default_card_resolver().get_base(card.base_card_id)
    return base.craft_class if base is not None else ""
