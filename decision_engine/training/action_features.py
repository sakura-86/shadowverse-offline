# -*- coding: utf-8 -*-
"""LegalAction -> model training action features."""

from __future__ import annotations

from typing import Any, Iterable

from decision_engine.actions.model import ActionKind, LegalAction, PlayKind
from decision_engine.state.snapshot import BattleSnapshot, CardView
from decision_engine.training.action_effect_tags import action_effect_slot, action_effect_tags
from decision_engine.training.feature_schema import TRAINING_FEATURE_SCHEMA_VERSION, assert_no_leaked_feature_keys, to_jsonable
from decision_engine.training.entity_position import ENTITY_POSITION_SELF_LEADER, position_for_uid
from decision_engine.training.state_features import card_features


def legal_actions_features(snapshot: BattleSnapshot, actions: Iterable[LegalAction], *, perspective: str = "me") -> list[dict[str, Any]]:
    return [action_features(snapshot, action, perspective=perspective) for action in actions]


def action_features(snapshot: BattleSnapshot, action: LegalAction, *, perspective: str = "me") -> dict[str, Any]:
    if action.kind in {ActionKind.MULLIGAN_CHANGE, ActionKind.MULLIGAN_END}:
        return _mulligan_action_features(snapshot, action, perspective=perspective)
    source_card = snapshot.cards_by_uid.get(action.source_uid)
    targets = [_target_feature(snapshot.cards_by_uid.get(uid), snapshot=snapshot, target_uid=uid, perspective=perspective) for group in action.target_groups for uid in group]
    effective_cost = action.cost if action.cost is not None else source_card.cost if source_card is not None else None
    player_pp = snapshot.me.pp if perspective == "me" else snapshot.opponent.pp
    features = {
        "schema": "ActionFeatures",
        "version": TRAINING_FEATURE_SCHEMA_VERSION,
        "action_kind": _action_kind(action.kind),
        # play_kind 保留为可读原始字段；action_scalar 使用以下四个 one-hot，禁止再将字符串哈希为伪数值。
        "play_kind": action.play_kind.value if action.play_kind is not None else None,
        "is_normal_play": action.play_kind is PlayKind.NORMAL,
        "is_enhance": action.play_kind is PlayKind.ENHANCE,
        "is_accelerate": action.play_kind is PlayKind.ACCELERATE,
        "is_crystal": action.play_kind is PlayKind.CRYSTAL,
        "variant": action.variant,
        "cost": effective_cost,
        "pp_before_action": player_pp,
        "unused_pp_after_action": player_pp - effective_cost if effective_cost is not None else None,
        "uses_full_pp": effective_cost is not None and player_pp - effective_cost == 0 and effective_cost > 0,
        "leaves_one_pp": effective_cost is not None and player_pp - effective_cost == 1 and effective_cost > 0,
        "source": _source_feature(source_card, snapshot=snapshot, source_uid=action.source_uid, perspective=perspective),
        "targets": targets,
        "target_group_count": len(action.target_groups),
        "target_count": sum(len(group) for group in action.target_groups),
        # ── 目标槽结构（pointer 动作表示）──
        # ``targets`` 是**平铺**的，本身不含 group 边界。下面两个字段还原它：
        # ``target_group_sizes[i]`` 是第 i 个槽的实体数（= arity），
        # ``target_group_specs[i]`` 是这个槽的语义标识（来自
        # `card_effect_specs.json` 的 spec_id，或 attack_follower 等固定来源）。
        #
        # 语义：``count=N`` 是**一个槽选 N 个**，所以 sizes[i]=N 而不是 N 个槽。
        # 没有 specs 的旧数据（replay 来源的 LegalAction 不带 specs）落成空列表，
        # 下游按「无 slot 信息」处理，不按位置猜语义。
        "target_group_sizes": [len(group) for group in action.target_groups],
        "target_group_specs": list(action.target_group_specs),
        "random_target_group_count": len(action.random_target_groups),
        "random_target_count": sum(len(group) for group in action.random_target_groups),
        "selected_index_count": len(action.selected_indexes),
        "material_count": len(action.material_uids),
        "has_target": bool(action.target_groups),
        "has_materials": bool(action.material_uids),
        "is_terminal_action": action.kind == ActionKind.TURN_END,
        "is_resource_action": action.kind in {ActionKind.PLAY, ActionKind.EXTRA_PP, ActionKind.FUSION},
        "is_combat_action": action.kind in {ActionKind.ATTACK_LEADER, ActionKind.ATTACK_FOLLOWER},
        "is_evolve_action": action.kind in {ActionKind.EVOLVE, ActionKind.SUPER_EVOLVE},
        "costs_extra_pp": action.variant == "extra_pp",
    }
    # 动作级效果标签：这个动作会产生什么（回血 / 抽牌 / 直伤 / 召唤 / 破坏 / 自毁 …）。
    # 只写为 1 的标签，缺失 key 在 tensorization 里读成 0.0，避免 jsonl 膨胀。
    effect_slot = action_effect_slot(features["action_kind"], features["play_kind"])
    effect_tags = action_effect_tags(
        source_card.base_card_id if source_card is not None else None,
        action_kind=features["action_kind"],
        play_kind=features["play_kind"],
    )
    features["effect_slot"] = effect_slot
    features["effect_tags"] = {tag: True for tag in effect_tags}
    result = to_jsonable(features)
    assert_no_leaked_feature_keys(result)
    return result


def _mulligan_action_features(
    snapshot: BattleSnapshot, action: LegalAction, *, perspective: str = "me"
) -> dict[str, Any]:
    """换牌动作的特征。

    换牌动作与局内动作**同构**：``source`` 复用 ``card_features``（那张手牌的
    cost / atk / 类型 / 关键词），因此模型判断"这张该不该换"所需的信息与判断
    "这张该不该打"来自同一套特征。

    ``action_scalar`` 里除 ``action_kind`` 外全部为默认值——换牌不消耗 PP、没有目标、
    不是终端/资源/战斗/进化动作。模型依靠 ``action_kind`` 的编码值区分它，
    这也是为什么本改动**不增加** ``action_scalar_width``（现有 checkpoint 可继续加载）。

    ``is_mulligan_change`` / ``is_mulligan_end`` 只写进 JSON 供人阅读与测试断言，
    不在 ``ACTION_VECTOR_FEATURE_ORDER`` 里，因此不进模型输入。
    """
    is_change = action.kind is ActionKind.MULLIGAN_CHANGE
    view = snapshot.me if perspective == "me" else snapshot.opponent
    source_card = snapshot.cards_by_uid.get(action.source_uid) if is_change else None
    zone = ("me_hand" if perspective == "me" else "opponent_hand") if is_change else "mulligan_none"
    features = {
        "schema": "ActionFeatures",
        "version": TRAINING_FEATURE_SCHEMA_VERSION,
        "action_kind": _action_kind(action.kind),
        "play_kind": None,
        "is_normal_play": False,
        "is_enhance": False,
        "is_accelerate": False,
        "is_crystal": False,
        "variant": action.variant,
        "cost": None,
        "pp_before_action": view.pp,
        "unused_pp_after_action": None,
        "uses_full_pp": False,
        "leaves_one_pp": False,
        "source": card_features(
            source_card,
            zone=zone,
            progress=(
                snapshot.condition_progress_by_uid.get(action.source_uid) if is_change else None
            ),
        ),
        "targets": [],
        "target_group_count": 0,
        "target_count": 0,
        "random_target_group_count": 0,
        "random_target_count": 0,
        "selected_index_count": 0,
        "material_count": 0,
        "has_target": False,
        "has_materials": False,
        "is_terminal_action": False,
        "is_resource_action": False,
        "is_combat_action": False,
        "is_evolve_action": False,
        "costs_extra_pp": False,
        # 只供阅读 / 断言，不进模型输入（不在 ACTION_VECTOR_FEATURE_ORDER 里）。
        "is_mulligan_change": is_change,
        "is_mulligan_end": not is_change,
        "effect_slot": None,
        "effect_tags": {},
    }
    result = to_jsonable(features)
    assert_no_leaked_feature_keys(result)
    return result


def _source_feature(card: CardView | None, *, snapshot: BattleSnapshot, source_uid: int, perspective: str = "me") -> dict[str, Any]:
    """动作 source 与 state card token 共用完整 card_features 契约。"""
    result = card_features(
        card,
        zone=_zone_for_uid(snapshot, source_uid, perspective),
        progress=snapshot.condition_progress_by_uid.get(source_uid),
    )
    # 固定位置坐标：生成式 pointer 用它把动作的 source 指向实体 token。
    # 无 source 的动作（TURN_END / EXTRA_PP，``source_uid=0`` 且 card 为 None）
    # 锚到己方主战者坐标：pointer 的 source 分布必须给它们一个有效位置，
    # 否则折算成动作 logit 时会被置 -inf、永远选不到。
    result["entity_position"] = (
        ENTITY_POSITION_SELF_LEADER
        if card is None
        else position_for_uid(snapshot, source_uid, perspective=perspective)
    )
    return result


def _target_feature(card: CardView | None, *, snapshot: BattleSnapshot, target_uid: int, perspective: str = "me") -> dict[str, Any]:
    """动作 target 与 state card token 共用完整 card_features 契约，额外标示相对阵营。"""
    zone = _zone_for_uid(snapshot, target_uid, perspective)
    result = card_features(
        card,
        zone=zone,
        progress=snapshot.condition_progress_by_uid.get(target_uid),
    )
    result["side"] = _side_for_uid(snapshot, target_uid, perspective)
    result["entity_position"] = position_for_uid(snapshot, target_uid, perspective=perspective)
    return result


def _zone_for_uid(snapshot: BattleSnapshot, uid: int, perspective: str = "me") -> str:
    if perspective == "me":
        player_view, opponent_view = snapshot.me, snapshot.opponent
    else:
        player_view, opponent_view = snapshot.opponent, snapshot.me
    if uid in player_view.hand:
        return "me_hand"
    if uid in player_view.field:
        return "me_field"
    if uid in opponent_view.hand:
        return "opponent_hand"
    if uid in opponent_view.field:
        return "opponent_field"
    if uid == 0:
        return "leader"
    return "unknown"


def _side_for_uid(snapshot: BattleSnapshot, uid: int, perspective: str = "me") -> str:
    if perspective == "me":
        player_view, opponent_view = snapshot.me, snapshot.opponent
    else:
        player_view, opponent_view = snapshot.opponent, snapshot.me
    if uid in player_view.hand or uid in player_view.field:
        return "me"
    if uid in opponent_view.hand or uid in opponent_view.field:
        return "opponent"
    if uid == 0:
        return "leader"
    return "unknown"


def _action_kind(kind: ActionKind | str) -> str:
    return kind.value if isinstance(kind, ActionKind) else str(kind)
