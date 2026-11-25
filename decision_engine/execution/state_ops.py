# -*- coding: utf-8 -*-
"""最小可回滚状态操作。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from decision_engine.state.hand_knowledge import HandCardKnowledge
from decision_engine.state.snapshot import BattleSnapshot, CardView, PlayerView


PLAYER_ALIASES = {"me", "opponent"}
ZONE_FIELDS = {"hand", "field"}

# ── 模板静态关键词 → CardView 字段 ────────────────────────────────────────────
#
# 入场物化（LocalPredictor._play 的 static keyword 段）与自由盘面构造（沙盒）
# 必须对「模板里写了什么关键词」得出同一个 CardView，否则同一张卡在两条路径下
# 的合法动作集合会不同。映射只维护这一份。
ENTRY_KEYWORD_CARD_FIELDS: dict[str, str] = {
    "guard": "has_guard",
    "storm": "has_storm",
    "rush": "has_rush",
    "bane": "has_bane",
    "drain": "has_drain",
    "ambush": "has_ambush",
    "cant_select": "has_cant_select",
    "intimidate": "has_cant_be_attacked",
    "induction": "has_induction",
    "cant_destroy": "has_cant_destroy",
    "ignore_guard_when_attacking": "can_ignore_guard",
    "barrier": "has_temp_shield",
    "temp_shield": "has_temp_shield",
    "damage_cut": "has_damage_cut",
}

# 外部协议 root 的静态补齐子集：只补不会与「协议已下发的临时状态」冲突的字段。
# 临时护盾一类状态由协议显式下发，水合时不能替协议做主。
EXTERNAL_STATIC_KEYWORD_CARD_FIELDS: dict[str, str] = {
    "guard": "has_guard",
    "storm": "has_storm",
    "rush": "has_rush",
    "bane": "has_bane",
    "drain": "has_drain",
    "ambush": "has_ambush",
    "cant_select": "has_cant_select",
    "intimidate": "has_cant_be_attacked",
    "induction": "has_induction",
    "cant_destroy": "has_cant_destroy",
    "ignore_guard_when_attacking": "can_ignore_guard",
    "damage_cut": "has_damage_cut",
}


def static_keyword_card_updates(card: CardView, template: Any, *, enable_attacks: bool = True) -> dict[str, Any]:
    """计算「把模板静态关键词落到 CardView」所需字段更新。

    ``template`` 只按鸭子类型读取 ``static_keywords``（每项有 ``keyword``），
    避免 state 操作层反向依赖 cards 模块。

    ``enable_attacks=False`` 用于「牌库/手牌构造」：只落关键词本身，不预先授予
    ``can_attack`` / ``attacks_left``——那是入场与回合开始的事实，不能在建局时替它决定。

    返回只包含需要改动的字段，可直接用于 ``replace(card, **updates)``。
    attack_limit / attacks_left 在这里统一取最大值，避免关键词顺序影响结果。
    """
    updates: dict[str, Any] = {}
    attack_limit = card.attack_limit
    attacks_left = card.attacks_left
    for keyword in getattr(template, "static_keywords", ()) or ():
        name = str(getattr(keyword, "keyword", "") or "")
        if not name:
            continue
        field_name = ENTRY_KEYWORD_CARD_FIELDS.get(name)
        if field_name is not None and not getattr(card, field_name):
            updates[field_name] = True
        if name == "storm" and card.card_type.is_follower:
            attack_limit = max(attack_limit, 1)
            if enable_attacks:
                attacks_left = max(attacks_left, 1)
                updates["can_attack"] = True
                updates["can_attack_leader"] = True
        elif name == "rush" and card.card_type.is_follower:
            attack_limit = max(attack_limit, 1)
            if enable_attacks:
                attacks_left = max(attacks_left, 1)
                updates["can_attack"] = True
        elif name.startswith("attack_limit_"):
            attack_limit = max(attack_limit, int(name.removeprefix("attack_limit_")))
    if attack_limit != card.attack_limit:
        updates["attack_limit"] = attack_limit
    if attacks_left != card.attacks_left:
        updates["attacks_left"] = attacks_left
    return updates
PLAYER_COUNTER_FIELDS = {
    "graveyard_count",
    "rally",
    "play_count",
    "deck_count",
    "result",
    "life",
    "max_life",
    "pp",
    "max_pp",
    "ep",
    "sep",
    "max_ep",
    "max_sep",
    "manual_evolve_count",
    "evolve_count",
    "followers_entered_battlefield",
    "attack_count",
}


def get_player(snapshot: BattleSnapshot, player: str) -> PlayerView:
    if player == "me":
        return snapshot.me
    if player == "opponent":
        return snapshot.opponent
    raise ValueError(f"unknown player alias: {player}")


def replace_player(snapshot: BattleSnapshot, player: str, value: PlayerView) -> BattleSnapshot:
    if player == "me":
        return replace(snapshot, me=value)
    if player == "opponent":
        return replace(snapshot, opponent=value)
    raise ValueError(f"unknown player alias: {player}")


def replace_card(snapshot: BattleSnapshot, uid: int, value: CardView) -> BattleSnapshot:
    if uid not in snapshot.cards_by_uid:
        raise KeyError(f"card uid not found: {uid}")
    cards = dict(snapshot.cards_by_uid)
    cards[uid] = value
    updated = replace(snapshot, cards_by_uid=cards)
    updated_me = _sync_player_deck_instance_card(updated.me, uid, value)
    updated_opponent = _sync_player_deck_instance_card(updated.opponent, uid, value)
    if updated_me is updated.me and updated_opponent is updated.opponent:
        return updated
    return replace(updated, me=updated_me, opponent=updated_opponent)


def _sync_player_deck_instance_card(player_view: PlayerView, uid: int, value: CardView) -> PlayerView:
    deck_instances = player_view.deck_instances
    if deck_instances is None or uid not in deck_instances.by_uid():
        return player_view
    return replace(player_view, deck_instances=deck_instances.update_card(uid, value))


class SetSnapshotField:
    def __init__(self, field_name: str, value: Any):
        self.name = f"set_snapshot_field:{field_name}"
        self.field_name = field_name
        self.value = value

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        if not hasattr(snapshot, self.field_name):
            raise AttributeError(f"BattleSnapshot has no field: {self.field_name}")
        return replace(snapshot, **{self.field_name: self.value})

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> "SetSnapshotField":
        return SetSnapshotField(self.field_name, getattr(before, self.field_name))


class SetPlayerField:
    def __init__(self, player: str, field_name: str, value: Any):
        if player not in PLAYER_ALIASES:
            raise ValueError(f"unknown player alias: {player}")
        self.name = f"set_player_field:{player}.{field_name}"
        self.player = player
        self.field_name = field_name
        self.value = value

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        player_view = get_player(snapshot, self.player)
        if not hasattr(player_view, self.field_name):
            raise AttributeError(f"PlayerView has no field: {self.field_name}")
        return replace_player(snapshot, self.player, replace(player_view, **{self.field_name: self.value}))

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> "SetPlayerField":
        return SetPlayerField(self.player, self.field_name, getattr(get_player(before, self.player), self.field_name))


class ModifyPlayerCounter:
    def __init__(self, player: str, field_name: str, delta: int):
        if field_name not in PLAYER_COUNTER_FIELDS:
            raise ValueError(f"unsupported player counter field: {field_name}")
        self.name = f"modify_player_counter:{player}.{field_name}"
        self.player = player
        self.field_name = field_name
        self.delta = delta

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        player_view = get_player(snapshot, self.player)
        current = getattr(player_view, self.field_name)
        return replace_player(snapshot, self.player, replace(player_view, **{self.field_name: current + self.delta}))

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> "ModifyPlayerCounter":
        return ModifyPlayerCounter(self.player, self.field_name, -self.delta)


class SetCardField:
    def __init__(self, uid: int, field_name: str, value: Any):
        self.name = f"set_card_field:{uid}.{field_name}"
        self.uid = uid
        self.field_name = field_name
        self.value = value

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        card = snapshot.cards_by_uid[self.uid]
        if not hasattr(card, self.field_name):
            raise AttributeError(f"CardView has no field: {self.field_name}")
        return replace_card(snapshot, self.uid, replace(card, **{self.field_name: self.value}))

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> "SetCardField":
        return SetCardField(self.uid, self.field_name, getattr(before.cards_by_uid[self.uid], self.field_name))


class ModifyCardCounter:
    def __init__(self, uid: int, field_name: str, delta: int):
        self.name = f"modify_card_counter:{uid}.{field_name}"
        self.uid = uid
        self.field_name = field_name
        self.delta = delta

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        card = snapshot.cards_by_uid[self.uid]
        current = getattr(card, self.field_name)
        value = current + self.delta
        if self.field_name == "atk":
            # 攻击力下限为 0：协议不会出现负数攻击力（录像证据 actual=0），
            # 连续减攻不能把内部值压到负数。
            value = max(0, value)
        return replace_card(snapshot, self.uid, replace(card, **{self.field_name: value}))

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> "ModifyCardCounter":
        return ModifyCardCounter(self.uid, self.field_name, -self.delta)


def _hand_count_updates(player_view: PlayerView, delta: int) -> dict[str, int]:
    """手牌区增减时同步维护手牌**张数**。

    `PlayerView.hand_count` 语义是「真实手牌张数」，包含协议隐藏手牌（对手手牌
    在 replay 里是 `unique_id=0` 占位，不进 `hand`）。本地执行器一旦把可见卡插入
    /移出手牌区，张数也必须跟着变，否则手牌上限判定用的是帧加载时的陈旧值：
      - 我方打出牌后仍按 9 张判上限，把正常抽牌误判成爆牌；
      - 对手 9 张隐藏手牌则相反，永远判不出爆牌。
    因此增量必须写在事务里（产生的同一处），而不是在判定端用 `max(len, count)` 兜。
    """
    if player_view.hand_count is None:
        return {}
    return {"hand_count": max(0, player_view.hand_count + delta)}


class InsertZoneUid:
    def __init__(self, player: str, zone: str, uid: int, index: int | None = None):
        if zone not in ZONE_FIELDS:
            raise ValueError(f"unsupported zone field: {zone}")
        self.name = f"insert_zone_uid:{player}.{zone}"
        self.player = player
        self.zone = zone
        self.uid = uid
        self.index = index

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        player_view = get_player(snapshot, self.player)
        zone_values = list(getattr(player_view, self.zone))
        index = len(zone_values) if self.index is None else self.index
        if index < 0 or index > len(zone_values):
            raise IndexError(f"zone insert index out of range: {index}")
        zone_values.insert(index, self.uid)
        updates: dict[str, Any] = {self.zone: tuple(zone_values)}
        if self.zone == "hand":
            updates.update(_hand_count_updates(player_view, 1))
        return replace_player(snapshot, self.player, replace(player_view, **updates))

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> "RemoveZoneUidAt":
        index = len(getattr(get_player(before, self.player), self.zone)) if self.index is None else self.index
        return RemoveZoneUidAt(self.player, self.zone, index, self.uid)


class RemoveZoneUidAt:
    def __init__(self, player: str, zone: str, index: int, expected_uid: int | None = None):
        if zone not in ZONE_FIELDS:
            raise ValueError(f"unsupported zone field: {zone}")
        self.name = f"remove_zone_uid_at:{player}.{zone}"
        self.player = player
        self.zone = zone
        self.index = index
        self.expected_uid = expected_uid

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        player_view = get_player(snapshot, self.player)
        zone_values = list(getattr(player_view, self.zone))
        uid = zone_values[self.index]
        if self.expected_uid is not None and uid != self.expected_uid:
            raise ValueError(f"zone uid mismatch: expected {self.expected_uid}, got {uid}")
        del zone_values[self.index]
        updates: dict[str, Any] = {self.zone: tuple(zone_values)}
        if self.zone == "hand":
            updates.update(_hand_count_updates(player_view, -1))
        return replace_player(snapshot, self.player, replace(player_view, **updates))

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> InsertZoneUid:
        uid = getattr(get_player(before, self.player), self.zone)[self.index]
        return InsertZoneUid(self.player, self.zone, uid, self.index)


class RemoveZoneUid:
    def __init__(self, player: str, zone: str, uid: int):
        if zone not in ZONE_FIELDS:
            raise ValueError(f"unsupported zone field: {zone}")
        self.name = f"remove_zone_uid:{player}.{zone}"
        self.player = player
        self.zone = zone
        self.uid = uid

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        zone_values = getattr(get_player(snapshot, self.player), self.zone)
        index = zone_values.index(self.uid)
        return RemoveZoneUidAt(self.player, self.zone, index, self.uid).apply(snapshot)

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> InsertZoneUid:
        index = getattr(get_player(before, self.player), self.zone).index(self.uid)
        return InsertZoneUid(self.player, self.zone, self.uid, index)


class MoveZoneUid:
    def __init__(self, from_player: str, from_zone: str, from_index: int, to_player: str, to_zone: str, to_index: int | None = None):
        self.name = f"move_zone_uid:{from_player}.{from_zone}->{to_player}.{to_zone}"
        self.from_player = from_player
        self.from_zone = from_zone
        self.from_index = from_index
        self.to_player = to_player
        self.to_zone = to_zone
        self.to_index = to_index

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        uid = getattr(get_player(snapshot, self.from_player), self.from_zone)[self.from_index]
        current = RemoveZoneUidAt(self.from_player, self.from_zone, self.from_index, uid).apply(snapshot)
        return InsertZoneUid(self.to_player, self.to_zone, uid, self.to_index).apply(current)

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> "MoveZoneUid":
        uid = getattr(get_player(before, self.from_player), self.from_zone)[self.from_index]
        to_zone_before = getattr(get_player(before, self.to_player), self.to_zone)
        inserted_index = len(to_zone_before) if self.to_index is None else self.to_index
        if self.from_player == self.to_player and self.from_zone == self.to_zone and inserted_index > self.from_index:
            inserted_index -= 1
        return MoveZoneUid(self.to_player, self.to_zone, inserted_index, self.from_player, self.from_zone, self.from_index)


class AddCard:
    def __init__(self, card: CardView):
        self.name = f"add_card:{card.uid}"
        self.card = card

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        if self.card.uid in snapshot.cards_by_uid:
            raise ValueError(f"card uid already exists: {self.card.uid}")
        cards = dict(snapshot.cards_by_uid)
        cards[self.card.uid] = self.card
        return replace(snapshot, cards_by_uid=cards)

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> "RemoveCard":
        return RemoveCard(self.card.uid, self.card)


class SetCardFull:
    """覆盖 cards_by_uid 中已有 uid 的整张卡（用于 deck->field 等 zone 迁移场景）。"""

    def __init__(self, uid: int, card: CardView):
        self.name = f"set_card_full:{uid}"
        self.uid = uid
        self.card = card

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        cards = dict(snapshot.cards_by_uid)
        cards[self.uid] = self.card
        return replace(snapshot, cards_by_uid=cards)

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> "SetCardFull":
        return SetCardFull(self.uid, before.cards_by_uid[self.uid])


class RemoveCard:
    def __init__(self, uid: int, expected_card: CardView | None = None):
        self.name = f"remove_card:{uid}"
        self.uid = uid
        self.expected_card = expected_card

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        card = snapshot.cards_by_uid[self.uid]
        if self.expected_card is not None and card != self.expected_card:
            raise ValueError(f"card mismatch for uid: {self.uid}")
        cards = dict(snapshot.cards_by_uid)
        del cards[self.uid]
        return replace(snapshot, cards_by_uid=cards, hand_knowledge=snapshot.hand_knowledge.without_card(self.uid))

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot) -> AddCard:
        return AddCard(before.cards_by_uid[self.uid])


class SetHandKnowledge:
    def __init__(self, knowledge: HandCardKnowledge):
        self.name = f"set_hand_knowledge:{knowledge.uid}"
        self.knowledge = knowledge

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        return replace(snapshot, hand_knowledge=snapshot.hand_knowledge.with_card(self.knowledge))

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot):
        previous = before.hand_knowledge.get(self.knowledge.uid)
        if previous is None:
            return RemoveHandKnowledge(self.knowledge.uid)
        return SetHandKnowledge(previous)


class RemoveHandKnowledge:
    def __init__(self, uid: int):
        self.name = f"remove_hand_knowledge:{uid}"
        self.uid = uid

    def apply(self, snapshot: BattleSnapshot) -> BattleSnapshot:
        return replace(snapshot, hand_knowledge=snapshot.hand_knowledge.without_card(self.uid))

    def inverse(self, before: BattleSnapshot, after: BattleSnapshot):
        previous = before.hand_knowledge.get(self.uid)
        if previous is None:
            return RemoveHandKnowledge(self.uid)
        return SetHandKnowledge(previous)
