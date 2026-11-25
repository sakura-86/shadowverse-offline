# -*- coding: utf-8 -*-
"""Puzzle solution / search route footprint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.state.snapshot import BattleSnapshot, CardType


@dataclass(frozen=True)
class SnapshotSignature:
    opp_life: int
    my_life: int
    my_pp: int
    my_ep: int
    my_sep: int
    my_hand: tuple[int, ...]
    my_field: tuple[int, ...]
    opp_field: tuple[int, ...]
    attack_ready: tuple[tuple[int, int], ...]
    my_destroyed: tuple[int, ...]
    opp_destroyed: tuple[int, ...]
    card_states: tuple[tuple[int, int, int, int, int], ...]


@dataclass(frozen=True)
class ActionFootprint:
    action_key: str
    reads: frozenset[str]
    writes: frozenset[str]
    emits: frozenset[str]
    requires: frozenset[str]

    def conflicts_with(self, other: "ActionFootprint") -> bool:
        self_writes = set(self.writes)
        other_writes = set(other.writes)
        if "attack_leader" in self.emits and "attack_leader" in other.emits:
            self_writes.discard("opponent.life")
            other_writes.discard("opponent.life")
        if self_writes & (other.reads | other_writes):
            return True
        if other_writes & (self.reads | self_writes):
            return True
        if self.emits & other.requires:
            return True
        if other.emits & self.requires:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_key": self.action_key,
            "reads": sorted(self.reads),
            "writes": sorted(self.writes),
            "emits": sorted(self.emits),
            "requires": sorted(self.requires),
        }


def snapshot_signature(snapshot: BattleSnapshot) -> SnapshotSignature:
    return SnapshotSignature(
        opp_life=snapshot.opponent.life,
        my_life=snapshot.me.life,
        my_pp=snapshot.me.pp,
        my_ep=snapshot.me.ep,
        my_sep=snapshot.me.sep,
        my_hand=tuple(_card_id(snapshot, uid) for uid in snapshot.me.hand),
        my_field=tuple(_card_id(snapshot, uid) for uid in snapshot.me.field),
        opp_field=tuple(_card_id(snapshot, uid) for uid in snapshot.opponent.field),
        attack_ready=tuple(
            sorted(
                (
                    _card_id(snapshot, uid),
                    max(0, snapshot.cards_by_uid[uid].atk),
                )
                for uid in snapshot.me.field
                if uid in snapshot.cards_by_uid and snapshot.cards_by_uid[uid].attacks_left > 0
            )
        ),
        my_destroyed=tuple(item.card_id for item in snapshot.me.destroyed_card_ids),
        opp_destroyed=tuple(item.card_id for item in snapshot.opponent.destroyed_card_ids),
        card_states=tuple(
            sorted(
                (
                    _card_id(snapshot, uid),
                    card.atk,
                    card.life,
                    card.attacks_left,
                    card.evolve_state,
                )
                for uid, card in snapshot.cards_by_uid.items()
                if uid in snapshot.me.field or uid in snapshot.opponent.field
            )
        ),
    )


def transition_footprint(before: BattleSnapshot, after: BattleSnapshot, action: LegalAction | None = None) -> ActionFootprint:
    reads = set(_action_reads(before, action))
    writes = set(_snapshot_writes(before, after))
    emits = set(_action_emits(action))
    requires = set(_action_requires(action))
    if action is None:
        action_key = _inferred_action_key(before, after, writes)
    else:
        action_key = action_key_for(action, before)
    return ActionFootprint(
        action_key=action_key,
        reads=frozenset(reads),
        writes=frozenset(writes),
        emits=frozenset(emits),
        requires=frozenset(requires),
    )


def action_key_for(action: LegalAction, snapshot: BattleSnapshot | None = None) -> str:
    card_key = str(action.source_uid)
    if snapshot is not None:
        card = snapshot.cards_by_uid.get(action.source_uid)
        if card is not None:
            card_key = str(card.base_card_id or card.card_id or action.source_uid)
    parts = [action.kind.value, card_key]
    if action.play_kind is not None:
        parts.append(action.play_kind.value)
    if action.target_groups:
        parts.append("targets=" + _groups_key(action.target_groups, snapshot))
    if action.random_target_groups:
        parts.append("random=" + _groups_key(action.random_target_groups, snapshot))
    if action.selected_indexes:
        parts.append("mode=" + ",".join(str(index) for index in action.selected_indexes))
    if action.material_uids:
        parts.append("materials=" + _uids_key(action.material_uids, snapshot))
    return ":".join(parts)


def footprint_distance(left: ActionFootprint, right: ActionFootprint) -> float:
    if left.action_key == right.action_key:
        return 0.0
    if _action_family(left.action_key) == _action_family(right.action_key) and _action_subject(left.action_key) == _action_subject(right.action_key):
        return 0.15
    left_keys = left.writes | left.emits
    right_keys = right.writes | right.emits
    if not left_keys and not right_keys:
        return 0.0 if left.action_key == right.action_key else 1.0
    intersection = len(left_keys & right_keys)
    union = len(left_keys | right_keys)
    return 1.0 - intersection / max(1, union)


def snapshot_distance(left: BattleSnapshot, right: BattleSnapshot) -> float:
    distance = abs(left.opponent.life - right.opponent.life) * 4.0
    distance += abs(left.me.life - right.me.life) * 0.5
    distance += abs(left.me.pp - right.me.pp) * 1.5
    distance += abs(left.me.ep - right.me.ep) * 2.0
    distance += abs(left.me.sep - right.me.sep) * 2.0
    distance += abs(len(left.me.hand) - len(right.me.hand)) * 1.0
    distance += abs(len(left.me.field) - len(right.me.field)) * 1.5
    distance += abs(len(left.opponent.field) - len(right.opponent.field)) * 2.0
    distance += _destroyed_history_distance(left.me.destroyed_card_ids, right.me.destroyed_card_ids) * 2.5
    distance += _destroyed_history_distance(left.opponent.destroyed_card_ids, right.opponent.destroyed_card_ids) * 1.0
    distance += abs(_ready_attack_damage(left) - _ready_attack_damage(right)) * 2.0
    return distance


def _action_family(action_key: str) -> str:
    return action_key.split(":", 1)[0]


def _action_subject(action_key: str) -> str:
    parts = action_key.split(":")
    return parts[1] if len(parts) > 1 else ""


def _inferred_action_key(before: BattleSnapshot, after: BattleSnapshot, writes: set[str]) -> str:
    played = tuple(uid for uid in before.me.hand if uid not in after.me.hand and uid in before.cards_by_uid)
    entered = tuple(uid for uid in after.me.field if uid not in before.me.field and uid in after.cards_by_uid)
    if len(played) == 1:
        card_id = _card_id(before, played[0])
        if entered and played[0] in entered:
            return f"play:{card_id}"
        return f"play:{card_id}"
    attack_sources = tuple(
        uid
        for uid in before.me.field
        if uid in before.cards_by_uid
        and uid in after.cards_by_uid
        and before.cards_by_uid[uid].attacks_left > after.cards_by_uid[uid].attacks_left
    )
    if len(attack_sources) == 1 and after.opponent.life < before.opponent.life:
        return f"attack_leader:{_card_id(before, attack_sources[0])}"
    evolved = tuple(
        uid
        for uid in before.me.field
        if uid in before.cards_by_uid
        and uid in after.cards_by_uid
        and after.cards_by_uid[uid].evolve_state > before.cards_by_uid[uid].evolve_state
    )
    if len(evolved) == 1:
        kind = "super_evolve" if after.me.sep < before.me.sep else "evolve"
        return f"{kind}:{_card_id(before, evolved[0])}"
    activated = tuple(
        uid
        for uid in before.me.field
        if uid in before.cards_by_uid
        and uid in after.cards_by_uid
        and (before.cards_by_uid[uid].countdown != after.cards_by_uid[uid].countdown or before.cards_by_uid[uid].stack != after.cards_by_uid[uid].stack)
    )
    if len(activated) == 1 and after.me.pp < before.me.pp:
        return f"activate:{_card_id(before, activated[0])}"
    return "transition:" + ",".join(sorted(writes)[:8])


def _action_reads(snapshot: BattleSnapshot, action: LegalAction | None) -> frozenset[str]:
    if action is None:
        return frozenset()
    reads = {"me.pp", "me.hand", "me.field", "opponent.field"}
    reads.add(f"card:{action.source_uid}")
    for uid in _target_uids(action):
        reads.add(f"card:{uid}")
    for uid in action.material_uids:
        reads.add(f"card:{uid}")
    if action.kind in {ActionKind.EVOLVE, ActionKind.SUPER_EVOLVE}:
        reads.add("me.ep" if action.kind is ActionKind.EVOLVE else "me.sep")
    if action.kind in {ActionKind.ATTACK_LEADER, ActionKind.ATTACK_FOLLOWER}:
        reads.add(f"card:{action.source_uid}.attacks_left")
        reads.add(f"card:{action.source_uid}.can_attack")
    return frozenset(reads)


def _action_emits(action: LegalAction | None) -> frozenset[str]:
    if action is None:
        return frozenset()
    mapping = {
        ActionKind.PLAY: ("play_card", "fanfare", "enter_field"),
        ActionKind.ATTACK_LEADER: ("attack", "attack_leader", "on_attack"),
        ActionKind.ATTACK_FOLLOWER: ("attack", "attack_follower", "on_attack", "clash"),
        ActionKind.EVOLVE: ("becomes_evolved",),
        ActionKind.SUPER_EVOLVE: ("becomes_evolved", "becomes_super_evolved"),
        ActionKind.FUSION: ("fusion",),
        ActionKind.ACTIVATE: ("activation",),
        ActionKind.SPECIAL_ACTION: ("special_action",),
        ActionKind.EXTRA_PP: ("extra_pp",),
        ActionKind.TURN_END: ("turn_end",),
        ActionKind.TURN_START: ("turn_start",),
    }
    return frozenset(mapping.get(action.kind, ()))


def _action_requires(action: LegalAction | None) -> frozenset[str]:
    if action is None:
        return frozenset()
    requires: set[str] = set()
    if action.target_groups:
        requires.add("selected_target_pool")
    if action.random_target_groups:
        requires.add("random_target_pool")
    if action.material_uids:
        requires.add("fusion_material_pool")
    return frozenset(requires)


def _snapshot_writes(before: BattleSnapshot, after: BattleSnapshot) -> frozenset[str]:
    writes: set[str] = set()
    for side in ("me", "opponent"):
        lhs = getattr(before, side)
        rhs = getattr(after, side)
        for field_name in ("life", "pp", "ep", "sep", "hand", "field", "graveyard_count", "play_count", "rally", "destroyed_card_ids"):
            if getattr(lhs, field_name) != getattr(rhs, field_name):
                writes.add(f"{side}.{field_name}")
    if set(before.cards_by_uid) != set(after.cards_by_uid):
        writes.add("cards_by_uid")
    for uid in sorted(set(before.cards_by_uid) & set(after.cards_by_uid)):
        left = before.cards_by_uid[uid]
        right = after.cards_by_uid[uid]
        for field_name in ("card_id", "base_card_id", "atk", "life", "max_life", "cost", "evolve_state", "attacks_left", "can_attack", "can_attack_leader", "countdown", "stack"):
            if getattr(left, field_name) != getattr(right, field_name):
                writes.add(f"card:{uid}.{field_name}")
    return frozenset(writes)


def _target_uids(action: LegalAction) -> tuple[int, ...]:
    return tuple(uid for group in action.target_groups for uid in group)


def _groups_key(groups: tuple[tuple[int, ...], ...], snapshot: BattleSnapshot | None) -> str:
    return ";".join(_uids_key(group, snapshot) for group in groups)


def _uids_key(uids: tuple[int, ...], snapshot: BattleSnapshot | None) -> str:
    if snapshot is None:
        return ",".join(str(uid) for uid in uids)
    return ",".join(str(_card_id(snapshot, uid)) for uid in uids)


def _card_id(snapshot: BattleSnapshot, uid: int) -> int:
    card = snapshot.cards_by_uid.get(uid)
    if card is None:
        return uid
    return card.base_card_id or card.card_id or uid


def _destroyed_history_distance(left, right) -> float:
    left_cards = tuple(item.card_id for item in left)
    right_cards = tuple(item.card_id for item in right)
    common = len(set(left_cards) & set(right_cards))
    return float(abs(len(left_cards) - len(right_cards)) + max(0, min(len(left_cards), len(right_cards)) - common))


def _ready_attack_damage(snapshot: BattleSnapshot) -> int:
    total = 0
    for uid in snapshot.me.field:
        card = snapshot.cards_by_uid.get(uid)
        if card is None or card.card_type is not CardType.FOLLOWER:
            continue
        if card.attacks_left > 0 and (card.can_attack or card.can_attack_leader):
            total += max(0, card.atk)
    return total
