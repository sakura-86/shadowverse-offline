# -*- coding: utf-8 -*-
"""Puzzle solution trace canonicalization/alignment tests."""

import unittest
from dataclasses import replace

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.trace_alignment import align_route_to_solution, make_solution_trace
from decision_engine.planning.trace_canonicalizer import canonical_blocks
from decision_engine.planning.trace_footprint import transition_footprint
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView


class TraceAlignmentTest(unittest.TestCase):
    def test_independent_attacks_share_one_canonical_block(self):
        start = _snapshot()
        first = _after_attack(start, 1, 17)
        second = _after_attack(first, 2, 14)
        a = transition_footprint(start, first, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=1))
        b = transition_footprint(first, second, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=2))

        blocks = canonical_blocks((a, b))

        self.assertEqual(len(blocks), 1)
        self.assertEqual(set(blocks[0].action_keys), {a.action_key, b.action_key})

    def test_alignment_accepts_commuted_actions_inside_block(self):
        start = _snapshot()
        solution_one = _after_attack(start, 1, 17)
        solution_two = _after_attack(solution_one, 2, 14)
        search_one = _after_attack(start, 2, 17)
        search_two = _after_attack(search_one, 1, 14)
        solution_footprints = (
            transition_footprint(start, solution_one, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=1)),
            transition_footprint(solution_one, solution_two, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=2)),
        )
        route_footprints = (
            transition_footprint(start, search_one, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=2)),
            transition_footprint(search_one, search_two, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=1)),
        )
        solution = make_solution_trace("case", (start, solution_one, solution_two), solution_footprints)

        alignment = align_route_to_solution(search_two, route_footprints, solution)

        self.assertEqual(alignment.matched_prefix, 1)
        self.assertGreater(alignment.score, 0.0)


def _snapshot() -> BattleSnapshot:
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=0, max_pp=0, field=(1, 2)),
        opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0),
        cards_by_uid={
            1: CardView(uid=1, card_id=1001, base_card_id=1001, name="A", atk=3, life=1, max_life=1, card_type=CardType.FOLLOWER, can_attack=True, can_attack_leader=True, attacks_left=1),
            2: CardView(uid=2, card_id=1002, base_card_id=1002, name="B", atk=3, life=1, max_life=1, card_type=CardType.FOLLOWER, can_attack=True, can_attack_leader=True, attacks_left=1),
        },
        source="test",
        is_ally_turn=True,
    )


def _after_attack(snapshot: BattleSnapshot, uid: int, opponent_life: int) -> BattleSnapshot:
    card = snapshot.cards_by_uid[uid]
    cards = {**snapshot.cards_by_uid, uid: replace(card, attacks_left=0, can_attack=False, can_attack_leader=False)}
    return replace(snapshot, opponent=replace(snapshot.opponent, life=opponent_life), cards_by_uid=cards)


if __name__ == "__main__":
    unittest.main()
