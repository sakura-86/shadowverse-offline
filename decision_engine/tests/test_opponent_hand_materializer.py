# -*- coding: utf-8 -*-
"""Opponent hand materializer tests."""

import unittest

from decision_engine.simulation.opponent_hand_materializer import OpponentHandMaterializer
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, DeckPoolCard, PlayerView


class FakeBase:
    base_card_id = 101
    name = "候选"
    cost = 2
    atk = 2
    life = 2
    card_type = "follower"
    type_id = 1
    tribes = (0,)


class FakeResolver:
    def get_base_by_card_id(self, card_id):
        return FakeBase() if card_id == 101 else None


class OpponentHandMaterializerTest(unittest.TestCase):
    def test_materializes_hypothetical_card_without_mutating_input(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2),
            opponent=PlayerView(life=20, max_life=20, pp=2, max_pp=2, hand=(5,)),
            cards_by_uid={5: CardView(uid=5, card_id=5, base_card_id=5, name="已有", card_type=CardType.SPELL)},
        )

        branch = OpponentHandMaterializer(resolver=FakeResolver()).materialize(snapshot, DeckPoolCard(card_id=101, count=1), branch_index=0)

        self.assertEqual(snapshot.opponent.hand, (5,))
        self.assertEqual(len(branch.snapshot.opponent.hand), 2)
        uid = branch.snapshot.opponent.hand[-1]
        self.assertEqual(branch.materialized_uid, uid)
        self.assertLess(uid, 0)
        self.assertEqual(branch.snapshot.cards_by_uid[uid].card_id, 101)
        self.assertIn("opponent_candidate_card_is_hypothetical", branch.assumptions)
        self.assertEqual(branch.unresolved_effects, ())

    def test_materialization_failure_is_structured(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2),
            opponent=PlayerView(life=20, max_life=20, pp=2, max_pp=2),
            cards_by_uid={},
        )

        branch = OpponentHandMaterializer(resolver=FakeResolver()).materialize(snapshot, DeckPoolCard(card_id=999, count=1), branch_index=0)

        self.assertEqual(branch.snapshot, snapshot)
        self.assertIn("opponent_candidate_materialization_failed:999", branch.unresolved_effects)


if __name__ == "__main__":
    unittest.main()
