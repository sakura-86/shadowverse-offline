# -*- coding: utf-8 -*-
"""HandKnowledge 状态模型测试。"""

import unittest

from decision_engine.state.hand_knowledge import HandKnowledgeState, known_generated_hand_card, unknown_opponent_hand_copy


class HandKnowledgeTest(unittest.TestCase):
    def test_unknown_opponent_hand_copy_is_material_only(self):
        knowledge = unknown_opponent_hand_copy(uid=9, owner="me", source_uid=3)

        self.assertTrue(knowledge.is_unknown)
        self.assertFalse(knowledge.playable)
        self.assertTrue(knowledge.usable_as_material)
        self.assertEqual(knowledge.origin, "copied_from_opponent_hand")

    def test_state_updates_are_immutable(self):
        state = HandKnowledgeState.empty()
        known = known_generated_hand_card(uid=7, owner="me", base_card_id=10844112, card_id=10844112)

        updated = state.with_card(known)
        bound = updated.bind_replay_uid(7, 70, card_id=10844112, base_card_id=10844112)
        removed = bound.without_card(7)

        self.assertIsNone(state.get(7))
        self.assertEqual(updated.get(7), known)
        self.assertEqual(bound.get(7).replay_bound_uid, 70)
        self.assertIsNone(removed.get(7))


if __name__ == "__main__":
    unittest.main()
