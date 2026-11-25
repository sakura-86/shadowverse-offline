# -*- coding: utf-8 -*-
"""本地枚举器：多选模式（选择 N 个能力）动作展开回归测试。"""

from __future__ import annotations

import unittest
from dataclasses import replace

from decision_engine.actions.local_enumerator import LocalActionEnumerator
from decision_engine.actions.model import ActionKind
from decision_engine.execution.local_match_setup import (
    LocalDeckList,
    LocalMatchSetup,
    LocalMatchSetupConfig,
)

# 11043310「俯瞰的目光」: 选择 2 个能力发动，模板有 4 个 option。
MODE_CARD_ID = 11043310


class ModeSelectionEnumerationTest(unittest.TestCase):
    def _snapshot_with_mode_card(self):
        deck_a = LocalDeckList(name="A", class_name="dragon", card_ids=(MODE_CARD_ID,) * 40)
        deck_b = LocalDeckList(name="B", class_name="dragon", card_ids=(MODE_CARD_ID,) * 40)
        setup = LocalMatchSetup().create(
            deck_a,
            deck_b,
            LocalMatchSetupConfig(
                seed=1,
                first_player="me",
                initial_hand_first=1,
                initial_hand_second=0,
            ),
        )
        snapshot = setup.snapshot
        self.assertIsNotNone(snapshot, setup.errors)
        snapshot = replace(snapshot, is_turn_ended=False, me=replace(snapshot.me, pp=10, max_pp=10))
        return snapshot

    def test_select_two_mode_options_enumerates_all_combinations(self) -> None:
        snapshot = self._snapshot_with_mode_card()
        hand_card = snapshot.cards_by_uid[snapshot.me.hand[0]]
        self.assertEqual(hand_card.base_card_id, MODE_CARD_ID)

        actions = LocalActionEnumerator().enumerate(snapshot)
        plays = [
            action
            for action in actions
            if action.kind is ActionKind.PLAY and action.source_uid == hand_card.uid
        ]
        self.assertTrue(plays, "mode card should be playable at 10 PP")

        selected = {action.selected_indexes for action in plays}
        self.assertTrue(all(len(indexes) == 2 for indexes in selected), selected)
        self.assertEqual(
            selected,
            {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)},
        )

    def test_single_selection_card_keeps_single_index(self) -> None:
        """没有 mode_choice_effect 的卡仍走单选路径，选择数回退为 1。"""
        deck_a = LocalDeckList(name="A", class_name="neutral", card_ids=(10001110,) * 40)
        deck_b = LocalDeckList(name="B", class_name="neutral", card_ids=(10001110,) * 40)
        setup = LocalMatchSetup().create(
            deck_a,
            deck_b,
            LocalMatchSetupConfig(seed=2, first_player="me", initial_hand_first=1, initial_hand_second=0),
        )
        snapshot = setup.snapshot
        self.assertIsNotNone(snapshot, setup.errors)
        card = snapshot.cards_by_uid[snapshot.me.hand[0]]
        self.assertEqual(card.base_card_id, 10001110)
        self.assertEqual(LocalActionEnumerator()._spec_mode_select_count(card), 1)


if __name__ == "__main__":
    unittest.main()
