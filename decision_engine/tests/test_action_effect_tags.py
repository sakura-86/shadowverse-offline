# -*- coding: utf-8 -*-
"""动作级效果标签（schema v10）的契约测试。"""

from __future__ import annotations

import unittest

from decision_engine.training.action_effect_tags import (
    SLOTS,
    TAGS,
    TAG_GROUPS,
    action_effect_slot,
    action_effect_tags,
    action_effect_vector,
    card_effect_tags,
)
from decision_engine.training.tensorization import (
    ACTION_EFFECT_TAG_DIM_INDICES,
    ACTION_VECTOR_DIM,
    ACTION_VECTOR_FEATURE_ORDER,
)


class ActionEffectTagContractTest(unittest.TestCase):
    def test_tag_set_is_29_and_grouped(self) -> None:
        self.assertEqual(len(TAGS), 29)
        self.assertEqual(len(set(TAGS)), 29)
        self.assertEqual(sum(len(tags) for tags in TAG_GROUPS.values()), 29)
        self.assertEqual(len(SLOTS), 7)

    def test_action_vector_width_includes_effect_tags(self) -> None:
        # 24 维基础元数据之前是 23 维；v10 追加 29 维效果标签。
        self.assertEqual(ACTION_VECTOR_DIM, 23 + len(TAGS))
        self.assertEqual(len(ACTION_EFFECT_TAG_DIM_INDICES), len(TAGS))
        self.assertEqual(
            ACTION_EFFECT_TAG_DIM_INDICES,
            tuple(range(ACTION_VECTOR_DIM - len(TAGS), ACTION_VECTOR_DIM)),
        )
        for name in ACTION_VECTOR_FEATURE_ORDER:
            if name.startswith("effect_tags."):
                self.assertIn(name.split(".", 1)[1], TAGS)

    def test_slot_mapping(self) -> None:
        self.assertEqual(action_effect_slot("play", "normal"), "play_normal")
        self.assertEqual(action_effect_slot("play", None), "play_normal")
        self.assertEqual(action_effect_slot("play", "enhance"), "play_enhance")
        self.assertEqual(action_effect_slot("play", "accelerate"), "play_accelerate")
        self.assertEqual(action_effect_slot("play", "crystal"), "play_crystal")
        self.assertEqual(action_effect_slot("evolve"), "evolve")
        self.assertEqual(action_effect_slot("super_evolve"), "super_evolve")
        self.assertEqual(action_effect_slot("activate"), "activate")
        for kind in ("attack_leader", "attack_follower", "turn_end", "fusion"):
            self.assertIsNone(action_effect_slot(kind))

    def test_evolve_effect_differs_between_two_candidates(self) -> None:
        """frame 41 的核心对比：进化密友会打 3，进化英灵什么都不会发生。"""
        companion = action_effect_tags(10134120, action_kind="evolve")
        spirit = action_effect_tags(90034130, action_kind="evolve")
        self.assertIn("damage", companion)
        self.assertEqual(spirit, frozenset())

    def test_accelerate_and_crystal_come_from_related_cards(self) -> None:
        """激奏/结晶的效果不在主卡文本里，必须从关联卡取。"""
        accelerate = action_effect_tags(10844120, action_kind="play", play_kind="accelerate")
        crystal = action_effect_tags(10952110, action_kind="play", play_kind="crystal")
        self.assertIn("pp_max_up", accelerate)
        self.assertIn("summon", crystal)
        # 结晶不应把关联卡的【启动】能力算成这次出牌的效果
        self.assertNotIn("countdown_change", crystal)

    def test_vector_is_fixed_width_and_zero_for_terminal(self) -> None:
        vector = action_effect_vector(10134120, action_kind="evolve")
        self.assertEqual(len(vector), len(TAGS))
        self.assertEqual(vector[TAGS.index("damage")], 1.0)
        self.assertEqual(sum(action_effect_vector(10134120, action_kind="turn_end")), 0.0)
        self.assertEqual(sum(action_effect_vector(None, action_kind="play", play_kind="normal")), 0.0)

    def test_unknown_card_returns_empty_not_guess(self) -> None:
        self.assertEqual(card_effect_tags(0), {})
        self.assertEqual(card_effect_tags(None), {})
        self.assertEqual(card_effect_tags(1), {})


if __name__ == "__main__":
    unittest.main()
