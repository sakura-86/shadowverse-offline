# -*- coding: utf-8 -*-
"""Entity 输入的信息边界契约（回归门）。

影之诗的对手手牌是**隐藏信息**。把它作为 ``VIS_VISIBLE`` entity token 喂进网络，
会让训练期看到实机拿不到的内容——训练/推理分布不一致，模型到实机必然失效。

本文件锁住两条：
1. 对手手牌**不得**出现在 entity token 序列里（内容不进网络）；
2. 对手手牌**张数**必须保留在 state_vector 标量里（信息不丢）。

第 2 条是第 1 条的前提：如果张数也丢了，"隐藏手牌内容"会退化成"连几张都不知道"。
"""

from __future__ import annotations

import unittest

from decision_engine.execution.local_match_setup import (
    LocalDeckList,
    LocalMatchSetup,
    LocalMatchSetupConfig,
)
from decision_engine.training.entity_tensorization import (
    OWNER_OPPONENT,
    ZONE_HAND,
    build_entity_tokens,
    default_card_vocabulary,
)
from decision_engine.training.state_features import state_features
from decision_engine.training.tensorization import (
    STATE_VECTOR_FEATURE_ORDER,
    state_vector_from_state_features,
)


def _snapshot(seed: int = 7):
    deck = LocalDeckList(name="boundary", class_name="boundary", card_ids=tuple([10161210] * 40))
    return LocalMatchSetup().create(deck, deck, LocalMatchSetupConfig(seed=seed, first_player="me")).snapshot


class EntityInformationBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = _snapshot()
        cls.payload = state_features(cls.snapshot, perspective="me")
        cls.vocabulary = default_card_vocabulary()

    def test_precondition_opponent_hand_is_non_empty(self) -> None:
        # 前置：快照里确实有对手手牌，否则下面的断言是空洞的。
        self.assertGreater(len(self.payload["opponent"]["hand_cards"]), 0)

    def test_opponent_hand_cards_are_not_emitted_as_tokens(self) -> None:
        tokens = build_entity_tokens(self.payload, vocabulary=self.vocabulary)
        leaked = [t for t in tokens if t.owner_id == OWNER_OPPONENT and t.zone_id == ZONE_HAND]
        self.assertEqual(leaked, [], "对手手牌不得作为 entity token 进入网络")

    def test_token_count_does_not_depend_on_opponent_hand_size(self) -> None:
        base = build_entity_tokens(self.payload, vocabulary=self.vocabulary)
        # 模拟"对手手牌更多"（内容不同、张数更多）——token 数不应变化。
        inflated = {
            **self.payload,
            "opponent": {
                **self.payload["opponent"],
                "hand_cards": list(self.payload["opponent"]["hand_cards"])
                + list(self.payload["opponent"]["hand_cards"]),
            },
        }
        self.assertEqual(len(build_entity_tokens(inflated, vocabulary=self.vocabulary)), len(base))

    def test_opponent_hand_count_survives_in_state_vector(self) -> None:
        self.assertIn("opponent.hand_count", STATE_VECTOR_FEATURE_ORDER)
        vector = state_vector_from_state_features(self.payload)
        index = STATE_VECTOR_FEATURE_ORDER.index("opponent.hand_count")
        self.assertEqual(
            vector[index],
            float(len(self.snapshot.opponent.hand)),
            "state_vector 里的 opponent.hand_count 必须等于真实手牌张数",
        )


if __name__ == "__main__":
    unittest.main()
