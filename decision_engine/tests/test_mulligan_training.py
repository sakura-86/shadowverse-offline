# -*- coding: utf-8 -*-
"""Mulligan 训练样本 schema、特征编码、集成测试。"""

import unittest

from decision_engine.cards import BaseCardInfo, CardInfo, CardResolver, MasterDataProvider
from decision_engine.execution import (
    EndTurnSelector,
    LocalDeckList,
    LocalMatchSetup,
    LocalMatchSetupConfig,
    LocalPredictor,
    LocalSelfPlayConfig,
    LocalSelfPlayRunner,
    MulliganTrace,
    PlayerLocalPredictor,
    perform_mulligan,
)
from decision_engine.state.snapshot import BattleSnapshot
from decision_engine.training.mulligan_features import (
    MULLIGAN_FEATURES_SCHEMA,
    mulligan_hand_card_features,
    mulligan_state_features,
)
from decision_engine.training.mulligan_example import (
    MULLIGAN_EXAMPLE_SCHEMA,
    MulliganTrainingExample,
    build_mulligan_training_example,
    validate_mulligan_example,
)


def _make_resolver() -> CardResolver:
    base_cards = {
        i: BaseCardInfo(base_card_id=i, name=f"卡牌{i}", type_id=1, card_type="follower", cost=i % 5, atk=i % 3, life=i % 3 + 1)
        for i in range(1001, 1011)
    }
    cards = {card_id: CardInfo(card_id=card_id, base_card_id=card_id) for card_id in base_cards}
    return CardResolver(MasterDataProvider(base_cards=base_cards, cards=cards))


def _setup(*, seed: int = 42) -> BattleSnapshot:
    result = LocalMatchSetup(_make_resolver()).create(
        LocalDeckList(name="TestA", class_name="Runecraft", card_ids=tuple(range(1001, 1011))),
        LocalDeckList(name="TestB", class_name="Havencraft", card_ids=tuple(range(1001, 1011))),
        LocalMatchSetupConfig(seed=seed, first_player="me"),
    )
    assert result.snapshot is not None
    return result.snapshot


class TestMulliganHandCardFeatures(unittest.TestCase):
    def test_encodes_known_card(self):
        snapshot = _setup()
        card = snapshot.cards_by_uid[snapshot.me.hand[0]]
        features = mulligan_hand_card_features(card)
        self.assertTrue(features["known"])
        self.assertIn("base_card_id", features)
        self.assertIn("cost", features)
        self.assertIn("card_type", features)
        self.assertIn("has_storm", features)
        self.assertIn("has_spell_boost", features)

    def test_encodes_none_card(self):
        features = mulligan_hand_card_features(None)
        self.assertFalse(features["known"])


class TestMulliganStateFeatures(unittest.TestCase):
    def test_encodes_full_state(self):
        snapshot = _setup()
        features = mulligan_state_features(
            snapshot,
            perspective="me",
            my_class="Runecraft",
            opponent_class="Havencraft",
        )
        self.assertEqual(features["schema"], MULLIGAN_FEATURES_SCHEMA)
        self.assertEqual(features["my_class"], "Runecraft")
        self.assertEqual(features["opponent_class"], "Havencraft")
        self.assertTrue(features["is_first"])
        self.assertEqual(features["hand_count"], 4)
        self.assertEqual(len(features["hand_cards"]), 4)
        # 每张手牌都是 known
        for card_feat in features["hand_cards"]:
            self.assertTrue(card_feat["known"])

    def test_opponent_perspective(self):
        snapshot = _setup()
        features = mulligan_state_features(
            snapshot,
            perspective="opponent",
            my_class="Havencraft",
            opponent_class="Runecraft",
        )
        self.assertEqual(features["my_class"], "Havencraft")
        self.assertFalse(features["is_first"])  # opponent 是后手


class TestMulliganTrainingExample(unittest.TestCase):
    def test_build_and_validate(self):
        snapshot = _setup()
        original_hand = list(snapshot.me.hand)
        cards_to_change = original_hand[:2]
        import random
        after_snapshot = perform_mulligan(snapshot, "me", cards_to_change, rng=random.Random(1))

        example = build_mulligan_training_example(
            snapshot=after_snapshot,
            player="me",
            cards_to_change=cards_to_change,
            my_class="Runecraft",
            opponent_class="Havencraft",
            sample_id="test-mulligan-001",
            source={"kind": "self_play_search", "episode_id": 0},
            trust={"level": "self_play_search", "accepted": True},
            value_target=1.0,
            generation_config={"kind": "mulligan_self_play"},
            metadata={"original_hand": original_hand},
        )

        self.assertEqual(example.schema, MULLIGAN_EXAMPLE_SCHEMA)
        self.assertEqual(example.sample_id, "test-mulligan-001")
        self.assertEqual(example.player, "me")
        self.assertEqual(example.value_target, 1.0)
        # mask: 前2张 True，后2张 False
        self.assertEqual(example.mulligan_mask, (True, True, False, False))
        # state_features 应该包含换牌后的手牌
        self.assertEqual(example.state_features["hand_count"], 4)

    def test_validate_rejects_empty_mask(self):
        with self.assertRaises(ValueError):
            validate_mulligan_example({
                "schema": MULLIGAN_EXAMPLE_SCHEMA,
                "version": 1,
                "sample_id": "x",
                "mulligan_mask": [],
                "state_features": {"hand_count": 0},
                "value_target": None,
            })

    def test_validate_rejects_invalid_value_target(self):
        with self.assertRaises(ValueError):
            validate_mulligan_example({
                "schema": MULLIGAN_EXAMPLE_SCHEMA,
                "version": 1,
                "sample_id": "x",
                "mulligan_mask": [True, False, False, False],
                "state_features": {"hand_count": 4},
                "value_target": 2.0,
            })

    def test_to_jsonable_output(self):
        snapshot = _setup()
        original_hand = list(snapshot.me.hand)
        import random
        after_snapshot = perform_mulligan(snapshot, "me", original_hand[:1], rng=random.Random(1))

        example = build_mulligan_training_example(
            snapshot=after_snapshot,
            player="me",
            cards_to_change=original_hand[:1],
            my_class="Runecraft",
            opponent_class="Havencraft",
            sample_id="test-mulligan-json",
            source={"kind": "self_play_search"},
            trust={"level": "self_play_search", "accepted": True},
            value_target=-1.0,
            generation_config={"kind": "mulligan_self_play"},
            metadata={"original_hand": original_hand},
        )
        jsonable = example.to_jsonable()
        self.assertEqual(jsonable["schema"], MULLIGAN_EXAMPLE_SCHEMA)
        self.assertIsInstance(jsonable["mulligan_mask"], list)
        self.assertEqual(jsonable["value_target"], -1.0)


class TestLocalSelfPlayRunnerMulliganTrace(unittest.TestCase):
    def test_runner_records_mulligan_traces(self):
        snapshot = _setup()

        class ChangeFirstCard:
            def select_mulligan(self, snapshot: BattleSnapshot, *, player: str) -> tuple[int, ...]:
                view = snapshot.me if player == "me" else snapshot.opponent
                return (view.hand[0],) if view.hand else ()

        runner = LocalSelfPlayRunner(predictor=PlayerLocalPredictor(LocalPredictor()), config=LocalSelfPlayConfig(max_actions_per_turn=1))
        result = runner.run(
            snapshot,
            me_selector=EndTurnSelector(),
            opponent_selector=EndTurnSelector(),
            me_mulligan=ChangeFirstCard(),
            opponent_mulligan=ChangeFirstCard(),
        )

        self.assertEqual(len(result.mulligan_traces), 2)
        me_trace = result.mulligan_traces[0]
        opp_trace = result.mulligan_traces[1]
        self.assertEqual(me_trace.player, "me")
        self.assertEqual(opp_trace.player, "opponent")
        self.assertEqual(len(me_trace.original_hand), 4)
        self.assertEqual(len(me_trace.cards_to_change), 1)
        self.assertIsNotNone(me_trace.post_mulligan_snapshot)


class TestSelfPlaySearchRunnerMulliganExamples(unittest.TestCase):
    def test_search_runner_generates_mulligan_examples_when_selector_provided(self):
        from decision_engine.planning.self_play_search import SelfPlaySearchConfig, SelfPlaySearchRunner

        class ChangeAllMulligan:
            def select_mulligan(self, snapshot: BattleSnapshot, *, player: str) -> tuple[int, ...]:
                view = snapshot.me if player == "me" else snapshot.opponent
                return tuple(view.hand)

        result = SelfPlaySearchRunner(
            config=SelfPlaySearchConfig(episodes=1, max_actions_per_turn=2, simulations=4),
            mulligan_selector=ChangeAllMulligan(),
        ).run(
            LocalDeckList(name="A", class_name="Runecraft", card_ids=(10161210,) * 8),
            LocalDeckList(name="B", class_name="Havencraft", card_ids=(10161210,) * 8),
        )

        # 应该有 mulligan examples
        mulligan_examples = [e for e in result.examples if e.get("schema") == MULLIGAN_EXAMPLE_SCHEMA]
        self.assertGreaterEqual(len(mulligan_examples), 1)
        # 验证内容
        mull = mulligan_examples[0]
        self.assertIn("mulligan_mask", mull)
        self.assertIn("state_features", mull)
        # value_target 取决于对局是否决出胜负
        if result.episodes[0].winner is not None:
            self.assertIsNotNone(mull.get("value_target"))
        else:
            self.assertIsNone(mull.get("value_target"))

    def test_search_runner_no_mulligan_examples_without_selector(self):
        from decision_engine.planning.self_play_search import SelfPlaySearchConfig, SelfPlaySearchRunner

        result = SelfPlaySearchRunner(
            config=SelfPlaySearchConfig(episodes=1, max_actions_per_turn=2, simulations=4),
        ).run(
            LocalDeckList(name="A", class_name="Runecraft", card_ids=(10161210,) * 8),
            LocalDeckList(name="B", class_name="Havencraft", card_ids=(10161210,) * 8),
        )

        mulligan_examples = [e for e in result.examples if e.get("schema") == MULLIGAN_EXAMPLE_SCHEMA]
        self.assertEqual(len(mulligan_examples), 0)


if __name__ == "__main__":
    unittest.main()
