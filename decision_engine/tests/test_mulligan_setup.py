# -*- coding: utf-8 -*-
"""起手4张和 mulligan（换牌）机制测试。"""

import random
import unittest

from decision_engine.cards import BaseCardInfo, CardInfo, CardResolver, MasterDataProvider
from decision_engine.execution import (
    LocalDeckList,
    LocalMatchSetup,
    LocalMatchSetupConfig,
    MulliganSelector,
    NoMulliganSelector,
    perform_mulligan,
)
from decision_engine.state.snapshot import BattleSnapshot


def _make_resolver() -> CardResolver:
    """创建包含10张不同牌的测试 resolver。"""
    base_cards = {
        i: BaseCardInfo(base_card_id=i, name=f"卡牌{i}", type_id=1, card_type="follower", cost=i % 5, atk=i % 3, life=i % 3 + 1)
        for i in range(1001, 1011)
    }
    cards = {card_id: CardInfo(card_id=card_id, base_card_id=card_id) for card_id in base_cards}
    return CardResolver(MasterDataProvider(base_cards=base_cards, cards=cards))


def _deck_10() -> LocalDeckList:
    """10张牌的测试卡组。"""
    return LocalDeckList(name="TestDeck", class_name="test", card_ids=tuple(range(1001, 1011)))


def _setup(*, seed: int = 42, first_player: str = "me") -> BattleSnapshot:
    """用默认配置创建初始 snapshot。"""
    result = LocalMatchSetup(_make_resolver()).create(
        _deck_10(), _deck_10(),
        LocalMatchSetupConfig(seed=seed, first_player=first_player),
    )
    assert result.snapshot is not None, f"setup failed: {result.errors}"
    return result.snapshot


class TestInitialHand(unittest.TestCase):
    """验证起手发4张牌。"""

    def test_default_config_deals_4_cards(self):
        """默认配置先后手各发4张。"""
        config = LocalMatchSetupConfig()
        self.assertEqual(config.initial_hand_first, 4)
        self.assertEqual(config.initial_hand_second, 4)

    def test_setup_creates_4_card_hand(self):
        """实际 setup 后双方各有4张手牌。"""
        snapshot = _setup()
        self.assertEqual(len(snapshot.me.hand), 4)
        self.assertEqual(len(snapshot.opponent.hand), 4)
        # 卡组剩余 10 - 4 = 6 张
        self.assertEqual(snapshot.me.deck_count, 6)
        self.assertEqual(snapshot.opponent.deck_count, 6)

    def test_hand_uids_are_unique(self):
        """手牌中的 uid 互不相同。"""
        snapshot = _setup()
        self.assertEqual(len(set(snapshot.me.hand)), 4)
        self.assertEqual(len(set(snapshot.opponent.hand)), 4)

    def test_hand_uids_exist_in_cards_by_uid(self):
        """手牌中每个 uid 都能在 cards_by_uid 中找到。"""
        snapshot = _setup()
        for uid in snapshot.me.hand:
            self.assertIn(uid, snapshot.cards_by_uid)
        for uid in snapshot.opponent.hand:
            self.assertIn(uid, snapshot.cards_by_uid)

    def test_different_seeds_give_different_hands(self):
        """不同 seed 应产生不同的手牌顺序。"""
        snap1 = _setup(seed=1)
        snap2 = _setup(seed=99)
        # 极小概率相同，但足够多牌的情况下基本不会
        hands1 = snap1.me.hand
        hands2 = snap2.me.hand
        self.assertNotEqual(hands1, hands2)


class TestPerformMulligan(unittest.TestCase):
    """验证 perform_mulligan 的正确性。"""

    def test_mulligan_no_change(self):
        """不换任何牌时 snapshot 保持不变。"""
        snapshot = _setup()
        after = perform_mulligan(snapshot, "me", [])
        self.assertEqual(after.me.hand, snapshot.me.hand)
        self.assertEqual(after.me.deck_count, snapshot.me.deck_count)

    def test_mulligan_replaces_cards(self):
        """换掉的牌不再在手牌中，手牌数量不变。"""
        snapshot = _setup()
        original_hand = snapshot.me.hand
        # 换掉前两张
        cards_to_change = list(original_hand[:2])
        after = perform_mulligan(snapshot, "me", cards_to_change, rng=random.Random(123))

        # 手牌数量不变
        self.assertEqual(len(after.me.hand), 4)
        # 换掉的 uid 不再出现在手牌中
        for uid in cards_to_change:
            self.assertNotIn(uid, after.me.hand)
        # 保留的 uid 仍然存在
        for uid in original_hand[2:]:
            self.assertIn(uid, after.me.hand)

    def test_mulligan_deck_count_unchanged(self):
        """换牌后卡组张数不变（换出N张 → 洗入N张 → 抽N张）。"""
        snapshot = _setup()
        cards_to_change = list(snapshot.me.hand[:3])
        after = perform_mulligan(snapshot, "me", cards_to_change, rng=random.Random(456))
        self.assertEqual(after.me.deck_count, snapshot.me.deck_count)

    def test_mulligan_all_cards(self):
        """换掉全部4张手牌。"""
        snapshot = _setup()
        original_hand = list(snapshot.me.hand)
        after = perform_mulligan(snapshot, "me", original_hand, rng=random.Random(789))

        self.assertEqual(len(after.me.hand), 4)
        # 由于洗入后可能抽到同一张（同一 deck instance），
        # 但至少 hand tuple 应该变化（极小概率相同）
        # 更重要的是：deck_count 保持一致
        self.assertEqual(after.me.deck_count, snapshot.me.deck_count)

    def test_mulligan_opponent(self):
        """对对手执行换牌。"""
        snapshot = _setup()
        original_hand = snapshot.opponent.hand
        cards_to_change = list(original_hand[:1])
        after = perform_mulligan(snapshot, "opponent", cards_to_change, rng=random.Random(111))

        self.assertEqual(len(after.opponent.hand), 4)
        self.assertNotIn(cards_to_change[0], after.opponent.hand)
        # me 手牌不受影响
        self.assertEqual(after.me.hand, snapshot.me.hand)

    def test_mulligan_invalid_uid_raises(self):
        """换掉不在手牌中的 uid 应报错。"""
        snapshot = _setup()
        with self.assertRaises(ValueError):
            perform_mulligan(snapshot, "me", [99999])

    def test_mulligan_deck_instances_consistency(self):
        """换牌后 deck_instances 的 hand 和 draw_pile 分区数量正确。"""
        snapshot = _setup()
        cards_to_change = list(snapshot.me.hand[:2])
        after = perform_mulligan(snapshot, "me", cards_to_change, rng=random.Random(222))

        state = after.me.deck_instances
        self.assertIsNotNone(state)
        # hand zone 应有4个 instance_id
        self.assertEqual(len(state.hand), 4)
        # draw_pile zone 应有6个 instance_id（10 - 4 = 6）
        self.assertEqual(len(state.draw_pile), 6)
        # 没有重叠
        self.assertFalse(set(state.hand) & set(state.draw_pile))

    def test_mulligan_new_cards_visible_in_cards_by_uid(self):
        """换牌后新抽到的牌在 cards_by_uid 中可查。"""
        snapshot = _setup()
        cards_to_change = list(snapshot.me.hand[:2])
        after = perform_mulligan(snapshot, "me", cards_to_change, rng=random.Random(333))
        for uid in after.me.hand:
            self.assertIn(uid, after.cards_by_uid)


class TestNoMulliganSelector(unittest.TestCase):
    """验证 NoMulliganSelector 行为。"""

    def test_no_mulligan_returns_empty(self):
        snapshot = _setup()
        selector = NoMulliganSelector()
        result = selector.select_mulligan(snapshot, player="me")
        self.assertEqual(result, ())


class TestMulliganWithSelfPlayRunner(unittest.TestCase):
    """验证 LocalSelfPlayRunner 的 mulligan 集成。"""

    def test_runner_accepts_mulligan_selector(self):
        """Runner 支持传入 mulligan selector 且不报错。"""
        from decision_engine.execution import (
            EndTurnSelector,
            LocalPredictor,
            LocalSelfPlayConfig,
            LocalSelfPlayRunner,
            PlayerLocalPredictor,
        )

        snapshot = _setup()

        class ChangeAllMulligan:
            def select_mulligan(self, snapshot: BattleSnapshot, *, player: str) -> tuple[int, ...]:
                view = snapshot.me if player == "me" else snapshot.opponent
                return tuple(view.hand)

        runner = LocalSelfPlayRunner(predictor=PlayerLocalPredictor(LocalPredictor()), config=LocalSelfPlayConfig(max_actions_per_turn=1))
        result = runner.run(
            snapshot,
            me_selector=EndTurnSelector(),
            opponent_selector=EndTurnSelector(),
            me_mulligan=ChangeAllMulligan(),
            opponent_mulligan=ChangeAllMulligan(),
        )
        # 能正常运行完成即可
        self.assertIsNotNone(result.final_snapshot)


if __name__ == "__main__":
    unittest.main()
