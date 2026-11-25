# -*- coding: utf-8 -*-
"""C1: canonical own remaining-deck 特征测试矩阵（方案 C）。

覆盖 DECK_ENTITY_TRANSFORMER_IMPLEMENTATION_PLAN.md §C1 验收：

- 相同多重集合、不同 draw order 的特征完全相同（无顺序泄漏）。
- 不同剩余构成产生不同特征。
- sum(remaining_count) == deck_count；不一致显式失败。
- ambiguous / invalid / missing 状态显式报告且不输出构成，无静默回退。
- perspective 对称：只为当前 perspective 玩家输出剩余牌组，不输出对手真实构成。
- 输出不含 instance_id / draw_pile / order_hint / visibility 等泄漏字段。
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from decision_engine.state.deck_instance import (
    DeckInstance,
    DeckInstanceState,
    make_deck_instance_state,
)
from decision_engine.state.snapshot import BattleSnapshot, CardView, CardType, PlayerView
from decision_engine.training.deck_entity_features import (
    DeckFeatureError,
    build_remaining_deck_features,
    remaining_deck_counter,
)
from decision_engine.training.state_features import state_features


def _deck_card(base_card_id: int) -> CardView:
    return CardView(uid=0, card_id=base_card_id + 900000, base_card_id=base_card_id, cost=2,
                    card_type=CardType.FOLLOWER)


def _deck_state_from_base_ids(base_ids, *, status: str = "available") -> DeckInstanceState:
    cards = tuple(_deck_card(bid) for bid in base_ids)
    state = make_deck_instance_state(cards, source="unit_test")
    if status != "available":
        state = replace(state, status=status)
    return state


class RemainingDeckFeatureHelperTest(unittest.TestCase):
    def test_aggregates_unique_base_card_ids_with_counts(self) -> None:
        state = _deck_state_from_base_ids([1000, 1000, 2000, 3000, 3000, 3000])
        features = build_remaining_deck_features(state, deck_count=6)
        self.assertEqual(features.deck_knowledge_status, "available")
        self.assertEqual(features.remaining_deck_count, 6)
        payload = features.to_jsonable()["remaining_deck_cards"]
        self.assertEqual(
            payload,
            [
                {"base_card_id": 1000, "remaining_count": 2},
                {"base_card_id": 2000, "remaining_count": 1},
                {"base_card_id": 3000, "remaining_count": 3},
            ],
        )

    def test_draw_order_permutation_is_invariant(self) -> None:
        base_ids = [1000, 2000, 3000, 1000, 3000, 3000]
        state = _deck_state_from_base_ids(base_ids)
        # 打乱 draw_pile 顺序，多重集合不变
        shuffled = replace(state, draw_pile=tuple(reversed(state.draw_pile)))
        a = build_remaining_deck_features(state).to_jsonable()
        b = build_remaining_deck_features(shuffled).to_jsonable()
        self.assertEqual(a, b)

    def test_different_composition_produces_different_features(self) -> None:
        a = build_remaining_deck_features(_deck_state_from_base_ids([1000, 1000])).to_jsonable()
        b = build_remaining_deck_features(_deck_state_from_base_ids([1000, 2000])).to_jsonable()
        self.assertNotEqual(a, b)

    def test_empty_deck_yields_empty_composition(self) -> None:
        state = DeckInstanceState(instances=(), draw_pile=(), status="available")
        features = build_remaining_deck_features(state, deck_count=0)
        self.assertEqual(features.remaining_deck_count, 0)
        self.assertEqual(features.remaining_deck_cards, ())

    def test_count_mismatch_raises(self) -> None:
        state = _deck_state_from_base_ids([1000, 2000, 3000])
        with self.assertRaises(DeckFeatureError):
            build_remaining_deck_features(state, deck_count=5)

    def test_missing_instance_reference_raises(self) -> None:
        state = _deck_state_from_base_ids([1000, 2000])
        # draw_pile 引用一个不存在的 instance_id
        broken = replace(state, draw_pile=state.draw_pile + (999,))
        with self.assertRaises(DeckFeatureError):
            remaining_deck_counter(broken)

    def test_non_available_status_raises_in_helper(self) -> None:
        for status in ("ambiguous", "invalid"):
            with self.subTest(status=status):
                state = _deck_state_from_base_ids([1000, 2000], status=status)
                with self.assertRaises(DeckFeatureError):
                    build_remaining_deck_features(state)

    def test_none_raises(self) -> None:
        with self.assertRaises(DeckFeatureError):
            build_remaining_deck_features(None)


class StateFeaturesRemainingDeckTest(unittest.TestCase):
    def _snapshot_with_my_deck(self, base_ids, *, status: str = "available", deck_count: int | None = None) -> BattleSnapshot:
        deck_instances = _deck_state_from_base_ids(base_ids, status=status)
        n = len(base_ids) if deck_count is None else deck_count
        me = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, deck_count=n, deck_instances=deck_instances)
        opp = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, deck_count=30)
        return BattleSnapshot(me=me, opponent=opp, cards_by_uid={}, is_ally_turn=True)

    def test_only_current_perspective_player_gets_remaining_deck(self) -> None:
        snapshot = self._snapshot_with_my_deck([1000, 1000, 2000])
        sf = state_features(snapshot, perspective="me")
        self.assertIn("remaining_deck", sf["player"])
        self.assertNotIn("remaining_deck", sf["opponent"])
        rd = sf["player"]["remaining_deck"]
        self.assertEqual(rd["deck_knowledge_status"], "available")
        self.assertEqual(rd["remaining_deck_count"], 3)
        self.assertEqual(
            rd["remaining_deck_cards"],
            [
                {"base_card_id": 1000, "remaining_count": 2},
                {"base_card_id": 2000, "remaining_count": 1},
            ],
        )

    def test_perspective_symmetry_opponent_view_hides_our_deck(self) -> None:
        # me 有 deck_instances，opponent 没有；从 opponent 视角看，player 是 opponent（无 deck），
        # 而作为对手的 me 不应输出 remaining_deck。
        snapshot = self._snapshot_with_my_deck([1000, 2000, 3000])
        sf = state_features(snapshot, perspective="opponent")
        # opponent 视角下 "player" 是 opponent 本人（无 deck_instances -> unavailable）
        self.assertEqual(sf["player"]["remaining_deck"]["deck_knowledge_status"], "unavailable")
        # "opponent" 是拥有真实牌组的 me，但不得输出其构成
        self.assertNotIn("remaining_deck", sf["opponent"])

    def test_missing_deck_instances_reports_unavailable_without_composition(self) -> None:
        me = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, deck_count=30, deck_instances=None)
        opp = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, deck_count=30)
        snapshot = BattleSnapshot(me=me, opponent=opp, cards_by_uid={}, is_ally_turn=True)
        sf = state_features(snapshot, perspective="me")
        rd = sf["player"]["remaining_deck"]
        self.assertEqual(rd["deck_knowledge_status"], "unavailable")
        self.assertEqual(rd["remaining_deck_cards"], [])

    def test_ambiguous_status_reports_status_without_composition(self) -> None:
        snapshot = self._snapshot_with_my_deck([1000, 2000], status="ambiguous")
        sf = state_features(snapshot, perspective="me")
        rd = sf["player"]["remaining_deck"]
        self.assertEqual(rd["deck_knowledge_status"], "ambiguous")
        self.assertEqual(rd["remaining_deck_cards"], [])

    def test_count_mismatch_under_available_raises(self) -> None:
        # available 状态但 deck_count 与 draw_pile 长度不一致 -> 显式失败
        snapshot = self._snapshot_with_my_deck([1000, 2000, 3000], deck_count=5)
        with self.assertRaises(DeckFeatureError):
            state_features(snapshot, perspective="me")

    def test_draw_order_permutation_invariant_in_state_features(self) -> None:
        snapshot = self._snapshot_with_my_deck([1000, 2000, 3000, 1000])
        shuffled_deck = replace(
            snapshot.me.deck_instances,
            draw_pile=tuple(reversed(snapshot.me.deck_instances.draw_pile)),
        )
        shuffled = replace(snapshot, me=replace(snapshot.me, deck_instances=shuffled_deck))
        a = state_features(snapshot, perspective="me")["player"]["remaining_deck"]
        b = state_features(shuffled, perspective="me")["player"]["remaining_deck"]
        self.assertEqual(a, b)

    def test_no_leaked_deck_fields_in_output(self) -> None:
        # state_features 内部已调用 assert_no_leaked_feature_keys；这里额外确认
        # 输出确实不含逐实例身份/顺序/可见性字段。
        snapshot = self._snapshot_with_my_deck([1000, 2000])
        rd = state_features(snapshot, perspective="me")["player"]["remaining_deck"]
        for entry in rd["remaining_deck_cards"]:
            self.assertEqual(set(entry.keys()), {"base_card_id", "remaining_count"})
        self.assertNotIn("draw_pile", rd)
        self.assertNotIn("instance_id", rd)


if __name__ == "__main__":
    unittest.main()
