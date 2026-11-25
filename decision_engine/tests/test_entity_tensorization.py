# -*- coding: utf-8 -*-
"""C2: entity tensorization 契约测试矩阵（方案 C）。

覆盖 DECK_ENTITY_TRANSFORMER_IMPLEMENTATION_PLAN.md §C2 验收：

- entity mask / zone / type / owner / count 字段逐项 golden test。
- batch 内不同 unique-deck 长度正确动态 padding。
- deck / hand token 排列变化不改变 canonical tensor 语义。
- field position 可区分。
- 新旧 schema 交叉加载显式失败。
- 全空区域下 [STATE] token 仍有效。
- serialization round-trip。
"""

from __future__ import annotations

import unittest

import torch

from decision_engine.state.deck_instance import make_deck_instance_state
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView
from decision_engine.training.card_vocabulary import CardVocabulary
from decision_engine.training.state_features import state_features
from decision_engine.training.entity_tensorization import (
    CARD_STAT_DIM,
    CREST_STAT_DIM,
    ENTITY_TYPE_STATE,
    ENTITY_TYPE_CARD,
    ENTITY_TYPE_DECK_CARD,
    ENTITY_LAYOUT_VERSION,
    OWNER_SELF,
    OWNER_OPPONENT,
    ZONE_HAND,
    ZONE_FIELD,
    ZONE_DECK,
    EntitySchemaMismatch,
    assert_entity_schema,
    build_entity_tokens,
    collate_entity_rows,
    entity_tensorize_example,
)


_VOCAB = CardVocabulary({1000: 2, 1001: 3, 1002: 4, 2000: 5, 3000: 6, 3001: 7})


def _deck_card(base_card_id: int) -> CardView:
    return CardView(uid=0, card_id=base_card_id + 900000, base_card_id=base_card_id, cost=2,
                    card_type=CardType.FOLLOWER)


def _example_from_snapshot(snapshot: BattleSnapshot, actions=()) -> dict:
    from decision_engine.training.action_features import legal_actions_features
    return {
        "state_features": state_features(snapshot, perspective="me"),
        "legal_action_features": legal_actions_features(snapshot, actions, perspective="me"),
        "policy_target": [1.0 / max(len(actions), 1)] * len(actions),
        "value_target": 0.5,
    }


def _snapshot(hand=(), field=(), opp_field=(), deck_base_ids=None) -> BattleSnapshot:
    cards: dict[int, CardView] = {}
    for uid in hand + field + opp_field:
        cards.setdefault(uid, CardView(uid=uid, card_id=1000 + uid, base_card_id=1000, cost=2,
                                       card_type=CardType.FOLLOWER, atk=1, life=1, max_life=1))
    deck_instances = None
    deck_count = 0
    if deck_base_ids is not None:
        deck_instances = make_deck_instance_state(tuple(_deck_card(b) for b in deck_base_ids), source="test")
        deck_count = len(deck_base_ids)
    me = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, hand=hand, field=field,
                    deck_count=deck_count, deck_instances=deck_instances)
    opp = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, field=opp_field, deck_count=30)
    return BattleSnapshot(me=me, opponent=opp, cards_by_uid=cards, is_ally_turn=True)


class EntityTokenBuildTest(unittest.TestCase):
    def test_state_token_is_first_and_always_present(self) -> None:
        tokens = build_entity_tokens(state_features(_snapshot(), perspective="me"), vocabulary=_VOCAB)
        self.assertEqual(tokens[0].type_id, ENTITY_TYPE_STATE)
        self.assertEqual(len(tokens), 1)  # 空场只有 [STATE]

    def test_deck_tokens_carry_count_and_ratio(self) -> None:
        tokens = build_entity_tokens(
            state_features(_snapshot(deck_base_ids=[1000, 1000, 2000, 2000]), perspective="me"),
            vocabulary=_VOCAB,
        )
        deck_tokens = [t for t in tokens if t.type_id == ENTITY_TYPE_DECK_CARD]
        self.assertEqual(len(deck_tokens), 2)  # 两种唯一卡
        for t in deck_tokens:
            self.assertEqual(t.zone_id, ZONE_DECK)
            self.assertEqual(t.owner_id, OWNER_SELF)
            self.assertEqual(t.count, 2.0)
            self.assertAlmostEqual(t.count_ratio, 0.5)

    def test_owner_and_zone_ids_are_correct(self) -> None:
        tokens = build_entity_tokens(
            state_features(_snapshot(hand=(1,), field=(2,), opp_field=(3,)), perspective="me"),
            vocabulary=_VOCAB,
        )
        by_zone_owner = {(t.zone_id, t.owner_id) for t in tokens if t.type_id == ENTITY_TYPE_CARD}
        self.assertIn((ZONE_HAND, OWNER_SELF), by_zone_owner)
        self.assertIn((ZONE_FIELD, OWNER_SELF), by_zone_owner)
        self.assertIn((ZONE_FIELD, OWNER_OPPONENT), by_zone_owner)

    def test_field_position_ids_are_distinct(self) -> None:
        tokens = build_entity_tokens(
            state_features(_snapshot(field=(2, 3, 4)), perspective="me"),
            vocabulary=_VOCAB,
        )
        field_positions = [t.position_id for t in tokens if t.zone_id == ZONE_FIELD]
        self.assertEqual(sorted(field_positions), [1, 2, 3])

    def test_hand_tokens_have_no_position(self) -> None:
        tokens = build_entity_tokens(
            state_features(_snapshot(hand=(1, 2, 3)), perspective="me"),
            vocabulary=_VOCAB,
        )
        for t in tokens:
            if t.zone_id == ZONE_HAND:
                self.assertEqual(t.position_id, 0)


class CollateTest(unittest.TestCase):
    def test_dynamic_padding_across_different_entity_counts(self) -> None:
        row_small = entity_tensorize_example(_example_from_snapshot(_snapshot(hand=(1,))), vocabulary=_VOCAB)
        row_large = entity_tensorize_example(
            _example_from_snapshot(_snapshot(hand=(1, 2), field=(3,), deck_base_ids=[1000, 2000, 3000])),
            vocabulary=_VOCAB,
        )
        batch = collate_entity_rows([row_small, row_large])
        E = batch["entity_mask"].shape[1]
        # 两条样本 pad 到同一 E = 当批最大实体数
        self.assertEqual(batch["entity_card_ids"].shape[1], E)
        self.assertEqual(int(batch["entity_mask"][0].sum()), len(row_small.entities))
        self.assertEqual(int(batch["entity_mask"][1].sum()), len(row_large.entities))
        # padding 位置 mask=False 且各 ID 为 0
        self.assertFalse(bool(batch["entity_mask"][0, len(row_small.entities)]))
        self.assertEqual(int(batch["entity_type_ids"][0, len(row_small.entities)]), 0)

    def test_deck_permutation_invariance_in_tensor(self) -> None:
        # 相同 deck 多重集合，DeckInstance 顺序不同 -> canonical tensor 相同
        from dataclasses import replace
        snap = _snapshot(deck_base_ids=[1000, 2000, 3000, 1000])
        shuffled_deck = replace(snap.me.deck_instances, draw_pile=tuple(reversed(snap.me.deck_instances.draw_pile)))
        snap2 = replace(snap, me=replace(snap.me, deck_instances=shuffled_deck))
        r1 = entity_tensorize_example(_example_from_snapshot(snap), vocabulary=_VOCAB)
        r2 = entity_tensorize_example(_example_from_snapshot(snap2), vocabulary=_VOCAB)
        b1 = collate_entity_rows([r1])
        b2 = collate_entity_rows([r2])
        for key in ("entity_card_ids", "entity_count", "entity_count_ratio", "entity_type_ids", "entity_zone_ids"):
            self.assertTrue(torch.equal(b1[key], b2[key]), msg=f"{key} not permutation-invariant")

    def test_empty_zones_still_have_valid_state_token(self) -> None:
        row = entity_tensorize_example(_example_from_snapshot(_snapshot()), vocabulary=_VOCAB)
        batch = collate_entity_rows([row])
        self.assertTrue(bool(batch["entity_mask"][0, 0]))
        self.assertEqual(int(batch["entity_type_ids"][0, 0]), ENTITY_TYPE_STATE)

    def test_dtypes_and_shapes(self) -> None:
        row = entity_tensorize_example(_example_from_snapshot(_snapshot(hand=(1,), deck_base_ids=[1000, 2000])), vocabulary=_VOCAB)
        batch = collate_entity_rows([row])
        self.assertEqual(batch["entity_card_ids"].dtype, torch.long)
        self.assertEqual(batch["entity_card_stats"].dtype, torch.float32)
        self.assertEqual(batch["entity_mask"].dtype, torch.bool)
        self.assertEqual(batch["entity_card_stats"].shape[-1], CARD_STAT_DIM)
        self.assertEqual(batch["entity_crest_stats"].shape[-1], CREST_STAT_DIM)


class SchemaGuardTest(unittest.TestCase):
    def test_roundtrip_jsonable_then_collate(self) -> None:
        row = entity_tensorize_example(_example_from_snapshot(_snapshot(hand=(1,), deck_base_ids=[1000])), vocabulary=_VOCAB)
        payload = row.to_jsonable()
        assert_entity_schema(payload)
        batch = collate_entity_rows([payload])
        self.assertEqual(int(batch["entity_mask"].sum()), len(row.entities))

    def test_wrong_schema_rejected(self) -> None:
        row = entity_tensorize_example(_example_from_snapshot(_snapshot(hand=(1,))), vocabulary=_VOCAB)
        payload = row.to_jsonable()
        payload["version"] = 999
        with self.assertRaises(EntitySchemaMismatch):
            collate_entity_rows([payload])

    def test_wrong_layout_rejected(self) -> None:
        row = entity_tensorize_example(_example_from_snapshot(_snapshot(hand=(1,))), vocabulary=_VOCAB)
        payload = row.to_jsonable()
        payload["entity_layout_version"] = ENTITY_LAYOUT_VERSION + 1
        with self.assertRaises(EntitySchemaMismatch):
            assert_entity_schema(payload)


if __name__ == "__main__":
    unittest.main()
