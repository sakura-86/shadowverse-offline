# -*- coding: utf-8 -*-
"""C7.4/C7.5: deck counterfactual suite（方案 C）。

验收：
- 同一可见场面、不同剩余牌组构成 -> entity 输入可区分（deck token 不同）。
- 相关固定场景（低费守护剩余构成变化）-> 输入上下文改变。
- 模型不改写合法动作集合（action count / mask 不因 deck 变化而改变）。
- 相同剩余多重集合、不同 draw order -> 输入完全相同（无泄漏）。
"""

from __future__ import annotations

import unittest
from dataclasses import replace

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

from decision_engine.state.deck_instance import make_deck_instance_state
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView
from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.training.card_vocabulary import CardVocabulary
from decision_engine.planning.entity_features import (
    entity_batch_from_snapshot,
    entity_example_from_snapshot,
    entity_snapshot_fingerprint,
)

# 10264110/10264120/10264122 系列在计划 C7.5 中作为固定相关场景；
# 这里用可控 vocab 表示低费守护相关的剩余构成变化。
_GUARD_LOW = 10264110
_GUARD_MID = 10264120
_OTHER = 20000
_VOCAB = CardVocabulary({_GUARD_LOW: 2, _GUARD_MID: 3, _OTHER: 4, 1000: 5})


def _deck_card(b: int) -> CardView:
    return CardView(uid=0, card_id=b + 900000, base_card_id=b, cost=1, card_type=CardType.FOLLOWER, has_guard=True)


def _fixed_board_snapshot(deck_base_ids) -> BattleSnapshot:
    """固定可见场面（手牌 1 张、场上 1 张），只改变剩余牌组。"""
    cards = {
        1: CardView(uid=1, card_id=1001, base_card_id=1000, cost=2, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER),
        2: CardView(uid=2, card_id=1002, base_card_id=1000, cost=2, atk=3, life=3, max_life=3, card_type=CardType.FOLLOWER, can_attack=True),
    }
    di = make_deck_instance_state(tuple(_deck_card(b) for b in deck_base_ids), source="cf")
    me = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=4, hand=(1,), field=(2,),
                    deck_count=len(deck_base_ids), deck_instances=di)
    opp = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=4, deck_count=30)
    return BattleSnapshot(me=me, opponent=opp, cards_by_uid=cards, is_ally_turn=True)


def _actions():
    return (LegalAction(kind=ActionKind.PLAY, source_uid=1, cost=2),
            LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=2),
            LegalAction(kind=ActionKind.TURN_END, source_uid=0))


class DeckCounterfactualInputTest(unittest.TestCase):
    def test_same_board_different_deck_distinguishable(self) -> None:
        # 多守护 vs 少守护剩余构成
        rowA = entity_example_from_snapshot(
            _fixed_board_snapshot([_GUARD_LOW, _GUARD_LOW, _GUARD_MID, _OTHER]), _actions(),
            perspective="me", vocabulary=_VOCAB)
        rowB = entity_example_from_snapshot(
            _fixed_board_snapshot([_OTHER, _OTHER, _OTHER, _OTHER]), _actions(),
            perspective="me", vocabulary=_VOCAB)
        deckA = [t for t in rowA.entities if t.zone_id == 5]  # ZONE_DECK
        deckB = [t for t in rowB.entities if t.zone_id == 5]
        self.assertNotEqual(
            {(t.card_id_index, t.count) for t in deckA},
            {(t.card_id_index, t.count) for t in deckB},
        )

    def test_fingerprint_changes_with_guard_composition(self) -> None:
        a = entity_snapshot_fingerprint(_fixed_board_snapshot([_GUARD_LOW, _GUARD_LOW]), "me", vocabulary=_VOCAB)
        b = entity_snapshot_fingerprint(_fixed_board_snapshot([_GUARD_LOW, _OTHER]), "me", vocabulary=_VOCAB)
        self.assertNotEqual(a, b)

    def test_legal_action_set_unchanged_by_deck(self) -> None:
        actions = _actions()
        batchA = entity_batch_from_snapshot(_fixed_board_snapshot([_GUARD_LOW, _GUARD_MID]), actions, perspective="me", vocabulary=_VOCAB)
        batchB = entity_batch_from_snapshot(_fixed_board_snapshot([_OTHER, _OTHER]), actions, perspective="me", vocabulary=_VOCAB)
        # 动作数与 mask 不因剩余牌组不同而改变（模型不改写合法动作集）
        self.assertEqual(batchA["action_mask"].shape, batchB["action_mask"].shape)
        if _TORCH:
            self.assertTrue(torch.equal(batchA["action_mask"], batchB["action_mask"]))

    def test_draw_order_permutation_no_effect(self) -> None:
        snap = _fixed_board_snapshot([_GUARD_LOW, _GUARD_MID, _OTHER, _GUARD_LOW])
        shuffled = replace(snap, me=replace(snap.me, deck_instances=replace(
            snap.me.deck_instances, draw_pile=tuple(reversed(snap.me.deck_instances.draw_pile)))))
        r1 = entity_example_from_snapshot(snap, _actions(), perspective="me", vocabulary=_VOCAB)
        r2 = entity_example_from_snapshot(shuffled, _actions(), perspective="me", vocabulary=_VOCAB)
        self.assertEqual(r1.to_jsonable(), r2.to_jsonable())


@unittest.skipUnless(_TORCH, "torch not available")
class DeckCounterfactualModelTest(unittest.TestCase):
    def test_model_output_can_differ_across_deck_composition(self) -> None:
        from decision_engine.training.entity_policy_value import EntityModelConfig, CrossZoneEntityTransformer
        torch.manual_seed(0)
        actions = _actions()
        bA = entity_batch_from_snapshot(_fixed_board_snapshot([_GUARD_LOW, _GUARD_LOW, _GUARD_MID]), actions, perspective="me", vocabulary=_VOCAB)
        bB = entity_batch_from_snapshot(_fixed_board_snapshot([_OTHER, _OTHER, _OTHER]), actions, perspective="me", vocabulary=_VOCAB)
        state_w = bA["state_scalar"].shape[1]
        action_w = bA["action_scalars"].shape[2]
        model = CrossZoneEntityTransformer(
            vocab_size=_VOCAB.vocab_size, state_scalar_width=state_w,
            action_scalar_width=action_w, config=EntityModelConfig(d_model=32, num_layers=2, num_heads=4, ffn_dim=64),
        )
        model.eval()
        with torch.no_grad():
            outA = model(bA)
            outB = model(bB)
        # deck token 进入 cross-attention，随机初始化模型的 value/policy 允许因 deck 不同而不同
        differs = (not torch.allclose(outA["value"], outB["value"], atol=1e-6)) or \
                  (not torch.allclose(outA["policy_logits"], outB["policy_logits"], atol=1e-6))
        self.assertTrue(differs, "deck composition 未能影响模型输出（deck token 未进入上下文？）")


if __name__ == "__main__":
    unittest.main()
