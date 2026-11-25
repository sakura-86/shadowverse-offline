# -*- coding: utf-8 -*-
"""C3/C4: Cross-Zone Entity Transformer + action cross-attention policy 测试。

覆盖 DECK_ENTITY_TRANSFORMER_IMPLEMENTATION_PLAN.md §C3/§C4 验收：

C3:
- forward 无 NaN/Inf，value ∈ [-1,1]。
- padding 内容变化不影响有效输出。
- deck/hand token 置换不改变输出（eval mode）。
- field 位置改变可改变 contextual representation。
- 全空区域下 [STATE] token 仍有效。

C4:
- policy logits shape 与 legal action count 一致。
- masked action 不参与 softmax，合法 prior 和约等于 1。
- 无 source / 无 target 动作不产生 NaN。
- 不同 padding 动作数不影响合法动作 logits。
"""

from __future__ import annotations

import unittest

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

from dataclasses import replace

from decision_engine.state.deck_instance import make_deck_instance_state
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView
from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.training.card_vocabulary import CardVocabulary
from decision_engine.training.state_features import state_features
from decision_engine.training.action_features import legal_actions_features
from decision_engine.training.entity_tensorization import (
    entity_tensorize_example,
    collate_entity_rows,
)

if _TORCH:
    from decision_engine.training.entity_policy_value import (
        EntityModelConfig,
        CrossZoneEntityTransformer,
    )


_VOCAB = CardVocabulary({1000: 2, 1001: 3, 2000: 4, 3000: 5})


def _deck_card(base_card_id: int) -> CardView:
    return CardView(uid=0, card_id=base_card_id + 900000, base_card_id=base_card_id, cost=2, card_type=CardType.FOLLOWER)


def _snapshot(hand=(), field=(), opp_field=(), deck_base_ids=None) -> BattleSnapshot:
    cards: dict[int, CardView] = {}
    for uid in hand + field + opp_field:
        cards.setdefault(uid, CardView(uid=uid, card_id=1000 + uid, base_card_id=1000, cost=2,
                                       card_type=CardType.FOLLOWER, atk=2, life=2, max_life=2, can_attack=True))
    di = None
    dc = 0
    if deck_base_ids is not None:
        di = make_deck_instance_state(tuple(_deck_card(b) for b in deck_base_ids), source="t")
        dc = len(deck_base_ids)
    me = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, hand=hand, field=field, deck_count=dc, deck_instances=di)
    opp = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, field=opp_field, deck_count=30)
    return BattleSnapshot(me=me, opponent=opp, cards_by_uid=cards, is_ally_turn=True)


def _batch(snapshot: BattleSnapshot, actions):
    example = {
        "state_features": state_features(snapshot, perspective="me"),
        "legal_action_features": legal_actions_features(snapshot, actions, perspective="me"),
        "policy_target": [1.0 / max(len(actions), 1)] * len(actions),
        "value_target": 0.3,
    }
    row = entity_tensorize_example(example, vocabulary=_VOCAB)
    return collate_entity_rows([row])


def _model(state_width: int, action_width: int) -> "CrossZoneEntityTransformer":
    torch.manual_seed(0)
    cfg = EntityModelConfig(d_model=32, num_layers=2, num_heads=4, ffn_dim=64)
    return CrossZoneEntityTransformer(
        vocab_size=_VOCAB.vocab_size, state_scalar_width=state_width,
        action_scalar_width=action_width, config=cfg,
    )


@unittest.skipUnless(_TORCH, "torch not available")
class EntityTransformerForwardTest(unittest.TestCase):
    def _actions(self, snapshot):
        return (
            LegalAction(kind=ActionKind.PLAY, source_uid=next(iter(snapshot.me.hand), 0), cost=2),
            LegalAction(kind=ActionKind.TURN_END, source_uid=0),
        )

    def test_forward_shapes_value_bound_no_nan(self) -> None:
        snap = _snapshot(hand=(1,), field=(2,), opp_field=(3,), deck_base_ids=[1000, 2000, 2000])
        actions = self._actions(snap)
        batch = _batch(snap, actions)
        model = _model(batch["state_scalar"].shape[1], batch["action_scalars"].shape[2])
        model.eval()
        out = model(batch)
        self.assertEqual(out["policy_logits"].shape, (1, len(actions)))
        self.assertEqual(out["value"].shape, (1,))
        self.assertTrue(torch.isfinite(out["value"]).all())
        self.assertGreaterEqual(float(out["value"].detach()), -1.0)
        self.assertLessEqual(float(out["value"].detach()), 1.0)
        # 合法 logits 有限
        legal = out["policy_logits"][batch["action_mask"]]
        self.assertTrue(torch.isfinite(legal).all())

    def test_backward_produces_gradients(self) -> None:
        snap = _snapshot(hand=(1,), deck_base_ids=[1000, 2000])
        actions = self._actions(snap)
        batch = _batch(snap, actions)
        model = _model(batch["state_scalar"].shape[1], batch["action_scalars"].shape[2])
        model.train()
        out = model(batch)
        loss = out["value"].sum() + out["policy_logits"][batch["action_mask"]].sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_all_zones_empty_state_token_valid(self) -> None:
        snap = _snapshot()  # 空场，仅 [STATE]
        actions = (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)
        batch = _batch(snap, actions)
        model = _model(batch["state_scalar"].shape[1], batch["action_scalars"].shape[2])
        model.eval()
        out = model(batch)
        self.assertTrue(torch.isfinite(out["value"]).all())

    def test_deck_permutation_invariance(self) -> None:
        snap = _snapshot(deck_base_ids=[1000, 2000, 3000, 1000])
        shuffled = replace(snap, me=replace(snap.me, deck_instances=replace(
            snap.me.deck_instances, draw_pile=tuple(reversed(snap.me.deck_instances.draw_pile)))))
        actions = (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)
        b1 = _batch(snap, actions)
        b2 = _batch(shuffled, actions)
        model = _model(b1["state_scalar"].shape[1], b1["action_scalars"].shape[2])
        model.eval()
        with torch.no_grad():
            v1 = model(b1)["value"]
            v2 = model(b2)["value"]
        self.assertTrue(torch.allclose(v1, v2, atol=1e-5))

    def test_field_position_sensitivity(self) -> None:
        # 两张内容不同的 field 卡，交换顺序 -> (content, position) 绑定改变 ->
        # contextual representation 允许不同（position embedding 生效）。
        cards = {
            2: CardView(uid=2, card_id=1002, base_card_id=1000, cost=2, atk=5, life=1, max_life=1, card_type=CardType.FOLLOWER),
            3: CardView(uid=3, card_id=2003, base_card_id=2000, cost=3, atk=1, life=5, max_life=5, card_type=CardType.FOLLOWER),
        }
        me_a = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, field=(2, 3), deck_count=0)
        me_b = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, field=(3, 2), deck_count=0)
        opp = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, deck_count=30)
        snap_a = BattleSnapshot(me=me_a, opponent=opp, cards_by_uid=cards, is_ally_turn=True)
        snap_b = BattleSnapshot(me=me_b, opponent=opp, cards_by_uid=cards, is_ally_turn=True)
        actions = (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)
        ba = _batch(snap_a, actions)
        bb = _batch(snap_b, actions)
        model = _model(ba["state_scalar"].shape[1], ba["action_scalars"].shape[2])
        model.eval()
        with torch.no_grad():
            ctx_a, _ = model.contextual_tokens(ba)
            ctx_b, _ = model.contextual_tokens(bb)
        # 交换顺序后 field token 的 contextual 表示不应完全相同
        self.assertFalse(torch.allclose(ctx_a, ctx_b, atol=1e-6))

    def test_padding_content_does_not_affect_valid_output(self) -> None:
        # 两条样本 batch：小样本的有效输出不应被大样本的 padding 影响
        snap_small = _snapshot(hand=(1,))
        snap_large = _snapshot(hand=(1, 2), field=(3,), deck_base_ids=[1000, 2000, 3000])
        actions = (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)
        r_small = entity_tensorize_example({
            "state_features": state_features(snap_small, perspective="me"),
            "legal_action_features": legal_actions_features(snap_small, actions, perspective="me"),
            "policy_target": [1.0], "value_target": 0.0,
        }, vocabulary=_VOCAB)
        r_large = entity_tensorize_example({
            "state_features": state_features(snap_large, perspective="me"),
            "legal_action_features": legal_actions_features(snap_large, actions, perspective="me"),
            "policy_target": [1.0], "value_target": 0.0,
        }, vocabulary=_VOCAB)
        model = _model(len(r_small.state_vector), len(r_small.action_scalar_vectors[0]))
        model.eval()
        with torch.no_grad():
            solo = model(collate_entity_rows([r_small]))["value"]
            mixed = model(collate_entity_rows([r_small, r_large]))["value"]
        self.assertTrue(torch.allclose(solo[0], mixed[0], atol=1e-5))


@unittest.skipUnless(_TORCH, "torch not available")
class EntityPolicyTest(unittest.TestCase):
    def test_masked_action_excluded_from_softmax(self) -> None:
        snap = _snapshot(hand=(1,))
        actions = (LegalAction(kind=ActionKind.PLAY, source_uid=1, cost=2),
                   LegalAction(kind=ActionKind.TURN_END, source_uid=0))
        # 手工构造带 padding 动作的两条 batch
        r = entity_tensorize_example({
            "state_features": state_features(snap, perspective="me"),
            "legal_action_features": legal_actions_features(snap, actions, perspective="me"),
            "policy_target": [0.5, 0.5], "value_target": 0.0,
        }, vocabulary=_VOCAB)
        r_pad = entity_tensorize_example({
            "state_features": state_features(snap, perspective="me"),
            "legal_action_features": legal_actions_features(snap, actions[:1], perspective="me"),
            "policy_target": [1.0], "value_target": 0.0,
        }, vocabulary=_VOCAB)
        model = _model(len(r.state_vector), len(r.action_scalar_vectors[0]))
        model.eval()
        batch = collate_entity_rows([r, r_pad])  # 第二条只有 1 个动作，会被 pad 到 2
        with torch.no_grad():
            out = model(batch)
        logits = out["policy_logits"]
        mask = batch["action_mask"]
        # 对第二条样本，padding 动作 logit 应为 -inf 且 softmax 概率≈0
        probs = torch.softmax(logits, dim=1)
        self.assertAlmostEqual(float(probs[1][mask[1]].sum()), 1.0, places=4)
        self.assertAlmostEqual(float(probs[1][~mask[1]].sum()), 0.0, places=5)

    def test_no_target_action_no_nan(self) -> None:
        snap = _snapshot()
        actions = (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)
        batch = _batch(snap, actions)
        model = _model(batch["state_scalar"].shape[1], batch["action_scalars"].shape[2])
        model.eval()
        with torch.no_grad():
            out = model(batch)
        self.assertTrue(torch.isfinite(out["policy_logits"][batch["action_mask"]]).all())

    def test_deterministic_eval(self) -> None:
        snap = _snapshot(hand=(1,), deck_base_ids=[1000, 2000])
        actions = (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)
        batch = _batch(snap, actions)
        model = _model(batch["state_scalar"].shape[1], batch["action_scalars"].shape[2])
        model.eval()
        with torch.no_grad():
            a = model(batch)["value"]
            b = model(batch)["value"]
        self.assertTrue(torch.equal(a, b))

    def test_config_validate_rejects_bad_head_div(self) -> None:
        with self.assertRaises(ValueError):
            EntityModelConfig(d_model=30, num_heads=4).validate()


if __name__ == "__main__":
    unittest.main()
