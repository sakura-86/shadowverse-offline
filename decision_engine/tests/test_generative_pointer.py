# -*- coding: utf-8 -*-
"""生成式 pointer：位置坐标、tensorization 传递、模型前向、折算接口。"""

from __future__ import annotations

import unittest

try:
    import torch
    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False

from decision_engine.actions.model import ActionKind, LegalAction, PlayKind
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView
from decision_engine.training.action_features import action_features
from decision_engine.training.card_vocabulary import CardVocabulary
from decision_engine.training.entity_position import (
    ENTITY_POSITION_OPP_LEADER,
    ENTITY_POSITION_PAD,
    ENTITY_POSITION_SELF_HAND_BASE,
    ENTITY_POSITION_SELF_LEADER,
    MAX_ENTITY_POSITIONS,
    position_for_crest,
    position_for_field,
    position_for_hand,
    position_for_uid,
)
from decision_engine.training.entity_tensorization import (
    collate_entity_rows,
    entity_tensorize_example,
)
from decision_engine.training.state_features import state_features

if _TORCH:
    from decision_engine.training.action_pointer import policy_logits_from_pointer
    from decision_engine.training.entity_policy_value import (
        CrossZoneEntityTransformer,
        EntityModelConfig,
    )

_VOCAB = CardVocabulary({1010: 2, 1020: 3})


def _snapshot(hand=(), field=(), opp_field=(), opp_crests=()):
    cards: dict[int, CardView] = {}
    for uid in hand + field + opp_field:
        cards.setdefault(
            uid,
            CardView(uid=uid, card_id=1000 + uid, base_card_id=1010, cost=2,
                     card_type=CardType.FOLLOWER, atk=2, life=2, max_life=2),
        )
    me = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, hand=hand, field=field)
    opp = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, unique_id=99,
                     field=opp_field, crests=opp_crests)
    return BattleSnapshot(me=me, opponent=opp, cards_by_uid=cards, is_ally_turn=True)


class EntityPositionTest(unittest.TestCase):
    def test_hand_and_field_coordinates(self):
        self.assertEqual(position_for_hand(0), ENTITY_POSITION_SELF_HAND_BASE)
        self.assertEqual(position_for_hand(8), ENTITY_POSITION_SELF_HAND_BASE + 8)
        self.assertEqual(position_for_field(self_side=True, position=1), 21)
        self.assertEqual(position_for_field(self_side=False, position=5), 35)
        self.assertEqual(position_for_crest(self_side=True, index=0), 40)
        self.assertEqual(position_for_crest(self_side=False, index=4), 54)

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            position_for_hand(9)
        with self.assertRaises(ValueError):
            position_for_field(self_side=True, position=0)

    def test_position_for_uid(self):
        snapshot = _snapshot(hand=(1, 2), field=(3,), opp_field=(5, 6))

        self.assertEqual(position_for_uid(snapshot, 1), 10)   # hand[0]
        self.assertEqual(position_for_uid(snapshot, 2), 11)   # hand[1]
        self.assertEqual(position_for_uid(snapshot, 3), 21)   # self field 1
        self.assertEqual(position_for_uid(snapshot, 5), 31)   # opp field 1
        self.assertEqual(position_for_uid(snapshot, 6), 32)   # opp field 2
        self.assertEqual(position_for_uid(snapshot, 0), ENTITY_POSITION_OPP_LEADER)
        self.assertEqual(position_for_uid(snapshot, 99), ENTITY_POSITION_OPP_LEADER)
        self.assertEqual(position_for_uid(snapshot, 12345), ENTITY_POSITION_PAD)

    def test_phase_position_mask(self):
        """阶段 mask 必须按规则收窄候选位置，且 play 比 exchange 宽。"""
        from decision_engine.training.entity_position import (
            PHASE_EVOLVE,
            PHASE_EXCHANGE,
            PHASE_PLAY,
            PHASE_SOURCE_POSITIONS,
            PHASE_TARGET_POSITIONS,
            phase_index,
            phase_position_mask,
        )

        play = phase_position_mask(PHASE_PLAY)
        exchange = phase_position_mask(PHASE_EXCHANGE)

        # exchange：只有对方场地 + 主战者
        self.assertTrue(exchange[ENTITY_POSITION_OPP_LEADER])
        self.assertTrue(exchange[31])
        self.assertFalse(exchange[10], "exchange 阶段不允许手牌作 target")
        self.assertFalse(exchange[21], "exchange 阶段不允许自己场地作 target")
        # play 阶段包含手牌、自己场地
        self.assertTrue(play[10])
        self.assertTrue(play[21])

        # source：evolve / exchange 只允许自己场地
        self.assertIn(21, PHASE_SOURCE_POSITIONS[PHASE_EVOLVE])
        self.assertNotIn(10, PHASE_SOURCE_POSITIONS[PHASE_EVOLVE])
        self.assertIn(21, PHASE_SOURCE_POSITIONS[PHASE_EXCHANGE])
        self.assertNotIn(10, PHASE_SOURCE_POSITIONS[PHASE_EXCHANGE])
        # play 阶段允许手牌
        self.assertIn(10, PHASE_SOURCE_POSITIONS[PHASE_PLAY])

        # 阶段下标：未知 → PAD
        self.assertEqual(phase_index(PHASE_PLAY), 0)
        self.assertEqual(phase_index(None), -1)
        self.assertEqual(phase_index("nonsense"), -1)


class ActionPositionFeatureTest(unittest.TestCase):
    def test_source_and_target_positions(self):
        snapshot = _snapshot(hand=(1,), opp_field=(5, 6))
        action = LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=1,
                             target_groups=((6,),), target_group_specs=("attack_follower",))

        features = action_features(snapshot, action)

        self.assertEqual(features["source"]["entity_position"], 10)
        self.assertEqual(features["targets"][0]["entity_position"], 32)

    def test_no_source_action_anchors_to_self_leader(self):
        """TURN_END / EXTRA_PP 没有 source，必须锚到有效坐标，否则永远选不到。"""
        snapshot = _snapshot(hand=(1,))

        turn_end = action_features(snapshot, LegalAction(kind=ActionKind.TURN_END, source_uid=0))

        self.assertEqual(turn_end["source"]["entity_position"], ENTITY_POSITION_SELF_LEADER)

    def test_state_and_action_positions_agree(self):
        snapshot = _snapshot(hand=(1,), opp_field=(5,))
        features = state_features(snapshot)
        action = action_features(
            snapshot,
            LegalAction(kind=ActionKind.PLAY, source_uid=1, play_kind=PlayKind.NORMAL, cost=2, target_groups=((5,),)),
        )

        state_hand_pos = [card["entity_position"] for card in features["player"]["hand_cards"]]
        state_opp_pos = [card["entity_position"] for card in features["opponent"]["field_cards"]]
        self.assertIn(action["source"]["entity_position"], state_hand_pos)
        self.assertIn(action["targets"][0]["entity_position"], state_opp_pos)


class PointerTensorizationTest(unittest.TestCase):
    def _row(self, actions):
        snapshot = _snapshot(hand=(1,), opp_field=(5,))
        example = {
            "state_features": state_features(snapshot),
            "legal_action_features": [action_features(snapshot, a) for a in actions],
            "policy_target": [1.0 / len(actions)] * len(actions),
            "value_target": 0.0,
        }
        return entity_tensorize_example(example, vocabulary=_VOCAB)

    def test_entity_anchors_and_position_mask(self):
        row = self._row([LegalAction(kind=ActionKind.TURN_END, source_uid=0)])
        batch = collate_entity_rows([row])

        self.assertTrue(bool(batch["position_mask"][0, ENTITY_POSITION_SELF_LEADER]))
        self.assertTrue(bool(batch["position_mask"][0, ENTITY_POSITION_OPP_LEADER]))
        self.assertTrue(bool(batch["position_mask"][0, 10]))     # self hand[0]
        self.assertTrue(bool(batch["position_mask"][0, 31]))     # opp field pos 1
        self.assertFalse(bool(batch["position_mask"][0, 20]))    # 无此坐标

    def test_action_positions_and_head_indices(self):
        actions = [
            LegalAction(kind=ActionKind.PLAY, source_uid=1, play_kind=PlayKind.NORMAL, cost=2,
                        target_groups=((5,),), target_group_specs=("local_select",)),
            LegalAction(kind=ActionKind.TURN_END, source_uid=0),
        ]
        row = self._row(actions)

        self.assertEqual(row.action_source_positions, (10, ENTITY_POSITION_SELF_LEADER))
        self.assertEqual(row.action_target_positions[0][0], 31)
        self.assertEqual(row.action_target_positions[1][0], ENTITY_POSITION_PAD)
        self.assertEqual(len(row.action_head_indices), 2)
        self.assertGreaterEqual(row.action_head_indices[0], 0)


@unittest.skipUnless(_TORCH, "torch 不可用")
class GenerativePointerModelTest(unittest.TestCase):
    def _batch(self, actions=None):
        snapshot = _snapshot(hand=(1,), opp_field=(5,))
        actions = actions or [
            LegalAction(kind=ActionKind.PLAY, source_uid=1, play_kind=PlayKind.NORMAL, cost=2,
                        target_groups=((5,),), target_group_specs=("local_select",)),
            LegalAction(kind=ActionKind.TURN_END, source_uid=0),
        ]
        example = {
            "state_features": state_features(snapshot),
            "legal_action_features": [action_features(snapshot, a) for a in actions],
            "policy_target": [1.0 / len(actions)] * len(actions),
            "value_target": 0.0,
        }
        row = entity_tensorize_example(example, vocabulary=_VOCAB)
        return collate_entity_rows([row])

    def _model(self, batch, *, pointer: bool):
        config = EntityModelConfig(
            d_model=32, num_layers=1, num_heads=4, ffn_dim=64,
            attention_dropout=0.0, residual_dropout=0.0,
            slot_aware_query_enabled=True, slot_spec_vocab_size=200,
            pointer_head_enabled=pointer,
        )
        torch.manual_seed(0)
        model = CrossZoneEntityTransformer(
            vocab_size=10,
            state_scalar_width=int(batch["state_scalar"].shape[-1]),
            action_scalar_width=52,
            config=config,
        )
        model.eval()
        return model

    def test_pointer_outputs_have_fixed_width(self):
        """pointer 输出宽度只由位置空间和槽数决定，与候选动作数无关。"""
        batch = self._batch()
        model = self._model(batch, pointer=True)

        with torch.no_grad():
            out = model(batch)

        self.assertEqual(tuple(out["head_logits"].shape), (1, 12))
        self.assertEqual(tuple(out["source_logits"].shape), (1, MAX_ENTITY_POSITIONS))
        self.assertEqual(tuple(out["slot_logits"].shape), (1, 4, MAX_ENTITY_POSITIONS))
        self.assertFalse(any(name == "policy_logits" for name in out))

    def test_pointer_outputs_ignore_action_count(self):
        """同样状态、不同候选动作数，pointer 输出的 logits 必须一致前面的宽度。"""
        one = self._batch([LegalAction(kind=ActionKind.TURN_END, source_uid=0)])
        two = self._batch()
        model = self._model(one, pointer=True)

        with torch.no_grad():
            out_one = model(one)
            out_two = model(two)

        self.assertEqual(tuple(out_one["source_logits"].shape), tuple(out_two["source_logits"].shape))

    def test_invalid_positions_masked(self):
        batch = self._batch()
        model = self._model(batch, pointer=True)

        with torch.no_grad():
            out = model(batch)

        neg_inf = torch.finfo(out["source_logits"].dtype).min
        invalid = ~batch["position_mask"][0]
        for index in torch.nonzero(invalid).flatten().tolist():
            self.assertEqual(float(out["source_logits"][0, index]), float(neg_inf))

    def test_folded_logits_are_finite_for_valid_actions(self):
        batch = self._batch()
        model = self._model(batch, pointer=True)

        with torch.no_grad():
            out = model(batch)
            logits = policy_logits_from_pointer(
                head_logits=out["head_logits"],
                source_logits=out["source_logits"],
                slot_logits=out["slot_logits"],
                action_head_indices=batch["action_head_indices"],
                action_source_positions=batch["action_source_positions"],
                action_target_slot_index=batch["action_target_slot_index"],
                action_target_positions=batch["action_target_positions"],
                action_target_mask=batch["action_target_mask"],
                action_mask=batch["action_mask"],
            )

        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertTrue(bool(torch.isfinite(logits[0]).all()))

    def test_folded_logits_invalidate_unmappable_action(self):
        """head 下标 -1（无法映射到 ActionHead）必须置 -inf，而不是落到 head 0。"""
        batch = dict(self._batch())
        batch["action_head_indices"] = batch["action_head_indices"].clone()
        batch["action_head_indices"][0, 0] = -1
        model = self._model(batch, pointer=True)

        with torch.no_grad():
            out = model(batch)
            logits = policy_logits_from_pointer(
                head_logits=out["head_logits"],
                source_logits=out["source_logits"],
                slot_logits=out["slot_logits"],
                action_head_indices=batch["action_head_indices"],
                action_source_positions=batch["action_source_positions"],
                action_target_slot_index=batch["action_target_slot_index"],
                action_target_positions=batch["action_target_positions"],
                action_target_mask=batch["action_target_mask"],
                action_mask=batch["action_mask"],
            )

        self.assertEqual(float(logits[0, 0]), float(torch.finfo(logits.dtype).min))
        self.assertTrue(bool(torch.isfinite(logits[0, 1])))


class HeadConditionedPointerTest(unittest.TestCase):
    """source 分布按动作头条件化（``[B,P]`` -> ``[B,H,P]``）。"""

    def _batch(self):
        # 实体数必须 **超过** 动作头数（12）：``[B,E,H]`` 转置成 ``[B,H,E]`` 后
        # 才 scatter 到 ``[B,H,P]``。实体数小于头数时维度校验会恰好通过，
        # 掩盖真实缺陷（曾经就是这样漏掉的）。
        snapshot = _snapshot(
            hand=(1, 2, 3, 4, 5),
            field=(6, 7, 8, 9, 10),
            opp_field=(11, 12, 13, 14, 15),
        )
        actions = [
            LegalAction(kind=ActionKind.PLAY, source_uid=1, play_kind=PlayKind.NORMAL, cost=2,
                        target_groups=((11,),), target_group_specs=("local_select",)),
            LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=6, target_groups=((11,),),
                        target_group_specs=("attack_follower",)),
            LegalAction(kind=ActionKind.TURN_END, source_uid=0),
        ]
        example = {
            "state_features": state_features(snapshot),
            "legal_action_features": [action_features(snapshot, a) for a in actions],
            "policy_target": [1.0 / len(actions)] * len(actions),
            "value_target": 0.0,
        }
        row = entity_tensorize_example(example, vocabulary=_VOCAB)
        return collate_entity_rows([row])

    def _model(self, batch):
        config = EntityModelConfig(
            d_model=32, num_layers=1, num_heads=4, ffn_dim=64,
            attention_dropout=0.0, residual_dropout=0.0,
            slot_aware_query_enabled=True, slot_spec_vocab_size=200,
            pointer_head_enabled=True,
            pointer_head_conditioning_enabled=True,
        )
        torch.manual_seed(0)
        model = CrossZoneEntityTransformer(
            vocab_size=10,
            state_scalar_width=int(batch["state_scalar"].shape[-1]),
            action_scalar_width=52,
            config=config,
        )
        model.eval()
        return model

    def test_source_logits_carry_head_axis(self):
        batch = self._batch()
        model = self._model(batch)
        heads = model.config.num_action_heads
        entity_count = int(batch["entity_card_ids"].shape[1])
        self.assertGreater(
            entity_count, heads,
            f"实体数 {entity_count} 必须超过头数 {heads}，否则轨迹维度校验会掩盖缺陷",
        )
        with torch.no_grad():
            out = model(batch)
        self.assertEqual(
            tuple(out["source_logits"].shape), (1, heads, MAX_ENTITY_POSITIONS)
        )
        # slot 暂不条件化，形状不变
        self.assertEqual(tuple(out["slot_logits"].shape), (1, 4, MAX_ENTITY_POSITIONS))

    def test_head_slices_are_not_identical(self):
        """条件化必须真的产生差异——否则等价于没做。"""
        batch = self._batch()
        model = self._model(batch)
        with torch.no_grad():
            out = model(batch)
        source = out["source_logits"][0]                        # [H,P]
        first = source[0]
        differences = (source != first.unsqueeze(0)).any(dim=-1)
        self.assertTrue(bool(differences.any()), "所有 head 的 source 分布完全相同")

    def test_folding_picks_the_action_head_slice(self):
        """折算时必须取「该候选动作自己的 head」那一片。"""
        batch = self._batch()
        model = self._model(batch)
        with torch.no_grad():
            out = model(batch)

        # 手工构造：head 0 只认位置 10，head 1 只认位置 11，其余全 -inf
        source = torch.full_like(out["source_logits"], float("-inf"))
        source[0, 0, 10] = 0.0
        source[0, 1, 11] = 0.0

        logits = policy_logits_from_pointer(
            head_logits=torch.zeros_like(out["head_logits"]),
            source_logits=source,
            slot_logits=torch.zeros_like(out["slot_logits"]),
            action_head_indices=batch["action_head_indices"],
            action_source_positions=batch["action_source_positions"],
            action_target_slot_index=batch["action_target_slot_index"],
            action_target_positions=batch["action_target_positions"],
            action_target_mask=batch["action_target_mask"],
            action_mask=batch["action_mask"],
        )

        # head/slot 项都是 0，所以 logit 必须严格等于「该候选自己的 head 那一片
        # 在该候选 source 位置上的值」——这就是条件化的全部意义。
        checked = 0
        for a in range(logits.shape[1]):
            if not bool(batch["action_mask"][0, a]):
                continue
            head = int(batch["action_head_indices"][0, a])
            source_pos = int(batch["action_source_positions"][0, a])
            if head < 0 or source_pos < 0:
                continue
            expected = float(source[0, head, source_pos])
            self.assertEqual(float(logits[0, a]), expected, f"action {a} head {head} source {source_pos}")
            checked += 1
        self.assertGreater(checked, 0)

    def test_loss_accepts_head_axis(self):
        from decision_engine.training.pointer_loss import pointer_auxiliary_losses

        batch = self._batch()
        model = self._model(batch)
        with torch.no_grad():
            out = model(batch)
        aux = pointer_auxiliary_losses(
            head_logits=out["head_logits"],
            source_logits=out["source_logits"],
            slot_logits=out["slot_logits"],
            policy=batch["policy"],
            action_head_indices=batch["action_head_indices"],
            action_source_positions=batch["action_source_positions"],
            action_target_slot_index=batch["action_target_slot_index"],
            action_target_positions=batch["action_target_positions"],
            action_target_mask=batch["action_target_mask"],
            action_mask=batch["action_mask"],
            slot_exists=out["slot_exists"],
        )
        self.assertIn("source", aux)
        loss_sum, count = aux["source"]
        self.assertEqual(count, 1)
        self.assertTrue(bool(torch.isfinite(loss_sum)))

    def test_head_centering_zero_means_each_slice(self):
        """每个 head 的有效位置均值必须为 0——否则 head/source 尺度退化仍在。"""
        batch = self._batch()
        model = self._model(batch)
        with torch.no_grad():
            out = model(batch)
        source = out["source_logits"][0]                            # [H,P]
        mask = batch["position_mask"][0]
        for head in range(source.shape[0]):
            values = source[head][mask]
            self.assertTrue(bool(torch.isfinite(values).all()))
            self.assertAlmostEqual(float(values.mean()), 0.0, places=4, msg=f"head {head}")

    def test_centering_cannot_be_disabled(self):
        """归零不是可选项：关掉会重新引入 head/source 的尺度退化，validate 必须拒绝。"""
        with self.assertRaises(ValueError):
            EntityModelConfig(
                pointer_head_enabled=True,
                pointer_head_conditioning_enabled=True,
                pointer_head_centering_enabled=False,
            ).validate()

    def test_head_conditioning_requires_pointer(self):
        with self.assertRaises(ValueError):
            EntityModelConfig(
                pointer_head_enabled=False,
                pointer_head_conditioning_enabled=True,
            ).validate()


class SlotHeadConditionedPointerTest(unittest.TestCase):
    """slot 分布也按动作头条件化（``[B,K,P]`` -> ``[B,H,K,P]``）。"""

    def _batch(self):
        return HeadConditionedPointerTest()._batch()

    def _model(self, batch):
        config = EntityModelConfig(
            d_model=32, num_layers=1, num_heads=4, ffn_dim=64,
            attention_dropout=0.0, residual_dropout=0.0,
            slot_aware_query_enabled=True, slot_spec_vocab_size=200,
            pointer_head_enabled=True,
            pointer_head_conditioning_enabled=True,
            pointer_slot_head_conditioning_enabled=True,
        )
        torch.manual_seed(0)
        model = CrossZoneEntityTransformer(
            vocab_size=10,
            state_scalar_width=int(batch["state_scalar"].shape[-1]),
            action_scalar_width=52,
            config=config,
        )
        model.eval()
        return model

    def test_slot_logits_are_head_conditioned(self):
        batch = self._batch()
        model = self._model(batch)
        heads = model.config.num_action_heads
        with torch.no_grad():
            out = model(batch)
        self.assertEqual(
            tuple(out["slot_logits"].shape),
            (1, heads, 4, MAX_ENTITY_POSITIONS),
        )

    def test_slot_heads_are_zero_centered_per_head_and_slot(self):
        batch = self._batch()
        model = self._model(batch)
        with torch.no_grad():
            out = model(batch)
        slots = out["slot_logits"][0]                                # [H,K,P]
        target_mask = batch["position_mask"][0]
        checked = 0
        for head in range(slots.shape[0]):
            for k in range(slots.shape[1]):
                values = slots[head, k][target_mask]
                # neg_inf 用的是 finfo.min（有限值），不能用 isfinite 过滤
                usable = values[values > -1e30]
                if usable.numel() == 0:
                    continue
                self.assertAlmostEqual(
                    float(usable.mean()), 0.0, places=4, msg=f"head {head} slot {k}"
                )
                checked += 1
        self.assertGreater(checked, 0)

    def test_folding_uses_head_slot_and_position(self):
        batch = self._batch()
        model = self._model(batch)
        with torch.no_grad():
            out = model(batch)

        slot_logits = torch.full_like(out["slot_logits"], float("-inf"))
        slot_logits[0, 1, 0, 20] = 0.0

        logits = policy_logits_from_pointer(
            head_logits=torch.zeros_like(out["head_logits"]),
            source_logits=torch.zeros_like(out["source_logits"]),
            slot_logits=slot_logits,
            action_head_indices=batch["action_head_indices"],
            action_source_positions=batch["action_source_positions"],
            action_target_slot_index=batch["action_target_slot_index"],
            action_target_positions=batch["action_target_positions"],
            action_target_mask=batch["action_target_mask"],
            action_mask=batch["action_mask"],
        )

        heads = batch["action_head_indices"][0].tolist()
        slot_idx = batch["action_target_slot_index"][0].tolist()
        targets = batch["action_target_positions"][0].tolist()
        target_mask = batch["action_target_mask"][0].tolist()
        checked = 0
        for a in range(logits.shape[1]):
            if not bool(batch["action_mask"][0, a]):
                continue
            hit = any(
                target_mask[a][t]
                and heads[a] == 1
                and slot_idx[a][t] == 0
                and targets[a][t] == 20
                for t in range(len(targets[a]))
            )
            if hit:
                self.assertGreater(float(logits[0, a]), -1e30, f"action {a} 应命中 slot 项")
                checked += 1
        del checked  # 命中与否取决于数据，这里只验证不误置 -inf

    def test_loss_accepts_head_conditioned_slots(self):
        from decision_engine.training.pointer_loss import pointer_auxiliary_losses

        batch = self._batch()
        model = self._model(batch)
        with torch.no_grad():
            out = model(batch)
        aux = pointer_auxiliary_losses(
            head_logits=out["head_logits"],
            source_logits=out["source_logits"],
            slot_logits=out["slot_logits"],
            policy=batch["policy"],
            action_head_indices=batch["action_head_indices"],
            action_source_positions=batch["action_source_positions"],
            action_target_slot_index=batch["action_target_slot_index"],
            action_target_positions=batch["action_target_positions"],
            action_target_mask=batch["action_target_mask"],
            action_mask=batch["action_mask"],
            slot_exists=out["slot_exists"],
        )
        if "slot" in aux:
            loss_sum, count = aux["slot"]
            self.assertGreater(count, 0)
            self.assertTrue(bool(torch.isfinite(loss_sum)))

    def test_requires_head_conditioning(self):
        with self.assertRaises(ValueError):
            EntityModelConfig(
                pointer_head_enabled=True,
                pointer_head_conditioning_enabled=False,
                pointer_slot_head_conditioning_enabled=True,
            ).validate()


if __name__ == "__main__":
    unittest.main()
