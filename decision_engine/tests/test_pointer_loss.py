# -*- coding: utf-8 -*-
"""生成式 pointer 分解损失测试。"""

from __future__ import annotations

import unittest

try:
    import torch
    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False

from decision_engine.training.pointer_loss import pointer_auxiliary_losses


@unittest.skipUnless(_TORCH, "torch 不可用")
class PointerAuxiliaryLossTest(unittest.TestCase):
    def _inputs(self, *, A=3, T=2, K=2, P=8, H=4, batch=2):
        policy = torch.zeros(batch, A)
        policy[:, 0] = 1.0
        head_logits = torch.zeros(batch, H)
        source_logits = torch.zeros(batch, P)
        slot_logits = torch.zeros(batch, K, P)
        action_head_indices = torch.zeros(batch, A, dtype=torch.long)
        action_head_indices[:, 0] = 1
        action_source_positions = torch.zeros(batch, A, dtype=torch.long)
        action_source_positions[:, 0] = 3
        action_target_slot_index = torch.full((batch, A, T), -1, dtype=torch.long)
        action_target_slot_index[:, 0, 0] = 0
        action_target_positions = torch.full((batch, A, T), -1, dtype=torch.long)
        action_target_positions[:, 0, 0] = 5
        action_target_mask = torch.zeros(batch, A, T, dtype=torch.bool)
        action_target_mask[:, 0, 0] = True
        action_mask = torch.zeros(batch, A, dtype=torch.bool)
        action_mask[:, 0] = True
        slot_exists = torch.zeros(batch, K, dtype=torch.bool)
        slot_exists[:, 0] = True
        return dict(
            head_logits=head_logits,
            source_logits=source_logits,
            slot_logits=slot_logits,
            policy=policy,
            action_head_indices=action_head_indices,
            action_source_positions=action_source_positions,
            action_target_slot_index=action_target_slot_index,
            action_target_positions=action_target_positions,
            action_target_mask=action_target_mask,
            action_mask=action_mask,
            slot_exists=slot_exists,
        )

    def test_returns_three_components(self):
        losses = pointer_auxiliary_losses(**self._inputs())

        self.assertEqual(set(losses), {"head", "source", "slot"})
        for name, (loss_sum, count) in losses.items():
            self.assertEqual(count, 2, name)
            self.assertTrue(torch.isfinite(loss_sum))

    def test_uniform_logits_give_log_vocab_size(self):
        """均匀 logits 下，soft CE 应等于 ln(有效候选数)。"""
        inputs = self._inputs()
        losses = pointer_auxiliary_losses(**inputs)

        head_sum, head_count = losses["head"]
        source_sum, source_count = losses["source"]
        # head 只有 1 个目标位置，均匀分布 → ln(H)
        self.assertAlmostEqual(float(head_sum) / head_count, float(torch.log(torch.tensor(4.0))), places=4)
        # source 同样 ln(P)，但只对目标位置归一化后计算
        self.assertAlmostEqual(float(source_sum) / source_count, float(torch.log(torch.tensor(8.0))), places=4)

    def test_head_target_aggregates_probability_mass(self):
        """两个动作同 head、各自 0.5 概率 → head 目标在该 head 上是 1.0。"""
        inputs = self._inputs(A=3)
        inputs["policy"] = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
        inputs["action_head_indices"] = torch.tensor([[1, 1, 0], [1, 1, 0]])
        inputs["action_mask"] = torch.tensor([[True, True, True], [True, True, True]])

        losses = pointer_auxiliary_losses(**inputs)

        head_sum, head_count = losses["head"]
        # 目标 head 概率和为 1，若 logits 全 0 则 head loss = ln(H)
        self.assertAlmostEqual(float(head_sum) / head_count, float(torch.log(torch.tensor(4.0))), places=4)

    def test_arity_two_slot_target_is_multi_label(self):
        """一个槽选 2 个：目标应给两个位置各一半质量，而不是只给一个。"""
        inputs = self._inputs(A=2, T=2, K=1, P=8)
        inputs["policy"] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        inputs["action_target_slot_index"] = torch.tensor([[[0, 0], [-1, -1]], [[0, 0], [-1, -1]]])
        inputs["action_target_positions"] = torch.tensor([[[2, 6], [-1, -1]], [[2, 6], [-1, -1]]])
        inputs["action_target_mask"] = torch.tensor([[[True, True], [False, False]], [[True, True], [False, False]]])
        inputs["slot_exists"] = torch.ones(2, 1, dtype=torch.bool)
        inputs["action_mask"] = torch.tensor([[True, True], [True, True]])

        losses = pointer_auxiliary_losses(**inputs)

        slot_sum, slot_count = losses["slot"]
        # 两个位置各 0.5，均匀 logits 下 = ln(8)
        self.assertAlmostEqual(float(slot_sum) / slot_count, float(torch.log(torch.tensor(8.0))), places=4)

    def test_no_slot_skips_slot_component(self):
        inputs = self._inputs()
        inputs["slot_exists"] = torch.zeros(2, 2, dtype=torch.bool)

        losses = pointer_auxiliary_losses(**inputs)

        self.assertNotIn("slot", losses)

    def test_all_masked_returns_empty(self):
        inputs = self._inputs()
        inputs["action_mask"] = torch.zeros(2, 3, dtype=torch.bool)

        losses = pointer_auxiliary_losses(**inputs)

        self.assertEqual(losses, {})


if __name__ == "__main__":
    unittest.main()
