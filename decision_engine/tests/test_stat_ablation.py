# -*- coding: utf-8 -*-
"""输入消融（ablate_stat_dims）：权重列恒 0 + 梯度屏蔽。

这一层是严格消融实验的地基：只有"网络真的看不到这些通道"成立，
同一批数据训出来的两个模型才能把差异归因给特征本身。
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - 无 torch 环境跳过
    torch = None

from decision_engine.training.structured_tensorization import (
    CARD_STAT_BOOST_PROGRESS_INDEX,
    CARD_STAT_EVOLVE_EFFECT_INDEX,
    CARD_STAT_DIM,
)

if torch is not None:
    from decision_engine.training.structured_policy_value import (
        StructuredModelConfig,
        StructuredPolicyValueNet,
    )


def _build(config):
    return StructuredPolicyValueNet(
        vocab_size=16,
        state_scalar_width=8,
        action_scalar_width=4,
        config=config,
    )


@unittest.skipIf(torch is None, "requires torch")
class StatAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.ablated = tuple(range(0, 3))  # 用前三维，便于构造输入
        self.config = StructuredModelConfig(ablate_stat_dims=self.ablated)
        self.model = _build(self.config)

    def test_weight_columns_are_zeroed(self) -> None:
        weight = self.model.card_encoder.stat_proj.weight.detach()
        for dim in self.ablated:
            self.assertTrue(torch.all(weight[:, dim] == 0))
        untouched = weight[:, 5].clone()
        self.assertTrue(torch.any(untouched != 0), "非消融列不应被清零")

    def test_forward_ignores_ablated_channels(self) -> None:
        card_ids = torch.zeros(2, 1, dtype=torch.long)
        stats = torch.rand(2, 1, CARD_STAT_DIM)
        mutated = stats.clone()
        for dim in self.ablated:
            mutated[:, :, dim] = 1e3
        base = self.model.card_encoder(card_ids, stats)
        changed = self.model.card_encoder(card_ids, mutated)
        self.assertTrue(torch.allclose(base, changed), "消融通道变化不应影响输出")

    def test_gradient_is_masked_on_ablated_columns(self) -> None:
        stats = torch.rand(1, 1, CARD_STAT_DIM, requires_grad=False)
        out = self.model.card_encoder(torch.zeros(1, 1, dtype=torch.long), stats)
        out.sum().backward()
        grad = self.model.card_encoder.stat_proj.weight.grad
        self.assertIsNotNone(grad)
        for dim in self.ablated:
            self.assertTrue(torch.all(grad[:, dim] == 0), "消融列梯度必须为 0")
        self.assertTrue(torch.any(grad[:, 5] != 0), "非消融列必须有梯度")

    def test_parameters_stay_zero_after_optimizer_step(self) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.5)
        for _ in range(3):
            optimizer.zero_grad()
            stats = torch.rand(2, 1, CARD_STAT_DIM)
            out = self.model.card_encoder(torch.zeros(2, 1, dtype=torch.long), stats)
            out.sum().backward()
            optimizer.step()
        weight = self.model.card_encoder.stat_proj.weight.detach()
        for dim in self.ablated:
            self.assertTrue(torch.all(weight[:, dim] == 0), "优化器不能把消融列拉起来")

    def test_without_ablation_channels_do_matter(self) -> None:
        model = _build(StructuredModelConfig())
        stats = torch.rand(1, 1, CARD_STAT_DIM)
        mutated = stats.clone()
        mutated[:, :, 0] = stats[:, :, 0] + 5.0
        base = model.card_encoder(torch.zeros(1, 1, dtype=torch.long), stats)
        changed = model.card_encoder(torch.zeros(1, 1, dtype=torch.long), mutated)
        self.assertFalse(torch.allclose(base, changed), "未消融时输入必须影响输出")

    def test_out_of_range_raises(self) -> None:
        model = _build(StructuredModelConfig())
        with self.assertRaises(ValueError):
            model.card_encoder.apply_stat_ablation((CARD_STAT_DIM,))

    def test_apply_is_idempotent(self) -> None:
        before = len(self.model.card_encoder.stat_proj.weight._backward_hooks or {})
        self.model.card_encoder.apply_stat_ablation(self.ablated)
        after = len(self.model.card_encoder.stat_proj.weight._backward_hooks or {})
        self.assertEqual(before, after, "重复施加不应重复注册 hook")

    def test_boost_progress_index_matches_tail(self) -> None:
        # schema v9 起，最后两维是进化效果标志，增幅进度三维退到倒数第五到倒数第三。
        self.assertEqual(CARD_STAT_BOOST_PROGRESS_INDEX,
                         (CARD_STAT_DIM - 5, CARD_STAT_DIM - 4, CARD_STAT_DIM - 3))
        self.assertEqual(CARD_STAT_EVOLVE_EFFECT_INDEX, (CARD_STAT_DIM - 2, CARD_STAT_DIM - 1))


if __name__ == "__main__":
    unittest.main()
