# -*- coding: utf-8 -*-
"""生成式 pointer 的损失。

四条监督，缺一不可：

1. **联合**（在训练循环里算）：把 pointer 折算成 ``[B,A]`` logit 后做 soft CE。
   与推理严格一致——保证折算出的动作分布就是真实策略。
2. **head**：把动作概率按动作头聚合，监督 ``head_logits``。
3. **source**：按 source 的位置坐标聚合，监督 ``source_logits``。
4. **slot**：按「第 k 个槽的 target 位置」聚合，监督 ``slot_logits``。

为什么 2–4 不能省：联合 CE 只给最终分布梯度，三个 pointer 头必须经过折算式
（``head + source + Σ slot``）才能「看见」监督，早期梯度很弱。分解监督让每个
分量直接有目标，收敛更快。但它**不替代**联合 CE——联合 CE 保证三个分量合起来
仍然是正确的动作分布。

**arity > 1 的处理**：一个槽选 N 个实体（如「选 2 个对方随从使其消失」）。目标的
做法是把这 N 个位置**都**分到概率质量，而不是压成 one-hot——否则「同一次选择里
选了两个」这件事在监督里就丢了。所以这里全部用 soft CE（目标是分布，不要求 one-hot）。
"""

from __future__ import annotations

from typing import Mapping


def _soft_cross_entropy(logits: object, target: object) -> object:
    """``-(target * log_softmax(logits)).sum()``，逐样本求和后返回 ``[B]``。"""
    import torch
    import torch.nn.functional as F

    log_probs = F.log_softmax(logits, dim=-1)
    # -inf 位置可能产生 NaN（-inf * 0），统一用 where 置 0
    contrib = torch.where(target > 0, target * log_probs, torch.zeros_like(log_probs))
    return -contrib.sum(dim=-1)


def _normalize(target: object) -> object:
    """把每个样本各自归一化；全零样本返回全零（不产生 NaN）。"""
    total = target.sum(dim=-1, keepdim=True)
    return torch_where_normalize(target, total)


def torch_where_normalize(target: object, total: object) -> object:
    import torch

    return torch.where(total > 0, target / total.clamp(min=1e-8), torch.zeros_like(target))


def pointer_auxiliary_losses(
    *,
    head_logits: object,
    source_logits: object,
    slot_logits: object,
    policy: object,
    action_head_indices: object,
    action_source_positions: object,
    action_target_slot_index: object,
    action_target_positions: object,
    action_target_mask: object,
    action_mask: object,
    slot_exists: object,
) -> Mapping[str, tuple[object, int]]:
    """计算 head/source/slot 三条分解损失。

    返回 ``{name: (loss_sum, sample_count)}``——调用方用 ``sum/count`` 得到均值，
    这样「这一批里有多少样本对这个分量有效」不会被静默摊平。

    输入约定（torch 张量）：

    - ``head_logits [B,H]``、``source_logits [B,P]``、``slot_logits [B,K,P]``
    - ``policy [B,A]``：动作分布目标（PPO/BC 的 policy_target）
    - ``action_head_indices [B,A]``（-1 = 无法映射）
    - ``action_source_positions [B,A]``（-1 = 无坐标）
    - ``action_target_slot_index [B,A,T]`` / ``action_target_positions [B,A,T]``
    - ``action_target_mask [B,A,T]``、``action_mask [B,A]``
    - ``slot_exists [B,K]``
    """
    import torch

    B, A = policy.shape
    H = head_logits.shape[-1]
    P = source_logits.shape[-1]
    K = slot_logits.shape[-2]
    T = action_target_positions.shape[-1]
    device = policy.device
    dtype = policy.dtype

    valid = action_mask & (action_head_indices >= 0)
    weights = torch.where(valid, policy, torch.zeros_like(policy))  # [B,A]

    losses: dict[str, tuple[object, int]] = {}

    # ── head ──
    head_target = torch.zeros(B, H, device=device, dtype=dtype)
    head_index = action_head_indices.clamp(min=0).long()
    head_target.scatter_add_(1, head_index, weights)
    head_target = _normalize(head_target)
    head_per_sample = _soft_cross_entropy(head_logits, head_target)
    head_valid = head_target.sum(dim=-1) > 0
    if head_valid.any():
        losses["head"] = (head_per_sample[head_valid].sum(), int(head_valid.sum().item()))

    # ── source ──
    # 条件化时 ``source_logits`` 是 ``[B,H,P]``，目标也按动作头分桶：
    # 「在 head h 下 source 该是哪个位置」。这与折算式的取片方式一致。
    source_ok = valid & (action_source_positions >= 0)
    source_weights = torch.where(source_ok, policy, torch.zeros_like(policy))
    source_position = action_source_positions.clamp(min=0).long()
    if source_logits.dim() == 3:
        head_flat = action_head_indices.clamp(min=0).long() * P + source_position    # [B,A]
        source_target = torch.zeros(B, H * P, device=device, dtype=dtype)
        source_target.scatter_add_(1, head_flat, source_weights)
        source_target = _normalize(source_target)
        source_per_sample = _soft_cross_entropy(
            source_logits.reshape(B, H * P), source_target
        )
    else:
        source_target = torch.zeros(B, P, device=device, dtype=dtype)
        source_target.scatter_add_(1, source_position, source_weights)
        source_target = _normalize(source_target)
        source_per_sample = _soft_cross_entropy(source_logits, source_target)
    source_valid = source_target.sum(dim=-1) > 0
    if source_valid.any():
        losses["source"] = (source_per_sample[source_valid].sum(), int(source_valid.sum().item()))

    # ── slot ──
    # 每个槽单独聚合：把所有动作中「该槽指向的位置 p」按动作概率加权。
    # head 条件化时 ``slot_logits`` 是 ``[B,H,K,P]``，目标按 (head, 位置) 分桶。
    slot_conditioned = slot_logits.dim() == 4
    slot_total = torch.zeros((), device=device, dtype=dtype)
    slot_count = 0
    for k in range(K):
        exists = slot_exists[:, k]  # [B]
        if not bool(exists.any()):
            continue
        # 只取「属于第 k 槽」且被 mask 的 target
        in_slot = (action_target_slot_index == k) & action_target_mask  # [B,A,T]
        position = action_target_positions.clamp(min=0).long()          # [B,A,T]
        weight_per_target = (weights.unsqueeze(-1) * in_slot.to(dtype))  # [B,A,T]
        if slot_conditioned:
            head_flat = (
                action_head_indices.clamp(min=0).long().unsqueeze(-1) * P + position
            )                                                            # [B,A,T]
            slot_k_target = torch.zeros(B, H * P, device=device, dtype=dtype)
            slot_k_target.scatter_add_(
                1, head_flat.reshape(B, A * T), weight_per_target.reshape(B, A * T)
            )
            slot_k_target = _normalize(slot_k_target)
            per_sample = _soft_cross_entropy(
                slot_logits[:, :, k, :].reshape(B, H * P), slot_k_target
            )
        else:
            slot_k_target = torch.zeros(B, P, device=device, dtype=dtype)
            flat_index = position.reshape(B, A * T)
            flat_weight = weight_per_target.reshape(B, A * T)
            slot_k_target.scatter_add_(1, flat_index, flat_weight)
            # arity>1：一个动作可能给同一槽贡献多个位置，本身就多标签；归一化后仍可和为 1
            slot_k_target = _normalize(slot_k_target)
            per_sample = _soft_cross_entropy(slot_logits[:, k, :], slot_k_target)
        valid_k = exists & (slot_k_target.sum(dim=-1) > 0)
        if bool(valid_k.any()):
            slot_total = slot_total + per_sample[valid_k].sum()
            slot_count += int(valid_k.sum().item())
    if slot_count > 0:
        losses["slot"] = (slot_total, slot_count)

    return losses
