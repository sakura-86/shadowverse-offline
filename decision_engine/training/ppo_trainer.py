# -*- coding: utf-8 -*-
"""PPO 更新：GAE(λ) + clipped surrogate + value MSE。

## 与采样阶段的一致性（容易错的点）

采样时只在"strict 可执行"的动作上分配概率（其余概率为 0），所以 rollout 里存的
``log_prob`` 是**条件于可执行集**的分布。训练时必须用同一组
``executable_indices`` 重新构造 mask，否则 ``π_new / π_old`` 的比值没有意义——
这正是 importance sampling 失效的经典来源。

## 稀疏奖励

影之诗的即时奖励恒为 0，只有每局最后一步拿 ±1。因此 ``δ_t = γV(s_{t+1}) − V(s_t)``，
advantage 的信息完全来自 value。value 学不准时 PPO 的梯度方向就是随机的——这也是
`docs/plan/VALUE_HEAD_SIGNAL_AUDIT.md` 要先测 value 判别力的原因。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .ppo_rollout import RolloutBatch, RolloutStep


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    update_epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    device: str = "cpu"
    #: 换牌步（``RolloutStep.is_mulligan``）的 advantage 是否改用 MC return。
    #:
    #: 换牌在 t=0、reward 在 t≈27，GAE(λ=0.95) 会把信号衰减到 ``0.95^27 ≈ 0.25``，
    #: 换牌几乎学不到东西。开启后换牌步直接用 ``episode_return − V(s)``（γ=1），
    #: 信号 100%，代价是方差更大、需要更大 batch 平均。
    mulligan_mc_return: bool = False
    #: 每批 PPO 更新**之前**先做多少次纯 value 更新（只算 value MSE，不碰 policy）。
    #: 0 = 关闭。用于诊断"critic 是欠拟合还是没信号"：如果预热后 value_loss 明显下降、
    #: 但 value 对胜负的 AUC 仍是 0.5，那问题就出在 value 输入没有判别信息，而不是优化不足。
    value_warmup_updates: int = 0


def compute_gae(
    episodes: Sequence[Sequence[RolloutStep]],
    *,
    gamma: float,
    lam: float,
    mulligan_mc_return: bool = False,
) -> dict[str, list[float]]:
    """按 episode 计算 advantage 与 return（展平顺序 = episodes 顺序）。

    ``reward`` 只在每局最后一步非零；其余步靠 ``γV(s') − V(s)`` 传播。

    ``mulligan_mc_return`` 打开时，**换牌步**的 advantage 改为 ``episode_return − V(s)``。
    换牌步是 episode 的第一步，在反向遍历里**最后**被处理，所以覆盖它不会影响
    其他步（其他步的 ``running`` 已经算完）；这里也因此不需要额外重置状态。
    """
    advantages: list[float] = []
    returns: list[float] = []
    for episode in episodes:
        episode_total = sum(step.reward for step in episode) if mulligan_mc_return else 0.0
        running = 0.0
        episode_adv: list[float] = [0.0] * len(episode)
        episode_ret: list[float] = [0.0] * len(episode)
        for index in range(len(episode) - 1, -1, -1):
            step = episode[index]
            if mulligan_mc_return and getattr(step, "is_mulligan", False):
                episode_adv[index] = episode_total - step.value
                episode_ret[index] = episode_total
                continue
            next_value = episode[index + 1].value if index + 1 < len(episode) else 0.0
            delta = step.reward + gamma * next_value - step.value
            running = delta + gamma * lam * running
            episode_adv[index] = running
            episode_ret[index] = running + step.value
        advantages.extend(episode_adv)
        returns.extend(episode_ret)
    return {"advantages": advantages, "returns": returns}


def _flatten(batch: RolloutBatch) -> list[RolloutStep]:
    return batch.all_steps()


def roc_auc(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Mann-Whitney U 形式的 ROC AUC（同分取平均秩），``labels > 0`` 为正类。

    用来回答"critic 到底有没有判别力"：value 若能区分胜负，AUC 应显著大于 0.5。
    ``value_loss`` 下降但 AUC 停在 0.5，说明它只学会了输出均值、advantage 仍是噪声。
    """
    positives = sum(1 for label in labels if label > 0)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        average_rank = (position + end) / 2.0 + 1.0
        for offset in range(position, end + 1):
            ranks[order[offset]] = average_rank
        position = end + 1
    rank_sum = sum(ranks[index] for index in range(len(scores)) if labels[index] > 0)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _predict_values(model: Any, rows: Sequence[dict], *, config: PPOConfig) -> list[float]:
    """对全部样本分块前向取 value（eval 模式，无梯度），用于训练后的诊断。"""
    import torch

    from decision_engine.training.structured_policy_value import _collate_structured

    values: list[float] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), config.batch_size):
            chunk = rows[start : start + config.batch_size]
            tensor = _collate_structured(chunk, device=config.device)
            _, value = model(tensor)
            values.extend(float(item) for item in value.tolist())
    if was_training:
        model.train()
    return values


def ppo_update(
    model: Any,
    optimizer: Any,
    batch: RolloutBatch,
    *,
    vocabulary: Any,
    config: PPOConfig,
) -> dict[str, float]:
    """用一个 on-policy batch 做若干 epoch 的 PPO 更新。"""
    import torch
    import torch.nn.functional as F

    from decision_engine.training.structured_policy_value import _collate_structured
    from decision_engine.training.structured_tensorization import (
        structured_tensorize_training_example,
    )

    steps = _flatten(batch)
    if not steps:
        return {"updates": 0.0}

    gae = compute_gae(
        batch.episodes,
        gamma=config.gamma,
        lam=config.lam,
        mulligan_mc_return=config.mulligan_mc_return,
    )
    advantages = gae["advantages"]
    returns = gae["returns"]

    rows: list[dict] = []
    kept: list[int] = []
    for position, step in enumerate(steps):
        try:
            rows.append(structured_tensorize_training_example(step.example, vocabulary=vocabulary).to_jsonable())
            kept.append(position)
        except Exception:
            continue
    if not rows:
        return {"updates": 0.0}

    steps = [steps[i] for i in kept]
    advantages = [advantages[i] for i in kept]
    returns = [returns[i] for i in kept]

    old_log_probs = torch.tensor([s.log_prob for s in steps], dtype=torch.float32)
    selected = torch.tensor([s.selected_index for s in steps], dtype=torch.long)
    advantage_t = torch.tensor(advantages, dtype=torch.float32)
    if config.normalize_advantage and len(advantage_t) > 1:
        advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)
    return_t = torch.tensor(returns, dtype=torch.float32)
    outcomes = [float(s.outcome) for s in steps]

    model.train()
    metrics = {
        "samples": float(len(rows)),
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "updates": 0.0,
        "value_warmup_updates": 0.0,
    }
    updates = 0

    # ── value 预热 ────────────────────────────────────────────────────
    # 只优化 value MSE，不碰 policy：用来区分"critic 欠拟合"与"value 输入无判别信息"。
    # 必须在 PPO 主循环前做，否则 policy 先动会让 advantage 与 value 目标一起漂移。
    if config.value_warmup_updates > 0:
        for _ in range(config.value_warmup_updates):
            order = torch.randperm(len(rows)).tolist()
            for start in range(0, len(order), config.batch_size):
                chunk = order[start : start + config.batch_size]
                batch_tensor = _collate_structured(
                    [rows[i] for i in chunk], device=config.device
                )
                _, warm_values = model(batch_tensor)
                warm_loss = F.mse_loss(warm_values, return_t[chunk])
                optimizer.zero_grad()
                warm_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                metrics["value_warmup_updates"] += 1.0
    for _ in range(config.update_epochs):
        order = torch.randperm(len(rows)).tolist()
        for start in range(0, len(order), config.batch_size):
            chunk = order[start : start + config.batch_size]
            chunk_rows = [rows[i] for i in chunk]
            batch_tensor = _collate_structured(chunk_rows, device=config.device)
            logits, values = model(batch_tensor)

            # 用采样阶段的可执行集重建 mask，保证 π_new 与 π_old 定义在同一支撑集上。
            allowed = torch.zeros_like(batch_tensor["action_mask"])
            for offset, position in enumerate(chunk):
                indices = steps[position].executable_indices
                if indices:
                    allowed[offset, list(indices)] = True
            combined = allowed & batch_tensor["action_mask"]
            # 极端情况（可执行集为空）退化为原始 mask，避免整行 -inf。
            empty_rows = combined.sum(dim=-1) == 0
            if bool(empty_rows.any()):
                combined[empty_rows] = batch_tensor["action_mask"][empty_rows]

            # 用 -1e9 而不是 -inf：整行被 mask 空时 log_softmax 会产出 NaN，
            # 而 NaN 会静默污染整个 loss（本实验踩过一次）。
            masked_logits = logits.masked_fill(~combined, -1.0e9)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            new_log_prob = log_probs.gather(1, selected[chunk].unsqueeze(1)).squeeze(1)
            old_log_prob = old_log_probs[chunk]
            advantage = advantage_t[chunk]
            if not bool(torch.isfinite(new_log_prob).all()):
                raise RuntimeError(
                    "PPO: new_log_prob 含非有限值——检查 executable_indices 与 collate 后 "
                    "action_mask 是否不重叠，或 selected_index 越界"
                )

            ratio = (new_log_prob - old_log_prob).exp()
            surrogate_1 = ratio * advantage
            surrogate_2 = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantage
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            value_loss = F.mse_loss(values, return_t[chunk])
            probabilities = log_probs.exp()
            entropy = -(probabilities * log_probs).nan_to_num(0.0).sum(dim=-1).mean()

            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = float((old_log_prob - new_log_prob).mean().item())
                clip_fraction = float(((ratio - 1.0).abs() > config.clip_ratio).float().mean().item())
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
            metrics["approx_kl"] += approx_kl
            metrics["clip_fraction"] += clip_fraction
            updates += 1

    model.eval()
    if updates:
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
            metrics[key] /= updates
    metrics["updates"] = float(updates)

    # ── critic 诊断 ───────────────────────────────────────────────────
    # 必须放在 ``value_loss`` 归一化之后：上面累加的是每个 minibatch 的和，直接拿去
    # 除 baseline 会让 R² 少除以 updates（本实验第一版因此报出 -9.67 的假象）。
    # value_loss 的绝对值不可跨批比较（return 的方差每批不同），所以同时给出
    # "预测常数 0 的 MSE"作为基线，R² 才是可读的收敛指标；AUC 判断判别力。
    count = len(returns)
    if count:
        return_mean = sum(returns) / count
        return_var = sum((item - return_mean) ** 2 for item in returns) / count
        baseline_mse = sum(item * item for item in returns) / count
        predicted = _predict_values(model, rows, config=config)
        value_mean = sum(predicted) / len(predicted)
        value_var = sum((item - value_mean) ** 2 for item in predicted) / len(predicted)
        metrics["return_mean"] = return_mean
        metrics["return_std"] = math.sqrt(return_var)
        metrics["return_abs_mean"] = sum(abs(item) for item in returns) / count
        metrics["value_pred_mean"] = value_mean
        metrics["value_pred_std"] = math.sqrt(value_var)
        metrics["value_auc"] = roc_auc(predicted, outcomes)
        metrics["value_baseline_mse"] = baseline_mse
        metrics["value_r2"] = 1.0 - (metrics["value_loss"] / baseline_mse) if baseline_mse > 0 else 0.0

    return metrics


__all__ = ["PPOConfig", "compute_gae", "ppo_update", "roc_auc"]
