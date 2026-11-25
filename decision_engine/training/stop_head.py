# -*- coding: utf-8 -*-
"""Stop head：独立预测「现在该不该结束回合」。

为什么单独做一条分支（见 ``docs/plan/SEARCH_MYOPIA_DIAGNOSIS.md``）：

- 当前 ``turn_end`` 的 logit 由 ``policy_head(state_emb, action_emb)`` 给出，而 turn_end
  没有动作特征（``source_uid=0``），在 21/26 个合法动作时 prior 被稀释到 0.002/0.001；
- 人类 replay 缺 1683 条 turn_end 标签，policy head 根本学不到它；
- 但 ``turn_end`` 其实是一个**二分类**问题（停 / 继续），不需要和具体动作争 policy 容量。

本模块只输出一个标量 ``stop_prob``，由 ``StructuredNeuralPrior`` 用它替换 turn_end 的
prior，再把剩余概率按原比例重分配给其他动作。

权重独立存放（``<model_dir>/stop_head.pt``），不改任何现有 ``model.pt``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

STOP_HEAD_SCHEMA = "StopHead"
STOP_HEAD_VERSION = 1


@dataclass(frozen=True)
class StopHeadConfig:
    state_dim: int = 256
    hidden: int = 64
    dropout: float = 0.0


def build_stop_head(config: StopHeadConfig):
    """构造 stop head 网络（延迟 import torch，保持模块可被非 torch 路径引用）。"""
    import torch
    from torch import nn

    class StopHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = [
                nn.Linear(config.state_dim, config.hidden),
                nn.ReLU(),
            ]
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            layers.append(nn.Linear(config.hidden, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, state_emb: "torch.Tensor") -> "torch.Tensor":
            """state_emb: [B, state_dim] -> stop logits [B]."""
            return self.net(state_emb).squeeze(-1)

    return StopHead()


def stop_head_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "stop_head.pt"


def save_stop_head(model_dir: str | Path, head, config: StopHeadConfig) -> Path:
    import torch

    path = stop_head_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": STOP_HEAD_SCHEMA,
            "version": STOP_HEAD_VERSION,
            "config": {
                "state_dim": config.state_dim,
                "hidden": config.hidden,
                "dropout": config.dropout,
            },
            "state_dict": head.state_dict(),
        },
        path,
    )
    return path


def load_stop_head(model_dir: str | Path, *, device: str = "cpu"):
    """加载 stop head；不存在时返回 ``None``（调用方据此保持旧行为）。"""
    import torch

    path = stop_head_path(model_dir)
    if not path.exists():
        return None
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != STOP_HEAD_SCHEMA:
        raise ValueError(f"stop head schema 不匹配: {path}")
    config = StopHeadConfig(**{k: v for k, v in (payload.get("config") or {}).items() if k in StopHeadConfig.__dataclass_fields__})
    head = build_stop_head(config)
    head.load_state_dict(payload["state_dict"])
    head.to(device)
    head.eval()
    return head


def apply_stop_prob(priors: Sequence[float], turn_end_index: int, stop_prob: float) -> tuple[float, ...]:
    """把 turn_end 的 prior 替换为 ``stop_prob``，其余动作按原比例缩放到 ``1 - stop_prob``。"""
    if turn_end_index < 0 or turn_end_index >= len(priors):
        raise ValueError(f"turn_end_index 越界: {turn_end_index} / {len(priors)}")
    if not 0.0 <= stop_prob <= 1.0:
        raise ValueError(f"stop_prob 必须在 [0,1]: {stop_prob}")
    corrected = [float(value) for value in priors]
    corrected[turn_end_index] = float(stop_prob)
    others = [index for index in range(len(corrected)) if index != turn_end_index]
    total_other = sum(corrected[index] for index in others)
    remaining = 1.0 - float(stop_prob)
    if total_other > 1e-9:
        scale = remaining / total_other
        for index in others:
            corrected[index] *= scale
    elif others:
        share = remaining / len(others)
        for index in others:
            corrected[index] = share
    return tuple(corrected)


def apply_stop_prob_to_prior_result(prior_result, actions, stop_prob: float):
    """在 ``PriorResult`` 上应用 stop_prob，保持 source/diagnostics。"""
    from decision_engine.planning.search_prior import PriorResult

    turn_end_index = -1
    for index, action in enumerate(actions):
        kind = action.kind.value if hasattr(action.kind, "value") else str(action.kind)
        if kind == "turn_end":
            turn_end_index = index
            break
    if turn_end_index < 0:
        return prior_result
    priors = apply_stop_prob(prior_result.priors, turn_end_index, stop_prob)
    diagnostics = dict(prior_result.diagnostics or {})
    diagnostics["stop_prob"] = float(stop_prob)
    return PriorResult(priors=priors, source=f"{prior_result.source}+stop_head", diagnostics=diagnostics)
