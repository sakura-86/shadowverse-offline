# -*- coding: utf-8 -*-
"""线性局面估值：只用 16 个客观局面量做逻辑回归。

## 为什么需要它

2026-10-30 的分布诊断发现：在 plan_turn 对局（AI 分布）上，一个**只用客观局面量的
逻辑回归**在按对局划分的测试集上 AUC 达到 **0.834**，而当时最强的神经网络 value 头
（`v10b_gen1`）只有 **0.610**。也就是说，我们从未用一个"达到简单基线"的 value 头
测过 PUCT——"AlphaZero 不适用"的负结果是在 value 远低于线性基线的前提下得到的。

本模块提供一个**不依赖神经网络**的 value 实现，用来把这个变量隔离出来：
如果线性 value 让 PUCT 明显变强，说明瓶颈在 value 质量；如果仍然完败，范式结论才真的成立。

## 口径

- 特征只取 `StateFeatures` 里协议级、双方可见的量：生命、场攻总和、手牌数、PP、场面数、
  守护数、回合数，以及它们的差值。**不含卡面 id、不含对手手牌内容、不含任何隐藏信息。**
- 训练标签与评测数据必须同分布（`docs/plan/VALUE_HEAD_SIGNAL_AUDIT.md` §7）。
- 输出经 sigmoid 映射回 `[-1, 1]`，与 `SearchValue` 协议一致。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .search_value import ValueResult, _winner

#: 客观局面量（顺序即模型输入顺序，改动必须同步重训）。
OBJECTIVE_FEATURES: tuple[str, ...] = (
    "my_life",
    "opp_life",
    "life_diff",
    "my_field_atk",
    "opp_field_atk",
    "field_atk_diff",
    "my_hand",
    "opp_hand",
    "hand_diff",
    "my_pp",
    "opp_pp",
    "my_field_count",
    "opp_field_count",
    "opp_guard",
    "my_guard",
    "turn",
)


def objective_feature_vector(state_features: Mapping[str, Any]) -> list[float]:
    """从 ``StateFeatures`` 字典提取 :data:`OBJECTIVE_FEATURES`。

    只读协议级字段；缺字段按 0 处理（与训练端的取值方式保持一致）。
    """
    player = state_features.get("player") or {}
    opponent = state_features.get("opponent") or {}
    pressure = state_features.get("pressure") or {}

    my_life = float(player.get("life") or 0)
    opp_life = float(opponent.get("life") or 0)
    my_field_atk = float(pressure.get("player_field_attack_total") or 0)
    opp_field_atk = float(pressure.get("opponent_field_attack_total") or 0)
    my_hand = float(player.get("hand_count") or 0)
    opp_hand = float(opponent.get("hand_count") or 0)

    return [
        my_life,
        opp_life,
        my_life - opp_life,
        my_field_atk,
        opp_field_atk,
        my_field_atk - opp_field_atk,
        my_hand,
        opp_hand,
        my_hand - opp_hand,
        float(player.get("pp") or 0),
        float(opponent.get("pp") or 0),
        float(player.get("field_count") or 0),
        float(opponent.get("field_count") or 0),
        float(pressure.get("opponent_guard_count") or 0),
        float(pressure.get("player_guard_count") or 0),
        float(player.get("turn") or 0),
    ]


@dataclass(frozen=True)
class LinearCoefficients:
    """标准化后的逻辑回归系数。"""

    weights: tuple[float, ...]
    bias: float
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    features: tuple[str, ...] = OBJECTIVE_FEATURES
    train_auc: float | None = None
    test_auc: float | None = None
    n_train: int | None = None

    def validate(self) -> None:
        width = len(self.features)
        if len(self.weights) != width or len(self.mean) != width or len(self.scale) != width:
            raise ValueError("线性 value 系数维度与特征名数量不一致")
        if any(value == 0 for value in self.scale):
            raise ValueError("线性 value 的标准化 scale 不能含 0")

    @classmethod
    def load(cls, path: Path) -> "LinearCoefficients":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        coefficients = cls(
            weights=tuple(float(v) for v in payload["weights"]),
            bias=float(payload["bias"]),
            mean=tuple(float(v) for v in payload["mean"]),
            scale=tuple(float(v) for v in payload["scale"]),
            features=tuple(payload.get("features") or OBJECTIVE_FEATURES),
            train_auc=payload.get("train_auc"),
            test_auc=payload.get("test_auc"),
            n_train=payload.get("n_train"),
        )
        coefficients.validate()
        return coefficients

    def save(self, path: Path) -> None:
        self.validate()
        payload = {
            "schema": "LinearStateValue",
            "version": 1,
            "features": list(self.features),
            "weights": list(self.weights),
            "bias": self.bias,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "train_auc": self.train_auc,
            "test_auc": self.test_auc,
            "n_train": self.n_train,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class LinearValue:
    """把 :class:`LinearCoefficients` 包装成 ``SearchValue``。

    终局直接返回 ±1（与其它 value 实现一致），非终局走线性 logit → sigmoid → ``2p-1``。
    """

    def __init__(self, coefficients: LinearCoefficients, *, source: str = "linear_state_value"):
        coefficients.validate()
        self._coefficients = coefficients
        self.source = source

    @classmethod
    def from_path(cls, path: Path, *, source: str | None = None) -> "LinearValue":
        return cls(LinearCoefficients.load(path), source=source or "linear_state_value")

    def _logit(self, vector: Sequence[float]) -> float:
        coefficients = self._coefficients
        total = coefficients.bias
        for value, weight, mean, scale in zip(
            vector, coefficients.weights, coefficients.mean, coefficients.scale
        ):
            total += weight * ((value - mean) / scale)
        return total

    def evaluate(self, snapshot: Any, *, player: str) -> ValueResult:
        if player not in ("me", "opponent"):
            raise ValueError(f"unsupported player: {player}")

        winner = _winner(snapshot)
        if winner is not None:
            if winner == "simultaneous":
                return ValueResult(
                    value=0.0, source=f"{self.source}:terminal_simultaneous", terminal=True
                )
            return ValueResult(
                value=1.0 if winner == player else -1.0,
                source=f"{self.source}:terminal_result",
                terminal=True,
                diagnostics={"player": player, "winner": winner},
            )

        from decision_engine.training.state_features import state_features

        features = state_features(snapshot, perspective=player)
        vector = objective_feature_vector(features)
        logit = self._logit(vector)
        probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
        value = 2.0 * probability - 1.0
        value = max(-1.0, min(1.0, value))
        result = ValueResult(
            value=value,
            source=self.source,
            terminal=False,
            diagnostics={"player": player, "logit": logit, "probability": probability},
        )
        result.validate()
        return result

    def clear_cache(self) -> None:  # pragma: no cover - 无缓存
        return None


__all__ = [
    "OBJECTIVE_FEATURES",
    "LinearCoefficients",
    "LinearValue",
    "objective_feature_vector",
]
