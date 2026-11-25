# -*- coding: utf-8 -*-
"""与卡组身份无关的通用换牌策略。

## 为什么需要它

`decision_engine/strategy/mulligan.py` 里的策略（`SpellboostRuneMulliganStrategy`、
`GuardHavenMulliganStrategy`）都是**卡组特定**的：它们按确切的 `base_card_id` 判断
"这张要不要留"。而训练/评估用的是卡组池（第 10 弹 35 个卡组、6 个职业），没有逐卡组的
换牌策略，逐个去写也不现实。

同时 A/B 的公平性要求**所有臂共用同一个换牌策略**（`plan_turn` / `policy` / `puct`
都没有换牌概念，只让某一臂换牌会污染比较）。所以需要一个与卡组无关的实现。

## 规则（保守，目标是降低开局方差而不是最优）

1. `cost <= always_keep_max_cost` 一律保留；
2. 其余保留 `cost <= keep_max_cost`；
3. 最多换 `max_changes` 张，从**最贵**的开始换。

保守的理由：全换会让手牌完全随机化，在"策略还不会打牌"的阶段反而放大方差。
这里只做"把明显打不出去的贵牌换掉"这一件事。

## 与已有路径的关系

- 本模块实现 `local_self_play.MulliganSelector` 协议
  （`select_mulligan(snapshot, *, player) -> tuple[int, ...]`，返回**要换掉的 uid**）。
- 不用 `StrategyMulliganSelector`，因为它把 snapshot 转成 `MulliganCard`，
  而 `MulliganCard` 不含 `cost` 字段；通用策略需要按费用判断。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CurveMulliganConfig:
    """通用曲线换牌参数。

    Attributes:
        always_keep_max_cost: cost 不超过它的手牌一定保留。
        keep_max_cost: 其余手牌中 cost 不超过它的保留。
        max_changes: 单次 mulligan 最多换掉的张数。
    """

    always_keep_max_cost: int = 2
    keep_max_cost: int = 4
    max_changes: int = 3


class CurveMulliganSelector:
    """按费用曲线换牌的通用策略（留低费、换高费）。

    满足 `local_self_play.MulliganSelector` 协议，可直接传给
    `StrictSearchAbRunner(mulligan_selector=...)`。

    这个类**只读 snapshot，不修改它**；换牌执行由调用方通过
    `perform_mulligan` 完成，从而保证 RNG 由调用方用 `seed` 派生（可复现）。
    """

    def __init__(self, config: CurveMulliganConfig | None = None) -> None:
        self.config = config or CurveMulliganConfig()
        if self.config.max_changes < 0:
            raise ValueError("max_changes 不能为负")
        if self.config.always_keep_max_cost > self.config.keep_max_cost:
            raise ValueError("always_keep_max_cost 不应大于 keep_max_cost")

    def select_mulligan(self, snapshot: Any, *, player: str) -> tuple[int, ...]:
        """返回要换掉的手牌 uid（按费用从高到低，最多 `max_changes` 张）。"""
        if player not in {"me", "opponent"}:
            raise ValueError(f"player 必须是 me/opponent，得到 {player!r}")
        view = snapshot.me if player == "me" else snapshot.opponent
        cards_by_uid = getattr(snapshot, "cards_by_uid", {}) or {}

        candidates: list[tuple[int, int]] = []
        for uid in view.hand:
            card = cards_by_uid.get(uid)
            if card is None:
                # 手牌 uid 在 cards_by_uid 里缺失属于数据不一致，不能猜费用。
                # 直接跳过（保守：不换它），而不是当成 0 费处理。
                continue
            cost = int(getattr(card, "cost", 0))
            if cost > self.config.keep_max_cost:
                candidates.append((cost, int(uid)))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return tuple(uid for _, uid in candidates[: self.config.max_changes])


def build_curve_mulligan_selector(
    *, always_keep_max_cost: int = 2, keep_max_cost: int = 4, max_changes: int = 3
) -> CurveMulliganSelector:
    """便捷构造器（供 CLI 用）。"""
    return CurveMulliganSelector(
        CurveMulliganConfig(
            always_keep_max_cost=always_keep_max_cost,
            keep_max_cost=keep_max_cost,
            max_changes=max_changes,
        )
    )


MULLIGAN_MODES = ("none", "curve")


def build_mulligan_selector(mode: str) -> CurveMulliganSelector | None:
    """按 CLI 模式构造 selector。``none`` 返回 ``None``（= 双方都不换牌）。

    三个 A/B / PPO 工具共用这一处映射，避免各自写出不一致的默认值。
    """
    if mode == "none":
        return None
    if mode == "curve":
        return build_curve_mulligan_selector()
    raise ValueError(f"未知 mulligan 模式: {mode!r}（可选 {MULLIGAN_MODES}）")


__all__ = [
    "MULLIGAN_MODES",
    "CurveMulliganConfig",
    "CurveMulliganSelector",
    "build_curve_mulligan_selector",
    "build_mulligan_selector",
]
