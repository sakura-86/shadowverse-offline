# -*- coding: utf-8 -*-
"""把「净血差」这个评估量搬到 PUCT 的叶节点上。

## 动机

`plan_turn` 之所以在 A/B 里全胜，不是因为它算得深，而是因为它有一个评估函数：

    净血差 = 我方打掉的主战者生命 - (对方剩余场攻 + 进化加成)

PUCT 没有这个量。它的叶节点值来自模型的 value head，而这个 head 已经测出是坏的
（Q≈0.5、排序反转）。于是 128 sims 内 PUCT 只能沿 prior 瞎逛。

本模块提供 :class:`TurnPlanValue`：用与 `plan_turn` **同一套净血差口径**给叶节点估值，
不搜索、不枚举、不碰合法性，成本 O(场面大小)。

## 与 `plan_turn` 评分的关系

`plan_turn` 的评分是「**这一回合**走完之后的净血差」——它在意「本回合能赚多少」。
叶节点估值要的是「**当前局面**谁占优」，所以用对称的形式：

    局面净值 = (我方生命 - 对方生命) + (我方威胁 - 对方威胁)

其中威胁 = 场上随从攻击力之和 + 该侧的进化加成（超进化点 +3 / 进化点 +2）。

这个式子和 `净血差` 同源：

    净血差 = 我方收益 - 对方收益
           = (对方生命下降) - (对方剩余威胁)
    局面净值 = 我方剩余生命 - 对方剩余威胁 - (对方剩余生命 - 我方剩余威胁)

两者只差一个「已经打掉多少」的常数项，对同一时刻的兄弟节点排序没有影响。

## 归一化

`SearchValue.evaluate` 要求输出在 [-1, 1]。用 ``tanh(score / scale)``：

- ``scale`` 默认 12。一局 20 点生命，8 点以上的局面净值差已经算明显优势，
  tanh(8/12) ≈ 0.58，不会一上来就饱和到 ±1。
- 打到 ±1 的只有真正的斩杀线附近，避免 value 在一开始就把所有边压成同一个值。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import tanh
from typing import Any, Mapping

from .search_value import ValueResult
from .turn_evaluation import opponent_evolve_bonus


@dataclass(frozen=True)
class TurnPlanValue:
    """叶节点估值：血量差 + 双方威胁差。

    ``player`` 是当前视角方；`SearchValue.evaluate` 在搜索里会按节点所属玩家传入，
    所以两侧都能用同一个实例。
    """

    scale: float = 12.0
    #: 场攻差的权重。1.0 = 与生命等权；场面是下回合的伤害来源，稍微加权更贴近实战。
    threat_weight: float = 1.0
    source: str = "turn_plan_net_life"

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("TurnPlanValue.scale must be positive")

    def _side(self, snapshot: Any, player: str):
        return (snapshot.me, snapshot.opponent) if player == "me" else (snapshot.opponent, snapshot.me)

    def _threat(self, snapshot: Any, view: Any) -> tuple[int, dict[str, Any]]:
        cards = snapshot.cards_by_uid
        atk = 0
        count = 0
        for uid in view.field:
            card = cards.get(uid)
            if card is None:
                continue
            atk += max(0, int(card.atk))
            count += 1
        bonus, kind = opponent_evolve_bonus(view)
        if count == 0:
            bonus, kind = 0, "no_follower"
        return atk + bonus, {"field_atk": atk, "evolve_bonus": bonus, "evolve_bonus_kind": kind, "count": count}

    def score(self, snapshot: Any, *, player: str) -> tuple[float, dict[str, Any]]:
        """局面净值（未归一化）。"""
        mine, theirs = self._side(snapshot, player)
        my_threat, my_detail = self._threat(snapshot, mine)
        opp_threat, opp_detail = self._threat(snapshot, theirs)
        life_delta = int(mine.life) - int(theirs.life)
        threat_delta = my_threat - opp_threat
        raw = life_delta + self.threat_weight * threat_delta
        return raw, {
            "life_delta": life_delta,
            "my_threat": my_threat,
            "opp_threat": opp_threat,
            "threat_delta": threat_delta,
            "my_threat_detail": my_detail,
            "opp_threat_detail": opp_detail,
        }

    def evaluate(self, snapshot: Any, *, player: str) -> ValueResult:
        mine, theirs = self._side(snapshot, player)
        # 终局直接 ±1，不做归一化——血量 <= 0 之后净值公式不再有意义。
        if int(theirs.life) <= 0:
            return ValueResult(value=1.0, source=f"{self.source}:terminal_win", terminal=True)
        if int(mine.life) <= 0:
            return ValueResult(value=-1.0, source=f"{self.source}:terminal_loss", terminal=True)

        raw, detail = self.score(snapshot, player=player)
        value = tanh(raw / self.scale)
        return ValueResult(
            value=value,
            source=self.source,
            terminal=False,
            diagnostics={"raw": raw, **detail, "scale": self.scale},
        )

    # 与 NeuralValueFunction 保持同一套缓存 / 清理接口，PuctSearch 会无条件调用
    def clear_cache(self) -> None:  # pragma: no cover - 无缓存
        return None


__all__ = ["TurnPlanValue"]
