# -*- coding: utf-8 -*-
"""换牌阶段的动作枚举。

## 为什么把换牌当动作，而不是局外的一次性调用

换牌是真实对局的第一个决策：玩家看着起手 4~5 张牌，决定换掉哪几张。把它排除在
动作空间之外，模型就永远学不会换牌，实战只能外挂一套按卡组手写的 heuristic
（``decision_engine/strategy/mulligan.py`` 里那些 ``*MulliganStrategy``），
新卡组一多就不可维护。

本模块把换牌表达成与局内动作**同构**的 ``LegalAction``：

- ``MULLIGAN_CHANGE(source_uid=card_uid)``：把手牌里的这张换掉
- ``MULLIGAN_END``：结束换牌，保留剩下的

这样 policy head、``action_features``、``PPORolloutCollector`` 全部可以原样复用，
**不需要额外的换牌网络**。

## 为什么逐张而不是枚举 2^k 个 mask

- 逐张直接复用现成的「N 选 1」框架，``action_scalar`` 维度不变
  （``action_kind`` 用 ``_stable_string_value`` 编码成标量，加新值不改宽度，
  现有 checkpoint 仍能加载）；
- 起手 4~5 张，平均只需换 1.47 张（见 ``docs/plan/PPO_IGNITION_CHECK.md`` §28.1），
  即约 1.47 次 CHANGE + 1 次 END ≈ 2.5 步/局。相对对局内约 27 步/局只增加 9% 采样成本；
- 枚举 2^k 个 mask 需要在 ``action_scalar`` 里塞 k 个 bit，会改 ``action_scalar_width``，
  让所有现有 checkpoint 维度不兼容。

## 顺序依赖说明

换牌在规则上是**同时**决定的（选定的牌一起洗回、一起重抽），逐张决策引入了一个
虚假的顺序。但每张牌独立洗回，顺序不影响最终手牌分布，所以这个参数化不改变
可达策略集合——模型学到的仍然是「什么样的牌该换」。

## 与 strict 验收的关系

``perform_mulligan`` 要洗牌重抽，结果不可能被 ``LocalPredictor`` EXACT 预测，
所以换牌动作**不进** strict 逐动作验收（runner 在换牌阶段不调用 ``predict``）。
这不是静默 fallback：换牌动作有独立的执行路径、独立的可复现 RNG 派生、独立的
合法性检查（张数上限、uid 必须在手牌里、不能重复换同一张）。
"""

from __future__ import annotations

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.state.snapshot import BattleSnapshot

#: 影之诗起手 4 张（先手）/ 5 张（后手），因此单次换牌最多 5 张。
MULLIGAN_MAX_CHANGES = 5


class MulliganActionEnumerator:
    """把「换掉某张手牌 / 结束换牌」枚举成 ``LegalAction``。

    与 ``LocalActionEnumerator`` 的接口保持一致（``enumerate(snapshot, player=...)``），
    额外接受 ``already_changed``：已经决定要换的 uid 不再作为候选，避免同一张牌被换两次。
    """

    def __init__(self, *, max_changes: int = MULLIGAN_MAX_CHANGES) -> None:
        if max_changes < 0:
            raise ValueError("max_changes 不能为负")
        self.max_changes = max_changes

    def enumerate(
        self,
        snapshot: BattleSnapshot,
        *,
        player: str,
        already_changed: tuple[int, ...] = (),
    ) -> tuple[LegalAction, ...]:
        """返回当前可选的换牌动作。

        Args:
            snapshot: 换牌前的 snapshot（本阶段内保持不变）。
            player: ``"me"`` / ``"opponent"``。
            already_changed: 已经决定换掉的 uid。已换的牌不再作为候选，
                且当其数量达到 ``max_changes`` 时只剩 ``MULLIGAN_END``。

        Returns:
            候选动作元组。``MULLIGAN_END`` 永远在最后，因此任何情况下都非空。
        """
        if player not in {"me", "opponent"}:
            raise ValueError(f"player 必须是 me/opponent，得到 {player!r}")
        view = snapshot.me if player == "me" else snapshot.opponent
        changed = set(already_changed)

        actions: list[LegalAction] = []
        if len(changed) < self.max_changes:
            for uid in view.hand:
                if uid in changed:
                    continue
                actions.append(
                    LegalAction(
                        kind=ActionKind.MULLIGAN_CHANGE,
                        source_uid=int(uid),
                        legality_source="local",
                        confidence="exact",
                        description="换掉这张手牌",
                    )
                )
        actions.append(
            LegalAction(
                kind=ActionKind.MULLIGAN_END,
                source_uid=0,
                legality_source="local",
                confidence="exact",
                description="结束换牌",
            )
        )
        return tuple(actions)

    @staticmethod
    def validate_changes(
        snapshot: BattleSnapshot, player: str, changes: tuple[int, ...]
    ) -> None:
        """校验一组换牌决策的结构合法性（不含洗牌结果）。

        换牌不进 strict 逐动作验收，但**必须有这一层结构检查**：否则一个 bug 就能
        让对局悄悄换掉不属于该玩家的牌、或超出张数上限，且不会被任何断言发现。
        """
        if len(changes) > MULLIGAN_MAX_CHANGES:
            raise ValueError(
                f"换牌张数 {len(changes)} 超过上限 {MULLIGAN_MAX_CHANGES}"
            )
        if len(set(changes)) != len(changes):
            raise ValueError(f"换牌决策里有重复 uid: {changes}")
        view = snapshot.me if player == "me" else snapshot.opponent
        hand = set(view.hand)
        outside = [uid for uid in changes if uid not in hand]
        if outside:
            raise ValueError(f"换牌决策包含不在手牌里的 uid: {outside}")


__all__ = ["MULLIGAN_MAX_CHANGES", "MulliganActionEnumerator"]
