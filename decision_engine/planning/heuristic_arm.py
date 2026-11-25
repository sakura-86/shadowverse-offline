# -*- coding: utf-8 -*-
"""L1 启发式对手：无搜索、零 predict 预算的基线策略。

## 定位

PPO 的对手阶梯（`docs/plan/PPO_IGNITION_CHECK.md` §4）里，L1 是最弱的一级：

| 级别 | 对手 | 每局耗时 |
|---|---|---|
| **L1** | 本模块：最高费出牌 + 斩杀优先 + 免费换怪 | ~50 ms |
| L2 | `plan_turn(max_actions=1)` 单步贪婪 | ~190 ms |
| L3 | `plan_turn(max_actions=3)` | ~600 ms |
| L4 | 完整 `plan_turn` | 1.5–46 s |

它存在的意义不是"强"，而是**给 PPO 一个连牌都不会打的起点对手**，并作为 L2 的对照
（如果 L2 的增益主要来自"用了评分函数"，那 L1→L2 的差距应该明显大于 L2 内部的随机性）。

## 决策顺序

1. **斩杀**：我方可用攻击力总和 ≥ 对方生命 → 直接打脸；
2. **出牌**：按 `cost` 从高到低（用户指定的规则；这条与真实目标无关，所以只放在最弱一级）；
3. **进化**：优先超进化（数值更强），其次普通进化；
4. **攻击**：能一刀换掉对方随从（攻 ≥ 目标血量）就先换，否则打脸；
5. **启动** → **回合结束**。

## 与"完整"实现的差距（有意为之）

- 不读 `evolve_desc`/"进化时效果"（那在 `UnifiedCardTemplate` 里，需要加载模板）；
  只用协议级 `can_evolve` / `can_super_evolve`。
- 不做攻击分配的搜索，按固定顺序取第一个可用项。
- 出牌只看费用，不看效果文本。

这些差距就是 L1 与 L2/L3/L4 的难度梯度来源。
"""

from __future__ import annotations

from typing import Any, Sequence

from decision_engine.actions.model import ActionKind
from decision_engine.planning.search_ab_match import SelectorResult
from decision_engine.planning.turn_plan_arm import strict_prediction_filter


def _views(snapshot: Any, player: str) -> tuple[Any, Any]:
    return (snapshot.me, snapshot.opponent) if player == "me" else (snapshot.opponent, snapshot.me)


def _first_target_uid(action: Any) -> int | None:
    for group in action.target_groups or ():
        for uid in group:
            return int(uid)
    return None


class HeuristicArmSelector:
    """L1：固定优先级启发式，不搜索、不调用 predictor 做评估。"""

    def __init__(
        self,
        predictor: Any,
        enumerator: Any,
        *,
        prediction_filter=strict_prediction_filter,
    ) -> None:
        self._predictor = predictor
        self._enumerator = enumerator
        self._prediction_filter = prediction_filter

    # ------------------------------------------------------------------
    def _executable(self, snapshot: Any, actions: Sequence[Any], player: str) -> list[int]:
        """runner 会拒绝任何一步非 strict 的预测（整局作废），所以要先筛。"""
        keep: list[int] = []
        for index, action in enumerate(actions):
            try:
                prediction = self._predictor.predict(snapshot, action, player=player)
            except Exception:
                continue
            if self._prediction_filter(prediction):
                keep.append(index)
        return keep

    def select(self, snapshot: Any, *, player: str, branch_seed: int) -> SelectorResult:
        actions = tuple(self._enumerator.enumerate(snapshot, player=player))
        if not actions:
            return SelectorResult(
                actions=(), selected_action_index=None,
                diagnostics={"arm": "heuristic_l1", "reject": "no_legal_actions"},
            )

        executable = self._executable(snapshot, actions, player)
        if not executable:
            return SelectorResult(
                actions=actions, selected_action_index=None,
                diagnostics={"arm": "heuristic_l1", "reject": "no_strict_action"},
            )

        mine, theirs = _views(snapshot, player)
        cards = snapshot.cards_by_uid

        def pick(indices: Sequence[int]) -> SelectorResult:
            return SelectorResult(
                actions=actions,
                selected_action_index=int(indices[0]),
                diagnostics={"arm": "heuristic_l1", "micro_simulations": 0, "macro_node_costs": 0,
                             "predictor_call_count": len(actions)},
            )

        by_kind: dict[ActionKind, list[int]] = {}
        for index in executable:
            by_kind.setdefault(actions[index].kind, []).append(index)

        # 1. 斩杀：可用攻击力总和够打到 0 血。
        leader_indices = by_kind.get(ActionKind.ATTACK_LEADER, [])
        if leader_indices:
            available = 0
            for uid in mine.field:
                card = cards.get(uid)
                if card is None or not card.can_attack:
                    continue
                available += max(0, int(card.atk))
            if available >= int(theirs.life):
                return pick(leader_indices)

        # 2. 出牌：费用从高到低。
        play_indices = by_kind.get(ActionKind.PLAY, [])
        if play_indices:
            play_indices = sorted(
                play_indices,
                key=lambda i: (
                    int(actions[i].cost if actions[i].cost is not None else -1),
                    actions[i].source_uid,
                ),
                reverse=True,
            )
            return pick(play_indices)

        # 3. 进化：超进化优先（数值更强），其次普通进化。
        for kind in (ActionKind.SUPER_EVOLVE, ActionKind.EVOLVE):
            indices = by_kind.get(kind, [])
            if indices:
                return pick(indices)

        # 4. 攻击：能一刀换掉对方随从就先换（攻 ≥ 目标血量），否则打脸。
        swaps = []
        for index in by_kind.get(ActionKind.ATTACK_FOLLOWER, []):
            action = actions[index]
            attacker = cards.get(action.source_uid)
            target_uid = _first_target_uid(action)
            target = cards.get(target_uid) if target_uid is not None else None
            if attacker is not None and target is not None and int(attacker.atk) >= int(target.life):
                swaps.append(index)
        if swaps:
            return pick(swaps)
        if leader_indices:
            return pick(leader_indices)
        follower_indices = by_kind.get(ActionKind.ATTACK_FOLLOWER, [])
        if follower_indices:
            return pick(follower_indices)

        # 5. 启动 → 回合结束。
        for kind in (ActionKind.ACTIVATE, ActionKind.TURN_END):
            indices = by_kind.get(kind, [])
            if indices:
                return pick(indices)

        # 兜底：取第一个可执行动作（仍受 strict 约束，不是静默降级到非法路径）。
        return pick(executable)


__all__ = ["HeuristicArmSelector"]
