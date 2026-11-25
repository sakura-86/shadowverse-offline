# -*- coding: utf-8 -*-
"""Extra PP 自动展开枚举器。

当 extra_pp 可用时，不产出独立 EXTRA_PP 动作，而是直接将"激活 extra_pp 后
解锁的新动作"合并到主动作列表中（用 variant="extra_pp" 标记）。

设计原理：
- Extra PP 本质是"展开选项空间"的前置操作，不是独立决策
- 先激活 extra_pp 才能看到 cost == pp+1 的出牌选项
- 最终是否"消耗"extra_pp 取决于回合结束时 PP 是否 >= 1
  （如果剩余 PP >= 1，实际不扣 extra_pp）
- T5 第一个 extra_pp 即将过期（T6 发放覆盖），应默认激活

Predictor 对 variant="extra_pp" 的动作需要先执行 activate_extra_pp 再执行实际动作。
"""

from __future__ import annotations

from dataclasses import replace

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.actions.dedup_enumerator import DeduplicatedActionEnumerator
from decision_engine.state.snapshot import BattleSnapshot


# T5 后手的第一个 extra_pp 在 T6 被覆盖，所以 T5 应该默认花掉
_FIRST_EXTRA_PP_EXPIRY_TURN = 5


class ExtraPpAwareEnumerator:
    """在去重枚举器基础上自动展开 extra_pp 解锁的动作。

    行为：
    1. 如果 extra_pp 不可用或已激活 → 退化为普通 DeduplicatedActionEnumerator
    2. 如果 extra_pp 可用：
       a. 用 DeduplicatedActionEnumerator 在当前 snapshot 上枚举基础动作（不含 EXTRA_PP）
       b. 在"pp+1"的虚拟 snapshot 上重新枚举，找出新解锁的动作
       c. 新解锁的动作用 variant="extra_pp" 标记，合并到结果中
       d. 移除独立的 EXTRA_PP 动作
    3. T5 且第一个 extra_pp 未使用：所有动作都在激活后状态枚举
       （因为第一个 extra_pp 不用就浪费了）
    """

    def __init__(self, base: DeduplicatedActionEnumerator | None = None):
        self.base = base or DeduplicatedActionEnumerator()

    def enumerate(self, snapshot: BattleSnapshot, *, player: str = "me") -> list[LegalAction]:
        # 检查 extra_pp 状态
        pv = snapshot.me if player == "me" else snapshot.opponent
        extra_pp = pv.extra_pp

        has_extra_pp = (
            extra_pp is not None
            and extra_pp.extra_pp > 0
            and not extra_pp.is_used_extra_pp_this_turn
        )

        if not has_extra_pp:
            # 无 extra_pp 可用，正常枚举
            return self.base.enumerate(snapshot, player=player)

        # T5 且持有第一个 extra_pp（即将在 T6 被覆盖）→ 全部在激活后枚举
        is_expiring = pv.turn == _FIRST_EXTRA_PP_EXPIRY_TURN and not pv.is_first_side

        if is_expiring:
            return self._enumerate_all_with_extra_pp(snapshot, player=player)

        # 正常模式：基础动作 + 额外解锁的动作
        return self._enumerate_merged(snapshot, player=player)

    def _enumerate_all_with_extra_pp(
        self, snapshot: BattleSnapshot, *, player: str
    ) -> list[LegalAction]:
        """T5 过期场景：所有动作在激活 extra_pp 后的状态枚举。

        所有动作都标记 variant="extra_pp"（因为都是在 pp+1 状态下枚举的）。
        turn_end 保持不标记（结束回合不需要先激活）。
        """
        activated_snapshot = self._activate_extra_pp_virtual(snapshot, player=player)
        actions = self.base.enumerate(activated_snapshot, player=player)

        result = []
        for a in actions:
            if a.kind == ActionKind.EXTRA_PP:
                continue  # 移除独立 EXTRA_PP
            if a.kind == ActionKind.TURN_END:
                result.append(a)  # turn_end 不标记
            else:
                result.append(replace(a, variant="extra_pp"))
        return result

    def _enumerate_merged(
        self, snapshot: BattleSnapshot, *, player: str
    ) -> list[LegalAction]:
        """正常模式：基础动作 + 新解锁的 extra_pp 动作合并。"""
        # 基础动作（不含 EXTRA_PP）
        base_actions = self.base.enumerate(snapshot, player=player)
        base_actions = [a for a in base_actions if a.kind != ActionKind.EXTRA_PP]

        # 在激活后的 snapshot 上枚举
        activated_snapshot = self._activate_extra_pp_virtual(snapshot, player=player)
        activated_actions = self.base.enumerate(activated_snapshot, player=player)

        # 找出新解锁的动作（在基础动作中不存在的）
        base_keys = set()
        for a in base_actions:
            base_keys.add(self._action_identity_key(a))

        new_actions = []
        for a in activated_actions:
            if a.kind == ActionKind.EXTRA_PP:
                continue
            if a.kind == ActionKind.TURN_END:
                continue  # turn_end 已在 base_actions 中
            key = self._action_identity_key(a)
            if key not in base_keys:
                new_actions.append(replace(a, variant="extra_pp"))

        return base_actions + new_actions

    def _activate_extra_pp_virtual(
        self, snapshot: BattleSnapshot, *, player: str
    ) -> BattleSnapshot:
        """构造"extra_pp 已激活"的虚拟 snapshot（PP+1, 标记已使用）。"""
        pv = snapshot.me if player == "me" else snapshot.opponent
        extra_pp = pv.extra_pp

        new_extra_pp = replace(
            extra_pp,
            preparation_extra_pp=1,
            is_used_extra_pp_this_turn=True,
        )
        new_pv = replace(pv, pp=pv.pp + 1, extra_pp=new_extra_pp)

        if player == "me":
            return replace(snapshot, me=new_pv)
        else:
            return replace(snapshot, opponent=new_pv)

    @staticmethod
    def _action_identity_key(action: LegalAction) -> tuple:
        """动作的身份 key，用于判断两个动作是否等价（忽略 variant）。"""
        return (
            action.kind,
            action.source_uid,
            action.play_kind,
            action.cost,
            action.target_groups,
            action.selected_indexes,
            action.material_uids,
        )
