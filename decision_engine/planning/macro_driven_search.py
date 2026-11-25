# -*- coding: utf-8 -*-
"""宏观新价值 → 动作级适配器（Macro-driven action search adapter）。

让 MacroTurnSearch（回合线级、逐动作即时收益价值）驱动 SelfPlaySearchRunner
（动作级循环），无需改动 runner / PUCT。

设计（混合，最贴合现有架构、风险最低）：
- policy 方向由微观 PUCT 提供（visit_counts → policy_target），训练 policy 头不退化。
- 实际走子（selected_action）由宏观新价值决定：MacroTurnSearch 选出 best_line，
  按序执行其动作序列；一条线执行完，下个我方回合重新宏观搜索。
- root_value 用宏观 backed_up_value（新价值尺度），仅作诊断，不参与训练 value 目标
  （训练 value 目标仍用对局最终 outcome，见 run_self_play_iteration 的 outcome_hard）。

这样：生成的对局轨迹反映新价值的决策，而 policy_target/value_target 的训练语义不变。
"""

from __future__ import annotations

from dataclasses import dataclass

from decision_engine.planning.puct import PuctSearchResult


class MacroDrivenSearch:
    """把 MacroTurnSearch 包装成 runner 期望的动作级 search 接口。

    对外只暴露 `search(snapshot, *, player)` 和 `config`/`prior`/`value`（runner 读取诊断用）。
    """

    def __init__(self, *, macro_search, micro_search, config):
        """
        macro_search: MacroTurnSearch（use_incremental=True，含 residual_fn）
        micro_search: 微观 PUCT（DiffAwarePuctSearch 或 PuctSearch），提供 policy 方向
        config: SelfPlaySearchConfig（runner 读 simulations 等）
        """
        self._macro = macro_search
        self._micro = micro_search
        self.config = config
        # runner 读取这些做诊断标签
        self.prior = getattr(micro_search, "prior", None)
        self.value = getattr(micro_search, "value", None)
        # 当前活动回合线的剩余动作队列（按 base_card_id/kind 匹配到微观 actions）
        self._pending_line = None
        self._pending_player = None

    def _micro_result(self, snapshot, player):
        """跑微观 PUCT，拿 policy 方向。DiffAwarePuctSearch 需先 notify_turn_end。"""
        if hasattr(self._micro, "notify_turn_end"):
            # DiffAwarePuctSearch：新回合起点重置 baseline
            pass  # baseline 由 runner 的回合边界管理；这里每次 search 都在同回合内
        return self._micro.search(snapshot, player=player)

    def _match_action_index(self, target_action, actions):
        """在微观 actions 里找到与宏观回合线下一动作匹配的索引。

        匹配依据：kind + source_uid（同一动作在同局面枚举稳定）。
        """
        if target_action is None:
            return None
        for i, a in enumerate(actions):
            if (a.kind == target_action.kind
                    and getattr(a, "source_uid", None) == getattr(target_action, "source_uid", None)):
                return i
        # 退化：只匹配 kind
        for i, a in enumerate(actions):
            if a.kind == target_action.kind:
                return i
        return None

    def search(self, snapshot, *, player):
        micro_res = self._micro_result(snapshot, player)
        actions = micro_res.actions
        if not actions:
            return micro_res  # 无合法动作，直接透传

        # 是否需要重新宏观规划：无活动回合线 / 换了行动方 / 上条线已执行完
        need_replan = (
            self._pending_line is None
            or self._pending_player != player
            or not self._pending_line
        )
        macro_root_value = micro_res.root_value
        if need_replan:
            try:
                macro_res = self._macro.search(snapshot, player=player)
                macro_root_value = macro_res.backed_up_value
                if macro_res.best_line is not None:
                    # 缓存 best_line 的动作序列（下一动作从队首取）
                    self._pending_line = list(macro_res.best_line.action_sequence)
                    self._pending_player = player
                else:
                    self._pending_line = []
            except Exception:
                # 宏观搜索失败：回退到微观决策（不阻断生成）
                self._pending_line = []

        # 取宏观回合线的下一个动作，匹配到微观 actions 索引
        target = self._pending_line.pop(0) if self._pending_line else None
        idx = self._match_action_index(target, actions)
        if idx is None:
            # 宏观建议的动作在当前微观枚举里找不到（预测偏离）：回退微观 selected
            idx = micro_res.selected_action_index
            # 偏离则清空回合线，下次重新规划
            self._pending_line = []

        # 构造结果：policy_target 用微观（训练 policy 不退化），selected 用宏观决策
        diag = dict(micro_res.diagnostics)
        diag["macro_driven"] = True
        diag["macro_root_value"] = macro_root_value
        diag["value_source"] = "macro_incremental"
        diag["policy_target_source"] = "micro_visit_counts"
        return PuctSearchResult(
            actions=actions,
            visit_counts=micro_res.visit_counts,
            policy_target=micro_res.policy_target,
            selected_action_index=idx,
            root_value=micro_res.root_value,   # 训练 value 语义不变（±1 outcome 由 runner 管）
            trace=micro_res.trace,
            diagnostics=diag,
        )

    def notify_turn_end(self):
        """回合结束：清空活动回合线，透传给微观（DiffAwarePuctSearch 重置 baseline）。"""
        self._pending_line = None
        self._pending_player = None
        if hasattr(self._micro, "notify_turn_end"):
            self._micro.notify_turn_end()
