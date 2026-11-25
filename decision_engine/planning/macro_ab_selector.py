# -*- coding: utf-8 -*-
"""MacroTurnSearch 接入严格 A/B selector 体系的适配层。

把 ``MacroTurnSearch``（macro_turn_search.py，未修改）包装为 search_ab_match 的
``ArmSelector``：每个己方 turn 从 MacroTurnSearch 选出一条完整 ``TurnLine``，随后
按该 ``action_sequence`` 逐步返回动作；每一步返回的动作必须仍在**当前 snapshot 的
本地 legal actions** 中，否则给出结构化拒绝（``no_selection`` / ``plan_stale``），
绝不 fallback 到别的启发式或猜测动作。

设计要点（对齐 architecture-boundaries / prohibited-patterns）：

- **不修改** macro_turn_search.py / puct.py / strict_self_play.py / search_ab_match.py
  的既有语义；本模块只做组合与适配。
- **同预测器 / branch seed**：Macro selector 的 ``MacroTurnSearch`` 由注入的
  ``MacroSearchFactory(shared_predictor, branch_seed)`` 构造，与 A/B runner 共享同一
  ``PlayerLocalPredictor``；branch_seed 由 runner 派生并透传，保证 (snapshot, action,
  branch_seed) 可复现。
- **计划以 turn 为界重置**：以 ``(player, 行动方 PlayerView.turn)`` 作为计划身份键；
  turn 切换（新回合）时旧计划失效，重新做一次 MacroTurnSearch。
- **逐步合法性校验**：每一步用注入的 enumerator 重新枚举当前 snapshot 的本地 legal
  actions，计划里的下一步必须能在其中按稳定 action key 匹配到，否则 ``plan_stale``。
- **绝不 fallback**：搜索无 best_line / 计划步骤匹配不到 legal action / 计划耗尽但
  turn_end 不合法，均返回 ``selected_action_index=None`` 的结构化拒绝，由 runner 判为
  整局作废。不猜测动作、不改走其它 selector。

成本诊断（对齐 search_ab_match 的 cost 键约定）：
- ``predictor_call_count`` / ``micro_simulations`` / ``macro_node_costs`` 从
  MacroResult.diagnostics（_MacroCreditLedger.to_diagnostics 产出）抽取。
- 仅在真正触发一次新 MacroTurnSearch 的决策步计入这些搜索成本；同一 turn 内后续
  按计划逐步返回的决策步只计一次 enumerator 调用（macro 成本记 0），避免把一次
  turn 级搜索成本重复计到每个动作步。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.execution.player_local_predictor import PlayerLocalPredictor
from decision_engine.planning.macro_turn_search import MacroResult, TurnLine
from decision_engine.planning.search_ab_match import (  # noqa: F401  (action_match_key 仅为兼容 re-export)
    SearchArm,
    SelectorResult,
    action_match_key,
)


# ---------------------------------------------------------------------------
# 注入契约
# ---------------------------------------------------------------------------


class LocalEnumerator(Protocol):
    """本地 legal action 枚举器（player-aware）。"""

    def enumerate(self, snapshot: Any, *, player: str = "me") -> list[LegalAction]:
        ...


# MacroTurnSearch 工厂：接收共享 predictor + 本次 turn 决策的 branch_seed，
# 返回一个带 ``.search(snapshot, player=...) -> MacroResult`` 的对象（通常是
# MacroTurnSearch）。与 PuctSearchFactory 对称，保证同预测器/同 branch seed。
MacroSearchFactory = Callable[[PlayerLocalPredictor, int], Any]

# enumerator 工厂：接收共享 predictor，返回 player-aware 的本地枚举器。
EnumeratorFactory = Callable[[PlayerLocalPredictor], LocalEnumerator]

# ``action_match_key`` 已下沉到 ``search_ab_match``（见该模块定义），此处只作兼容
# re-export，供既有调用方与 ``test_macro_ab_selector`` 继续按旧路径引用。


# ---------------------------------------------------------------------------
# per-turn 计划
# ---------------------------------------------------------------------------


@dataclass
class _TurnPlan:
    """某个己方 turn 的完整 TurnLine 计划及消费进度。"""

    turn_key: tuple
    line: TurnLine
    action_sequence: tuple[LegalAction, ...]
    step_index: int = 0
    turn_end_returned: bool = False

    def next_planned_action(self) -> LegalAction | None:
        if self.step_index < len(self.action_sequence):
            return self.action_sequence[self.step_index]
        return None

    def is_sequence_exhausted(self) -> bool:
        return self.step_index >= len(self.action_sequence)


# ---------------------------------------------------------------------------
# MacroTurnPlanArmSelector
# ---------------------------------------------------------------------------


class MacroTurnPlanArmSelector:
    """把 MacroTurnSearch 的完整 TurnLine 计划适配为逐步 ArmSelector。

    生命周期：每个己方 turn 首个决策步触发一次 ``MacroTurnSearch.search``，缓存
    ``best_line.action_sequence`` 为本 turn 计划；后续决策步不重搜，只按计划顺序取下
    一步。turn 切换（turn_key 变化）时旧计划自动失效并重搜。

    合法性：每一步返回前用 enumerator 枚举当前 snapshot 的本地 legal actions，计划步
    骤必须能匹配到其中一个（按 action_match_key），否则 plan_stale 结构化拒绝。计划
    动作序列消费完后返回 legal actions 中的 TURN_END；若 TURN_END 不合法，plan_stale。
    """

    def __init__(
        self,
        predictor: PlayerLocalPredictor,
        search_factory: MacroSearchFactory,
        enumerator: LocalEnumerator,
    ):
        self._predictor = predictor
        self._search_factory = search_factory
        self._enumerator = enumerator
        self._plans: dict[str, _TurnPlan] = {}

    # -- turn 身份 --------------------------------------------------------

    @staticmethod
    def _turn_key(snapshot: Any, player: str) -> tuple:
        pv = snapshot.me if player == "me" else snapshot.opponent
        return (player, int(getattr(pv, "turn", 0)))

    # -- 主入口 -----------------------------------------------------------

    def select(self, snapshot: Any, *, player: str, branch_seed: int) -> SelectorResult:
        actions = tuple(self._enumerator.enumerate(snapshot, player=player))
        turn_key = self._turn_key(snapshot, player)

        plan = self._plans.get(player)
        searched_now = False
        search_diag: Mapping[str, Any] = {}

        # turn 切换或首次进入本 turn：重置并重新搜索完整计划。
        if plan is None or plan.turn_key != turn_key:
            plan, search_diag = self._build_plan(
                snapshot, player=player, branch_seed=branch_seed, turn_key=turn_key
            )
            searched_now = True
            if plan is None:
                # 搜索未产出 best_line：结构化 no_selection，绝不 fallback。
                return SelectorResult(
                    actions=actions,
                    selected_action_index=None,
                    diagnostics=self._cost_diag(
                        search_diag,
                        searched_now,
                        reason="no_selection:macro_no_best_line",
                        extra={
                            "macro_result_summary": {
                                key: search_diag.get(key)
                                for key in (
                                    "root_line_count",
                                    "macro_leaf_count",
                                    "terminal_leaf_count",
                                    "micro_search_count",
                                    "complete_line_count",
                                    "pending_prefix_count",
                                    "frontier_edge_count",
                                    "frontier_expanded",
                                    "frontier_pending",
                                    "lazy_budget_exhausted_events",
                                )
                            }
                        },
                    ),
                )
            self._plans[player] = plan

        # 取本 turn 计划的下一步。
        planned = plan.next_planned_action()
        if planned is not None:
            index = self._match_index(planned, actions)
            if index is None:
                # 计划动作已不在当前 legal actions 中：plan_stale，绝不猜测替代动作。
                return SelectorResult(
                    actions=actions,
                    selected_action_index=None,
                    diagnostics=self._cost_diag(
                        search_diag,
                        searched_now,
                        reason="plan_stale:planned_action_not_legal",
                        extra={
                            "planned_action": _describe_action(planned),
                            "step_index": plan.step_index,
                        },
                    ),
                )
            plan.step_index += 1
            return SelectorResult(
                actions=actions,
                selected_action_index=index,
                diagnostics=self._cost_diag(search_diag, searched_now),
            )

        # 计划的非 turn_end 动作序列已耗尽：返回 legal 的 TURN_END 收束本回合。
        if plan.turn_end_returned:
            # 不应发生（runner 在 turn_end 后会切换玩家并重置）；防御性 plan_stale。
            return SelectorResult(
                actions=actions,
                selected_action_index=None,
                diagnostics=self._cost_diag(
                    search_diag, searched_now, reason="plan_stale:plan_already_completed"
                ),
            )
        end_index = self._match_index(
            LegalAction(kind=ActionKind.TURN_END, source_uid=0), actions
        )
        if end_index is None:
            return SelectorResult(
                actions=actions,
                selected_action_index=None,
                diagnostics=self._cost_diag(
                    search_diag, searched_now, reason="plan_stale:turn_end_not_legal"
                ),
            )
        plan.turn_end_returned = True
        return SelectorResult(
            actions=actions,
            selected_action_index=end_index,
            diagnostics=self._cost_diag(search_diag, searched_now),
        )

    # -- helpers ----------------------------------------------------------

    def _build_plan(
        self, snapshot: Any, *, player: str, branch_seed: int, turn_key: tuple
    ) -> tuple[_TurnPlan | None, Mapping[str, Any]]:
        search = self._search_factory(self._predictor, branch_seed)
        result: MacroResult = search.search(snapshot, player=player)
        diag = dict(result.diagnostics)
        if result.best_line is None:
            return None, diag
        sequence = tuple(result.best_line.action_sequence)
        return (
            _TurnPlan(turn_key=turn_key, line=result.best_line, action_sequence=sequence),
            diag,
        )

    @staticmethod
    def _match_index(target: LegalAction, actions: tuple[LegalAction, ...]) -> int | None:
        key = action_match_key(target)
        for i, candidate in enumerate(actions):
            if action_match_key(candidate) == key:
                return i
        return None

    @staticmethod
    def _cost_diag(
        search_diag: Mapping[str, Any],
        searched_now: bool,
        *,
        reason: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """把 MacroResult.diagnostics 映射到 search_ab_match 标准 cost 键。

        只有真正触发一次新 MacroTurnSearch 的决策步（searched_now=True）才计搜索成本；
        同一 turn 内后续按计划逐步返回的步骤 macro/predictor/micro 成本记 0，避免把
        一次 turn 级搜索成本重复累加到每个动作步。
        """
        if searched_now:
            predictor_calls = int(search_diag.get("predictor_call_count") or 0)
            micro_sims = int(search_diag.get("micro_executed_simulations") or 0)
            macro_nodes = int(search_diag.get("macro_turn_expansion_count") or 0) + int(
                search_diag.get("macro_successor_node_count") or 0
            )
            micro_search_count = int(search_diag.get("micro_search_count") or 0)
            macro_turn_transitions = int(search_diag.get("macro_turn_transition_count") or 0)
        else:
            predictor_calls = 0
            micro_sims = 0
            macro_nodes = 0
            micro_search_count = 0
            macro_turn_transitions = 0
        cost: dict[str, Any] = {
            "predictor_call_count": predictor_calls,
            "micro_simulations": micro_sims,
            "macro_node_costs": macro_nodes,
            "micro_search_count": micro_search_count,
            "macro_turn_transition_count": macro_turn_transitions,
            "macro_searched_this_step": searched_now,
            "search_name": "macro_turn_search",
        }
        if reason is not None:
            cost["macro_rejection_reason"] = reason
        if extra:
            cost.update(extra)
        return cost


def _describe_action(action: LegalAction) -> dict[str, Any]:
    return {
        "kind": action.kind.value,
        "source_uid": action.source_uid,
        "play_kind": action.play_kind.value if action.play_kind is not None else None,
        "target_groups": [list(g) for g in action.target_groups],
        "variant": action.variant,
    }


# ---------------------------------------------------------------------------
# 两臂 factory：baseline legacy Macro vs candidate lazy/static Macro
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroArmConfig:
    """一臂 Macro selector 的可序列化配置（用于 config hash / manifest 区分）。

    只声明 MacroTurnSearch 的行为开关；不含 predictor / branch seed（那些由 runner
    共享注入，保证两臂同预测器 / 同 branch scheduler）。
    """

    variant: str = "legacy"  # "legacy" | "lazy" | "static"
    macro_depth: int = 3
    k: int = 3
    root_k: int = 5
    use_incremental: bool = False
    extra_pp_signature_portfolio: bool = False
    lazy_mode: bool = False
    static_resource_mode: bool = False
    bucket_active_quota: int = 2
    # 全局 credit 预算（lazy 模式真实 enforce；legacy 只观测）。None 表示不声明。
    max_predictor_calls: int | None = None
    max_micro_simulations: int | None = None
    max_macro_nodes: int | None = None
    max_turn_transitions: int | None = None

    def to_config_mapping(self) -> dict[str, Any]:
        """产出 SearchArm.config 用的可序列化映射，能区分 legacy / candidate。"""
        return {
            "search": "macro_turn_search",
            "variant": self.variant,
            "macro_depth": self.macro_depth,
            "k": self.k,
            "root_k": self.root_k,
            "use_incremental": self.use_incremental,
            "extra_pp_signature_portfolio": self.extra_pp_signature_portfolio,
            "lazy_mode": self.lazy_mode,
            "static_resource_mode": self.static_resource_mode,
            "bucket_active_quota": self.bucket_active_quota,
            "credit_budget": {
                "max_predictor_calls": self.max_predictor_calls,
                "max_micro_simulations": self.max_micro_simulations,
                "max_macro_nodes": self.max_macro_nodes,
                "max_turn_transitions": self.max_turn_transitions,
            },
        }


def legacy_macro_arm_config(**overrides: Any) -> MacroArmConfig:
    """baseline：legacy k/root_k 递归 Macro（lazy/static 均关闭）。"""
    base: dict[str, Any] = {
        "variant": "legacy",
        "lazy_mode": False,
        "static_resource_mode": False,
    }
    base.update(overrides)
    return MacroArmConfig(**base)


def lazy_macro_arm_config(**overrides: Any) -> MacroArmConfig:
    """candidate：资源感知惰性前沿 Macro（lazy_mode=True，全局 credit 真实 enforce）。"""
    base: dict[str, Any] = {
        "variant": "lazy",
        "lazy_mode": True,
        "static_resource_mode": False,
    }
    base.update(overrides)
    return MacroArmConfig(**base)


def static_macro_arm_config(**overrides: Any) -> MacroArmConfig:
    """candidate：静态资源 Macro evaluator（static_resource_mode=True）。"""
    base: dict[str, Any] = {
        "variant": "static",
        "lazy_mode": False,
        "static_resource_mode": True,
    }
    base.update(overrides)
    return MacroArmConfig(**base)


def build_macro_arm(
    name: str,
    config: MacroArmConfig,
    *,
    macro_search_factory: MacroSearchFactory,
    enumerator_factory: EnumeratorFactory,
) -> SearchArm:
    """构造一臂 Macro ``SearchArm``。

    macro_search_factory(shared_predictor, branch_seed) -> MacroTurnSearch-like：
        由调用方注入，负责用共享 predictor + branch_seed 构造 MacroTurnSearch，保证同
        预测器 / 同 branch scheduler。config 的行为开关应在此 factory 内落实到
        MacroTurnSearch 构造参数（本模块不替调用方决定微观搜索/leaf/terminal 组件）。
    enumerator_factory(shared_predictor) -> LocalEnumerator：
        由调用方注入，产出 player-aware 本地枚举器（如 ExtraPpAwareEnumerator）。

    这样两臂可注入不同 config / 不同 MacroTurnSearch 组装，但共享同一 predictor。
    """

    def arm_factory(shared_predictor: PlayerLocalPredictor) -> MacroTurnPlanArmSelector:
        enumerator = enumerator_factory(shared_predictor)
        return MacroTurnPlanArmSelector(
            shared_predictor,
            macro_search_factory,
            enumerator,
        )

    return SearchArm(name=name, factory=arm_factory, config=config.to_config_mapping())
