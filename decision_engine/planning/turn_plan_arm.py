# -*- coding: utf-8 -*-
"""把 `plan_turn`（回合规划器）和「纯 policy prior」接成 A/B 对战的两臂。

`search_ab_match.StrictSearchAbRunner` 要求两臂实现 `ArmSelector.select`，共享同一个
`PlayerLocalPredictor`，每对 (镜像牌组, seed) 跑 candidate 先手 / 后手两局。

两个 selector：

- :class:`TurnPlanArmSelector`（candidate 臂）：每个己方 turn 用 :func:`plan_turn`
  规划整条路线，然后逐步返回。规划器只用本地规则预测器，**不读模型**。
- :class:`PolicyPriorArmSelector`（baseline 臂）：模型 policy head 的 prior 取 argmax，
  无搜索。这是「不用搜索器」的对照。

两臂都不做启发式兜底：规划步骤在当前 legal actions 里找不到、或没有合法动作，都返回
``selected_action_index=None`` 的结构化拒绝，由 runner 判整局作废。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.execution.player_local_predictor import PlayerLocalPredictor
from decision_engine.planning.search_ab_match import SearchArm, SelectorResult, action_match_key
from decision_engine.planning.turn_planner import plan_turn


#: A/B runner 的 strict 口径：只有 EXACT / SPEC_ONLY 且无 unresolved 的预测才会被执行。
STRICT_CERTAINTY_VALUES = frozenset({"exact", "spec_only"})


def strict_prediction_filter(prediction: Any) -> bool:
    certainty = getattr(prediction, "certainty", None)
    value = getattr(certainty, "value", certainty)
    return str(value) in STRICT_CERTAINTY_VALUES and not getattr(prediction, "unresolved_effects", ())


def rank_burn_pp_candidates(actions: Sequence[LegalAction], *, pp: int) -> list[int]:
    """无事可做时的出牌顺序：把剩余 PP 花完，**费用从高到低**。

    为什么需要它：``plan_turn`` 只规划"能改变净血差"的动作。只能铺场时它返回空路线，
    纯规则锚点若直接 ``turn_end`` 就永远不建场面，等于白送对局（实测 plan-vs-plan
    24 对全胜、opponent 从不 play）。这里把"花完 PP"作为确定性的推进规则：
    高费先打（价值密度高、且更容易把 PP 用尽），同费用保持枚举顺序以保证可复现。

    只返回 ``PLAY`` 且 ``cost <= pp`` 的下标；``cost is None`` 视为 0（保守可打）。
    """
    candidates = [
        (index, action)
        for index, action in enumerate(actions)
        if action.kind is ActionKind.PLAY and (action.cost is None or action.cost <= pp)
    ]
    # 费用降序；同费用按枚举顺序（Python sort 稳定，显式带 index 更清楚）。
    candidates.sort(key=lambda pair: (-(pair[1].cost or 0), pair[0]))
    return [index for index, _ in candidates]


def _turn_number(snapshot: Any, player: str) -> int:
    view = snapshot.me if player == "me" else snapshot.opponent
    return int(getattr(view, "turn", 0) or 0)


@dataclass
class TurnPlanArmConfig:
    """A 臂的规划配置。"""

    max_actions: int = 12
    allow_turn_end: bool = True
    #: 传给 ``plan_turn`` 的搜索结构：``phased``（默认，快）/ ``dfs``（严格最优但会爆）。
    search_mode: str = "phased"
    #: 只在「规划器给出了有区分度的最优解」时才由规划器决定。False 表示永远用规划器。
    require_discrimination: bool = False
    #: prior 引导的候选裁剪：每层只保留 prior 前 ``prior_top_k`` 个候选（0 = 关闭）。
    #:
    #: 目的不是「更准」，而是**用剪枝换深度**——根分支从 13 降到 5 后，同样的时间
    #: 预算能展开更深的回合路线。``turn_end`` 始终保留（否则无法结束回合）。
    prior_top_k: int = 0
    #: 在哪几层做裁剪（含）。0 = 只裁根节点。``plan_turn`` 每个节点调一次 prior 是
    #: 模型前向，深度放大后成本会线性上升，所以默认只裁根。
    prior_depth: int = 0

    def to_config_mapping(self) -> dict[str, Any]:
        return {
            "arm": "turn_plan",
            "max_actions": self.max_actions,
            "allow_turn_end": self.allow_turn_end,
            "search_mode": self.search_mode,
            "require_discrimination": self.require_discrimination,
            "prior_top_k": self.prior_top_k,
            "prior_depth": self.prior_depth,
        }


@dataclass
class PolicyArmConfig:
    """B 臂（纯 prior argmax）的配置。"""

    temperature: float = 0.0  # 0 = argmax，只做确定性基线

    def to_config_mapping(self) -> dict[str, Any]:
        return {"arm": "policy_prior", "temperature": self.temperature}


@dataclass
class _ActivePlan:
    turn_key: tuple
    route: tuple[LegalAction, ...]
    step_index: int = 0
    score: Any = None

    def exhausted(self) -> bool:
        return self.step_index >= len(self.route)


class TurnPlanArmSelector:
    """candidate 臂：每个己方 turn 规划一次，然后按计划逐步返回。"""

    def __init__(
        self,
        predictor: PlayerLocalPredictor,
        enumerator: Any,
        *,
        config: TurnPlanArmConfig | None = None,
        fallback_prior: Any = None,
    ) -> None:
        self._predictor = predictor
        self._enumerator = enumerator
        self._config = config or TurnPlanArmConfig()
        self._fallback_prior = fallback_prior
        # 裁剪用的 prior 与兜底 prior 是同一个模型，但语义分开：兜底只在「规划器给不出
        # 有区分度的解」时用，裁剪则直接改变搜索的候选集。用同一实例、不同参数。
        self._prune_prior = fallback_prior if self._config.prior_top_k > 0 else None
        self._plan: _ActivePlan | None = None
        self._replans = 0
        self._planned_nodes = 0

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _match(actions: tuple[LegalAction, ...], intended: LegalAction) -> int | None:
        wanted = action_match_key(intended)
        for index, action in enumerate(actions):
            if action_match_key(action) == wanted:
                return index
        return None

    def _legal_actions(self, snapshot: Any, player: str) -> tuple[LegalAction, ...]:
        return tuple(self._enumerator.enumerate(snapshot, player=player))

    def _make_plan(self, snapshot: Any, player: str) -> _ActivePlan:
        plan = plan_turn(
            snapshot,
            predictor=self._predictor,
            enumerator=self._enumerator,
            player=player,
            max_actions=self._config.max_actions,
            allow_turn_end=self._config.allow_turn_end,
            search_mode=self._config.search_mode,
            # 和 runner 的 strict 验收用同一个过滤器：规划器不能规划出 runner 拒绝执行的动作
            prediction_filter=strict_prediction_filter,
            # prior 引导的候选裁剪（prior_top_k=0 时不生效）
            action_prior=self._prune_prior,
            prior_top_k=self._config.prior_top_k,
            prior_depth=self._config.prior_depth,
        )
        self._replans += 1
        self._planned_nodes += plan.nodes
        return _ActivePlan(turn_key=(player, _turn_number(snapshot, player)), route=tuple(plan.route), score=plan.score)

    def _prior_choice(self, snapshot: Any, actions: tuple[LegalAction, ...], player: str) -> int | None:
        if self._fallback_prior is None or not actions:
            return None
        result = self._fallback_prior.priors(snapshot, actions, player=player)
        priors = tuple(getattr(result, "priors", ()) or ())
        if len(priors) != len(actions):
            return None
        return max(range(len(actions)), key=lambda index: priors[index])

    def _burn_pp_choice(
        self,
        snapshot: Any,
        actions: tuple[LegalAction, ...],
        player: str,
    ) -> int | None:
        """把剩余 PP 花完：按费用降序找第一个预测可 strict 执行的 ``PLAY``。

        必须过 ``strict_prediction_filter``：runner 会拒绝预测非 EXACT/SPEC_ONLY 的动作
        并判整局作废——盲目选一张打不出去的牌，比空转更糟。
        """
        view = snapshot.me if player == "me" else snapshot.opponent
        pp = int(getattr(view, "pp", 0) or 0)
        for index in rank_burn_pp_candidates(actions, pp=pp):
            try:
                prediction = self._predictor.predict(snapshot, actions[index], player=player)
            except Exception:
                continue
            if strict_prediction_filter(prediction):
                return index
        return None

    # -- ArmSelector ------------------------------------------------------

    def select(self, snapshot: Any, *, player: str, branch_seed: int) -> SelectorResult:
        actions = self._legal_actions(snapshot, player)
        if not actions:
            return SelectorResult(
                actions=(),
                selected_action_index=None,
                diagnostics={"reject": "no_legal_actions", "arm": "turn_plan"},
            )

        turn_key = (player, _turn_number(snapshot, player))
        same_turn = self._plan is not None and self._plan.turn_key == turn_key
        # 只在「本回合第一次」或「上一轮**非空**路线已走完」时重新规划。
        # 空路线（plan_turn 判定本回合无收益）每回合只规划一次：否则每次 select 都会重跑
        # 一遍 plan_turn 搜索，铺场后分支因子变大，单局会从 2s 涨到分钟级。
        if not same_turn or (self._plan is not None and self._plan.exhausted() and self._plan.route):
            self._plan = self._make_plan(snapshot, player)

        intended = self._plan.route[self._plan.step_index] if not self._plan.exhausted() else None
        diagnostics: dict[str, Any] = {
            "arm": "turn_plan",
            "predictor_call_count": self._planned_nodes,
            "macro_node_costs": self._planned_nodes,
            "micro_simulations": 0,
            "replans": self._replans,
            "plan_length": len(self._plan.route),
            "plan_step": self._plan.step_index,
        }

        if intended is not None:
            match = self._match(actions, intended)
            if match is not None:
                self._plan.step_index += 1
                return SelectorResult(actions=actions, selected_action_index=match, diagnostics=diagnostics)

            # 计划作废：规划时用的是 `branch_seed=None` 的确定性预测，runner 执行时用
            # `branch_seed(seed, turn, action_index)`，遇到随机目标/抽牌会走向不同分支，
            # 后续 uid 就对不上了。从**当前**快照重新规划一次，第一步一定是从当前
            # legal actions 枚举出来的，必然可匹配。
            self._plan = self._make_plan(snapshot, player)
            diagnostics["replans"] = self._replans
            diagnostics["replan_reason"] = "plan_stale"
            if not self._plan.route:
                diagnostics["reject"] = "empty_plan_after_replan"
                return SelectorResult(actions=actions, selected_action_index=None, diagnostics=diagnostics)
            match = self._match(actions, self._plan.route[0])
            if match is None:
                diagnostics["reject"] = "plan_stale_after_replan"
                diagnostics["intended_action"] = str(self._plan.route[0].kind.value)
                return SelectorResult(actions=actions, selected_action_index=None, diagnostics=diagnostics)
            self._plan.step_index = 1
            diagnostics["plan_length"] = len(self._plan.route)
            diagnostics["plan_step"] = 1
            return SelectorResult(actions=actions, selected_action_index=match, diagnostics=diagnostics)

        # plan_turn 无话可说时，先把剩余 PP 花完（高费优先），再考虑 prior / 结束回合。
        burn = self._burn_pp_choice(snapshot, actions, player)
        if burn is not None:
            diagnostics["fallback"] = "burn_pp"
            diagnostics["burn_pp_cost"] = actions[burn].cost
            return SelectorResult(actions=actions, selected_action_index=burn, diagnostics=diagnostics)

        index = self._prior_choice(snapshot, actions, player)
        if index is None:
            # plan_turn 只规划"能改变局面"的动作；当本回合确实无事可做（route 为空）时，
            # 唯一合法的推进就是结束回合。这不是启发式兜底：TURN_END 是 enumerator 给出的
            # 合法动作，且只在「没有任何可规划动作」时才会走到这里。
            if self._config.allow_turn_end:
                turn_end = next(
                    (i for i, action in enumerate(actions) if action.kind is ActionKind.TURN_END),
                    None,
                )
                if turn_end is not None:
                    diagnostics["fallback"] = "turn_end_only"
                    return SelectorResult(actions=actions, selected_action_index=turn_end, diagnostics=diagnostics)
            diagnostics["reject"] = "empty_plan_no_fallback"
            return SelectorResult(actions=actions, selected_action_index=None, diagnostics=diagnostics)
        diagnostics["fallback"] = "policy_prior"
        return SelectorResult(actions=actions, selected_action_index=index, diagnostics=diagnostics)


class PolicyPriorArmSelector:
    """baseline 臂：模型 policy head 的 prior 取 argmax，没有搜索。

    按 prior 从高到低找一个**预测可被 strict 验收**的动作：runner 要求每一步执行的
    预测都是 EXACT/SPEC_ONLY 且无 unresolved，纯 argmax 会挑到 PARTIAL 的动作，整局
    直接作废。这不是给 baseline 加搜索，只是让它在和 candidate 相同的验收口径下
    能被比较。
    """

    def __init__(
        self,
        predictor: PlayerLocalPredictor,
        enumerator: Any,
        *,
        prior: Any,
        prediction_filter: Callable[[Any], bool] = strict_prediction_filter,
    ) -> None:
        self._predictor = predictor
        self._enumerator = enumerator
        self._prior = prior
        self._prediction_filter = prediction_filter

    def select(self, snapshot: Any, *, player: str, branch_seed: int) -> SelectorResult:
        actions = tuple(self._enumerator.enumerate(snapshot, player=player))
        if not actions:
            return SelectorResult(
                actions=(), selected_action_index=None, diagnostics={"reject": "no_legal_actions", "arm": "policy_prior"}
            )
        result = self._prior.priors(snapshot, actions, player=player)
        priors = tuple(getattr(result, "priors", ()) or ())
        if len(priors) != len(actions):
            return SelectorResult(
                actions=actions,
                selected_action_index=None,
                diagnostics={"reject": "prior_length_mismatch", "arm": "policy_prior"},
            )
        order = sorted(range(len(actions)), key=lambda index: -priors[index])
        checked = 0
        for index in order:
            try:
                prediction = self._predictor.predict(snapshot, actions[index], player=player)
            except Exception:
                checked += 1
                continue
            checked += 1
            if self._prediction_filter(prediction):
                return SelectorResult(
                    actions=actions,
                    selected_action_index=index,
                    diagnostics={
                        "arm": "policy_prior",
                        "predictor_call_count": checked,
                        "macro_node_costs": 0,
                        "micro_simulations": 0,
                        "top_prior": round(float(priors[index]), 6),
                        "prior_sum": round(float(sum(priors)), 6),
                        "strict_checked": checked,
                    },
                )
        return SelectorResult(
            actions=actions,
            selected_action_index=None,
            diagnostics={
                "arm": "policy_prior",
                "reject": "no_strict_action",
                "predictor_call_count": checked,
                "strict_checked": checked,
            },
        )

    def select_mulligan(
        self, snapshot: Any, actions: Sequence[LegalAction], *, player: str, branch_seed: int
    ) -> SelectorResult:
        """换牌阶段：按 prior 取 argmax，**不做 strict 可执行性筛选**。

        换牌要洗牌重抽，``LocalPredictor`` 无法 EXACT 预测，逐候选 predict 只会全部
        失败。合法性由 ``MulliganActionEnumerator.validate_changes`` 做结构检查。
        """
        actions = tuple(actions)
        if not actions:
            return SelectorResult(
                actions=(),
                selected_action_index=None,
                diagnostics={"arm": "policy_prior", "reject": "no_mulligan_actions"},
            )
        result = self._prior.priors(snapshot, actions, player=player)
        priors = tuple(getattr(result, "priors", ()) or ())
        if len(priors) != len(actions):
            return SelectorResult(
                actions=actions,
                selected_action_index=None,
                diagnostics={"arm": "policy_prior", "reject": "prior_length_mismatch"},
            )
        best = max(range(len(actions)), key=lambda index: priors[index])
        return SelectorResult(
            actions=actions,
            selected_action_index=best,
            diagnostics={
                "arm": "policy_prior",
                "predictor_call_count": 0,
                "macro_node_costs": 0,
                "micro_simulations": 0,
                "top_prior": round(float(priors[best]), 6),
                "mulligan_stage": True,
            },
        )


@dataclass
class MixedFirstActionConfig:
    """分工臂：**模型决定本回合第一个动作**，其余交给 ``plan_turn`` 补全。

    动机（``docs/plan/PPO_IGNITION_CHECK.md`` §35）：模型的候选覆盖率已经很高
    （plan_turn 首动作 95.6% 在模型 top-5 里），但 top-1 只有 54.6%，且训练会让它
    继续下降。所以让模型做"选第一个动作"这件事（这个位置承载跨回合意图：
    这回合走脸还是解场、要不要留资源），把"这一回合剩下的最优序列"交给穷举规划器。

    与 ``TurnPlanArmSelector`` 的唯一差别：每回合的**第一次** ``select`` 不走
    ``plan_turn``，而是用模型 prior 取 argmax（仍然要过 ``strict_prediction_filter``）；
    从第二步起完全走 ``plan_turn``。
    """

    max_actions: int = 12
    allow_turn_end: bool = True
    search_mode: str = "phased"

    def to_config_mapping(self) -> dict[str, Any]:
        return {
            "arm": "mixed_first_action",
            "max_actions": self.max_actions,
            "allow_turn_end": self.allow_turn_end,
            "search_mode": self.search_mode,
        }


class MixedFirstActionSelector:
    """每回合第一步用模型 prior，其余步用 ``plan_turn``。"""

    def __init__(
        self,
        predictor: PlayerLocalPredictor,
        enumerator: Any,
        *,
        plan_config: TurnPlanArmConfig,
        prior: Any,
        fallback_prior: Any = None,
    ) -> None:
        self._planner = TurnPlanArmSelector(
            predictor, enumerator, config=plan_config, fallback_prior=fallback_prior
        )
        self._first = PolicyPriorArmSelector(predictor, enumerator, prior=prior)
        self._turn_key: tuple | None = None
        self._first_done = False
        #: 诊断计数，便于核对"模型真的只决定了首动作"。
        self.model_first_decisions = 0
        self.plan_decisions = 0

    @staticmethod
    def _tag(result: SelectorResult, phase: str) -> SelectorResult:
        diagnostics = dict(result.diagnostics or {})
        diagnostics["arm"] = "mixed_first_action"
        diagnostics["phase"] = phase
        return SelectorResult(
            actions=result.actions,
            selected_action_index=result.selected_action_index,
            diagnostics=diagnostics,
        )

    def select(self, snapshot: Any, *, player: str, branch_seed: int) -> SelectorResult:
        turn_key = (player, _turn_number(snapshot, player))
        if turn_key != self._turn_key:
            self._turn_key = turn_key
            self._first_done = False

        if not self._first_done:
            self._first_done = True
            self.model_first_decisions += 1
            return self._tag(
                self._first.select(snapshot, player=player, branch_seed=branch_seed),
                "model_first",
            )

        self.plan_decisions += 1
        return self._tag(
            self._planner.select(snapshot, player=player, branch_seed=branch_seed),
            "plan_rest",
        )

    def select_mulligan(self, snapshot: Any, actions: Sequence[LegalAction], *, player: str, branch_seed: int) -> SelectorResult:
        return self._first.select_mulligan(snapshot, actions, player=player, branch_seed=branch_seed)


def build_turn_plan_arm(
    name: str,
    config: TurnPlanArmConfig,
    *,
    enumerator_factory: Callable[[PlayerLocalPredictor], Any],
    prior: Any = None,
) -> SearchArm:
    def arm_factory(shared_predictor: PlayerLocalPredictor) -> TurnPlanArmSelector:
        return TurnPlanArmSelector(
            shared_predictor,
            enumerator_factory(shared_predictor),
            config=config,
            fallback_prior=prior,
        )

    return SearchArm(name=name, factory=arm_factory, config=config.to_config_mapping())


def build_policy_prior_arm(
    name: str,
    config: PolicyArmConfig,
    *,
    enumerator_factory: Callable[[PlayerLocalPredictor], Any],
    prior: Any,
) -> SearchArm:
    def arm_factory(shared_predictor: PlayerLocalPredictor) -> PolicyPriorArmSelector:
        return PolicyPriorArmSelector(shared_predictor, enumerator_factory(shared_predictor), prior=prior)

    return SearchArm(name=name, factory=arm_factory, config=config.to_config_mapping())


def build_mixed_first_action_arm(
    name: str,
    config: MixedFirstActionConfig,
    *,
    enumerator_factory: Callable[[PlayerLocalPredictor], Any],
    prior: Any,
    fallback_prior: Any = None,
) -> SearchArm:
    """模型决定本回合首动作、``plan_turn`` 补全其余步的 A/B 臂。"""

    def arm_factory(shared_predictor: PlayerLocalPredictor) -> MixedFirstActionSelector:
        return MixedFirstActionSelector(
            shared_predictor,
            enumerator_factory(shared_predictor),
            plan_config=TurnPlanArmConfig(
                max_actions=config.max_actions,
                allow_turn_end=config.allow_turn_end,
                search_mode=config.search_mode,
            ),
            prior=prior,
            fallback_prior=fallback_prior,
        )

    return SearchArm(name=name, factory=arm_factory, config=config.to_config_mapping())


class StrictPuctArmSelector:
    """PUCT 臂 + strict 动作筛选。

    `ArmSelector` 返回的动作由 runner 重新预测并按 strict 口径验收；`PuctSearch`
    内部不知道这个约束，会选出预测为 PARTIAL 的动作，导致整局作废（实测 4 对里
    作废 2 对）。

    这里的做法：先看 PUCT 自己选的那个动作能不能 strict 预测；不能就按 root 的
    visit count 从高到低找第一个能 strict 预测的。**不改搜索语义**——PUCT 已经算完
    了，这里只是在它的候选里换一个可执行的动作。
    """

    def __init__(
        self,
        predictor: PlayerLocalPredictor,
        search_factory: Callable[[PlayerLocalPredictor, int], Any],
        *,
        prediction_filter: Callable[[Any], bool] = strict_prediction_filter,
    ) -> None:
        self._predictor = predictor
        self._search_factory = search_factory
        self._prediction_filter = prediction_filter

    def select(self, snapshot: Any, *, player: str, branch_seed: int) -> SelectorResult:
        search = self._search_factory(self._predictor, branch_seed)
        result = search.search(snapshot, player=player)
        actions = tuple(getattr(result, "actions", ()) or ())
        if not actions:
            return SelectorResult(
                actions=(), selected_action_index=None, diagnostics={"reject": "no_legal_actions", "arm": "puct_strict"}
            )
        visits = tuple(getattr(result, "visit_counts", ()) or ())
        diag = dict(getattr(result, "diagnostics", {}) or {})
        diagnostics: dict[str, Any] = {
            "arm": "puct_strict",
            "predictor_call_count": int(diag.get("predictor_call_count") or 0),
            "micro_simulations": int(diag.get("executed_simulations") or 0),
            "macro_node_costs": int(diag.get("turn_transition_count") or 0),
        }

        order: list[int] = []
        selected = getattr(result, "selected_action_index", None)
        if selected is not None and 0 <= selected < len(actions):
            order.append(int(selected))
        if len(visits) == len(actions):
            for index in sorted(range(len(actions)), key=lambda i: -int(visits[i])):
                if index not in order:
                    order.append(index)
        else:
            for index in range(len(actions)):
                if index not in order:
                    order.append(index)

        checked = 0
        for index in order:
            try:
                prediction = self._predictor.predict(snapshot, actions[index], player=player)
            except Exception:
                checked += 1
                continue
            checked += 1
            if self._prediction_filter(prediction):
                diagnostics["strict_checked"] = checked
                if selected is not None and index != selected:
                    diagnostics["puct_choice_overridden"] = True
                diagnostics["visits"] = int(visits[index]) if index < len(visits) else 0
                return SelectorResult(actions=actions, selected_action_index=index, diagnostics=diagnostics)
        diagnostics["strict_checked"] = checked
        diagnostics["reject"] = "no_strict_action"
        return SelectorResult(actions=actions, selected_action_index=None, diagnostics=diagnostics)


def build_strict_puct_arm(
    name: str,
    config: Mapping[str, Any],
    *,
    search_factory: Callable[[PlayerLocalPredictor, int], Any],
) -> SearchArm:
    def arm_factory(shared_predictor: PlayerLocalPredictor) -> StrictPuctArmSelector:
        return StrictPuctArmSelector(shared_predictor, search_factory)

    return SearchArm(name=name, factory=arm_factory, config=dict(config))


__all__ = [
    "StrictPuctArmSelector",
    "build_strict_puct_arm",
    "TurnPlanArmConfig",
    "TurnPlanArmSelector",
    "PolicyArmConfig",
    "PolicyPriorArmSelector",
    "build_turn_plan_arm",
    "build_policy_prior_arm",
    "strict_prediction_filter",
]
