# -*- coding: utf-8 -*-
"""PUCT tree search — two-player, multi-turn.

支持跨回合搜索：turn_end 后自动切换 active_player，执行 start_turn，
枚举对方动作并继续展开，直到达到 max_turns 深度限制或终局。

关键语义：
- active_player: 当前节点由谁决策 ("me" / "opponent")
- turn_number: 从 root 的 turn 开始计数的逻辑回合序号（每次 turn_end 递增）
- max_turns: 搜索覆盖的总回合数（默认 3 = 我方 + 对方 + 我方下回合）
- value backprop: 跨越 player 边界时取反（对方好 = 我方差）
- max_turns=1 时退化为旧行为（turn_end 是叶节点）

抽牌处理（初版）：
- drawn_card=None: turn_start 时减 deck_count 但不加入实际卡
- 后续将改为 unknown card 建模
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace as dc_replace
from math import sqrt
from typing import Any, Mapping, Protocol, Sequence

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.actions.player_local_enumerator import PlayerLocalActionEnumerator
from decision_engine.actions.player_perspective import opponent_as_me, restore_opponent_perspective
from decision_engine.execution.execution_context import LocalExecutionContext
from decision_engine.planning.search_prior import SearchPrior, UniformSearchPrior
from decision_engine.planning.search_value import SearchValue, TerminalValue
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot

_logger = logging.getLogger(__name__)


class SearchPredictor(Protocol):
    def predict(self, snapshot: BattleSnapshot, action: LegalAction, *, player: str = "me"):
        ...


class TurnTransitioner(Protocol):
    """回合切换接口：执行 turn_end + 对方 start_turn。"""
    def transition_turn(self, snapshot: BattleSnapshot, ending_player: str) -> BattleSnapshot:
        ...


@dataclass(frozen=True)
class _ForcedDeepeningOutcome:
    """forced greedy-prior 单次深度展开的结果（reservation / high-prior deepening 共用）。"""
    executed: bool
    prediction_error_count: int
    prediction_error_details: list[dict[str, Any]]


@dataclass
class _UcbSimulationDeltas:
    """单次常规 UCB simulation 的诊断增量。

    把原先内联在 search() UCB 循环里的局部累加器抽出为显式返回值，使
    search() 与可恢复的 MicroTurnCompletionSession 共用同一条 UCB simulation
    执行路径（单一真源），从而保证 (snapshot, action, branch_seed) 一致性、
    reservation/deepening/UCB 三阶段语义不因复用而分叉。

    depth_reached：本次 simulation 抵达的最大深度（供 max_depth_reached 更新）。
    trace_entry：simulation_trace 条目（None 表示未采集）。
    """

    executed: bool = False
    depth_reached: int = 0
    prediction_error_count: int = 0
    prediction_error_details: list[dict[str, Any]] = field(default_factory=list)
    partial_prediction_count: int = 0
    partial_prediction_details: list[str] = field(default_factory=list)
    greedy_rollout_count: int = 0
    cross_turn_count: int = 0
    trace_entry: dict[str, Any] | None = None


@dataclass(frozen=True)
class PuctActionTrace:
    action_index: int
    action_kind: str
    prior: float
    visit_count: int
    q_value: float
    u_value: float
    rejected: bool = False
    rejection_reason: str = ""


@dataclass(frozen=True)
class PuctSearchResult:
    actions: tuple[LegalAction, ...]
    visit_counts: tuple[int, ...]
    policy_target: tuple[float, ...]
    selected_action_index: int | None
    root_value: float
    trace: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class PuctConfig:
    simulations: int = 64
    max_depth: int = 9
    max_turns: int = 1
    c_puct: float = 1.5
    discount: float = 1.0
    stop_on_predict_error: bool = False
    reject_predict_error: bool = False
    branch_seed: int | None = 0
    # Root Dirichlet noise for exploration (AlphaZero style)
    dirichlet_alpha: float = 0.3      # noise concentration (lower = more peaked)
    dirichlet_epsilon: float = 0.25   # noise weight (0 = no noise, 1 = pure noise)
    # Prior deepening: 对 root 节点 prior > 阈值的动作，预留 simulations 做深度优先展开
    # 确保高 prior 的复杂 combo（play → evolve → attack）至少被完整探索一次
    prior_deepening_threshold: float = 0.15   # prior > 此值的动作触发 deepening
    prior_deepening_sims: int = 4             # 每个触发动作分配的 deepening simulation 次数（最小值）
    prior_deepening_ratio: float = 0.3        # deepening 阶段占总 simulations 的比例上限
    # Root action reservation: 为 prior 较低的非 turn_end 根动作预留 simulation 预算，
    # 强制做 greedy-prior 深度展开，保证被 high-prior deepening / UCB 忽视的动作至少被访问一次。
    # 目的：探索罕见但可能关键的路线（如低 prior 的斩杀/解场），消除 prior 单点主导导致的探索盲区。
    # 默认 0（关闭）时完全保持原有行为，不引入任何额外 simulation 或语义变化。
    # 该 phase 在 high-prior deepening 之前执行；三个 phase（reservation + deepening + UCB）
    # 合计严格不超过 simulations，并尽量等于 simulations。
    # 根动作按 (prior 升序, action_index 升序) 排序选取，保证低 prior 优先且平局用稳定索引打破。
    root_reservation_ratio: float = 0.0       # reservation 阶段占总 simulations 的比例上限（0 = 关闭）
    root_reservation_min_visits: int = 1      # 每个被预留的低 prior 根动作强制访问的最小次数
    # 给 root 的 turn_end 边单独预留的强制评估次数（0 = 关闭，保持旧行为）。
    # turn_end 的 prior 通常极低（21~26 个动作时只有 0.001~0.02），在 UCB 里竞争不过
    # 已积累访问的攻击边，于是永远拿不到 Q；但它可能正是最优解。
    # 见 docs/plan/SEARCH_MYOPIA_DIAGNOSIS.md。
    turn_end_min_visits: int = 0
    # 同上，但按 simulations 的比例给出预算（与 min_visits 取较大者）。
    # 实测固定次数不够：S3 在 128 sims 下 16 次够，256 sims 下 16 次又输给积累了
    # 更多访问的攻击边；按比例缩放才能在任意预算下都评估充分。
    # 0.25 是实测下限：S1（8 个动作）需要 32/128 才能让 turn_end 的 Q 优势兑现。
    # 该预算只用于「把 Q 测出来」，最终仍由 visit-count 选择；S5（不该停）在 0.25
    # 下依旧选 play，不会误停。
    turn_end_reservation_ratio: float = 0.25
    # 开启「未访问优先」：先给每个合法子边各一次评估，再交给 UCB 竞争。
    # 默认关闭以保持既有搜索语义；与 turn_end_reservation_ratio 叠加时对
    # 动作数很多的根节点更稳（S4 在 26 个动作下靠它拿到 67 visits）。
    unvisited_first: bool = False
    # Debug-only: 保留前 N 次常规 simulation 的根节点 PUCB 选择与回传。
    # 0 表示关闭；该 trace 不参与选择，不改变搜索语义。
    simulation_trace_limit: int = 0
    #: **未使用行动值（PP）的惩罚系数**。0 = 关闭，保持旧搜索语义。
    #:
    #: 动机：`turn_end` 被当成普通动作和别的比 Q，而它的 Q 是**局面优势**——
    #: 不会因为放弃剩余 PP 而变差。`plan_turn` 的 DFS 只展开非 turn_end 动作，
    #: 结构上不可能放弃行动值，所以它从不「提前停手」。
    #:
    #: 惩罚只加在「**主动**放弃」上：`_handle_turn_end` 里检查父节点的动作集合
    #: 是否还有非 turn_end 选项。被迫停手（没 PP / 没手牌 / 场地满）不罚，
    #: 因为那时 `pp` 通常已经是 0，或者 `has_alternative` 为 False。
    #:
    #: 量级参考：`BoardAdvantageValue` 输出在 [-1, 1]，4 费牌约值几点 net_life，
    #: 所以 1 PP 大概在 0.02–0.05。
    unused_pp_penalty: float = 0.0


# ---------------------------------------------------------------------------
# PARTIAL prediction 标记（有可用 snapshot 但含未解析效果）
# ---------------------------------------------------------------------------


class _PartialPrediction:
    """Wrapper: predict 成功但有 unresolved_effects，作为叶节点用 value 估值。"""
    __slots__ = ("prediction", "unresolved_effects")

    def __init__(self, prediction, unresolved_effects: tuple[str, ...]):
        self.prediction = prediction
        self.unresolved_effects = unresolved_effects

    @property
    def snapshot(self):
        return self.prediction.snapshot


# ---------------------------------------------------------------------------
# 默认回合切换实现
# ---------------------------------------------------------------------------


class _PredictionError:
    """Wrapper: predict 失败，保留错误细节用于诊断。"""
    __slots__ = ("action", "unresolved_effects", "description", "player")

    def __init__(self, action: LegalAction, unresolved_effects: tuple[str, ...], description: str, player: str):
        self.action = action
        self.unresolved_effects = unresolved_effects
        self.description = description
        self.player = player


# ---------------------------------------------------------------------------
# 默认回合切换实现（DefaultTurnTransitioner）
# ---------------------------------------------------------------------------


class DefaultTurnTransitioner:
    """使用 LocalPredictor 执行回合切换（包含 event queue effects）。

    ending_player turn_end 已在 predictor 中处理完毕，这里只负责
    对方 start_turn（刷新 PP、can_attack、countdown、谢幕曲等）。
    """

    def __init__(self, predictor=None, *, branch_seed: int | None = None):
        from decision_engine.execution.rule_kernel import CoreRuleKernel
        self._kernel = CoreRuleKernel()
        self._predictor = predictor  # PlayerLocalPredictor | None
        self._branch_seed = branch_seed  # 与 self-play / PUCT 动作预测统一注入 execution_context
        # P0 credit ledger：只观测实际由此 transitioner 发起的 predictor 调用，
        # 不参与任何搜索选择或预算截断。
        self.predictor_call_count = 0

    def transition_turn(self, snapshot: BattleSnapshot, ending_player: str) -> BattleSnapshot:
        """在 ending_player 的 turn_end after-state 基础上，执行对方 start_turn。"""
        next_player = "opponent" if ending_player == "me" else "me"

        # 如果有 predictor，通过完整路径执行（包含 event queue effects）
        if self._predictor is not None:
            from decision_engine.actions.model import ActionKind, LegalAction
            from decision_engine.execution.local_match_setup import draw_next_from_deck_instances
            drawn_card, draw_snapshot = draw_next_from_deck_instances(snapshot, next_player)
            turn_start_action = LegalAction(kind=ActionKind.TURN_START, source_uid=0)
            generated = (drawn_card,) if drawn_card is not None else ()
            # 与 self-play (_start_turn) / PUCT 动作预测 (_predict_with_seed) 统一：
            # 注入带 branch_seed 的 execution_context，消除 turn_start 随机效果的 draw_binding
            # 与 deck_instances 不一致（DeckFeatureError 根因之一：三条路径 execution_context 割裂）。
            method = getattr(self._predictor, "predict_with_replay_context", None)
            if method is not None and self._branch_seed is not None:
                from decision_engine.execution.execution_context import LocalExecutionContext
                ctx = LocalExecutionContext(branch_seed=self._branch_seed)
                self.predictor_call_count += 1
                result = method(
                    draw_snapshot, turn_start_action, player=next_player,
                    generated_cards=generated, execution_context=ctx,
                )
            else:
                self.predictor_call_count += 1
                result = self._predictor.predict_with_generated_cards(
                    draw_snapshot, turn_start_action, player=next_player, generated_cards=generated
                )
            return result.snapshot

        # Fallback: 裸 kernel（无效果执行，用于无 predictor 的轻量搜索）
        result = self._kernel.start_turn(snapshot, next_player, drawn_card=None)
        return result.snapshot


# ---------------------------------------------------------------------------
# 搜索树节点
# ---------------------------------------------------------------------------


class _PuctNode:
    """搜索树节点。每个节点对应一个 snapshot + 当前决策方 + 合法动作集。"""

    __slots__ = (
        "snapshot", "actions", "priors", "children",
        "visit_counts", "value_sums", "rejection_reasons",
        "total_visits", "is_terminal", "terminal_value",
        "is_depth_leaf", "active_player", "turn_number",
        "turn_action_depth",
    )

    def __init__(
        self,
        snapshot: BattleSnapshot,
        actions: tuple[LegalAction, ...],
        priors: tuple[float, ...],
        *,
        active_player: str = "me",
        turn_number: int = 1,
        is_terminal: bool = False,
        terminal_value: float = 0.0,
        is_depth_leaf: bool = False,
        turn_action_depth: int = 0,
    ):
        self.snapshot = snapshot
        self.actions = actions
        self.priors = priors
        self.children: list[_PuctNode | None] = [None] * len(actions)
        self.visit_counts: list[int] = [0] * len(actions)
        self.value_sums: list[float] = [0.0] * len(actions)
        self.rejection_reasons: list[str] = [""] * len(actions)
        self.total_visits: int = 0
        self.is_terminal = is_terminal
        self.terminal_value = terminal_value
        self.is_depth_leaf = is_depth_leaf
        self.active_player = active_player
        self.turn_number = turn_number
        # 从所属回合的回合根节点起算的 0-based 回合内深度。root（回合根）为 0；
        # 回合内每展开一层非 turn_end 动作 +1；跨回合 _handle_turn_end 产生的新回合根重置为 0。
        # 预测某条边时，该边的 1-based 回合内 action_index = parent.turn_action_depth + 1，
        # 用于派生与 strict self-play / search_ab_match 执行侧一致的 branch_seed。
        self.turn_action_depth = turn_action_depth

    def q_value(self, index: int) -> float:
        visits = self.visit_counts[index]
        return self.value_sums[index] / visits if visits > 0 else 0.0

    def select_child(self, c_puct: float) -> int:
        """选择 UCB 最高的未被 reject 的子边。"""
        parent_visits = self.total_visits + 1
        best_index = 0
        best_score = float("-inf")
        for i in range(len(self.actions)):
            if self.rejection_reasons[i]:
                continue
            q = self.q_value(i)
            u = c_puct * self.priors[i] * sqrt(parent_visits) / (1 + self.visit_counts[i])
            score = q + u
            if score > best_score:
                best_score = score
                best_index = i
        return best_index


# ---------------------------------------------------------------------------
# PuctSearch 主类
# ---------------------------------------------------------------------------


class PuctSearch:
    def __init__(
        self,
        *,
        enumerator: PlayerLocalActionEnumerator | None = None,
        child_enumerator: PlayerLocalActionEnumerator | None = None,
        predictor: SearchPredictor | None = None,
        prior: SearchPrior | None = None,
        value: SearchValue | None = None,
        turn_transitioner: TurnTransitioner | None = None,
        config: PuctConfig | None = None,
    ):
        self.enumerator = enumerator or PlayerLocalActionEnumerator()
        self.child_enumerator = child_enumerator or PlayerLocalActionEnumerator()
        self.predictor = predictor
        self.prior = prior or UniformSearchPrior()
        self.value = value or TerminalValue()
        self.turn_transitioner = turn_transitioner
        self.config = config or PuctConfig()
        self.baseline_name = "n1_two_player_puct"
        self.baseline_version = 2
        self.neural_network_used = False

    def search(self, snapshot: BattleSnapshot, *, player: str = "me") -> PuctSearchResult:
        root_player = player  # root 视角，backprop 始终从此视角计算 value
        # P0 credit ledger：计数只服务于诊断，不参与PUCT选择、value或预算。
        self._predictor_call_count = 0
        self._turn_transition_count = 0
        self._turn_transition_predictor_call_count = 0

        # Clear prior state cache for this new search
        if hasattr(self.prior, "clear_cache"):
            self.prior.clear_cache()
        if hasattr(self.value, "clear_cache"):
            self.value.clear_cache()

        # 构建 root 节点
        actions = tuple(self.enumerator.enumerate(snapshot, player=player))
        if not actions:
            raise ValueError("puct search requires at least one legal action")
        prior_result = self.prior.priors(snapshot, actions, player=player)
        prior_result.validate(len(actions))

        # === 规则化 Prior 修正 ===
        # 当存在明显更优动作时，压低 turn_end 的 prior，避免浪费搜索预算
        root_priors = _apply_prior_corrections(actions, prior_result.priors)

        # Root Dirichlet noise: 确保低 prior 动作也能被探索
        if self.config.dirichlet_epsilon > 0 and len(actions) > 1:
            import numpy as np
            rng = np.random.default_rng(self.config.branch_seed)
            noise = rng.dirichlet([self.config.dirichlet_alpha] * len(actions))
            eps = self.config.dirichlet_epsilon
            root_priors = tuple(
                (1 - eps) * p + eps * n for p, n in zip(root_priors, noise)
            )

        root = _PuctNode(
            snapshot, actions, root_priors,
            active_player=player, turn_number=1,
        )

        simulations = max(1, self.config.simulations)
        total_budget = simulations  # 严格预算基准：reservation + deepening + UCB 合计不得超过
        executed_simulations = 0
        prediction_error_count = 0
        prediction_error_details: list[dict[str, Any]] = []
        partial_prediction_count = 0
        partial_prediction_details: list[str] = []
        greedy_rollout_count = 0
        greedy_rollout_max_steps = 0
        max_depth_reached = 0
        cross_turn_count = 0
        simulation_trace: list[dict[str, Any]] = []

        # 诊断：root reservation 阶段统计
        reservation_eligible = 0
        reservation_requested = 0
        reservation_completed = 0

        # === Turn-end Exploration Phase ===
        # 给 root 的 turn_end 边固定预算，先把它的 Q 评估出来。
        # 动机：turn_end 的 prior 在 21~26 个动作时只有 0.001~0.02，UCB 里竞争不过
        # 已积累访问的攻击边，于是永远拿不到 Q（实测 0~1 visits），而它可能正是最优解。
        # 默认 turn_end_min_visits=0 时整段跳过，保持旧行为。
        turn_end_completed = 0
        turn_end_budget = max(
            int(self.config.turn_end_min_visits),
            int(simulations * max(0.0, self.config.turn_end_reservation_ratio)),
        )
        if turn_end_budget > 0:
            for i in range(len(root.actions)):
                if not _is_turn_end_action(root.actions[i]) or root.rejection_reasons[i]:
                    continue
                for _ in range(turn_end_budget):
                    if executed_simulations >= total_budget:
                        break
                    outcome = self._run_forced_deepening_simulation(root, i, root_player)
                    prediction_error_count += outcome.prediction_error_count
                    for detail in outcome.prediction_error_details:
                        if len(prediction_error_details) < 10:
                            prediction_error_details.append(detail)
                    if not outcome.executed:
                        break
                    turn_end_completed += 1
                    executed_simulations += 1
            simulations = max(1, simulations - turn_end_completed)

        # === Root Action Reservation Phase ===
        # 为 prior 较低的非 turn_end 根动作预留 simulation 预算，强制 greedy-prior 深度展开，
        # 保证被 high-prior deepening / UCB 忽视的低 prior 动作至少被访问 min_visits 次。
        # 默认 root_reservation_ratio=0 时整段跳过，保持原有行为不变。
        # 根动作排序：prior 升序（低 prior 优先），平局用 action_index 升序打破（稳定索引）。
        # 该 phase 严格在 high-prior deepening 之前执行，其消耗从总 simulations 中扣除。
        if self.config.root_reservation_ratio > 0 and self.config.root_reservation_min_visits > 0:
            eligible_edges = [
                i for i in range(len(root.actions))
                if not _is_turn_end_action(root.actions[i]) and not root.rejection_reasons[i]
            ]
            # turn_end 单独预留：低 prior 的 turn_end 在 UCB 里永远竞争不过已积累
            # 访问的攻击边（S3 实测 0~1 visits），而它可能正是最优解。给它固定的
            # 评估预算，先把 Q 测出来，再由最终选择逻辑决定是否采用。
            turn_end_edges = [
                i for i in range(len(root.actions))
                if _is_turn_end_action(root.actions[i]) and not root.rejection_reasons[i]
            ] if self.config.turn_end_min_visits > 0 else []
            reservation_eligible = len(eligible_edges)
            if eligible_edges:
                # 低 prior 优先，稳定索引打破平局
                ordered = sorted(eligible_edges, key=lambda i: (root.priors[i], i))
                min_visits = self.config.root_reservation_min_visits
                # 总预算上限 = simulations * ratio，且不超过覆盖所有 eligible edge 所需
                budget_cap = max(1, int(simulations * self.config.root_reservation_ratio))
                needed = len(ordered) * min_visits
                reservation_budget = min(budget_cap, needed, simulations)

                for i in ordered:
                    if reservation_completed >= reservation_budget:
                        break
                    for _ in range(min_visits):
                        if reservation_completed >= reservation_budget or root.rejection_reasons[i]:
                            break
                        reservation_requested += 1
                        outcome = self._run_forced_deepening_simulation(root, i, root_player)
                        prediction_error_count += outcome.prediction_error_count
                        for detail in outcome.prediction_error_details:
                            if len(prediction_error_details) < 10:
                                prediction_error_details.append(detail)
                        if outcome.executed:
                            reservation_completed += 1
                            executed_simulations += 1
                        else:
                            # edge 被 reject（如 predict_error + reject_predict_error），
                            # 不再对该 edge 重复请求
                            break

                simulations = max(1, simulations - reservation_completed)

        # === Prior Deepening Phase ===
        # 对高 prior 的 root 动作强制做深度优先展开，确保复杂 combo 至少被完整探索
        if self.config.prior_deepening_threshold > 0 and self.config.prior_deepening_ratio > 0:
            # 计算触发 deepening 的动作数
            deepening_actions = [i for i in range(len(root.actions)) if root.priors[i] >= self.config.prior_deepening_threshold]
            deepening_done = 0
            if deepening_actions:
                # 总预算 = simulations * ratio，均分给每个触发动作
                total_deepening_budget = max(1, int(simulations * self.config.prior_deepening_ratio))
                per_action_budget = max(self.config.prior_deepening_sims, total_deepening_budget // len(deepening_actions))

                for i in deepening_actions:
                    # 对这个动作做 per_action_budget 次强制 simulation
                    for _ in range(per_action_budget):
                        if deepening_done >= total_deepening_budget or root.rejection_reasons[i]:
                            break
                        # 严格预算保护：deepening 不得使总执行数超过 total_budget
                        if executed_simulations >= total_budget:
                            break
                        outcome = self._run_forced_deepening_simulation(root, i, root_player)
                        prediction_error_count += outcome.prediction_error_count
                        for detail in outcome.prediction_error_details:
                            if len(prediction_error_details) < 10:
                                prediction_error_details.append(detail)
                        if outcome.executed:
                            deepening_done += 1
                            executed_simulations += 1
                        else:
                            break

            # 减去已使用的 deepening budget
            simulations = max(1, simulations - deepening_done)

        # 最终常规 UCB simulation 次数：严格钳制到剩余预算，
        # 使 reservation + deepening + UCB 合计不超过 total_budget，且尽量等于。
        ucb_simulations = max(0, min(simulations, total_budget - executed_simulations))
        for _ in range(ucb_simulations):
            collect_trace = len(simulation_trace) < self.config.simulation_trace_limit
            deltas = self._run_ucb_simulation(
                root, root_player,
                collect_trace=collect_trace,
                simulation_index=executed_simulations + 1,
            )
            prediction_error_count += deltas.prediction_error_count
            for detail in deltas.prediction_error_details:
                if len(prediction_error_details) < 10:
                    prediction_error_details.append(detail)
            partial_prediction_count += deltas.partial_prediction_count
            for pd in deltas.partial_prediction_details:
                if len(partial_prediction_details) < 5:
                    partial_prediction_details.append(pd)
            greedy_rollout_count += deltas.greedy_rollout_count
            cross_turn_count += deltas.cross_turn_count
            if deltas.depth_reached > max_depth_reached:
                max_depth_reached = deltas.depth_reached
            if deltas.trace_entry is not None:
                simulation_trace.append(deltas.trace_entry)
            executed_simulations += 1

        # 构建结果
        selected_index = None if all(root.rejection_reasons) else _argmax_stable(root.visit_counts)
        rejected_count = sum(1 for r in root.rejection_reasons if r)
        if rejected_count > 0:
            rejected_details = [(i, _action_kind(actions[i]), root.rejection_reasons[i]) for i in range(len(actions)) if root.rejection_reasons[i]]
            _logger.warning(
                "PUCT: %d/%d actions rejected (predict errors=%d). Rejected: %s. "
                "Visits: %s. Priors: %s",
                rejected_count, len(actions), prediction_error_count,
                [(idx, kind, reason) for idx, kind, reason in rejected_details],
                list(root.visit_counts),
                [round(p, 4) for p in root.priors],
            )
        if selected_index is not None and root.visit_counts[selected_index] > 0:
            sel_kind = _action_kind(actions[selected_index])
            sel_prior = root.priors[selected_index]
            # Warn if selected action has low prior but high visits (prior-visit mismatch)
            best_prior_idx = max(range(len(root.priors)), key=lambda i: root.priors[i])
            if best_prior_idx != selected_index and root.priors[best_prior_idx] > 0.5:
                _logger.warning(
                    "PUCT: selected %s (prior=%.3f, visits=%d) but highest-prior action is %s "
                    "(prior=%.3f, visits=%d, q=%.3f). Reason: value backup overrode policy.",
                    sel_kind, sel_prior, root.visit_counts[selected_index],
                    _action_kind(actions[best_prior_idx]),
                    root.priors[best_prior_idx], root.visit_counts[best_prior_idx],
                    root.q_value(best_prior_idx),
                )

        selected_index = None if all(root.rejection_reasons) else _argmax_stable(root.visit_counts)
        policy_target = _normalize(root.visit_counts)

        root_trace = [
            PuctActionTrace(
                action_index=i,
                action_kind=_action_kind(action),
                prior=root.priors[i],
                visit_count=root.visit_counts[i],
                q_value=root.q_value(i),
                u_value=self.config.c_puct * root.priors[i] * sqrt(root.total_visits + 1) / (1 + root.visit_counts[i]),
                rejected=bool(root.rejection_reasons[i]),
                rejection_reason=root.rejection_reasons[i],
            ).__dict__
            for i, action in enumerate(actions)
        ]
        root_trace_map = {
            "root": root_trace,
            "value_sum": list(root.value_sums),
            "mean_value": [root.q_value(i) for i in range(len(actions))],
            "max_depth_reached": max_depth_reached,
            "cross_turn_count": cross_turn_count,
            "last_value_source": getattr(self.value, "source", "terminal"),
            "last_value_terminal": isinstance(self.value, TerminalValue),
            "prediction_error_count": prediction_error_count,
            "prediction_error_details": prediction_error_details,
            "after_state_source": "PredictionResult.snapshot",
            "simulation_trace": simulation_trace,
        }
        diagnostics = {
            "baseline_name": self.baseline_name,
            "baseline_version": self.baseline_version,
            "search_name": "puct_two_player",
            "search_scope": f"multi_turn_{self.config.max_turns}" if self.config.max_turns > 1 else "intra_turn_tree",
            "neural_network_used": self.neural_network_used,
            "legal_action_source": "PlayerLocalActionEnumerator",
            "predictor_source": type(self.predictor).__name__ if self.predictor is not None else "none",
            "after_state_source": "PredictionResult.snapshot",
            "prior_source": getattr(self.prior, "source", "uniform"),
            "value_source": getattr(self.value, "source", "terminal"),
            "policy_target_source": "visit_counts",
            "requested_simulations": self.config.simulations,
            "executed_simulations": executed_simulations,
            "requested_max_depth": self.config.max_depth,
            "effective_max_depth": max_depth_reached,
            "max_turns": self.config.max_turns,
            "cross_turn_count": cross_turn_count,
            "turn_transition_count": self._turn_transition_count,
            "action_predictor_call_count": self._predictor_call_count,
            "turn_transition_predictor_call_count": self._turn_transition_predictor_call_count,
            "predictor_call_count": (
                self._predictor_call_count + self._turn_transition_predictor_call_count
            ),
            "c_puct": self.config.c_puct,
            "discount": self.config.discount,
            "action_count": len(actions),
            "selected_action_index": selected_index,
            "prediction_error_count": prediction_error_count,
            "prediction_error_details": prediction_error_details,
            "partial_prediction_count": partial_prediction_count,
            "partial_prediction_details": partial_prediction_details,
            "greedy_rollout_count": greedy_rollout_count,
            "reservation_eligible": reservation_eligible,
            "reservation_requested": reservation_requested,
            "reservation_completed": reservation_completed,
            "rejected_action_count": sum(1 for r in root.rejection_reasons if r),
            "rejection_reasons": [r for r in root.rejection_reasons if r],
            "non_goals": ["not_runtime_selector", "not_heuristic_predictor", "not_protocol_fallback"],
        }
        root_value = sum(root.value_sums) / sum(root.visit_counts) if sum(root.visit_counts) > 0 else 0.0
        self._last_root = root  # 保留树结构供调试工具读取
        return PuctSearchResult(
            actions=actions,
            visit_counts=tuple(root.visit_counts),
            policy_target=policy_target,
            selected_action_index=selected_index,
            root_value=root_value,
            trace=root_trace_map,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # 可恢复微观回合完成：root 构建 + 增量 UCB 推进（P1）
    # ------------------------------------------------------------------

    def build_completion_root(
        self, snapshot: BattleSnapshot, *, player: str = "me"
    ) -> "_PuctNode":
        """构建一个与 search() 完全一致的 root 决策节点（不跑任何 simulation）。

        与 search() 的 root 构建逐行等价：enumerate → prior → _apply_prior_corrections
        → root Dirichlet noise（同一 branch_seed），从而保证由本方法建立、随后用
        advance_ucb_simulations() 增量推进的树，与一次性 search() 得到的树在语义上
        属于同一分布（相同 branch_seed 下 prior/noise 一致）。

        MicroTurnCompletionSession 用它建立一次持有的 root，后续 advance 只在这棵树上
        追加 simulation，不重置 visit/value/tree。
        """
        actions = tuple(self.enumerator.enumerate(snapshot, player=player))
        if not actions:
            raise ValueError("puct completion session requires at least one legal action")
        # 初始化 P0 credit 计数器（_predict / turn transition 会累加），
        # 使 advance 可在未先调用 search 的情况下安全统计 predictor 调用。
        if not hasattr(self, "_predictor_call_count"):
            self._predictor_call_count = 0
        if not hasattr(self, "_turn_transition_count"):
            self._turn_transition_count = 0
        if not hasattr(self, "_turn_transition_predictor_call_count"):
            self._turn_transition_predictor_call_count = 0
        prior_result = self.prior.priors(snapshot, actions, player=player)
        prior_result.validate(len(actions))
        root_priors = _apply_prior_corrections(actions, prior_result.priors)
        if self.config.dirichlet_epsilon > 0 and len(actions) > 1:
            import numpy as np
            rng = np.random.default_rng(self.config.branch_seed)
            noise = rng.dirichlet([self.config.dirichlet_alpha] * len(actions))
            eps = self.config.dirichlet_epsilon
            root_priors = tuple(
                (1 - eps) * p + eps * n for p, n in zip(root_priors, noise)
            )
        return _PuctNode(
            snapshot, actions, root_priors,
            active_player=player, turn_number=1,
        )

    def advance_ucb_simulations(
        self,
        root: "_PuctNode",
        root_player: str,
        credit: int,
        *,
        executed_before: int = 0,
    ) -> dict[str, Any]:
        """在已展开的 root 树上追加 credit 次常规 UCB simulation，返回诊断增量。

        不重置 visit_counts / value_sums / children：credit 是在既有树上继续积累访问，
        真正复用已展开的子树而不是从头搜索。reservation/deepening 阶段不在此重复执行
        （它们属于一次性 root 预算），advance 只做常规 UCB。

        executed_before：本次 advance 之前已执行的 simulation 总数（供 trace
        simulation_index 与 simulation_trace_limit 连续计数）。
        """
        credit = max(0, int(credit))
        agg = {
            "executed": 0,
            "prediction_error_count": 0,
            "prediction_error_details": [],
            "partial_prediction_count": 0,
            "partial_prediction_details": [],
            "greedy_rollout_count": 0,
            "cross_turn_count": 0,
            "max_depth_reached": 0,
            "simulation_trace": [],
        }
        trace_limit = self.config.simulation_trace_limit
        for i in range(credit):
            collect_trace = (executed_before + i) < trace_limit
            deltas = self._run_ucb_simulation(
                root, root_player,
                collect_trace=collect_trace,
                simulation_index=executed_before + i + 1,
            )
            agg["executed"] += 1
            agg["prediction_error_count"] += deltas.prediction_error_count
            for detail in deltas.prediction_error_details:
                if len(agg["prediction_error_details"]) < 10:
                    agg["prediction_error_details"].append(detail)
            agg["partial_prediction_count"] += deltas.partial_prediction_count
            for pd in deltas.partial_prediction_details:
                if len(agg["partial_prediction_details"]) < 5:
                    agg["partial_prediction_details"].append(pd)
            agg["greedy_rollout_count"] += deltas.greedy_rollout_count
            agg["cross_turn_count"] += deltas.cross_turn_count
            if deltas.depth_reached > agg["max_depth_reached"]:
                agg["max_depth_reached"] = deltas.depth_reached
            if deltas.trace_entry is not None:
                agg["simulation_trace"].append(deltas.trace_entry)
        return agg

    # ------------------------------------------------------------------
    # 单次常规 UCB simulation（search 与 completion session 共用）
    # ------------------------------------------------------------------

    def _run_ucb_simulation(
        self,
        root: "_PuctNode",
        root_player: str,
        *,
        collect_trace: bool = False,
        simulation_index: int = 0,
    ) -> "_UcbSimulationDeltas":
        """在已构建的 root 树上执行一次常规 UCB simulation，并回传诊断增量。

        这是从旧 search() UCB 循环体逐行抽出的单一真源：search() 每轮调用它累加
        增量，MicroTurnCompletionSession.advance() 也调用它在同一棵已展开树上追加
        simulation credit（不重置 visit / value_sum / 树结构）。两条路径因此共享
        完全一致的展开/估值/回传语义与 branch_seed 注入路径。
        """
        deltas = _UcbSimulationDeltas()

        trace_entry: dict[str, Any] | None = None
        if collect_trace:
            root_parent_visits = root.total_visits
            root_scores = _root_pucb_scores(root, self.config.c_puct)
            trace_entry = {
                "simulation_index": simulation_index,
                "phase": "ucb",
                "root_parent_visits_before": root_parent_visits,
                "root_edges_before": root_scores,
            }

        path: list[tuple[_PuctNode, int]] = []
        node = root
        depth = 0
        leaf_value: float | None = None

        while depth < self.config.max_depth:
            # 终局或深度限制叶节点
            if node.is_terminal or node.is_depth_leaf:
                leaf_value = node.terminal_value
                break

            # 选择子边（对手节点选择对 root 最差的动作）
            edge_index = _select_child_for_player(node, self.config.c_puct, root_player, unvisited_first=self.config.unvisited_first)
            if trace_entry is not None and not path:
                trace_entry["selected_root_action_index"] = edge_index
                trace_entry["selected_root_action"] = _action_identity(node.actions[edge_index])
            path.append((node, edge_index))

            child = node.children[edge_index]
            if child is not None:
                # 已展开的子节点，继续往下走
                node = child
                depth += 1
                continue

            # 未展开：predict 并展开
            action = node.actions[edge_index]
            is_turn_end = _is_turn_end_action(action)

            # 回合内 1-based 动作序列深度：该边是当前回合从回合根起的第
            # (node.turn_action_depth + 1) 个动作。据此派生 branch_seed，与执行侧对齐。
            edge_action_index = node.turn_action_depth + 1
            prediction = self._predict(
                node.snapshot, action, player=node.active_player, action_index=edge_action_index
            )

            if prediction is None or isinstance(prediction, _PredictionError):
                # predict 失败（PREDICT_ERROR 或 predictor 不可用）
                deltas.prediction_error_count += 1
                if isinstance(prediction, _PredictionError) and len(deltas.prediction_error_details) < 10:
                    deltas.prediction_error_details.append({
                        "action_kind": action.kind.value,
                        "source_uid": action.source_uid,
                        "player": prediction.player,
                        "unresolved": list(prediction.unresolved_effects[:3]),
                        "description": prediction.description[:120],
                    })
                if self.config.reject_predict_error:
                    node.rejection_reasons[edge_index] = "predict_error"
                    path.pop()
                    leaf_value = None
                    break
                leaf_value = 0.0
                break

            # PARTIAL prediction: 有可用状态但含未解析效果
            # 尝试继续展开：如果 after-state 有合法动作，不截断搜索
            if isinstance(prediction, _PartialPrediction):
                deltas.partial_prediction_count += 1
                if len(deltas.partial_prediction_details) < 5:
                    deltas.partial_prediction_details.append(
                        f"{_action_kind(action)}(src={action.source_uid}):"
                        f"{','.join(prediction.unresolved_effects)}"
                    )
                child_snapshot = prediction.snapshot

                # 尝试枚举 after-state 的合法动作
                try:
                    child_actions = tuple(self.child_enumerator.enumerate(
                        child_snapshot, player=node.active_player
                    ))
                except Exception:
                    child_actions = ()

                if child_actions:
                    # 有合法动作 → 当作正常节点展开（best-effort）
                    child_prior = self.prior.priors(child_snapshot, child_actions, player=node.active_player)
                    child_prior.validate(len(child_actions))
                    child_node = _PuctNode(
                        child_snapshot, child_actions, _apply_prior_corrections(child_actions, child_prior.priors),
                        active_player=node.active_player,
                        turn_number=node.turn_number,
                        turn_action_depth=node.turn_action_depth + 1,
                    )
                    node.children[edge_index] = child_node
                    node = child_node
                    depth += 1
                    continue  # 继续往下展开
                else:
                    # 无合法动作 → 回退为 depth_leaf
                    value_result = self.value.evaluate(child_snapshot, player=node.active_player)
                    leaf_v = self._to_root_value(value_result.value, node.active_player, root_player)
                    child_node = _PuctNode(
                        child_snapshot, (), (),
                        active_player=node.active_player,
                        turn_number=node.turn_number,
                        is_depth_leaf=True, terminal_value=leaf_v,
                    )
                    node.children[edge_index] = child_node
                    depth += 1
                    leaf_value = leaf_v
                    break

            child_snapshot = prediction.snapshot
            depth += 1

            # 判断子节点是否为终局
            child_terminal_value = self._terminal_value(child_snapshot, root_player)
            if child_terminal_value is not None:
                child_node = _PuctNode(
                    child_snapshot, (), (),
                    active_player=node.active_player,
                    turn_number=node.turn_number,
                    is_terminal=True, terminal_value=child_terminal_value,
                )
                node.children[edge_index] = child_node
                leaf_value = child_terminal_value
                break

            # turn_end 动作：回合切换逻辑
            if is_turn_end or getattr(child_snapshot, "is_turn_ended", False):
                child_node = self._handle_turn_end(
                    child_snapshot, node, root_player, depth,
                )
                node.children[edge_index] = child_node
                if child_node.is_depth_leaf or child_node.is_terminal:
                    leaf_value = child_node.terminal_value
                else:
                    # 跨回合节点已展开，继续往下 traverse
                    deltas.cross_turn_count += 1
                    node = child_node
                    continue
                break

            # 到达 max_depth：greedy rollout 到 turn_end
            if depth >= self.config.max_depth:
                # child_snapshot 是执行当前边后的 after-state，其回合内深度为
                # node.turn_action_depth + 1；rollout 从它继续，首步为第
                # node.turn_action_depth + 2 个回合内动作。
                leaf_v = self._greedy_rollout_to_turn_end(
                    child_snapshot, node.active_player, root_player,
                    start_action_index=node.turn_action_depth + 2,
                )
                deltas.greedy_rollout_count += 1
                child_actions = tuple(self.child_enumerator.enumerate(child_snapshot, player=node.active_player))
                if not child_actions:
                    child_node = _PuctNode(
                        child_snapshot, (), (),
                        active_player=node.active_player,
                        turn_number=node.turn_number,
                        is_terminal=True, terminal_value=leaf_v,
                    )
                else:
                    child_prior = self.prior.priors(child_snapshot, child_actions, player=node.active_player)
                    child_prior.validate(len(child_actions))
                    child_node = _PuctNode(
                        child_snapshot, child_actions, _apply_prior_corrections(child_actions, child_prior.priors),
                        active_player=node.active_player,
                        turn_number=node.turn_number,
                        is_depth_leaf=True, terminal_value=leaf_v,
                    )
                node.children[edge_index] = child_node
                leaf_value = leaf_v
                break

            # 正常展开中间节点
            child_actions = tuple(self.child_enumerator.enumerate(child_snapshot, player=node.active_player))
            if not child_actions:
                value_result = self.value.evaluate(child_snapshot, player=node.active_player)
                leaf_v = self._to_root_value(value_result.value, node.active_player, root_player)
                child_node = _PuctNode(
                    child_snapshot, (), (),
                    active_player=node.active_player,
                    turn_number=node.turn_number,
                    is_terminal=True, terminal_value=leaf_v,
                )
                node.children[edge_index] = child_node
                leaf_value = leaf_v
                break

            child_prior = self.prior.priors(child_snapshot, child_actions, player=node.active_player)
            child_prior.validate(len(child_actions))
            child_node = _PuctNode(
                child_snapshot, child_actions, _apply_prior_corrections(child_actions, child_prior.priors),
                active_player=node.active_player,
                turn_number=node.turn_number,
                turn_action_depth=node.turn_action_depth + 1,
            )
            node.children[edge_index] = child_node
            node = child_node
            # 继续循环往更深走

        # rejected edge: 没有 path 需要 backprop
        if leaf_value is None:
            if not path:
                # 本次 simulation 仍视为已执行（与旧 search 语义一致：executed += 1）
                deltas.executed = True
                deltas.depth_reached = depth
                return deltas
            value_result = self.value.evaluate(node.snapshot, player=node.active_player)
            leaf_value = self._to_root_value(value_result.value, node.active_player, root_player)

        deltas.depth_reached = depth

        # 反向传播（value 始终从 root_player 视角存储）
        for (path_node, path_index) in reversed(path):
            path_node.visit_counts[path_index] += 1
            path_node.value_sums[path_index] += leaf_value
            path_node.total_visits += 1

        if trace_entry is not None:
            root_index = trace_entry.get("selected_root_action_index")
            trace_entry.update({
                "depth_reached": depth,
                "leaf_value_from_root": leaf_value,
                "root_visits_after": root.total_visits,
                "root_edge_after": (
                    _root_pucb_scores(root, self.config.c_puct)[root_index]
                    if root_index is not None else None
                ),
            })
            deltas.trace_entry = trace_entry

        deltas.executed = True
        return deltas

    # ------------------------------------------------------------------
    # 强制深度优先展开（forced greedy-prior deepening）
    # ------------------------------------------------------------------

    def _run_forced_deepening_simulation(
        self,
        root: "_PuctNode",
        forced_edge: int,
        root_player: str,
    ) -> "_ForcedDeepeningOutcome":
        """从 root 强制选择 forced_edge，之后沿最高 prior 子边深度优先展开一次。

        与旧 high-prior deepening 使用完全相同的展开/估值/回传语义，因此
        reservation 与 high-prior deepening 共享同一条执行路径，避免语义分叉。

        返回 _ForcedDeepeningOutcome：executed 表示本次是否真正跑了一次 simulation
        （被 reject 的 edge 不算），prediction_error_count / prediction_error_details
        为本次产生的 predict 错误增量，由调用方累加到全局诊断。
        """
        prediction_error_count = 0
        prediction_error_details: list[dict[str, Any]] = []

        if root.rejection_reasons[forced_edge]:
            return _ForcedDeepeningOutcome(False, prediction_error_count, prediction_error_details)

        path: list[tuple[_PuctNode, int]] = [(root, forced_edge)]
        node = root
        depth = 0
        leaf_value: float | None = None
        forced_first = forced_edge  # 第一层强制

        while depth < self.config.max_depth:
            edge_index = forced_first if depth == 0 else _select_greedy_prior(node)
            if edge_index is None:
                break
            if depth > 0:
                path.append((node, edge_index))

            child = node.children[edge_index]
            if child is not None:
                node = child
                depth += 1
                forced_first = None
                continue

            # 展开
            action = node.actions[edge_index]
            is_turn_end = _is_turn_end_action(action)
            # 与 UCB 路径一致：该边是当前回合从回合根起的第 (node.turn_action_depth + 1)
            # 个动作，据此派生与执行侧一致的 branch_seed。
            edge_action_index = node.turn_action_depth + 1
            prediction = self._predict(
                node.snapshot, action, player=node.active_player, action_index=edge_action_index
            )

            if prediction is None or isinstance(prediction, _PredictionError):
                prediction_error_count += 1
                if isinstance(prediction, _PredictionError) and len(prediction_error_details) < 10:
                    prediction_error_details.append({
                        "action_kind": action.kind.value,
                        "source_uid": action.source_uid,
                        "player": prediction.player,
                        "unresolved": list(prediction.unresolved_effects[:3]),
                        "description": prediction.description[:120],
                    })
                if self.config.reject_predict_error:
                    node.rejection_reasons[edge_index] = "predict_error"
                    path.pop()
                    leaf_value = None
                    break
                leaf_value = 0.0
                break

            if isinstance(prediction, _PartialPrediction):
                child_snapshot = prediction.snapshot
                try:
                    child_actions = tuple(self.child_enumerator.enumerate(child_snapshot, player=node.active_player))
                except Exception:
                    child_actions = ()
                if child_actions:
                    child_prior = self.prior.priors(child_snapshot, child_actions, player=node.active_player)
                    child_prior.validate(len(child_actions))
                    child_node = _PuctNode(child_snapshot, child_actions, _apply_prior_corrections(child_actions, child_prior.priors), active_player=node.active_player, turn_number=node.turn_number, turn_action_depth=node.turn_action_depth + 1)
                    node.children[edge_index] = child_node
                    node = child_node
                    depth += 1
                    continue
                else:
                    value_result = self.value.evaluate(child_snapshot, player=node.active_player)
                    leaf_value = self._to_root_value(value_result.value, node.active_player, root_player)
                    child_node = _PuctNode(child_snapshot, (), (), active_player=node.active_player, turn_number=node.turn_number, is_depth_leaf=True, terminal_value=leaf_value)
                    node.children[edge_index] = child_node
                    break

            child_snapshot = prediction.snapshot
            depth += 1

            child_terminal = self._terminal_value(child_snapshot, root_player)
            if child_terminal is not None:
                child_node = _PuctNode(child_snapshot, (), (), active_player=node.active_player, turn_number=node.turn_number, is_terminal=True, terminal_value=child_terminal)
                node.children[edge_index] = child_node
                leaf_value = child_terminal
                break

            if is_turn_end or getattr(child_snapshot, "is_turn_ended", False):
                child_node = self._handle_turn_end(child_snapshot, node, root_player, depth)
                node.children[edge_index] = child_node
                if child_node.is_depth_leaf or child_node.is_terminal:
                    leaf_value = child_node.terminal_value
                else:
                    node = child_node
                    continue
                break

            child_actions = tuple(self.child_enumerator.enumerate(child_snapshot, player=node.active_player))
            if not child_actions:
                value_result = self.value.evaluate(child_snapshot, player=node.active_player)
                leaf_v = self._to_root_value(value_result.value, node.active_player, root_player)
                child_node = _PuctNode(child_snapshot, (), (), active_player=node.active_player, turn_number=node.turn_number, is_terminal=True, terminal_value=leaf_v)
                node.children[edge_index] = child_node
                leaf_value = leaf_v
                break

            child_prior = self.prior.priors(child_snapshot, child_actions, player=node.active_player)
            child_prior.validate(len(child_actions))
            child_node = _PuctNode(child_snapshot, child_actions, _apply_prior_corrections(child_actions, child_prior.priors), active_player=node.active_player, turn_number=node.turn_number, turn_action_depth=node.turn_action_depth + 1)
            node.children[edge_index] = child_node
            node = child_node
            continue

        # Leaf evaluation
        if leaf_value is None:
            # rejected edge：无 path 可回传，视为未执行
            if not path:
                return _ForcedDeepeningOutcome(False, prediction_error_count, prediction_error_details)
            value_result = self.value.evaluate(node.snapshot, player=node.active_player)
            leaf_value = self._to_root_value(value_result.value, node.active_player, root_player)

        # Backprop
        for (path_node, path_index) in reversed(path):
            path_node.visit_counts[path_index] += 1
            path_node.value_sums[path_index] += leaf_value
            path_node.total_visits += 1

        return _ForcedDeepeningOutcome(True, prediction_error_count, prediction_error_details)

    # ------------------------------------------------------------------
    # 回合切换处理
    # ------------------------------------------------------------------

    def _handle_turn_end(
        self,
        turn_end_snapshot: BattleSnapshot,
        parent_node: _PuctNode,
        root_player: str,
        current_depth: int,
    ) -> _PuctNode:
        """处理 turn_end 后的回合切换。

        如果 turn_number 已到达 max_turns，则作为叶节点估值。
        否则执行对方 start_turn 并枚举对方动作，创建新的决策节点。
        """
        current_turn = parent_node.turn_number
        next_turn = current_turn + 1
        ending_player = parent_node.active_player
        next_player = "opponent" if ending_player == "me" else "me"

        # max_turns=1 退化为旧行为：turn_end 是叶节点
        if next_turn > self.config.max_turns:
            value_result = self.value.evaluate(turn_end_snapshot, player=ending_player)
            leaf_v = self._to_root_value(value_result.value, ending_player, root_player)
            # 未使用行动值的惩罚：只惩罚「主动放弃」（父节点还有非 turn_end 选项）。
            # 注意读的是 parent_node.snapshot——turn_end_snapshot 里 PP 已被回合结束重置。
            penalty = float(self.config.unused_pp_penalty)
            if penalty > 0.0:
                ending_view = (
                    parent_node.snapshot.me
                    if ending_player == "me"
                    else parent_node.snapshot.opponent
                )
                has_alternative = any(
                    action.kind is not ActionKind.TURN_END for action in parent_node.actions
                )
                if has_alternative:
                    leaf_v -= penalty * int(ending_view.pp)
            return _PuctNode(
                turn_end_snapshot, (), (),
                active_player=ending_player,
                turn_number=current_turn,
                is_depth_leaf=True, terminal_value=leaf_v,
            )

        # 执行回合切换：对方 start_turn
        if self.turn_transitioner is not None:
            self._turn_transition_count += 1
            before_transition_predictor_calls = getattr(
                self.turn_transitioner, "predictor_call_count", None
            )
            next_snapshot = self.turn_transitioner.transition_turn(turn_end_snapshot, ending_player)
            after_transition_predictor_calls = getattr(
                self.turn_transitioner, "predictor_call_count", None
            )
            if isinstance(before_transition_predictor_calls, int) and isinstance(
                after_transition_predictor_calls, int
            ):
                self._turn_transition_predictor_call_count += max(
                    0, after_transition_predictor_calls - before_transition_predictor_calls
                )
        else:
            # 无 transitioner 时退化为叶节点
            value_result = self.value.evaluate(turn_end_snapshot, player=ending_player)
            leaf_v = self._to_root_value(value_result.value, ending_player, root_player)
            return _PuctNode(
                turn_end_snapshot, (), (),
                active_player=ending_player,
                turn_number=current_turn,
                is_depth_leaf=True, terminal_value=leaf_v,
            )

        # 检查回合切换后是否为终局
        terminal_v = self._terminal_value(next_snapshot, root_player)
        if terminal_v is not None:
            return _PuctNode(
                next_snapshot, (), (),
                active_player=next_player,
                turn_number=next_turn,
                is_terminal=True, terminal_value=terminal_v,
            )

        # 枚举下一位玩家的合法动作
        next_actions = tuple(self.child_enumerator.enumerate(next_snapshot, player=next_player))
        if not next_actions:
            # 无合法动作（不应该发生，防御）
            value_result = self.value.evaluate(next_snapshot, player=next_player)
            leaf_v = self._to_root_value(value_result.value, next_player, root_player)
            return _PuctNode(
                next_snapshot, (), (),
                active_player=next_player,
                turn_number=next_turn,
                is_terminal=True, terminal_value=leaf_v,
            )

        # 创建对方决策节点
        next_prior = self.prior.priors(next_snapshot, next_actions, player=next_player)
        next_prior.validate(len(next_actions))
        return _PuctNode(
            next_snapshot, next_actions, _apply_prior_corrections(next_actions, next_prior.priors),
            active_player=next_player,
            turn_number=next_turn,
        )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _predict(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        *,
        player: str,
        action_index: int | None = None,
    ) -> Any | None:
        """预测单个动作的 after-state。

        action_index（私有参数，1-based）：该动作在**当前回合内**从回合根起算的动作
        序列深度。搜索树按此深度派生 branch_seed，使之与 strict self-play /
        search_ab_match 的执行侧 ``base + action_index`` 完全一致（见
        _predict_with_seed）。None 表示普通单步调用（无树上下文），退化为旧单步
        语义，保持 PuctSearch 对外 API 兼容。
        """
        if self.predictor is None:
            return None

        # LocalPredictor.predict 不接受 player 参数，通过视角交换处理
        if player == "opponent":
            swapped = opponent_as_me(snapshot)
            prediction = self._predict_with_seed(swapped, action, player="me", action_index=action_index)
            if prediction is None:
                return None
            certainty = getattr(prediction, "certainty", None)
            unresolved_effects = tuple(getattr(prediction, "unresolved_effects", ()) or ())
            if certainty is PredictionCertainty.PREDICT_ERROR:
                return _PredictionError(
                    action=action,
                    unresolved_effects=unresolved_effects,
                    description=getattr(prediction, "description", "") or "",
                    player=player,
                )
            if certainty in {PredictionCertainty.UNKNOWN, PredictionCertainty.PARTIAL} and unresolved_effects:
                # 尝试 summon 物化重试
                retried = self._retry_summon_materialization(
                    swapped, action, prediction, player="me", action_index=action_index
                )
                if retried is not None:
                    restored_snapshot = restore_opponent_perspective(retried.snapshot, snapshot)
                    return dc_replace(retried, snapshot=restored_snapshot)
                restored_snapshot = restore_opponent_perspective(prediction.snapshot, snapshot)
                return _PartialPrediction(dc_replace(prediction, snapshot=restored_snapshot), unresolved_effects)
            # 恢复视角
            restored_snapshot = restore_opponent_perspective(prediction.snapshot, snapshot)
            return dc_replace(prediction, snapshot=restored_snapshot)
        else:
            prediction = self._predict_with_seed(snapshot, action, player=player, action_index=action_index)
            certainty = getattr(prediction, "certainty", None)
            unresolved_effects = tuple(getattr(prediction, "unresolved_effects", ()) or ())
            if certainty is PredictionCertainty.PREDICT_ERROR:
                return _PredictionError(
                    action=action,
                    unresolved_effects=unresolved_effects,
                    description=getattr(prediction, "description", "") or "",
                    player=player,
                )
            if certainty in {PredictionCertainty.UNKNOWN, PredictionCertainty.PARTIAL} and unresolved_effects:
                # 尝试 summon 物化重试
                retried = self._retry_summon_materialization(
                    snapshot, action, prediction, player="me", action_index=action_index
                )
                if retried is not None:
                    return retried
                return _PartialPrediction(prediction, unresolved_effects)
            return prediction

    def _retry_summon_materialization(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        prediction,
        *,
        player: str,
        action_index: int | None = None,
    ):
        """尝试物化 summon 效果并重试 predict。

        当 prediction 返回 PARTIAL 且含 generated_cards_not_materializable:summon 时，
        从 deck_instances 中随机选取符合条件的卡牌作为 generated_cards 重试。
        成功时返回新的 prediction，否则返回 None。
        """
        unresolved = tuple(getattr(prediction, "unresolved_effects", ()) or ())
        if not any("generated_cards_not_materializable:summon" in u for u in unresolved):
            return None

        # 需要 deck_instances 才能物化
        view = snapshot.me  # 此时 snapshot 已经是 me 视角
        if view.deck_instances is None:
            return None

        # 从模板找到 summon 操作的 source_pool 条件
        from decision_engine.execution.generated_card_materializer import source_pool_cards
        card = snapshot.cards_by_uid.get(action.source_uid)
        if card is None:
            return None

        templates = getattr(self, "_templates", None)
        if templates is None:
            # 尝试从 predictor 获取
            inner = getattr(self.predictor, "predictor", self.predictor)
            templates = getattr(inner, "templates_by_base_card_id", None) or getattr(inner, "_templates", None)
            if templates is None:
                return None
            self._templates = templates

        template = templates.get(card.base_card_id)
        if template is None:
            return None

        # 找 fanfare/manual_super_evolve 中带 source_pool 的 summon 操作
        import random as _random
        trigger_events = ("fanfare", "manual_super_evolve", "manual_evolve")
        generated_cards = []
        # 统一 seed 派生：搜索树上下文（action_index 非 None）用 base + action_index，
        # 与 strict self-play / search_ab_match 执行侧完全一致；None 时退化为旧单步语义。
        seed = self._branch_seed_for(action, action_index)

        for effect in template.executable_effects:
            if effect.trigger.event not in trigger_events:
                continue
            for op_idx, operation in enumerate(effect.operations):
                if operation.op != "summon":
                    continue
                sp = operation.params.get("source_pool")
                if sp is None:
                    continue
                # 从 snapshot 中筛选符合 source_pool 条件的卡牌
                candidates = source_pool_cards(snapshot, sp)
                if not candidates:
                    continue
                count = sp.get("count", 1)
                rng = _random.Random(seed + op_idx)
                selected = rng.sample(candidates, min(count, len(candidates)))
                generated_cards.extend(selected)

        if not generated_cards:
            return None

        # 用 generated_cards 重试 predict
        method = getattr(self.predictor, "predict_with_replay_context", None)
        if method is None:
            return None

        ctx = LocalExecutionContext(branch_seed=seed)
        self._predictor_call_count += 1
        retried = method(snapshot, action, player=player, generated_cards=tuple(generated_cards), execution_context=ctx)
        if retried is None:
            return None

        retry_certainty = getattr(retried, "certainty", None)
        retry_unresolved = tuple(getattr(retried, "unresolved_effects", ()) or ())
        # 如果重试后仍然有 summon 未解析，放弃
        if any("generated_cards_not_materializable:summon" in u for u in retry_unresolved):
            return None

        return retried

    def _branch_seed_for(self, action: LegalAction, action_index: int | None) -> int:
        """派生单次预测的 branch_seed。

        这是 PUCT 搜索与 strict self-play / search_ab_match 执行侧共享的 seed 契约核心。

        - 搜索树上下文（action_index 为 1-based 回合内动作序列深度）：
          seed = config.branch_seed + action_index。
          调用方把 config.branch_seed 视作「当前回合 base」（strict/search_ab 用
          ``seed*100000 + turn*100`` 作为 base，并以 ``base + action_index`` 执行第
          action_index 个动作），因此搜索树内第 action_index 层动作与执行侧用同一 seed，
          随机效果 / 生成物物化结果一致，Macro TurnLine 的 end_snapshot 不再偏离。

        - 普通单步调用（action_index 为 None，无树上下文）：
          退化为旧单步语义 config.branch_seed + hash(source_uid) % 10000，
          保证 PuctSearch 对外 API 行为向后兼容。
        """
        base = self.config.branch_seed or 0
        if action_index is not None:
            return base + int(action_index)
        return base + hash(action.source_uid) % 10000

    def _predict_with_seed(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        *,
        player: str,
        action_index: int | None = None,
    ):
        """有 branch_seed 时注入 execution_context 以消除随机效果 PARTIAL。

        action_index（1-based 回合内动作序列深度）用于派生与执行侧一致的 branch_seed，
        见 _branch_seed_for。
        """
        self._predictor_call_count += 1
        if self.config.branch_seed is None:
            return self.predictor.predict(snapshot, action, player=player)
        seed = self._branch_seed_for(action, action_index)
        method = getattr(self.predictor, "predict_with_replay_context", None)
        if method is not None:
            ctx = LocalExecutionContext(branch_seed=seed)
            return method(snapshot, action, player=player, execution_context=ctx)
        return self.predictor.predict(snapshot, action, player=player)

    def _terminal_value(self, snapshot: BattleSnapshot, root_player: str) -> float | None:
        """如果 snapshot 是终局状态，返回从 root_player 视角的 value。"""
        value_result = self.value.evaluate(snapshot, player=root_player)
        if value_result.terminal:
            return value_result.value
        return None

    def _to_root_value(self, value: float, evaluating_player: str, root_player: str) -> float:
        """将从 evaluating_player 视角的 value 转换为 root_player 视角。"""
        if evaluating_player == root_player:
            return value
        return -value

    # Greedy rollout 最大步数限制（防止无限循环）
    _GREEDY_ROLLOUT_MAX_STEPS = 20

    def _greedy_rollout_to_turn_end(
        self,
        snapshot: BattleSnapshot,
        active_player: str,
        root_player: str,
        *,
        start_action_index: int | None = None,
    ) -> float:
        """从 max_depth 处 greedy rollout 到 turn_end，返回 root_player 视角的 value。

        当搜索到达 max_depth 但当前回合尚未结束时调用。
        按 prior 最高的动作一直走到 turn_end，然后用 value head 评估。
        这保证了 Q 值反映的是"回合结束时的完整局面评估"，
        而不是被 combo 中间状态的不准确估值污染。

        如果 rollout 超过 _GREEDY_ROLLOUT_MAX_STEPS 步仍未到 turn_end，
        则用当前 snapshot 的 value 评估（fallback）。

        start_action_index（1-based 回合内动作序列深度）：rollout 首步在当前回合内的
        action_index。每走一步递增，使 rollout 中每个动作预测用与执行侧一致的
        branch_seed（base + action_index）。None 时退化为旧单步 seed 语义（兼容）。
        """
        current = snapshot
        # rollout 内逐步递增的回合内 action_index（None 表示无树上下文，走旧单步 seed）。
        action_index = start_action_index
        for step in range(self._GREEDY_ROLLOUT_MAX_STEPS):
            # 枚举合法动作
            try:
                actions = tuple(self.child_enumerator.enumerate(current, player=active_player))
            except Exception:
                break
            if not actions:
                break

            # 选 turn_end（如果有的话直接结束）
            turn_end_action = None
            best_prior_action = None
            best_prior = -1.0
            for a in actions:
                if _is_turn_end_action(a):
                    turn_end_action = a
                    break
                # 不走 prior，直接找 turn_end 以外 prior 最高的
                # 但这里我们不调 prior（开销大），用简单规则：优先攻击 > 进化 > play
                # 实际上还是用 prior 更准确
            
            if turn_end_action is not None:
                # 到达 turn_end，执行并评估。turn_end 本身也占一个回合内 action_index，
                # 与执行侧 TURN_END 的 branch_seed 对齐。
                prediction = self._predict(
                    current, turn_end_action, player=active_player, action_index=action_index
                )
                if prediction is not None and not isinstance(prediction, _PredictionError):
                    pred_snap = prediction.snapshot if hasattr(prediction, 'snapshot') else None
                    if pred_snap is not None:
                        value_result = self.value.evaluate(pred_snap, player=active_player)
                        return self._to_root_value(value_result.value, active_player, root_player)
                # turn_end predict 失败，用当前状态评估
                break

            # 没有 turn_end（不应该发生，但防御），选 prior 最高的非 turn_end 动作
            try:
                prior_result = self.prior.priors(current, actions, player=active_player)
                best_idx = max(range(len(actions)), key=lambda i: prior_result.priors[i])
            except Exception:
                best_idx = 0

            action = actions[best_idx]
            prediction = self._predict(current, action, player=active_player, action_index=action_index)
            if prediction is None or isinstance(prediction, _PredictionError):
                break
            pred_snap = prediction.snapshot if hasattr(prediction, 'snapshot') else None
            if pred_snap is None:
                break
            current = pred_snap
            # 推进到回合内下一个 action_index（None 时保持 None）。
            if action_index is not None:
                action_index += 1

        # Fallback: rollout 未到达 turn_end，用最后状态评估
        _logger.debug(
            "greedy_rollout_to_turn_end: reached max steps (%d) without turn_end, using fallback value",
            self._GREEDY_ROLLOUT_MAX_STEPS,
        )
        value_result = self.value.evaluate(current, player=active_player)
        return self._to_root_value(value_result.value, active_player, root_player)

    # ------------------------------------------------------------------
    # select_child 支持对手视角
    # ------------------------------------------------------------------


def _select_greedy_prior(node: _PuctNode) -> int | None:
    """选择 prior 最高的未被 reject 的子边（用于 deepening 阶段的深度优先展开）。"""
    if not node.actions:
        return None
    best_index = None
    best_prior = -1.0
    for i in range(len(node.actions)):
        if node.rejection_reasons[i]:
            continue
        # 跳过 turn_end（deepening 目的是展开回合内 combo）
        if _is_turn_end_action(node.actions[i]):
            continue
        if node.priors[i] > best_prior:
            best_prior = node.priors[i]
            best_index = i
    # 如果所有非 turn_end 动作都被 reject 了，选 turn_end
    if best_index is None:
        for i in range(len(node.actions)):
            if not node.rejection_reasons[i]:
                return i
    return best_index


def _root_pucb_scores(node: _PuctNode, c_puct: float) -> list[dict[str, Any]]:
    """返回根节点选择前的 PUCB 构成；仅供诊断，不参与选择。"""
    parent_visits = node.total_visits + 1
    scores: list[dict[str, Any]] = []
    for index, action in enumerate(node.actions):
        visits = node.visit_counts[index]
        q_value = node.q_value(index)
        prior = node.priors[index]
        u_value = c_puct * prior * sqrt(parent_visits) / (1 + visits)
        scores.append({
            "action_index": index,
            "action": _action_identity(action),
            "prior": prior,
            "visits": visits,
            "q_value": q_value,
            "u_value": u_value,
            "pucb_value": q_value + u_value,
            "rejected": bool(node.rejection_reasons[index]),
        })
    return scores


def _action_identity(action: LegalAction) -> dict[str, Any]:
    """保留区分语义变体所需的完整动作标识，供 trace 审计。"""
    return {
        "kind": _action_kind(action),
        "source_uid": action.source_uid,
        "cost": action.cost,
        "play_kind": action.play_kind.value if action.play_kind is not None else None,
        "target_groups": [list(group) for group in action.target_groups],
        "random_target_groups": [list(group) for group in action.random_target_groups],
        "selected_indexes": list(action.selected_indexes),
        "material_uids": list(action.material_uids),
        "variant": action.variant,
    }


def _select_child_for_player(node: _PuctNode, c_puct: float, root_player: str, *, unvisited_first: bool = False) -> int:
    """选择 UCB 最高的子边。对手节点取 -q（选择对 root 最差的动作）。

    ``unvisited_first``：开启时只要还有未被评估过的合法子边，就先按 prior 降序各访问一次。
    未访问边的 ``q_value`` 是 0（中性），而其他动作的 q 已经是 0.5+，低 prior 的
    ``turn_end`` 由此永远排不进去。默认关闭以保持既有搜索语义；需要消除这类探索盲区时
    由 ``PuctConfig.unvisited_first`` 打开。
    """
    if unvisited_first:
        unvisited = [
            i for i in range(len(node.actions))
            if not node.rejection_reasons[i] and node.visit_counts[i] == 0
        ]
        if unvisited:
            return max(unvisited, key=lambda i: (node.priors[i], -i))
    parent_visits = node.total_visits + 1
    # 对手节点：对手想选让 root_player 最差的动作 = -q 最大
    negate = (node.active_player != root_player)
    best_index = 0
    best_score = float("-inf")
    for i in range(len(node.actions)):
        if node.rejection_reasons[i]:
            continue
        q = node.q_value(i)
        if negate:
            q = -q
        u = c_puct * node.priors[i] * sqrt(parent_visits) / (1 + node.visit_counts[i])
        score = q + u
        if score > best_score:
            best_score = score
            best_index = i
    return best_index


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

# Prior 修正：turn_end 的最大 prior 上限（当存在攻击/进化类动作时）。
# 默认 0.0 = **关闭**。原实现固定压到 0.05，实测会让 policy 即使学会给 turn_end
# 更高 prior，也被压回 0.039 并拿到 0 visits（见 docs/plan/SEARCH_MYOPIA_DIAGNOSIS.md
# 「prior cap 的独立影响」）。保留环境变量以便需要旧行为时显式打开。
_TURN_END_PRIOR_CAP_DEFAULT = 0.0


def _turn_end_prior_cap() -> float:
    import os as _os

    raw = _os.environ.get("EP_TURN_END_PRIOR_CAP")
    if raw is None or raw == "":
        return _TURN_END_PRIOR_CAP_DEFAULT
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"EP_TURN_END_PRIOR_CAP 必须是数字，收到 {raw!r}") from exc


def _apply_prior_corrections(
    actions: tuple[LegalAction, ...],
    priors: tuple[float, ...],
) -> tuple[float, ...]:
    """规则化 prior 修正：当存在主动动作时压低 turn_end 的 prior。

    规则：
    - 当存在 attack_follower / attack_leader / super_evolve / evolve 动作时，
      turn_end 的 prior 压到上限（默认 0.0 = 不压制）
    - 压掉的概率按比例重分配给其他动作
    - 上限由 ``EP_TURN_END_PRIOR_CAP`` 控制；<= 0 时整段跳过

    历史说明：这层修正原本是「防低级送回合」的保护，但它同时封死了「该停手」的
    唯一入口——policy 学会的 prior 会被再压一道。现在默认关闭，停手与否交给
    unvisited-first 探索 + value 判断。
    """
    cap = _turn_end_prior_cap()
    if cap <= 0.0:
        return priors
    if len(actions) <= 1:
        return priors

    # 检查是否存在主动战斗/进化动作
    has_proactive_action = False
    turn_end_index = -1
    for i, a in enumerate(actions):
        kind = a.kind.value if hasattr(a.kind, "value") else str(a.kind)
        if kind in ("attack_follower", "attack_leader", "super_evolve", "evolve"):
            has_proactive_action = True
        if kind == "turn_end":
            turn_end_index = i

    if not has_proactive_action or turn_end_index < 0:
        return priors

    # turn_end 的 prior 已经足够低，不需要修正
    te_prior = priors[turn_end_index]
    if te_prior <= cap:
        return priors

    # 压低 turn_end，重分配给其他动作
    excess = te_prior - cap
    other_total = 1.0 - te_prior
    corrected = list(priors)
    corrected[turn_end_index] = cap

    if other_total > 1e-8:
        # 按比例重分配
        scale = (other_total + excess) / other_total
        for i in range(len(corrected)):
            if i != turn_end_index:
                corrected[i] *= scale
    else:
        # 极端情况：其他动作 prior 全是 0，均分
        n_other = len(actions) - 1
        share = excess / n_other
        for i in range(len(corrected)):
            if i != turn_end_index:
                corrected[i] += share

    return tuple(corrected)


def _is_turn_end_action(action: LegalAction) -> bool:
    kind = action.kind.value if hasattr(action.kind, "value") else str(action.kind)
    return kind == "turn_end"


def _normalize(values: Sequence[int]) -> tuple[float, ...]:
    total = float(sum(values))
    if total <= 0.0:
        count = len(values)
        if count <= 0:
            return ()
        return tuple(1.0 / count for _ in values)
    return tuple(value / total for value in values)


def _argmax_stable(values: Sequence[int]) -> int | None:
    if not values:
        return None
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def _action_kind(action: LegalAction) -> str:
    return action.kind.value if hasattr(action.kind, "value") else str(action.kind)
