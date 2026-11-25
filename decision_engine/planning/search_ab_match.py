# -*- coding: utf-8 -*-
"""Strict paired *search / selector* A/B self-play harness.

与 ``strict_self_play`` 的区别：``strict_self_play`` 比较的是同一搜索下的两种
**value 臂**（terminal vs diff），产出训练样本。本模块比较的是两种 **搜索 /
selector 臂** 直接对局的胜负——candidate arm 与 opponent arm 在同一局里各控一方，
用镜像牌组 + 固定 seed 跑 candidate 先手 / 后手两局，只把双方全程 strict
(EXACT / SPEC_ONLY、无 unresolved) 且 terminal 的成对结果纳入统计。

复用契约（不修改 macro_turn_search.py / puct.py / strict_self_play.py）：

- strict 确定性契约：每一个被执行动作的 prediction 必须 certainty ∈
  {exact, spec_only} 且 unresolved_effects 为空，否则整局拒绝。
- ``LocalMatchSetup`` fixed seed：镜像局用同一 ``seed`` + 同一牌组构造 setup，
  两局只交换 first_player。
- ``PlayerLocalPredictor`` / ``LocalPredictor``：执行与 turn transition 全部走本地
  执行器；SvHook / Oracle 不参与决策。
- branch scheduler：与 strict_self_play 相同的 ``seed * 100000 + turn * 100 +
  action`` 派生，保证 (snapshot, action, branch_seed) 可复现。

非目标（明确不做）：

- 不接入 runtime selector，不做 promotion / guarded rollout。
- one-sided sign test 只是简单显著性提示，**不是** proof（不能称 proof）。
- 不使用任何 human replay 数据；对局全部由本地 self-play 生成。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.actions.mulligan_enumerator import MulliganActionEnumerator
from decision_engine.execution.execution_context import LocalExecutionContext
from decision_engine.execution.local_match_setup import (
    LocalDeckList,
    LocalMatchSetup,
    LocalMatchSetupConfig,
    draw_next_from_deck_instances,
    perform_mulligan,
)
from decision_engine.execution.player_local_predictor import PlayerLocalPredictor
from decision_engine.execution.rule_kernel import deckout_winner_from_results
from decision_engine.simulation.predictor import PredictionResult

STRICT_CERTAINTY_VALUES = frozenset({"exact", "spec_only"})


def action_match_key(action: LegalAction) -> tuple:
    """稳定 action 匹配键，用于判断计划步骤是否仍在当前 legal actions 中。

    只比较决策语义相关字段（动作类别、来源、出牌方式、目标、进化/选择/融合材料、
    variant），**忽略** 非语义字段（``cost`` 近似值、``legality_source`` / ``confidence``
    / ``description`` 这类注释性差异），避免因来源标注不同而误判 plan_stale。

    本函数原在 ``macro_ab_selector``。``turn_plan_arm``（锚点 A/B 的两臂）只需要它，
    不该因此依赖已冻结的 macro 搜索模块，故下沉到这里；``macro_ab_selector`` 保留
    re-export 以兼容既有调用方。
    """
    return (
        action.kind,
        action.source_uid,
        action.play_kind,
        tuple(action.target_groups),
        tuple(action.random_target_groups),
        tuple(action.selected_indexes),
        tuple(action.material_uids),
        action.variant,
    )


# ---------------------------------------------------------------------------
# Selector 抽象：两臂可注入不同搜索 / selector factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectorResult:
    """一次决策的结果 + 可审计代价诊断。

    selected_action_index 为 None 表示 selector 无法给出选择（视为拒绝，整局作废）。
    diagnostics 应尽量含以下 cost 键（缺失记 0）：

    - predictor_call_count：本次决策消耗的 predictor 调用数
    - micro_simulations：micro 层 simulation 次数（PUCT executed_simulations）
    - macro_node_costs：macro 层展开的节点数 / 代价
    """

    actions: tuple[LegalAction, ...]
    selected_action_index: int | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class ArmSelector(Protocol):
    """单臂决策器：给定 snapshot + 决策方，返回 SelectorResult。"""

    def select(self, snapshot: Any, *, player: str, branch_seed: int) -> SelectorResult:
        ...


# selector factory：由每臂注入，接收共享 predictor，返回 ArmSelector。
SelectorFactory = Callable[[PlayerLocalPredictor], ArmSelector]


@dataclass(frozen=True)
class SearchArm:
    """A/B 中的一臂。

    name：臂标识（如 "candidate" / "opponent"）。
    factory：注入的 selector factory；不同臂可注入不同搜索 / selector。
    config：可序列化配置（用于 config hash 与 manifest 记录）。
    """

    name: str
    factory: SelectorFactory
    config: Mapping[str, Any] = field(default_factory=dict)

    def config_hash(self) -> str:
        return config_hash(self.config)


def config_hash(config: Mapping[str, Any]) -> str:
    """对可序列化配置做稳定 hash（sha256 前 16 hex）。"""
    payload = json.dumps(_jsonable(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ---------------------------------------------------------------------------
# PUCT selector 适配器：把 puct.py 的 PuctSearch 包成 ArmSelector（不修改 puct.py）
# ---------------------------------------------------------------------------


# PuctSearch 工厂：接收共享 predictor + 本次决策的 branch_seed，返回一个带
# .search(snapshot, player=...) 的对象（通常是 PuctSearch）。
PuctSearchFactory = Callable[[PlayerLocalPredictor, int], Any]


class PuctArmSelector:
    """把返回 PuctSearchResult 的搜索对象适配为 ArmSelector。

    diagnostics 从 PuctSearchResult.diagnostics 抽取 predictor_call_count /
    executed_simulations(micro) / turn_transition_count 等，映射到标准 cost 键。
    """

    def __init__(self, predictor: PlayerLocalPredictor, search_factory: PuctSearchFactory):
        self._predictor = predictor
        self._search_factory = search_factory

    def select(self, snapshot: Any, *, player: str, branch_seed: int) -> SelectorResult:
        search = self._search_factory(self._predictor, branch_seed)
        result = search.search(snapshot, player=player)
        diag = dict(result.diagnostics)
        cost = {
            "predictor_call_count": int(diag.get("predictor_call_count") or 0),
            "action_predictor_call_count": int(diag.get("action_predictor_call_count") or 0),
            "turn_transition_predictor_call_count": int(diag.get("turn_transition_predictor_call_count") or 0),
            "micro_simulations": int(diag.get("executed_simulations") or 0),
            "requested_simulations": int(diag.get("requested_simulations") or 0),
            "macro_node_costs": int(diag.get("turn_transition_count") or 0),
            "cross_turn_count": int(diag.get("cross_turn_count") or 0),
            "effective_max_depth": int(diag.get("effective_max_depth") or 0),
            "partial_prediction_count": int(diag.get("partial_prediction_count") or 0),
            "prediction_error_count": int(diag.get("prediction_error_count") or 0),
            "rejected_action_count": int(diag.get("rejected_action_count") or 0),
            "search_value_source": diag.get("value_source"),
            "search_name": diag.get("search_name"),
        }
        return SelectorResult(
            actions=tuple(result.actions),
            selected_action_index=result.selected_action_index,
            diagnostics=cost,
        )


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchAbConfig:
    max_actions_per_turn: int = 12


# ---------------------------------------------------------------------------
# 单局结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameOutcome:
    game_id: str
    seed: int
    first_player: str            # "me" | "opponent" —— 谁先手
    candidate_side: str          # candidate 控制哪一方 ("me" | "opponent")
    accepted: bool               # 全程 strict 且 terminal
    winner_side: str | None      # "me" | "opponent" | None
    candidate_won: bool | None   # candidate 是否获胜（accepted 时才有意义）
    reason: str
    turns: int
    steps: int
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class PairResult:
    """一对 (镜像牌组, seed) 下 candidate 先手 / 后手两局的成对结果。"""

    pair_id: str
    matchup_id: str
    seed: int
    game_candidate_first: GameOutcome
    game_candidate_second: GameOutcome

    @property
    def accepted(self) -> bool:
        return self.game_candidate_first.accepted and self.game_candidate_second.accepted

    @property
    def rejection_reason(self) -> str | None:
        if self.accepted:
            return None
        if not self.game_candidate_first.accepted:
            return f"candidate_first:{self.game_candidate_first.reason}"
        return f"candidate_second:{self.game_candidate_second.reason}"

    def outcome_label(self) -> str:
        """pair-level 结果：win / loss / split（仅 accepted 时有效）。

        影之诗是零和博弈：一对局要么我方胜、要么对方胜，不存在"无胜负"。
        ``candidate_won is None`` 只可能来自终局判定缺陷（local predictor
        未产出终局，且没有回合上限兜底）。此时**必须显式失败**，不能因为
        ``None`` 为 falsy 就被下面 ``w1 and w2`` 静默判成 ``split`` 并按 0.5
        胜场计入分子——那会把"跑不完的局"伪装成"一胜一负"。
        """
        if not self.accepted:
            return "rejected"
        w1 = self.game_candidate_first.candidate_won
        w2 = self.game_candidate_second.candidate_won
        if w1 is None or w2 is None:
            raise AssertionError(
                f"pair {self.pair_id}（matchup={self.matchup_id} seed={self.seed}）"
                "标为 accepted，但存在 winner=None 的对局。"
                f"原因：first={self.game_candidate_first.reason!r} / second={self.game_candidate_second.reason!r}。"
                "零和对局不存在无胜负结果：这是终局判定缺陷，不得计作 split / 平局。"
            )
        if w1 and w2:
            return "win"
        if not w1 and not w2:
            return "loss"
        return "split"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


#: 最近 K 步动作进入样本。与 entity tensorization 的 HISTORY_ACTION_LIMIT 对齐：
#: 4 步覆盖"本回合内 combo 链"，跨回合长程依赖交给 value head。
_HISTORY_ACTION_LIMIT = 4


def branch_seed(seed: int, turn_index: int, action_index: int) -> int:
    """与 strict_self_play._action_seed 一致的 branch scheduler。"""
    return seed * 100_000 + turn_index * 100 + action_index


class StrictSearchAbRunner:
    """严格配对 search A/B self-play runner。

    两臂共享同一 ``PlayerLocalPredictor``（保证执行语义一致），只在决策 selector
    上不同。每对 (镜像牌组, seed) 跑两局：candidate 先手、candidate 后手。
    """

    def __init__(
        self,
        *,
        predictor: PlayerLocalPredictor,
        candidate_arm: SearchArm,
        opponent_arm: SearchArm,
        config: SearchAbConfig | None = None,
        step_recorder: Callable[..., None] | None = None,
        mulligan_selector: object | None = None,
        candidate_mulligan_arm: SearchArm | None = None,
        opponent_mulligan_arm: SearchArm | None = None,
        initial_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.predictor = predictor
        self.candidate_arm = candidate_arm
        self.opponent_arm = opponent_arm
        self.config = config or SearchAbConfig()
        # 初始状态变换钩子：**只用于受控实验**（例如"给 candidate 侧多 1 个 EP"），
        # 在换牌之后、回合循环之前调用一次。签名 ``(snapshot) -> snapshot``。
        # 它不得修改动作枚举/预测语义，也不得作为生产路径的默认行为。
        self._initial_transform = initial_transform
        # 决策记录钩子：在每个决策点、动作执行前调用，供导出/画像使用。
        # 约定 recorder 不得修改 snapshot / selection / action，也不得抛异常影响对局
        # ——它是旁路观察者，不是业务流程的一部分。签名见 _record_step。
        self._step_recorder = step_recorder
        self._candidate_selector = candidate_arm.factory(predictor)
        self._opponent_selector = opponent_arm.factory(predictor)
        #: 对局开始前的换牌策略。**两臂共用**（见 ``_run_game`` 里的说明）。
        #: ``None`` = 双方都不换牌，与加入本参数之前的行为完全一致。
        self.mulligan_selector = mulligan_selector
        #: candidate 臂的换牌决策臂。给了它之后 candidate 方不再使用
        #: ``mulligan_selector``，而是把换牌当作**动作空间的一部分**由策略自己选
        #: （``MULLIGAN_CHANGE`` / ``MULLIGAN_END``，走同一个 policy head）。
        #: 对手方仍用 ``mulligan_selector``（通常 ``None`` = 不换）。
        self._candidate_mulligan_arm = candidate_mulligan_arm
        self._candidate_mulligan_selector = (
            candidate_mulligan_arm.factory(predictor) if candidate_mulligan_arm is not None else None
        )
        #: opponent 臂的换牌决策臂。用于评估「只有一方会换牌」的对局
        #: （例如 candidate = 起点模型不换、opponent = 训练后模型会换）。
        self._opponent_mulligan_arm = opponent_mulligan_arm
        self._opponent_mulligan_selector = (
            opponent_mulligan_arm.factory(predictor) if opponent_mulligan_arm is not None else None
        )
        for arm, selector in (
            (candidate_mulligan_arm, self._candidate_mulligan_selector),
            (opponent_mulligan_arm, self._opponent_mulligan_selector),
        ):
            if arm is not None and not hasattr(selector, "select_mulligan"):
                raise TypeError(
                    "换牌臂的 selector 必须实现 select_mulligan(snapshot, actions, *, player, branch_seed)"
                    f"；{arm.name} 的 {type(selector).__name__} 没有这个方法"
                )
        self._mulligan_enumerator = MulliganActionEnumerator()

    def _record_step(
        self,
        *,
        game_id: str,
        seed: int,
        turn_index: int,
        step_index: int,
        player: str,
        arm_name: str,
        is_candidate_turn: bool,
        snapshot: Any,
        selection: Any,
        action: LegalAction,
        history_actions: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if self._step_recorder is None:
            return
        self._step_recorder(
            game_id=game_id,
            seed=seed,
            turn_index=turn_index,
            step_index=step_index,
            player=player,
            arm_name=arm_name,
            is_candidate_turn=is_candidate_turn,
            snapshot=snapshot,
            selection=selection,
            action=action,
            history_actions=tuple(history_actions),
        )


    # -- pair -------------------------------------------------------------

    def run_pair(
        self,
        deck_a: LocalDeckList,
        deck_b: LocalDeckList,
        *,
        matchup_id: str,
        seed: int,
        pair_id: str | None = None,
    ) -> PairResult:
        """镜像牌组 + 固定 seed 下跑 candidate 先手 / 后手两局。

        镜像：两局使用相同的 deck_a / deck_b 与相同 seed，只交换 first_player，
        并交换 candidate 控制的一方，保证先后手对称。
        """
        pair_id = pair_id or f"{matchup_id}:seed={seed}"
        # game 1：candidate 先手（candidate 控 me，first_player=me）
        game1 = self._run_game(
            deck_a,
            deck_b,
            game_id=f"{pair_id}:candidate_first",
            seed=seed,
            first_player="me",
            candidate_side="me",
        )
        # game 2：candidate 后手（candidate 控 me，first_player=opponent）
        game2 = self._run_game(
            deck_a,
            deck_b,
            game_id=f"{pair_id}:candidate_second",
            seed=seed,
            first_player="opponent",
            candidate_side="me",
        )
        return PairResult(
            pair_id=pair_id,
            matchup_id=matchup_id,
            seed=seed,
            game_candidate_first=game1,
            game_candidate_second=game2,
        )

    # -- single game ------------------------------------------------------

    def _run_game(
        self,
        deck_a: LocalDeckList,
        deck_b: LocalDeckList,
        *,
        game_id: str,
        seed: int,
        first_player: str,
        candidate_side: str,
    ) -> GameOutcome:
        setup = LocalMatchSetup().create(
            deck_a, deck_b, LocalMatchSetupConfig(seed=seed, first_player=first_player)
        )
        if setup.snapshot is None:
            return self._rejected(game_id, seed, first_player, candidate_side, "match_setup_error", 0, 0, {})

        snapshot = setup.snapshot
        cost_totals = {"predictor_call_count": 0, "micro_simulations": 0, "macro_node_costs": 0}
        arm_costs: dict[str, dict[str, int]] = {
            self.candidate_arm.name: {"predictor_call_count": 0, "micro_simulations": 0, "macro_node_costs": 0, "decisions": 0},
            self.opponent_arm.name: {"predictor_call_count": 0, "micro_simulations": 0, "macro_node_costs": 0, "decisions": 0},
        }
        step_count = 0

        # 换牌（mulligan）作为对局开始前的一步。
        #
        # 两种模式：
        # - ``mulligan_selector``（旧行为）：**所有臂共用**同一个策略。`plan_turn` / `policy` /
        #   `puct` 都没有换牌概念，若只让某一臂换牌，A/B 结果会被这个差异污染。
        # - ``candidate_mulligan_arm``（新）：candidate 把换牌当作**动作空间的一部分**，
        #   由策略自己逐张决定换/留，走同一个 policy head 与同一条 PPO rollout 记录路径；
        #   对手方仍用 ``mulligan_selector``（通常 None = 不换）。
        #
        # **不进入 strict 逐动作验收**：`perform_mulligan` 要洗牌重抽，结果不可能被
        # `LocalPredictor` EXACT 预测。换牌动作的合法性由 `MulliganActionEnumerator.
        # validate_changes` 做结构检查（张数上限 / uid 必须在手牌 / 不允许重复），
        # RNG 由 `seed` 派生固定，保证 `(deck, seed)` 可复现。
        if (
            self.mulligan_selector is not None
            or self._candidate_mulligan_selector is not None
            or self._opponent_mulligan_selector is not None
        ):
            for offset, player in enumerate(("me", "opponent")):
                mulligan_rng = random.Random(seed * 31 + offset + 1)
                if player == candidate_side:
                    arm_selector, arm = self._candidate_mulligan_selector, self._candidate_mulligan_arm
                else:
                    arm_selector, arm = self._opponent_mulligan_selector, self._opponent_mulligan_arm
                if arm_selector is not None:
                    result = self._run_candidate_mulligan(
                        snapshot, player,
                        arm_name=arm.name,
                        is_candidate_side=(player == candidate_side),
                        game_id=game_id, seed=seed, rng=mulligan_rng, step_count=step_count,
                    )
                    if isinstance(result, str):
                        return self._rejected(
                            game_id, seed, first_player, candidate_side, result,
                            0, step_count, cost_totals,
                        )
                    snapshot, step_count = result
                else:
                    changes = (
                        self.mulligan_selector.select_mulligan(snapshot, player=player)
                        if self.mulligan_selector is not None else ()
                    )
                    self._mulligan_enumerator.validate_changes(snapshot, player, tuple(changes))
                    snapshot = perform_mulligan(snapshot, player, tuple(changes), rng=mulligan_rng)

        # 受控实验的初始状态变换（例如"candidate 侧多 1 个 EP"）。放在换牌之后，
        # 避免 perform_mulligan 重建 snapshot 时丢掉改动。
        if self._initial_transform is not None:
            snapshot = self._initial_transform(snapshot)

        current_player = "me" if snapshot.is_ally_turn else "opponent"        # 最近 K 步已执行动作（跨回合），供 policy 学习 combo / 顺序依赖。
        # 只记录 base_card_id（不记 uid，避免不可泛化特征）。
        recent_actions: deque[dict[str, Any]] = deque(maxlen=_HISTORY_ACTION_LIMIT)
        last_turn = 0

        # 不设回合上限：仅在本地终局判定或执行失败时结束。
        turn_index = 0
        while True:
            turn_index += 1
            last_turn = turn_index
            winner = _winner(snapshot)
            if winner is not None:
                return self._accepted(game_id, seed, first_player, candidate_side, winner, turn_index, step_count, cost_totals, arm_costs)

            started = self._start_turn(snapshot, current_player, seed=seed, turn_index=turn_index)
            if isinstance(started, str):
                return self._rejected(game_id, seed, first_player, candidate_side, started, turn_index, step_count, cost_totals)
            snapshot = started
            winner = _winner(snapshot)
            if winner is not None:
                return self._accepted(game_id, seed, first_player, candidate_side, winner, turn_index, step_count, cost_totals, arm_costs)

            is_candidate_turn = current_player == candidate_side
            selector = self._candidate_selector if is_candidate_turn else self._opponent_selector
            arm_name = self.candidate_arm.name if is_candidate_turn else self.opponent_arm.name

            for action_index in range(1, self.config.max_actions_per_turn + 1):
                seed_for_search = branch_seed(seed, turn_index, 0)
                try:
                    selection = selector.select(snapshot, player=current_player, branch_seed=seed_for_search)
                except Exception as exc:  # 结构化拒绝，不做静默 fallback
                    return self._rejected(
                        game_id, seed, first_player, candidate_side,
                        f"select_error:{arm_name}:{type(exc).__name__}", turn_index, step_count, cost_totals,
                    )
                self._accumulate_cost(cost_totals, arm_costs[arm_name], selection.diagnostics)
                arm_costs[arm_name]["decisions"] += 1

                if selection.selected_action_index is None or not selection.actions:
                    return self._rejected(
                        game_id, seed, first_player, candidate_side,
                        f"no_selection:{arm_name}", turn_index, step_count,
                        {
                            **cost_totals,
                            "selector_rejection": dict(selection.diagnostics),
                        },
                    )
                action = selection.actions[selection.selected_action_index]
                before_snapshot = snapshot
                self._record_step(
                    game_id=game_id,
                    seed=seed,
                    turn_index=turn_index,
                    step_index=step_count,
                    player=current_player,
                    arm_name=arm_name,
                    is_candidate_turn=is_candidate_turn,
                    snapshot=snapshot,
                    selection=selection,
                    action=action,
                    history_actions=tuple(recent_actions),
                )
                prediction = self.predictor.predict_with_replay_context(
                    snapshot,
                    action,
                    player=current_player,
                    execution_context=LocalExecutionContext(branch_seed=branch_seed(seed, turn_index, action_index)),
                )
                if not _is_strict(prediction.certainty) or prediction.unresolved_effects:
                    return self._rejected(
                        game_id, seed, first_player, candidate_side,
                        f"executed_prediction_{_certainty_value(prediction.certainty)}:{arm_name}",
                        turn_index, step_count,
                        {**cost_totals, "executed_rejection": _prediction_rejection(prediction, action, snapshot, arm_name)},
                    )
                step_count += 1
                snapshot = prediction.snapshot
                # 执行成功后才记入历史（被拒绝的动作没有发生）。
                # 卡信息取自执行前的 before_snapshot：随从可能已被这次攻击打死。
                _source_card = before_snapshot.cards_by_uid.get(action.source_uid)
                recent_actions.append({
                    "kind": action.kind.value,
                    "source_base_card_id": (
                        int(_source_card.base_card_id) if _source_card is not None else None
                    ),
                    "actor": current_player,
                })
                winner = _winner(snapshot)
                if winner is not None:
                    return self._accepted(game_id, seed, first_player, candidate_side, winner, turn_index, step_count, cost_totals, arm_costs)
                if action.kind is ActionKind.TURN_END or snapshot.is_turn_ended:
                    break
            else:
                return self._rejected(game_id, seed, first_player, candidate_side, f"max_actions_per_turn:{arm_name}", turn_index, step_count, cost_totals)
            current_player = "opponent" if current_player == "me" else "me"

    def _run_candidate_mulligan(
        self,
        snapshot: Any,
        player: str,
        *,
        arm_name: str,
        is_candidate_side: bool,
        game_id: str,
        seed: int,
        rng: random.Random,
        step_count: int,
    ) -> tuple[Any, int] | str:
        """换牌臂的决策阶段：策略自己逐张决定换/留。

        换牌动作走与局内动作同一条选择链路（``select_mulligan`` → policy head），
        每个换牌决策都会经过 ``step_recorder``，因此会进入 PPO 的轨迹与梯度。
        换牌的实际执行在收集完决策后**一次性**做（规则上换掉的牌是一起洗回重抽的）。

        Returns:
            ``(换牌后的 snapshot, 新的 step_count)``；失败时返回结构化拒绝原因。
        """
        selector = (
            self._candidate_mulligan_selector if is_candidate_side else self._opponent_mulligan_selector
        )
        changes: list[int] = []
        # 上界 = 最多换 max_changes 张 + 1 次 MULLIGAN_END。
        for _ in range(self._mulligan_enumerator.max_changes + 1):
            actions = self._mulligan_enumerator.enumerate(
                snapshot, player=player, already_changed=tuple(changes)
            )
            selection = selector.select_mulligan(
                snapshot, actions, player=player, branch_seed=branch_seed(seed, 0, len(changes))
            )
            if selection.selected_action_index is None or not selection.actions:
                return f"mulligan_no_selection:{arm_name}"
            action = selection.actions[selection.selected_action_index]
            self._record_step(
                game_id=game_id,
                seed=seed,
                turn_index=0,
                step_index=step_count,
                player=player,
                arm_name=arm_name,
                is_candidate_turn=is_candidate_side,
                snapshot=snapshot,
                selection=selection,
                action=action,
            )
            step_count += 1
            if action.kind is ActionKind.MULLIGAN_END:
                break
            if action.kind is not ActionKind.MULLIGAN_CHANGE:
                return f"mulligan_unexpected_action:{action.kind.value}:{arm_name}"
            changes.append(int(action.source_uid))
        try:
            self._mulligan_enumerator.validate_changes(snapshot, player, tuple(changes))
        except ValueError as exc:
            return f"mulligan_invalid:{type(exc).__name__}:{arm_name}"
        return perform_mulligan(snapshot, player, tuple(changes), rng=rng), step_count

    def _start_turn(self, snapshot: Any, player: str, *, seed: int, turn_index: int) -> Any | str:
        drawn_card, draw_snapshot = draw_next_from_deck_instances(snapshot, player)
        action = LegalAction(kind=ActionKind.TURN_START, source_uid=0)
        result = self.predictor.predict_with_replay_context(
            draw_snapshot,
            action,
            player=player,
            generated_cards=(drawn_card,) if drawn_card is not None else (),
            execution_context=LocalExecutionContext(branch_seed=branch_seed(seed, turn_index, -1)),
        )
        if not _is_strict(result.certainty) or result.unresolved_effects:
            return f"turn_start_prediction_{_certainty_value(result.certainty)}"
        return result.snapshot

    def _accumulate_cost(self, totals: dict[str, int], arm_total: dict[str, int], diagnostics: Mapping[str, Any]) -> None:
        for key in ("predictor_call_count", "micro_simulations", "macro_node_costs"):
            value = int(diagnostics.get(key) or 0)
            totals[key] += value
            arm_total[key] += value

    def _accepted(
        self,
        game_id: str,
        seed: int,
        first_player: str,
        candidate_side: str,
        winner: str,
        turns: int,
        steps: int,
        cost_totals: Mapping[str, Any],
        arm_costs: Mapping[str, Any],
    ) -> GameOutcome:
        # simultaneous 视为无胜者 → 不纳入（accepted=False，因为没有明确 terminal winner）
        if winner not in {"me", "opponent"}:
            return self._rejected(game_id, seed, first_player, candidate_side, f"non_decisive:{winner}", turns, steps, cost_totals)
        candidate_won = winner == candidate_side
        diagnostics = {
            "cost_totals": dict(cost_totals),
            "arm_costs": {name: dict(value) for name, value in arm_costs.items()},
            "branch_scheduler": {"formula": "seed*100000 + turn*100 + action", "seed": seed},
        }
        return GameOutcome(
            game_id=game_id,
            seed=seed,
            first_player=first_player,
            candidate_side=candidate_side,
            accepted=True,
            winner_side=winner,
            candidate_won=candidate_won,
            reason=f"terminal:{winner}",
            turns=turns,
            steps=steps,
            diagnostics=diagnostics,
        )

    #: ``_rejected`` 收到的 payload 里哪些键是 cost 累加项。其余键（``selector_rejection``
    #: / ``executed_rejection`` 等）是诊断上下文，必须提到顶层——否则排查时会被埋在
    #: ``cost_totals`` 下面，看起来像"作废但没有任何原因"。
    _COST_KEYS = frozenset({"predictor_call_count", "micro_simulations", "macro_node_costs"})

    def _rejected(
        self,
        game_id: str,
        seed: int,
        first_player: str,
        candidate_side: str,
        reason: str,
        turns: int,
        steps: int,
        payload: Mapping[str, Any],
    ) -> GameOutcome:
        cost_totals = {key: value for key, value in payload.items() if key in self._COST_KEYS}
        extras = {key: value for key, value in payload.items() if key not in self._COST_KEYS}
        diagnostics = {
            "cost_totals": dict(cost_totals),
            "branch_scheduler": {"formula": "seed*100000 + turn*100 + action", "seed": seed},
            **extras,
        }
        return GameOutcome(
            game_id=game_id,
            seed=seed,
            first_player=first_player,
            candidate_side=candidate_side,
            accepted=False,
            winner_side=None,
            candidate_won=None,
            reason=reason,
            turns=turns,
            steps=steps,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 汇总 + one-sided sign test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbSummary:
    accepted_pairs: int
    rejected_pairs: int
    win: int
    loss: int
    split: int
    equal: int
    sign_test: Mapping[str, Any]
    rejection_reasons: Mapping[str, int]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "accepted_pairs": self.accepted_pairs,
            "rejected_pairs": self.rejected_pairs,
            "pair_outcomes": {"win": self.win, "loss": self.loss, "split": self.split, "equal": self.equal},
            "one_sided_sign_test": dict(self.sign_test),
            "rejection_reasons": dict(self.rejection_reasons),
            "disclaimer": "one-sided sign test 仅为显著性提示，非 proof",
        }


def candidate_win_counts(summary: "AbSummary") -> tuple[int, int]:
    """``(candidate 胜局, 总局数)``；分母是 2 × accepted_pairs（每对两局）。

    ``pair_outcomes`` 记录的是 **candidate** 的胜负，`split` 是"同 seed 下一胜一负"，
    所以按 0.5 计入胜场：

        candidate 胜局 = 2 * win + split

    ## 为什么单独成一个函数

    这个换算很容易写反：`tools/run_turn_plan_ab.py` 里被测模型在 **baseline** 臂
    （candidate 是 `plan_turn` 对手），所以那边要用 `2*loss + split`；
    `tools/run_cross_deck_ab.py` 里被测模型在 **candidate** 臂，要用 `2*win + split`。
    两处各写一遍时，第二处确实照抄错了方向（同配置两次运行报出相反的 75%/25%）。
    现在统一走这里。
    """
    return 2 * int(summary.win) + int(summary.split), 2 * int(summary.accepted_pairs)


def candidate_win_rate(summary: "AbSummary") -> float | None:
    """candidate 臂的胜率（0–1），无有效局时返回 ``None``。"""
    wins, games = candidate_win_counts(summary)
    return (wins / games) if games > 0 else None


def _sign_test_p_value(wins: int, losses: int) -> float:
    """双臂 pair 中 win vs loss 的 one-sided sign test p 值（H0: p=0.5）。
    忽略 split / equal（无方向 tie）。返回 P(X >= wins | n=wins+losses, p=0.5)。
    这是简单二项检验，**不是** proof。
    """
    n = wins + losses
    if n == 0:
        return 1.0
    # 精确二项尾概率（n 不大，直接累加）
    from math import comb

    tail = sum(comb(n, k) for k in range(wins, n + 1))
    return tail / (2 ** n)


def summarize(pairs: Sequence[PairResult]) -> AbSummary:
    # equal 结构性恒 0：零和对局没有"无胜负"标签。保留字段仅为 JSON 向后兼容，
    # 不要据此推断"平局数"——真要统计平局只能看 rejected_reasons 里的终局缺陷。
    win = loss = split = equal = 0
    accepted = 0
    rejected = 0
    rejection_reasons: dict[str, int] = {}
    for pair in pairs:
        if not pair.accepted:
            rejected += 1
            reason = pair.rejection_reason or "unknown"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        accepted += 1
        label = pair.outcome_label()
        if label == "win":
            win += 1
        elif label == "loss":
            loss += 1
        elif label == "split":
            split += 1
        else:
            # outcome_label 对 accepted 的 pair 只返回 win / loss / split；
            # rejected 已在上方 continue。走到这里说明 label 语义被改坏了。
            raise AssertionError(f"未预期的 pair label {label!r}（pair={pair.pair_id}）")

    p_value = _sign_test_p_value(win, loss)
    sign_test = {
        "wins": win,
        "losses": loss,
        "ignored_split": split,
        "ignored_equal": equal,
        "n_directional": win + loss,
        "null_hypothesis": "candidate win-rate = 0.5 among decisive pairs",
        "p_value_one_sided": p_value,
        "is_proof": False,
    }
    return AbSummary(
        accepted_pairs=accepted,
        rejected_pairs=rejected,
        win=win,
        loss=loss,
        split=split,
        equal=equal,
        sign_test=sign_test,
        rejection_reasons=rejection_reasons,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _certainty_value(certainty: Any) -> str:
    return str(getattr(certainty, "value", certainty))


def _is_strict(certainty: Any) -> bool:
    return _certainty_value(certainty) in STRICT_CERTAINTY_VALUES


def _prediction_rejection(prediction: PredictionResult, action: LegalAction, snapshot: Any, arm: str) -> dict[str, Any]:
    source = snapshot.cards_by_uid.get(action.source_uid)
    return {
        "arm": arm,
        "certainty": _certainty_value(prediction.certainty),
        "action_kind": action.kind.value,
        "source_uid": action.source_uid,
        "source_base_card_id": getattr(source, "base_card_id", None),
        "description": str(prediction.description)[:240],
        "unresolved_effects": list(prediction.unresolved_effects),
    }


def _winner(snapshot: Any) -> str | None:
    if snapshot.me.life <= 0 and snapshot.opponent.life <= 0:
        return "simultaneous"
    if snapshot.me.life <= 0:
        return "opponent"
    if snapshot.opponent.life <= 0:
        return "me"
    if snapshot.me.result or snapshot.opponent.result:
        winner = deckout_winner_from_results(int(snapshot.me.result or 0), int(snapshot.opponent.result or 0))
        if winner is not None:
            return winner
        if snapshot.me.result and snapshot.opponent.result:
            return "simultaneous"
        if snapshot.me.result:
            return "me"
        if snapshot.opponent.result:
            return "opponent"
    return None
