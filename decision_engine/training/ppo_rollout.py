# -*- coding: utf-8 -*-
"""PPO rollout：策略采样 arm + 轨迹收集。

## 为什么不能直接用现有 self-play runner

`SelfPlaySearchStep` / `StrictSelfPlayStep` 记录的是 PUCT 的 ``search_result`` 与
``selected_index``，**没有 ``log_prob``，也没有 ``value``**，而且动作是 argmax/搜索选出的、
不是按策略分布采样的。PPO 的 on-policy 校正需要 ``log π(a|s)`` 与 ``V(s)``，所以需要单独的
采样路径。

## 设计

- ``SampledPolicyArmSelector`` 实现与 ``PolicyPriorArmSelector`` 相同的 ``select`` 协议，
  因此可以原样插进 ``StrictSearchAbRunner``，复用它的回合切换、strict 验收和整局管理。
- 采样前先按 strict 口径**过滤掉不可执行的动作**，在剩余动作上重新归一化再采样。
  这样 ``log_prob`` 始终对应"实际可执行动作上的条件分布"，PPO 的 importance ratio 才成立。
  代价是每步多花 ``len(actions)`` 次 predictor 调用（实测均值 5.7 次）。
- ``log_prob`` / ``value`` 通过 ``SelectorResult.diagnostics`` 传出，**不改 dataclass**。
- 每个 batch 收集固定局数后立即张量化并训练，避免把整批 snapshot 留在内存里
  （47 万 step 的 snapshot 引用会撑爆内存）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from decision_engine.planning.search_ab_match import SelectorResult
from decision_engine.planning.turn_plan_arm import strict_prediction_filter


class SampledPolicyArmSelector:
    """按策略分布采样动作（PPO 的 rollout 臂）。"""

    def __init__(
        self,
        predictor: Any,
        enumerator: Any,
        *,
        model: Any,
        vocabulary: Any,
        device: str = "cpu",
        temperature: float = 1.0,
        seed: int = 0,
        prediction_filter=strict_prediction_filter,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature 必须为正")
        self._predictor = predictor
        self._enumerator = enumerator
        self._model = model
        self._vocabulary = vocabulary
        self._device = device
        self._temperature = temperature
        self._rng = random.Random(seed)
        self._prediction_filter = prediction_filter

    # ------------------------------------------------------------------
    def _logits_and_value(self, snapshot: Any, actions: Sequence[Any], player: str):
        import torch

        from decision_engine.planning._fast_tensorize import fast_tensorize_for_inference

        batch = fast_tensorize_for_inference(
            snapshot, actions, player=player, device=self._device, vocabulary=self._vocabulary
        )
        with torch.no_grad():
            logits, value = self._model(batch)
        return logits[0], float(value[0].item())

    def select(self, snapshot: Any, *, player: str, branch_seed: int) -> SelectorResult:
        actions = tuple(self._enumerator.enumerate(snapshot, player=player))
        if not actions:
            return SelectorResult(
                actions=(), selected_action_index=None,
                diagnostics={"arm": "ppo_sample", "reject": "no_legal_actions"},
            )

        # 先按 strict 口径筛掉不可执行动作：runner 拒绝任何一步非 strict 的预测（整局作废），
        # 因此在"不可执行动作"上分配概率既浪费又会让 log_prob 与真实可执行分布不一致。
        executable: list[int] = []
        checked = 0
        for index, action in enumerate(actions):
            try:
                prediction = self._predictor.predict(snapshot, action, player=player)
            except Exception:
                checked += 1
                continue
            checked += 1
            if self._prediction_filter(prediction):
                executable.append(index)
        if not executable:
            return SelectorResult(
                actions=actions, selected_action_index=None,
                diagnostics={"arm": "ppo_sample", "reject": "no_strict_action", "predictor_call_count": checked},
            )

        import torch

        # 用**全部** actions 前向，再把不可执行的位置 mask 成 -inf 后归一化。
        # 若只把 executable 子集喂给网络，logits 长度就与训练时（全部 actions）不同，
        # 训练阶段算出的 log_prob 会与采样时的 index 错位。
        logits, value = self._logits_and_value(snapshot, actions, player)
        allowed = torch.tensor(executable, dtype=torch.long)
        masked = torch.full_like(logits, float("-inf"))
        masked[allowed] = logits[allowed]
        # temperature 只在采样时生效；log_prob 必须与采样分布一致，所以按同一温度缩放。
        scaled = masked / self._temperature
        log_probs = torch.log_softmax(scaled, dim=-1)
        probs = log_probs.exp()
        total = float(probs.sum().item())
        if not math.isfinite(total) or total <= 0:
            return SelectorResult(
                actions=actions, selected_action_index=None,
                diagnostics={"arm": "ppo_sample", "reject": "degenerate_distribution"},
            )

        draw = self._rng.random()
        cumulative = 0.0
        selected_index = executable[-1]
        for position, probability in enumerate(probs.tolist()):
            cumulative += probability
            if draw <= cumulative:
                selected_index = position
                break

        return SelectorResult(
            actions=actions,
            selected_action_index=selected_index,
            diagnostics={
                "arm": "ppo_sample",
                "predictor_call_count": checked,
                "micro_simulations": 0,
                "macro_node_costs": 0,
                "log_prob": float(log_probs[selected_index].item()),
                "value": value,
                "executable_count": len(executable),
                "executable_indices": tuple(executable),
                "entropy": float(-(probs * log_probs).sum().item()),
            },
        )

    def select_mulligan(
        self, snapshot: Any, actions: Sequence[Any], *, player: str, branch_seed: int
    ) -> SelectorResult:
        """换牌阶段的动作选择。

        与 ``select`` 的唯一区别：**不做 strict 可执行性筛选**。

        ``select`` 会对每个候选调用 ``predictor.predict`` 把不可执行动作 mask 掉，
        但换牌要洗牌重抽，``LocalPredictor`` 无法 EXACT 预测，逐候选 predict 只会
        全部失败。换牌动作的合法性由 ``MulliganActionEnumerator.validate_changes``
        做结构检查（张数上限 / uid 必须在手牌 / 不允许重复），因此这里可以直接对
        全部候选打分——这不是静默 fallback，而是换牌阶段的验收语义与局内不同。

        ``actions`` 由调用方（runner）传入，而不是内部枚举：换牌候选依赖
        ``already_changed``（已决定换掉的牌不再出现），这个状态只有 runner 持有。
        """
        actions = tuple(actions)
        if not actions:
            return SelectorResult(
                actions=(),
                selected_action_index=None,
                diagnostics={"arm": "ppo_sample", "reject": "no_mulligan_actions"},
            )

        import torch

        logits, value = self._logits_and_value(snapshot, actions, player)
        # temperature 只在采样时生效；log_prob 必须与采样分布一致，所以按同一温度缩放。
        scaled = logits / self._temperature
        log_probs = torch.log_softmax(scaled, dim=-1)
        probs = log_probs.exp()
        total = float(probs.sum().item())
        if not math.isfinite(total) or total <= 0:
            return SelectorResult(
                actions=actions,
                selected_action_index=None,
                diagnostics={"arm": "ppo_sample", "reject": "degenerate_distribution"},
            )

        draw = self._rng.random()
        cumulative = 0.0
        selected_index = len(actions) - 1
        for position, probability in enumerate(probs.tolist()):
            cumulative += probability
            if draw <= cumulative:
                selected_index = position
                break

        return SelectorResult(
            actions=actions,
            selected_action_index=selected_index,
            diagnostics={
                "arm": "ppo_sample",
                "predictor_call_count": 0,
                "micro_simulations": 0,
                "macro_node_costs": 0,
                "log_prob": float(log_probs[selected_index].item()),
                "value": value,
                # 换牌不做 strict 筛选，全部候选都可执行——这个字段的语义是
                # "进入采样分布的动作集合"，不是"通过 LocalPredictor 验收的动作集合"。
                "executable_count": len(actions),
                "executable_indices": tuple(range(len(actions))),
                "entropy": float(-(probs * log_probs).sum().item()),
                "mulligan_stage": True,
            },
        )


# ----------------------------------------------------------------------
# 轨迹收集
# ----------------------------------------------------------------------


@dataclass
class RolloutStep:
    """一步决策。``example`` 是该步的 ``TrainingExample.to_jsonable()`` 结果。

    存 JSON 而不是现成 tensor 是有意的：训练侧统一走
    ``structured_tensorize_training_example`` + ``_collate_structured``，那条路径已处理
    action 维 padding、mask、history 字段等全部边界；手工拼 tensor 容易在 N 不一致时出错。
    """

    sample_id: str
    player: str
    example: dict
    selected_index: int
    log_prob: float
    value: float
    executable_count: int
    executable_indices: tuple[int, ...]
    turn_index: int
    step_index: int
    reward: float = 0.0
    #: 该步所属对局的终局结果（+1 胜 / -1 负），``finalize_game`` 时回填。
    #: 只用于诊断 critic 的判别力（value 与胜负的 AUC），不参与任何 loss。
    outcome: float = 0.0
    #: 是否是换牌阶段（局前）的决策。换牌在 t=0、reward 在 t≈27，
    #: GAE(λ=0.95) 会让优势信号衰减到约 0.25（0.95^27），因此可选用 MC return
    #: （``PPOConfig.mulligan_mc_return``）给换牌步直接注入终局回报。
    is_mulligan: bool = False


@dataclass
class RolloutBatch:
    """一个 on-policy batch：按对局分组，便于 GAE 与优势归一化。"""

    episodes: list[list[RolloutStep]] = field(default_factory=list)
    dropped_episodes: int = 0
    total_episodes: int = 0

    def add_episode(self, steps: list[RolloutStep]) -> None:
        self.episodes.append(steps)
        self.total_episodes += 1

    def drop_episode(self) -> None:
        self.dropped_episodes += 1
        self.total_episodes += 1

    def all_steps(self) -> list[RolloutStep]:
        return [step for episode in self.episodes for step in episode]

    def __len__(self) -> int:
        return len(self.all_steps())


class PPORolloutCollector:
    """交给 ``StrictSearchAbRunner`` 的 ``step_recorder``。

    只为**被采样臂控制的一方**记录（对局中每方轮流决策，对手臂的步子不该进梯度）。

    ## 稠密 reward

    纯终局 ±1 的 advantage 完全来自 ``γV(s') − V(s)``。作为对照，TavernAgent 用的是
    终局 **placement（1–8 名，8 个值）**——同样是稀疏终局，但每局只有一个标量且带序。
    这里用本项目的等价物：**回合内累计净血差的增量**。

    ``score_terminal(round_start, snapshot).net_life`` 就是"从回合开始到当前"的累计净血差
    （``our_damage = theirs_before.life − theirs_after.life``），相邻两步相减即该步的增量。
    这样"解掉对方威胁"也会得到正 reward，而不只是"打脸"。

    终局仍额外加 ±1，作为锚点。
    """

    def __init__(self, *, actor_side: str = "me", reward_scale: float = 0.05) -> None:
        self._actor_side = actor_side
        self._reward_scale = reward_scale
        self._pending: dict[str, list[RolloutStep]] = {}
        #: game_id -> (turn_index, round_start_snapshot, 上一步累计净血差)
        self._round_state: dict[str, tuple[int, Any, float]] = {}
        self.batch = RolloutBatch()

    def _net_life_increment(self, game_id: str, snapshot: Any, player: str, turn_index: int) -> float:
        """本步的累计净血差增量（回合内按 ``round_start`` 计算）。"""
        from decision_engine.planning.turn_evaluation import score_terminal

        state = self._round_state.get(game_id)
        if state is None or state[0] != turn_index:
            # 新回合（或该局第一步）：以当前 snapshot 作为回合起点，累计净值从 0 起算。
            round_start = snapshot
            previous = score_terminal(round_start, snapshot, player=player).net_life
            self._round_state[game_id] = (turn_index, round_start, previous)
            return 0.0
        _, round_start, previous = state
        current = score_terminal(round_start, snapshot, player=player).net_life
        self._round_state[game_id] = (turn_index, round_start, current)
        return float(current - previous)

    def __call__(
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
        action: Any,
        history_actions: Any = (),
    ) -> None:
        if player != self._actor_side:
            return
        diagnostics = dict(getattr(selection, "diagnostics", {}) or {})
        if diagnostics.get("arm") != "ppo_sample":
            return
        log_prob = diagnostics.get("log_prob")
        value = diagnostics.get("value")
        if log_prob is None or value is None:
            return

        index = getattr(selection, "selected_action_index", None)
        if index is None:
            return

        from decision_engine.training.example_builder import build_training_example

        actions = tuple(selection.actions)
        example = build_training_example(
            snapshot=snapshot,
            legal_actions=actions,
            sample_id=f"{game_id}:t{turn_index}:s{step_index}",
            source={
                "kind": "ppo_rollout",
                "game_id": game_id,
                "seed": seed,
                "turn_index": turn_index,
                "step_index": step_index,
            },
            trust={"level": "ppo_rollout", "accepted": True, "policy_source": "sampled_policy"},
            perspective=player,
            selected_action_index=int(index),
            value_target=None,
            generation_config={"kind": "ppo_rollout", "seed": seed, "game_id": game_id},
            metadata={"log_prob": float(log_prob), "value": float(value)},
        )
        increment = self._net_life_increment(game_id, snapshot, player, int(turn_index))
        step = RolloutStep(
            sample_id=f"{game_id}:t{turn_index}:s{step_index}",
            player=player,
            example=example.to_jsonable(),
            selected_index=int(index),
            log_prob=float(log_prob),
            value=float(value),
            executable_count=int(diagnostics.get("executable_count") or len(actions)),
            executable_indices=tuple(int(i) for i in (diagnostics.get("executable_indices") or range(len(actions)))),
            turn_index=int(turn_index),
            step_index=int(step_index),
            reward=self._reward_scale * increment,
            is_mulligan=bool(diagnostics.get("mulligan_stage")),
        )
        self._pending.setdefault(game_id, []).append(step)

    def finalize_game(self, game: Any) -> None:
        """终局额外加 ±1 作为锚点；作废局整局丢弃。"""
        game_id = getattr(game, "game_id", None)
        steps = self._pending.pop(game_id, []) if game_id else []
        self._round_state.pop(game_id, None) if game_id else None
        if not steps:
            self.batch.drop_episode()
            return
        if not getattr(game, "accepted", False) or getattr(game, "winner_side", None) not in {"me", "opponent"}:
            # 对局没打完 → 中间决策没有可靠终值，整局丢弃，不猜 0。
            self.batch.drop_episode()
            return
        winner = game.winner_side
        for position, step in enumerate(steps):
            terminal = 1.0 if step.player == winner else -1.0
            step.outcome = terminal
            step.reward = (step.reward or 0.0) + (terminal if position == len(steps) - 1 else 0.0)
        self.batch.add_episode(steps)
