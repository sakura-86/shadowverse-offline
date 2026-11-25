# -*- coding: utf-8 -*-
"""Value Target 回填（self-play 后处理）。

对 self-play 对局的每一帧 F（行动方 P），定位 4 回合窗口终点 C_end
（P 的下下个回合的末尾 after-snapshot），计算 value_target 写回样本。

设计见 docs/plan/VALUE_TARGET_DESIGN.md。不改搜索，纯数据后处理，低风险。
"""

from __future__ import annotations

from typing import Any

from decision_engine.planning.value_target import (
    ValueTargetWeights, DEFAULT_VALUE_TARGET_WEIGHTS, compute_value_target,
)


def _after_snapshot(step) -> Any:
    """一帧动作执行后的 snapshot（优先 prediction.snapshot，回退 step.snapshot）。"""
    pred = getattr(step, "prediction", None)
    snap = getattr(pred, "snapshot", None) if pred is not None else None
    return snap if snap is not None else step.snapshot


def _turn_key(step) -> tuple:
    """回合标识：(player, turn_index)。同一回合的连续帧 turn_index 相同、player 相同。"""
    return (step.player, step.turn_index)


def compute_targets_for_episode(
    steps,
    *,
    weights: ValueTargetWeights = DEFAULT_VALUE_TARGET_WEIGHTS,
    lookahead_my_turns: int = 1,
) -> dict:
    """对一局 steps 计算每帧的 value_target。返回 {sample_id: value_target}。

    窗口：帧 F（行动方 P）→ P 之后第 `lookahead_my_turns` 个我方回合末尾。
    - lookahead_my_turns=1（默认，实验最优）：A→C（对方1回合B + 我方1回合C），
      "看对方1回合+我方1回合"的最短兑现窗口。gate 验证 6 卡组全面胜过 outcome（66-31）。
    - lookahead_my_turns=2：A→E（跨我方2个后续回合，旧行为，窗口过长稀释信号，已被证明劣于1）。
    - lookahead_my_turns=3：A→G（6回合窗口）。
    - 先按出现顺序切回合段（连续相同 (player, turn_index) 归为一段）。
    - end = 第 lookahead_my_turns 个 player==P 的回合段最后一帧 after-snapshot；
      不存在（对局提前结束）则用整局最后一帧 after-snapshot（含终局，趋近 outcome）。
    """
    if not steps:
        return {}

    # 1. 切回合段（保序）
    segments = []  # list of (player, [step,...])
    for st in steps:
        key = _turn_key(st)
        if segments and _turn_key(segments[-1][1][-1]) == key:
            segments[-1][1].append(st)
        else:
            segments.append((st.player, [st]))

    # 每段的 player 列表，用于定位"第 N 个同 player 回合段"
    seg_players = [p for p, _ in segments]
    last_after = _after_snapshot(steps[-1])  # 整局最后 after（兜底 end）

    out: dict = {}
    for seg_idx, (player, seg_steps) in enumerate(segments):
        # 找 seg_idx 之后第 lookahead_my_turns 个 player==player 的段索引
        seen = 0
        c_seg_idx = None
        for j in range(seg_idx + 1, len(segments)):
            if seg_players[j] == player:
                seen += 1
                if seen == lookahead_my_turns:
                    c_seg_idx = j
                    break
        if c_seg_idx is not None:
            end_snapshot = _after_snapshot(segments[c_seg_idx][1][-1])
        else:
            end_snapshot = last_after  # 提前结束：用整局末尾

        for st in seg_steps:
            start_snapshot = st.snapshot
            target = compute_value_target(start_snapshot, end_snapshot, player, weights=weights)
            out[st.sample_id] = target
    return out


def compute_targets_multi_window(
    steps,
    window_weights,
    *,
    weights: ValueTargetWeights = DEFAULT_VALUE_TARGET_WEIGHTS,
) -> dict:
    """多窗口加权 value_target：sum(w_i · value_target(lookahead_i)) / sum(w_i)。

    window_weights: 序列 [(lookahead_my_turns, weight), ...]，
      例 [(1, 0.8), (2, 0.2)] = 0.8·短窗(A→C) + 0.2·长窗(A→E)。
    各窗口的 value_target 已是 tanh 归一值，加权后仍在 (-1, 1)。
    权重自动归一化（除以权重和）。
    """
    if not window_weights:
        raise ValueError("window_weights 不能为空")
    total_w = sum(w for _, w in window_weights)
    if total_w <= 0:
        raise ValueError("窗口权重和必须 > 0")

    per_window = [
        compute_targets_for_episode(steps, weights=weights, lookahead_my_turns=la)
        for la, _ in window_weights
    ]
    out: dict = {}
    # 以第一个窗口的 sample_id 全集为准（各窗口 sample_id 相同）
    for sid in per_window[0]:
        acc = 0.0
        for (la, w), tgt_map in zip(window_weights, per_window):
            acc += w * tgt_map.get(sid, 0.0)
        out[sid] = acc / total_w
    return out


def backfill_run_result(
    result,
    *,
    weights: ValueTargetWeights = DEFAULT_VALUE_TARGET_WEIGHTS,
    lookahead_my_turns: int = 1,
    window_weights=None,
) -> dict:
    """对整个 SelfPlaySearchRunResult 回填。返回 {sample_id: value_target}（所有 episode 合并）。

    - window_weights 非空：多窗口加权（如 [(1,0.8),(2,0.2)]），忽略 lookahead_my_turns。
    - 否则：单窗口 lookahead_my_turns。
    """
    targets: dict = {}
    for ep in result.episodes:
        if window_weights:
            targets.update(compute_targets_multi_window(ep.steps, window_weights, weights=weights))
        else:
            targets.update(compute_targets_for_episode(ep.steps, weights=weights, lookahead_my_turns=lookahead_my_turns))
    return targets


# ---------------------------------------------------------------------------
# MC discounted：从终局真实胜负反推、按回合折现（MDP value，无代理资源量）
# ---------------------------------------------------------------------------


def compute_mc_discounted_for_episode(steps, winner, *, gamma: float = 0.97) -> dict:
    """蒙特卡洛终局 return，按回合折现。返回 {sample_id: value_target}。

    MDP 语义：一局制游戏唯一真实 reward 是终局胜负 z∈{+1,-1}，中间步 reward=0。
    故状态 s_t 的真实 value = γ^(T_turn - t_turn) · z_P，其中：
      - z_P：该帧行动方 P 视角的终局胜负（P 赢=+1，P 输=-1）；
      - t_turn：该帧的 turn_index；T_turn：本局最后一帧的 turn_index；
      - 折现以回合为单位（距终局的回合数），不按动作数。
    终局帧 |value|=1，越早期 |value| 越小（早期不确定性自然衰减）。

    winner: "me" / "opponent" / None。None（无胜负）→ 该 episode 不回填（返回空）。
    simultaneous 不会作为 winner 传入（runner 只给 me/opponent/None）。
    """
    if not steps or winner not in ("me", "opponent"):
        return {}
    # 本局终局回合 = 最后一帧 turn_index
    t_final = max(int(st.turn_index) for st in steps)
    out: dict = {}
    for st in steps:
        z_p = 1.0 if winner == st.player else -1.0
        turns_to_end = max(0, t_final - int(st.turn_index))
        out[st.sample_id] = (gamma ** turns_to_end) * z_p
    return out


def backfill_run_result_mc(result, *, gamma: float = 0.97) -> dict:
    """对整个 SelfPlaySearchRunResult 做 mc_discounted 回填（所有 episode 合并）。

    用 episode 的真实 winner（runner 权威判定）+ 按回合折现，无 resource_exchange 代理。
    无胜负的 episode（winner=None）其样本不在返回表中，调用方保留原 value_target。
    """
    targets: dict = {}
    for ep in result.episodes:
        targets.update(compute_mc_discounted_for_episode(ep.steps, ep.winner, gamma=gamma))
    return targets


# ---------------------------------------------------------------------------
# TD(λ) / n-step：前向视角 λ-return，bootstrap 用规则估值（零 neural 依赖）
# ---------------------------------------------------------------------------


def compute_td_lambda_for_episode(steps, winner, *, gamma: float = 0.97, lam: float = 0.7,
                                  bootstrap_value=None) -> dict:
    """前向视角 λ-return，按回合折现。返回 {sample_id: value_target}。

    动机（bias-variance 谱系的中间地带）：
      - resource_exchange：有偏、低方差（固定窗口资源代理，经验上最强）。
      - mc_discounted（λ=1 的极限）：无偏、高方差（纯 MC 终局 return，bishop 等长局最差）。
      - TD(λ)：用 n-step return 的几何加权（λ 控权重衰减）在两者间连续插值。
        λ→0 退化为 1-step TD（最低方差、偏差来自 bootstrap）；λ→1 退化为 mc_discounted。

    MDP 语义（一局制，中间 reward=0，唯一真实 reward 是终局 z∈{+1,-1}）：
      按"该 player 视角的回合段序列"组织。设 P 的回合段为 k=0..K，各段全局 turn
      记为 τ_k（该段的 turn_index），终局全局 turn τ_T（本局最后一帧 turn_index），
      每段末尾 after-snapshot 的规则估值 V(s_k)（P 视角，终局=±1），终局 z_P。
      **折现按全局 turn_index 差**（与 mc_discounted 同一把尺子，保证 λ=1 严格退化为 mc）：
      n-step return（中间 reward=0）：
        G^{(n)}_k = γ^{(τ_{k+n} - τ_k)} · V(s_{k+n})     若 k+n < K（bootstrap 尾部）
        G^{(n)}_k = γ^{(τ_T   - τ_k)} · z_P              若 k+n ≥ K（已到终局，真实 return）
      λ-return（前向，到终局截断）：
        G^λ_k = (1-λ) Σ_{n=1}^{K-k-1} λ^{n-1} G^{(n)}_k + λ^{K-k-1} · G^{(K-k)}_k
      λ=1 时只剩末项 γ^{(τ_T-τ_k)}·z_P，即 mc_discounted；λ=0 时只剩 n=1 的 1-step TD。
      一个回合段内所有帧共享该段的 G^λ_k（同一回合视角价值相同）。

    折现以全局回合为单位（turn_index 差），与 mc_discounted 一致。
    bootstrap_value: SearchValue 实例（默认 BoardAdvantageValue，零 neural 依赖，
      与 value 去偏实验同源，保证 TD(λ) 的增益来自 n-step 混合而非 neural value 自举）。
    winner: "me"/"opponent"/None；None → 不回填（返回空）。
    """
    if not steps or winner not in ("me", "opponent"):
        return {}
    if bootstrap_value is None:
        from decision_engine.planning.search_value import BoardAdvantageValue
        bootstrap_value = BoardAdvantageValue()

    # 1. 切回合段（保序），记录每段 player 与段末 after-snapshot
    segments = []  # (player, [step,...])
    for st in steps:
        key = _turn_key(st)
        if segments and _turn_key(segments[-1][1][-1]) == key:
            segments[-1][1].append(st)
        else:
            segments.append((st.player, [st]))

    # 终局全局 turn（本局最后一帧 turn_index，与 mc_discounted 的 t_final 一致）
    t_final = max(int(st.turn_index) for st in steps)

    # 2. 为每个 player 视角建立"回合段序列"（按出现顺序），并预存段末规则估值
    #    V 以"该段 player 自身视角"求值；λ-return 对每个 player 用其自己的段序列。
    out: dict = {}
    for target_player in ("me", "opponent"):
        # 该 player 的回合段（按序）：(全局turn τ_k, 段末V, seg_steps)
        p_segments = [seg_steps for pl, seg_steps in segments if pl == target_player]
        if not p_segments:
            continue
        K = len(p_segments) - 1  # 最后一段索引
        z_p = 1.0 if winner == target_player else -1.0

        # 预存每段的全局 turn τ_k 与段末 after-snapshot 的规则估值 V(s_k)
        seg_turns = [int(seg_steps[0].turn_index) for seg_steps in p_segments]
        seg_values = []
        for seg_steps in p_segments:
            snap = _after_snapshot(seg_steps[-1])
            seg_values.append(bootstrap_value.evaluate(snap, player=target_player).value)

        # 对每段 k 计算 λ-return（折现指数用全局 turn 差 τ - τ_k）
        for k in range(len(p_segments)):
            steps_to_end = K - k  # P 还剩多少个回合段到终局
            tau_k = seg_turns[k]
            if steps_to_end <= 0:
                # P 的最后一个回合段：之后 P 不再行动，return 只剩到全局终局的折现真实 return。
                # 注意 P 最后动完未必就是全局终局（对方还要走最后一个回合），
                # 故折现指数仍用全局 turn 差 (t_final - τ_k)，与 mc_discounted 一致。
                g_lambda = (gamma ** max(0, t_final - tau_k)) * z_p
            else:
                g_lambda = 0.0
                # n=1..steps_to_end-1：bootstrap 尾部 G^{(n)} = γ^{(τ_{k+n}-τ_k)} · V(s_{k+n})
                for n in range(1, steps_to_end):
                    disc = gamma ** max(0, seg_turns[k + n] - tau_k)
                    g_n = disc * seg_values[k + n]
                    g_lambda += (1.0 - lam) * (lam ** (n - 1)) * g_n
                # n=steps_to_end：到终局的完整 return γ^{(τ_T-τ_k)}·z，权重 λ^{steps_to_end-1}
                g_terminal = (gamma ** max(0, t_final - tau_k)) * z_p
                g_lambda += (lam ** (steps_to_end - 1)) * g_terminal

            # 该段所有帧共享 g_lambda
            seg_steps = p_segments[k]
            for st in seg_steps:
                out[st.sample_id] = g_lambda
    return out


def backfill_run_result_td(result, *, gamma: float = 0.97, lam: float = 0.7,
                           bootstrap_value=None) -> dict:
    """对整个 SelfPlaySearchRunResult 做 TD(λ) 回填（所有 episode 合并）。

    bootstrap_value 默认 BoardAdvantageValue（零 neural 依赖）。
    无胜负 episode（winner=None）其样本不在返回表中，调用方保留原 value_target。
    """
    if bootstrap_value is None:
        from decision_engine.planning.search_value import BoardAdvantageValue
        bootstrap_value = BoardAdvantageValue()
    targets: dict = {}
    for ep in result.episodes:
        targets.update(compute_td_lambda_for_episode(ep.steps, ep.winner, gamma=gamma,
                                                     lam=lam, bootstrap_value=bootstrap_value))
    return targets
