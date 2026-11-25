# -*- coding: utf-8 -*-
"""Value Target：数据驱动的局面价值监督信号（4回合窗口净资源交换）。

设计见 docs/plan/VALUE_TARGET_DESIGN.md。

核心：局面价值 = 从该帧(回合A)到我方下下个回合末(C_end)的通用硬资源净交换，
经 tanh 归一。只用跨体系通用标尺（血量/EP/SEP/手牌卡差/额外PP），
一切长期性/身材/存在性/体系特有靠 value head 从网络特征学。

与 step_reward 的区别：
- step_reward 给单动作打分（含潜在价值 PP ramp/增幅），用于搜索排序。
- value_target 表达局面真实优劣（只近端通用硬资源），用于训练 value head。
两者不复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import tanh

from decision_engine.execution.rule_kernel import deckout_winner_from_results
from decision_engine.state.snapshot import BattleSnapshot


@dataclass(frozen=True)
class ValueTargetWeights:
    """value target 权重（硬锚：1 血 = 10 分）。见 VALUE_TARGET_DESIGN.md。"""
    w_life: float = 10.0    # 每点净血量交换（锚）
    w_ep: float = 20.0      # 每点净 EP 交换（1EP=2PP=20）
    w_sep: float = 30.0     # 每点净 SEP 交换（1.5×EP）
    w_hand: float = 12.0    # 每张净手牌卡差（存量，略低于抽牌流量15）
    w_xpp: float = 10.0     # 每点额外 PP 持有差（=1点PP）
    w_first: float = 10.0   # 先后手节奏基准（先手+/后手−，=1费节奏差=1PP）
                            # 推导：后手每回合面对 [N,N+1] 慢一费=−1PP；先手 [N,N] 持平。
                            # 额外PP(w_xpp)恰好抵消一回合的慢一费 → w_first==w_xpp。
    v_win: float = 200.0    # 终局（tanh 饱和到 ±1，统一正负一）
    scale: float = 90.0     # tanh 除数（数据校准：中位|NET|≈42→tanh0.44，p75≈78→tanh0.71，
                            # 终局±200→tanh±0.98。见 VALUE_TARGET_DESIGN.md scale校准节）


DEFAULT_VALUE_TARGET_WEIGHTS = ValueTargetWeights()


def _winner(snapshot: BattleSnapshot) -> str | None:
    """终局判断（与 incremental_value._winner 一致）。"""
    if snapshot.me.life <= 0 and snapshot.opponent.life <= 0:
        return "simultaneous"
    if snapshot.me.life <= 0:
        return "opponent"
    if snapshot.opponent.life <= 0:
        return "me"
    result_winner = deckout_winner_from_results(snapshot.me.result, snapshot.opponent.result)
    if result_winner is not None:
        return result_winner
    if snapshot.me.result and snapshot.opponent.result:
        return "simultaneous"
    if snapshot.me.result:
        return "me"
    if snapshot.opponent.result:
        return "opponent"
    return None


def _sides(snapshot: BattleSnapshot, root_player: str):
    """返回 (my_view, opp_view)，始终以 root_player 侧为 my。"""
    if root_player == "me":
        return snapshot.me, snapshot.opponent
    return snapshot.opponent, snapshot.me


def _extra_pp_held(view) -> int:
    """玩家当前持有的额外 PP 数（后手特权静态资源）。"""
    xp = getattr(view, "extra_pp", None)
    if xp is None:
        return 0
    return int(getattr(xp, "extra_pp", 0) or 0)


def compute_value_target_net(
    start: BattleSnapshot,
    end: BattleSnapshot,
    root_player: str,
    *,
    weights: ValueTargetWeights = DEFAULT_VALUE_TARGET_WEIGHTS,
) -> float:
    """计算窗口 [start, end] 的净资源交换分（未归一化 NET，root_player 视角）。

    只看 end vs start 的净差（不看过程掉血/回血/抽牌/打牌分别计），
    自动消除"掉血权重≠回血权重导致凭空盈亏"的矛盾。
    """
    # 终局：窗口内分出胜负 → ±v_win（tanh 后饱和到 ±1）
    winner = _winner(end)
    if winner is not None:
        if winner == "simultaneous":
            return 0.0
        return weights.v_win if winner == root_player else -weights.v_win

    my_s, opp_s = _sides(start, root_player)
    my_e, opp_e = _sides(end, root_player)

    net = 0.0

    # 1. 净血量交换：(对方掉的血) − (我方掉的血)。掉血 = start − end（可负=回血）。
    opp_life_lost = opp_s.life - opp_e.life
    my_life_lost = my_s.life - my_e.life
    net += weights.w_life * (opp_life_lost - my_life_lost)

    # 2. 净 EP 交换：对方消耗 − 我方消耗（对方消耗=对方赚不到，我方赚）。
    opp_ep_used = opp_s.ep - opp_e.ep
    my_ep_used = my_s.ep - my_e.ep
    net += weights.w_ep * (opp_ep_used - my_ep_used)

    # 3. 净 SEP 交换
    opp_sep_used = opp_s.sep - opp_e.sep
    my_sep_used = my_s.sep - my_e.sep
    net += weights.w_sep * (opp_sep_used - my_sep_used)

    # 4. 净手牌卡差（存量）：(end 我方手牌−对方手牌) − (start 我方手牌−对方手牌)
    hand_diff_end = len(my_e.hand) - len(opp_e.hand)
    hand_diff_start = len(my_s.hand) - len(opp_s.hand)
    net += weights.w_hand * (hand_diff_end - hand_diff_start)

    # 5. 额外 PP 持有差（end 时刻，后手关键静态资源）
    net += weights.w_xpp * (_extra_pp_held(my_e) - _extra_pp_held(opp_e))

    # 6. 先后手节奏基准（静态身份，用 start 帧 root 侧 is_first_side，整局不变）：
    #    先手 [N,N] 节奏持平 → +w_first；后手 [N,N+1] 慢一费 → −w_first。
    #    与额外PP项互补：后手持额外PP(临时反转[N+1,N+1])时 +w_xpp 恰好抵消 −w_first → NET=0。
    if getattr(my_s, "is_first_side", False):
        net += weights.w_first
    else:
        net -= weights.w_first

    return net


def compute_value_target(
    start: BattleSnapshot,
    end: BattleSnapshot,
    root_player: str,
    *,
    weights: ValueTargetWeights = DEFAULT_VALUE_TARGET_WEIGHTS,
) -> float:
    """value target ∈ (-1, 1)：tanh(NET / scale)。

    终局 NET 含 ±v_win → tanh 饱和到 ±1（与终局胜负统一）；
    非终局 NET 是资源优势 → tanh 落在 (-1, 1)，保留优势程度区分。
    """
    net = compute_value_target_net(start, end, root_player, weights=weights)
    return tanh(net / weights.scale)
