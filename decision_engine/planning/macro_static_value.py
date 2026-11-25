# -*- coding: utf-8 -*-
"""P4 静态资源 Macro evaluator（Static Resource Macro Evaluator）。

独立模块，不依赖也不修改 macro_turn_search.py / puct.py / incremental_value.py /
strict_self_play.py。提供一个纯静态、可配置、可注入的宏观局面评估：把「当前
snapshot 的资源存量」折算成一个标量 StaticScore，root-relative 宏观价值为双方
StaticScore 之差。

设计要点（合同）：

1. 权重集中在 `MacroStaticWeights`，全部可配置：
   - HP=10、MaxPP=10、EP=20、SEP=30（每单位分值）
   - 终局值=1000（压倒任何中间分累加）
   - board residue 默认 0（保留可注入接口）
2. 手牌用「边际递减表」而非线性：第 k 张手牌（1-based）的边际分见
   `DEFAULT_HAND_MARGINAL`=(15,15,12,9,7,5,3,1,0)。第 9 张及以后边际为 0。
3. root-relative 宏观价值 = my.StaticScore - opp.StaticScore；`player` 支持
   "me" / "opponent" 并严格对称（swap 双方 → 取负）。
4. Extra PP 只依赖本地快照生命周期事实（是否持有 token、本回合是否已用、是否先手），
   **不从 is_used 反推「已消费」的资源价值**；「能否兑现成 +1PP 的选择权」由可注入
   `ExtraPpRedeemablePredicate` 决定，默认不做任何猜测（返回 False → 只按持有事实给
   一个 EXPIRING/HOLD 分级的保守 option 值）。T5 后手持有的、即将在 T6 被覆盖的
   token 判定为 EXPIRING，其 option 值严格低于普通 HOLD。
5. 纹章价值用可注入 `CrestValuationProvider`，默认不猜（返回 0）。另提供一个
   **通用转换估值 provider**：只依据「已知的 source_pp / source_ep / source_sep 成本」
   把纹章折算成费用当量分值，不写任何卡名/卡 id 硬编码。
6. 终局：任一方 life<=0 或 deckout 结果 → 返回 ±v_win，量纲上压倒中间分。
7. board residue：默认 0；`BoardValuationProvider` 可注入以引入场面残值。

本模块不读取协议 Can*/Select* 字段，只用本地可观察 snapshot 状态推导。
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Callable, Optional, Protocol, Sequence

from decision_engine.state.snapshot import BattleSnapshot, PlayerView


# ---------------------------------------------------------------------------
# 权重与常量
# ---------------------------------------------------------------------------

# 手牌边际递减表（1-based：index 0 = 第 1 张的边际分）。
# 第 9 张（index 8）及以后边际为 0：手牌上限附近再加牌不产生宏观价值。
DEFAULT_HAND_MARGINAL: tuple[int, ...] = (15, 15, 12, 9, 7, 5, 3, 1, 0)

# T5 后手第一个 extra_pp 在 T6 发放时会被覆盖（EXTRA_PP_CAP=1），
# 因此 T5 后手持有的、尚未使用的 token 视为 EXPIRING。此常量仅用于本模块内部
# 分级判定，与其它模块无耦合。
_FIRST_EXTRA_PP_EXPIRY_TURN: int = 5


@dataclass(frozen=True)
class MacroStaticWeights:
    """静态宏观评估权重（全部可配置）。

    量纲锚：1 点 HP = 10 分（与 incremental_value 的 1费=10分 尺度一致，便于并用）。
    """

    w_hp: float = 10.0        # 每点当前 life
    w_max_pp: float = 10.0    # 每点 max_pp（长期资源上限）
    w_ep: float = 20.0        # 每点 EP
    w_sep: float = 30.0       # 每点 SEP
    v_win: float = 1000.0     # 终局胜负（压倒任何中间分累加）

    # 手牌边际表；默认见 DEFAULT_HAND_MARGINAL。
    hand_marginal: tuple[int, ...] = DEFAULT_HAND_MARGINAL

    # Extra PP option 保守值：HOLD（普通持有，未来可能兑现）与 EXPIRING
    # （即将被覆盖，价值更低）。默认 HOLD=5（半费）、EXPIRING=1（远低于 HOLD）。
    # 满足可兑现 predicate 时使用 REDEEMABLE（更高，但仍是期权，默认 8 < 一费 10）。
    w_extra_pp_hold: float = 5.0
    w_extra_pp_expiring: float = 1.0
    w_extra_pp_redeemable: float = 8.0

    # board residue 默认 0（保留可注入 provider 引入）。
    w_board_residue: float = 0.0


DEFAULT_WEIGHTS = MacroStaticWeights()


# ---------------------------------------------------------------------------
# Extra PP option 分级
# ---------------------------------------------------------------------------


class ExtraPpStatus(str, Enum):
    """本地快照生命周期事实推导出的 extra_pp option 分级。"""

    NONE = "none"              # 不持有 / 先手 / 无 token → 无 option
    EXPIRING = "expiring"      # T5 后手持有、即将被覆盖 → 最低 option 值
    HOLD = "hold"              # 普通持有、未来可能保留 → 普通 option 值
    REDEEMABLE = "redeemable"  # 可注入 predicate 判定「本回合/下次行动可兑现」→ 最高 option 值


# predicate(snapshot, view) -> bool：该方持有的 token 是否「可兑现成 +1PP 解锁某手牌」。
# 默认 predicate 永远返回 False（不猜测）。
ExtraPpRedeemablePredicate = Callable[[BattleSnapshot, PlayerView], bool]


def default_extra_pp_redeemable(_snapshot: BattleSnapshot, _view: PlayerView) -> bool:
    """默认可兑现 predicate：不做任何猜测，一律 False。"""
    return False


def classify_extra_pp(
    snapshot: BattleSnapshot,
    view: PlayerView,
    *,
    redeemable: ExtraPpRedeemablePredicate = default_extra_pp_redeemable,
) -> ExtraPpStatus:
    """从本地快照生命周期事实推导 extra_pp option 分级。

    仅依赖：is_first_side、extra_pp token 是否 >0、is_used_extra_pp_this_turn、turn。
    **不从 is_used 反推已消费资源的价值**——is_used 只用于判定「本回合已用掉 →
    不再是未来 option（NONE）」，而不是把已消费的 PP 折算成分值。

    分级顺序：
    - 先手 / 无 token / 本回合已用 → NONE
    - 否则若 redeemable predicate 判定可兑现 → REDEEMABLE
    - 否则若 T5(turn>=5) 后手持有（即将被覆盖）→ EXPIRING
    - 否则 → HOLD
    """
    if view.is_first_side:
        return ExtraPpStatus.NONE
    extra = view.extra_pp
    if extra is None or int(extra.extra_pp) <= 0:
        return ExtraPpStatus.NONE
    if bool(extra.is_used_extra_pp_this_turn):
        # 已用掉：不再是未来选择权。不把「已消费」折算成价值。
        return ExtraPpStatus.NONE
    if redeemable(snapshot, view):
        return ExtraPpStatus.REDEEMABLE
    if int(view.turn) >= _FIRST_EXTRA_PP_EXPIRY_TURN:
        return ExtraPpStatus.EXPIRING
    return ExtraPpStatus.HOLD


def _extra_pp_option_value(status: ExtraPpStatus, weights: MacroStaticWeights) -> float:
    if status is ExtraPpStatus.REDEEMABLE:
        return weights.w_extra_pp_redeemable
    if status is ExtraPpStatus.HOLD:
        return weights.w_extra_pp_hold
    if status is ExtraPpStatus.EXPIRING:
        return weights.w_extra_pp_expiring
    return 0.0


# ---------------------------------------------------------------------------
# 纹章估值 provider（可注入，默认不猜）
# ---------------------------------------------------------------------------


class CrestValuationProvider(Protocol):
    """把一个玩家的纹章区折算成分值。默认实现返回 0（不猜）。"""

    def __call__(self, snapshot: BattleSnapshot, view: PlayerView) -> float: ...


def zero_crest_valuation(_snapshot: BattleSnapshot, _view: PlayerView) -> float:
    """默认纹章估值：不做任何猜测，一律 0。"""
    return 0.0


@dataclass(frozen=True)
class CrestSourceCost:
    """一张纹章的「已知来源成本」事实（不含卡名/卡 id 语义）。

    只描述「获得这个纹章花了多少资源」：source_pp（一次性 PP 费用）、
    source_ep / source_sep（进化/超进化点消耗）。由调用方从可靠数据填充，
    本模块不推断、不硬编码任何卡。
    """

    source_pp: float = 0.0
    source_ep: float = 0.0
    source_sep: float = 0.0


class GenericCrestConversionValuation:
    """通用纹章转换估值 provider：按已知 source 成本折算费用当量分值。

    折算规则（费用当量，与 MacroStaticWeights 同尺度）：
        value = source_pp * w_pp_unit + source_ep * w_ep + source_sep * w_sep

    其中 w_pp_unit 默认取「1 PP = 1 HP = w_hp 分」（10 分），w_ep / w_sep 直接
    复用权重。不写任何卡名硬编码——纹章到 CrestSourceCost 的映射由调用方通过
    `cost_lookup` 注入（key 为纹章的 card_id，value 为 CrestSourceCost）。
    未在表中的纹章记 0（不猜）。
    """

    def __init__(
        self,
        cost_lookup: dict[int, CrestSourceCost],
        *,
        pp_unit_value: float | None = None,
    ) -> None:
        self._cost_lookup = dict(cost_lookup)
        self._pp_unit_value = pp_unit_value

    def __call__(self, snapshot: BattleSnapshot, view: PlayerView) -> float:
        weights = _weights_from_context(snapshot)
        pp_unit = self._pp_unit_value if self._pp_unit_value is not None else weights.w_hp
        total = 0.0
        for crest in getattr(view, "crests", ()) or ():
            crest_id = int(getattr(crest, "card_id", 0) or 0)
            cost = self._cost_lookup.get(crest_id)
            if cost is None:
                continue
            total += cost.source_pp * pp_unit
            total += cost.source_ep * weights.w_ep
            total += cost.source_sep * weights.w_sep
        return total


# GenericCrestConversionValuation 需要权重来折算；权重不是全局单例，
# 通过一个轻量 thread-unsafe 但足够（同步单线程评估）的上下文变量传递。
_ACTIVE_WEIGHTS: MacroStaticWeights = DEFAULT_WEIGHTS


def _weights_from_context(_snapshot: BattleSnapshot) -> MacroStaticWeights:
    return _ACTIVE_WEIGHTS


# ---------------------------------------------------------------------------
# board residue provider（可注入，默认 0）
# ---------------------------------------------------------------------------


class BoardValuationProvider(Protocol):
    """把一个玩家的场面折算成残值分值。默认实现返回 0。"""

    def __call__(self, snapshot: BattleSnapshot, view: PlayerView) -> float: ...


def zero_board_valuation(_snapshot: BattleSnapshot, _view: PlayerView) -> float:
    """默认 board residue：0。"""
    return 0.0


# ---------------------------------------------------------------------------
# 终局判定
# ---------------------------------------------------------------------------


def _winner(snapshot: BattleSnapshot) -> Optional[str]:
    """终局判断（绝对方："me"/"opponent"/"simultaneous"/None）。

    与 incremental_value._winner / search_value._winner 语义一致，但本模块自持一份
    以保持独立（不 import 那些文件的私有函数）。
    """
    from decision_engine.execution.rule_kernel import deckout_winner_from_results

    me_dead = snapshot.me.life <= 0
    opp_dead = snapshot.opponent.life <= 0
    if me_dead and opp_dead:
        return "simultaneous"
    if me_dead:
        return "opponent"
    if opp_dead:
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


# ---------------------------------------------------------------------------
# 单侧 StaticScore
# ---------------------------------------------------------------------------


def _hand_score(hand_size: int, marginal: Sequence[int]) -> float:
    """按边际递减表累加前 hand_size 张手牌的边际分。

    超出表长度的手牌按最后一项的边际（默认 0）累加，因此第 9 张及以后不再加分。
    """
    total = 0.0
    if hand_size <= 0 or not marginal:
        return total
    last = marginal[-1]
    for i in range(hand_size):
        total += marginal[i] if i < len(marginal) else last
    return total


def static_side_score(
    snapshot: BattleSnapshot,
    view: PlayerView,
    *,
    weights: MacroStaticWeights = DEFAULT_WEIGHTS,
    crest_valuation: CrestValuationProvider = zero_crest_valuation,
    board_valuation: BoardValuationProvider = zero_board_valuation,
    extra_pp_redeemable: ExtraPpRedeemablePredicate = default_extra_pp_redeemable,
) -> float:
    """单侧静态资源分（不含终局；终局在 root-relative 层短路）。

    = HP*w_hp + max_pp*w_max_pp + EP*w_ep + SEP*w_sep
      + hand_marginal(hand_size)
      + extra_pp option 值（分级）
      + crest_valuation（默认 0）
      + board_residue*w_board_residue（默认 0）
    """
    global _ACTIVE_WEIGHTS
    prev_weights = _ACTIVE_WEIGHTS
    _ACTIVE_WEIGHTS = weights
    try:
        score = 0.0
        score += max(0, int(view.life)) * weights.w_hp
        score += int(view.max_pp) * weights.w_max_pp
        score += int(view.ep) * weights.w_ep
        score += int(view.sep) * weights.w_sep
        score += _hand_score(len(view.hand), weights.hand_marginal)

        status = classify_extra_pp(snapshot, view, redeemable=extra_pp_redeemable)
        score += _extra_pp_option_value(status, weights)

        score += crest_valuation(snapshot, view)
        score += board_valuation(snapshot, view) * weights.w_board_residue
        return score
    finally:
        _ACTIVE_WEIGHTS = prev_weights


# ---------------------------------------------------------------------------
# root-relative 宏观价值
# ---------------------------------------------------------------------------


def _sides(snapshot: BattleSnapshot, player: str) -> tuple[PlayerView, PlayerView]:
    """返回 (my_view, opp_view)，以 player 侧为 my。"""
    if player == "me":
        return snapshot.me, snapshot.opponent
    if player == "opponent":
        return snapshot.opponent, snapshot.me
    raise ValueError(f"player 必须是 'me' 或 'opponent'，收到 {player!r}")


def macro_static_value(
    snapshot: BattleSnapshot,
    player: str = "me",
    *,
    weights: MacroStaticWeights = DEFAULT_WEIGHTS,
    crest_valuation: CrestValuationProvider = zero_crest_valuation,
    board_valuation: BoardValuationProvider = zero_board_valuation,
    extra_pp_redeemable: ExtraPpRedeemablePredicate = default_extra_pp_redeemable,
) -> float:
    """root-relative 宏观静态价值（player 视角）。

    终局短路：任一方 life<=0 / deckout 结果 → 返回 ±v_win（压倒任何中间分）。
    否则 = my.StaticScore - opp.StaticScore。

    对称性：macro_static_value(s, "me") == -macro_static_value(s, "opponent")。
    """
    winner = _winner(snapshot)
    if winner is not None:
        if winner == "simultaneous":
            return 0.0
        return weights.v_win if winner == player else -weights.v_win

    my_view, opp_view = _sides(snapshot, player)
    my_score = static_side_score(
        snapshot, my_view,
        weights=weights, crest_valuation=crest_valuation,
        board_valuation=board_valuation, extra_pp_redeemable=extra_pp_redeemable,
    )
    opp_score = static_side_score(
        snapshot, opp_view,
        weights=weights, crest_valuation=crest_valuation,
        board_valuation=board_valuation, extra_pp_redeemable=extra_pp_redeemable,
    )
    return my_score - opp_score


__all__ = [
    "DEFAULT_HAND_MARGINAL",
    "MacroStaticWeights",
    "DEFAULT_WEIGHTS",
    "ExtraPpStatus",
    "ExtraPpRedeemablePredicate",
    "default_extra_pp_redeemable",
    "classify_extra_pp",
    "CrestValuationProvider",
    "zero_crest_valuation",
    "CrestSourceCost",
    "GenericCrestConversionValuation",
    "BoardValuationProvider",
    "zero_board_valuation",
    "static_side_score",
    "macro_static_value",
]
