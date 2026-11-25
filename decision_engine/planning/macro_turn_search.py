# -*- coding: utf-8 -*-
"""分层跨回合搜索（Hierarchical Multi-Turn Search）。

宏观回合树 + 微观动作树的分层前瞻搜索。设计见
docs/plan/HIERARCHICAL_MULTITURN_SEARCH_PLAN.md。

核心结构：
- 微观层：现有 PUCT（max_turns=1，只搜本回合，无奇偶效应），产出 top-K 回合线。
- 宏观层：把每条回合线的结束状态切到对手回合，递归微观搜索到 MACRO_DEPTH，
  minimax backup 把多回合价值回传当前回合。

本文件按 M0~M4 分阶段构建：
- M0（本段）：从微观 PUCT 树提取"到 turn_end 的完整回合线"。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from decision_engine.actions.model import LegalAction
from decision_engine.state.snapshot import BattleSnapshot

if TYPE_CHECKING:
    from decision_engine.planning.puct import _PuctNode, PuctSearch


def _callable_accepts_kwarg(fn, name: str) -> bool:
    """显式探测可调用对象是否接受指定关键字参数（E3 residual_fn 适配用）。

    通过 inspect 签名判断，而非 try/except 吞异常：
    - 显式声明该关键字参数，或
    - 存在 **kwargs（VAR_KEYWORD）。
    fn 为 None 时返回 False。
    """
    if fn is None:
        return False
    import inspect

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# E2: Extra PP 未来签名
# ---------------------------------------------------------------------------
#
# 目的：同一"剩余手牌"下，Extra PP token 的未来处置（花掉 / 保留 / 即将过期）
# 是真实的战略分歧，不应被 dedup_by_hand 折叠为同一条线。
#
# 语义（只从本地可观察状态推导，不依赖协议 Can* 字段）：
# - NONE：先手方（不获得 extra_pp），或回合开始时未持有 token → 无未来差异。
# - SPENT：回合开始持有 token，且到 turn_end after-state 时 held token 递减 → 实际花掉。
# - HOLD：回合开始持有 token，到 turn_end after-state 时 token 未递减 → 保留到下回合。
#         注意：回合内瞬态的 "activated-but-not-spent"（is_used_extra_pp_this_turn=True
#         但 held 数量未变）必须规范化为 HOLD，不能把瞬态 is_used 当作未来差异。
# - EXPIRING：HOLD 语义，但发生在 T5 后手——此时持有的第一个 token 会在 T6 发放时
#             被覆盖（EXTRA_PP_CAP=1），未花即浪费。与普通 HOLD 语义不同，单独分桶。
# - UNKNOWN：没有可观察的 turn-end after-state（缺少 turn_end_edge post-snapshot），
#            无法判定 token 是否被消耗。此时不得伪称与任何确定状态相同，保守分离。
#
# 参考 rule_kernel：
# - EXTRA_PP_GRANT_TURNS = {1, 6}，EXTRA_PP_CAP = 1（held token 上限 1）。
# - turn_start：后手方 held = min(held + grant, CAP)，覆盖旧 token。
# - 实际消耗：_pay_pp_transactions 在真正用到 extra_pp 时把 extra_pp.extra_pp 递减 1。

EXTRA_PP_SIG_NONE = "NONE"
EXTRA_PP_SIG_SPENT = "SPENT"
EXTRA_PP_SIG_HOLD = "HOLD"
EXTRA_PP_SIG_EXPIRING = "EXPIRING"
EXTRA_PP_SIG_UNKNOWN = "UNKNOWN"

# T5 后手持有的第一个 extra_pp 在 T6 发放时被覆盖（与 extra_pp_enumerator 保持一致）。
_FIRST_EXTRA_PP_EXPIRY_TURN = 5


def _held_extra_pp(snapshot: BattleSnapshot, player: str) -> int:
    """某玩家在给定局面下持有的 extra_pp token 数量（0 表示无 token）。"""
    pv = snapshot.me if player == "me" else snapshot.opponent
    extra = pv.extra_pp
    if extra is None:
        return 0
    return max(int(extra.extra_pp), 0)


def extra_pp_future_signature(
    turn_start: BattleSnapshot,
    player: str,
    turn_end_edge: tuple | None,
) -> str | None:
    """推导一条回合线的 Extra PP 未来签名。

    turn_start：回合起始局面（微观搜索 root.snapshot），用于确定回合开始时持有的 token。
    turn_end_edge：(pre_turn_end_snapshot, turn_end_action, post_turn_end_snapshot) 或 None。
        post-snapshot 是 turn_end 结算后的 after-state，用于判定 token 是否被实际消耗。

    Returns:
        E2 签名字符串（NONE/SPENT/HOLD/EXPIRING/UNKNOWN），
        或 None（默认兼容：无法从可观察状态推导时不写签名，交由调用方保守分离）。
    """
    pv = turn_start.me if player == "me" else turn_start.opponent
    # 先手方不获得 extra_pp；即使协议给了 extra_pp=0 的对象也视为 NONE。
    if pv.is_first_side:
        return EXTRA_PP_SIG_NONE

    held_at_start = _held_extra_pp(turn_start, player)
    if held_at_start <= 0:
        # 回合开始无 token → 无未来差异。
        return EXTRA_PP_SIG_NONE

    # 有 token：需要 turn-end after-state 才能判定花掉 / 保留。
    if turn_end_edge is None:
        return EXTRA_PP_SIG_UNKNOWN
    post = turn_end_edge[2] if len(turn_end_edge) >= 3 else None
    if post is None:
        return EXTRA_PP_SIG_UNKNOWN

    held_after = _held_extra_pp(post, player)
    if held_after < held_at_start:
        # held token 递减 → 实际花掉（不受回合内瞬态 is_used 影响）。
        return EXTRA_PP_SIG_SPENT

    # held 未递减 → 保留（含 activated-but-not-spent 瞬态，规范化为 HOLD）。
    is_expiring = pv.turn == _FIRST_EXTRA_PP_EXPIRY_TURN and not pv.is_first_side
    return EXTRA_PP_SIG_EXPIRING if is_expiring else EXTRA_PP_SIG_HOLD


# ---------------------------------------------------------------------------
# 回合线数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnLine:
    """一条完整回合线：从回合起始局面，打出一串动作，到 turn_end 前的局面。

    Attributes:
        action_sequence: 本回合按顺序打出的动作（不含 turn_end 本身）。
        end_snapshot: 打完 action_sequence、正要 turn_end 时的局面（turn_end 前）。
        remaining_hand_key: 去重键 = 结束时我方剩余手牌 base_card_id 多重集（排序元组）。
        active_player: 该回合的行动方。
        turn_number: 该回合的逻辑回合序号。
        visit_count: 微观搜索中这条线（turn_end 边）的访问次数（供诊断/排序参考）。
        diff_score: 该回合线的价值评分（M1=DiffSearchValue；S2 后=step_reward 累加）。
        edges: 从回合起始到 end_snapshot 的逐边序列 (prev_snapshot, action, next_snapshot)，
               不含 turn_end；供逐动作 step_reward 累加（S2）。空 tuple 表示未收集。
        turn_end_edge: 可选的 (pre_turn_end_snapshot, turn_end, post_turn_end_snapshot)；
               用于将自动 turn_end 效果纳入评分，不改变 end_snapshot/bucket 的pre-turn-end语义。
        extra_pp_signature: E2 Extra PP 未来签名（NONE/SPENT/HOLD/EXPIRING/UNKNOWN）或 None。
               默认 None（兼容旧构造）；由 extract_turn_lines 在存在 turn_end after-state
               时物化。dedup_by_hand 的去重键包含该签名，保证同剩余手牌下 SPENT/HOLD/
               EXPIRING 不相互覆盖；None/UNKNOWN 视为不同状态保守分离，不伪称相同。
    """

    action_sequence: tuple[LegalAction, ...]
    end_snapshot: BattleSnapshot
    remaining_hand_key: tuple[int, ...]
    active_player: str
    turn_number: int
    visit_count: int = 0
    diff_score: float = 0.0
    edges: tuple = ()
    turn_end_edge: tuple | None = None
    extra_pp_signature: str | None = None


def remaining_hand_key(snapshot: BattleSnapshot, player: str) -> tuple[int, ...]:
    """计算某玩家在给定局面下的剩余手牌 base_card_id 多重集（排序元组）。

    这是回合线去重键：打出的牌相同（剩余手牌相同）= 同一战略的不同执行。
    """
    ps = snapshot.me if player == "me" else snapshot.opponent
    ids: list[int] = []
    for uid in ps.hand:
        card = snapshot.cards_by_uid.get(uid)
        if card is not None:
            ids.append(int(card.base_card_id))
        else:
            # 手牌 uid 无对应 CardView（隐藏牌等），用 uid 占位保证键稳定
            ids.append(-int(uid))
    return tuple(sorted(ids))


# ---------------------------------------------------------------------------
# M0: 从微观 PUCT 树提取回合线
# ---------------------------------------------------------------------------


def _is_turn_end(action: LegalAction) -> bool:
    kind = action.kind.value if hasattr(action.kind, "value") else str(action.kind)
    return kind == "turn_end"


def extract_turn_lines(root: "_PuctNode") -> list[TurnLine]:
    """从微观 PUCT 树（root 为当前回合决策节点）提取所有到达 turn_end 的回合线。

    遍历策略：DFS 已展开的树。对每个仍属于本回合（active_player/turn_number 与 root 相同、
    非终局/深度叶）的节点，检查其每条边：
    - 若该边是 turn_end 动作 → 一条回合线终结。end_snapshot = 该节点的 snapshot
      （已打完本回合所有非 turn_end 动作，正要 turn_end）。action_sequence = root→此节点路径。
    - 若该边有已展开的 child 且 child 仍属本回合 → 继续 DFS。
      （跨回合 child：active_player 变化，由 _handle_turn_end 产生——不再深入，本回合线止于此。）

    注意：PUCT 树是稀疏的（只含被搜索访问过的分支），因此提取的是搜索认为值得探索的线，
    不是全枚举。探索保证（让延迟收益线 D 被访问）由 M3 处理。

    Returns:
        回合线列表（可能含 remaining_hand_key 重复项，去重在 M1）。
    """
    root_player = root.active_player
    root_turn = root.turn_number
    turn_start_snapshot = root.snapshot
    lines: list[TurnLine] = []
    seen_nodes: set[int] = set()

    def dfs(node: "_PuctNode", path: tuple[LegalAction, ...], edges: tuple) -> None:
        # 防环（树本应无环，防御）
        if id(node) in seen_nodes:
            return
        seen_nodes.add(id(node))

        # 只在本回合、非终局、非深度叶的节点上找 turn_end 边
        if node.is_terminal or node.is_depth_leaf:
            return
        if node.active_player != root_player or node.turn_number != root_turn:
            return

        for i, action in enumerate(node.actions):
            if _is_turn_end(action):
                # 一条回合线：end_snapshot = 本节点（turn_end 前）
                # 仅当这条边被访问过（visit_count>0）才算搜索探索过的线
                vc = node.visit_counts[i]
                if vc <= 0:
                    continue
                key = remaining_hand_key(node.snapshot, root_player)
                turn_end_edge = (
                    (node.snapshot, action, node.children[i].snapshot)
                    if node.children[i] is not None else None
                )
                # E2：仅在存在 turn_end after-state（child post-snapshot）时能判定
                # 花掉/保留；否则 extra_pp_future_signature 返回 UNKNOWN 或 None，
                # 保守分离，不伪称与确定状态相同。
                signature = extra_pp_future_signature(
                    turn_start_snapshot, root_player, turn_end_edge
                )
                lines.append(
                    TurnLine(
                        action_sequence=path,
                        end_snapshot=node.snapshot,
                        remaining_hand_key=key,
                        active_player=root_player,
                        turn_number=root_turn,
                        visit_count=vc,
                        edges=edges,
                        turn_end_edge=turn_end_edge,
                        extra_pp_signature=signature,
                    )
                )
                continue
            child = node.children[i]
            if child is None:
                continue
            # 只沿本回合内的边深入（跨回合 child 的 active_player 会变）
            if child.active_player != root_player or child.turn_number != root_turn:
                continue
            # 收集这条边 (prev_snapshot, action, next_snapshot) 供逐动作累加
            edge = (node.snapshot, action, child.snapshot)
            dfs(child, path + (action,), edges + (edge,))

    dfs(root, (), ())
    return lines


# ---------------------------------------------------------------------------
# M1: 回合线评分、去重、top-K
# ---------------------------------------------------------------------------


def score_turn_lines(
    lines: list[TurnLine],
    turn_start: BattleSnapshot,
    player: str,
    *,
    use_incremental: bool = False,
    weights=None,
    threshold_fn=None,
) -> list[TurnLine]:
    """给每条回合线打分。

    turn_start = 回合起始局面（微观搜索的 root.snapshot）。
    line.end_snapshot = 打完动作、正要 turn_end 的局面。

    use_incremental=False（默认，旧行为）：DiffSearchValue 首尾差 compute_diff_score.normalized。
    use_incremental=True（逐动作即时收益）：沿 line.edges 与可用的 line.turn_end_edge
    累加 step_reward（不丢过程损失与自动回合结束效果）；若两者皆缺失，回退首尾差近似。
    新旧并存、可配置切换（满足可回滚要求）。

    返回带 diff_score 的新回合线列表（不修改入参）。
    """
    if use_incremental:
        from decision_engine.planning.incremental_value import (
            accumulate_path_reward, DEFAULT_WEIGHTS, default_threshold_fn,
        )
        from decision_engine.planning.diff_value import compute_diff_score
        w = weights or DEFAULT_WEIGHTS
        tfn = threshold_fn if threshold_fn is not None else default_threshold_fn()
        scored: list[TurnLine] = []
        for ln in lines:
            scoring_edges = ln.edges + ((ln.turn_end_edge,) if ln.turn_end_edge is not None else ())
            if scoring_edges:
                score = accumulate_path_reward(scoring_edges, player, weights=w, threshold_fn=tfn)
            else:
                # 无边序列（防御）：回退首尾差
                score = compute_diff_score(turn_start, ln.end_snapshot, player=player).normalized
            scored.append(
                TurnLine(
                    action_sequence=ln.action_sequence, end_snapshot=ln.end_snapshot,
                    remaining_hand_key=ln.remaining_hand_key, active_player=ln.active_player,
                    turn_number=ln.turn_number, visit_count=ln.visit_count,
                    diff_score=score, edges=ln.edges, turn_end_edge=ln.turn_end_edge,
                    extra_pp_signature=ln.extra_pp_signature,
                )
            )
        return scored

    from decision_engine.planning.diff_value import compute_diff_score

    scored: list[TurnLine] = []
    for ln in lines:
        diff = compute_diff_score(turn_start, ln.end_snapshot, player=player)
        scored.append(
            TurnLine(
                action_sequence=ln.action_sequence,
                end_snapshot=ln.end_snapshot,
                remaining_hand_key=ln.remaining_hand_key,
                active_player=ln.active_player,
                turn_number=ln.turn_number,
                visit_count=ln.visit_count,
                diff_score=diff.normalized,
                edges=ln.edges,
                turn_end_edge=ln.turn_end_edge,
                extra_pp_signature=ln.extra_pp_signature,
            )
        )
    return scored


def _dedup_key(line: TurnLine) -> tuple:
    """去重键 = 剩余手牌 + Extra PP 未来签名。

    E2：同一剩余手牌下，Extra PP 的未来处置（SPENT/HOLD/EXPIRING）是真实战略分歧，
    必须分到不同桶，不能相互覆盖。

    两类"未确定"分量的处理不同：
    - signature is None：签名未被物化（默认兼容旧构造，如未经 extract_turn_lines
      的线）。此时退回旧行为，仅按剩余手牌合并（不因缺失签名而人为拆分）。
    - signature == UNKNOWN：已计算但无可观察 turn-end after-state，无法判定 token
      是否被消耗。不得伪称与任何确定状态相同，用回合线自身 id 作签名分量，
      保证每条这样的线独占一个桶（保守分离，绝不与 SPENT/HOLD/EXPIRING 或彼此合并）。
    """
    sig = line.extra_pp_signature
    if sig is None:
        # 未物化签名：兼容旧行为，仅按剩余手牌合并。
        return (line.remaining_hand_key, None)
    if sig == EXTRA_PP_SIG_UNKNOWN:
        # 不可判定：保守分离，永不与其它线合并。
        return (line.remaining_hand_key, EXTRA_PP_SIG_UNKNOWN, id(line))
    return (line.remaining_hand_key, sig)


def dedup_by_hand(lines: list[TurnLine]) -> list[TurnLine]:
    """按剩余手牌键 + Extra PP 未来签名去重。

    同键（打出的牌相同 且 Extra PP 未来处置相同 = 同一战略）只保留 DiffSearchValue
    最高的一条。出牌/攻击顺序、进化对象、爆能与否等执行差异由 diff_score 合并。

    E2：Extra PP 的 SPENT/HOLD/EXPIRING 是不同的未来状态，即使剩余手牌相同也分开保留。
    无可观察 turn-end after-state（None/UNKNOWN）的线保守分离，不伪称与确定状态相同。
    """
    best: dict[tuple, TurnLine] = {}
    for ln in lines:
        k = _dedup_key(ln)
        cur = best.get(k)
        if cur is None or ln.diff_score > cur.diff_score:
            best[k] = ln
        elif ln.diff_score == cur.diff_score and ln.visit_count > cur.visit_count:
            # 分数相同时用访问次数破平（搜索更信任的执行）
            best[k] = ln
    return list(best.values())


def select_extra_pp_signature_portfolio(lines: list[TurnLine], k: int) -> list[TurnLine]:
    """在固定 ``k`` 内保留不同确定 Extra PP future signature 的最高分代表。

    只使用 TurnLine 的本地 ``extra_pp_signature``、``diff_score`` 与稳定输入顺序，
    不读取 human action、runtime after-context 或卡名。每个确定签名
    （NONE/SPENT/HOLD/EXPIRING）先保留最高分代表；UNKNOWN 缺少可观察 turn-end
    after-state，不获得强制名额；余位按原全局分数顺序填充。
    """
    if k <= 0:
        return []
    ranked = sorted(lines, key=lambda line: line.diff_score, reverse=True)
    best_by_signature: dict[str | None, TurnLine] = {}
    for line in ranked:
        signature = line.extra_pp_signature
        if signature == EXTRA_PP_SIG_UNKNOWN:
            continue
        if signature not in best_by_signature:
            best_by_signature[signature] = line
    selected = sorted(best_by_signature.values(), key=lambda line: line.diff_score, reverse=True)[:k]
    selected_ids = {id(line) for line in selected}
    for line in ranked:
        if len(selected) >= k:
            break
        if id(line) not in selected_ids:
            selected.append(line)
            selected_ids.add(id(line))
    return selected


def top_k_turn_lines(
    root: "_PuctNode",
    turn_start: BattleSnapshot,
    player: str,
    k: int = 3,
    *,
    use_incremental: bool = False,
    weights=None,
    threshold_fn=None,
    extra_pp_signature_portfolio: bool = False,
) -> list[TurnLine]:
    """完整 M0+M1 流程：提取、评分、资源状态去重并选择 top-K 完整回合线。"""
    lines = extract_turn_lines(root)
    scored = score_turn_lines(lines, turn_start, player, use_incremental=use_incremental,
                              weights=weights, threshold_fn=threshold_fn)
    deduped = dedup_by_hand(scored)
    deduped.sort(key=lambda ln: ln.diff_score, reverse=True)
    return (
        select_extra_pp_signature_portfolio(deduped, k)
        if extra_pp_signature_portfolio
        else deduped[:k]
    )


# ---------------------------------------------------------------------------
# P1：可恢复的完整回合 micro completion session
# ---------------------------------------------------------------------------
#
# 目标（见 HIERARCHICAL_MULTITURN_SEARCH_PLAN v17 待办）：
# - Macro edge 必须是 strict 的完整 TurnLine（起始局面 → 一串动作 → turn_end）。
# - 微观搜索对未完成前缀返回 PENDING，并给出可恢复的树句柄；后续预算从该前缀继续，
#   而不是丢弃或用中间 Diff leaf 伪装战略价值。
# - 深度耗尽的未完成前缀严禁进入完整线、Macro backup、candidate 或训练标签。
#
# 设计要点：
# - Session 持有同一个 PuctSearch 与同一棵 root 树；advance(credit) 只在既有树上追加
#   常规 UCB simulation（不重置 visit/value/tree），真正复用已展开子树。
# - complete_lines 复用 extract_turn_lines：它只收敛到被访问过的 turn_end 边，天然
#   不会把 depth-leaf / 中间前缀当作完整线。
# - pending_prefixes 单独枚举「本回合内、非终局、被截断为 depth-leaf 且尚未 turn_end」
#   的前缀节点，标记 PENDING，供后续 advance 恢复；绝不进入 complete_lines。


PENDING_STATUS = "PENDING"


@dataclass(frozen=True)
class PendingPrefix:
    """一条被深度耗尽截断、尚未到达 turn_end 的未完成微观前缀。

    action_sequence：从回合起始到该前缀节点的动作序列（不含 turn_end）。
    prefix_snapshot：该前缀节点的局面（尚未 turn_end）。
    remaining_hand_key：该前缀处的剩余手牌多重集键（用于诊断/去重参考）。
    depth_reached：该前缀节点在树中的深度（供恢复预算参考）。
    visit_count：微观搜索对该前缀边累计的访问次数。
    status：固定 PENDING —— 明确它是未完成前缀，不得当作完整线或 Macro 价值。

    未完成前缀绝不携带 diff_score 战略价值：它只是一个可恢复句柄，不参与 backup。
    """

    action_sequence: tuple[LegalAction, ...]
    prefix_snapshot: BattleSnapshot
    remaining_hand_key: tuple[int, ...]
    depth_reached: int
    visit_count: int
    status: str = PENDING_STATUS


@dataclass(frozen=True)
class CompletionResult:
    """一次 micro completion session 的可消费快照。

    complete_lines：strict 完整回合线（只含到达 turn_end 的路径）。可被 Macro 直接
        包装消费（每条即一个 Macro edge）。
    pending_prefixes：深度耗尽、未完成的前缀（PENDING）。不得进入 Macro backup /
        candidate / 训练标签；仅用于诊断和后续 advance 恢复。
    executed_simulations：session 至今累计执行的常规 UCB simulation 数（跨多次 advance 累加）。
    root_active_player / root_turn_number：该完成器对应的回合边界身份。
    diagnostics：累计诊断（predictor 调用、greedy rollout、跨回合等），只观测。
    """

    complete_lines: tuple[TurnLine, ...]
    pending_prefixes: tuple[PendingPrefix, ...]
    executed_simulations: int
    root_active_player: str
    root_turn_number: int
    diagnostics: Mapping[str, Any]

    def has_pending(self) -> bool:
        return bool(self.pending_prefixes)


def _collect_pending_prefixes(root, root_player: str, root_turn: int) -> list[PendingPrefix]:
    """DFS 已展开树，收集本回合内被截断为 depth-leaf 的未完成前缀。

    只收集满足以下全部条件的节点：
    - 属于本回合（active_player == root_player 且 turn_number == root_turn）；
    - is_depth_leaf 为 True（因 max_depth 被截断）；
    - is_terminal 为 False（不是真正终局）；
    - 尚未 turn_end（snapshot.is_turn_ended 非 True）。

    终局叶、turn_end 后的跨回合节点、以及已到达 turn_end 的完整线都不算 pending。
    """
    pending: list[PendingPrefix] = []
    seen: set[int] = set()

    def dfs(node, path: tuple[LegalAction, ...], depth: int, via_turn_end: bool) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))

        in_turn = node.active_player == root_player and node.turn_number == root_turn

        # depth-leaf 未完成前缀：本回合内、非终局、尚未 turn_end、且不是经 turn_end 边到达。
        #
        # 经 turn_end 边到达的 depth-leaf 是「回合边界结算结果」（例如 max_turns=1 时
        # _handle_turn_end 把 turn_end after-state 作为 depth-leaf），属于完整线一侧，
        # 不是未完成前缀 —— 它已由 extract_turn_lines 作为完整线的 turn_end 边覆盖。
        if (
            in_turn
            and not via_turn_end
            and getattr(node, "is_depth_leaf", False)
            and not getattr(node, "is_terminal", False)
            and not getattr(node.snapshot, "is_turn_ended", False)
        ):
            pending.append(
                PendingPrefix(
                    action_sequence=path,
                    prefix_snapshot=node.snapshot,
                    remaining_hand_key=remaining_hand_key(node.snapshot, root_player),
                    depth_reached=depth,
                    visit_count=node.total_visits,
                )
            )
            # depth-leaf 无子节点可继续，止步。
            return

        # 终局 / 跨回合边界节点不再深入本回合前缀收集。
        if getattr(node, "is_terminal", False) or getattr(node, "is_depth_leaf", False):
            return
        if not in_turn:
            return

        for i, action in enumerate(node.actions):
            child = node.children[i]
            if child is None:
                continue
            edge_is_turn_end = _is_turn_end(action)
            dfs(child, path + (action,), depth + 1, edge_is_turn_end)

    dfs(root, (), 0, False)
    return pending


class MicroTurnCompletionSession:
    """可恢复的完整回合 micro completion session（P1）。

    持有同一个 PuctSearch 与同一棵 root 树，可多次 advance 追加 simulation credit
    而不重置树/visit 累积；每次 result() 复用 extract_turn_lines 提取 strict 完整线，
    并单独枚举 PENDING 未完成前缀。complete() 便捷方法返回 top-K 完整线供 Macro 包装。

    与 PuctSearch.search 的兼容性：
    - Session 用 PuctSearch.build_completion_root 构建与 search 语义一致的 root，
      再用 advance_ucb_simulations 追加常规 UCB simulation。
    - 不改变 PuctSearch.search 的默认行为；search 仍是一次性入口。
    - branch_seed 路径一致：advance 复用 PuctSearch._predict_with_seed，因此同一
      (snapshot, action, branch_seed) 产出与 search 相同的预测。

    用法：
        session = MicroTurnCompletionSession(search, snapshot, player="me")
        session.advance(64)
        session.advance(64)      # 在同一棵树上继续积累，不重置
        result = session.result()
        macro_edges = result.complete_lines        # strict 完整线，可交给 Macro
        pending = result.pending_prefixes           # PENDING，绝不进入 backup
    """

    def __init__(
        self,
        search: "PuctSearch",
        snapshot: BattleSnapshot,
        *,
        player: str = "me",
    ):
        self._search = search
        self._root_player = player
        self._root = search.build_completion_root(snapshot, player=player)
        self._root_turn = self._root.turn_number
        self._turn_start_snapshot = snapshot
        self._executed_simulations = 0
        # 累计诊断（跨多次 advance）。
        self._prediction_error_count = 0
        self._prediction_error_details: list[dict[str, Any]] = []
        self._partial_prediction_count = 0
        self._greedy_rollout_count = 0
        self._cross_turn_count = 0
        self._max_depth_reached = 0
        self._advance_count = 0

    @property
    def root(self):
        """暴露持有的 root 树（供诊断/审计；调用方不应重置其 visit）。"""
        return self._root

    @property
    def executed_simulations(self) -> int:
        return self._executed_simulations

    def advance(self, credit: int) -> int:
        """在既有 root 树上追加 credit 次常规 UCB simulation。

        返回本次实际执行的 simulation 数。多次 advance 累积到同一棵树，visit_counts
        与 value_sums 持续累加，不重置。
        """
        credit = max(0, int(credit))
        if credit == 0:
            return 0
        agg = self._search.advance_ucb_simulations(
            self._root,
            self._root_player,
            credit,
            executed_before=self._executed_simulations,
        )
        self._executed_simulations += agg["executed"]
        self._prediction_error_count += agg["prediction_error_count"]
        for detail in agg["prediction_error_details"]:
            if len(self._prediction_error_details) < 10:
                self._prediction_error_details.append(detail)
        self._partial_prediction_count += agg["partial_prediction_count"]
        self._greedy_rollout_count += agg["greedy_rollout_count"]
        self._cross_turn_count += agg["cross_turn_count"]
        if agg["max_depth_reached"] > self._max_depth_reached:
            self._max_depth_reached = agg["max_depth_reached"]
        self._advance_count += 1
        return agg["executed"]

    def complete_lines(self) -> list[TurnLine]:
        """提取当前树中的 strict 完整回合线（只含到达 turn_end 的路径）。"""
        return extract_turn_lines(self._root)

    def pending_prefixes(self) -> list[PendingPrefix]:
        """枚举当前树中被深度耗尽截断的未完成前缀（PENDING）。"""
        return _collect_pending_prefixes(self._root, self._root_player, self._root_turn)

    def result(self) -> CompletionResult:
        """汇总当前 session 状态为 CompletionResult。"""
        lines = tuple(self.complete_lines())
        pending = tuple(self.pending_prefixes())
        diagnostics = {
            "session_advance_count": self._advance_count,
            "executed_simulations": self._executed_simulations,
            "complete_line_count": len(lines),
            "pending_prefix_count": len(pending),
            "max_depth_reached": self._max_depth_reached,
            "prediction_error_count": self._prediction_error_count,
            "prediction_error_details": list(self._prediction_error_details),
            "partial_prediction_count": self._partial_prediction_count,
            "greedy_rollout_count": self._greedy_rollout_count,
            "cross_turn_count": self._cross_turn_count,
            # completion session 只把 strict 完整线暴露给 Macro；PENDING 永不进入 backup。
            "pending_excluded_from_macro": True,
        }
        return CompletionResult(
            complete_lines=lines,
            pending_prefixes=pending,
            executed_simulations=self._executed_simulations,
            root_active_player=self._root_player,
            root_turn_number=self._root_turn,
            diagnostics=diagnostics,
        )

    def top_k_complete_lines(
        self,
        k: int = 3,
        *,
        use_incremental: bool = False,
        weights=None,
        threshold_fn=None,
        extra_pp_signature_portfolio: bool = False,
    ) -> list[TurnLine]:
        """M0+M1 完整流程作用在 session 持有的树上，返回 top-K strict 完整线。

        可直接作为 Macro edge 候选被 build_completion_macro_turn_lines_fn 包装消费。
        未完成前缀不会出现在此结果中（extract_turn_lines 只收敛到 turn_end 边）。
        """
        return top_k_turn_lines(
            self._root,
            self._turn_start_snapshot,
            self._root_player,
            k,
            use_incremental=use_incremental,
            weights=weights,
            threshold_fn=threshold_fn,
            extra_pp_signature_portfolio=extra_pp_signature_portfolio,
        )


# ---------------------------------------------------------------------------
# P0：全局 credit 账本（只观测，不改变当前 top-K 行为）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroCreditBudget:
    """后续 lazy Macro 的全局预算声明。

    P0 只记录实际消耗与声明上限，绝不据此截断候选、改变递归或影响选择；
    真正 enforcement 由后续 lazy frontier 阶段接入。None 表示未声明上限。
    """

    max_predictor_calls: int | None = None
    max_micro_simulations: int | None = None
    max_macro_nodes: int | None = None
    max_turn_transitions: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_predictor_calls", self.max_predictor_calls),
            ("max_micro_simulations", self.max_micro_simulations),
            ("max_macro_nodes", self.max_macro_nodes),
            ("max_turn_transitions", self.max_turn_transitions),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")

    def to_diagnostics(self) -> dict[str, int | None]:
        return {
            "budget_max_predictor_calls": self.max_predictor_calls,
            "budget_max_micro_simulations": self.max_micro_simulations,
            "budget_max_macro_nodes": self.max_macro_nodes,
            "budget_max_turn_transitions": self.max_turn_transitions,
        }


@dataclass(frozen=True)
class MicroSearchOutcome:
    """可选的 micro search 返回包装，附带实际 PUCT diagnostics。

    旧 ``micro_search_fn -> root`` 注入仍完整兼容；生产路径使用本类型让
    Macro 汇总真实 simulation 与 predictor 调用，而不从候选数量猜测成本。
    """

    root: object
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnTransitionOutcome:
    """可选的 Macro 回合切换包装，记录外层 TURN_END/TURN_START 的真实成本。"""

    snapshot: BattleSnapshot
    predictor_calls: int = 0
    turn_transitions: int = 1


@dataclass
class _MacroCreditLedger:
    """单次 Macro 搜索的可序列化全局成本账本。"""

    macro_turn_expansion_count: int = 0
    macro_successor_node_count: int = 0
    macro_turn_transition_count: int = 0
    macro_transition_predictor_call_count: int = 0
    micro_search_count: int = 0
    micro_executed_simulations: int = 0
    micro_predictor_call_count: int = 0
    micro_turn_transition_count: int = 0

    def record_micro(self, diagnostics: Mapping[str, Any]) -> None:
        self.micro_search_count += 1
        self.micro_executed_simulations += _non_negative_int(
            diagnostics.get("executed_simulations")
        )
        self.micro_predictor_call_count += _non_negative_int(
            diagnostics.get("predictor_call_count")
        )
        self.micro_turn_transition_count += _non_negative_int(
            diagnostics.get("turn_transition_count")
        )

    def record_macro_transition(self, outcome: TurnTransitionOutcome) -> None:
        self.macro_successor_node_count += 1
        self.macro_turn_transition_count += max(0, outcome.turn_transitions)
        self.macro_transition_predictor_call_count += max(0, outcome.predictor_calls)

    def to_diagnostics(self, budget: MacroCreditBudget) -> dict[str, Any]:
        predictor_calls = (
            self.micro_predictor_call_count
            + self.macro_transition_predictor_call_count
        )
        values: dict[str, Any] = {
            "credit_schema": "MacroGlobalCredits/v1",
            "credit_mode": "observe_only",
            "macro_turn_expansion_count": self.macro_turn_expansion_count,
            "macro_successor_node_count": self.macro_successor_node_count,
            "macro_turn_transition_count": self.macro_turn_transition_count,
            "macro_transition_predictor_call_count": self.macro_transition_predictor_call_count,
            "micro_search_count": self.micro_search_count,
            "micro_executed_simulations": self.micro_executed_simulations,
            "micro_predictor_call_count": self.micro_predictor_call_count,
            "micro_turn_transition_count": self.micro_turn_transition_count,
            "predictor_call_count": predictor_calls,
            "credit_budget_enforced": False,
        }
        values.update(budget.to_diagnostics())
        return values


def _non_negative_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _unwrap_micro_outcome(value: object) -> tuple[object, Mapping[str, Any]]:
    if isinstance(value, MicroSearchOutcome):
        return value.root, value.diagnostics
    return value, {}


def _unwrap_transition_outcome(value: object) -> TurnTransitionOutcome:
    if isinstance(value, TurnTransitionOutcome):
        return value
    if not isinstance(value, BattleSnapshot):
        raise TypeError(
            "transition_fn must return BattleSnapshot or TurnTransitionOutcome"
        )
    return TurnTransitionOutcome(snapshot=value)


# ---------------------------------------------------------------------------
# Macro result
# ---------------------------------------------------------------------------


def _opponent(player: str) -> str:
    return "opponent" if player == "me" else "me"


@dataclass(frozen=True)
class MacroResult:
    """宏观搜索结果。

    backed_up_value: 从 root_player 视角，当前回合最优战略线经多回合展开后的 minimax 价值。
    best_line: 当前回合选中的回合线（其 action_sequence 是要执行的动作序列）。
    ranked_lines: 当前回合所有候选线，按 backed_up_value 降序（供 policy_target 构建）。
    line_values: 与 ranked_lines 对齐的 backed-up value 列表。
    diagnostics: 诊断信息（展开的宏观叶节点数、是否全落在回合边界等）。
    """

    backed_up_value: float
    best_line: TurnLine | None
    ranked_lines: tuple[TurnLine, ...]
    line_values: tuple[float, ...]
    diagnostics: dict


class MacroTurnSearch:
    """分层跨回合搜索：宏观回合树编排微观 PUCT，做 top-K minimax。

    统一以 root_player 视角贯穿价值（我方节点取 max、对手节点取 min），
    避免逐层取反的易错性。叶节点估值与终局判断均直接输出 root_player 视角。

    依赖注入（便于测试与生产复用）：
    - micro_search_fn(snapshot, player) -> micro_root(_PuctNode)：单回合微观 PUCT，返回树根。
      生产用 DiffAwarePuctSearch(max_turns=1).search() 后取 ._last_root。
    - transition_fn(end_snapshot, ending_player) -> next_snapshot：输入为回合线的 TURN_END 前局面，必须先结算 ending_player 的 TURN_END，再执行对方 TURN_START。
      生产用 DefaultTurnTransitioner.transition_turn。
    - leaf_value_fn(snapshot, root_player) -> float：宏观叶节点估值（root_player 视角）。
    - terminal_fn(snapshot, root_player) -> float | None：终局判断（root_player 视角），非终局返回 None。
    """

    def __init__(
        self,
        *,
        micro_search_fn,
        transition_fn,
        leaf_value_fn,
        terminal_fn,
        turn_lines_fn=None,
        macro_depth: int = 3,
        k: int = 3,
        root_k: int = 5,
        extra_pp_signature_portfolio: bool = False,
        use_incremental: bool = False,
        weights=None,
        threshold_fn=None,
        residual_fn=None,
        credit_budget: MacroCreditBudget | None = None,
        lazy_mode: bool = False,
        bucket_active_quota: int = 2,
        frontier_perspective: str | None = None,
        static_resource_mode: bool = False,
        static_weights=None,
        static_crest_valuation=None,
        static_board_valuation=None,
        static_extra_pp_redeemable=None,
    ):
        self.micro_search_fn = micro_search_fn
        self.transition_fn = transition_fn
        self.leaf_value_fn = leaf_value_fn
        self.terminal_fn = terminal_fn
        # turn_lines_fn(micro_root, snapshot, player, k) -> list[TurnLine]
        # 默认用 M0+M1 的 top_k_turn_lines；测试可注入简化版聚焦 minimax。
        self.turn_lines_fn = turn_lines_fn or top_k_turn_lines
        self.macro_depth = macro_depth
        self.k = k
        # 根层（当前回合）放宽 top-K：延迟收益线 D 当前回合分数往往一般，
        # 若用内层 k=3 纯分数截断会在宏观 backup 之前误伤 D。根层用更大的候选池
        # （默认 5），让 D 有机会进入并由 backed-up value 翻盘。成本 root_k×k×k。
        self.root_k = root_k
        self.extra_pp_signature_portfolio = extra_pp_signature_portfolio
        # 逐动作即时收益模型（PER_ACTION_INCREMENTAL_VALUE_PLAN）：
        # True 时路径价值 = Σ 每回合线的 step_reward 累加（含跨回合），
        # 地平线叶记 0（冷启动 value head 只训练不参与）；回合线评分也用 step_reward 累加。
        # False 时保持旧行为（回合线用 DiffSearchValue、地平线叶用 leaf_value_fn）。
        self.use_incremental = use_incremental
        self.weights = weights
        self.threshold_fn = threshold_fn
        # 方案A 地平线叶残值：use_incremental 时，若提供 residual_fn(snapshot, root_player)->float，
        # 地平线叶用「场面残值」(pp_of_stat) 而非记0，补过程纯涌现搜不到的未来场面价值。
        # None 时退回记0（冷启动可选）。
        self.residual_fn = residual_fn
        self.credit_budget = credit_budget or MacroCreditBudget()
        # P0 只记录配置和消耗；任何上限都尚未影响当前递归 top-K 选择。
        # E3：地平线叶把「当前 next_player（该叶将立即行动方）」作为 next_player 传给 residual_fn，
        # 以便 horizon_leaf_residual 计算 Extra PP option residual。为兼容既有仅接受
        # (snapshot, root_player) 的二参 residual_fn 注入，用显式签名探测的适配器，而非吞异常：
        # - 若 residual_fn 接受 next_player 关键字参数 → 调用时传 next_player。
        # - 否则按旧二参调用（既有数值不变）。
        self._residual_accepts_next_player = _callable_accepts_kwarg(residual_fn, "next_player")
        # P3 lazy Macro frontier（明确 opt-in）：
        # - lazy_mode=False（默认）：完全保留 legacy k/root_k 递归行为与所有现有诊断键。
        # - lazy_mode=True：根层用资源感知惰性前沿（FutureResourceSignature +
        #   BoardFingerprint 注册 edge），同资源桶初始最多两个 board-distinct 代表，
        #   其余 pending；全局 MacroCreditBudget 真实 enforce；每次 selection 只展开一条
        #   successor，不重跑已展开微观搜索。未完成 prefix 绝不进入 candidate/backup。
        self.lazy_mode = bool(lazy_mode)
        self.bucket_active_quota = max(0, int(bucket_active_quota))
        # frontier 视角：默认与搜索 root_player 一致（在 search 时按 player 解析）。
        self.frontier_perspective = frontier_perspective
        # P4 static resource Macro evaluator（明确 opt-in）：
        # - static_resource_mode=False（默认）：完全保留 legacy 行为，leaf/terminal
        #   仍走注入的 leaf_value_fn / terminal_fn，incremental 逻辑不变。
        # - static_resource_mode=True：Macro 的地平线叶与终局改用 macro_static_value
        #   （root-relative 绝对资源价值），且**不与 incremental line reward 双计**——
        #   静态模式强制使用绝对 leaf value，不累加 line.diff_score。
        #   weights / crest / board / extra_pp 均可注入；缺省沿用 macro_static_value 默认
        #   （crest/board provider 默认返回 0，board_residue 权重默认 0 → board 默认不计）。
        self.static_resource_mode = bool(static_resource_mode)
        self.static_weights = static_weights
        self.static_crest_valuation = static_crest_valuation
        self.static_board_valuation = static_board_valuation
        self.static_extra_pp_redeemable = static_extra_pp_redeemable

    def _static_kwargs(self) -> dict:
        """把注入的 static provider 折叠成 macro_static_value 的关键字参数。

        只传显式注入的 provider；未注入的分量用 macro_static_value 的模块默认
        （crest/board 默认 0、weights 默认 DEFAULT_WEIGHTS、redeemable 默认不猜）。
        """
        kwargs: dict = {}
        if self.static_weights is not None:
            kwargs["weights"] = self.static_weights
        if self.static_crest_valuation is not None:
            kwargs["crest_valuation"] = self.static_crest_valuation
        if self.static_board_valuation is not None:
            kwargs["board_valuation"] = self.static_board_valuation
        if self.static_extra_pp_redeemable is not None:
            kwargs["extra_pp_redeemable"] = self.static_extra_pp_redeemable
        return kwargs

    def _static_leaf_value(self, snapshot, root_player: str) -> float:
        """静态资源模式下的地平线叶估值（root_player 视角绝对资源价值）。"""
        from decision_engine.planning.macro_static_value import macro_static_value

        return macro_static_value(snapshot, root_player, **self._static_kwargs())

    def _static_terminal(self, snapshot, root_player: str):
        """静态资源模式下的终局判断（root_player 视角）。

        复用 macro_static_value 的终局短路：非终局返回 None，终局返回 ±v_win/0。
        """
        from decision_engine.planning.macro_static_value import (
            _winner as _static_winner,
            DEFAULT_WEIGHTS as _STATIC_DEFAULT_WEIGHTS,
        )

        winner = _static_winner(snapshot)
        if winner is None:
            return None
        weights = self.static_weights or _STATIC_DEFAULT_WEIGHTS
        if winner == "simultaneous":
            return 0.0
        return weights.v_win if winner == root_player else -weights.v_win

    @property
    def _incremental_active(self) -> bool:
        """静态资源模式下强制关闭 incremental 累加，避免与静态绝对 leaf value 双计。

        static_resource_mode=True 时返回 False：Macro backup 只用绝对静态 leaf value，
        不再累加 line.diff_score（否则同时叠加即时 line reward 与绝对资源分即双计）。
        """
        return self.use_incremental and not self.static_resource_mode

    def _leaf_value(self, snapshot, root_player: str) -> float:
        """地平线叶估值统一入口：静态模式用绝对资源价值，否则用注入的 leaf_value_fn。"""
        if self.static_resource_mode:
            return self._static_leaf_value(snapshot, root_player)
        return self.leaf_value_fn(snapshot, root_player)

    def _terminal(self, snapshot, root_player: str):
        """终局判断统一入口：静态模式用静态终局短路，否则用注入的 terminal_fn。"""
        if self.static_resource_mode:
            return self._static_terminal(snapshot, root_player)
        return self.terminal_fn(snapshot, root_player)

    def _call_residual(self, snapshot, root_player, next_player) -> float:
        """地平线叶残值调用适配器（E3 合同要点 6）。

        next_player = 该地平线叶将立即行动的一方。仅当注入的 residual_fn 支持
        next_player 关键字参数时才传入；否则退回二参调用保持旧值。
        """
        if self._residual_accepts_next_player:
            return self.residual_fn(snapshot, root_player, next_player=next_player)
        return self.residual_fn(snapshot, root_player)

    def search(self, snapshot: BattleSnapshot, *, player: str = "me") -> MacroResult:
        """从当前回合（player 行动）出发，做分层跨回合 minimax，返回当前回合的最优战略线。"""
        if self.lazy_mode:
            return self._search_lazy(snapshot, player=player)
        root_player = player
        ledger = _MacroCreditLedger()
        diag = {
            "macro_depth": self.macro_depth,
            "k": self.k,
            "root_k": self.root_k,
            "macro_leaf_count": 0,          # 宏观叶节点数（回合边界估值次数）
            "terminal_leaf_count": 0,       # 终局叶节点数
            "micro_search_count": 0,        # 微观搜索调用次数
            "root_line_count": 0,           # 当前回合去重后候选线数
            "root_player": root_player,
            "incremental_reward_perspective": "root_relative" if self.use_incremental else None,
            "incremental_root_turn_line_count": 0,
            "incremental_opponent_turn_line_count": 0,
        }

        # 展开当前回合的 top-(root_k) 线（放宽），并对每条线递归展开后续回合
        lines, line_values = self._expand_turn(
            snapshot,
            player,
            self.macro_depth,
            root_player,
            diag,
            ledger,
            k_override=self.root_k,
        )
        diag["root_line_count"] = len(lines)
        diag.update(ledger.to_diagnostics(self.credit_budget))

        if not lines:
            # 无候选线（不应发生）：逐动作模型记 0，旧行为/静态模式用叶估值
            v = 0.0 if self._incremental_active else self._leaf_value(snapshot, root_player)
            return MacroResult(v, None, (), (), diag)

        # 当前回合是 root_player，取 backed-up value 最大的线
        order = sorted(range(len(lines)), key=lambda i: line_values[i], reverse=True)
        ranked = tuple(lines[i] for i in order)
        ranked_vals = tuple(line_values[i] for i in order)
        diag.update(ledger.to_diagnostics(self.credit_budget))
        return MacroResult(
            backed_up_value=ranked_vals[0],
            best_line=ranked[0],
            ranked_lines=ranked,
            line_values=ranked_vals,
            diagnostics=diag,
        )

    # -----------------------------------------------------------------
    # P3: lazy Macro frontier 搜索路径（opt-in，legacy 路径不受影响）
    # -----------------------------------------------------------------

    def _root_candidate_lines(self, snapshot, player, k, ledger):
        """取根层完整回合线候选（复用 legacy 微观搜索 + turn_lines_fn）。

        lazy 模式下 k 应传较宽的值（root_k 或更宽），因为 frontier 才是真正的
        有界机制；这里只负责产出 strict 完整线，未完成 prefix 由 P1 completion
        session / extract_turn_lines 语义天然排除，不会进入候选。
        """
        micro_outcome = self.micro_search_fn(snapshot, player)
        micro_root, micro_diagnostics = _unwrap_micro_outcome(micro_outcome)
        ledger.record_micro(micro_diagnostics)
        if self.turn_lines_fn is top_k_turn_lines:
            # lazy 合同：不以 root_k/top_k 决定资格。所有已形成的严格完整
            # TurnLine 均注册为 frontier edge；资源/场面代表和全局预算只控制
            # 后续何时展开，绝不在这里永久删除候选。
            raw_lines = extract_turn_lines(micro_root)
            return score_turn_lines(
                raw_lines,
                snapshot,
                player,
                use_incremental=self.use_incremental,
                weights=self.weights,
                threshold_fn=self.threshold_fn,
            )
        # 自定义注入函数的候选范围由调用方合同负责；production default 不走此分支。
        return self.turn_lines_fn(micro_root, snapshot, player, k)

    def _search_lazy(self, snapshot: BattleSnapshot, *, player: str = "me") -> MacroResult:
        """资源感知惰性宏观前沿搜索（P3）。

        流程：
        1. 根层微观搜索取宽候选完整线（root_k）。
        2. 按 FutureResourceSignature 分桶、BoardFingerprint 区分，注册为 frontier
           edge：同资源桶初始最多两个 board-distinct 代表激活，其余 pending，不删除。
        3. 在全局 MacroCreditBudget（真实 enforce）内，每次 expand_one 只展开一条
           successor（对手回合 transition + 内层 minimax），不重跑已展开微观搜索。
        4. 预算耗尽或无可展开 edge 时停止；pending edge 记录结构化 reason，绝不删除。
        5. 只有**已展开**（有 backed_up_value）的 edge 才进入 candidate/ranked；未展开 /
           pending edge 绝不进入 macro candidate 或 backup。
        """
        from decision_engine.planning.macro_frontier import (
            ExpansionOutcome,
            LazyBudgetTracker,
            MacroFrontier,
        )

        root_player = player
        ledger = _MacroCreditLedger()
        perspective = self.frontier_perspective or player
        tracker = LazyBudgetTracker(self.credit_budget)

        diag = {
            "macro_depth": self.macro_depth,
            "k": self.k,
            "root_k": self.root_k,
            "macro_leaf_count": 0,
            "terminal_leaf_count": 0,
            "micro_search_count": 0,
            "root_line_count": 0,
            "root_player": root_player,
            "lazy_mode": True,
            "incremental_reward_perspective": "root_relative" if self.use_incremental else None,
            "incremental_root_turn_line_count": 0,
            "incremental_opponent_turn_line_count": 0,
            "lazy_successor_expansions": 0,
        }

        # 1. 根层候选完整线（宽度用 root_k；未完成 prefix 天然不入）。
        ledger.macro_turn_expansion_count += 1
        root_lines = self._root_candidate_lines(snapshot, player, self.root_k, ledger)
        diag["micro_search_count"] = ledger.micro_search_count
        diag["root_line_count"] = len(root_lines)

        # 2. 注册 frontier（分桶 + board-distinct 初始激活）。
        frontier = MacroFrontier(
            tracker,
            perspective=perspective,
            bucket_active_quota=self.bucket_active_quota,
        )
        frontier.register_lines(root_lines)

        # 3. expander：展开一条 edge = transition 到对手回合 + 内层 minimax。
        #    真实成本上报给 tracker 用于预算 enforcement；预算耗尽返回未展开。
        def expander(edge):
            line = edge.line
            # transition 预算预检查（每次展开一条 successor = 一次回合切换）。
            allowed_tt, reason_tt = tracker.check_turn_transition(1)
            if not allowed_tt:
                return ExpansionOutcome(expanded=False, pending_reason=reason_tt)

            transition_outcome = _unwrap_transition_outcome(
                self.transition_fn(line.end_snapshot, player)
            )
            # transition 的 predictor 成本预检查（先估后消费，超限则不透支）。
            trans_pred = max(0, transition_outcome.predictor_calls)
            allowed_pc, reason_pc = tracker.check_predictor_calls(trans_pred)
            if not allowed_pc:
                return ExpansionOutcome(expanded=False, pending_reason=reason_pc)

            ledger.record_macro_transition(transition_outcome)
            tracker.consume_turn_transition(max(0, transition_outcome.turn_transitions))
            tracker.consume_predictor_calls(trans_pred)
            next_snapshot = transition_outcome.snapshot

            child_value = self._minimax_lazy(
                next_snapshot,
                _opponent(player),
                self.macro_depth - 1,
                root_player,
                diag,
                ledger,
                tracker,
            )
            if self.use_incremental:
                root_relative_reward = (
                    line.diff_score if player == root_player else -line.diff_score
                )
                diag["incremental_root_turn_line_count"] += 1
                backed = root_relative_reward + child_value
            else:
                backed = child_value
            return ExpansionOutcome(
                expanded=True,
                successor_snapshot=next_snapshot,
                backed_up_value=backed,
            )

        # 4. progressive widening：每次只展开一条 successor，预算耗尽即停。
        while frontier.has_expandable():
            expansion = frontier.expand_one(expander)
            if expansion.expanded_edge is not None:
                diag["lazy_successor_expansions"] += 1
                continue
            if expansion.exhausted:
                # 预算耗尽：停止 widening；pending edge 已记录 reason，不删除。
                break
            # 无可展开 edge（非耗尽）：结束循环。
            break

        # 5. 只收集已展开 edge 作为 candidate（未完成 / pending 绝不进入）。
        expanded = frontier.expanded_edges()
        diag.update(ledger.to_diagnostics(self.credit_budget))
        diag.update(tracker.to_diagnostics())
        diag.update(frontier.to_diagnostics())

        if not expanded:
            v = 0.0 if self.use_incremental else self.leaf_value_fn(snapshot, root_player)
            return MacroResult(v, None, (), (), diag)

        # root_player 回合取 max。
        expanded.sort(key=lambda e: e.backed_up_value, reverse=True)
        ranked = tuple(e.line for e in expanded)
        ranked_vals = tuple(float(e.backed_up_value) for e in expanded)
        return MacroResult(
            backed_up_value=ranked_vals[0],
            best_line=ranked[0],
            ranked_lines=ranked,
            line_values=ranked_vals,
            diagnostics=diag,
        )

    def _minimax_lazy(
        self,
        snapshot,
        player,
        macro_depth,
        root_player,
        diag,
        ledger: _MacroCreditLedger,
        tracker,
    ) -> float:
        """lazy 模式下的内层 minimax（复用 legacy minimax 语义，但预算真实 enforce）。

        与 legacy `_minimax` 的差别：内层每一步微观搜索前检查 micro simulation 预算；
        micro search 上报的实际 predictor 调用也计入并检查。预算耗尽时不再深入展开，
        直接以地平线叶残值 / leaf_value_fn 结算（不静默删除、不伪装成成功）。

        注意：内层不删除已获 root 保护的 candidate（那些由 root frontier 管理）；
        这里只负责对手反制估计。预算耗尽时以叶结算是显式的价值下界处理，非 fallback。
        """
        tv = self.terminal_fn(snapshot, root_player)
        if tv is not None:
            diag["terminal_leaf_count"] += 1
            return tv

        if macro_depth <= 0:
            diag["macro_leaf_count"] += 1
            if self.use_incremental:
                return (
                    self._call_residual(snapshot, root_player, player)
                    if self.residual_fn is not None else 0.0
                )
            return self.leaf_value_fn(snapshot, root_player)

        # 内层微观搜索预算预检查：用声明的 micro_simulations 上限保守估一次搜索的
        # 规模（若无法预知，用 1 作为最小占位单位，保证至少检查一次预算是否已耗尽）。
        allowed_sim, _ = tracker.check_micro_simulations(1)
        if not allowed_sim:
            # 内层微观预算耗尽：以地平线叶残值结算（显式下界，非 fallback）。
            diag["macro_leaf_count"] += 1
            if self.use_incremental:
                return (
                    self._call_residual(snapshot, root_player, player)
                    if self.residual_fn is not None else 0.0
                )
            return self.leaf_value_fn(snapshot, root_player)

        ledger.macro_turn_expansion_count += 1
        micro_outcome = self.micro_search_fn(snapshot, player)
        micro_root, micro_diagnostics = _unwrap_micro_outcome(micro_outcome)
        ledger.record_micro(micro_diagnostics)
        diag["micro_search_count"] = ledger.micro_search_count

        # 消耗内层实际 micro simulation 与 predictor 调用（观测到多少消耗多少）。
        executed = _non_negative_int(micro_diagnostics.get("executed_simulations")) or 1
        tracker.consume_micro_simulations(executed)
        tracker.consume_predictor_calls(
            _non_negative_int(micro_diagnostics.get("predictor_call_count"))
        )

        if self.turn_lines_fn is top_k_turn_lines:
            # lazy 内层遵守与根层相同的资格合同：所有 strict 完整线都注册/可见，
            # 预算只控制展开，不以 k 静默删除。
            lines = score_turn_lines(
                extract_turn_lines(micro_root),
                snapshot,
                player,
                use_incremental=self.use_incremental,
                weights=self.weights,
                threshold_fn=self.threshold_fn,
            )
        else:
            lines = self.turn_lines_fn(micro_root, snapshot, player, self.k)

        values: list[float] = []
        for line in lines:
            # 内层 transition 预算检查。
            allowed_tt, _ = tracker.check_turn_transition(1)
            if not allowed_tt:
                break
            transition_outcome = _unwrap_transition_outcome(
                self.transition_fn(line.end_snapshot, player)
            )
            trans_pred = max(0, transition_outcome.predictor_calls)
            allowed_pc, _ = tracker.check_predictor_calls(trans_pred)
            if not allowed_pc:
                break
            ledger.record_macro_transition(transition_outcome)
            tracker.consume_turn_transition(max(0, transition_outcome.turn_transitions))
            tracker.consume_predictor_calls(trans_pred)
            next_snapshot = transition_outcome.snapshot
            child_value = self._minimax_lazy(
                next_snapshot, _opponent(player), macro_depth - 1,
                root_player, diag, ledger, tracker,
            )
            if self.use_incremental:
                root_relative_reward = (
                    line.diff_score if player == root_player else -line.diff_score
                )
                if player == root_player:
                    diag["incremental_root_turn_line_count"] += 1
                else:
                    diag["incremental_opponent_turn_line_count"] += 1
                values.append(root_relative_reward + child_value)
            else:
                values.append(child_value)

        if not values:
            diag["macro_leaf_count"] += 1
            if self.use_incremental:
                return (
                    self._call_residual(snapshot, root_player, player)
                    if self.residual_fn is not None else 0.0
                )
            return self.leaf_value_fn(snapshot, root_player)

        if player == root_player:
            return max(values)
        return min(values)

    def _expand_turn(
        self,
        snapshot,
        player,
        macro_depth,
        root_player,
        diag,
        ledger: _MacroCreditLedger,
        k_override=None,
    ):
        """展开 player 的一个回合：微观搜索取 top-K 线，对每条线递归后续回合。

        k_override: 根层传 root_k（放宽）；内层递归为 None → 用 self.k。

        返回 (lines, values)：lines 为本回合 top-K 回合线，values 为各线的 backed-up value
        （root_player 视角，已含后续回合的 minimax 结果）。
        """
        k = k_override if k_override is not None else self.k
        ledger.macro_turn_expansion_count += 1
        micro_outcome = self.micro_search_fn(snapshot, player)
        micro_root, micro_diagnostics = _unwrap_micro_outcome(micro_outcome)
        ledger.record_micro(micro_diagnostics)
        # 保持既有diagnostic键，同时使它与全局账本一致。
        diag["micro_search_count"] = ledger.micro_search_count
        # 默认 top_k_turn_lines 支持 use_incremental；注入的简化 turn_lines_fn 只收 4 参数
        if self.turn_lines_fn is top_k_turn_lines:
            lines = self.turn_lines_fn(micro_root, snapshot, player, k,
                                       use_incremental=self.use_incremental,
                                       weights=self.weights, threshold_fn=self.threshold_fn,
                                       extra_pp_signature_portfolio=self.extra_pp_signature_portfolio)
        else:
            lines = self.turn_lines_fn(micro_root, snapshot, player, k)

        values: list[float] = []
        for line in lines:
            # 切到对手回合。P0 接受旧 snapshot 返回，也允许生产路径返回明确成本。
            transition_outcome = _unwrap_transition_outcome(
                self.transition_fn(line.end_snapshot, player)
            )
            ledger.record_macro_transition(transition_outcome)
            next_snapshot = transition_outcome.snapshot
            child_value = self._minimax(
                next_snapshot,
                _opponent(player),
                macro_depth - 1,
                root_player,
                diag,
                ledger,
            )
            if self._incremental_active:
                # score_turn_lines 的 diff_score 是当前行动方视角；宏观 minimax
                # 所有价值必须固定为 root_player 视角，因此对手本回合收益取负。
                root_relative_reward = line.diff_score if player == root_player else -line.diff_score
                if player == root_player:
                    diag["incremental_root_turn_line_count"] += 1
                else:
                    diag["incremental_opponent_turn_line_count"] += 1
                values.append(root_relative_reward + child_value)
            else:
                # 旧行为/静态模式：价值来自后续回合 minimax（叶用 value head 或
                # 静态绝对资源分），回合线分只用于 top_k 排序，不累加 line.diff_score。
                values.append(child_value)
        return lines, values

    def _minimax(
        self,
        snapshot,
        player,
        macro_depth,
        root_player,
        diag,
        ledger: _MacroCreditLedger,
    ) -> float:
        """递归 minimax：返回从 root_player 视角，snapshot 出发的最优多回合价值。

        player: 当前回合行动方。macro_depth: 剩余可展开回合数。
        """
        # 终局
        tv = self._terminal(snapshot, root_player)
        if tv is not None:
            diag["terminal_leaf_count"] += 1
            return tv

        # 回合数预算耗尽：地平线叶
        if macro_depth <= 0:
            diag["macro_leaf_count"] += 1
            # 逐动作模型：地平线叶用场面残值(pp_of_stat)补搜不到的未来场面价值；
            # 无 residual_fn 时记0（冷启动可选）。旧行为/静态模式用绝对叶估值。
            # E3：next_player = player（该叶将立即行动方），用于 Extra PP option residual。
            if self._incremental_active:
                return self._call_residual(snapshot, root_player, player) if self.residual_fn is not None else 0.0
            return self._leaf_value(snapshot, root_player)

        lines, values = self._expand_turn(
            snapshot, player, macro_depth, root_player, diag, ledger
        )
        if not values:
            diag["macro_leaf_count"] += 1
            if self._incremental_active:
                return self._call_residual(snapshot, root_player, player) if self.residual_fn is not None else 0.0
            return self._leaf_value(snapshot, root_player)

        # minimax：我方回合取 max，对手回合取 min（均为 root_player 视角）
        if player == root_player:
            return max(values)
        return min(values)



# ---------------------------------------------------------------------------
# M4: 生产组装（用真实部件构建 MacroTurnSearch）
# ---------------------------------------------------------------------------


def build_production_macro_search(
    *,
    checkpoint_path: str,
    templates_by_base_card_id,
    micro_simulations: int = 64,
    micro_max_depth: int = 8,
    c_puct: float = 1.5,
    macro_depth: int = 3,
    k: int = 3,
    root_k: int = 5,
    extra_pp_signature_portfolio: bool = True,
    device: str = "cpu",
    use_incremental: bool = False,
    static_resource_mode: bool = False,
    static_weights=None,
    static_crest_valuation=None,
    static_board_valuation=None,
    static_extra_pp_redeemable=None,
) -> "MacroTurnSearch":
    """用真实部件组装 MacroTurnSearch（structured backend champion）。

    - micro_search_fn: DiffAwarePuctSearch(max_turns=1)，每次调用重置 baseline 后搜索，返回内部树根。
    - transition_fn:   DefaultTurnTransitioner.transition_turn（对手 start_turn）。
    - leaf_value_fn:   StructuredNeuralValueFunction.evaluate（root_player 视角绝对局面价值，终局±1）。
    - terminal_fn:     同上，terminal=True 时返回其 value，否则 None。

    static_resource_mode（默认 False）：opt-in 静态资源 Macro evaluator。开启时地平线叶
    与终局改用 macro_static_value 的 root-relative 绝对资源价值，且不与 incremental line
    reward 双计（内部强制使用绝对 leaf value，不累加 line.diff_score）。weights / crest /
    board / extra_pp provider 均可注入，缺省沿用 macro_static_value 的默认（board 默认 0）。
    """
    from decision_engine.planning.puct import PuctConfig, DefaultTurnTransitioner
    from decision_engine.planning.search_prior import StructuredNeuralPrior
    from decision_engine.planning.search_value import StructuredNeuralValueFunction
    from decision_engine.planning.diff_value import DiffAwarePuctSearch
    from decision_engine.actions.extra_pp_enumerator import ExtraPpAwareEnumerator
    from decision_engine.execution.local_predictor import LocalPredictor
    from decision_engine.execution.player_local_predictor import PlayerLocalPredictor

    predictor = PlayerLocalPredictor(LocalPredictor(templates_by_base_card_id=templates_by_base_card_id))
    enumerator = ExtraPpAwareEnumerator()
    prior = StructuredNeuralPrior(checkpoint_path=checkpoint_path)
    transitioner = DefaultTurnTransitioner(predictor=predictor)
    micro_cfg = PuctConfig(
        simulations=micro_simulations, max_depth=micro_max_depth,
        max_turns=1, c_puct=c_puct,
    )
    micro = DiffAwarePuctSearch(
        enumerator=enumerator, child_enumerator=enumerator, predictor=predictor,
        prior=prior, turn_transitioner=transitioner, config=micro_cfg,
    )
    value_fn = StructuredNeuralValueFunction(checkpoint_path=checkpoint_path, device=device)

    def micro_search_fn(snapshot, player):
        # 重置 DiffSearchValue baseline，确保 turn_start = 本回合起始 snapshot
        micro.notify_turn_end()
        result = micro.search(snapshot, player=player)
        return MicroSearchOutcome(root=micro._search._last_root, diagnostics=result.diagnostics)

    def transition_fn(end_snapshot, ending_player):
        # TurnLine.end_snapshot 是“正要 TURN_END”的局面；不能直接调用只负责
        # next TURN_START 的 transitioner，否则会跳过 turn_end 事件并把错误时序
        # 的状态交给对手宏观回合。
        from decision_engine.actions.model import ActionKind, LegalAction
        from decision_engine.execution.execution_context import LocalExecutionContext

        before_transition_calls = transitioner.predictor_call_count
        ended = predictor.predict_with_replay_context(
            end_snapshot,
            LegalAction(kind=ActionKind.TURN_END, source_uid=0),
            player=ending_player,
            # Macro search 当前不携带 ProofSchedulerContext；此处仅是启发式宏观
            # 搜索的固定随机调度，不得用于 strict certificate。
            execution_context=LocalExecutionContext(branch_seed=0),
        )
        next_snapshot = transitioner.transition_turn(ended.snapshot, ending_player)
        return TurnTransitionOutcome(
            snapshot=next_snapshot,
            predictor_calls=1 + max(0, transitioner.predictor_call_count - before_transition_calls),
            turn_transitions=1,
        )

    def leaf_value_fn(snapshot, root_player):
        return value_fn.evaluate(snapshot, player=root_player).value

    def terminal_fn(snapshot, root_player):
        res = value_fn.evaluate(snapshot, player=root_player)
        return res.value if res.terminal else None

    # 方案A：use_incremental 时，地平线叶用场面残值(pp_of_stat)，
    # 终局用 incremental 尺度(±v_win=±1000)对齐过程 step_reward 量纲。
    residual_fn = None
    inc_terminal_fn = terminal_fn
    if use_incremental:
        from decision_engine.planning.incremental_value import (
            horizon_leaf_residual, DEFAULT_WEIGHTS, _winner,
        )

        def residual_fn(snapshot, root_player, *, next_player=None):
            # E3：把 horizon 叶将立即行动的一方作为 next_player 传入，启用 Extra PP
            # option residual；MacroTurnSearch._call_residual 会在支持时传 next_player。
            return horizon_leaf_residual(snapshot, root_player, next_player=next_player)

        def inc_terminal_fn(snapshot, root_player):
            w = _winner(snapshot)
            if w is None:
                return None
            if w == "simultaneous":
                return 0.0
            return DEFAULT_WEIGHTS.v_win if w == root_player else -DEFAULT_WEIGHTS.v_win

    return MacroTurnSearch(
        micro_search_fn=micro_search_fn,
        transition_fn=transition_fn,
        leaf_value_fn=leaf_value_fn,
        terminal_fn=inc_terminal_fn,
        macro_depth=macro_depth,
        k=k,
        root_k=root_k,
        extra_pp_signature_portfolio=extra_pp_signature_portfolio,
        use_incremental=use_incremental,
        residual_fn=residual_fn,
        static_resource_mode=static_resource_mode,
        static_weights=static_weights,
        static_crest_valuation=static_crest_valuation,
        static_board_valuation=static_board_valuation,
        static_extra_pp_redeemable=static_extra_pp_redeemable,
    )
