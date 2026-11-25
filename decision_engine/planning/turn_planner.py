# -*- coding: utf-8 -*-
"""回合规划：把这一回合的动作排成一条**收益最高**的路线。

和 `tools/find_winning_route.py` 的区别（两者解决不同问题，不要混用）：

- `find_winning_route` 追的是**硬目标**：能不能斩杀 / 能不能清场。谜题验收用它，
  因为谜题的答案帧就是斩杀或清场。
- `plan_turn` 追的是**收益**：没有斩杀路线时，怎么分配攻击让净血差最大。对局决策
  用它。清场只是「让对方场攻归零」的手段，不是目标本身。

## 搜索方式

穷举本回合所有能改变局面的动作序列，对每个终局调用
:func:`decision_engine.planning.turn_evaluation.score_terminal`，取评分最高的一条。
评分只依赖终局状态（外加固定的回合起始快照），所以同一状态只需要展开一次。

动作展开顺序按一组启发式排（只影响**先试哪个**，不影响最优性——所有动作都会被
展开，顺序只决定剪枝能多早生效）：

1. **攻击目标优先选对方场攻最高的随从**——它下回合最可能打掉我们的血；
2. **不可打脸的随从（Rush 等）的攻击先展开**——它们留到最后也没法打脸；
3. **目标选「攻击力刚好 ≥ 血量」的**——能一次带走就一次带走，节约攻击次数；
4. **己方攻击力降序**——大随从先选目标，避免好目标被小随从占掉。

## 上界剪枝

    剩余可打脸伤害 = 能打脸的随从攻击力 × 剩余攻击次数 + 击飞上限
    乐观净血差   = 已造成伤害 + 剩余可打脸伤害 - 0（假设对方场面被清空）

上界不如已找到的最优解就剪掉整棵子树。「对方场面清空」是乐观假设，不会剪掉更优
的分支。已经找到斩杀时，任何「当前还斩不了」的分支直接剪。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from decision_engine.actions.model import ActionKind

from .tactical_summary import summarize_turn
from .turn_evaluation import TurnScore, score_terminal, state_fingerprint

#: 会改变场面或主战者生命的动作。其余（TURN_START / EXTRA_PP / FUSION 等）跳过。
_PLAN_KINDS = {
    ActionKind.ATTACK_LEADER,
    ActionKind.ATTACK_FOLLOWER,
    ActionKind.PLAY,
    ActionKind.EVOLVE,
    ActionKind.SUPER_EVOLVE,
    ActionKind.ACTIVATE,
    ActionKind.TURN_END,
}


@dataclass
class TurnPlan:
    """一条回合路线的规划结果。"""

    route: list
    score: TurnScore
    before: Any
    after: Any
    nodes: int = 0
    pruned_by_bound: int = 0
    #: 本次规划是否因为 ``node_budget`` 被截断。截断时 ``route`` 不保证最优。
    truncated: bool = False
    #: 撞到预算的分支次数（用于诊断爆炸点有多严重）。
    budget_hits: int = 0
    #: 本次规划耗时（秒）。用于定位「哪些局面贵」，不进入评分。
    elapsed_seconds: float = 0.0
    #: memo 命中次数。命中低说明大量路径重复展开（排列冗余）。
    memo_hits: int = 0
    #: 所有展开节点的子节点数总和；除以 nodes 即平均分支因子。
    branch_sum: int = 0
    #: 单个节点的最大分支因子。
    max_branch: int = 0
    #: 阶段 1 停下出牌的原因（``no_hand`` / ``no_pp`` / ``field_full`` / ``no_playable``）。
    #: ``field_full`` 表示"还有 PP 但场地满"，是交换后折回出牌的依据。
    play_stop_reason: str | None = None
    #: 各阶段展开的节点数（诊断用）。
    phase_visits: dict | None = None
    #: prior 剪枝累计裁掉的候选数（0 表示剪枝没生效或未开启）。
    prior_pruned: int = 0


def _side(snapshot, player: str):
    return (snapshot.me, snapshot.opponent) if player == "me" else (snapshot.opponent, snapshot.me)


def optimistic_bound(state, before_snapshot, *, player: str = "me") -> int:
    """从 ``state`` 出发能拿到的净血差上界（乐观）。"""
    _, theirs = _side(state, player)
    _, theirs_before = _side(before_snapshot, player)
    dealt = max(0, int(theirs_before.life) - int(theirs.life))
    summary = summarize_turn(state, player=player)
    leader_damage = int(summary.details.get("leader_damage", 0))
    trample = int(summary.details.get("trample_damage", 0))
    return dealt + leader_damage + trample


def can_still_lethal(state, *, player: str = "me") -> bool:
    """从 ``state`` 出发还有没有斩杀可能（乐观口径）。"""
    summary = summarize_turn(state, player=player)
    _, theirs = _side(state, player)
    leader_damage = int(summary.details.get("leader_damage", 0))
    trample = int(summary.details.get("trample_damage", 0))
    return int(theirs.life) <= leader_damage + trample


#: ---- 分阶段搜索（phased）的常量 -------------------------------------------
#: 阶段 1 只允许出牌/启动；阶段 2 只允许进化；阶段 3 贪心推进攻击与回合结束。
#: 这样把「出牌与攻击任意交错」的排列空间压成「阶段内顺序 + 阶段间有序」。
_PLAY_PHASE_KINDS = frozenset({ActionKind.PLAY, ActionKind.ACTIVATE})
_EVOLVE_PHASE_KINDS = frozenset({ActionKind.EVOLVE, ActionKind.SUPER_EVOLVE})
_ATTACK_PHASE_KINDS = frozenset({ActionKind.ATTACK_FOLLOWER, ActionKind.ATTACK_LEADER})

#: 场地格数上限（随从与护符共享）。
_FIELD_LIMIT = 5

_PHASE_PLAY = "play"
_PHASE_EVOLVE = "evolve"
_PHASE_EXCHANGE = "exchange"

#: 阶段 1 的停止原因。``field_full`` 是唯一值得折回的：场面交换腾出格子后
#: 还能继续出牌；其余原因折回也没有新动作。
STOP_NO_HAND = "no_hand"
STOP_NO_PP = "no_pp"
STOP_FIELD_FULL = "field_full"
STOP_NO_PLAYABLE = "no_playable"


def _field_count(snapshot, player: str) -> int:
    view, _ = _side(snapshot, player)
    return len(getattr(view, "field", ()) or ())


def play_stop_reason(snapshot, player: str = "me", *, play_actions=()) -> str | None:
    """阶段 1 为什么打不出牌了；还能出牌时返回 ``None``。

    这个记号不只是诊断：``STOP_FIELD_FULL``（**还有 PP 但场地满**）是折回阶段 1 的
    唯一信号——阶段 3 的交换如果把格子腾出来，就应该回去把牌打完。
    """
    if play_actions:
        return None
    view, _ = _side(snapshot, player)
    hand = tuple(getattr(view, "hand", ()) or ())
    if not hand:
        return STOP_NO_HAND
    if _field_count(snapshot, player) >= _FIELD_LIMIT:
        return STOP_FIELD_FULL
    cards = snapshot.cards_by_uid
    pp = int(getattr(view, "pp", 0) or 0)
    # 手里最低费的牌都放不下，才算「PP 不足」；否则是「有 PP 但没有能打出的牌」。
    costs = [int(getattr(cards.get(uid), "cost", 0) or 0) for uid in hand if uid in cards]
    if costs and min(costs) > pp:
        return STOP_NO_PP
    return STOP_NO_PLAYABLE


def _action_sort_key(snapshot, action) -> tuple:
    """动作展开顺序。只影响搜索效率，不影响最优性。"""
    cards = snapshot.cards_by_uid
    kind = action.kind
    if kind is ActionKind.ATTACK_FOLLOWER:
        source = cards.get(action.source_uid)
        target_uid = next((uid for group in (action.target_groups or ()) for uid in group), None)
        target = cards.get(target_uid) if target_uid is not None else None
        if source is None or target is None:
            return (2, 0, 0, 0, 0, 0)
        # 目标：对方场攻最高的先解
        target_threat = -int(target.atk)
        # 攻击者：不能打脸的（Rush）先动，留着也没用
        must_use_now = 0 if not bool(getattr(source, "can_attack_leader", False)) else 1
        # 能一次带走就带走；带不走的按「还差多少」排后面
        source_atk = int(source.atk)
        target_life = int(target.life)
        kill_gap = 0 if source_atk >= target_life else (target_life - source_atk)
        return (0, target_threat, must_use_now, kill_gap, -source_atk, 0)
    if kind is ActionKind.ATTACK_LEADER:
        source = cards.get(action.source_uid)
        atk = int(source.atk) if source is not None else 0
        return (1, 0, 0, 0, -atk, 0)
    if kind is ActionKind.SUPER_EVOLVE:
        return (3, 0, 0, 0, 0, 0)
    if kind is ActionKind.EVOLVE:
        return (4, 0, 0, 0, 0, 0)
    if kind is ActionKind.PLAY:
        card = cards.get(action.source_uid)
        cost = int(getattr(card, "cost", 0) or 0) if card is not None else 0
        return (5, 0, 0, 0, 0, -cost)
    if kind is ActionKind.ACTIVATE:
        return (6, 0, 0, 0, 0, 0)
    if kind is ActionKind.TURN_END:
        return (9, 0, 0, 0, 0, 0)
    return (7, 0, 0, 0, 0, 0)


def is_deciding_player_turn(snapshot, player: str = "me") -> bool:
    """该决策方此刻是否处于自己的回合（视角换算的唯一出口）。

    ``snapshot.is_ally_turn`` 的语义是"是否**我方**回合"，不是"是否**决策方**回合"。
    两者在 ``player="opponent"`` 时相反。这里把换算集中到一处，避免各调用点各写一遍
    ``not is_ally_turn`` 或干脆漏掉——漏掉一次，``plan_turn`` 就会在对手回合判成
    "回合已结束"并返回空路线，对手永远不会攻击/进化。

    同类使用点审计（2026-10-30）：其余 3 处读 ``is_ally_turn`` 的地方都是"我方绝对
    视角"，语义与字面一致，**不是**同类缺陷——``local_predictor._refresh_local_action_flags``
    （只在我方回合刷新进化标志）、``cross_turn_transition.apply_after_ally_turn_end``
    （要求我方回合结束）、``replay_validator._predict_with_replay_context``（TURN_START
    取反得**下一回合**玩家，因为 ``rule_kernel.start_turn`` 会把该字段重置为新回合方）。
    但任何接受 ``player`` 参数的函数都不许直接读 ``is_ally_turn``：那是绝对视角字段，
    必须经由本函数换算。
    """
    ally_active = bool(getattr(snapshot, "is_ally_turn", True))
    return ally_active if player == "me" else not ally_active


def plan_turn(
    snapshot,
    *,
    predictor,
    enumerator,
    player: str = "me",
    max_actions: int = 12,
    allow_turn_end: bool = True,
    prediction_filter=None,
    node_budget: int | None = None,
    search_mode: str = "phased",
    bound_pruning: bool = True,
    action_prior=None,
    prior_top_k: int = 0,
    prior_depth: int = 0,
) -> TurnPlan:
    """求这一回合的最优动作序列。

    ``max_actions`` 是这一回合最多走多少个动作；超过就按当前状态评分（截断）。
    正常回合用不到 12 个动作，触顶说明出现了异常循环。

    ``prediction_filter`` 可选：给一个 ``prediction -> bool``，返回 False 的分支直接
    跳过。A/B runner 的 strict 口径要求每一步执行的预测都是 EXACT/SPEC_ONLY 且没有
    unresolved，规划器必须用同一个过滤器，否则会规划出一条 runner 拒绝执行的路线
    （整局作废）。

    ``search_mode`` 选择搜索结构：

    - ``"phased"``（默认）：按「出牌 → 进化 → 场面交换」分阶段推进，交换腾出场地时
      折回出牌。把「出牌与攻击任意交错」的排列空间压成「阶段内顺序 + 阶段间有序」：
      实测最贵决策点 73s → 2.9s、p90 0.50s → 0.17s（见
      ``docs/plan/PLAN_TURN_COST_PROFILE.md``）。已知在约 4%（3/76）的大场面局面上
      会漏解，原因是"先攻击再出牌"存在真实顺序依赖；人工复核这 3 例后判定可接受
      （差异集中在 net_life/站场等次级项，且 phased 站场更多）。
    - ``"dfs"``：任意交错穷举，保证在评分函数下最优，但会在大场面分支爆炸。
      需要严格最优或做对照验收时用。

    ``node_budget`` 可选：本次规划的节点上限。**默认 ``None``（不限制），生产路径不要
    设置**——超预算的分支会按"当前状态已结算"处理，得到的是近似解，``TurnPlan.truncated``
    会置 True。它存在的唯一目的是**诊断**：某些局面会分支爆炸到分钟级，没有安全网就
    无法采样到爆炸点。真正的修复应当针对爆炸来源，而不是长期挂着预算。
    """
    if search_mode not in {"phased", "dfs"}:
        raise ValueError(f"search_mode 只能是 'phased' 或 'dfs'，收到 {search_mode!r}")
    before = snapshot
    started_at = time.perf_counter()
    best: dict[str, Any] = {"score": None}
    counters = {"nodes": 0, "pruned": 0, "budget_hits": 0, "memo_hits": 0, "branch_sum": 0, "max_branch": 0}
    memo: dict[tuple, tuple[TurnScore, tuple]] = {}
    bounds: dict[tuple, tuple[int, bool]] = {}
    phase_counts: dict[str, int] = {}

    def _apply_prior_prune(actions, state, depth):
        """prior 引导的候选裁剪（``prior_top_k=0`` 时原样返回）。

        目的不是「更准」，而是**用剪枝换深度**——分支从 13 降到 5 后，同样的时间
        预算能展开更深的回合路线。``turn_end`` 始终保留（否则无法结束回合）。

        注意 ``depth`` 是**剩余预算**（从 ``max_actions`` 递减），不是层数；
        根节点 ``depth == max_actions``，所以层数是 ``max_actions - depth``。
        """
        if action_prior is None or prior_top_k <= 0:
            return actions
        level = max_actions - depth
        if level > prior_depth:
            return actions
        if len(actions) <= prior_top_k:
            return actions
        prior_values = action_prior.priors(state, tuple(actions), player=player).priors
        if len(prior_values) != len(actions):
            raise ValueError(
                f"prior 长度 {len(prior_values)} 与候选数 {len(actions)} 不一致"
            )
        order = sorted(range(len(actions)), key=lambda index: -float(prior_values[index]))
        kept = order[:prior_top_k]
        turn_end_index = next(
            (i for i, a in enumerate(actions) if a.kind is ActionKind.TURN_END), None
        )
        if turn_end_index is not None and turn_end_index not in kept:
            kept.append(turn_end_index)
        kept_set = set(kept)
        pruned = [a for i, a in enumerate(actions) if i in kept_set]
        counters["prior_pruned"] = counters.get("prior_pruned", 0) + (len(actions) - len(pruned))
        return pruned

    def _dfs(state, depth: int, path_keys: set) -> tuple[TurnScore, tuple] | None:
        actions = [
            action for action in enumerator.enumerate(state, player=player) if action.kind in _PLAN_KINDS
        ]
        if not allow_turn_end:
            actions = [action for action in actions if action.kind is not ActionKind.TURN_END]
        actions = _apply_prior_prune(actions, state, depth)
        # 视角换算统一走 is_deciding_player_turn：is_ally_turn 是"我方回合"，
        # 直接取反判断会让 opponent 侧永远被判成 turn_over（返回空路线）。
        turn_over = bool(getattr(state, "is_turn_ended", False)) or not is_deciding_player_turn(state, player)
        if turn_over or not actions or depth <= 0:
            return score_terminal(before, state, player=player), ()

        key = state_fingerprint(state, player)
        if key in path_keys:
            return None
        cached = memo.get(key)
        if cached is not None:
            counters["memo_hits"] += 1
            return cached

        current_best = best["score"]
        if bound_pruning and current_best is not None:
            bound = bounds.get(key)
            if bound is None:
                bound = (optimistic_bound(state, before, player=player), can_still_lethal(state, player=player))
                bounds[key] = bound
            net_bound, lethal_reachable = bound
            # 已经找到斩杀：任何当前还斩不掉的分支都比它差。
            if current_best.lethal and not lethal_reachable:
                counters["pruned"] += 1
                return None
            # 还没斩杀：净血差上界不如当前最优就剪（相等不剪——可能靠存活数/身材赢）。
            if not current_best.lethal and net_bound < current_best.net_life:
                counters["pruned"] += 1
                return None

        # 诊断安全网：超预算的分支按"当前状态已结算"处理。这只在显式传 node_budget 时
        # 生效，生产路径必须保持 None，否则搜索会退化成近似解。
        if node_budget is not None and counters["nodes"] >= node_budget:
            counters["budget_hits"] += 1
            return score_terminal(before, state, player=player), ()
        path_keys.add(key)
        counters["nodes"] += 1
        counters["branch_sum"] += len(actions)
        if len(actions) > counters["max_branch"]:
            counters["max_branch"] = len(actions)
        local_best: tuple[TurnScore, tuple] | None = None
        for action in sorted(actions, key=lambda item: _action_sort_key(state, item)):
            try:
                prediction = predictor.predict(state, action, player=player)
            except Exception:
                continue
            if prediction_filter is not None and not prediction_filter(prediction):
                continue
            sub = _dfs(prediction.snapshot, depth - 1, path_keys)
            if sub is None:
                continue
            candidate = (sub[0], (action,) + sub[1])
            if local_best is None or candidate[0].key() > local_best[0].key():
                local_best = candidate
                if best["score"] is None or candidate[0].key() > best["score"].key():
                    best["score"] = candidate[0]
        path_keys.discard(key)
        # 只缓存成功结果：缓存 None 会把「因路径上的环被剪」当成「这个状态无解」，
        # 从别的路径再次到达时就会漏掉更优解。
        if local_best is not None:
            memo[key] = local_best
        return local_best

    # ---- 分阶段搜索（phased） ------------------------------------------
    def _predict_or_none(state, action):
        """预测并应用 strict 过滤器；不可用返回 None（不静默放行）。"""
        try:
            prediction = predictor.predict(state, action, player=player)
        except Exception:
            return None
        if prediction_filter is not None and not prediction_filter(prediction):
            return None
        return prediction

    def _dfs_exchange(state, depth: int, path_keys: set, allow_fold_back: bool = True, was_full: bool = False):
        """阶段 3：攻击与回合结束的完整穷举（**不贪心**）。

        曾实现成贪心（按 ``_action_sort_key`` 取第一个可执行攻击），实测在 76 个决策点里
        有 15 个找到更差的解，其中一例直接漏掉斩杀。原因是"最优解场/打脸路线"本身需要
        搜索——贪心选不出它。攻击受 ``attacks_left`` 限制，分支天然有限，穷举得起。

        ``was_full`` 表示进入交换阶段时场地是满的。若交换腾出了格子，则折回出牌阶段
        （``allow_fold_back`` 只允许折回一次，避免 play↔exchange 循环）。
        """
        if depth <= 0:
            return score_terminal(before, state, player=player), ()
        if bool(getattr(state, "is_turn_ended", False)) or not is_deciding_player_turn(state, player):
            return score_terminal(before, state, player=player), ()

        actions = [
            action
            for action in enumerator.enumerate(state, player=player)
            if action.kind in _ATTACK_PHASE_KINDS
            or (action.kind is ActionKind.TURN_END and allow_turn_end)
        ]
        if not actions:
            return score_terminal(before, state, player=player), ()

        key = (_PHASE_EXCHANGE, state_fingerprint(state, player))
        if key in path_keys:
            return None
        cached = memo.get(key)
        if cached is not None:
            counters["memo_hits"] += 1
            return cached

        current_best = best["score"]
        if bound_pruning and current_best is not None:
            bound = bounds.get(key)
            if bound is None:
                bound = (optimistic_bound(state, before, player=player), can_still_lethal(state, player=player))
                bounds[key] = bound
            net_bound, lethal_reachable = bound
            if current_best.lethal and not lethal_reachable:
                counters["pruned"] += 1
                return None
            if not current_best.lethal and net_bound < current_best.net_life:
                counters["pruned"] += 1
                return None

        # 诊断安全网：与 _dfs 同语义。phased 是生产默认路径，缺这段会让
        # node_budget 在默认配置下静默失效（2026-11-26 修复）。
        if node_budget is not None and counters["nodes"] >= node_budget:
            counters["budget_hits"] += 1
            return score_terminal(before, state, player=player), ()

        path_keys.add(key)
        counters["nodes"] += 1
        counters["branch_sum"] += len(actions)
        if len(actions) > counters["max_branch"]:
            counters["max_branch"] = len(actions)
        phase_counts[_PHASE_EXCHANGE] = phase_counts.get(_PHASE_EXCHANGE, 0) + 1

        local_best: tuple[TurnScore, tuple] | None = None
        try:
            for action in sorted(actions, key=lambda item: _action_sort_key(state, item)):
                # 折回：准备结束回合时，若交换已腾出格子就回去把牌打完。
                if (
                    action.kind is ActionKind.TURN_END
                    and allow_fold_back
                    and was_full
                    and _field_count(state, player) < _FIELD_LIMIT
                ):
                    fold_play = [
                        item for item in enumerator.enumerate(state, player=player) if item.kind in _PLAY_PHASE_KINDS
                    ]
                    if fold_play:
                        folded = _dfs_phased(state, depth, path_keys, _PHASE_PLAY, allow_fold_back=False)
                        if folded is not None and (local_best is None or folded[0].key() > local_best[0].key()):
                            local_best = folded
                            if best["score"] is None or folded[0].key() > best["score"].key():
                                best["score"] = folded[0]
                prediction = _predict_or_none(state, action)
                if prediction is None:
                    continue
                sub = _dfs_exchange(prediction.snapshot, depth - 1, path_keys, allow_fold_back, was_full)
                if sub is None:
                    continue
                combined = (sub[0], (action,) + sub[1])
                if local_best is None or combined[0].key() > local_best[0].key():
                    local_best = combined
                    if best["score"] is None or combined[0].key() > best["score"].key():
                        best["score"] = combined[0]
        finally:
            path_keys.discard(key)

        if local_best is not None:
            memo[key] = local_best
        return local_best

    def _dfs_phased(state, depth: int, path_keys: set, phase: str, allow_fold_back: bool = True):
        # 交换阶段整体委托给 _dfs_exchange：它和本函数用同一个 (phase, fingerprint) key，
        # 若在这里先 add 再进去，会被自己的环检测判成环并返回 None。
        if phase == _PHASE_EXCHANGE:
            return _dfs_exchange(
                state,
                depth,
                path_keys,
                allow_fold_back=allow_fold_back,
                was_full=_field_count(state, player) >= _FIELD_LIMIT,
            )
        if depth <= 0:
            return score_terminal(before, state, player=player), ()
        if bool(getattr(state, "is_turn_ended", False)) or not is_deciding_player_turn(state, player):
            return score_terminal(before, state, player=player), ()

        actions = [
            action for action in enumerator.enumerate(state, player=player) if action.kind in _PLAN_KINDS
        ]
        if not allow_turn_end:
            actions = [action for action in actions if action.kind is not ActionKind.TURN_END]
        actions = _apply_prior_prune(actions, state, depth)
        if not actions:
            return score_terminal(before, state, player=player), ()

        # memo 的 key 必须含 phase：同一局面在「还能出牌」和「只能攻击」两个阶段
        # 的后续动作集不同，混用会互相污染。
        key = (phase, state_fingerprint(state, player))
        if key in path_keys:
            return None
        cached = memo.get(key)
        if cached is not None:
            counters["memo_hits"] += 1
            return cached

        current_best = best["score"]
        if bound_pruning and current_best is not None:
            bound = bounds.get(key)
            if bound is None:
                bound = (optimistic_bound(state, before, player=player), can_still_lethal(state, player=player))
                bounds[key] = bound
            net_bound, lethal_reachable = bound
            if current_best.lethal and not lethal_reachable:
                counters["pruned"] += 1
                return None
            if not current_best.lethal and net_bound < current_best.net_life:
                counters["pruned"] += 1
                return None

        # 诊断安全网：与 _dfs 同语义。phased 是生产默认路径，缺这段会让
        # node_budget 在默认配置下静默失效（2026-11-26 修复）。
        if node_budget is not None and counters["nodes"] >= node_budget:
            counters["budget_hits"] += 1
            return score_terminal(before, state, player=player), ()

        path_keys.add(key)
        counters["nodes"] += 1
        counters["branch_sum"] += len(actions)
        if len(actions) > counters["max_branch"]:
            counters["max_branch"] = len(actions)
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

        local_best: tuple[TurnScore, tuple] | None = None

        def _take(result, prefix: tuple = ()) -> None:
            nonlocal local_best
            if result is None:
                return
            combined = (result[0], prefix + result[1])
            if local_best is None or combined[0].key() > local_best[0].key():
                local_best = combined
                if best["score"] is None or combined[0].key() > best["score"].key():
                    best["score"] = combined[0]

        try:
            if phase == _PHASE_PLAY:
                play_actions = [action for action in actions if action.kind in _PLAY_PHASE_KINDS]
                # 出口：本阶段随时可以停止出牌，进入进化阶段（不消耗 depth）。
                _take(_dfs_phased(state, depth, path_keys, _PHASE_EVOLVE, allow_fold_back))
                for action in sorted(play_actions, key=lambda item: _action_sort_key(state, item)):
                    prediction = _predict_or_none(state, action)
                    if prediction is None:
                        continue
                    _take(
                        _dfs_phased(prediction.snapshot, depth - 1, path_keys, _PHASE_PLAY, allow_fold_back),
                        (action,),
                    )
            elif phase == _PHASE_EVOLVE:
                evolve_actions = [action for action in actions if action.kind in _EVOLVE_PHASE_KINDS]
                # 出口：也可以不进化（进化不是每回合都必须）。
                _take(_dfs_phased(state, depth, path_keys, _PHASE_EXCHANGE, allow_fold_back))
                for action in sorted(evolve_actions, key=lambda item: _action_sort_key(state, item)):
                    prediction = _predict_or_none(state, action)
                    if prediction is None:
                        continue
                    _take(
                        _dfs_phased(prediction.snapshot, depth - 1, path_keys, _PHASE_EVOLVE, allow_fold_back),
                        (action,),
                    )
        finally:
            path_keys.discard(key)

        if local_best is not None:
            memo[key] = local_best
        return local_best

    if search_mode == "phased":
        root_result = _dfs_phased(snapshot, max_actions, set(), _PHASE_PLAY)
        initial_play = [
            action
            for action in enumerator.enumerate(snapshot, player=player)
            if action.kind in _PLAY_PHASE_KINDS
        ]
        stop_reason = play_stop_reason(snapshot, player, play_actions=initial_play)
    else:
        root_result = _dfs(snapshot, max_actions, set())
        stop_reason = None
    if root_result is None:
        raise RuntimeError("plan_turn: 根节点没有任何可评分的终局（回合内没有合法动作）")

    score, route = root_result
    final_state = before
    for action in route:
        final_state = predictor.predict(final_state, action, player=player).snapshot
    return TurnPlan(
        route=list(route),
        score=score,
        before=before,
        after=final_state,
        nodes=counters["nodes"],
        pruned_by_bound=counters["pruned"],
        truncated=counters["budget_hits"] > 0,
        budget_hits=counters["budget_hits"],
        elapsed_seconds=time.perf_counter() - started_at,
        memo_hits=counters["memo_hits"],
        branch_sum=counters["branch_sum"],
        max_branch=counters["max_branch"],
        play_stop_reason=stop_reason,
        phase_visits=dict(phase_counts),
        prior_pruned=counters.get("prior_pruned", 0),
    )


__all__ = ["TurnPlan", "plan_turn", "optimistic_bound", "can_still_lethal"]
