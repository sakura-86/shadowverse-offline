# -*- coding: utf-8 -*-
"""Diff-based value function: 用回合开始/结束的场面 diff 作为 leaf evaluation。

设计思路：
- value 衡量的是"这个回合做了多少贡献"
- 基线 = 回合开始时的场面状态
- 得分 = 清场分 + 铺场分 + 打脸/回复/资源 + 纹章 + 护符
- 归一化后乘以最终惩罚系数（对方剩余随从越多越扣分）
- 什么都不做 = -1，满分操作 = +1

用于 PUCT 搜索的 leaf evaluation：
- 搜索路径走到 turn_end 时，用 turn_end after-state 与 turn_start 的 diff 计算分数
- 不依赖 value head，提供可靠的即时训练信号
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from decision_engine.state.snapshot import BattleSnapshot, CardView, PlayerView


# ──────────────────────────────────────────────────────────────────────
# 随从打分公式
# ──────────────────────────────────────────────────────────────────────

def score_follower(card: CardView) -> float:
    """对单个随从打分：固定底分 + atk×0.5 + life×增益倍率。
    
    固定底分 2：体现"多一个/少一个随从"本身的价值。
    增益倍率由关键字决定（可叠乘）：
    - 守护: ×1.1
    - 灵气(drain): ×1.3
    - 谢幕曲(last_words/has_last_words): ×1.5
    - 减伤(cant_be_damaged/damage_reduction): ×2.0
    """
    if card is None:
        return 0.0
    
    life_multiplier = 1.0
    if getattr(card, 'has_guard', False):
        life_multiplier *= 1.1
    if getattr(card, 'has_drain', False):
        life_multiplier *= 1.3
    if getattr(card, 'has_last_words', False):
        life_multiplier *= 1.5
    # 减伤：cant_be_attacked 或 damage_reduction 类属性
    if getattr(card, 'has_cant_be_attacked', False):
        life_multiplier *= 2.0
    
    return 2.0 + card.atk * 0.5 + card.life * life_multiplier


def score_field(cards: Sequence[CardView | None]) -> float:
    """对场面所有随从打分（总和）。"""
    return sum(score_follower(c) for c in cards if c is not None)


# ──────────────────────────────────────────────────────────────────────
# Diff 计算
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DiffScore:
    """回合 diff 各项分数明细。"""
    clear_score: float = 0.0       # 清场分（对方随从减少的分数）
    board_score: float = 0.0       # 铺场分（我方随从增加的分数 × 惩罚系数）
    damage_score: float = 0.0      # 打脸/回复/资源分
    crest_score: float = 0.0       # 纹章奖励
    amulet_score: float = 0.0      # 护符分
    raw_total: float = 0.0         # 归一化前总分
    final_penalty: float = 1.0     # 最终惩罚系数
    normalized: float = 0.0        # 归一化后最终分 [-1, 1]


def compute_diff_score(
    turn_start: BattleSnapshot,
    turn_end: BattleSnapshot,
    *,
    player: str = "me",
) -> DiffScore:
    """计算一个回合的 diff 分数。
    
    Args:
        turn_start: 回合开始时的 snapshot
        turn_end: 回合结束时的 after-state（turn_end 效果已执行）
        player: 计算视角
    
    Returns:
        DiffScore 包含各项明细和最终归一化分数
    """
    # 确定视角
    if player == "me":
        my_start = turn_start.me
        opp_start = turn_start.opponent
        my_end = turn_end.me
        opp_end = turn_end.opponent
    else:
        my_start = turn_start.opponent
        opp_start = turn_start.me
        my_end = turn_end.opponent
        opp_end = turn_end.me
    
    # 获取场面卡牌
    my_field_start = [turn_start.cards_by_uid.get(uid) for uid in my_start.field]
    opp_field_start = [turn_start.cards_by_uid.get(uid) for uid in opp_start.field]
    my_field_end = [turn_end.cards_by_uid.get(uid) for uid in my_end.field]
    opp_field_end = [turn_end.cards_by_uid.get(uid) for uid in opp_end.field]
    
    # 过滤 None
    my_field_start = [c for c in my_field_start if c is not None]
    opp_field_start = [c for c in opp_field_start if c is not None]
    my_field_end = [c for c in my_field_end if c is not None]
    opp_field_end = [c for c in opp_field_end if c is not None]
    
    result = DiffScore()
    
    # ── 1. 清场分 ──
    opp_score_start = score_field(opp_field_start)
    opp_score_end = score_field(opp_field_end)
    result.clear_score = max(0.0, opp_score_start - opp_score_end)
    
    # ── 2. 铺场分 ──
    my_score_start = score_field(my_field_start)
    my_score_end = score_field(my_field_end)
    raw_board = max(0.0, my_score_end - my_score_start)
    
    # 惩罚系数：max[(我方随从数-对方随从数)/我方随从数, 0]
    my_count_end = len(my_field_end)
    opp_count_end = len(opp_field_end)
    if my_count_end > 0:
        board_penalty = max((my_count_end - opp_count_end) / my_count_end, 0.0)
    else:
        board_penalty = 0.0
    result.board_score = raw_board * board_penalty
    
    # ── 3. 打脸/回复 ──
    # 打脸伤害 = 对方 life 减少量
    face_damage = max(0.0, opp_start.life - opp_end.life)
    # 回复 = 我方 life 增加量
    heal = max(0.0, my_end.life - my_start.life)
    # 对方场攻（回合结束时）= 未解场的威胁
    opp_field_atk_end = sum(c.atk for c in opp_field_end)
    
    # 打脸/回复分数：只有超过对方场攻的部分才有正收益
    # 逻辑：如果对面还有场攻 10，你打了 5 点脸，实际净收益为负（因为下回合对面打你 10）
    # 只有当打脸+回复 > 对方场攻时，超出部分才算有效收益
    effective_damage_heal = max(0.0, face_damage + heal - opp_field_atk_end)
    result.damage_score = 2.0 * effective_damage_heal
    
    # ── 3b. EP/SEP 资源消耗 ──
    ep_used = max(0, my_start.ep - my_end.ep)   # 本回合消耗的 EP
    sep_used = max(0, my_start.sep - my_end.sep)  # 本回合消耗的 SEP
    
    # EP 消耗是否有效：
    # - 如果对方回合开始时场上无随从 且 打脸伤害为 0 → EP 浪费（无解场/无直伤收益）
    # - 否则 EP 消耗是合理的（用于解场或打脸）
    ep_penalty = 0.0
    if ep_used > 0:
        had_targets = len(opp_field_start) > 0
        did_face_damage = face_damage > 0
        if not had_targets and not did_face_damage:
            # 无目标且无打脸 → 完全浪费
            ep_penalty = 4.0 * ep_used
        # 有目标或有打脸时不扣分（EP 用于解场/打脸是合理的）
    
    sep_penalty = 0.0
    if sep_used > 0:
        had_targets = len(opp_field_start) > 0
        did_face_damage = face_damage > 0
        if not had_targets and not did_face_damage:
            sep_penalty = 8.0 * sep_used
    
    # ── 3c. 额外PP 消耗 ──
    # 额外PP是否有效：使用后是否打出了 cost > 原始 max_pp 的牌
    extra_pp_penalty = 0.0
    extra_pp_start = getattr(my_start, 'extra_pp', None)
    extra_pp_end = getattr(my_end, 'extra_pp', None)
    extra_pp_was_used = False
    if extra_pp_start is not None and extra_pp_end is not None:
        if (not getattr(extra_pp_start, 'is_used_extra_pp_this_turn', False) and
            getattr(extra_pp_end, 'is_used_extra_pp_this_turn', False)):
            extra_pp_was_used = True
    
    if extra_pp_was_used:
        # 检查本回合是否打出了 cost > max_pp 的牌
        # 通过比较场上新增卡牌的 cost 来判断
        base_max_pp = my_start.max_pp  # 不含额外PP的原始 max_pp
        played_high_cost = False
        start_field_uids_set = set(my_start.field)
        for card in my_field_end:
            if card is not None and card.uid not in start_field_uids_set:
                if card.cost is not None and card.cost > base_max_pp:
                    played_high_cost = True
                    break
        if not played_high_cost:
            # 用了额外PP但没打出高费牌 → 浪费
            extra_pp_penalty = 6.0
    
    result.damage_score -= (ep_penalty + sep_penalty + extra_pp_penalty)
    
    # ── 4. 纹章奖励 ──
    # 新获得的纹章 = turn_end 有但 turn_start 没有的纹章
    start_crest_ids = set()
    for crest in my_start.crests:
        cid = getattr(crest, 'base_card_id', None) or getattr(crest, 'card_id', 0)
        start_crest_ids.add(cid)
    
    for crest in my_end.crests:
        cid = getattr(crest, 'base_card_id', None) or getattr(crest, 'card_id', 0)
        if cid and cid not in start_crest_ids:
            # 纹章来源卡的 cost：用 crest 的 card_id 推断
            # 纹章通常与来源卡关联，但 cost 不直接在 crest 对象上
            # 简化：用一个映射表或默认值
            crest_source_cost = _estimate_crest_source_cost(cid)
            result.crest_score += 2.0 * crest_source_cost
    
    # ── 5. 护符分 ──
    # 新增的护符 = turn_end 场上有但 turn_start 场上没有的护符
    start_field_uids = set(my_start.field)
    for card in my_field_end:
        if card is not None and card.uid not in start_field_uids:
            card_type = getattr(card, 'card_type', None)
            type_val = card_type.value if hasattr(card_type, 'value') else str(card_type)
            if type_val in ('amulet', 'countdown_amulet'):
                result.amulet_score += 2.0 * (card.cost or 0)
    
    # ── 总分 ──
    raw_positive = (
        result.clear_score
        + result.board_score
        + result.damage_score
        + result.crest_score
        + result.amulet_score
    )
    
    # ── 6. 手牌浪费惩罚 ──
    # 如果本回合消耗了手牌（手牌数减少）但正收益极低，说明打出的牌没有贡献
    # 每张浪费的手牌扣 3 分（一张手牌的机会成本）
    hand_start = len(my_start.hand)
    hand_end = len(my_end.hand)
    cards_played = max(0, hand_start - hand_end)  # 净消耗的手牌数
    
    # ── 6b. PP 最大值提升奖励 ──
    # 提升最大 PP 相当于下回合能多用 1 费，价值显著
    # 但当 max_pp >= 9 时不再有价值（下回合自然到 10）
    pp_ramp_bonus = 0.0
    max_pp_gained = max(0, my_end.max_pp - my_start.max_pp)
    if max_pp_gained > 0 and my_start.max_pp <= 8:
        # 只有 start max_pp <= 8 时提升才有价值
        effective_ramp = min(max_pp_gained, 9 - my_start.max_pp)  # 最多提升到 9（=下回合10）
        pp_ramp_bonus = 6.0 * effective_ramp  # 每提升 1 点 max_pp 值 6 分（> 手牌消耗 3 分）
    
    waste_penalty = 0.0
    if cards_played > 0:
        # 手牌边际成本：每消耗 1 张手牌记固定成本（资源的时间价值）。
        # 收益项（clear/board/damage/crest/amulet/pp_ramp）照常计入正分，
        # 手牌成本始终从总分扣除——正常出牌收益 > 成本→净正；
        # 空融合（消耗手牌但无场面/打脸收益）收益≈0→净负。
        #
        # 这修正了旧"全或无"逻辑的漏洞：旧逻辑只要整回合有任一正收益
        # 就豁免全部手牌消耗，导致"打1张好牌+空融3张"被误判为好回合。
        # 边际成本 C=2：与随从底分(2.0)同量纲，空融合明确转负，
        # 而高收益操作（board/damage 远超成本）几乎不受影响。
        _HAND_MARGINAL_COST = 2.0
        waste_penalty = _HAND_MARGINAL_COST * cards_played

    result.raw_total = raw_positive + pp_ramp_bonus - waste_penalty
    
    # ── 最终惩罚系数 ──
    # 1 - 0.1 × 对方剩余随从数
    result.final_penalty = max(0.0, 1.0 - 0.1 * opp_count_end)
    
    return result


def normalize_diff_score(score: DiffScore) -> float:
    """将 diff 分数通过 tanh 非线性归一化到 [-1, 1]。
    
    使用 tanh(raw_score / scale) 实现：
    - 低分段变化明显（区分度高）
    - 高分段边际递减（自然压缩）
    - 不依赖动态 max_score，同一输入永远返回同一输出
    - 自动限制在 [-1, 1] 范围内
    
    scale=15：得 15 分时 Q≈0.76，得 30 分时 Q≈0.96
    """
    from math import tanh as _tanh
    
    # 先应用最终惩罚系数到 raw_total
    penalized = score.raw_total * score.final_penalty
    
    # tanh 归一化
    return _tanh(penalized / _DIFF_NORMALIZE_SCALE)


# tanh 归一化的 scale 参数（控制"多少分≈Q=0.76"）
_DIFF_NORMALIZE_SCALE = 15.0


# ──────────────────────────────────────────────────────────────────────
# 纹章来源 cost 映射
# ──────────────────────────────────────────────────────────────────────

# 已知的纹章 card_id → 来源卡 cost 映射
_CREST_SOURCE_COST_MAP = {
    10864112: 7,   # 飞跃的姐妹 纹章 ← 10864110 (7费)
    10204122: 3,   # ← 10204120 (3费)
    10264122: 6,   # ← 10264120 (6费)
    10364122: 4,   # ← 10364120 (4费)
}


def _estimate_crest_source_cost(crest_card_id: int) -> float:
    """估算纹章来源卡的 cost。"""
    if crest_card_id in _CREST_SOURCE_COST_MAP:
        return _CREST_SOURCE_COST_MAP[crest_card_id]
    # 未知纹章，给一个保守默认值
    return 4.0



# ──────────────────────────────────────────────────────────────────────
# SearchValue 接口实现
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DiffValueResult:
    """满足 SearchValue 协议的返回值。"""
    value: float
    source: str = "diff_value"
    terminal: bool = False
    diagnostics: dict = None

    def __post_init__(self):
        if self.diagnostics is None:
            self.diagnostics = {}

    def validate(self) -> None:
        if self.value < -1.0 or self.value > 1.0:
            raise ValueError(f"diff value must be in [-1.0, 1.0], got {self.value}")


class DiffSearchValue:
    """基于回合 diff 的 SearchValue 实现。
    
    用于 PUCT 搜索的 leaf evaluation：
    - 持有 turn_start snapshot 作为基线
    - evaluate() 时虚拟执行 turn_end，用 turn_end after-state 计算 diff
    - 保证所有节点（中间/叶节点）的评估语义一致：
      "如果此刻选择结束回合，回合结束效果执行完毕后场面有多好"
    
    使用方式：
        diff_value = DiffSearchValue(turn_start_snapshot=snapshot, predictor=predictor)
        search = PuctSearch(..., value=diff_value)
    """
    
    source = "diff_value"
    
    def __init__(self, turn_start_snapshot: BattleSnapshot, *, player: str = "me", predictor=None):
        self._turn_start = turn_start_snapshot
        self._player = player
        self._predictor = predictor
    
    def evaluate(self, snapshot: BattleSnapshot, *, player: str = "me") -> DiffValueResult:
        """计算当前 snapshot 虚拟 turn_end 后相对于 turn_start 的 diff value。
        
        对所有节点统一：虚拟执行 turn_end → 用 after-state 算 diff。
        如果虚拟 turn_end 失败（predictor 不可用或执行异常），fallback 用当前 snapshot 直接算。
        """
        # 虚拟执行 turn_end 得到 after-state
        eval_snapshot = self._virtual_turn_end(snapshot, player)
        
        # 计算 diff
        diff = compute_diff_score(self._turn_start, eval_snapshot, player=player)
        
        # tanh 归一化
        value = normalize_diff_score(diff)
        
        # Clamp 安全
        value = max(-1.0, min(1.0, value))
        
        return DiffValueResult(
            value=value,
            source="diff_value",
            terminal=False,
            diagnostics={
                "raw_total": diff.raw_total,
                "penalized": diff.raw_total * diff.final_penalty,
                "clear": diff.clear_score,
                "board": diff.board_score,
                "damage": diff.damage_score,
                "penalty": diff.final_penalty,
                "virtual_turn_end": eval_snapshot is not snapshot,
            },
        )
    
    def _virtual_turn_end(self, snapshot: BattleSnapshot, player: str) -> BattleSnapshot:
        """虚拟执行 turn_end，返回 after-state。失败时返回原 snapshot。"""
        if self._predictor is None:
            return snapshot
        
        # 如果 snapshot 已经是 turn_ended 状态，不需要再执行
        if getattr(snapshot, 'is_turn_ended', False):
            return snapshot
        
        try:
            from decision_engine.actions.model import ActionKind, LegalAction
            from decision_engine.execution.execution_context import LocalExecutionContext
            
            turn_end_action = LegalAction(kind=ActionKind.TURN_END, source_uid=0)
            result = self._predictor.predict_with_replay_context(
                snapshot, turn_end_action, player=player,
                execution_context=LocalExecutionContext(branch_seed=0),
            )
            if result is not None and result.snapshot is not None:
                return result.snapshot
        except Exception:
            pass
        
        return snapshot
    
    @property
    def max_raw_score(self) -> float:
        """已废弃，保留接口兼容。"""
        return 0.0
    
    def reset(self, turn_start_snapshot: BattleSnapshot, *, player: str = "me"):
        """重置为新回合的基线（复用实例时使用）。"""
        self._turn_start = turn_start_snapshot
        self._player = player


# ──────────────────────────────────────────────────────────────────────
# DiffAwarePuctSearch: 自动管理 turn_start baseline 的 PuctSearch 包装
# ──────────────────────────────────────────────────────────────────────

class DiffAwarePuctSearch:
    """包装 PuctSearch，自动在每回合重置 DiffSearchValue 的 baseline。
    
    与 SelfPlaySearchRunner 兼容：它在每步调用 search.search(snapshot, player=...)，
    本类追踪当前回合的 turn_start snapshot，并在每次 search 前更新 DiffSearchValue。
    
    用法：
        diff_search = DiffAwarePuctSearch(
            enumerator=enumerator,
            child_enumerator=enumerator,
            predictor=predictor,
            prior=prior_fn,
            turn_transitioner=...,
            config=puct_config,
        )
        # 传给 SelfPlaySearchRunner 当作 search 使用
        runner = SelfPlaySearchRunner(predictor=predictor, search=diff_search, config=...)
    """
    
    def __init__(self, *, enumerator, child_enumerator, predictor, prior, turn_transitioner, config):
        from decision_engine.planning.puct import PuctSearch
        
        self._enumerator = enumerator
        self._child_enumerator = child_enumerator
        self._predictor = predictor
        self._prior = prior
        self._turn_transitioner = turn_transitioner
        self._config = config
        
        # 当前回合的 diff value（每回合重置）
        self._diff_value = None
        self._current_turn_start = None
        self._current_player = None
        
        # 内部 PuctSearch 实例（每次 search 时重建 value）
        self._search = None
    
    @property
    def config(self):
        return self._config
    
    @property
    def prior(self):
        return self._prior
    
    @property
    def value(self):
        return self._diff_value
    
    def search(self, snapshot: BattleSnapshot, *, player: str = "me"):
        """执行搜索，自动管理 DiffSearchValue 的 turn_start baseline。
        
        如果 player 或回合切换了，自动重置 baseline。
        """
        from decision_engine.planning.puct import PuctSearch
        
        # 检测是否需要重置 baseline（新回合或 player 变化）
        if self._current_player != player or self._current_turn_start is None:
            self._current_turn_start = snapshot
            self._current_player = player
            self._diff_value = DiffSearchValue(
                turn_start_snapshot=snapshot, player=player,
                predictor=self._predictor,
            )
        
        # 构建/重建 PuctSearch（value 可能变了）
        self._search = PuctSearch(
            enumerator=self._enumerator,
            child_enumerator=self._child_enumerator,
            predictor=self._predictor,
            prior=self._prior,
            value=self._diff_value,
            turn_transitioner=self._turn_transitioner,
            config=self._config,
        )
        
        return self._search.search(snapshot, player=player)
    
    def notify_turn_end(self):
        """外部通知回合结束，下次 search 时会重置 baseline。"""
        self._current_turn_start = None
        self._current_player = None
