# -*- coding: utf-8 -*-
"""逐动作即时收益价值模型（Per-Action Incremental Value）。

设计见 docs/plan/PER_ACTION_INCREMENTAL_VALUE_PLAN.md。

核心：节点价值 = 从回合起始到该节点每个动作的即时收益 step_reward 累加。
- 移除单回合补偿系数（final_penalty/board_penalty/减对方场攻/资源惩罚/tanh）——
  多回合搜索让这些"预估未来"的系数被真实即时后果取代。
- 逐动作调用天然不丢"过程损失"（增幅达阈值变身、增幅牌当回合打出）。
- 冷启动阶段地平线叶记 0（value head 只训练不参与）。

符号约定（方案 a）：所有项以 root_player 的 my/opp 侧计算状态增量。
对手行动导致我方 life 下降 → 自然记为负，无需按行动方翻转符号。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.execution.rule_kernel import MAX_FIELD_SIZE, deckout_winner_from_results
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView


# ---------------------------------------------------------------------------
# 权重常量（S5 标定，先给合理初值；复用 score_follower 量纲：底分2 + atk×0.5）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRewardWeights:
    """PP 本位标定权重（1 PP = 10 分；见 docs/plan/ENGINE_POTENTIAL_SHAPING_DESIGN.md §11）。

    标定体系经领域专家多轮压力测试，从真实锚卡反解收敛（§11.2 原子锚点）：
    - 1 PP = 1 点打脸 = 1 点有效回复 = 10 分（硬锚）
    - EP = 3 PP = 30（EP→PP 方向定价；稀缺战略资源买入含溢价，用途更广，优于旧 2PP）
    - SEP = 4.5 PP = 45（= 1.5 × EP；抽牌/身材 buff 双维度 1.5 倍）
    - 增幅 = 0.5 PP = 5（1 有效增幅半价）
    - 最大 PP = 1 PP/点 = 10（静态资源读数，敌我对称；双方到 10 封顶时领先自动归零）
    - 手牌 1 张 = 1.5 PP = 15（抽牌/生成的价值全部由手牌张数变化吸收，不设独立抽牌项）

    §11 统一原则：势 = 所有可观测状态按 PP 读数；动作效果（伤害/回复/解场/抽牌/
    获取资源/跳费）不单独记"动作奖励"，其价值已体现在它改变的状态量里（telescoping）。
    board 维度过程即计（w_count 数量项 + atk/life 细粒度），替换旧的地平线叶 pp_of_stat
    超线性残值。最大 PP 由静态资源读数体现（非跳费动作奖励），避免后期封顶后仍虚高。
    量纲：1 PP = 10 分。
    """
    w_face: float = 10.0         # 每点打脸伤害（锚：1 PP）
    w_heal: float = 10.0         # 每点有效回复（对称打脸；溢出因血量读数不涨自动不计）
    w_boost: float = 5.0         # 每次有效魔力增幅（0.5 PP，朝质变阈值推进）
    w_boost_linear: float = 5.0  # 无阈值线性增幅每次（0.5 PP 当量）
    # 预支系数：手牌中有饱和点的增幅卡时，把"完成积累的回报"按进度权重计入持有势。
    # 形式：R × (sb/cap)^power，R = 最高阈值的总 reward。
    # - 0（默认）→ 关闭，行为与 §12.19 完全一致；这是实验参数，验证到能提升胜率前不默认开。
    # - power=0 → 整份常量（零进度也给满 R，实测会诱发"囤积不打"）
    # - power=2 → 渐进（零进度给 0，接近满档才快速上升；"临门一脚"激励）
    w_boost_anticipation: float = 0.0
    w_boost_anticipation_power: float = 2.0
    # 阈值合并：只用该卡的最高档作为唯一饱和点，抹平多档折点。
    # 为什么：多档进度势的边际在跨过中间档后骤降（坦忒拉 10 档：7.5 → 2.5 分/次），
    # 而打一张纯法术燃料的手牌成本固定 15 分，于是"越接近满档越不划算"，与直觉相反。
    # 合并成单档后每增幅边际恒定；中间档的实物收益（衍生卡进手牌）仍由 w_hand 计价，
    # 不会因为势不再单独记账就变得不可观测。默认关闭 = 与既有行为完全一致。
    boost_single_tier: bool = False
    # 进度势曲线指数：Φ ∝ (progress)^power。1.0 = 线性（现状）；>1 = 凸，后段加速，
    # "临门一脚"更值钱（低档兑现的机会成本变小）。总量守恒：sb=cap 时恒为 Σ reward。
    w_boost_curve_power: float = 1.0
    # 进度势整体缩放：1.0 = 现状；0.0 = 完全移除"持有增幅势"（配合 w_boost_gate 使用）。
    # 存在的理由见 w_boost_gate：进度势让"攒的激励"和"打出的机会成本"由同一个 Φ 决定，
    # 任何缩放都会同时移动两边，所以需要能把它单独关掉。
    w_boost_potential_scale: float = 1.0
    # 跨阈值闸门奖励：sb 首次跨过某个阈值 t_i 的那一步，一次性给 reward(t_i) × coef。
    # **不进 Φ**，所以拿到的回报不会在打出时被扣回——这是把"攒的激励"与"打出的机会成本"
    # 解耦的唯一途径。代价：它不是 potential-based（路径无关性丧失），
    # 需要用小系数并配合 w_boost_potential_scale 控制总量。0 = 关闭（默认，行为不变）。
    w_boost_gate: float = 0.0
    # 满档释放（"达成阈值即释放"）：卡已经攒到最高阈值时，打出**不再回收**它的持有势。
    #
    # 为什么是这一条而不是别的："攒的激励"（逐次 +Φ 差分）和"打出的机会成本"（3b 块扣回
    # Φ）在纯势函数里是同一个数，任何缩放都同时移动两边。而实验证明两头都失败：抬高持有
    # 势 → 满档不打（囤积），压低持有势 → 低档就乱打（§12.25/12.26）。
    #
    # 本开关只切一刀：低档打出仍然贵（早打被惩罚），满档打出免费（该打就打）。
    # 代价：3b 块不再是严格 Φ 差分 ⇒ 不再满足 potential-based 的路径无关性，
    # 属于受控偏置。默认 False = 行为与旧版逐位一致。
    boost_release_at_cap: bool = False
    w_hand: float = 15.0         # 每张手牌（1.5 PP；抽牌/生成价值全由手牌张数吸收）
    w_crest: float = 10.0        # 每点纹章来源费用（卡PP/2 由调用处折半，此处为 PP→分）
    w_amulet: float = 10.0       # 每点护符费用（PP→分；四类公式在 _amulet_gain 内）
    w_countdown_progress: float = 10.0  # 己方倒计时每推进1（未来固定效果提前1回合，≈1 PP）
    # board 维度（§11.3，过程即计，敌我对称）：
    w_board_count: float = 30.0  # 每个随从底分（解场位 ≈ 单解 3 PP；保证数量优于身材）
    w_board_atk: float = 4.0     # 每点 atk（打脸威胁/换场主动性；0.4 PP）
    w_board_life: float = 1.0    # 每点 life（被动耐久，细粒度区分 5/2 vs 5/3；0.1 PP）
    # 每点"额外攻击次数"的预期伤害期权（1 PP/点，与 w_face 同量纲）：
    # 双击/多次攻击时 attacks_left≥2 的那部分攻击是尚未兑现的伤害，必须在这里可见，
    # 否则"给双击"这个动作在势里是 +0。与血量维度对冲，不双重计价（见 _board_signal 注释）。
    w_board_extra_attack: float = 10.0
    w_max_pp: float = 10.0       # 每点最大 PP（1 PP；静态资源读数，敌我对称，封顶后领先归零）
    w_pp_temp: float = 10.0      # 每点一次性 PP 恢复（1 PP）
    w_ep: float = 30.0           # 每点 EP 变化（3 PP；进化消耗记负成本 / 恢复记正收益）
    w_sep: float = 45.0          # 每点 SEP 变化（4.5 PP = 1.5 EP）
    v_win: float = 1000.0        # 终局胜负（量纲上压倒中间收益）


DEFAULT_WEIGHTS = StepRewardWeights()


def weights_from_env(base: StepRewardWeights = DEFAULT_WEIGHTS) -> StepRewardWeights:
    """从环境变量覆盖权重，供对照实验扫参数（不改代码、不污染默认）。

    支持覆盖的环境变量（对应 StepRewardWeights 字段，全大写加前缀 EP_）：
      EP_W_BOARD_COUNT / EP_W_BOARD_ATK / EP_W_BOARD_LIFE / EP_W_BOOST / EP_W_FACE ...
      布尔字段（EP_BOOST_SINGLE_TIER）用 1/0 或 true/false 覆盖。
    未设置的字段保持 base 默认。仅用于实验编排，生产默认不设任何 EP_ 变量。
    """
    import os
    from dataclasses import fields, replace
    overrides = {}
    for f in fields(StepRewardWeights):
        env_key = "EP_" + f.name.upper()
        val = os.environ.get(env_key)
        if val is None:
            continue
        if isinstance(f.default, bool):
            overrides[f.name] = val.strip().lower() in ("1", "true", "yes", "on")
            continue
        try:
            overrides[f.name] = float(val)
        except ValueError:
            pass
    return replace(base, **overrides) if overrides else base


def _winner(snapshot: BattleSnapshot) -> str | None:
    """终局判断（绝对方："me"/"opponent"/"simultaneous"/None）。与 search_value._winner 一致。"""
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
    """返回 (my_view, opp_view) —— 始终以 root_player 侧为 my。"""
    if root_player == "me":
        return snapshot.me, snapshot.opponent
    return snapshot.opponent, snapshot.me


def _field_cards(snapshot: BattleSnapshot, view) -> list[CardView]:
    cards = [snapshot.cards_by_uid.get(uid) for uid in view.field]
    return [c for c in cards if c is not None]


def _hand_boost_by_uid(snapshot: BattleSnapshot, view) -> dict[int, tuple[int, int]]:
    """uid -> (spell_boost_count, base_card_id) for 手牌中受增幅牌。"""
    out: dict[int, tuple[int, int]] = {}
    for uid in view.hand:
        c = snapshot.cards_by_uid.get(uid)
        if c is not None and getattr(c, "has_spell_boost", False):
            out[uid] = (int(getattr(c, "spell_boost_count", 0)), int(c.base_card_id))
    return out


# threshold_fn(base_card_id) -> tuple[int, ...] 增幅质变阈值列表（S1 提供）。None 表示无阈值查表（全线性）。
ThresholdFn = Callable[[int], Sequence[int]]


# ---------------------------------------------------------------------------
# S1: 增幅阈值查表（从卡牌数据 supplement_texts 解析）
# ---------------------------------------------------------------------------


def build_spellboost_threshold_table(card_database_path: str | None = None) -> dict[int, tuple[int, ...]]:
    """分档解锁型增幅卡的"质变阈值"列表（委托 ``cards/spellboost.py``）。

    阈值提取是卡牌知识，唯一实现在 ``decision_engine.cards.spellboost``：value 侧
    （本模块）和 policy 侧（``training/state_features.py`` 的进度饱和度）必须同源，
    否则两边对"这张攒满了没有"会给出不一致的读数。

    supplement_texts 里：
    - identifier == "variable_x_threshold_hand"  args=["5"]      → 阈值 (5,)
    - identifier == "variable_x_threshold2_hand" args=["10","20"] → 阈值 (10, 20)
    - identifier == "variable_x_hand"（无阈值，纯线性 X 成长，如暴风破）→ 不入表（视为线性）

    返回 base_card_id -> 阈值元组（升序）。无阈值/未受增幅牌不在表中。
    """
    from decision_engine.cards.spellboost import boost_threshold_table

    return boost_threshold_table(card_database_path)


def make_threshold_fn(table: dict[int, tuple[int, ...]]) -> ThresholdFn:
    """把阈值查表包成 threshold_fn。表中无此牌 → 返回 ()（线性）。"""
    def fn(base_card_id: int) -> tuple[int, ...]:
        return table.get(int(base_card_id), ())
    return fn


_DEFAULT_THRESHOLD_TABLE: dict[int, tuple[int, ...]] | None = None


def default_threshold_fn() -> ThresholdFn:
    """默认阈值函数（懒加载缓存全局查表）。"""
    global _DEFAULT_THRESHOLD_TABLE
    if _DEFAULT_THRESHOLD_TABLE is None:
        _DEFAULT_THRESHOLD_TABLE = build_spellboost_threshold_table()
    return make_threshold_fn(_DEFAULT_THRESHOLD_TABLE)


def _boost_potential(sb: int, base_card_id: int,
                     threshold_fn: ThresholdFn | None, w: StepRewardWeights) -> float:
    """一张受增幅牌在 spell_boost_count=sb 时的**持有态势**（绝对值，分制，1PP=10分）。

    多阈值进度势叠加（用户定稿）：每个阈值 t_i 独立贡献一份"进度势"，总势 = 各档之和。
      每档 reward(t_i) = (t_i − t_{i-1}) × 0.5PP        （新增解锁跨度的一半，t_0=0）
      每档进度势        = reward(t_i) × min(sb, t_i) / t_i
      engine_potential  = Σ_i  reward(t_i) × min(sb, t_i) / t_i

    精妙处：低档增幅同时为所有更高档凝聚进度势。在低档提前兑现（卡离手）会**丢失为高档
    积攒的进度势**（telescoping 自然扣除），这正是"忍到最高阈值"的真实机会成本——无需人为
    注入 bias，凸性与"早兑现惩罚"从机制内在结构涌现（纯 potential-based，无偏）。

    例（坦忒拉，阈值 10/20；reward 各 (10-0)*.5=5PP、(20-10)*.5=5PP）：
      sb=5 → 5*5/10 + 5*5/20 = 2.5+1.25 = 3.75PP
      sb=10→ 5 + 2.5 = 7.5PP ；sb=20→ 5 + 5 = 10PP
    （分制 ×10：0.5PP ↔ w_boost=5。）

    无阈值牌：线性 sb × w_boost_linear（每增幅 0.5PP 当量，不饱和）。

    两个形状参数（默认关闭，实验用）：
    - ``boost_single_tier``：只用最高档，抹平多档折点，边际不再在中间档后骤降。
    - ``w_boost_curve_power``：进度项改为 ``(min(sb,t)/t)^power``，>1 时后段加速。
      两者都**保持总量守恒**（sb 达到最高档时仍等于 Σ reward），只改支付曲线形状，
      因此仍是合法 potential（路径无关、不改变终局最优策略的保证不变）。
    """
    thresholds = tuple(sorted(threshold_fn(base_card_id))) if threshold_fn is not None else ()
    if not thresholds:
        return max(0, sb) * w.w_boost_linear
    sb = max(0, sb)
    if w.boost_single_tier:
        thresholds = (thresholds[-1],)
    curve_power = w.w_boost_curve_power
    # w_boost = 0.5PP 的分制系数；reward(t_i) 分制 = (t_i - t_{i-1}) × w_boost
    total = 0.0
    prev_t = 0
    for t in thresholds:
        if t <= 0:
            continue
        span = t - prev_t                      # 该档新增解锁跨度
        reward = span * w.w_boost              # (跨度) × 0.5PP（分制）
        ratio = min(sb, t) / t                 # 进度比例
        total += reward * (ratio ** curve_power if curve_power != 1.0 else ratio)
        prev_t = t
    # 预支：把手牌中这张卡"完成积累的回报"按进度权重记上。
    # power=0 → 整份常量（零进度也给 R，边际不变）；
    # power>0 → 渐进，(sb/cap)^power 让低进度几乎不预支、接近满档才快速上升，
    #           于是"再攒一点"的边际在临门处被叠加放大，而"低进度打出"仍然便宜。
    if prev_t > 0 and w.w_boost_anticipation:
        ratio = min(sb, prev_t) / prev_t
        power = w.w_boost_anticipation_power
        weight = 1.0 if power <= 0 else ratio ** power
        total += prev_t * w.w_boost * w.w_boost_anticipation * weight
    return total


def _boost_marginal_value(
    prev_sb: int, next_sb: int, base_card_id: int,
    threshold_fn: ThresholdFn | None, w: StepRewardWeights,
) -> float:
    """本步增幅的边际势 = 持有态势之差（路径无关，首尾差）。

    保留此函数名与签名（step_reward 与单元测试复用）；内部改为多阈值进度势叠加的差分。
    """
    return _boost_potential(next_sb, base_card_id, threshold_fn, w) \
        - _boost_potential(prev_sb, base_card_id, threshold_fn, w)


def _at_boost_cap(
    sb: int, base_card_id: int, threshold_fn: ThresholdFn | None,
) -> bool:
    """该卡是否已攒到最高阈值（无阈值卡恒为 False）。"""
    if threshold_fn is None:
        return False
    thresholds = tuple(sorted(threshold_fn(base_card_id)))
    if not thresholds:
        return False
    return max(0, sb) >= max(thresholds)


def _boost_gate_reward(
    prev_sb: int, next_sb: int, base_card_id: int,
    threshold_fn: ThresholdFn | None, w: StepRewardWeights,
) -> float:
    """本步**跨过**的每个阈值的闸门奖励（一次性，不进 Φ，因此打出时不会被扣回）。

    与 ``_boost_marginal_value`` 的本质区别：后者是势的差分，攒到的势挂在卡上，打出时
    连同它一起扣掉（= 打出的机会成本）；这里发出去就是发出去了，卡离手也不回收。
    这正是"达成阈值即释放"的可实现版本——代价是它不是 potential-based。

    仅当 ``prev_sb < t_i <= next_sb`` 时对第 i 档发放 ``reward(t_i) × w_boost_gate``；
    单步跨多档逐档发放，不会漏档。``w_boost_gate == 0`` 时恒为 0（默认关闭）。
    """
    if not w.w_boost_gate:
        return 0.0
    thresholds = tuple(sorted(threshold_fn(base_card_id))) if threshold_fn is not None else ()
    if not thresholds:
        return 0.0
    prev_sb = max(0, prev_sb)
    next_sb = max(0, next_sb)
    if w.boost_single_tier:
        thresholds = (thresholds[-1],)
    total = 0.0
    prev_t = 0
    for t in thresholds:
        if t <= 0:
            continue
        if prev_sb < t <= next_sb:
            total += (t - prev_t) * w.w_boost * w.w_boost_gate
        prev_t = t
    return total


def _hand_value(snapshot: BattleSnapshot, view, weights: StepRewardWeights) -> float:
    """手牌读数：普通卡按 ``w_hand``，分档奖励卡按"解锁它所需的增幅势"计价。

    为什么不能一视同仁：坦忒拉达到 10/20 档各给一张 **1 费**奖励卡（苍奏之四 /
    宏大的回归），但这两张卡的牌效约等于 5 费卡。若按普通手牌 ``w_hand``(1.5PP) 计，
    解锁瞬间势会凭空蒸发——打出时 3b 块没收 ``Φ(20)=10PP``，而两张奖励卡只记 3PP，
    净 −7PP，于是"越攒越不该打"，与真实收益方向相反。

    按解锁跨度计价后，没收的 10PP 正好由两张奖励卡承接（10 增幅 × 0.5PP = 5PP 各一张），
    势在"坦忒拉手牌 → 奖励卡手牌"之间守恒，不会无故下降。

    注意 ``view.hand`` 是 **uid 列表**（不是 CardView），必须回 ``snapshot.cards_by_uid``
    取卡；直接对 uid 取属性会静默退化成"按张数计价"。
    """
    from decision_engine.cards.spellboost import boost_reward_card_spans
    spans = boost_reward_card_spans()
    total = 0.0
    for uid in view.hand:
        card = snapshot.cards_by_uid.get(uid)
        bid = int(getattr(card, "base_card_id", 0) or 0) if card is not None else 0
        span = spans.get(bid)
        if span:
            total += span * weights.w_boost
        else:
            total += weights.w_hand
    return total


def step_reward(
    prev: BattleSnapshot,
    action: LegalAction,
    next_snapshot: BattleSnapshot,
    root_player: str,
    *,
    weights: StepRewardWeights = DEFAULT_WEIGHTS,
    threshold_fn: ThresholdFn | None = None,
) -> float:
    """单个动作对 root_player 的即时净收益（无终局系数、无 tanh）。

    所有项以 root_player 的 my/opp 侧计算状态增量（方案 a）：
    对手行动导致我方 life↓ 自然记负。
    """
    # 终局：动作后若已分胜负，直接返回大数值（覆盖中间收益）
    winner = _winner(next_snapshot)
    if winner is not None:
        if winner == "simultaneous":
            return 0.0
        return weights.v_win if winner == root_player else -weights.v_win

    my_prev, opp_prev = _sides(prev, root_player)
    my_next, opp_next = _sides(next_snapshot, root_player)

    reward = 0.0

    # §11 统一原则：势 = 所有可观测状态按 PP 读数之差（telescoping）。
    # board / 血量 / 手牌张数 / max_pp / EP·SEP 均为「状态读数」，动作效果不单独记奖励。

    # 1. 打脸：对方 life 下降（推进唯一胜利条件）
    reward += max(0.0, opp_prev.life - opp_next.life) * weights.w_face
    # 我方被打脸（对手回合随从攻击/直伤）记负——这是"不解场"后果的真实涌现
    reward -= max(0.0, my_prev.life - my_next.life) * weights.w_face

    # 2. 回复：我方 life 上升（溢出因 life 读数不涨自动不计，无需 clamp）
    reward += max(0.0, my_next.life - my_prev.life) * weights.w_heal

    # 3. 魔力增幅：手牌 spell_boost_count 增量（朝阈值边际价值，逐动作不丢过程）
    boost_prev = _hand_boost_by_uid(prev, my_prev)
    boost_next = _hand_boost_by_uid(next_snapshot, my_next)
    for uid, (nsb, bid) in boost_next.items():
        psb = boost_prev.get(uid, (0, bid))[0] if uid in boost_prev else 0
        if weights.w_boost_potential_scale:
            reward += weights.w_boost_potential_scale * _boost_marginal_value(
                psb, nsb, bid, threshold_fn, weights)
        # 闸门奖励：跨阈一次性发放，不挂在卡上，所以打出不会被扣回。
        reward += _boost_gate_reward(psb, nsb, bid, threshold_fn, weights)
    # 3b. 卡离手（打出/被弃/回牌组）：它的整份持有势消失，必须如实扣回。
    #     旧实现只遍历 next 的手牌，离手卡根本不参与差分 ⇒ "提前兑现"在势里零代价，
    #     这既违反 §6 的 telescoping 守恒，也正是早兑现学不会的直接原因。
    #     只扣"进度势"那部分（闸门奖励已发放，不在 Φ 里，无需回收）。
    if weights.w_boost_potential_scale:
        for uid, (psb, bid) in boost_prev.items():
            if uid in boost_next:
                continue
            # 满档释放：已经攒到最高阈值的卡打出去不再回收持有势（低档仍然回收）。
            if weights.boost_release_at_cap and _at_boost_cap(psb, bid, threshold_fn):
                continue
            reward -= weights.w_boost_potential_scale * _boost_potential(
                psb, bid, threshold_fn, weights)

    # 4. 手牌读数差值：吸收抽牌/生成（+）与打出/消耗（−）。
    #    不设独立抽牌项——所有过牌/生成的价值都体现在手牌读数变化上（§11，用户修正）。
    #    这一项同时取代旧的 w_draw（手牌增）与 c_hand（手牌耗）：读数差一把尺子。
    #    分档奖励卡按"解锁所需增幅势"计价，与 3b 块的没收额守恒（见 _hand_value）。
    reward += _hand_value(next_snapshot, my_next, weights) - _hand_value(prev, my_prev, weights)

    # 5. 纹章 / 护符（费用当量）
    reward += _crest_gain(prev, next_snapshot, my_prev, my_next) * weights.w_crest
    reward += _amulet_gain(prev, next_snapshot, my_prev, my_next) * weights.w_amulet
    reward += _countdown_progress(prev, next_snapshot, my_prev, my_next) * weights.w_countdown_progress
    reward -= _countdown_progress(prev, next_snapshot, opp_prev, opp_next) * weights.w_countdown_progress

    # 6. board 维度（§11.3，过程即计，敌我对称状态读数之差）：
    #    每随从 w_board_count（解场位，数量优先）+ atk×w_board_atk + life×w_board_life。
    #    随从兑现（打脸→life diff 已在项1、被解→读数降）由 telescoping 自然处理。
    board_before = _board_signal(prev, my_prev, weights) - _board_signal(prev, opp_prev, weights)
    board_after = _board_signal(next_snapshot, my_next, weights) - _board_signal(next_snapshot, opp_next, weights)
    reward += board_after - board_before

    # 7. 最大 PP（静态资源读数之差 × 1 PP，敌我对称）：跳费的价值由读数领先涌现；
    #    双方到 10 封顶时领先自动归零，不再虚高（替代旧 w_pp_ramp 动作奖励）。
    max_pp_before = my_prev.max_pp - opp_prev.max_pp
    max_pp_after = my_next.max_pp - opp_next.max_pp
    reward += (max_pp_after - max_pp_before) * weights.w_max_pp

    # 9. EP / SEP：双侧交换（我方消耗记负成本 / 对方消耗记正收益）。
    #    我方进化消耗 1 EP → -w_ep（进化成本，与进化收益相抵，净值决定是否值得）。
    #    对方进化消耗（ep↓）→ +w_ep：我方场面逼对方交进化点解场 = 我方赚（PP交换优势）。
    reward += (my_next.ep - my_prev.ep) * weights.w_ep
    reward += (my_next.sep - my_prev.sep) * weights.w_sep
    reward -= (opp_next.ep - opp_prev.ep) * weights.w_ep
    reward -= (opp_next.sep - opp_prev.sep) * weights.w_sep

    # 手牌消耗成本已并入项 4（手牌张数读数之差），不再单独扣 c_hand。

    return reward



def _countdown_progress(prev_snapshot, next_snapshot, player_prev, player_next) -> int:
    """返回该玩家场上倒计时护符本边实际推进的总格数。"""
    total = 0
    next_field = set(player_next.field)
    for uid in player_prev.field:
        before = prev_snapshot.cards_by_uid.get(uid)
        if before is None or int(getattr(before, "countdown", 0) or 0) <= 0:
            continue
        after = next_snapshot.cards_by_uid.get(uid) if uid in next_field else None
        after_countdown = int(getattr(after, "countdown", 0) or 0) if after is not None else 0
        total += max(0, int(before.countdown) - after_countdown)
    return total


def _crest_gain(prev, next_snapshot, my_prev, my_next) -> float:
    """新增纹章的来源费用总和（沿用旧 diff _estimate_crest_source_cost）。"""
    from decision_engine.planning.diff_value import _estimate_crest_source_cost
    start_ids = set()
    for crest in getattr(my_prev, "crests", ()):
        cid = getattr(crest, "base_card_id", None) or getattr(crest, "card_id", 0)
        start_ids.add(cid)
    total = 0.0
    for crest in getattr(my_next, "crests", ()):
        cid = getattr(crest, "base_card_id", None) or getattr(crest, "card_id", 0)
        if cid and cid not in start_ids:
            source_cost = _estimate_crest_source_cost(cid)
            total += _PERSISTENT_CREST_ACQUISITION_PP.get(cid, source_cost)
    return total


def _amulet_gain(prev, next_snapshot, my_prev, my_next) -> float:
    """新增护符的费用总和。"""
    start_field_uids = set(my_prev.field)
    total = 0.0
    for uid in my_next.field:
        if uid in start_field_uids:
            continue
        card = next_snapshot.cards_by_uid.get(uid)
        if card is None:
            continue
        card_type = getattr(card, "card_type", None)
        type_val = card_type.value if hasattr(card_type, "value") else str(card_type)
        if type_val in ("amulet", "countdown_amulet"):
            total += (card.cost or 0)
    return total


# ---------------------------------------------------------------------------
# 路径累加（S2）：沿搜索路径逐边累加 step_reward
# ---------------------------------------------------------------------------


def accumulate_path_reward(
    edges,
    root_player: str,
    *,
    weights: StepRewardWeights = DEFAULT_WEIGHTS,
    threshold_fn: ThresholdFn | None = None,
) -> float:
    """沿一条路径逐边累加 step_reward，返回 root_player 视角的累加即时收益。

    edges: 可迭代的 (prev_snapshot, action, next_snapshot) 三元组序列，
           按从 root 到叶的顺序。每个三元组是一条已展开的搜索树边。

    这是逐动作价值的核心：不做首尾快照差，逐边累加，
    天然不丢"过程损失"（增幅达阈值变身、增幅牌当回合打出）。
    冷启动阶段地平线叶不加 value head（本函数只累加即时收益，叶外由调用方记 0）。
    """
    total = 0.0
    for prev, action, nxt in edges:
        total += step_reward(prev, action, nxt, root_player,
                             weights=weights, threshold_fn=threshold_fn)
    return total


# ---------------------------------------------------------------------------
# 方案 A：地平线叶的「场面残值」——把场面价值转化为"打出它需要多少 PP × 10分"
# ---------------------------------------------------------------------------
#
# 用户思路：场面价值 = 打出这个场面需要花多少 PP（PP=硬锚 1费=10分）。
# PP↔身材 非线性（超线性）：拟合自纯香草 菲泽(3费9身材)/猛犸象(7费20身材)：
#     body(c) = 3.107*c - 0.0357*c^2      （身材总量作为费用的函数，超线性）
# 逆函数 pp_of_stat(body) 给出"打出 body 点身材需要几费"。
# 解场残值：对方场面作为我方叶价值的负项；用神纹誓言(6费全清)锚上限 60 分（边际递减）。
#
# 过程 step_reward 不记身材/解场（方案A纯涌现），仅此叶残值补"场面身材"这一未记维度。
# 已在过程算过即时价值的（打脸=life diff、抽牌=手牌流量）均不重复计入叶残值。

# body(c) = a*c + b*c^2 拟合系数（菲泽/猛犸象）
_STAT_A = 3.107
_STAT_B = -0.0357
_COST_UNIT = 10.0            # 1费 = 10分
_CLEAR_RESIDUAL_CAP = 60.0   # 解场残值上限（神纹誓言6费全清 = 60分）


def pp_of_stat(body: float) -> float:
    """打出 body 点身材总量（atk+life）需要多少费。逆超线性曲线。

    解 a*c + b*c^2 = body（b<0，取物理根，饱和到曲线顶点身材上限）。
    """
    if body <= 0:
        return 0.0
    a, b = _STAT_A, _STAT_B
    disc = a * a + 4 * b * body
    if disc <= 0:
        # body 超过曲线顶点（理论最大身材）：用顶点费用 + 线性外推
        c_vertex = -a / (2 * b)              # 顶点费用
        body_vertex = a * c_vertex + b * c_vertex * c_vertex
        # 超出部分按顶点处边际（≈ a + 2*b*c_vertex ≈ 0）线性，用保守斜率外推
        return c_vertex + max(0.0, (body - body_vertex)) / max(1e-6, a + 2 * b * c_vertex + 1.0)
    c = (-a + disc ** 0.5) / (2 * b)
    return max(0.0, c)


def _total_body(snapshot: BattleSnapshot, view) -> float:
    """场上随从的总身材（atk + life）。护符无身材（atk/life 为 0）自然不计。"""
    total = 0.0
    for uid in view.field:
        c = snapshot.cards_by_uid.get(uid)
        if c is None:
            continue
        atk = getattr(c, "atk", 0) or 0
        life = getattr(c, "life", 0) or 0
        total += max(0, atk) + max(0, life)
    return total


def _is_follower(card) -> bool:
    """是否随从（护符/纹章不计入 board 数量项，由各自维度处理）。"""
    ct = getattr(card, "card_type", None)
    val = ct.value if hasattr(ct, "value") else str(ct)
    return val in ("follower", "FOLLOWER", str(CardType.FOLLOWER))


def _board_signal(snapshot: BattleSnapshot, view, weights: StepRewardWeights) -> float:
    """§11.3 board 维度信号：Σ_随从 (w_board_count + atk×w_board_atk + life×w_board_life
    + 额外攻击次数×预期伤害期权)。

    - w_board_count（数量项，解场位，大头）保证"数量优于身材"（多一随从=逼对手多一次单解）。
    - atk/life 细粒度项让网络能分辨 5/2 优于 5/3；w_board_atk > w_board_life（攻=威胁，主）。
    - **额外攻击期权**：本回合还能攻击 N 次（``attacks_left``）时，第 2 次及以后的攻击是
      "预期伤害"而非"存在价值"——「1回合可以攻击2次」（宏大的回归 90034350）或超进化
      带来的多次攻击，若不计这一项，给它双击这个动作在势里是 +0，只有当搜索真的把第二刀
      砍出去、血量变了才体现；而单回合搜索常常走不到那一步。
    - 只对随从计数；护符/纹章不占 board_count，由 _amulet_gain / _crest_gain 维度处理。
    过程即计、敌我对称（调用处 my − opp），随从被解/兑现由读数之差 telescoping 自然处理。

    防双重计价（§11.3 "打脸不得重复计"）的对冲推演（5 攻随从获得双击并打满）：
      获得双击 attacks_left 2 → +atk×w_board_extra_attack（期权）
      打第一刀 → 血量 +atk×w_face，attacks_left 2→1 → 期权如数扣回（净 0）
      打第二刀 → 血量 +atk×w_face（期权已为 0）
    合计 = 2×atk 点伤害，与"只按血量计"完全一致，没有多算；而"获得双击但最终没打第二刀"
    会被如实记为期权落空（真实机会成本）。
    """
    total = 0.0
    # 只有"当前行动方"的额外攻击才是马上能兑现的期权。对手随从残留的 attacks_left
    # （上回合给了双击却没打出去）在对手回合开始时会重置，对我方不构成当前威胁，
    # 若计入会让负势虚高。
    is_active_side = (view is snapshot.me) == bool(snapshot.is_ally_turn)
    for uid in view.field:
        c = snapshot.cards_by_uid.get(uid)
        if c is None or not _is_follower(c):
            continue
        atk = max(0, getattr(c, "atk", 0) or 0)
        life = max(0, getattr(c, "life", 0) or 0)
        # attacks_left 是"本回合剩余攻击次数"，回合开始由 rule_kernel 重置为 attack_limit(通常 1)，
        # 所以双击只在当回合有效，跨回合搜索里期权会自然回落，不会虚高。
        can_act = bool(getattr(c, "can_attack", False) or getattr(c, "can_attack_leader", False))
        attacks_left = max(0, getattr(c, "attacks_left", 0) or 0)
        extra_attacks = max(0, attacks_left - 1) if (is_active_side and can_act) else 0
        total += (
            weights.w_board_count
            + atk * weights.w_board_atk
            + extra_attacks * atk * weights.w_board_extra_attack
            + life * weights.w_board_life
        )
    return total


# 持续纹章采用“即时获取折扣 + horizon 持续残值”而非每回合重复奖励。
# 10264122 来自6费维尔伯特的手动进化：即时按半张来源卡=3PP，
# horizon 再保留1EP等价=20分；实际进化的EP消耗仍由step_reward单独记-20。
_PERSISTENT_CREST_ACQUISITION_PP = {
    10264122: 3.0,
}
_PERSISTENT_CREST_HORIZON_VALUE = {
    10264122: 20.0,
}


# ---------------------------------------------------------------------------
# E3：Extra PP 保守 horizon option residual
# ---------------------------------------------------------------------------
#
# 语义：后手方持有的、本回合尚未使用的 extra_pp token 提供一个「未来可临时 +1 PP」
# 的选择权（option）。只有当该 token 在下一次该方行动时确实能解锁某张手牌的
# pp+1 play mode 时，这个 option 才有真实价值。
#
# 该残值是「保守固定 option 值」，只在 horizon_leaf_residual 被显式传入 next_player 时
# 加入（Macro 逐动作模型的地平线叶评估）。默认调用（next_player=None）保持既有数值不变，
# 不影响任何非 Macro-horizon 的既有语义。
#
# 关键防误奖规则（合同要点 2/4）：只对「horizon 时当前 next_player 可立即行动的一方」
# 评估其持有的 token option。T5 后手若已结束回合、next_player 为另一方，则该方持有的、
# 即将在 T6 发放时被覆盖（EXTRA_PP_CAP=1）的 token 不再是「即将可用的选择权」，不计。
# 只有当 next_player 仍是该 token 持有方（如 T5 当前该方尚未结束回合）时才计。
#
# 取值：保守固定，<= 一费价值的一半。一费在 step_reward 中为 w_pp_temp=10 分，
# 故 option 值取 5.0（= 半费）。option 是「可能用得上」的期权，不是已兑现资源，
# 因此显著低于真正获得 1 点一次性 PP 的价值，避免过度奖励持有 token。
_EXTRA_PP_OPTION_RESIDUAL = 5.0

# T5 后手第一个 extra_pp 在 T6 发放时被覆盖（与 macro_turn_search / extra_pp_enumerator
# 的 _FIRST_EXTRA_PP_EXPIRY_TURN 保持一致），仅用于文档说明；实际是否计入由
# next_player 是否等于 token 持有方决定，不在此处单独判 turn。


def _extra_pp_option_available(snapshot: BattleSnapshot, view) -> bool:
    """该玩家当前是否持有一个「未来可解锁 pp+1 play mode」的 extra_pp option。

    仅从本地可观察状态推导（不依赖协议 Can* 字段）：
    - 后手（非先手）——先手方不获得 extra_pp。
    - 持有 extra_pp token（extra_pp.extra_pp > 0）。
    - 本回合未使用（not is_used_extra_pp_this_turn）——已用掉则不再是未来 option。
    - 当前已知手牌中存在一张本地可合法的、恰好被 +1 PP 解锁的 play mode：
      某个 play-mode 费用恰等于 pp + 1（当前 pp 买不起，+1 PP 后可打出）。

    未知手牌（无对应 CardView）不计；spent / 先手 / 无解锁手牌均返回 False。
    """
    if view.is_first_side:
        return False
    extra = view.extra_pp
    if extra is None or int(extra.extra_pp) <= 0 or bool(extra.is_used_extra_pp_this_turn):
        return False
    unlock_cost = int(view.pp) + 1
    field_full = len(view.field) >= MAX_FIELD_SIZE
    for uid in view.hand:
        card = snapshot.cards_by_uid.get(uid)
        if card is None:
            # 未知手牌：无法本地确认解锁 play mode，保守不计。
            continue
        if _card_has_play_mode_cost(card, unlock_cost, field_full=field_full):
            return True
    return False


def _card_has_play_mode_cost(card: CardView, target_cost: int, *, field_full: bool) -> bool:
    """该手牌是否存在费用恰为 target_cost 且当前场地可合法打出的 play mode。

    普通/增强模式的非 spell 卡、以及结晶均需场地空位；激奏为 spell 模式，
    不占场位。该检查与 LocalActionEnumerator 的 Extra PP 解锁条件保持一致，
    只用本地 CardView，不读取协议 Can*。
    """
    normal_or_enhance_needs_slot = card.card_type is not CardType.SPELL
    if int(card.cost) == target_cost and not (field_full and normal_or_enhance_needs_slot):
        return True
    if card.can_enhance and any(int(c) == target_cost for c in (card.enhance_costs or ())):
        if not (field_full and normal_or_enhance_needs_slot):
            return True
    if any(int(c) == target_cost for c in (card.accelerate_costs or ())):
        return True
    if any(int(c) == target_cost for c in (card.crystal_costs or ())):
        if not field_full:
            return True
    return False


def _extra_pp_option_residual(
    snapshot: BattleSnapshot,
    root_player: str,
    next_player: str,
) -> float:
    """E3 option residual（root_player 视角，对称正/负）。

    仅评估 next_player 可立即行动的一方（防 T5 结束后误奖即将被覆盖的 token）：
    - 若该方 == root_player → 正的保守 option 值。
    - 若该方 == root_player 的对手 → 负的保守 option 值。
    该方不满足 option 条件（先手 / spent / 无解锁手牌 / 无 token）→ 0。
    非 me/opponent 的 next_player 视为无可评估方 → 0。
    """
    if next_player not in ("me", "opponent"):
        return 0.0
    acting_view = snapshot.me if next_player == "me" else snapshot.opponent
    if not _extra_pp_option_available(snapshot, acting_view):
        return 0.0
    return _EXTRA_PP_OPTION_RESIDUAL if next_player == root_player else -_EXTRA_PP_OPTION_RESIDUAL


def horizon_leaf_residual(
    snapshot: BattleSnapshot,
    root_player: str,
    *,
    cost_unit: float = _COST_UNIT,
    clear_cap: float = _CLEAR_RESIDUAL_CAP,
    next_player: str | None = None,
) -> float:
    """地平线叶的残值（root_player 视角）。

    §11 重构后：**场面身材已由 step_reward 的 board 维度过程即计**，叶不再用
    pp_of_stat 身材残值（否则与过程 board 项双重计价）。叶残值现只保留：
      我方持续纹章保守残值 − 对方持续纹章保守残值
      [+ E3 Extra PP option residual，仅当传入 next_player]

    生命差 / 手牌存量 / 场面身材均不在叶计入（已在过程 step_reward 记过）；
    纹章持续残值只在 horizon 计算一次，获取动作的即时折扣已在 step_reward 处理。

    E3（合同）：option residual 只在显式传入 next_player 时加入；默认调用
    （next_player=None）只返回纹章残值差。
    """
    my_view, opp_view = _sides(snapshot, root_player)
    my_crest_residual = _persistent_crest_horizon_value(my_view)
    opp_crest_residual = _persistent_crest_horizon_value(opp_view)
    residual = my_crest_residual - opp_crest_residual
    if next_player is not None:
        residual += _extra_pp_option_residual(snapshot, root_player, next_player)
    return residual


def _persistent_crest_horizon_value(view) -> float:
    """仍在纹章区的已知持续能力保守残值；仅在Macro horizon计算一次。"""
    total = 0.0
    for crest in getattr(view, "crests", ()):
        crest_id = getattr(crest, "base_card_id", None) or getattr(crest, "card_id", 0)
        total += _PERSISTENT_CREST_HORIZON_VALUE.get(int(crest_id or 0), 0.0)
    return total


# ---------------------------------------------------------------------------
# PUCT leaf value 接入（逐动作 MDP 粒度，取代 MacroDrivenSearch 的回合线抽象）
# ---------------------------------------------------------------------------
#
# 架构修正：MacroDrivenSearch 在"回合线(TurnLine)"粒度做 minimax，违背了
# (a) MDP 的逐动作原子转移粒度、(b) potential-based shaping 的 per-transition
# telescoping 定义前提。正确做法 = 逐动作 PUCT，leaf value = 从本回合 turn_start
# 到叶节点的 step_reward 累加。
#
# 关键简化（已验证）：本势函数是纯 potential-based（全状态读数 + spell_boost 单调），
# 路径无关 → Σ step_reward(逐边) == step_reward(turn_start → leaf 首尾差)。
# 故无需逐边路径，直接用首尾差一次性算（仿 DiffSearchValue），不改 puct.py。

from math import tanh as _tanh

# tanh 归一化 scale（1PP=10分）。
# scale=150：使"一回合典型净势"(raw 50~150 分)落在 tanh 的近线性区(|v|<0.7)，保留势梯度。
# 诊断教训：scale=50 过小 → raw>90 全挤进 tanh 饱和尾(v≈0.95+)，10→20 增幅势差被压成 0.03，
# 搜索看不到；scale=150 把同一势差拉到 0.11（3×），进度势才真正进入 PUCT 决策。
_INCREMENTAL_NORMALIZE_SCALE = 150.0


class IncrementalSearchValue:
    """逐动作 PUCT 的 leaf value：Φ(leaf) − Φ(turn_start) 的 PP 本位势差，tanh 归一化。

    满足 search_value.SearchValue 协议（evaluate(snapshot, player) -> ValueResult）。
    与 DiffSearchValue 同构（存 turn_start 基线、可选 virtual turn_end、reset 复用），
    但势用 §11 重标定的 step_reward（PP 本位、board 过程即计），且天然路径无关。

    用法（干净单变量对照，取代 MacroDrivenSearch）：
        value = IncrementalSearchValue(turn_start, player=player, predictor=predictor)
        search = PuctSearch(..., value=value)
    """

    source = "incremental_value"

    def __init__(self, turn_start_snapshot: BattleSnapshot, *, player: str = "me",
                 predictor=None, weights: StepRewardWeights = DEFAULT_WEIGHTS,
                 threshold_fn: "ThresholdFn | None" = None,
                 scale: float = _INCREMENTAL_NORMALIZE_SCALE):
        self._turn_start = turn_start_snapshot
        self._player = player
        self._predictor = predictor
        self._weights = weights
        self._threshold_fn = threshold_fn if threshold_fn is not None else default_threshold_fn()
        self._scale = scale

    def evaluate(self, snapshot: BattleSnapshot, *, player: str = "me"):
        from decision_engine.planning.search_value import ValueResult

        # 终局：step_reward 的终局项返回 ±v_win，tanh 后 → ±1。
        winner = _winner(snapshot)
        if winner is not None:
            if winner == "simultaneous":
                return ValueResult(value=0.0, source="terminal_simultaneous", terminal=True,
                                   diagnostics={"player": player, "winner": winner})
            v = 1.0 if winner == player else -1.0
            return ValueResult(value=v, source="terminal_result", terminal=True,
                               diagnostics={"player": player, "winner": winner})

        eval_snapshot = self._virtual_turn_end(snapshot, player)

        # 路径无关：首尾状态差一次性算（已验证等价于逐边累加）。
        # 用一个占位 action（step_reward 的非终局分支只依赖 prev/next 状态差）。
        raw = step_reward(self._turn_start, _LEAF_PLACEHOLDER_ACTION, eval_snapshot, player,
                          weights=self._weights, threshold_fn=self._threshold_fn)
        raw += horizon_leaf_residual(eval_snapshot, player)

        value = _tanh(raw / self._scale)
        value = max(-1.0, min(1.0, value))
        return ValueResult(value=value, source="incremental_value", terminal=False,
                           diagnostics={"raw": raw, "scale": self._scale,
                                        "virtual_turn_end": eval_snapshot is not snapshot})

    def _virtual_turn_end(self, snapshot: BattleSnapshot, player: str) -> BattleSnapshot:
        """虚拟执行 turn_end，返回 after-state（仿 DiffSearchValue）。失败返回原 snapshot。"""
        if self._predictor is None or getattr(snapshot, "is_turn_ended", False):
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

    def reset(self, turn_start_snapshot: BattleSnapshot, *, player: str = "me"):
        self._turn_start = turn_start_snapshot
        self._player = player


# step_reward 非终局分支只用 prev/next 状态差，action 仅用于终局前记录；用固定占位。
_LEAF_PLACEHOLDER_ACTION = LegalAction(kind=ActionKind.TURN_END, source_uid=0)


class IncrementalAwarePuctSearch:
    """逐动作 PuctSearch 包装，每回合自动重置 IncrementalSearchValue 的 turn_start baseline。

    与 DiffAwarePuctSearch 同构，但 leaf value = PP 本位 step_reward 势差（§11）。
    这是取代 MacroDrivenSearch（回合线粒度）的正确架构：严格逐动作 MDP 粒度，
    policy target = 干净的逐动作 visit 分布，势函数 telescoping 结构完整。
    """

    def __init__(self, *, enumerator, child_enumerator, predictor, prior,
                 turn_transitioner, config, weights: StepRewardWeights = DEFAULT_WEIGHTS,
                 threshold_fn: "ThresholdFn | None" = None):
        self._enumerator = enumerator
        self._child_enumerator = child_enumerator
        self._predictor = predictor
        self._prior = prior
        self._turn_transitioner = turn_transitioner
        self._config = config
        self._weights = weights
        self._threshold_fn = threshold_fn
        self._value = None
        self._current_turn_start = None
        self._current_player = None
        self._search = None

    @property
    def config(self):
        return self._config

    @property
    def prior(self):
        return self._prior

    @property
    def value(self):
        return self._value

    def search(self, snapshot: BattleSnapshot, *, player: str = "me"):
        from decision_engine.planning.puct import PuctSearch
        if self._current_player != player or self._current_turn_start is None:
            self._current_turn_start = snapshot
            self._current_player = player
            self._value = IncrementalSearchValue(
                turn_start_snapshot=snapshot, player=player, predictor=self._predictor,
                weights=self._weights, threshold_fn=self._threshold_fn,
            )
        self._search = PuctSearch(
            enumerator=self._enumerator,
            child_enumerator=self._child_enumerator,
            predictor=self._predictor,
            prior=self._prior,
            value=self._value,
            turn_transitioner=self._turn_transitioner,
            config=self._config,
        )
        return self._search.search(snapshot, player=player)

    def notify_turn_end(self):
        self._current_turn_start = None
        self._current_player = None
