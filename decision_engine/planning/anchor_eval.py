# -*- coding: utf-8 -*-
"""锚点评测：候选对 ``plan_turn`` 的配对胜率与分级判定。

为什么需要它
============

项目历史上最大的方法论问题不是"某个信号不好用"，而是**没有外部锚点**：
self-play 只看 ``gen_N`` vs ``gen_{N-1}``，36 局就能得出"迭代成立"的结论
（见 ``docs/blog/诊断历程_SelfPlay不涨与价值信号_面试材料.md`` §9、§11、§16、§17）。

本模块把验收钉在唯一有数据支撑的强基线 ``plan_turn`` 上：严格配对镜像、同 seed、
候选与锚点只差 selector。评测口径复用 :mod:`decision_engine.planning.search_ab_match`。

两档判定
========

========  ==========  ==========================  ==============================================
档位      每臂局数    能回答的问题                允许输出的结论
========  ==========  ==========================  ==============================================
screen    >= 48       是否值得投入确认档          ``screen_promote`` / ``screen_reject`` /
                                                  ``screen_inconclusive``
confirm   >= 400      是否显著优于锚点            ``confirmed_better`` / ``confirmed_worse`` /
                                                  ``confirmed_not_better``
========  ==========  ==========================  ==============================================

档位在**调用时**决定，不由数据决定：screen 档即使 48 局全胜也只输出
``screen_promote``，不能写成"更好"。把"筛选不许当结论"做成返回值类型约束，而不是
流程约定——项目历史上正是这条失守。

统计口径
========

不用 ``summarize`` 的 one-sided sign test，因为它**丢弃 split**。本模块把
``split``（同 seed 一胜一负）记作 0.5 胜场，按 game 级口径算胜率，并用 Wilson
区间给确认档做双侧 95% 的保守下界。

零和终局口径：**影之诗不存在平局**
==================================

一对局只有"我方胜"或"对方胜"两种终局。因此：

- ``split`` **不是**平局，它是"同 seed 下先手赢、后手输"的 1:1。报告中必须写
  "split = 一胜一负"，不得简写成"平局 7 次"。
- 任何 ``winner=None`` 的对局（终局缺失、``max_actions_per_turn`` 截断、
  local predictor 未产出终局、``simultaneous``）都是**终局判定缺陷**，唯一正确处理是
  ``accepted=False`` 整对作废：既不进胜率分子，也不进分母。
- **禁止**把无胜负局记作 draw、记作 0.5 胜场、或从分子分母里按"忽略"处理而不作废计数。
  用回合上限把没打完的局凑成"有胜负"，等于伪造实验结果。
- ``AbSummary.equal`` / ``AnchorResult.equal`` 结构性恒 0，只为 JSON 向后兼容保留；
  它不是"平局数"。要考察终局质量请看 ``rejected_reasons``。

这条曾经真实失守过：``outcome_label`` 用 ``w1 and w2`` 判断，而 ``None`` 是 falsy，
于是 ``(True, None)`` 被静默判成 ``split`` 并按 0.5 胜场计入分子。现在由
``test_anchor_eval.ZeroSumTerminalTest`` 锁死为显式 ``AssertionError``。

已知限制：``plan_turn`` 还不是可用的对称锚点（三层原因）
======================================================

1. **不铺场**（已在臂层部分补上）：``plan_turn`` 只看净血差，打随从当回合记为 0 分，
   于是返回空路线。``turn_plan_arm`` 现在补了 ``burn_pp``——无事可做时按费用从高到低
   把剩余 PP 花完。补完后 opponent 确实开始铺场（play 9 次/局）。

2. **已修：``plan_turn`` 对 opponent 侧空转**。它曾用 ``not snapshot.is_ally_turn``
   判断"回合是否结束"，而 ``is_ally_turn`` 是"**我方**回合"——对手回合恒为 False，
   于是对手侧永远返回空路线，只会走 ``burn_pp`` 打牌、从不攻击/进化。修复后 opponent
   的动作分布与 me 对称（``attack_follower`` 各 25 次）。换算已抽成
   ``turn_planner.is_deciding_player_turn``，由 ``test_plan_turn_perspective`` 锁住。

3. **满场搜索成本爆炸**（未修）：双方都铺场后 ``plan_turn`` 的分支因子显著变大，
   单对从 2.4s 涨到 74s。24 对 ≈ 30 分钟，confirm 档 200 对 ≈ 4 小时，且
   ``plan_turn`` 没有节点预算参数。

4. **对局节奏依赖卡组，单卡组数字不可外推**（2026-10-30 观测）：``rotation_evolve_haven``
   是守护/回复体系；终局缺失时按 ``rejected:no_terminal`` 作废，
   说明该对局中位数回合数明显高于真实天梯（8-10 回合）。这是**卡组特性**，不能据此
   判定 ``plan_turn`` 评分缺陷。因此：``run_anchor_match.py`` 现在把 accepted 局的
   回合数中位数与胜率一起打印；任何对外结论仍须遵守 ``docs/CURRENT.md`` 的
   "单卡组数字不可用，6 卡组 ≥96 局"。

结论：统计/档位/门控层可用；**在第 2、3 条解决前，``tools/run_anchor_match.py``
的对局结果不得写进任何结论**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from decision_engine.planning.search_ab_match import AbSummary, PairResult, summarize

#: 判定值。screen 与 confirm 的返回值集合**不相交**，防止筛选结论被当成确认结论。
VERDICT_INSUFFICIENT = "insufficient_samples"
VERDICT_SCREEN_PROMOTE = "screen_promote"
VERDICT_SCREEN_REJECT = "screen_reject"
VERDICT_SCREEN_INCONCLUSIVE = "screen_inconclusive"
VERDICT_CONFIRMED_BETTER = "confirmed_better"
VERDICT_CONFIRMED_WORSE = "confirmed_worse"
VERDICT_CONFIRMED_NOT_BETTER = "confirmed_not_better"


@dataclass(frozen=True)
class AnchorTierSpec:
    """一个判定档位的样本量要求与可检出效应量。"""

    name: str
    #: 每臂局数下限（一对 = 两局，两臂各一局，因此每臂 ``2 * pairs`` 局）。
    min_games_per_arm: int
    #: 允许该档位输出的结论集合。
    allowed_verdicts: frozenset[str]

    @property
    def min_pairs(self) -> int:
        return self.min_games_per_arm // 2

    @property
    def detectable_effect_pp(self) -> float:
        """该档位在 alpha=.05（单侧）/ power=.8 下能检出的最小效应量，单位 pp。

        由 ``min_games_per_arm`` 反解，保证"标注的效应量"与"要求的局数"永远自洽，
        不会出现"写着检出 7pp、实际只能检出 6pp"这种夸大。
        """
        z_alpha = 1.6448536269514722
        z_beta = 0.8416212335729143
        return 100.0 * (z_alpha + z_beta) * 0.5 / math.sqrt(self.min_games_per_arm)


SCREEN_TIER = AnchorTierSpec(
    name="screen",
    min_games_per_arm=48,
    allowed_verdicts=frozenset(
        {VERDICT_INSUFFICIENT, VERDICT_SCREEN_PROMOTE, VERDICT_SCREEN_REJECT, VERDICT_SCREEN_INCONCLUSIVE}
    ),
)

CONFIRM_TIER = AnchorTierSpec(
    name="confirm",
    min_games_per_arm=400,
    allowed_verdicts=frozenset(
        {VERDICT_INSUFFICIENT, VERDICT_CONFIRMED_BETTER, VERDICT_CONFIRMED_WORSE, VERDICT_CONFIRMED_NOT_BETTER}
    ),
)

TIER_SPECS: Mapping[str, AnchorTierSpec] = {"screen": SCREEN_TIER, "confirm": CONFIRM_TIER}

#: screen 档的"值得继续"门槛：game 级胜率需达到该值。
SCREEN_PROMOTE_WIN_RATE = 0.60
#: screen 档的"直接放弃"门槛：game 级胜率不高于该值。
SCREEN_REJECT_WIN_RATE = 0.40


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间（双尾，默认 95%）。

    ``wins`` 允许是 0.5 的倍数（split 记 0.5 胜场），因此用 float。
    """
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def required_games_per_arm(effect_pp: float, *, alpha_one_sided: float = 0.05, power: float = 0.8) -> int:
    """检出 ``effect_pp`` 个百分点效应所需的每臂局数（单侧）。

    正态近似：``n = (z_alpha + z_beta)^2 * 0.25 / d^2``。
    """
    if effect_pp <= 0:
        raise ValueError("effect_pp 必须为正")
    z_alpha = 1.6448536269514722  # 单侧 5%
    z_beta = 0.8416212335729143  # power 0.8
    d = effect_pp / 100.0
    return math.ceil((z_alpha + z_beta) ** 2 * 0.25 / (d * d))


def required_pairs(effect_pp: float) -> int:
    """``required_games_per_arm`` 换算成 seed 对数（每对两局）。"""
    return math.ceil(required_games_per_arm(effect_pp) / 2)


@dataclass(frozen=True)
class AnchorResult:
    """一次锚点评测的结构化结论。"""

    tier: str
    accepted_pairs: int
    rejected_pairs: int
    win: int
    loss: int
    split: int
    equal: int
    #: 有效对局数（每对两局）。
    games: int
    #: game 级候选胜场（win 记 2、split 记 1）。
    game_wins: float
    win_rate: float
    ci_low: float
    ci_high: float
    verdict: str
    required_games_per_arm: int
    rejection_reasons: Mapping[str, int]

    @property
    def is_conclusive(self) -> bool:
        """是否达到该档位的样本量要求。"""
        return self.verdict != VERDICT_INSUFFICIENT

    @property
    def is_confirmed(self) -> bool:
        """是否达到"确认优于锚点"级别（screen 档永远为 False）。"""
        return self.verdict == VERDICT_CONFIRMED_BETTER

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "verdict": self.verdict,
            "verdict_meaning": _VERDICT_MEANING.get(self.verdict, self.verdict),
            "accepted_pairs": self.accepted_pairs,
            "rejected_pairs": self.rejected_pairs,
            "pair_outcomes": {"win": self.win, "loss": self.loss, "split": self.split, "equal": self.equal},
            "games_per_arm": self.games,
            "game_wins": self.game_wins,
            "win_rate": round(self.win_rate, 6),
            "wilson_95": [round(self.ci_low, 6), round(self.ci_high, 6)],
            "required_games_per_arm": self.required_games_per_arm,
            "rejection_reasons": dict(self.rejection_reasons),
            "is_confirmed_better": self.is_confirmed,
            "note": (
                "screen 档只用于决定是否进入 confirm 档；它不能证明候选更好。"
                if self.tier == "screen"
                else "confirm 档以 Wilson 95% 下界 > 0.5 判定显著优于锚点。"
            ),
        }


_VERDICT_MEANING: Mapping[str, str] = {
    VERDICT_INSUFFICIENT: "样本量未达该档位要求，不下结论",
    VERDICT_SCREEN_PROMOTE: "方向明确，值得投入 confirm 档复核；不构成'更好'的证据",
    VERDICT_SCREEN_REJECT: "方向明确为劣，可停止该候选",
    VERDICT_SCREEN_INCONCLUSIVE: "筛选档区分不出，需换更大效应量的改动或直接进 confirm",
    VERDICT_CONFIRMED_BETTER: "显著优于锚点（Wilson 95% 下界 > 0.5）",
    VERDICT_CONFIRMED_WORSE: "显著劣于锚点（Wilson 95% 上界 < 0.5）",
    VERDICT_CONFIRMED_NOT_BETTER: "达到样本量但仍无法区分于锚点",
}


def _resolve_tier(tier: str | AnchorTierSpec) -> AnchorTierSpec:
    if isinstance(tier, AnchorTierSpec):
        return tier
    try:
        return TIER_SPECS[tier]
    except KeyError as exc:
        raise ValueError(f"未知档位 {tier!r}，可选 {sorted(TIER_SPECS)}") from exc


def evaluate(pairs: Sequence[PairResult], *, tier: str | AnchorTierSpec = "screen") -> AnchorResult:
    """对一批配对结果做锚点判定。

    ``pairs`` 的每个元素是一对（同 seed、交换先后手）对局，与
    :class:`StrictSearchAbRunner` 的产出一致。
    """
    spec = _resolve_tier(tier)
    return _evaluate_summary(summarize(pairs), spec)


def evaluate_summary(summary: AbSummary, *, tier: str | AnchorTierSpec = "screen") -> AnchorResult:
    """对已汇总的 :class:`AbSummary` 做锚点判定（便于离线重算历史数据）。"""
    return _evaluate_summary(summary, _resolve_tier(tier))


def _evaluate_summary(summary: AbSummary, spec: AnchorTierSpec) -> AnchorResult:
    accepted = summary.accepted_pairs
    games = 2 * accepted
    # split = 同 seed 一胜一负 → 各记一局；win 对 = 两局全胜。
    game_wins = float(2 * summary.win + summary.split)
    win_rate = (game_wins / games) if games else 0.0
    ci_low, ci_high = wilson_interval(game_wins, games)

    verdict = _decide(spec, accepted_pairs=accepted, win_rate=win_rate, ci_low=ci_low, ci_high=ci_high)
    if verdict not in spec.allowed_verdicts:  # 防御：档位不得输出越权结论
        raise AssertionError(f"档位 {spec.name} 输出了未授权结论 {verdict}")

    return AnchorResult(
        tier=spec.name,
        accepted_pairs=accepted,
        rejected_pairs=summary.rejected_pairs,
        win=summary.win,
        loss=summary.loss,
        split=summary.split,
        equal=summary.equal,
        games=games,
        game_wins=game_wins,
        win_rate=win_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        verdict=verdict,
        required_games_per_arm=spec.min_games_per_arm,
        rejection_reasons=dict(summary.rejection_reasons),
    )


def _decide(
    spec: AnchorTierSpec,
    *,
    accepted_pairs: int,
    win_rate: float,
    ci_low: float,
    ci_high: float,
) -> str:
    if accepted_pairs < spec.min_pairs:
        return VERDICT_INSUFFICIENT
    if spec.name == "screen":
        if win_rate >= SCREEN_PROMOTE_WIN_RATE:
            return VERDICT_SCREEN_PROMOTE
        if win_rate <= SCREEN_REJECT_WIN_RATE:
            return VERDICT_SCREEN_REJECT
        return VERDICT_SCREEN_INCONCLUSIVE
    # confirm：只在 CI 明确落在 0.5 的一侧时下结论。
    if ci_low > 0.5:
        return VERDICT_CONFIRMED_BETTER
    if ci_high < 0.5:
        return VERDICT_CONFIRMED_WORSE
    return VERDICT_CONFIRMED_NOT_BETTER


def plan_markdown(result: AnchorResult) -> str:
    """人类可读的结论摘要（CLI 用）。"""
    lines = [
        f"# 锚点评测（{result.tier} 档）",
        "",
        f"- 结论：**{result.verdict}** — {_VERDICT_MEANING.get(result.verdict, '')}",
        f"- 有效对：{result.accepted_pairs}（作废 {result.rejected_pairs}）→ {result.games} 局/臂",
        f"- pair 结果：win {result.win} / loss {result.loss} / split {result.split}"
        "（split = 同 seed 一胜一负，非平局）",
        f"- game 级胜率：{result.win_rate:.3f}，Wilson 95% = [{result.ci_low:.3f}, {result.ci_high:.3f}]",
        f"- 该档位要求：{result.required_games_per_arm} 局/臂",
    ]
    if result.rejection_reasons:
        reasons = "，".join(f"{k}:{v}" for k, v in sorted(result.rejection_reasons.items(), key=lambda kv: -kv[1])[:5])
        lines.append(f"- 主要作废原因：{reasons}")
    if result.tier == "screen" and result.verdict == VERDICT_SCREEN_PROMOTE:
        lines.append("")
        lines.append("> screen 档通过**不等于**候选更好。请用 `--tier confirm` 复核后再写结论。")
    return "\n".join(lines)


__all__ = [
    "AnchorResult",
    "AnchorTierSpec",
    "CONFIRM_TIER",
    "SCREEN_TIER",
    "TIER_SPECS",
    "evaluate",
    "evaluate_summary",
    "plan_markdown",
    "required_games_per_arm",
    "required_pairs",
    "wilson_interval",
]
