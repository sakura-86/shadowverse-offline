# -*- coding: utf-8 -*-
"""魔力增幅（spellboost）饱和点模型：卡牌知识侧的唯一出处。

一只受增幅的卡，其"继续攒增幅还能不能换来新东西"分三种：

- ``threshold``：效果在若干固定增幅数上分档解锁（如 10834110 的 10/20 档）。
  饱和点 = 最高阈值；攒到那里之后继续增幅不再有收益。
- ``cost_reduction``：每次增幅让自身费用 -1，降到 0 费封顶（如 10134310 初始 18 费）。
  饱和点 = 印刷费用。
- ``none``：线性成长（``variable_x_hand`` 的 X+n，如暴风破）或随从 +1/+1 型，
  没有饱和点——它的增幅收益在打出那一刻全部兑现，不存在"攒到某个数解锁质变"。

为什么把这个放到 ``cards/``：饱和点（cap）是纯卡牌知识，value 侧
（``planning/incremental_value.py`` 的多阈值进度势）和 policy 侧
（``training/state_features.py`` 的进度饱和度特征）必须用同一份判定，否则两边
对"这张满了没有"会给出互相矛盾的说法。运行时不查文本、不做正则，只在构造期
从 ``card_database.json`` 建一次表。

数据来源（全部自动提取，无逐卡硬编码）：
- 阈值型：``supplement_texts[].identifier`` ∈ {``variable_x_threshold_hand``,
  ``variable_x_threshold2_hand``}，阈值取 ``args``。
- 降费型：``clean_skill_desc`` 含精确子串 ``【魔力增幅时】使本卡牌的费用-``
  （85803310 那种"使随机 1 张手牌的费用-1"不是自身降费，被这个前缀排除）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

#: 分档解锁型：饱和点 = 最高阈值。
CAP_KIND_THRESHOLD = "threshold"
#: 自我降费型：饱和点 = 印刷费用（降到 0 费为止）。
CAP_KIND_COST_REDUCTION = "cost_reduction"
#: 线性成长 / 无饱和点：不计进度饱和度。
CAP_KIND_NONE = "none"

_THRESHOLD_IDENTIFIERS = frozenset({
    "variable_x_threshold_hand",
    "variable_x_threshold2_hand",
})
#: 自身降费的精确文本前缀。"使自己的手牌中的随机1张…的费用-1" 不匹配（不是自身降费）。
_COST_REDUCTION_MARKER = "【魔力增幅时】使本卡牌的费用-"

#: 归一化上限：卡池里最贵的降费型是 18 费（超越次元），取 20 作为无量纲分母。
CAP_NORMALIZER = 20.0

_DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"
)


@dataclass(frozen=True)
class BoostCapProfile:
    """一只增幅卡的饱和点画像。

    ``cap`` 为 ``None`` 表示没有饱和点（线性成长/随从 buff 型），此时
    ``saturation()`` 恒为 0——刻意区别于"有饱和点但一点没攒"（也是 0），
    由调用方用 ``has_cap`` 区分，不做隐式猜测。
    """

    base_card_id: int
    kind: str
    thresholds: tuple[int, ...] = ()
    cap: int | None = None

    @property
    def has_cap(self) -> bool:
        return self.cap is not None and self.cap > 0

    def saturation(self, spell_boost_count: int) -> float:
        """进度饱和度：``min(count, cap) / cap`` ∈ [0, 1]；无饱和点 → 0.0。"""
        if not self.has_cap:
            return 0.0
        counted = max(0, int(spell_boost_count))
        return min(counted, int(self.cap)) / float(self.cap)

    def remaining(self, spell_boost_count: int) -> int:
        """距饱和点还差几次增幅；无饱和点 → 0。"""
        if not self.has_cap:
            return 0
        counted = max(0, int(spell_boost_count))
        return max(0, int(self.cap) - counted)


_NO_CAP = BoostCapProfile(base_card_id=0, kind=CAP_KIND_NONE, thresholds=(), cap=None)


def _iter_cards(payload: object) -> list[Mapping[str, object]]:
    if isinstance(payload, list):
        return [c for c in payload if isinstance(c, Mapping)]
    if isinstance(payload, Mapping):
        cards = payload.get("cards")
        if isinstance(cards, list):
            return [c for c in cards if isinstance(c, Mapping)]
        return [c for c in payload.values() if isinstance(c, Mapping)]
    return []


def _profile_from_card(card: Mapping[str, object]) -> BoostCapProfile | None:
    base_card_id = card.get("base_card_id") or card.get("card_id")
    if not base_card_id:
        return None
    try:
        base_card_id = int(base_card_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    thresholds: list[int] = []
    for supplement in card.get("supplement_texts") or ():  # type: ignore[union-attr]
        if not isinstance(supplement, Mapping):
            continue
        if str(supplement.get("identifier", "")) not in _THRESHOLD_IDENTIFIERS:
            continue
        for raw in supplement.get("args") or ():  # type: ignore[union-attr]
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                thresholds.append(value)
    if thresholds:
        ordered = tuple(sorted(set(thresholds)))
        return BoostCapProfile(
            base_card_id=base_card_id,
            kind=CAP_KIND_THRESHOLD,
            thresholds=ordered,
            cap=ordered[-1],
        )

    description = str(card.get("clean_skill_desc") or "")
    if _COST_REDUCTION_MARKER in description:
        try:
            printed_cost = int(card.get("cost") or 0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            printed_cost = 0
        if printed_cost > 0:
            return BoostCapProfile(
                base_card_id=base_card_id,
                kind=CAP_KIND_COST_REDUCTION,
                thresholds=(),
                cap=printed_cost,
            )
    return None


def build_boost_cap_table(card_database_path: str | Path | None = None) -> dict[int, BoostCapProfile]:
    """从卡牌数据库构建 ``base_card_id -> BoostCapProfile``（只含真有饱和点的卡）。"""
    path = Path(card_database_path) if card_database_path is not None else _DEFAULT_DATABASE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    table: dict[int, BoostCapProfile] = {}
    for card in _iter_cards(payload):
        profile = _profile_from_card(card)
        if profile is not None:
            table[profile.base_card_id] = profile
    return table


_TABLE_CACHE: dict[int, BoostCapProfile] | None = None


def build_boost_reward_card_spans(
    card_database_path: str | Path | None = None,
) -> dict[int, int]:
    """``奖励卡 base_card_id -> 解锁该卡所需的增幅跨度``（= 它应当持有的势当量）。

    为什么需要：分档解锁型卡（坦忒拉）达到 10/20 档时各给一张 1 费奖励卡，但这两张
    1 费卡的牌效约等于 5 费卡。若按普通手牌 ``w_hand`` 计价，解锁的那一刻势会凭空蒸发：
    打出时没收 ``Φ(20)=10PP``，奖励卡只记 3PP，净 −7PP ⇒ "越攒越不该打"。
    按解锁跨度计价后，被没收的势正好由奖励卡承接（各 10 增幅 × 0.5PP = 5PP）。

    映射来源：``related_base_card_ids`` 与 ``thresholds`` **按序一一对应**（坦忒拉为
    ``[90034340, 90034350]`` ↔ ``(10, 20)``）。数量不一致时整张卡跳过——宁可漏算，
    也不把"相关卡"误当"奖励卡"。
    """
    path = Path(card_database_path) if card_database_path is not None else _DEFAULT_DATABASE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    spans: dict[int, int] = {}
    for card in _iter_cards(payload):
        profile = _profile_from_card(card)
        if profile is None or profile.kind != CAP_KIND_THRESHOLD:
            continue
        related: list[int] = []
        for raw in card.get("related_base_card_ids") or ():
            try:
                related.append(int(raw))
            except (TypeError, ValueError):
                related = []
                break
        if len(related) != len(profile.thresholds):
            continue
        prev = 0
        for reward_id, threshold in zip(related, profile.thresholds):
            if threshold > prev:
                spans[reward_id] = threshold - prev
            prev = threshold
    return spans


_REWARD_SPAN_CACHE: dict[int, int] | None = None


def boost_reward_card_spans() -> dict[int, int]:
    """懒加载缓存的奖励卡跨度表（奖励卡 → 解锁所需增幅次数）。"""
    global _REWARD_SPAN_CACHE
    if _REWARD_SPAN_CACHE is None:
        _REWARD_SPAN_CACHE = build_boost_reward_card_spans()
    return _REWARD_SPAN_CACHE


def boost_cap_table() -> dict[int, BoostCapProfile]:
    """懒加载缓存的饱和点表。"""
    global _TABLE_CACHE
    if _TABLE_CACHE is None:
        _TABLE_CACHE = build_boost_cap_table()
    return _TABLE_CACHE


def reset_boost_cap_cache() -> None:
    """清空缓存（测试或数据重建后使用）。"""
    global _TABLE_CACHE, _REWARD_SPAN_CACHE
    _TABLE_CACHE = None
    _REWARD_SPAN_CACHE = None


def boost_cap_profile(base_card_id: int | None) -> BoostCapProfile:
    """查一只卡的饱和点画像；不在表里 → 无饱和点（kind=none, cap=None）。"""
    if not base_card_id:
        return _NO_CAP
    return boost_cap_table().get(int(base_card_id), _NO_CAP)


def boost_threshold_table(card_database_path: str | Path | None = None) -> dict[int, tuple[int, ...]]:
    """兼容 value 侧旧接口：只导出分档解锁型卡的阈值元组。"""
    table = (
        build_boost_cap_table(card_database_path)
        if card_database_path is not None
        else boost_cap_table()
    )
    return {
        base_card_id: profile.thresholds
        for base_card_id, profile in table.items()
        if profile.kind == CAP_KIND_THRESHOLD and profile.thresholds
    }


def boost_cap_features(base_card_id: int | None, spell_boost_count: int) -> tuple[float, float, float]:
    """三只 policy 侧特征的唯一算法：``(saturation, cap_ratio, remaining_ratio)``。

    - ``saturation``    = ``min(sb, cap) / cap``，0 = 一点没攒、1 = 已满（打出零机会成本）
    - ``cap_ratio``     = ``cap / 20``，0 表示这张牌根本没有饱和点（线性成长/随从 buff 型）
    - ``remaining_ratio``= ``(cap - min(sb, cap)) / 20``，距满还差几次增幅

    无饱和点时三者皆为 0。``cap_ratio == 0`` 是"没有饱和点"的唯一判据，
    不要把 ``saturation == 0`` 读成"没攒"——满不足与不存在必须能被网络区分。
    """
    profile = boost_cap_profile(base_card_id)
    if not profile.has_cap:
        return 0.0, 0.0, 0.0
    cap = int(profile.cap)  # type: ignore[arg-type]
    return (
        profile.saturation(spell_boost_count),
        cap / CAP_NORMALIZER,
        profile.remaining(spell_boost_count) / CAP_NORMALIZER,
    )
