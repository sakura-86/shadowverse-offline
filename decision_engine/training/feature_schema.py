# -*- coding: utf-8 -*-
"""Shared schema helpers for training features."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping

# Dense state features were reduced from the legacy 61-dimensional layout to
# the current 41-dimensional layout. Artifacts from the old layout must not be
# loaded against the current encoder.
#
# v3 (方案 C C1)：state_features 为当前 perspective 玩家新增 own remaining-deck
# 特征（player.remaining_deck: deck_knowledge_status + remaining_deck_count +
# remaining_deck_cards）。dense scalar 布局未变，但结构化特征结构已扩展，
# 旧 schema 数据不得与新特征契约混用。
#
# v6 (A 组条件特征)：卡级新增 4 维「打出效果条件」三态特征
# （has_conditional_effect / conditional_effect_satisfied /
#   conditional_effect_unsatisfied / conditional_effect_unknown），
# CARD_STAT_DIM 29 → 33。条件进度由 LocalPredictor 在每次 action 转移后物化
# （execution/condition_progress.py），self-play 与 runtime 共用同一路径；
# 特征编码只读 snapshot，不做额外求值。旧 v5 artifact 因卡张量宽度变化不可复用。
#
# v7 (留牌潜在价值)：再新增 2 维静态来源
# （has_discard_payoff 本卡牌被舍弃时发动 / has_hand_cost_reduction 在手牌中降费），
# CARD_STAT_DIM 33 → 35。这两类卡即使没有任何「效果条件」，留在手里也有价值。
#
# v8 (增幅进度饱和度)：再新增 3 维，让 policy 侧直接看到"这张增幅牌还差多少攒满"。
# - boost_saturation    = min(sb, cap)/cap，1 表示已到最高阈值/0 费（打出零机会成本）
# - boost_cap_ratio     = cap/20，0 表示该卡没有饱和点（线性成长/随从 buff 型）
# - boost_cap_remaining = (cap - min(sb, cap))/20，距满还差几次增幅
# CARD_STAT_DIM 35 → 38。cap 的判定（阈值型=最高阈值、降费型=印刷费用）在
# cards/spellboost.py，与 planning 侧多阈值进度势同源，不做第二套逻辑。
#
# v9 (进化效果可见性)：再新增 2 维，让 policy 头能区分"进化这张卡会不会触发额外效果"。
# - has_evolve_effect       = card_database.evolve_desc 非空
# - has_super_evolve_effect = card_database.super_evolve_desc 非空
# CARD_STAT_DIM 38 → 40。826 个随从里 225 个有 evolve_desc、120 个有 super_evolve_desc。
# 动机：policy 的输入只有 [state_emb ; action_emb]，看不到"进化它会打 3"这类效果，
# 于是只能按身材挑"更值得进化"的那只。实测 v8 模型在"进化 5/5 突进英灵 vs 进化
# 4/4 密友(打3)"上给前者 prior 0.364、后者 0.145；人类数据训出的 policy 是
# 0.034 vs 0.160。这两个位就是那部分信息的最粗编码。
#
# v10 (动作级效果标签)：action scalar 向量从 24 维扩到 53 维，新增 29 维
# multi-hot「这个动作会产生什么效果」（heal / draw / damage / summon / destroy /
# self_sacrifice / gain_storm ...）。标签由 training/action_effect_tags.py 从效果文本
# 抽取，覆盖全部 1798 张卡；槽位按 play_normal / play_enhance / play_accelerate /
# play_crystal / evolve / super_evolve / activate 分开，所以"打出"和"进化"是两套标签。
# 动机：v9 只给了"进化会不会有效果"两个布尔位，v10 给出"具体是什么效果"，例如
# frame 41 里 evolve 密友 = damage、evolve 英灵 = 空。
# 旧 v9 action_scalar 宽度 24，与 53 维模型不兼容，必须重训。
TRAINING_FEATURE_SCHEMA_VERSION = 10
TRAINING_EXAMPLE_SCHEMA = "TrainingExample"
TRAINING_EXAMPLE_VERSION = 1
LEAKED_FEATURE_KEYS = frozenset({
    "after_context",
    "raw_dir",
    "raw_root",
    "path",
    "before_path",
    "next_path",
    "record_index",
    "transition_index",
    "uid",
    "unique_id",
    "source_uid",
    "target_uid",
    "material_uids",
    # Deck entity 泄漏防御（方案 C C1.5）：剩余牌组特征只允许聚合后的
    # base_card_id + remaining_count，禁止逐实例身份、抽牌顺序或可见性原始字段。
    "instance_id",
    "draw_pile",
    "order_hint",
    "visibility",
})


@dataclass(frozen=True)
class FeatureFrame:
    schema: str
    version: int
    features: Mapping[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "features": to_jsonable(self.features),
        }


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        return [to_jsonable(item) for item in sorted(value)]
    return value


def assert_no_leaked_feature_keys(value: Any, *, path: str = "features") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in LEAKED_FEATURE_KEYS:
                raise ValueError(f"training feature contains leaked key: {path}.{key_text}")
            assert_no_leaked_feature_keys(item, path=f"{path}.{key_text}")
        return
    if isinstance(value, tuple | list):
        for index, item in enumerate(value):
            assert_no_leaked_feature_keys(item, path=f"{path}[{index}]")
