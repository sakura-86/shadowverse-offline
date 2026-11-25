# -*- coding: utf-8 -*-
"""Mulligan 训练样本 schema 与构建器。

Mulligan 样本与对局内 TrainingExample 不同：
- 动作空间是每张手牌独立的二元决策（留/换），不是 N 选 1
- 标签来自对局胜负回传（value_target）
- 每局每方只产生 1 个 mulligan 样本
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from decision_engine.state.snapshot import BattleSnapshot
from decision_engine.training.feature_schema import to_jsonable
from decision_engine.training.mulligan_features import mulligan_state_features

MULLIGAN_EXAMPLE_SCHEMA = "MulliganTrainingExample"
MULLIGAN_EXAMPLE_VERSION = 1


@dataclass(frozen=True)
class MulliganTrainingExample:
    """Mulligan 阶段训练样本。

    Attributes:
        sample_id: 唯一样本标识。
        source: 数据来源元信息。
        trust: 信任等级信息。
        player: 视角玩家 "me" / "opponent"。
        state_features: mulligan_state_features 编码结果。
        mulligan_mask: 每张手牌的决策，True = 换掉，False = 保留。
            长度等于 hand_count。
        value_target: 该局最终胜负，+1.0 = 赢，-1.0 = 输，None = 未结算。
        generation_config: 生成配置。
        metadata: 附加元数据。
    """
    schema: str = MULLIGAN_EXAMPLE_SCHEMA
    version: int = MULLIGAN_EXAMPLE_VERSION
    sample_id: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    trust: dict[str, Any] = field(default_factory=dict)
    player: str = "me"
    state_features: dict[str, Any] = field(default_factory=dict)
    mulligan_mask: tuple[bool, ...] = ()
    value_target: float | None = None
    generation_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        validate_mulligan_example(self)

    def to_jsonable(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "version": self.version,
            "sample_id": self.sample_id,
            "source": to_jsonable(self.source),
            "trust": to_jsonable(self.trust),
            "player": self.player,
            "state_features": to_jsonable(self.state_features),
            "mulligan_mask": list(self.mulligan_mask),
            "value_target": self.value_target,
            "generation_config": to_jsonable(self.generation_config),
            "metadata": to_jsonable(self.metadata),
        }


def validate_mulligan_example(example: MulliganTrainingExample | Mapping[str, Any]) -> None:
    """验证 mulligan 训练样本的基本约束。"""
    data = _to_mapping(example)
    if data.get("schema") != MULLIGAN_EXAMPLE_SCHEMA:
        raise ValueError(f"mulligan example schema must be {MULLIGAN_EXAMPLE_SCHEMA}")
    if data.get("version") != MULLIGAN_EXAMPLE_VERSION:
        raise ValueError(f"mulligan example version must be {MULLIGAN_EXAMPLE_VERSION}")
    sample_id = data.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("mulligan example sample_id must be a non-empty string")
    mulligan_mask = data.get("mulligan_mask")
    if not isinstance(mulligan_mask, tuple | list):
        raise ValueError("mulligan example mulligan_mask must be a sequence of booleans")
    if not mulligan_mask:
        raise ValueError("mulligan example mulligan_mask must not be empty")
    for index, value in enumerate(mulligan_mask):
        if not isinstance(value, bool):
            raise ValueError(f"mulligan example mulligan_mask[{index}] must be bool")
    state_features = data.get("state_features")
    if not isinstance(state_features, Mapping):
        raise ValueError("mulligan example state_features must be a mapping")
    hand_count = state_features.get("hand_count")
    if isinstance(hand_count, int) and hand_count != len(mulligan_mask):
        raise ValueError("mulligan example mulligan_mask length must equal state_features.hand_count")
    value_target = data.get("value_target")
    if value_target is not None:
        if not isinstance(value_target, int | float):
            raise ValueError("mulligan example value_target must be numeric or None")
        if float(value_target) < -1.0 or float(value_target) > 1.0:
            raise ValueError("mulligan example value_target must be in [-1.0, 1.0]")


def build_mulligan_training_example(
    *,
    snapshot: BattleSnapshot,
    player: str,
    cards_to_change: Sequence[int],
    my_class: str = "",
    opponent_class: str = "",
    sample_id: str,
    source: Mapping[str, Any],
    trust: Mapping[str, Any],
    value_target: float | None = None,
    generation_config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MulliganTrainingExample:
    """从 snapshot 和换牌决策构建 MulliganTrainingExample。

    Args:
        snapshot: 换牌完成后的 snapshot（hand 已经是最终手牌）。
            注意：mulligan_mask 标记的是哪些牌被换掉了，
            但 state_features 编码的是换牌后的最终手牌。
            这样网络学习的是"起手拿到这些牌时的胜率"。
        player: "me" 或 "opponent"。
        cards_to_change: 实际被换掉的 uid 列表（来自 mulligan 前的手牌）。
        my_class: 视角玩家职业名。
        opponent_class: 对手职业名。
        sample_id: 唯一标识。
        source: 数据来源。
        trust: 信任等级。
        value_target: 胜负标签，对局结束后回填。
        generation_config: 生成配置。
        metadata: 附加元数据。

    Returns:
        MulliganTrainingExample 实例。
    """
    view = snapshot.me if player == "me" else snapshot.opponent
    change_set = set(cards_to_change)

    # mulligan_mask: 基于换牌前的手牌生成
    # 但由于 snapshot 是换牌后的状态，我们需要从 metadata 获取原始手牌
    # 这里我们改为：基于传入的 cards_to_change uid 数量生成 mask
    # state_features 编码的是换牌后的最终手牌
    features = mulligan_state_features(
        snapshot,
        perspective=player,
        my_class=my_class,
        opponent_class=opponent_class,
    )

    # 从 metadata 获取原始手牌来生成 mask
    original_hand = metadata.get("original_hand") if metadata else None
    if original_hand is not None and isinstance(original_hand, (list, tuple)):
        mulligan_mask = tuple(uid in change_set for uid in original_hand)
    else:
        # fallback: 用当前 hand 长度 + cards_to_change 数量推断
        # 换掉 N 张 → mask 中有 N 个 True
        hand_size = len(view.hand)
        change_count = len(cards_to_change)
        # 无法精确对应每个位置，标记前 change_count 个为 True（仅作为 fallback）
        mulligan_mask = tuple(i < change_count for i in range(hand_size))

    gen_config = dict(generation_config or {})
    if "kind" not in gen_config:
        gen_config["kind"] = "mulligan_self_play"

    example = MulliganTrainingExample(
        sample_id=sample_id,
        source=dict(source),
        trust=dict(trust),
        player=player,
        state_features=features,
        mulligan_mask=mulligan_mask,
        value_target=value_target,
        generation_config=gen_config,
        metadata=dict(metadata or {}),
    )
    example.validate()
    return example


def _to_mapping(example: MulliganTrainingExample | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(example, MulliganTrainingExample):
        return {
            "schema": example.schema,
            "version": example.version,
            "sample_id": example.sample_id,
            "source": example.source,
            "trust": example.trust,
            "player": example.player,
            "state_features": example.state_features,
            "mulligan_mask": example.mulligan_mask,
            "value_target": example.value_target,
            "generation_config": example.generation_config,
            "metadata": example.metadata,
        }
    return example
