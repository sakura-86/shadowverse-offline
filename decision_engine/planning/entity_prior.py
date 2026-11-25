# -*- coding: utf-8 -*-
"""Entity backend PUCT/runtime prior + value evaluator（方案 C 阶段 C6）.

从 BattleSnapshot + LegalAction[] 经 canonical entity 路径产出：
- policy prior（softmax over legal actions only）
- leaf value（contextual [STATE] token）

只消费 LegalAction，不生成/删除/修改动作（C6.7）。
DeckInstance available-only 契约由上游特征层如实标记；本类不静默 fallback（C6.8）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from decision_engine.actions.model import LegalAction
from decision_engine.state.snapshot import BattleSnapshot
from decision_engine.planning.search_prior import PriorResult
from decision_engine.planning.entity_features import (
    entity_batch_from_snapshot,
    entity_snapshot_fingerprint,
)


class EntityNeuralEvaluator:
    """加载 entity checkpoint，提供 PUCT prior 与 leaf value。"""

    source: str = "entity_transformer"

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu", temperature: float = 1.0):
        self._device = device
        self._temperature = temperature
        self._state_cache: dict[tuple, Any] = {}
        self._load(Path(checkpoint_path))

    def clear_cache(self) -> None:
        self._state_cache.clear()

    def _load(self, checkpoint_path: Path) -> None:
        try:
            import torch
            from decision_engine.training.card_vocabulary import load_card_vocabulary
            from decision_engine.training.entity_checkpoint import load_entity_checkpoint
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(f"EntityNeuralEvaluator requires PyTorch: {e}") from e

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        vocab_path = checkpoint_path.parent / "card_vocabulary.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"entity checkpoint requires card_vocabulary.json: {vocab_path}")
        vocabulary = load_card_vocabulary(vocab_path)
        # strict load：backend/architecture/vocab 不匹配会显式抛错
        self._model = load_entity_checkpoint(checkpoint_path, vocabulary=vocabulary, device=self._device)
        self._torch = torch
        self._vocabulary = vocabulary

    def _forward(self, snapshot: BattleSnapshot, actions: Sequence[LegalAction], player: str) -> tuple[Any, Any]:
        batch = entity_batch_from_snapshot(
            snapshot, actions, perspective=player, vocabulary=self._vocabulary, device=self._device,
        )
        with self._torch.no_grad():
            out = self._model(batch)
        return out["policy_logits"], out["value"]

    def priors(self, snapshot: BattleSnapshot, actions: Sequence[LegalAction], *, player: str) -> PriorResult:
        if player not in ("me", "opponent"):
            raise ValueError(f"unsupported player: {player}")
        action_count = len(actions)
        if action_count <= 0:
            raise ValueError("entity prior requires at least one legal action")

        logits, _ = self._forward(snapshot, actions, player)
        raw = logits[0, :action_count]
        if self._temperature != 1.0:
            raw = raw / self._temperature
        probs = self._torch.softmax(raw, dim=0)
        # softmax 是 float32 张量，逐元素转成 Python float 后累积误差可能使
        # 求和偏离 1.0 超过 PriorResult.validate 的 1e-6 容差（动作数越多越明显）。
        # 在产出端显式归一化，保证严格满足"prior 和为 1.0"契约，而不是放宽消费端校验。
        raw_values = [float(p) for p in probs]
        total = sum(raw_values)
        if total <= 0.0:
            raise ValueError("entity prior softmax produced non-positive sum")
        prior_values = tuple(v / total for v in raw_values)
        result = PriorResult(
            priors=prior_values, source=self.source,
            diagnostics={
                "action_count": action_count, "player": player,
                "temperature": self._temperature,
                "card_vocabulary_digest": self._vocabulary.digest,
                "backend": "torch_entity_transformer",
            },
        )
        result.validate(action_count)
        return result

    def value(self, snapshot: BattleSnapshot, actions: Sequence[LegalAction], *, player: str) -> float:
        if player not in ("me", "opponent"):
            raise ValueError(f"unsupported player: {player}")
        if len(actions) <= 0:
            raise ValueError("entity value requires at least one legal action")
        _, v = self._forward(snapshot, actions, player)
        return float(v[0])

    def fingerprint(self, snapshot: BattleSnapshot, player: str) -> tuple:
        return entity_snapshot_fingerprint(snapshot, player, vocabulary=self._vocabulary)


class EntityNeuralValueFunction:
    """entity backend 的 PUCT 叶节点 value function（实现 SearchValue 协议）。

    终局直接返回 ±1；否则用 contextual [STATE] token 的 value head。
    与 EntityNeuralEvaluator 共享同一 checkpoint / vocabulary / canonical 路径。
    """

    source: str = "entity_transformer_value"

    def __init__(self, evaluator: "EntityNeuralEvaluator | None" = None, *, checkpoint_path: str | Path | None = None, device: str = "cpu"):
        if evaluator is not None:
            self._ev = evaluator
        elif checkpoint_path is not None:
            self._ev = EntityNeuralEvaluator(checkpoint_path, device=device)
        else:
            raise ValueError("EntityNeuralValueFunction requires evaluator or checkpoint_path")

    def evaluate(self, snapshot: BattleSnapshot, *, player: str):
        from decision_engine.planning.search_value import ValueResult, _winner
        if player not in ("me", "opponent"):
            raise ValueError(f"unsupported player: {player}")
        winner = _winner(snapshot)
        if winner is not None:
            if winner == "simultaneous":
                return ValueResult(value=0.0, source="entity_transformer_terminal_simultaneous", terminal=True, diagnostics={"player": player, "winner": winner})
            v = 1.0 if winner == player else -1.0
            return ValueResult(value=v, source="entity_transformer_terminal_result", terminal=True, diagnostics={"player": player, "winner": winner})

        # 非终局：用一个 synthetic turn_end 动作构建 batch（value 来自 [STATE] token，与动作无关）
        from decision_engine.actions.model import ActionKind, LegalAction
        synthetic = (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)
        v = self._ev.value(snapshot, synthetic, player=player)
        v = max(-1.0, min(1.0, v))
        result = ValueResult(value=v, source=self.source, terminal=False, diagnostics={"player": player, "raw_value": v})
        result.validate()
        return result
