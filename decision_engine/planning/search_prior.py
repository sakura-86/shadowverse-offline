# -*- coding: utf-8 -*-
"""Neutral search prior interfaces for Phase N1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from decision_engine.actions.model import LegalAction
from decision_engine.state.snapshot import BattleSnapshot


@dataclass(frozen=True)
class PriorResult:
    priors: tuple[float, ...]
    source: str = "uniform"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, action_count: int) -> None:
        if len(self.priors) != action_count:
            raise ValueError("prior length must equal legal action count")
        if any(value < 0.0 for value in self.priors):
            raise ValueError("prior values must be non-negative")
        if self.priors and abs(sum(self.priors) - 1.0) > 1e-6:
            raise ValueError("prior values must sum to 1.0")


class SearchPrior(Protocol):
    def priors(self, snapshot: BattleSnapshot, actions: Sequence[LegalAction], *, player: str) -> PriorResult:
        ...


class UniformSearchPrior:
    def priors(self, snapshot: BattleSnapshot, actions: Sequence[LegalAction], *, player: str) -> PriorResult:
        del snapshot
        if player not in {"me", "opponent"}:
            raise ValueError(f"unsupported search prior player: {player}")
        action_count = len(actions)
        if action_count <= 0:
            raise ValueError("uniform prior requires at least one legal action")
        value = 1.0 / action_count
        result = PriorResult(priors=tuple(value for _ in actions), source="uniform", diagnostics={"action_count": action_count, "player": player})
        result.validate(action_count)
        return result


class StructuredNeuralPrior:
    """用 structured encoder 的 policy head 输出作为 PUCT 搜索先验。

    从 BattleSnapshot + legal_actions → structured tensorize → model forward → softmax → priors。
    """

    source: str = "structured_neural"

    def __init__(self, checkpoint_path: str | None = None, *, device: str = "cpu", temperature: float = 1.0):
        self._device = device
        self._temperature = temperature
        self._model = None
        self._state_cache: dict[tuple, dict[str, Any]] = {}  # fingerprint → state tensors
        # 增幅饱和 gate（实验参数，默认 0 = 完全关闭，行为与旧版逐位一致）。
        # 语义：在 softmax 前给每个动作的 logit 加 ``β × saturation``，saturation 取该动作
        # source 卡的 ``min(sb, cap)/cap``（1 = 已满档/0 费，此刻打出不再丢失可攒进度）。
        # 为什么放在推理侧而不是模型参数里：六个加载点都是 strict=True，新增 nn.Parameter
        # 会让全部旧 checkpoint 失效；推理侧加项既不改 state_dict，也能直接扫 β 找方向。
        # 方向验证通过后再考虑做成可学习参数（那时需要统一放宽加载的 strict）。
        import os as _os
        raw_gate = _os.environ.get("EP_BOOST_GATE")
        try:
            self._boost_gate = float(raw_gate) if raw_gate else 0.0
        except ValueError as exc:
            raise ValueError(f"EP_BOOST_GATE 必须是数字，收到 {raw_gate!r}") from exc
        if checkpoint_path is not None:
            self._load_model(checkpoint_path)

    def clear_cache(self):
        """Clear state tensor cache. Call at start of each PUCT search step."""
        self._state_cache.clear()

    def _load_model(self, checkpoint_path: str) -> None:
        from pathlib import Path
        try:
            import torch
            from decision_engine.training.card_vocabulary import load_card_vocabulary
            from decision_engine.training.structured_policy_value import (
                StructuredModelConfig, StructuredPolicyValueNet, _collate_structured
            )
        except ImportError as e:
            raise RuntimeError(f"StructuredNeuralPrior requires PyTorch: {e}") from e

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(path, map_location=self._device, weights_only=True)
        vocab_size = ckpt["vocab_size"]
        state_scalar_width = ckpt["state_scalar_width"]
        action_scalar_width = ckpt["action_scalar_width"]
        cfg_dict = ckpt.get("config", {})
        config = StructuredModelConfig(**{k: v for k, v in cfg_dict.items() if k in StructuredModelConfig.__dataclass_fields__})

        vocabulary_path = path.parent / "card_vocabulary.json"
        if vocabulary_path.exists():
            vocabulary = load_card_vocabulary(vocabulary_path)
            if vocabulary.vocab_size != vocab_size:
                raise ValueError("checkpoint vocabulary size does not match card_vocabulary.json")
            vocabulary_mode = "base_card_id"
        else:
            from decision_engine.training.card_vocabulary import build_legacy_card_vocabulary
            database_path = Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"
            vocabulary = build_legacy_card_vocabulary(database_path)
            if vocabulary.vocab_size != vocab_size:
                raise ValueError("legacy checkpoint vocabulary is incompatible with current card database")
            vocabulary_mode = "legacy_database_order"

        model = StructuredPolicyValueNet(
            vocab_size=vocab_size,
            state_scalar_width=state_scalar_width,
            action_scalar_width=action_scalar_width,
            config=config,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self._device)
        model.eval()
        self._model = model
        self._torch = torch
        self._collate = _collate_structured
        self._vocabulary = vocabulary
        self._vocabulary_mode = vocabulary_mode

    def priors(self, snapshot: BattleSnapshot, actions: Sequence[LegalAction], *, player: str) -> PriorResult:
        if player not in ("me", "opponent"):
            raise ValueError(f"unsupported player: {player}")
        action_count = len(actions)
        if action_count <= 0:
            raise ValueError("neural prior requires at least one legal action")

        if self._model is None:
            raise RuntimeError("StructuredNeuralPrior: model not loaded")

        from decision_engine.planning._fast_tensorize import (
            encode_state, encode_actions, snapshot_fingerprint,
        )

        # State encoding with cache
        fp = snapshot_fingerprint(snapshot, player)
        state_tensors = self._state_cache.get(fp)
        if state_tensors is None:
            state_tensors = encode_state(snapshot, player, self._device, vocabulary=self._vocabulary)
            self._state_cache[fp] = state_tensors

        # Action encoding (always fresh — different actions per node)
        action_tensors = encode_actions(snapshot, actions, player, self._device, vocabulary=self._vocabulary)

        # Combine into batch
        batch = {**state_tensors, **action_tensors}

        torch = self._torch
        with torch.no_grad():
            logits, _ = self._model(batch)
            if self._boost_gate != 0.0:
                from decision_engine.training.structured_tensorization import (
                    CARD_STAT_BOOST_PROGRESS_INDEX,
                )
                saturation_index = CARD_STAT_BOOST_PROGRESS_INDEX[0]
                saturation = batch["action_source_stats"][:, :action_count, saturation_index]
                logits = logits[:, :action_count] + self._boost_gate * saturation

        # 取前 action_count 个 logits，应用 temperature，softmax
        raw_logits = logits[0, :action_count]
        if self._temperature != 1.0:
            raw_logits = raw_logits / self._temperature
        probs = torch.softmax(raw_logits, dim=0)
        prior_values = tuple(float(p) for p in probs)

        result = PriorResult(priors=prior_values, source=self.source, diagnostics={"action_count": action_count, "player": player, "temperature": self._temperature, "card_identity_kind": self._vocabulary_mode, "card_vocabulary_digest": self._vocabulary.digest})
        result.validate(action_count)
        return result
