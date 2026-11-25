# -*- coding: utf-8 -*-
"""Neutral search value interfaces for Phase N1."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import tanh
from typing import Any, Mapping, Protocol

from decision_engine.execution.rule_kernel import deckout_winner_from_results
from decision_engine.state.snapshot import BattleSnapshot, CardType
from decision_engine.training.tensorization import ACTION_VECTOR_DIM


@dataclass(frozen=True)
class ValueResult:
    value: float
    source: str
    terminal: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.value < -1.0 or self.value > 1.0:
            raise ValueError("search value must be in [-1.0, 1.0]")
        if not self.source:
            raise ValueError("search value source must be non-empty")


class SearchValue(Protocol):
    def evaluate(self, snapshot: BattleSnapshot, *, player: str) -> ValueResult:
        ...


class BoardAdvantageValue:
    """基于场面优势的叶节点估值。

    设计视角：对方回合开始时的场面评价。

    随从有效威胁 = min(atk, life)：
    - atk 代表潜在伤害，life 代表存活能力。
    - 非疾驰随从必须活到下个回合才能打脸，所以 atk 再高、life 为 1 也容易被解。
    - 反之 life 高但 atk=0 也不构成威胁。
    - 取两者较小值代表"能实际兑现的压力"。

    评分规则：
    - 我方场面有效威胁总和越高越好。
    - 对方场面有效威胁总和越高越差，权重为我方的 2 倍（留场 = 被打脸风险）。
    - life 差作为辅助信号。
    - 终局直接返回 ±1。

    输出 [-1, 1]，用 tanh 归一化。
    """

    source: str = "board_advantage"

    def __init__(
        self,
        *,
        my_board_weight: float = 1.0,
        opp_board_weight: float = 2.0,
        life_weight: float = 0.3,
        scale: float = 10.0,
    ):
        self.my_board_weight = my_board_weight
        self.opp_board_weight = opp_board_weight
        self.life_weight = life_weight
        self.scale = scale

    def evaluate(self, snapshot: BattleSnapshot, *, player: str) -> ValueResult:
        if player not in ("me", "opponent"):
            raise ValueError(f"unsupported player: {player}")

        # 终局直接返回确定值
        winner = _winner(snapshot)
        if winner is not None:
            if winner == "simultaneous":
                return ValueResult(value=0.0, source="terminal_simultaneous", terminal=True, diagnostics={"player": player, "winner": winner})
            v = 1.0 if winner == player else -1.0
            return ValueResult(value=v, source="terminal_result", terminal=True, diagnostics={"player": player, "winner": winner})

        # 确定己方/对方 PlayerView
        if player == "me":
            my_view = snapshot.me
            opp_view = snapshot.opponent
        else:
            my_view = snapshot.opponent
            opp_view = snapshot.me

        # 计算场面有效威胁（只计随从，不计护符）
        # 有效威胁 = min(atk, life)：攻击力和存活能力中的短板
        my_board_threat = 0
        opp_board_threat = 0
        for uid in my_view.field:
            card = snapshot.cards_by_uid.get(uid)
            if card is not None and card.card_type.is_follower:
                atk = max(0, card.atk)
                life = max(0, card.life)
                my_board_threat += min(atk, life) if atk > 0 else 0
        for uid in opp_view.field:
            card = snapshot.cards_by_uid.get(uid)
            if card is not None and card.card_type.is_follower:
                atk = max(0, card.atk)
                life = max(0, card.life)
                opp_board_threat += min(atk, life) if atk > 0 else 0

        # 场面分：我方场面加分，对方场面扣分（2x 权重）
        board_score = (self.my_board_weight * my_board_threat
                       - self.opp_board_weight * opp_board_threat)

        # life 辅助信号
        life_score = self.life_weight * (my_view.life - opp_view.life)

        # 总分用 tanh 压到 [-1, 1]
        raw = (board_score + life_score) / self.scale
        value = tanh(raw)

        diagnostics = {
            "player": player,
            "my_board_threat": my_board_threat,
            "opp_board_threat": opp_board_threat,
            "board_score": board_score,
            "life_score": life_score,
            "raw": raw,
            "value": value,
        }
        result = ValueResult(value=value, source=self.source, terminal=False, diagnostics=diagnostics)
        result.validate()
        return result


class NeuralValueFunction:
    """用训练出的 value head 作为 PUCT 叶节点评估函数。

    从 BattleSnapshot 提取 state_features → state_vector → model forward → value。
    终局时直接返回 ±1，不走 model。
    """

    source: str = "neural_value"

    def __init__(self, checkpoint_path: str | None = None, *, device: str = "cpu"):
        self._device = device
        self._model = None
        if checkpoint_path is not None:
            self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> None:
        from pathlib import Path
        # 延迟导入避免循环依赖和无 torch 环境
        try:
            import torch
            from decision_engine.training.mlp_policy_value import MlpPolicyValueNet, MlpPolicyValueConfig
        except ImportError as e:
            raise RuntimeError(f"NeuralValueFunction requires PyTorch: {e}") from e

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(path, map_location=self._device, weights_only=True)
        # 支持两种格式：直接 state_dict 或包装的 checkpoint dict
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_width = state_dict.get("state_vector_width", 65)
            action_width = state_dict.get("action_vector_width", ACTION_VECTOR_DIM)
            state_dict = state_dict["model_state_dict"]
        else:
            # 从权重推断维度
            state_width = state_dict["state_encoder.0.weight"].shape[1]
            action_width = state_dict["action_encoder.0.weight"].shape[1]
        config = MlpPolicyValueConfig()
        model = MlpPolicyValueNet(state_width=state_width, action_width=action_width, config=config)
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()
        self._model = model
        self._torch = torch

    def evaluate(self, snapshot: BattleSnapshot, *, player: str) -> ValueResult:
        if player not in ("me", "opponent"):
            raise ValueError(f"unsupported player: {player}")

        # 终局直接返回确定值
        winner = _winner(snapshot)
        if winner is not None:
            if winner == "simultaneous":
                return ValueResult(value=0.0, source="neural_terminal_simultaneous", terminal=True, diagnostics={"player": player, "winner": winner})
            v = 1.0 if winner == player else -1.0
            return ValueResult(value=v, source="neural_terminal_result", terminal=True, diagnostics={"player": player, "winner": winner})

        if self._model is None:
            raise RuntimeError("NeuralValueFunction: model not loaded (no checkpoint provided)")

        # BattleSnapshot → state_features → state_vector → forward
        from decision_engine.training.state_features import state_features
        from decision_engine.training.tensorization import state_vector_from_state_features

        features = state_features(snapshot, perspective=player)
        state_vector = state_vector_from_state_features(features)

        torch = self._torch
        state_t = torch.tensor([state_vector], dtype=torch.float32, device=self._device)
        # Value head 只需要 state，但 forward 需要 action_batch 和 mask
        # 用 dummy action (1 action, action_width dims) 来走 forward
        action_width = self._model.action_encoder[0].in_features
        action_t = torch.zeros(1, 1, action_width, dtype=torch.float32, device=self._device)
        mask_t = torch.ones(1, 1, dtype=torch.bool, device=self._device)

        with torch.no_grad():
            _, value = self._model(state_t, action_t, mask_t)

        v = float(value[0].item())
        # Clamp to [-1, 1]
        v = max(-1.0, min(1.0, v))

        result = ValueResult(value=v, source=self.source, terminal=False, diagnostics={"player": player, "raw_value": v})
        result.validate()
        return result


class StructuredNeuralValueFunction:
    """用 structured encoder 的 value head 作为 PUCT 叶节点评估。

    从 BattleSnapshot 提取 structured features → collate → forward → value。
    终局直接返回 ±1。
    """

    source: str = "structured_neural_value"

    def __init__(self, checkpoint_path: str | None = None, *, device: str = "cpu"):
        self._device = device
        self._model = None
        self._state_cache: dict[tuple, dict] = {}  # fingerprint → state tensors
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
            raise RuntimeError(f"StructuredNeuralValueFunction requires PyTorch: {e}") from e

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
        else:
            from decision_engine.training.card_vocabulary import build_legacy_card_vocabulary
            database_path = Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"
            vocabulary = build_legacy_card_vocabulary(database_path)
            if vocabulary.vocab_size != vocab_size:
                raise ValueError("legacy checkpoint vocabulary is incompatible with current card database")

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

    def evaluate(self, snapshot: BattleSnapshot, *, player: str) -> ValueResult:
        if player not in ("me", "opponent"):
            raise ValueError(f"unsupported player: {player}")

        # 终局直接返回
        winner = _winner(snapshot)
        if winner is not None:
            if winner == "simultaneous":
                return ValueResult(value=0.0, source="structured_neural_terminal_simultaneous", terminal=True, diagnostics={"player": player, "winner": winner})
            v = 1.0 if winner == player else -1.0
            return ValueResult(value=v, source="structured_neural_terminal_result", terminal=True, diagnostics={"player": player, "winner": winner})

        if self._model is None:
            raise RuntimeError("StructuredNeuralValueFunction: model not loaded")

        from decision_engine.planning._fast_tensorize import (
            encode_state, encode_value_input, snapshot_fingerprint,
        )

        # State encoding with cache（与 prior 同款 fingerprint，可跨 prior/value 复用）
        fp = snapshot_fingerprint(snapshot, player)
        state_tensors = self._state_cache.get(fp)
        if state_tensors is None:
            state_tensors = encode_state(snapshot, player, self._device, vocabulary=self._vocabulary)
            self._state_cache[fp] = state_tensors

        batch = encode_value_input(
            snapshot, player, self._device,
            vocabulary=self._vocabulary, state_tensors=state_tensors,
        )

        torch = self._torch
        with torch.no_grad():
            _, value = self._model(batch)

        v = float(value[0].item())
        v = max(-1.0, min(1.0, v))

        result = ValueResult(value=v, source=self.source, terminal=False, diagnostics={"player": player, "raw_value": v})
        result.validate()
        return result


class TerminalValue:
    def evaluate(self, snapshot: BattleSnapshot, *, player: str) -> ValueResult:
        if player not in {"me", "opponent"}:
            raise ValueError(f"unsupported search value player: {player}")
        winner = _winner(snapshot)
        if winner is None:
            result = ValueResult(value=0.0, source="non_terminal_zero_baseline", terminal=False, diagnostics={"player": player})
        elif winner == "simultaneous":
            result = ValueResult(value=0.0, source="terminal_simultaneous", terminal=True, diagnostics={"player": player, "winner": winner})
        else:
            result = ValueResult(value=1.0 if winner == player else -1.0, source="terminal_result", terminal=True, diagnostics={"player": player, "winner": winner})
        result.validate()
        return result


class NeuralSearchValue:
    """用神经网络的 value head 给 PUCT 叶节点估值。

    与 ``BoardAdvantageValue`` / ``DiffSearchValue`` 这些 heuristic 估值器的区别：
    它直接复用 PPO 训练出来的 value head，所以 prior 与 value 来自**同一个模型**，
    不会出现「prior 说这步好、heuristic value 说这步差」的语义冲突。

    ``value_fn`` 是 ``(snapshot, player) -> float``，由调用方注入（通常是
    ``StructuredNeuralPrior`` 持有的模型）。这样本模块不依赖训练侧代码，避免循环导入。

    模型 value head 在 ``value_activation=linear`` 下输出无界，而 ``ValueResult``
    要求 [-1, 1]，所以这里显式 clip——这是边界保护，不是静默截断：原始值保留在
    ``diagnostics.raw_value`` 里，越界时可以直接看出来。
    """

    def __init__(self, value_fn: Any, *, source: str = "structured_neural_value", clip: float = 1.0):
        if not callable(value_fn):
            raise TypeError("NeuralSearchValue 需要可调用的 value_fn(snapshot, player) -> float")
        self._value_fn = value_fn
        self.source = source
        self._clip = float(clip)
        if self._clip <= 0:
            raise ValueError("clip 必须为正")

    def evaluate(self, snapshot: BattleSnapshot, *, player: str) -> ValueResult:
        if player not in ("me", "opponent"):
            raise ValueError(f"unsupported player: {player}")
        winner = _winner(snapshot)
        if winner is not None:
            if winner == "simultaneous":
                result = ValueResult(value=0.0, source="terminal_simultaneous", terminal=True, diagnostics={"player": player, "winner": winner})
            else:
                result = ValueResult(value=1.0 if winner == player else -1.0, source="terminal_result", terminal=True, diagnostics={"player": player, "winner": winner})
            result.validate()
            return result
        raw = float(self._value_fn(snapshot, player))
        result = ValueResult(
            value=max(-self._clip, min(self._clip, raw)),
            source=self.source,
            terminal=False,
            diagnostics={"player": player, "raw_value": raw},
        )
        result.validate()
        return result


def _winner(snapshot: BattleSnapshot) -> str | None:
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
