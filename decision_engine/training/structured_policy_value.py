# -*- coding: utf-8 -*-
"""Structured Policy-Value Network with card-level encoders.

Architecture:
    State Encoder:
        scalar_state (schema-defined d) → MLP → state_scalar_emb
        hand_cards[9] → card_encoder → mean_pool → hand_emb
        field_cards[5] → card_encoder → mean_pool → field_emb
        opp_field_cards[5] → card_encoder → mean_pool → opp_field_emb
        state_emb = MLP(concat(state_scalar_emb, hand_emb, field_emb, opp_field_emb))

    Action Encoder:
        action_scalar (schema-defined d) → MLP → action_scalar_emb
        source_card → card_encoder → source_emb
        target_cards[5] → card_encoder → mean_pool → target_emb
        action_emb = MLP(concat(action_scalar_emb, source_emb, target_emb))

    Card Encoder (shared):
        card_id → embedding(vocab_size, embed_dim)
        card_stats (25d) → Linear → stat_emb
        card_emb = ReLU(Linear(concat(embedding, stat_emb)))

    Heads:
        policy: per-action dot(state_emb, action_emb) or MLP(concat)
        value: MLP(state_emb) → tanh → scalar
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

from decision_engine.training.card_vocabulary import CardVocabulary, save_card_vocabulary
from decision_engine.training.feature_schema import TRAINING_FEATURE_SCHEMA_VERSION
from decision_engine.training.tensorization import ACTION_VECTOR_DIM
from decision_engine.training.structured_tensorization import CARD_STAT_DIM

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass(frozen=True)
class StructuredModelConfig:
    """Configuration for StructuredPolicyValueNet."""
    # Card encoder
    card_embed_dim: int = 32
    card_stat_hidden: int = 32
    card_emb_dim: int = 64  # output of card encoder

    # State encoder
    state_scalar_hidden: int = 128
    state_fusion_hidden: int = 256

    # Action encoder
    action_scalar_hidden: int = 64
    action_fusion_hidden: int = 128

    # Heads
    policy_hidden: int = 128
    value_hidden: int = 128

    # Training
    value_loss_weight: float = 1.0
    value_head_dropout: float = 0.0
    history_enabled: bool = False
    history_length: int = 4
    seed: int = 42

    #: value 输出激活。``tanh`` 是历史契约（输出恒在 [-1,1]，PUCT leaf value 与
    #: ``[-1,1]`` 的 BC 标签直接对齐），``linear`` 保留无界输出。
    #:
    #: 为什么需要 ``linear``：PPO 的 return 是"稠密项累积 + 终局 ±1"的**加性和**，
    #: 量级约 ±1.7（γ=0.99、一局 32 步时终局项本身就被打折到 ±0.74）。tanh 在 ±1
    #: 附近梯度趋零，critic 会卡在饱和区：实测 `run_greedy_500` 25 批的
    #: ``value_loss`` 全程在 0.49–1.32 波动、无下降趋势，等于对 return 只输出常数。
    #: 这与 `entity_policy_value.py` 的 ``value_activation`` 是同一套契约和同一个坑。
    #:
    #: 默认仍是 ``tanh``：旧 checkpoint 的 config 里没有这个字段，取默认值后行为不变；
    #: 且 tanh 无参数，切换开关不改 state_dict，双向换成都能直接加载。
    value_activation: str = "tanh"

    def __post_init__(self) -> None:
        if self.value_activation not in ("tanh", "linear"):
            raise ValueError(f"value_activation 不支持: {self.value_activation!r}")

    # Set attention pooling (replaces mean pooling when enabled)
    use_set_attention: bool = False
    num_attention_heads: int = 2
    attention_dropout: float = 0.0

    # 输入消融：把 card stat 的这些下标通道权重恒置 0 且屏蔽梯度。
    # 语义 = "网络看不到这些特征"，用于严格消融实验（同数据、同推理路径，
    # 唯一变量是该特征是否参与计算）。空元组 = 不消融。
    ablate_stat_dims: tuple[int, ...] = ()
    #: action scalar 向量里要屏蔽的输入维度（v10 效果标签消融用）
    ablate_action_dims: tuple[int, ...] = ()


if _TORCH_AVAILABLE:
    class SetAttentionPooling(nn.Module):
        """Self-attention over a set of card embeddings, then masked mean pool.

        Each card attends to all other cards in the set, capturing pairwise
        interactions (e.g. combo synergies) before aggregation.
        """

        def __init__(self, dim: int, num_heads: int = 2, dropout: float = 0.0):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(dim)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            x: [B, N, dim]  card embeddings
            mask: [B, N] bool  True = valid card position
            Returns: [B, dim]
            """
            B, N, dim = x.shape

            # 对于某些batch中集合完全为空的情况，直接返回零向量
            any_valid = mask.any(dim=1)  # [B]
            if not any_valid.all():
                # 混合处理：有卡的走 attention，全空的返回零
                result = torch.zeros(B, dim, device=x.device, dtype=x.dtype)
                if any_valid.any():
                    valid_idx = any_valid.nonzero(as_tuple=True)[0]
                    valid_x = x[valid_idx]
                    valid_mask = mask[valid_idx]
                    key_padding_mask = ~valid_mask
                    attn_out, _ = self.attn(valid_x, valid_x, valid_x, key_padding_mask=key_padding_mask)
                    normed = self.norm(valid_x + attn_out)
                    mask_exp = valid_mask.unsqueeze(-1).float()
                    pooled = (normed * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)
                    result[valid_idx] = pooled
                return result

            # Normal path: all batch items have at least one valid card
            key_padding_mask = ~mask  # [B, N]
            attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
            x = self.norm(x + attn_out)  # residual connection + LayerNorm

            # Masked mean pooling over attended representations
            mask_expanded = mask.unsqueeze(-1).float()  # [B, N, 1]
            masked_sum = (x * mask_expanded).sum(dim=1)  # [B, dim]
            count = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1]
            return masked_sum / count

    class CardEncoder(nn.Module):
        """共享的单卡编码器：embedding + stat MLP → card_emb."""

        def __init__(self, vocab_size: int, config: StructuredModelConfig):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, config.card_embed_dim, padding_idx=0)
            self.stat_proj = nn.Linear(CARD_STAT_DIM, config.card_stat_hidden)
            self.fuse = nn.Sequential(
                nn.Linear(config.card_embed_dim + config.card_stat_hidden, config.card_emb_dim),
                nn.ReLU(),
            )
            self._ablated_stat_dims: tuple[int, ...] = ()
            if config.ablate_stat_dims:
                self.apply_stat_ablation(config.ablate_stat_dims)

        def apply_stat_ablation(self, dims: Sequence[int]) -> None:
            """把指定输入通道的权重恒置 0 并屏蔽其梯度（输入消融）。幂等。

            为什么用"权重列置零 + 梯度 mask"而不是"把输入数据置零"：
            后者要求训练与推理两条路径同步改，任一侧漏改就是静默的分布不一致；
            而权重列恒 0 时，输入是什么都不影响前向，训练与推理天然同口径。
            """
            norm = tuple(sorted({int(d) for d in dims}))
            if not norm or self._ablated_stat_dims == norm:
                return
            width = self.stat_proj.weight.shape[1]
            out_of_range = [d for d in norm if d < 0 or d >= width]
            if out_of_range:
                raise ValueError(
                    f"ablate_stat_dims out of range: {out_of_range} (stat width={width})"
                )
            index = torch.tensor(norm, dtype=torch.long)
            with torch.no_grad():
                self.stat_proj.weight[:, index] = 0.0

            def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
                grad = grad.clone()
                grad[:, index] = 0.0
                return grad

            self.stat_proj.weight.register_hook(_mask_grad)
            self._ablated_stat_dims = norm

        def forward(self, card_ids: torch.Tensor, card_stats: torch.Tensor) -> torch.Tensor:
            """
            card_ids: [..., ] int tensor
            card_stats: [..., CARD_STAT_DIM] float tensor
            Returns: [..., card_emb_dim]
            """
            emb = self.embedding(card_ids)
            stat_emb = F.relu(self.stat_proj(card_stats))
            return self.fuse(torch.cat([emb, stat_emb], dim=-1))

    class StructuredPolicyValueNet(nn.Module):
        """Structured policy-value network with card-level encoding."""

        def __init__(self, *, vocab_size: int, state_scalar_width: int, action_scalar_width: int, config: StructuredModelConfig):
            super().__init__()
            self.config = config
            self.vocab_size = vocab_size

            # Shared card encoder
            self.card_encoder = CardEncoder(vocab_size, config)

            # Set attention pooling (optional, replaces mean pooling)
            if config.use_set_attention:
                self.card_set_attention = SetAttentionPooling(
                    dim=config.card_emb_dim,
                    num_heads=config.num_attention_heads,
                    dropout=config.attention_dropout,
                )
                self.crest_set_attention = SetAttentionPooling(
                    dim=config.card_emb_dim,
                    num_heads=config.num_attention_heads,
                    dropout=config.attention_dropout,
                )
            else:
                self.card_set_attention = None
                self.crest_set_attention = None

            # Crest encoder (simpler: embedding + 3d stats)
            self.crest_embedding = nn.Embedding(vocab_size, config.card_embed_dim, padding_idx=0)
            self.crest_stat_proj = nn.Linear(3, config.card_stat_hidden)  # CREST_STAT_DIM=3
            self.crest_fuse = nn.Sequential(
                nn.Linear(config.card_embed_dim + config.card_stat_hidden, config.card_emb_dim),
                nn.ReLU(),
            )

            # State scalar encoder
            self.state_scalar_encoder = nn.Sequential(
                nn.Linear(state_scalar_width, config.state_scalar_hidden),
                nn.ReLU(),
                nn.Linear(config.state_scalar_hidden, config.state_scalar_hidden),
                nn.ReLU(),
            )

            # State fusion: concat scalar + 4 * card_emb_dim (hand, field, opp_hand, opp_field) + 2 * card_emb_dim (crests)
            state_cat_dim = config.state_scalar_hidden + 6 * config.card_emb_dim
            self.state_fusion = nn.Sequential(
                nn.Linear(state_cat_dim, config.state_fusion_hidden),
                nn.ReLU(),
                nn.Linear(config.state_fusion_hidden, config.state_fusion_hidden),
                nn.ReLU(),
            )

            if config.history_enabled:
                self.history_position_embedding = nn.Embedding(config.history_length, state_scalar_width)
                self.history_encoder = nn.GRU(state_scalar_width, config.state_fusion_hidden, batch_first=True)
                self.history_fusion = nn.Sequential(
                    nn.Linear(2 * config.state_fusion_hidden, config.state_fusion_hidden),
                    nn.ReLU(),
                )

            # Action scalar encoder
            self.action_scalar_encoder = nn.Sequential(
                nn.Linear(action_scalar_width, config.action_scalar_hidden),
                nn.ReLU(),
                nn.Linear(config.action_scalar_hidden, config.action_scalar_hidden),
                nn.ReLU(),
            )

            self._ablated_action_dims: tuple[int, ...] = ()
            if config.ablate_action_dims:
                self.apply_action_dim_ablation(config.ablate_action_dims)

            # Action fusion: concat scalar + source_emb + target_pool_emb
            action_cat_dim = config.action_scalar_hidden + 2 * config.card_emb_dim
            self.action_fusion = nn.Sequential(
                nn.Linear(action_cat_dim, config.action_fusion_hidden),
                nn.ReLU(),
                nn.Linear(config.action_fusion_hidden, config.action_fusion_hidden),
                nn.ReLU(),
            )

            # Policy head: state_emb + action_emb → logit
            self.policy_head = nn.Sequential(
                nn.Linear(config.state_fusion_hidden + config.action_fusion_hidden, config.policy_hidden),
                nn.ReLU(),
                nn.Linear(config.policy_hidden, 1),
            )

            # Value head: state_emb → value
            # 激活不在 Sequential 里固化：``value_activation`` 决定是否套 tanh。tanh 无参数，
            # 所以放在前向里套既能保持 state_dict 与旧 checkpoint 完全一致，又能双向切换。
            value_layers = [
                nn.Linear(config.state_fusion_hidden, config.value_hidden),
                nn.ReLU(),
            ]
            if config.value_head_dropout > 0:
                value_layers.append(nn.Dropout(config.value_head_dropout))
            value_layers.append(nn.Linear(config.value_hidden, 1))
            self.value_head = nn.Sequential(*value_layers)

        def apply_action_dim_ablation(self, dims: Sequence[int]) -> None:
            """把 action scalar 指定输入通道的权重恒置 0 并屏蔽梯度（输入消融）。幂等。

            与 ``CardEncoder.apply_stat_ablation`` 同一套口径：改权重列而不是改数据，
            训练与推理共享同一前向路径，不会出现"训练时置零、推理时没置零"的静默不一致。
            """
            norm = tuple(sorted({int(d) for d in dims}))
            if not norm or self._ablated_action_dims == norm:
                return
            first = self.action_scalar_encoder[0]
            width = first.weight.shape[1]
            out_of_range = [d for d in norm if d < 0 or d >= width]
            if out_of_range:
                raise ValueError(
                    f"ablate_action_dims out of range: {out_of_range} (action width={width})"
                )
            index = torch.tensor(norm, dtype=torch.long)
            with torch.no_grad():
                first.weight[:, index] = 0.0

            def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
                grad = grad.clone()
                grad[:, index] = 0.0
                return grad

            first.weight.register_hook(_mask_grad)
            self._ablated_action_dims = norm

        def _encode_card_set(self, card_ids: torch.Tensor, card_stats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            Encode a set of cards with attention pooling (if enabled) or mean pooling.
            card_ids: [B, N]
            card_stats: [B, N, CARD_STAT_DIM]
            mask: [B, N] bool
            Returns: [B, card_emb_dim]
            """
            # [B, N, card_emb_dim]
            card_embs = self.card_encoder(card_ids, card_stats)

            if self.card_set_attention is not None:
                return self.card_set_attention(card_embs, mask)

            # Fallback: masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()  # [B, N, 1]
            masked_sum = (card_embs * mask_expanded).sum(dim=1)  # [B, card_emb_dim]
            count = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1]
            return masked_sum / count

        def _encode_crest_set(self, crest_ids: torch.Tensor, crest_stats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            Encode a set of crests with attention pooling (if enabled) or mean pooling.
            crest_ids: [B, N]
            crest_stats: [B, N, 3]
            mask: [B, N] bool
            Returns: [B, card_emb_dim]
            """
            emb = self.crest_embedding(crest_ids)
            stat_emb = F.relu(self.crest_stat_proj(crest_stats))
            crest_embs = self.crest_fuse(torch.cat([emb, stat_emb], dim=-1))

            if self.crest_set_attention is not None:
                return self.crest_set_attention(crest_embs, mask)

            # Fallback: masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()
            masked_sum = (crest_embs * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1.0)
            return masked_sum / count

        def encode_state(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
            """Encode state into state_emb. Returns [B, state_fusion_hidden]."""
            # Scalar state
            scalar_emb = self.state_scalar_encoder(batch["state_scalar"])  # [B, state_scalar_hidden]

            # Card sets
            hand_emb = self._encode_card_set(batch["hand_card_ids"], batch["hand_card_stats"], batch["hand_card_mask"])
            field_emb = self._encode_card_set(batch["field_card_ids"], batch["field_card_stats"], batch["field_card_mask"])
            opp_hand_emb = self._encode_card_set(batch["opp_hand_card_ids"], batch["opp_hand_card_stats"], batch["opp_hand_card_mask"])
            opp_field_emb = self._encode_card_set(batch["opp_field_card_ids"], batch["opp_field_card_stats"], batch["opp_field_card_mask"])

            # Crest sets
            crest_emb = self._encode_crest_set(batch["crest_card_ids"], batch["crest_stats"], batch["crest_mask"])
            opp_crest_emb = self._encode_crest_set(batch["opp_crest_card_ids"], batch["opp_crest_stats"], batch["opp_crest_mask"])

            # Fuse
            state_cat = torch.cat([scalar_emb, hand_emb, field_emb, opp_hand_emb, opp_field_emb, crest_emb, opp_crest_emb], dim=-1)
            return self.state_fusion(state_cat)

        def encode_value_state(self, state_emb: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
            """Fuse current state with optional turn-boundary scalar history for value only."""
            if not self.config.history_enabled:
                return state_emb
            history = batch["history_state_scalar"]
            history_mask = batch["history_mask"]
            positions = torch.arange(self.config.history_length, device=history.device)
            history = history + self.history_position_embedding(positions).unsqueeze(0)
            history = history * history_mask.unsqueeze(-1).float()
            encoded, _ = self.history_encoder(history)
            last_positions = torch.where(
                history_mask,
                torch.arange(self.config.history_length, device=history.device).unsqueeze(0),
                torch.zeros_like(history_mask, dtype=torch.long),
            ).max(dim=1).values
            history_emb = encoded[torch.arange(encoded.shape[0], device=encoded.device), last_positions]
            history_emb = history_emb * history_mask.any(dim=1, keepdim=True).float()
            return self.history_fusion(torch.cat([state_emb, history_emb], dim=-1))

        def encode_actions(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
            """Encode actions. Returns [B, max_actions, action_fusion_hidden]."""
            # action_scalar: [B, max_actions, action_scalar_width]
            B, N, _ = batch["action_scalars"].shape
            action_scalar_flat = batch["action_scalars"].view(B * N, -1)
            action_scalar_emb = self.action_scalar_encoder(action_scalar_flat).view(B, N, -1)

            # Source card: [B, max_actions] → [B, max_actions, card_emb_dim]
            source_ids_flat = batch["action_source_ids"].view(B * N)
            source_stats_flat = batch["action_source_stats"].view(B * N, -1)
            source_emb = self.card_encoder(source_ids_flat, source_stats_flat).view(B, N, -1)

            # Target cards: [B, max_actions, MAX_TARGETS, ...] → mean pool → [B, max_actions, card_emb_dim]
            T = batch["action_target_ids"].shape[2]  # MAX_ACTION_TARGETS
            target_ids_flat = batch["action_target_ids"].view(B * N * T)
            target_stats_flat = batch["action_target_stats"].view(B * N * T, -1)
            target_embs = self.card_encoder(target_ids_flat, target_stats_flat).view(B, N, T, -1)
            target_mask = batch["action_target_mask"].unsqueeze(-1).float()  # [B, N, T, 1]
            target_sum = (target_embs * target_mask).sum(dim=2)  # [B, N, card_emb_dim]
            target_count = target_mask.sum(dim=2).clamp(min=1.0)  # [B, N, 1]
            target_pool = target_sum / target_count

            # Fuse
            action_cat = torch.cat([action_scalar_emb, source_emb, target_pool], dim=-1)
            action_emb_flat = self.action_fusion(action_cat.view(B * N, -1)).view(B, N, -1)
            return action_emb_flat

        def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.
            Returns:
                logits: [B, max_actions] (masked)
                value: [B]
            """
            state_emb = self.encode_state(batch)   # [B, state_fusion_hidden]
            action_emb = self.encode_actions(batch)  # [B, max_actions, action_fusion_hidden]

            # Policy: per-action
            B, N, _ = action_emb.shape
            state_expanded = state_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, state_fusion_hidden]
            policy_input = torch.cat([state_expanded, action_emb], dim=-1)  # [B, N, state+action]
            logits = self.policy_head(policy_input.view(B * N, -1)).view(B, N)  # [B, N]

            # Mask invalid actions
            action_mask = batch["action_mask"]  # [B, N] bool
            logits = logits.masked_fill(~action_mask, -1.0e9)

            # Value may consume temporal context; policy remains current-state only.
            value_state_emb = self.encode_value_state(state_emb, batch)
            value = self.value_head(value_state_emb).squeeze(-1)  # [B]
            # tanh 只在配置为 ``tanh`` 时套，用于保持旧 checkpoint 的 [-1,1] 输出契约；
            # PPO 的 return 无界，必须用 ``linear``，否则 critic 会锁在饱和区。
            if self.config.value_activation == "tanh":
                value = torch.tanh(value)

            return logits, value


# ──────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StructuredTrainingResult:
    checkpoint_path: Path
    train_metrics: dict[str, Any]
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    """Paths to all generated artifact files (run_manifest.json, model_artifact.json, etc.)"""


def _collate_structured(rows: Sequence[Mapping[str, Any]], *, device: str) -> dict[str, torch.Tensor]:
    """Collate a batch of structured tensorization results into tensors."""
    B = len(rows)
    # Infer dimensions
    first = rows[0]
    state_width = len(first["state_vector"])
    action_width = len(first["action_scalar_vectors"][0]) if first["action_scalar_vectors"] else ACTION_VECTOR_DIM
    max_actions = max(len(row["action_scalar_vectors"]) for row in rows)
    card_stat_dim = len(first["hand_card_stats"][0]) if first["hand_card_stats"] else CARD_STAT_DIM
    max_hand = len(first["hand_card_ids"])
    max_field = len(first["field_card_ids"])
    max_opp_field = len(first["opp_field_card_ids"])
    max_opp_hand = len(first.get("opp_hand_card_ids", ())) or 9
    max_crests = len(first["crest_card_ids"])
    max_targets = len(first["action_target_card_ids"][0]) if first["action_target_card_ids"] else 5

    # Allocate tensors
    state_scalar = torch.zeros(B, state_width, dtype=torch.float32, device=device)
    hand_card_ids = torch.zeros(B, max_hand, dtype=torch.long, device=device)
    hand_card_stats = torch.zeros(B, max_hand, card_stat_dim, dtype=torch.float32, device=device)
    hand_card_mask = torch.zeros(B, max_hand, dtype=torch.bool, device=device)
    field_card_ids = torch.zeros(B, max_field, dtype=torch.long, device=device)
    field_card_stats = torch.zeros(B, max_field, card_stat_dim, dtype=torch.float32, device=device)
    field_card_mask = torch.zeros(B, max_field, dtype=torch.bool, device=device)
    opp_field_card_ids = torch.zeros(B, max_opp_field, dtype=torch.long, device=device)
    opp_field_card_stats = torch.zeros(B, max_opp_field, card_stat_dim, dtype=torch.float32, device=device)
    opp_field_card_mask = torch.zeros(B, max_opp_field, dtype=torch.bool, device=device)
    opp_hand_card_ids = torch.zeros(B, max_opp_hand, dtype=torch.long, device=device)
    opp_hand_card_stats = torch.zeros(B, max_opp_hand, card_stat_dim, dtype=torch.float32, device=device)
    opp_hand_card_mask = torch.zeros(B, max_opp_hand, dtype=torch.bool, device=device)

    crest_card_ids = torch.zeros(B, max_crests, dtype=torch.long, device=device)
    crest_stats_t = torch.zeros(B, max_crests, 3, dtype=torch.float32, device=device)  # CREST_STAT_DIM=3
    crest_mask_t = torch.zeros(B, max_crests, dtype=torch.bool, device=device)
    opp_crest_card_ids = torch.zeros(B, max_crests, dtype=torch.long, device=device)
    opp_crest_stats_t = torch.zeros(B, max_crests, 3, dtype=torch.float32, device=device)
    opp_crest_mask_t = torch.zeros(B, max_crests, dtype=torch.bool, device=device)

    action_scalars = torch.zeros(B, max_actions, action_width, dtype=torch.float32, device=device)
    action_mask = torch.zeros(B, max_actions, dtype=torch.bool, device=device)
    action_source_ids = torch.zeros(B, max_actions, dtype=torch.long, device=device)
    action_source_stats = torch.zeros(B, max_actions, card_stat_dim, dtype=torch.float32, device=device)
    action_target_ids = torch.zeros(B, max_actions, max_targets, dtype=torch.long, device=device)
    action_target_stats = torch.zeros(B, max_actions, max_targets, card_stat_dim, dtype=torch.float32, device=device)
    action_target_mask = torch.zeros(B, max_actions, max_targets, dtype=torch.bool, device=device)

    history_length = len(first.get("history_state_vectors", ())) or 4
    history_state_scalar = torch.zeros(B, history_length, state_width, dtype=torch.float32, device=device)
    history_mask = torch.zeros(B, history_length, dtype=torch.bool, device=device)

    policy = torch.zeros(B, max_actions, dtype=torch.float32, device=device)
    value = torch.zeros(B, dtype=torch.float32, device=device)
    value_mask = torch.zeros(B, dtype=torch.bool, device=device)

    for b, row in enumerate(rows):
        # State scalar
        sv = row["state_vector"]
        state_scalar[b, :len(sv)] = torch.tensor(sv, dtype=torch.float32)

        # Hand cards
        for i, (cid, stats, m) in enumerate(zip(row["hand_card_ids"], row["hand_card_stats"], row["hand_card_mask"])):
            hand_card_ids[b, i] = int(cid)
            hand_card_stats[b, i] = torch.tensor(stats, dtype=torch.float32)
            hand_card_mask[b, i] = bool(m)

        # Field cards
        for i, (cid, stats, m) in enumerate(zip(row["field_card_ids"], row["field_card_stats"], row["field_card_mask"])):
            field_card_ids[b, i] = int(cid)
            field_card_stats[b, i] = torch.tensor(stats, dtype=torch.float32)
            field_card_mask[b, i] = bool(m)

        # Opp field cards
        for i, (cid, stats, m) in enumerate(zip(row["opp_field_card_ids"], row["opp_field_card_stats"], row["opp_field_card_mask"])):
            opp_field_card_ids[b, i] = int(cid)
            opp_field_card_stats[b, i] = torch.tensor(stats, dtype=torch.float32)
            opp_field_card_mask[b, i] = bool(m)

        # Opp hand cards (完美信息模式)
        for i, (cid, stats, m) in enumerate(zip(row.get("opp_hand_card_ids", ()), row.get("opp_hand_card_stats", ()), row.get("opp_hand_card_mask", ()))):
            opp_hand_card_ids[b, i] = int(cid)
            opp_hand_card_stats[b, i] = torch.tensor(stats, dtype=torch.float32)
            opp_hand_card_mask[b, i] = bool(m)

        # Crests
        for i, (cid, stats, m) in enumerate(zip(row["crest_card_ids"], row["crest_stats"], row["crest_mask"])):
            crest_card_ids[b, i] = int(cid)
            crest_stats_t[b, i] = torch.tensor(stats, dtype=torch.float32)
            crest_mask_t[b, i] = bool(m)

        for i, (cid, stats, m) in enumerate(zip(row["opp_crest_card_ids"], row["opp_crest_stats"], row["opp_crest_mask"])):
            opp_crest_card_ids[b, i] = int(cid)
            opp_crest_stats_t[b, i] = torch.tensor(stats, dtype=torch.float32)
            opp_crest_mask_t[b, i] = bool(m)

        # Historical turn-boundary scalar states, oldest -> newest.
        for i, vector in enumerate(row.get("history_state_vectors", ())):
            history_state_scalar[b, i] = torch.tensor(vector, dtype=torch.float32)
        for i, present in enumerate(row.get("history_mask", ())):
            history_mask[b, i] = bool(present)

        # Actions
        n_actions = len(row["action_scalar_vectors"])
        for i in range(n_actions):
            action_scalars[b, i] = torch.tensor(row["action_scalar_vectors"][i], dtype=torch.float32)
            action_mask[b, i] = True
            action_source_ids[b, i] = int(row["action_source_card_ids"][i])
            action_source_stats[b, i] = torch.tensor(row["action_source_card_stats"][i], dtype=torch.float32)
            for j in range(max_targets):
                if j < len(row["action_target_card_ids"][i]):
                    action_target_ids[b, i, j] = int(row["action_target_card_ids"][i][j])
                    action_target_stats[b, i, j] = torch.tensor(row["action_target_card_stats"][i][j], dtype=torch.float32)
                    action_target_mask[b, i, j] = bool(row["action_target_mask"][i][j])

            policy[b, i] = row["policy_target"][i]

        # Value
        vt = row.get("value_target")
        if vt is not None:
            value[b] = float(vt)
            value_mask[b] = True

    return {
        "state_scalar": state_scalar,
        "hand_card_ids": hand_card_ids,
        "hand_card_stats": hand_card_stats,
        "hand_card_mask": hand_card_mask,
        "field_card_ids": field_card_ids,
        "field_card_stats": field_card_stats,
        "field_card_mask": field_card_mask,
        "opp_field_card_ids": opp_field_card_ids,
        "opp_field_card_stats": opp_field_card_stats,
        "opp_field_card_mask": opp_field_card_mask,
        "opp_hand_card_ids": opp_hand_card_ids,
        "opp_hand_card_stats": opp_hand_card_stats,
        "opp_hand_card_mask": opp_hand_card_mask,
        "crest_card_ids": crest_card_ids,
        "crest_stats": crest_stats_t,
        "crest_mask": crest_mask_t,
        "opp_crest_card_ids": opp_crest_card_ids,
        "opp_crest_stats": opp_crest_stats_t,
        "opp_crest_mask": opp_crest_mask_t,
        "action_scalars": action_scalars,
        "action_mask": action_mask,
        "action_source_ids": action_source_ids,
        "action_source_stats": action_source_stats,
        "action_target_ids": action_target_ids,
        "action_target_stats": action_target_stats,
        "action_target_mask": action_target_mask,
        "history_state_scalar": history_state_scalar,
        "history_mask": history_mask,
        "policy": policy,
        "value": value,
        "value_mask": value_mask,
    }


def _batches(items, batch_size: int):
    """按 batch_size 产出批次。list/tuple 用切片；其它可迭代对象流式累积。"""
    if isinstance(items, (list, tuple)):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
        return
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch



def _load_warm_start(
    *,
    model: "StructuredPolicyValueNet",
    initial_checkpoint: Path | None,
    vocabulary: CardVocabulary | None,
    state_scalar_width: int,
    action_scalar_width: int,
    config: StructuredModelConfig,
    device: str,
) -> dict[str, Any]:
    """Load compatible weights and append-only embedding rows from a prior run."""
    if initial_checkpoint is None:
        return {"enabled": False, "reason": "no_initial_checkpoint"}
    if vocabulary is None:
        raise ValueError("warm-start requires an explicit CardVocabulary")
    if not initial_checkpoint.exists():
        raise FileNotFoundError(f"initial checkpoint not found: {initial_checkpoint}")

    checkpoint = torch.load(initial_checkpoint, map_location=device, weights_only=True)
    if int(checkpoint.get("state_scalar_width") or 0) != state_scalar_width:
        raise ValueError("warm-start state scalar width mismatch")
    if int(checkpoint.get("action_scalar_width") or 0) != action_scalar_width:
        raise ValueError("warm-start action scalar width mismatch")
    previous_config = StructuredModelConfig(**{
        key: value for key, value in dict(checkpoint.get("config") or {}).items()
        if key in StructuredModelConfig.__dataclass_fields__
    })
    architecture_fields = (
        "card_embed_dim", "card_stat_hidden", "card_emb_dim", "state_scalar_hidden",
        "state_fusion_hidden", "action_scalar_hidden", "action_fusion_hidden",
        "policy_hidden", "value_hidden", "value_head_dropout", "history_enabled",
        "history_length", "use_set_attention", "num_attention_heads", "attention_dropout",
    )
    if any(getattr(previous_config, field) != getattr(config, field) for field in architecture_fields):
        raise ValueError("warm-start model architecture mismatch")

    from decision_engine.training.card_vocabulary import build_legacy_card_vocabulary, load_card_vocabulary

    previous_vocab_path = initial_checkpoint.parent / "card_vocabulary.json"
    if previous_vocab_path.exists():
        previous_vocabulary = load_card_vocabulary(previous_vocab_path)
        previous_vocabulary_mode = "base_card_id"
    else:
        database_path = Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"
        previous_vocabulary = build_legacy_card_vocabulary(database_path)
        if previous_vocabulary.vocab_size != int(checkpoint.get("vocab_size") or 0):
            raise ValueError("legacy warm-start checkpoint is incompatible with current card database")
        previous_vocabulary_mode = "legacy_database_order"
    for base_card_id, old_index in previous_vocabulary.base_card_id_to_index.items():
        if vocabulary.base_card_id_to_index.get(base_card_id) != old_index:
            raise ValueError("warm-start vocabulary must preserve every prior base_card_id index")

    previous_state = checkpoint.get("model_state_dict")
    if not isinstance(previous_state, Mapping):
        raise ValueError("warm-start checkpoint lacks model_state_dict")
    current_state = model.state_dict()
    embedding_keys = {"card_encoder.embedding.weight", "crest_embedding.weight"}
    inherited_keys = 0
    for key, value in previous_state.items():
        if key not in current_state:
            raise ValueError(f"warm-start checkpoint has unknown parameter: {key}")
        if key in embedding_keys:
            if value.shape[1:] != current_state[key].shape[1:] or value.shape[0] > current_state[key].shape[0]:
                raise ValueError(f"warm-start embedding shape mismatch: {key}")
            current_state[key][:value.shape[0]].copy_(value)
        elif value.shape != current_state[key].shape:
            raise ValueError(f"warm-start parameter shape mismatch: {key}")
        else:
            current_state[key].copy_(value)
        inherited_keys += 1
    model.load_state_dict(current_state)
    return {
        "enabled": True,
        "initial_checkpoint": str(initial_checkpoint),
        "previous_vocabulary_digest": previous_vocabulary.digest,
        "previous_vocabulary_mode": previous_vocabulary_mode,
        "vocabulary_digest": vocabulary.digest,
        "inherited_base_cards": len(previous_vocabulary.base_card_id_to_index),
        "added_base_cards": vocabulary.vocab_size - previous_vocabulary.vocab_size,
        "inherited_parameter_keys": inherited_keys,
        "optimizer_state_inherited": False,
    }


def train_structured_policy_value(
    *,
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    run_id: str,
    vocab_size: int,
    vocabulary: CardVocabulary | None = None,
    initial_checkpoint: Path | None = None,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    config: StructuredModelConfig | None = None,
    freeze_value: bool = False,
    freeze_policy: bool = False,
    freeze_backbone: bool = False,
) -> StructuredTrainingResult:
    """Train structured policy-value model."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for structured training")

    cfg = config or StructuredModelConfig()
    torch.manual_seed(cfg.seed)

    from decision_engine.training.sharded_dataset import ShardedRowDataset

    raw_train = dataset.get("train", ())
    raw_val = dataset.get("val", ())
    # 惰性数据集保持流式；普通序列一次性转 list。
    train_rows = raw_train if isinstance(raw_train, ShardedRowDataset) else list(raw_train)
    val_rows = raw_val if isinstance(raw_val, ShardedRowDataset) else list(raw_val)
    if len(train_rows) == 0:
        raise ValueError("Need at least one training sample")

    # Infer widths（惰性数据集用 first_row 取样，不加载整表）
    _fr = train_rows.first_row() if isinstance(train_rows, ShardedRowDataset) else train_rows[0]
    state_scalar_width = len(_fr["state_vector"])
    action_scalar_width = len(_fr["action_scalar_vectors"][0]) if _fr["action_scalar_vectors"] else ACTION_VECTOR_DIM

    model = StructuredPolicyValueNet(
        vocab_size=vocab_size,
        state_scalar_width=state_scalar_width,
        action_scalar_width=action_scalar_width,
        config=cfg,
    ).to(device)

    if vocabulary is not None and vocabulary.vocab_size != vocab_size:
        raise ValueError("vocabulary vocab_size must match trainer vocab_size")
    warm_start = _load_warm_start(
        model=model,
        initial_checkpoint=initial_checkpoint,
        vocabulary=vocabulary,
        state_scalar_width=state_scalar_width,
        action_scalar_width=action_scalar_width,
        config=cfg,
        device=device,
    )
    # warm-start 会把权重整块覆盖掉，消融必须在加载之后重新施加一次
    # （`apply_stat_ablation` 幂等，无 warm-start 时这里是空操作）。
    if cfg.ablate_stat_dims:
        model.card_encoder.apply_stat_ablation(cfg.ablate_stat_dims)
    if cfg.ablate_action_dims:
        model.apply_action_dim_ablation(cfg.ablate_action_dims)

    # freeze_value: 保留 warm-start(如 vt_cold27)的 value 头(及其独享的 history 时序层),
    # 只训练 policy 头 + 共享 backbone。用于验证"人类锚定 value + 搜索增强 policy"路线。
    # 注意: policy loss 仍会更新共享 backbone(encode_state), value 头输入分布可能漂移,
    # 但 value 头自身权重被冻结,保留人类胜负关联的映射。
    if freeze_value:
        _frozen = 0
        for _name, _p in model.named_parameters():
            if _name.startswith("value_head") or _name.startswith("history_"):
                _p.requires_grad = False
                _frozen += 1
        import logging as _lg
        _lg.getLogger(__name__).info("  freeze_value: 冻结 %d 个 value/history 参数张量", _frozen)

    # freeze_policy: 与 freeze_value 对称的另一半——冻结共享 backbone 与 policy 头，
    # 只训练 value 头（含其独享的 history_ 时序层）。
    #
    # 为什么要这个开关：PUCT 的最终选择由 Q 值主导而非 prior。self-play 日志里大量
    # "selected X but highest-prior action is Y. Reason: value backup overrode policy."
    # 说明在 22 sims 的预算下，改 policy 对实际出牌影响很小，改 value 才影响决策。
    # 因此"人类数据到底能教什么"必须把 policy 和 value 分开测：
    #   freeze_value  → 只让人类数据改 policy（已测，2.1%）
    #   freeze_policy → 只让人类数据改 value（本开关）
    # 冻结 backbone 而非只冻 policy 头，是为了让 policy 输出**逐位不变**，
    # 从而把"胜率变化"完全归因到 value 头，避免共享 backbone 漂移污染对照。
    if freeze_policy:
        _frozen = 0
        for _name, _p in model.named_parameters():
            if _name.startswith("value_head") or _name.startswith("history_"):
                continue
            _p.requires_grad = False
            _frozen += 1
        import logging as _lg
        _lg.getLogger(__name__).info("  freeze_policy: 冻结 %d 个 backbone/policy 参数张量", _frozen)

    # freeze_backbone: 冻结**特征提取**（card_encoder / state_* / action_* 与 set attention），
    # 只训练 policy_head 与 value_head（含 value 独享的 history_ 时序层）。
    #
    # 与 freeze_policy 的区别只在 policy 头：
    #   freeze_policy  → 只训 value 头（policy 输出逐位不变）
    #   freeze_backbone→ 训 policy 头 + value 头（policy 输入表示不变，输出可变）
    #
    # 为什么需要这一格：实测人类数据能教会 policy 头的"何时兑现"条件
    # （bc_human_v2 在 sb>=20 时 top1 45.2%，与人类 37.1% 对齐），但整网胜率崩到
    # 6.2%。`sfull` 已量化"破坏 backbone"本身值约 10 点，所以需要一条
    # "让人类数据改行为、但不重定义表示"的路径。
    if freeze_backbone:
        _keep = ("policy_head", "value_head", "history_")
        _frozen = 0
        for _name, _p in model.named_parameters():
            if _name.startswith(_keep):
                continue
            _p.requires_grad = False
            _frozen += 1
        import logging as _lg
        _lg.getLogger(__name__).info("  freeze_backbone: 冻结 %d 个特征提取参数张量", _frozen)

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    rng = Random(cfg.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_reports: list[dict[str, Any]] = []

    # Early stopping state
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience = 5
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        if isinstance(train_rows, ShardedRowDataset):
            train_rows.shuffle(rng)
        else:
            rng.shuffle(train_rows)
        model.train()
        train_losses: list[dict[str, float]] = []

        for batch_rows in _batches(train_rows, batch_size):
            batch = _collate_structured(batch_rows, device=device)
            optimizer.zero_grad()
            logits, values = model(batch)

            # Policy loss: cross-entropy with soft targets
            log_probs = F.log_softmax(logits, dim=-1)
            policy_loss = -(batch["policy"] * log_probs).sum(dim=-1).mean()

            # Value loss: MSE on masked samples
            value_loss = torch.tensor(0.0, device=device)
            if batch["value_mask"].any():
                value_loss = F.mse_loss(values[batch["value_mask"]], batch["value"][batch["value_mask"]])

            total_loss = policy_loss + cfg.value_loss_weight * value_loss
            total_loss.backward()
            optimizer.step()

            train_losses.append({
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "total_loss": total_loss.item(),
            })

        # Epoch summary
        avg_policy = sum(d["policy_loss"] for d in train_losses) / len(train_losses)
        avg_value = sum(d["value_loss"] for d in train_losses) / len(train_losses)
        avg_total = sum(d["total_loss"] for d in train_losses) / len(train_losses)

        # Val eval —— 分批处理避免整个验证集一次性张量化导致 OOM。
        # 损失按样本数加权聚合，语义与"整集一次性 mean"一致。
        val_metrics = {}
        if len(val_rows) > 0:
            model.eval()
            eval_batch_size = max(batch_size, 256)
            policy_loss_sum = 0.0
            policy_sample_count = 0
            value_sq_error_sum = 0.0
            value_sample_count = 0
            with torch.no_grad():
                for val_chunk in _batches(val_rows, eval_batch_size):
                    val_batch = _collate_structured(val_chunk, device=device)
                    val_logits, val_values = model(val_batch)
                    val_log_probs = F.log_softmax(val_logits, dim=-1)
                    per_sample_ce = -(val_batch["policy"] * val_log_probs).sum(dim=-1)
                    policy_loss_sum += per_sample_ce.sum().item()
                    policy_sample_count += per_sample_ce.shape[0]
                    mask = val_batch["value_mask"]
                    if mask.any():
                        diff = val_values[mask] - val_batch["value"][mask]
                        value_sq_error_sum += (diff * diff).sum().item()
                        value_sample_count += int(mask.sum().item())
            val_policy_loss = policy_loss_sum / max(1, policy_sample_count)
            val_value_loss = value_sq_error_sum / value_sample_count if value_sample_count else 0.0
            val_metrics = {"val_policy_loss": val_policy_loss, "val_value_loss": val_value_loss}

        report = {"epoch": epoch, "train_policy_loss": avg_policy, "train_value_loss": avg_value, "train_total_loss": avg_total, **val_metrics}
        epoch_reports.append(report)

        # Early stopping check
        if val_rows:
            val_total = val_metrics.get("val_policy_loss", 0) + cfg.value_loss_weight * val_metrics.get("val_value_loss", 0)
            if val_total < best_val_loss:
                best_val_loss = val_total
                best_epoch = epoch
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val_total: {best_val_loss:.4f})")
                    break

        if epoch % 10 == 0 or epoch == epochs:
            vp_str = f"{val_metrics.get('val_policy_loss', 0):.4f}" if val_metrics else "?"
            print(f"  epoch {epoch}: train_policy={avg_policy:.4f} train_value={avg_value:.4f} val_policy={vp_str}")

    # Restore best model weights if early stopping was used
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"  Restored best model from epoch {best_epoch}")

    # Save checkpoint
    if vocabulary is not None:
        save_card_vocabulary(output_dir / "card_vocabulary.json", vocabulary)
    checkpoint_path = output_dir / "model.pt"
    torch.save({
        "schema": "StructuredPolicyValueCheckpoint",
        "version": 1,
        "run_id": run_id,
        "vocab_size": vocab_size,
        "card_identity_kind": "base_card_id",
        "card_vocabulary_digest": vocabulary.digest if vocabulary is not None else None,
        "state_scalar_width": state_scalar_width,
        "action_scalar_width": action_scalar_width,
        "config": {
            "card_embed_dim": cfg.card_embed_dim,
            "card_stat_hidden": cfg.card_stat_hidden,
            "card_emb_dim": cfg.card_emb_dim,
            "state_scalar_hidden": cfg.state_scalar_hidden,
            "state_fusion_hidden": cfg.state_fusion_hidden,
            "action_scalar_hidden": cfg.action_scalar_hidden,
            "action_fusion_hidden": cfg.action_fusion_hidden,
            "policy_hidden": cfg.policy_hidden,
            "value_hidden": cfg.value_hidden,
            "value_head_dropout": cfg.value_head_dropout,
            "history_enabled": cfg.history_enabled,
            "history_length": cfg.history_length,
            "use_set_attention": cfg.use_set_attention,
            "num_attention_heads": cfg.num_attention_heads,
            "attention_dropout": cfg.attention_dropout,
            "value_loss_weight": cfg.value_loss_weight,
            "ablate_stat_dims": list(cfg.ablate_stat_dims),
        },
        "model_state_dict": model.state_dict(),
    }, checkpoint_path)

    metrics = {
        "run_id": run_id,
        "epochs": epoch_reports,
        "final_train_policy_loss": epoch_reports[-1]["train_policy_loss"],
        "final_train_value_loss": epoch_reports[-1]["train_value_loss"],
        "final_val_policy_loss": epoch_reports[best_epoch - 1].get("val_policy_loss") if best_epoch > 0 else epoch_reports[-1].get("val_policy_loss"),
        "final_val_value_loss": epoch_reports[best_epoch - 1].get("val_value_loss") if best_epoch > 0 else epoch_reports[-1].get("val_value_loss"),
        "best_epoch": best_epoch,
        "early_stopped": patience_counter >= patience,
        "vocab_size": vocab_size,
        "card_identity_kind": "base_card_id",
        "card_vocabulary_digest": vocabulary.digest if vocabulary is not None else None,
        "warm_start": warm_start,
        "state_scalar_width": state_scalar_width,
        "action_scalar_width": action_scalar_width,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
    }

    # ── 产出 artifact bundle ──
    artifact_paths = _write_artifact_bundle(
        output_dir=output_dir,
        run_id=run_id,
        metrics=metrics,
        vocab_size=vocab_size,
        state_scalar_width=state_scalar_width,
        action_scalar_width=action_scalar_width,
        train_samples=len(train_rows),
        val_samples=len(val_rows),
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        input_dir=str(dataset.get("_source_path", "") or ""),
        vocabulary=vocabulary,
        warm_start=warm_start,
    )

    return StructuredTrainingResult(checkpoint_path=checkpoint_path, train_metrics=metrics, artifact_paths=artifact_paths)


def _write_artifact_bundle(
    *,
    output_dir: Path,
    run_id: str,
    metrics: dict[str, Any],
    vocab_size: int,
    state_scalar_width: int,
    action_scalar_width: int,
    train_samples: int,
    val_samples: int,
    epochs: int,
    checkpoint_path: Path,
    input_dir: str,
    vocabulary: CardVocabulary | None,
    warm_start: Mapping[str, Any],
) -> dict[str, Path]:
    """训练完成后自动产出完整 artifact bundle。

    产出文件：
        - train_metrics.json: 训练指标
        - model_artifact.json: 模型描述（runtime 加载需要）
        - run_manifest.json: 训练运行清单（runtime 加载需要）
        - candidate_report.json: 候选模型报告
        - promotion_decision.json: 晋升决策（默认 pending）
    """
    paths: dict[str, Path] = {}

    # 1. train_metrics.json
    train_metrics_path = output_dir / "train_metrics.json"
    _write_json(train_metrics_path, metrics)
    paths["train_metrics"] = train_metrics_path

    # 2. model_artifact.json
    model_artifact_path = output_dir / "model_artifact.json"
    model_artifact = {
        "schema": "PolicyValueModelArtifact",
        "version": 1,
        "run_id": run_id,
        "backend": "torch_structured",
        "model_kind": "policy_value_dual_head",
        "input_tensorization_schema": "StructuredTensorization",
        "dataset_schema": "StructuredPolicyValueDataset",
        "state_vector_width": state_scalar_width,
        "action_vector_width": action_scalar_width,
        "policy_head_width": action_scalar_width,
        "value_head_output": "scalar",
        "feature_schema_version": TRAINING_FEATURE_SCHEMA_VERSION,
        "metadata": {
            "checkpoint": str(checkpoint_path.name),
            "candidate": True,
            "source": run_id,
            "vocab_size": vocab_size,
            "card_identity_kind": "base_card_id",
            "card_vocabulary_file": "card_vocabulary.json" if vocabulary is not None else None,
            "card_vocabulary_digest": vocabulary.digest if vocabulary is not None else None,
            "warm_start": dict(warm_start),
        },
    }
    _write_json(model_artifact_path, model_artifact)
    paths["model_artifact"] = model_artifact_path

    # 3. run_manifest.json
    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest = {
        "schema": "PolicyValueTrainingRunManifest",
        "version": 1,
        "run_id": run_id,
        "input_dir": input_dir,
        "output_dir": str(output_dir),
        "tensor_dataset_summary": {
            "schema": "TensorDatasetSummary",
            "total_samples": train_samples + val_samples,
            "state_vector_width": state_scalar_width,
            "action_vector_width": action_scalar_width,
        },
        "training_summary": {
            "backend": "torch_structured",
            "candidate": True,
            "epochs": epochs,
            "vocab_size": vocab_size,
            "train_samples": train_samples,
            "val_samples": val_samples,
        },
    }
    _write_json(run_manifest_path, run_manifest)
    paths["run_manifest"] = run_manifest_path

    # 4. candidate_report.json
    candidate_report_path = output_dir / "candidate_report.json"
    candidate_report = {
        "schema": "PolicyValueCandidateRunReport",
        "version": 1,
        "run_id": run_id,
        "run_dir": str(output_dir),
        "backend": "torch_structured",
        "artifact_status": "trained",
        "quality_summary": {
            "policy_loss_final": metrics.get("final_train_policy_loss"),
            "value_loss_final": metrics.get("final_train_value_loss"),
            "val_policy_loss_final": metrics.get("final_val_policy_loss"),
            "val_value_loss_final": metrics.get("final_val_value_loss"),
            "train_samples": train_samples,
            "val_samples": val_samples,
        },
        "runtime_loader_contract": {
            "schema": "PolicyValueRuntimeLoaderContract",
            "version": 1,
            "required_files": ["run_manifest.json", "candidate_report.json", "promotion_decision.json", "model_artifact.json"],
            "backend": "torch_structured",
            "artifact_schema": "PolicyValueModelArtifact",
            "artifact_version": 1,
            "artifact_status": "trained",
        },
    }
    _write_json(candidate_report_path, candidate_report)
    paths["candidate_report"] = candidate_report_path

    # 5. promotion_decision.json
    promotion_decision_path = output_dir / "promotion_decision.json"
    promotion_decision = {
        "schema": "PolicyValuePromotionDecisionReport",
        "version": 1,
        "candidate_run_id": run_id,
        "baseline_run_id": "",
        "status": "pending",
        "warnings": ["auto_generated_pending_evaluation"],
        "evaluation_summary": {
            "backend": "torch_structured",
            "artifact_status": "trained",
        },
    }
    _write_json(promotion_decision_path, promotion_decision)
    paths["promotion_decision"] = promotion_decision_path

    return paths


def _write_json(path: Path, payload: Any) -> None:
    """写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
