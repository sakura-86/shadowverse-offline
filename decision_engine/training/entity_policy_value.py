# -*- coding: utf-8 -*-
"""Cross-Zone Entity Transformer (方案 C 阶段 C3/C4).

用一个跨区域 Entity Transformer 替代旧 structured 模型的「六区域摘要拼接」。
序列由 entity_tensorization.collate_entity_rows 产出：

    [STATE] + self/opp hand/field cards + crests + self remaining-deck unique cards

流程（§5.3）：

    type-specific encoder (card / crest / state)
      -> common d_model projection
      -> add type/zone/owner/visibility/position embedding + count numeric projection
      -> TransformerEncoder × N (key_padding_mask 屏蔽 padding)
      -> contextual entity tokens
           ├─ contextual [STATE] token -> value head (tanh, [-1,1])
           └─ (C4) legal action query -> action-to-entity cross-attention -> policy logit

约束：

- padding token 通过 key_padding_mask 屏蔽，不作为 attention key/value。
- [STATE] token 始终有效（序列首位），避免全空 collection 导致 attention 全 mask。
- deck/hand 排列不变（不加顺序位置）；field 通过 position embedding 可区分。
- 不在 Transformer 内实现规则合法性或卡名级 attention bias。
- 所有超参写入 config，可序列化进 checkpoint / artifact；架构不匹配拒绝加载。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch 缺失时仅配置可用
    _TORCH_AVAILABLE = False

from decision_engine.training.entity_tensorization import (
    CARD_STAT_DIM,
    CREST_STAT_DIM,
)
from decision_engine.training.entity_position import (
    ENTITY_POSITION_OPP_LEADER,
    ENTITY_POSITION_SELF_LEADER,
    PHASE_NAMES,
    PHASE_SOURCE_POSITIONS,
    PHASE_TARGET_POSITIONS,
)

# entity/zone/owner/visibility ID 的枚举上界（append-only；扩充时同步增大）
NUM_ENTITY_TYPES = 8
NUM_ZONES = 8
NUM_OWNERS = 4
NUM_VISIBILITIES = 4
MAX_FIELD_POSITIONS = 8  # 0=none/state/hand/deck, 1..5 field 槽位，留裕量


@dataclass(frozen=True)
class EntityModelConfig:
    """Entity Transformer 配置（实验起点，非 schema 常量）。"""
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 256
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1

    card_embed_dim: int = 64
    card_stat_hidden: int = 64

    state_scalar_hidden: int = 128

    value_hidden: int = 128
    policy_hidden: int = 128

    value_loss_weight: float = 1.0
    #: value 输出激活。``tanh`` 是历史契约（输出恒在 [-1,1]，PUCT leaf value 直接可用），
    #: 但它在 ±1 附近梯度趋零：配 Adam lr=1e-3 时一旦第一层权重被推大就永久锁死，
    #: 实测两个 arm 的 value 都对所有样本输出 +1（见
    #: docs/plan/ENTITY_V2_DATA_RECIPE_EXPERIMENT.md §9）。``linear`` 保留无界输出，
    #: 梯度不消失；默认仍为 tanh，旧 checkpoint 加载行为不变。
    value_activation: str = "tanh"
    #: 在 card token 上叠加"卡牌效果文本向量"（见 `tools/build_card_text_embeddings.py`）。
    #: 默认关闭：保持旧 checkpoint 的 state_dict 完全不变，打开后才新增
    #: `card_text_embedding` / `card_text_proj` 两个参数组。
    card_text_embedding_enabled: bool = False
    card_text_embedding_dim: int = 128
    #: 文本通道投影的初始化缩放。0.5 = 与 id/stat 通道同量级但不压过它们。
    #: 实测 0.5 时对"没见过的卡"没有任何可测改善（card-disjoint split 下 val top-1
    #: 仍是 0.03–0.06），调大它可以验证是"信号太弱被淹没"还是"信号本身没用"。
    card_text_proj_scale: float = 0.5
    #: 是否把 `entity_card_ids` 喂给卡牌编码器。置 False 时全部替换为 UNKNOWN 索引，
    #: 模型只能靠 `card_stat`、结构特征和卡牌文本向量区分卡——用来回答"文本向量能否
    #: 独立承担认卡"，而不是和 id embedding 并列时被淹没。
    use_card_id: bool = True
    seed: int = 42
    # 架构标识，写入 artifact；加载时不匹配拒绝
    architecture: str = "cross_zone_entity_transformer"
    architecture_version: int = 1
    effect_spec_encoder_enabled: bool = False

    #: 动作 query 使用**目标槽**结构（pointer 动作表示），而不是把所有 target
    #: 平均池化成一个向量。开启后新增 ``slot_spec_embedding`` 等参数，
    #: 因此旧 checkpoint 必须保持该项为 False 才能加载。
    #:
    #: 数据侧需要 ``action_slot_spec_ids`` / ``action_slot_mask`` /
    #: ``action_target_slot_index``；训练数据里没有槽信息时（replay 来源的
    #: ``LegalAction`` 不带 ``target_group_specs``）自动退化为旧的 mean pool，
    #: 所以同一个模型可以同时跑旧数据与新数据。
    slot_aware_query_enabled: bool = False
    #: 单个动作的最大目标槽数，必须与 ``entity_tensorization.MAX_ACTION_SLOTS`` 一致。
    max_action_slots: int = 4
    #: slot spec 词表大小（``SlotSpecVocabulary.vocab_size``）。启用 slot-aware
    #: query 时必须 > 0。
    slot_spec_vocab_size: int = 0

    #: **生成式 pointer**：不再对候选动作打分，而是分开输出
    #: ``head_logits [B,H]`` / ``source_logits [B,P]`` / ``slot_logits [B,K,P]``，
    #: 由 ``resolve_pointer_decode`` 反查回 ``LegalAction``。
    #:
    #: 指向空间是固定位置坐标（见 ``entity_position.py``），所以不需要局内 uid。
    #: 打开后不再产出 ``policy_logits [B,A]``，旧训练管线需要走
    #: ``policy_logits_from_pointer`` 折算成动作分布。
    pointer_head_enabled: bool = False
    #: 动作头数量，必须 == ``len(training.action_pointer.ActionHead)``。
    num_action_heads: int = 12
    #: pointer 指向空间大小，必须 == ``entity_position.MAX_ENTITY_POSITIONS``。
    max_pointer_positions: int = 64
    #: 分解监督权重（head / source / slot）。联合 CE 始终计算；这三项是辅助信号，
    #: 让三个 pointer 头在早期就能直接看到梯度，而不是只经由折算式传导。
    #: 置 0 表示只用联合 CE。
    pointer_head_loss_weight: float = 0.1
    pointer_source_loss_weight: float = 0.1
    pointer_slot_loss_weight: float = 0.1
    #: 按回合内阶段收窄 pointer 的候选位置（``entity_position.PHASE_SOURCE_POSITIONS``
    #: / ``PHASE_TARGET_POSITIONS``）。**零新增参数**——阶段是规则确定的，不该让
    #: 网络从数据里学「哪个阶段能用哪些位置」。
    #:
    #: 需要 batch 带 ``phase_ids``；旧数据（无 phase）自动全放行，不退化成全 mask。
    pointer_phase_mask_enabled: bool = False
    #: **按动作头条件化 source 分布**：把 ``source_logits`` 从 ``[B,P]`` 扩成
    #: ``[B,H,P]``。
    #:
    #: 动机：``source`` 的合法集强依赖动作头——``ATTACK`` 的 source 是场上随从、
    #: ``PLAY`` 的 source 是手牌、``TURN_END`` 的 source 是主战者。共享一个
    #: ``[B,P]`` 分布等于逼网络对同一位置在不同动作头下给出同一个 logit，
    #: 这是「source 混淆」的结构性来源。阶段 mask（``pointer_phase_mask_enabled``）
    #: 只约束「哪个阶段」，不区分阶段内的动作头。
    #:
    #: 与 ``slot_logits`` 不同，这里**不需要教师强制**：模型一次输出全部 H 份
    #: source logits，折算时每个候选动作取自己 head 的那一片，训练与推理严格一致。
    pointer_head_conditioning_enabled: bool = False
    #: 条件化后对每个 head 的 source 分布做**零均值化**（只在该 head 的有效位置上）。
    #:
    #: 必需，不是可选技巧：加法分解 ``logit(a) = head[h] + source[h][s]`` 里，
    #: 「给 source[h] 全体加常数 c、同时 head[h] -= c」不改变任何 logit，
    #: 于是有 H 个不可辨识方向（无条件化时只有 1 个）。这些方向会把梯度浪费在
    #: 互相抵消的尺度漂移上，实测 top-1 从 91.9% 掉到 82.6%。归零后尺度只由
    #: ``head_logits`` 承担，退化方向回到 1 个。
    #:
    #: 置 False 会重新引入退化方向，**没有理由这么做**——``validate`` 直接拒绝这个
    #: 组合，而不是留一个「静默变差」的开关。
    pointer_head_centering_enabled: bool = True
    #: **按动作头条件化 slot 分布**：``slot_logits`` 从 ``[B,K,P]`` 扩成
    #: ``[B,H,K,P]``。
    #:
    #: §3.12 证明 head 条件化在 phase mask 之上整体是负收益（数据稀释），但在
    #: ``n6p`` 桶（≥6 候选）上是正收益（87.97% vs 86.08%）。slot 的语义**不能**
    #: 从 phase 推断（phase 只约束位置，不约束「这一槽在选什么」），所以这里值得
    #: 单独验证一次，而不是把 §3.12 的结论直接外推。
    #:
    #: 依赖 ``pointer_head_conditioning_enabled=True``：slot 条件化是它的延伸，
    #: 没有理由单独开启（slot 的 head 维度需要 source 已经按 head 分开才有意义）。
    pointer_slot_head_conditioning_enabled: bool = False

    def validate(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model {self.d_model} 必须能被 num_heads {self.num_heads} 整除")
        if self.value_activation not in ("tanh", "linear"):
            raise ValueError(f"value_activation 不支持: {self.value_activation!r}")
        if self.slot_aware_query_enabled and self.slot_spec_vocab_size <= 0:
            raise ValueError("slot_aware_query_enabled=True 时 slot_spec_vocab_size 必须 > 0")
        if self.pointer_head_enabled:
            if self.num_action_heads <= 0:
                raise ValueError("pointer_head_enabled=True 时 num_action_heads 必须 > 0")
            if self.max_pointer_positions <= 0:
                raise ValueError("pointer_head_enabled=True 时 max_pointer_positions 必须 > 0")
            if self.max_action_slots <= 0:
                raise ValueError("pointer_head_enabled=True 时 max_action_slots 必须 > 0")
        if self.pointer_head_conditioning_enabled and not self.pointer_head_enabled:
            raise ValueError("pointer_head_conditioning_enabled=True 需要 pointer_head_enabled=True")
        if self.pointer_head_conditioning_enabled and not self.pointer_head_centering_enabled:
            # 关闭归零会让 head / source 出现 H 个不可辨识方向，实测 top-1 掉 6.5pp。
            # 这不是「可选调参」，是分解式的数学约束，所以直接拒绝这个组合。
            raise ValueError(
                "pointer_head_conditioning_enabled=True 时 pointer_head_centering_enabled 必须为 True"
            )
        if self.pointer_slot_head_conditioning_enabled and not self.pointer_head_conditioning_enabled:
            raise ValueError(
                "pointer_slot_head_conditioning_enabled=True 需要 pointer_head_conditioning_enabled=True"
            )
        if self.max_action_slots <= 0:
            raise ValueError(f"max_action_slots 必须 > 0，实际 {self.max_action_slots}")

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


if _TORCH_AVAILABLE:

    class EntityCardEncoder(nn.Module):
        """card / deck-card 实体编码器：base_card_id embedding + card stat MLP。"""

        def __init__(self, vocab_size: int, config: EntityModelConfig):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, config.card_embed_dim, padding_idx=0)
            self.stat_proj = nn.Linear(CARD_STAT_DIM, config.card_stat_hidden)
            self.fuse = nn.Linear(config.card_embed_dim + config.card_stat_hidden, config.d_model)

        def forward(self, card_ids: torch.Tensor, card_stats: torch.Tensor) -> torch.Tensor:
            emb = self.embedding(card_ids)
            stat = F.relu(self.stat_proj(card_stats))
            return self.fuse(torch.cat([emb, stat], dim=-1))

    class EntityCrestEncoder(nn.Module):
        """crest 实体编码器：base_card_id embedding + 3d crest stat MLP。"""

        def __init__(self, vocab_size: int, config: EntityModelConfig):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, config.card_embed_dim, padding_idx=0)
            self.stat_proj = nn.Linear(CREST_STAT_DIM, config.card_stat_hidden)
            self.fuse = nn.Linear(config.card_embed_dim + config.card_stat_hidden, config.d_model)

        def forward(self, card_ids: torch.Tensor, crest_stats: torch.Tensor) -> torch.Tensor:
            emb = self.embedding(card_ids)
            stat = F.relu(self.stat_proj(crest_stats))
            return self.fuse(torch.cat([emb, stat], dim=-1))

    class CrossZoneEntityTransformer(nn.Module):
        """Cross-Zone Entity Transformer + value head。

        C4 的 action-to-entity cross-attention policy head 在同一模块内实现，
        便于共享 contextual entity tokens。
        """

        def __init__(self, *, vocab_size: int, state_scalar_width: int, action_scalar_width: int, config: EntityModelConfig):
            super().__init__()
            config.validate()
            self.config = config
            self.vocab_size = vocab_size
            self.state_scalar_width = state_scalar_width
            self.action_scalar_width = action_scalar_width
            d = config.d_model

            # type-specific encoders
            self.card_encoder = EntityCardEncoder(vocab_size, config)
            self.crest_encoder = EntityCrestEncoder(vocab_size, config)

            # [STATE] scalar token 投影
            self.state_scalar_encoder = nn.Sequential(
                nn.Linear(state_scalar_width, config.state_scalar_hidden),
                nn.ReLU(),
                nn.Linear(config.state_scalar_hidden, d),
            )

            # 结构化 embedding（相加融合到 d_model）
            self.type_embedding = nn.Embedding(NUM_ENTITY_TYPES, d, padding_idx=0)
            self.zone_embedding = nn.Embedding(NUM_ZONES, d, padding_idx=0)
            self.owner_embedding = nn.Embedding(NUM_OWNERS, d, padding_idx=0)
            self.visibility_embedding = nn.Embedding(NUM_VISIBILITIES, d, padding_idx=0)
            self.position_embedding = nn.Embedding(MAX_FIELD_POSITIONS, d, padding_idx=0)
            # count 数值投影（deck token 的 count / ratio）
            self.count_proj = nn.Linear(2, d)

            # 卡牌效果文本向量表。用离线 TF-IDF/SVD 等向量初始化（见
            # `tools/build_card_text_embeddings.py`），但**保持可学习**：
            # 初始化只负责给"伤害 1 vs 伤害 2"这类区分一个语义起点，
            # self-play 数据到位后它能自己微调到更适合决策的表示。
            if config.card_text_embedding_enabled:
                self.card_text_embedding = nn.Embedding(
                    vocab_size, config.card_text_embedding_dim, padding_idx=0
                )
                self.card_text_proj = nn.Linear(config.card_text_embedding_dim, d, bias=False)
                with torch.no_grad():
                    # 用 `[I; 0] * 0.5` 初始化：初始时前 min(d, dim) 个输出维度直通文本
                    # 向量，其余维度从 0 开始学。若用随机初始化，离线算好的语义会被立刻
                    # 打散——"文本向量有信号"和"模型能在训练早期用上信号"就变成两件不相干
                    # 的事。0.5 让文本通道与 id/stat 通道同量级但不压过它们。
                    spread = min(d, config.card_text_embedding_dim)
                    weight = torch.zeros(d, config.card_text_embedding_dim)
                    weight[:spread, :spread] = torch.eye(spread) * config.card_text_proj_scale
                    self.card_text_proj.weight.copy_(weight)

            self.input_norm = nn.LayerNorm(d)
            self.input_dropout = nn.Dropout(config.residual_dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=config.num_heads,
                dim_feedforward=config.ffn_dim,
                dropout=config.attention_dropout,
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=config.num_layers, enable_nested_tensor=False
            )

            # value head：来自 contextual [STATE] token
            self.value_head = nn.Sequential(
                nn.Linear(d, config.value_hidden),
                nn.ReLU(),
                nn.Linear(config.value_hidden, 1),
            )

            # ── C4 action query + cross-attention policy ──
            self.action_scalar_encoder = nn.Sequential(
                nn.Linear(action_scalar_width, config.policy_hidden),
                nn.ReLU(),
                nn.Linear(config.policy_hidden, d),
            )
            self.action_source_proj = nn.Linear(config.card_embed_dim + CARD_STAT_DIM, d)
            self.action_target_proj = nn.Linear(config.card_embed_dim + CARD_STAT_DIM, d)
            self.action_fuse = nn.Linear(3 * d, d)
            self.cross_attention = nn.MultiheadAttention(
                d, config.num_heads, dropout=config.attention_dropout, batch_first=True
            )
            self.policy_head = nn.Sequential(
                nn.Linear(3 * d, config.policy_hidden),
                nn.ReLU(),
                nn.Linear(config.policy_hidden, 1),
            )

            # ── 目标槽（pointer 动作表示）──
            # 只有 slot_aware_query_enabled 时才建参数，保证旧 checkpoint 的
            # state_dict 完全不变。
            if config.slot_aware_query_enabled:
                self.slot_spec_embedding = nn.Embedding(
                    config.slot_spec_vocab_size, d, padding_idx=0
                )
                self.slot_position_embedding = nn.Embedding(config.max_action_slots, d)
                self.slot_arity_proj = nn.Linear(1, d)
                self.slot_target_proj = nn.Linear(d, d)
                self.slot_fuse = nn.Linear(3 * d, d)

            # ── 生成式 pointer ──
            # 指向空间是固定位置坐标（entity_position.py）。anchor embedding 多留一行
            # 给「无坐标」（-1），所以 padding_idx 用最后一行而不是 0——0 是合法坐标
            # （self leader），若用 padding_idx=0 它的向量会恒为零。
            if config.pointer_head_enabled:
                self.pointer_anchor_embedding = nn.Embedding(
                    config.max_pointer_positions + 1, d, padding_idx=config.max_pointer_positions
                )
                self.pointer_attn = nn.MultiheadAttention(
                    d, config.num_heads, dropout=config.attention_dropout, batch_first=True
                )
                self.pointer_fuse = nn.Linear(3 * d, d)
                self.pointer_out = nn.Linear(d, 1)
                self.slot_index_embedding = nn.Embedding(config.max_action_slots, d)
                self.head_head = nn.Linear(d, config.num_action_heads)
                # slot 分布的主战者项（slot 暂不做 head 条件化，见 B 阶段）
                self.leader_head = nn.Linear(d, 2)
                if config.pointer_head_conditioning_enabled:
                    # source 分布按动作头分片：``[B,P]`` -> ``[B,H,P]``。
                    # slot 分布仍共享 ``[B,K,P]``——slot 的语义依赖动作本身，
                    # 扩到 ``[B,H,K,P]`` 是下一步。
                    self.source_head_proj = nn.Linear(d, config.num_action_heads)
                    self.leader_head_cond = nn.Linear(d, 2 * config.num_action_heads)
                    if config.pointer_slot_head_conditioning_enabled:
                        self.slot_head_proj = nn.Linear(d, config.num_action_heads)
                        self.slot_leader_head_cond = nn.Linear(
                            d, 2 * config.num_action_heads
                        )

        # ── entity encoding ──
        def encode_entities(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            """返回 (entity_embeddings [B,E,d], key_padding_mask [B,E] True=pad)。"""
            # raw_card_ids 永远保留，供文本向量查表；card_ids 可能被替换成 UNKNOWN。
            raw_card_ids = batch["entity_card_ids"]
            if self.config.use_card_id:
                card_ids = raw_card_ids
            else:
                # UNKNOWN_INDEX = 1（见 CardVocabulary）。全部卡牌都变成"同一种未知卡"，
                # id embedding 提供的身份信息被彻底切断。
                card_ids = torch.ones_like(raw_card_ids)
            card_stats = batch["entity_card_stats"]
            crest_stats = batch["entity_crest_stats"]
            type_ids = batch["entity_type_ids"]
            B, E = card_ids.shape
            d = self.config.d_model

            card_emb = self.card_encoder(card_ids, card_stats)      # [B,E,d]
            crest_emb = self.crest_encoder(card_ids, crest_stats)   # [B,E,d]

            # [STATE] token content 用 scalar 投影
            state_content = self.state_scalar_encoder(batch["state_scalar"])  # [B,d]

            from decision_engine.training.entity_tensorization import (
                ENTITY_TYPE_STATE,
                ENTITY_TYPE_CREST,
            )
            is_state = (type_ids == ENTITY_TYPE_STATE).unsqueeze(-1)   # [B,E,1]
            is_crest = (type_ids == ENTITY_TYPE_CREST).unsqueeze(-1)

            # 选择 type-specific content：state / crest / card(default, 含 deck card)
            content = torch.where(is_crest, crest_emb, card_emb)
            content = torch.where(is_state, state_content.unsqueeze(1).expand(B, E, d), content)

            # 结构化 embedding 相加
            struct = (
                self.type_embedding(type_ids)
                + self.zone_embedding(batch["entity_zone_ids"])
                + self.owner_embedding(batch["entity_owner_ids"])
                + self.visibility_embedding(batch["entity_visibility_ids"])
                + self.position_embedding(batch["entity_position_ids"].clamp(max=MAX_FIELD_POSITIONS - 1))
            )
            count_feat = torch.stack([batch["entity_count"], batch["entity_count_ratio"]], dim=-1)  # [B,E,2]
            struct = struct + self.count_proj(count_feat)

            if self.config.card_text_embedding_enabled:
                # card / crest 实体叠加效果文本向量；[STATE] token 的 card_ids 是 pad(0)，
                # padding_idx=0 的行恒为零向量，不会污染。
                struct = struct + self.card_text_proj(self.card_text_embedding(raw_card_ids))

            if self.config.pointer_head_enabled:
                # 给每个实体叠加它的固定位置坐标，pointer 才能把「第 P 个位置」与
                # 对应实体关联起来（见 entity_position.py）。
                struct = struct + self.pointer_anchor_embedding(
                    self._anchor_index(batch["entity_anchor_ids"])
                )

            x = self.input_norm(content + struct)
            x = self.input_dropout(x)

            key_padding_mask = ~batch["entity_mask"]  # True = padding
            return x, key_padding_mask

        def initialize_card_text_embeddings(self, matrix: torch.Tensor) -> None:
            """用离线文本向量初始化卡牌文本 embedding 表。

            ``matrix`` 形状 ``[vocab_size, card_text_embedding_dim]``，行索引与
            ``CardVocabulary`` 一致。未被覆盖的行保持零初始化（等价于"该卡没有文本向量"）。
            初始化后表仍可学习——这里只给起点，不冻结。
            """
            if not self.config.card_text_embedding_enabled:
                raise ValueError("card_text_embedding_enabled=False，无法初始化卡牌文本 embedding")
            expected = (self.vocab_size, self.config.card_text_embedding_dim)
            if tuple(matrix.shape) != expected:
                raise ValueError(f"卡牌文本 embedding 形状不符: {tuple(matrix.shape)} != {expected}")
            with torch.no_grad():
                self.card_text_embedding.weight.copy_(matrix.to(self.card_text_embedding.weight.dtype))
                padding_index = self.card_text_embedding.padding_idx
                if padding_index is not None:
                    # padding_idx 只保证梯度为 0，copy_ 仍会写入。这里显式清零，
                    # 防止资产里出现 0 号 key 时把无意义向量灌给 [STATE] token。
                    self.card_text_embedding.weight[padding_index].zero_()

        def contextual_tokens(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            x, key_padding_mask = self.encode_entities(batch)
            ctx = self.transformer(x, src_key_padding_mask=key_padding_mask)  # [B,E,d]
            return ctx, key_padding_mask

        def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            ctx, key_padding_mask = self.contextual_tokens(batch)  # [B,E,d]

            # value 来自 contextual [STATE] token（序列首位，index 0，始终有效）
            state_ctx = ctx[:, 0, :]                          # [B,d]
            value = self.value_head(state_ctx).squeeze(-1)  # [B]
            if self.config.value_activation == "tanh":
                value = torch.tanh(value)

            if self.config.pointer_head_enabled:
                outputs = self._pointer_outputs(batch, ctx, key_padding_mask, state_ctx)
                outputs["value"] = value
                return outputs

            # ── C4 policy：action query -> cross-attention over entity tokens ──
            action_query = self._action_query(batch)         # [B,A,d]
            attended, _ = self.cross_attention(
                action_query, ctx, ctx, key_padding_mask=key_padding_mask
            )                                                 # [B,A,d]
            A = action_query.shape[1]
            fused = torch.cat([
                action_query,
                attended,
                state_ctx.unsqueeze(1).expand(-1, A, -1),
            ], dim=-1)                                        # [B,A,3d]
            policy_logits = self.policy_head(fused).squeeze(-1)  # [B,A]

            # 非法/padding 动作 mask 为 -inf，softmax 只覆盖合法动作
            action_mask = batch["action_mask"]               # [B,A] bool
            neg_inf = torch.finfo(policy_logits.dtype).min
            masked_logits = policy_logits.masked_fill(~action_mask, neg_inf)
            return {
                "policy_logits": masked_logits,
                "value": value,
            }

        # ── 生成式 pointer ──
        def _anchor_index(self, anchor_ids: torch.Tensor) -> torch.Tensor:
            """把 -1（无坐标）映射到 embedding 的 padding 行（index = max_positions）。"""
            pad = self.config.max_pointer_positions
            return torch.where(anchor_ids < 0, torch.full_like(anchor_ids, pad), anchor_ids)

        def _phase_mask(
            self,
            phase_ids: torch.Tensor,
            table: Mapping[str, tuple[int, ...]],
            num_positions: int,
        ) -> torch.Tensor:
            """按阶段取位置 mask，返回 ``[B,P]``。

            ``phase_ids == -1``（旧数据无阶段）时**全放行**——不能退化成全 mask，
            否则旧数据会没有任何可选的 source。
            """
            rows = []
            for name in PHASE_NAMES:
                allowed = set(table.get(name, ()))
                rows.append(torch.tensor(
                    [index in allowed for index in range(num_positions)], device=phase_ids.device
                ))
            matrix = torch.stack(rows)                                   # [NUM_PHASES, P]
            safe = phase_ids.clamp(min=0)
            selected = matrix[safe]                                      # [B,P]
            has_phase = (phase_ids >= 0).unsqueeze(-1)
            return torch.where(has_phase, selected, torch.ones_like(selected))

        def _pointer_outputs(
            self,
            batch: Mapping[str, torch.Tensor],
            ctx: torch.Tensor,
            key_padding_mask: torch.Tensor,
            state_ctx: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            """分开输出 head / source / slot 分布，而不是对候选动作打分。

            ``source_logits`` 与 ``slot_logits`` 都是对**固定位置坐标**的分布，
            宽度与候选动作数无关——这正是它能扩展到序列输出的原因。
            """
            B, E, d = ctx.shape
            P = self.config.max_pointer_positions
            K = self.config.max_action_slots
            device = ctx.device
            neg_inf = torch.finfo(ctx.dtype).min

            anchor_ids = batch["entity_anchor_ids"]                     # [B,E]（-1 = 无坐标）
            valid = anchor_ids >= 0
            anchors = self._anchor_index(anchor_ids).clamp(max=P - 1)

            anchor_query = self.pointer_anchor_embedding(self._anchor_index(anchor_ids))  # [B,E,d]
            attended, _ = self.pointer_attn(
                anchor_query, ctx, ctx, key_padding_mask=key_padding_mask
            )                                                           # [B,E,d]
            fused = self.pointer_fuse(torch.cat([
                anchor_query,
                attended,
                state_ctx.unsqueeze(1).expand(-1, E, -1),
            ], dim=-1))                                                 # [B,E,d]

            # ── source 分布 ──
            # 条件化时按动作头出 ``[B,H,P]``：每个候选动作取自己 head 的那一片，
            # 训练与折算严格一致（不需要教师强制）。
            conditioning = self.config.pointer_head_conditioning_enabled
            H = self.config.num_action_heads
            if conditioning:
                entity_logits_h = self.source_head_proj(fused)          # [B,E,H]
                masked_h = torch.where(
                    valid.unsqueeze(-1),
                    entity_logits_h,
                    torch.full_like(entity_logits_h, neg_inf),
                )
                source_logits = torch.full((B, H, P), neg_inf, device=device, dtype=ctx.dtype)
                source_logits.scatter_(
                    2,
                    anchors.unsqueeze(1).expand(-1, H, -1),
                    masked_h.transpose(1, 2),
                )
            else:
                entity_logits = self.pointer_out(fused).squeeze(-1)     # [B,E]
                masked_entity = torch.where(
                    valid, entity_logits, torch.full_like(entity_logits, neg_inf)
                )
                source_logits = torch.full((B, P), neg_inf, device=device, dtype=ctx.dtype)
                source_logits.scatter_(1, anchors, masked_entity)

            # ── slot 分布 ──
            # 每个槽加一个可学偏移，让「第几个槽」影响打分。
            # 按 head 条件化时输出 ``[B,H,K,P]``：槽的语义依赖动作本身（同一槽号在
            # ATTACK 下是 attack_follower、在 PLAY 下是那张卡的效果 spec），
            # head 是它的主要决定因素。
            slot_cond = self.config.pointer_slot_head_conditioning_enabled
            slot_bias = self.slot_index_embedding(torch.arange(K, device=device))  # [K,d]
            slot_input = fused.unsqueeze(1) + slot_bias.view(1, K, 1, d)           # [B,K,E,d]
            if slot_cond:
                slot_entity_logits = self.slot_head_proj(slot_input).permute(0, 3, 1, 2)  # [B,H,K,E]
                slot_entity_valid = valid.view(B, 1, 1, E).expand(-1, H, -1, -1)         # [B,H,K,E]
                masked_slot = torch.where(
                    slot_entity_valid,
                    slot_entity_logits,
                    torch.full_like(slot_entity_logits, neg_inf),
                )
                slot_logits = torch.full((B, H, K, P), neg_inf, device=device, dtype=ctx.dtype)
                slot_logits.scatter_(
                    3, anchors.view(B, 1, 1, E).expand(-1, H, K, -1), masked_slot
                )
            else:
                slot_entity_logits = self.pointer_out(slot_input).squeeze(-1)          # [B,K,E]
                masked_slot = torch.where(
                    valid.unsqueeze(1),
                    slot_entity_logits,
                    torch.full_like(slot_entity_logits, neg_inf),
                )
                slot_logits = torch.full((B, K, P), neg_inf, device=device, dtype=ctx.dtype)
                slot_logits.scatter_(2, anchors.unsqueeze(1).expand(-1, K, -1), masked_slot)

            # 主战者没有实体 token，用 state_ctx 直接出 logits
            if slot_cond:
                # 独立的 leader 头：复用 source 的会重新把两项绑在一起，又制造退化
                slot_leader_h = self.slot_leader_head_cond(state_ctx).view(B, H, 2)   # [B,H,2]
                slot_logits[:, :, :, ENTITY_POSITION_SELF_LEADER] = slot_leader_h[:, :, 0].unsqueeze(-1)
                slot_logits[:, :, :, ENTITY_POSITION_OPP_LEADER] = slot_leader_h[:, :, 1].unsqueeze(-1)
            else:
                slot_leader = self.leader_head(state_ctx)                  # [B,2]
                slot_logits[:, :, ENTITY_POSITION_SELF_LEADER] = slot_leader[:, 0].unsqueeze(1)
                slot_logits[:, :, ENTITY_POSITION_OPP_LEADER] = slot_leader[:, 1].unsqueeze(1)
            if conditioning:
                leader_h = self.leader_head_cond(state_ctx).view(B, H, 2)  # [B,H,2]
                source_logits[:, :, ENTITY_POSITION_SELF_LEADER] = leader_h[:, :, 0]
                source_logits[:, :, ENTITY_POSITION_OPP_LEADER] = leader_h[:, :, 1]
            else:
                source_logits[:, ENTITY_POSITION_SELF_LEADER] = slot_leader[:, 0]
                source_logits[:, ENTITY_POSITION_OPP_LEADER] = slot_leader[:, 1]

            position_mask = batch["position_mask"]                      # [B,P]
            source_mask = position_mask
            slot_position_mask = position_mask
            if self.config.pointer_phase_mask_enabled and "phase_ids" in batch:
                phase = batch["phase_ids"]                              # [B]
                source_mask = position_mask & self._phase_mask(phase, PHASE_SOURCE_POSITIONS, P)
                slot_position_mask = position_mask & self._phase_mask(phase, PHASE_TARGET_POSITIONS, P)
            if conditioning:
                source_logits = source_logits.masked_fill(~source_mask.unsqueeze(1), neg_inf)
                if self.config.pointer_head_centering_enabled:
                    # 消解 head / source 的 per-head 常数偏移退化（见 config 注释）
                    weight = source_mask.to(ctx.dtype).unsqueeze(1)          # [B,1,P]
                    denom = weight.sum(-1, keepdim=True).clamp(min=1.0)      # [B,1,1]
                    masked_vals = torch.where(
                        source_mask.unsqueeze(1), source_logits, torch.zeros_like(source_logits)
                    )
                    avg = masked_vals.sum(-1, keepdim=True) / denom          # [B,H,1]
                    source_logits = (source_logits - avg).masked_fill(
                        ~source_mask.unsqueeze(1), neg_inf
                    )
            else:
                source_logits = source_logits.masked_fill(~source_mask, neg_inf)
            # 第 k 个槽是否存在：候选动作里只要有一个用到第 k 槽就算存在
            slot_exists = batch["action_slot_mask"].any(dim=1)          # [B,K]
            if slot_cond:
                slot_valid = (
                    slot_position_mask.view(B, 1, 1, P) & slot_exists.view(B, 1, K, 1)
                )                                                        # [B,1,K,P]
                slot_logits = slot_logits.masked_fill(~slot_valid, neg_inf)
                # per-(head, slot) 零均值：``head[h] + source[h][s] + Σ slot[h,k,s]``
                # 里「slot[h,k] 全体加常数」可被 head[h] 吸收，有 H*K 个不可辨识方向。
                weight = slot_position_mask.view(B, 1, 1, P).to(ctx.dtype)
                denom = weight.sum(-1, keepdim=True).clamp(min=1.0)
                masked_vals = torch.where(
                    slot_valid, slot_logits, torch.zeros_like(slot_logits)
                )
                avg = masked_vals.sum(-1, keepdim=True) / denom          # [B,H,K,1]
                slot_logits = (slot_logits - avg).masked_fill(~slot_valid, neg_inf)
            else:
                slot_valid = slot_position_mask.unsqueeze(1) & slot_exists.unsqueeze(-1)
                slot_logits = slot_logits.masked_fill(~slot_valid, neg_inf)

            return {
                "head_logits": self.head_head(state_ctx),               # [B,H]
                "source_logits": source_logits,                         # [B,P] 或 [B,H,P]
                "slot_logits": slot_logits,                             # [B,K,P]
                "slot_exists": slot_exists,                             # [B,K]
            }

        def _action_query(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
            scalar = self.action_scalar_encoder(batch["action_scalars"])  # [B,A,d]
            src_emb = self.card_encoder.embedding(batch["action_source_ids"])   # [B,A,card_embed]
            src = self.action_source_proj(torch.cat([src_emb, batch["action_source_stats"]], dim=-1))  # [B,A,d]
            tgt_emb = self.card_encoder.embedding(batch["action_target_ids"])   # [B,A,T,card_embed]
            tgt_in = torch.cat([tgt_emb, batch["action_target_stats"]], dim=-1)
            tgt = self.action_target_proj(tgt_in)                               # [B,A,T,d]
            tmask = batch["action_target_mask"]                                 # [B,A,T]
            legacy_pool = self._masked_target_pool(tgt, tmask)                  # [B,A,d]
            if self.config.slot_aware_query_enabled:
                tgt_pool = self._slot_aware_target_pool(batch, tgt, tmask, legacy_pool)
            else:
                tgt_pool = legacy_pool
            return self.action_fuse(torch.cat([scalar, src, tgt_pool], dim=-1))

        @staticmethod
        def _masked_target_pool(tgt: torch.Tensor, tmask: torch.Tensor) -> torch.Tensor:
            """把所有 target 平均池化成一个向量（旧行为，无槽结构）。"""
            weights = tmask.unsqueeze(-1).float()
            return (tgt * weights).sum(dim=2) / weights.sum(dim=2).clamp(min=1.0)

        def _slot_aware_target_pool(
            self,
            batch: Mapping[str, torch.Tensor],
            tgt: torch.Tensor,
            tmask: torch.Tensor,
            legacy_pool: torch.Tensor,
        ) -> torch.Tensor:
            """按目标槽分组池化，让 query 知道「几次选择、每次选了什么语义」。

            没有槽信息时（旧训练数据 / replay 来源的 ``LegalAction``）退回
            ``legacy_pool``，而不是让 query 变成零向量。
            """
            slot_index = batch.get("action_target_slot_index")
            slot_spec_ids = batch.get("action_slot_spec_ids")
            slot_mask = batch.get("action_slot_mask")
            if slot_index is None or slot_spec_ids is None or slot_mask is None:
                return legacy_pool
            num_slots = slot_spec_ids.shape[-1]
            slot_range = torch.arange(num_slots, device=tgt.device).view(1, 1, 1, num_slots)
            # target -> slot 归属；slot_index=-1（pad / 无边界信息）与任何 slot 都不相等
            assign = (slot_index.unsqueeze(-1) == slot_range) & tmask.unsqueeze(-1)  # [B,A,T,K]
            weights = assign.float()
            counts = weights.sum(dim=2)                                              # [B,A,K]
            pooled = torch.einsum("batk,batd->bakd", weights, tgt) / counts.clamp(min=1.0).unsqueeze(-1)

            spec_emb = self.slot_spec_embedding(slot_spec_ids)                       # [B,A,K,d]
            arity_emb = self.slot_arity_proj(counts.unsqueeze(-1))                   # [B,A,K,d]
            target_emb = self.slot_target_proj(pooled)                               # [B,A,K,d]
            slot_token = self.slot_fuse(torch.cat([spec_emb, arity_emb, target_emb], dim=-1))
            positions = torch.arange(num_slots, device=tgt.device)
            slot_token = F.relu(slot_token + self.slot_position_embedding(positions))

            slot_weights = slot_mask.unsqueeze(-1).float()                           # [B,A,K,1]
            slot_pool = (slot_token * slot_weights).sum(dim=2) / slot_weights.sum(dim=2).clamp(min=1.0)
            has_slots = slot_mask.any(dim=-1, keepdim=True)                          # [B,A,1]
            return torch.where(has_slots, slot_pool, legacy_pool)
