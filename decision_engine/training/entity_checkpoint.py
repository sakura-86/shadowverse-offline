# -*- coding: utf-8 -*-
"""Entity backend 训练、checkpoint、strict load、partial warm-start（方案 C 阶段 C5）。

- 训练循环：policy cross-entropy(soft target) + value MSE(masked)，early stopping，
  deterministic seed。
- checkpoint：写入 backend/architecture/config/vocab digest；strict load 校验
  backend、architecture、architecture_version、vocab_size 一致，否则显式拒绝。
- partial warm-start：仅复制 shape 与语义一致的 card/crest embedding 与 stat encoder，
  Transformer / 结构化 embedding / cross-attention / heads 新初始化，产出 machine-readable
  migration report（copied / new / rejected keys）。
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from decision_engine.training.card_vocabulary import CardVocabulary, save_card_vocabulary
from decision_engine.training.entity_tensorization import (
    ENTITY_TENSORIZATION_SCHEMA,
    ENTITY_TENSORIZATION_VERSION,
    ENTITY_LAYOUT_VERSION,
    collate_entity_rows,
)
from decision_engine.training.tensorization import ACTION_VECTOR_DIM
from decision_engine.training.action_pointer import policy_logits_from_pointer
from decision_engine.training.pointer_loss import pointer_auxiliary_losses

ENTITY_BACKEND = "torch_entity_transformer"
ENTITY_CHECKPOINT_SCHEMA = "EntityPolicyValueCheckpoint"
ENTITY_CHECKPOINT_VERSION = 1

# 仅这些参数前缀在 partial warm-start 时允许从 structured checkpoint 继承
# （shape 与语义一致的共享 embedding / stat encoder）。
_WARM_START_EMBEDDING_KEYS = {
    "card_encoder.embedding.weight": "card_encoder.embedding.weight",
    "crest_embedding.weight": "crest_encoder.embedding.weight",
}


class EntityCheckpointError(ValueError):
    """entity checkpoint 加载时 backend / architecture / vocab 不匹配的显式拒绝。"""


@dataclass(frozen=True)
class EntityTrainingResult:
    checkpoint_path: Path
    train_metrics: Mapping[str, Any]
    artifact_paths: Mapping[str, str]


def _batches(rows: Sequence[Any], batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]


def _infer_widths(row: Any) -> tuple[int, int]:
    if hasattr(row, "state_vector"):
        sv = row.state_vector
        actions = row.action_scalar_vectors
    else:
        sv = row["state_vector"]
        actions = row["action_scalar_vectors"]
    state_width = len(sv)
    action_width = len(actions[0]) if actions else ACTION_VECTOR_DIM
    return state_width, action_width


def _candidate_size_bucket(n_legal: int) -> str:
    """按候选动作数分桶（只用于独立评测的细分诊断）。"""
    if n_legal <= 1:
        return "n1"
    if n_legal == 2:
        return "n2"
    if n_legal <= 5:
        return "n3_5"
    return "n6p"


def _evaluate_model(
    model: Any,
    rows: Sequence[Any],
    *,
    batch_size: int,
    device: str,
    breakdown: bool = False,
) -> dict[str, float]:
    """在一个样本集上算 policy / value 指标。

    为什么除了 loss 还要 ``policy_top1`` / ``policy_top3``
    ----------------------------------------------------
    ``val_policy_loss`` 对**不同数据集不可比**：动作空间大小不同，交叉熵基线
    ``ln(n_legal)`` 就不同（``n_legal`` 中位 5 vs 6 差 0.18）。而 anchor 评测
    （``tools/run_anchor_match.py --candidate policy``）用的正是 policy 的 argmax，
    所以 top-1 accuracy 才是与下游强度直接对应的量。

    为什么 value 要配 ``value_pos_ratio``
    ------------------------------------
    value 是二分类（胜/负）。训练集 93% 为正时，"全押正"也能拿 93% 准确率，
    单看 ``value_sign_acc`` 会把塌缩当成学会了。两个指标必须一起读。

    为什么要单独报 ``*_decisions``
    ----------------------------
    候选动作只有 1 个的样本**没有任何决策可言**：softmax 只有一项，loss 恒为 0，
    top-1 必然命中。它们在 val 里占到 22.7%（121/534），会把 top-1 整体抬高，
    并**稀释架构之间的真实差距**。所以额外报「只在 ``n_legal >= 2`` 上算」的
    top-1 / top-3，架构对比只应读这一组。
    """
    if not rows:
        return {}

    model.eval()
    p_sum = 0.0
    p_n = 0
    v_sq = 0.0
    v_n = 0
    top1 = 0
    top3 = 0
    d_top1 = 0
    d_top3 = 0
    d_n = 0
    bucket_hits: dict[str, list[int]] = {}
    sign_hits = 0
    v_pos = 0
    with torch.no_grad():
        for chunk in _batches(rows, max(batch_size, 256)):
            vb = collate_entity_rows(chunk, device=device)
            vout = model(vb)
            if model.config.pointer_head_enabled:
                logits = policy_logits_from_pointer(
                    head_logits=vout["head_logits"],
                    source_logits=vout["source_logits"],
                    slot_logits=vout["slot_logits"],
                    action_head_indices=vb["action_head_indices"],
                    action_source_positions=vb["action_source_positions"],
                    action_target_slot_index=vb["action_target_slot_index"],
                    action_target_positions=vb["action_target_positions"],
                    action_target_mask=vb["action_target_mask"],
                    action_mask=vb["action_mask"],
                )
            else:
                logits = vout["policy_logits"]
            vlp = F.log_softmax(logits, dim=-1)
            per = -(vb["policy"] * vlp.masked_fill(~vb["action_mask"], 0.0)).sum(dim=-1)
            p_sum += per.sum().item()
            p_n += int(per.shape[0])

            masked = logits.masked_fill(~vb["action_mask"], float("-inf"))
            target = vb["policy"].argmax(dim=-1)
            ranked = masked.topk(min(3, masked.shape[-1]), dim=-1).indices
            hit1 = ranked[:, 0] == target
            hit3 = (ranked == target.unsqueeze(-1)).any(dim=-1)
            top1 += int(hit1.sum().item())
            top3 += int(hit3.sum().item())
            # 只在真正有选择余地的样本上统计（n_legal >= 2）
            has_choice = vb["action_mask"].sum(dim=-1) >= 2
            if bool(has_choice.any()):
                d_top1 += int(hit1[has_choice].sum().item())
                d_top3 += int(hit3[has_choice].sum().item())
                d_n += int(has_choice.sum().item())
            if breakdown:
                sizes = vb["action_mask"].sum(dim=-1).tolist()
                for row_index, n_legal in enumerate(sizes):
                    slot = bucket_hits.setdefault(_candidate_size_bucket(int(n_legal)), [0, 0, 0])
                    slot[0] += int(hit1[row_index].item())
                    slot[1] += int(hit3[row_index].item())
                    slot[2] += 1

            if vb["value_mask"].any():
                vm = vb["value_mask"]
                diff = vout["value"][vm] - vb["value"][vm]
                v_sq += (diff * diff).sum().item()
                v_n += int(vm.sum().item())
                pred_v = vout["value"][vm]
                true_v = vb["value"][vm]
                sign_hits += int(((pred_v >= 0) == (true_v > 0)).sum().item())
                v_pos += int((true_v > 0).sum().item())

    metrics = {
        "samples": p_n,
        "policy_loss": p_sum / max(1, p_n),
        "policy_top1": top1 / max(1, p_n),
        "policy_top3": top3 / max(1, p_n),
        "policy_decisions": d_n,
        "policy_top1_decisions": d_top1 / d_n if d_n else 0.0,
        "policy_top3_decisions": d_top3 / d_n if d_n else 0.0,
        "value_loss": v_sq / v_n if v_n else 0.0,
        "value_sign_acc": sign_hits / v_n if v_n else 0.0,
        "value_pos_ratio": v_pos / v_n if v_n else 0.0,
    }
    if breakdown:
        for name, (hits1, hits3, count) in sorted(bucket_hits.items()):
            metrics[f"policy_count_{name}"] = count
            metrics[f"policy_top1_{name}"] = hits1 / count if count else 0.0
            metrics[f"policy_top3_{name}"] = hits3 / count if count else 0.0
    return metrics


def train_entity_policy_value(
    *,
    dataset: Mapping[str, Sequence[Any]],
    output_dir: Path,
    run_id: str,
    vocabulary: CardVocabulary,
    config: "Any" = None,
    initial_checkpoint: Path | None = None,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    value_learning_rate: "float | None" = None,
    patience: int = 5,
    device: str = "cpu",
    progress: "Any" = None,
    card_text_matrix: "Any" = None,
) -> EntityTrainingResult:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for entity training")
    from decision_engine.training.entity_policy_value import EntityModelConfig, CrossZoneEntityTransformer

    cfg = config or EntityModelConfig()
    cfg.validate()
    torch.manual_seed(cfg.seed)

    train_rows = list(dataset.get("train", ()))
    val_rows = list(dataset.get("val", ()))
    #: 按来源（human / teacher）分开的 val 集。混合 val loss 会把"两个来源的
    #: 动作空间大小不同、value 分布不同"摊平成一个数，看不出模型在哪份数据上落后。
    val_groups = {name: list(rows) for name, rows in (dataset.get("val_groups") or {}).items()}
    if not train_rows:
        raise ValueError("Need at least one training sample")

    state_width, action_width = _infer_widths(train_rows[0])
    model = CrossZoneEntityTransformer(
        vocab_size=vocabulary.vocab_size,
        state_scalar_width=state_width,
        action_scalar_width=action_width,
        config=cfg,
    ).to(device)

    warm_start = _partial_warm_start(
        model=model, initial_checkpoint=initial_checkpoint,
        vocabulary=vocabulary, device=device,
    )
    # 卡牌文本向量必须在 warm-start 之后灌入：`_partial_warm_start` 结尾会
    # `model.load_state_dict(current)`，虽然只改 card/crest embedding，但放在之后
    # 才不会依赖它的实现细节。
    if card_text_matrix is not None:
        model.initialize_card_text_embeddings(
            torch.tensor(card_text_matrix, dtype=torch.float32, device=device)
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if value_learning_rate is not None and value_learning_rate != learning_rate:
        # policy 与 value 的梯度尺度差一个量级（policy 交叉熵 ~1，value MSE ~0.2，
        # 且 value 只有 3 层）。同一个 lr 下要么 policy 学得慢、要么 value 被推爆：
        # 实测 lr=1e-3 时 tanh value 一步锁死在 +1，lr=3e-4 时 value 能降但 policy
        # 收敛变慢。分开给 lr 是让两边都在各自合适的步长上。
        value_prefixes = ("value_head.",)
        value_params = [p for n, p in model.named_parameters() if n.startswith(value_prefixes)]
        other_params = [p for n, p in model.named_parameters() if not n.startswith(value_prefixes)]
        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": learning_rate},
                {"params": value_params, "lr": value_learning_rate},
            ]
        )
    rng = Random(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_reports: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        rng.shuffle(train_rows)
        model.train()
        losses: list[dict[str, float]] = []
        for chunk in _batches(train_rows, batch_size):
            batch = collate_entity_rows(chunk, device=device)
            optimizer.zero_grad()
            out = model(batch)
            if cfg.pointer_head_enabled:
                logits = policy_logits_from_pointer(
                    head_logits=out["head_logits"],
                    source_logits=out["source_logits"],
                    slot_logits=out["slot_logits"],
                    action_head_indices=batch["action_head_indices"],
                    action_source_positions=batch["action_source_positions"],
                    action_target_slot_index=batch["action_target_slot_index"],
                    action_target_positions=batch["action_target_positions"],
                    action_target_mask=batch["action_target_mask"],
                    action_mask=batch["action_mask"],
                )
            else:
                logits = out["policy_logits"]
            values = out["value"]
            log_probs = F.log_softmax(logits, dim=-1)
            # padding 动作 policy target 为 0，log_softmax 处已 mask -inf -> 贡献 0
            policy_loss = -(batch["policy"] * log_probs.masked_fill(~batch["action_mask"], 0.0)).sum(dim=-1).mean()
            value_loss = torch.tensor(0.0, device=device)
            if batch["value_mask"].any():
                value_loss = F.mse_loss(values[batch["value_mask"]], batch["value"][batch["value_mask"]])
            total = policy_loss + cfg.value_loss_weight * value_loss
            aux_values: dict[str, float] = {}
            if cfg.pointer_head_enabled:
                aux = pointer_auxiliary_losses(
                    head_logits=out["head_logits"],
                    source_logits=out["source_logits"],
                    slot_logits=out["slot_logits"],
                    policy=batch["policy"],
                    action_head_indices=batch["action_head_indices"],
                    action_source_positions=batch["action_source_positions"],
                    action_target_slot_index=batch["action_target_slot_index"],
                    action_target_positions=batch["action_target_positions"],
                    action_target_mask=batch["action_target_mask"],
                    action_mask=batch["action_mask"],
                    slot_exists=out["slot_exists"],
                )
                for name, weight in (
                    ("head", cfg.pointer_head_loss_weight),
                    ("source", cfg.pointer_source_loss_weight),
                    ("slot", cfg.pointer_slot_loss_weight),
                ):
                    if weight <= 0 or name not in aux:
                        continue
                    loss_sum, count = aux[name]
                    component = loss_sum / max(count, 1)
                    total = total + weight * component
                    aux_values[f"pointer_{name}"] = float(component.item())
            total.backward()
            optimizer.step()
            losses.append({"policy": policy_loss.item(), "value": value_loss.item(), "total": total.item(), **aux_values})

        avg_policy = sum(d["policy"] for d in losses) / len(losses)
        avg_value = sum(d["value"] for d in losses) / len(losses)
        avg_total = sum(d["total"] for d in losses) / len(losses)
        aux_avg = {
            key: sum(d[key] for d in losses if key in d) / max(sum(1 for d in losses if key in d), 1)
            for key in ("pointer_head", "pointer_source", "pointer_slot")
            if any(key in d for d in losses)
        }

        val_metrics: dict[str, float] = {}
        if val_rows:
            base = _evaluate_model(model, val_rows, batch_size=batch_size, device=device)
            val_metrics = {
                "val_policy_loss": base["policy_loss"],
                "val_value_loss": base["value_loss"],
                "val_policy_top1": base["policy_top1"],
                "val_policy_top3": base["policy_top3"],
                # 架构对比只应读这三个：单动作样本（无决策）会把 top-1 抬高并稀释差距
                "val_policy_decisions": base["policy_decisions"],
                "val_policy_top1_decisions": base["policy_top1_decisions"],
                "val_policy_top3_decisions": base["policy_top3_decisions"],
                "val_value_sign_acc": base["value_sign_acc"],
                "val_value_pos_ratio": base["value_pos_ratio"],
            }
            # 分来源指标：val loss 依赖动作空间大小（交叉熵基线 ln(n_legal) 不同），
            # 跨数据集不可比，所以每个来源单独报 loss / top-1 / value 校准情况。
            for group_name, group_rows in val_groups.items():
                group_metrics = _evaluate_model(model, group_rows, batch_size=batch_size, device=device)
                for key, value in group_metrics.items():
                    val_metrics[f"val_{group_name}_{key}"] = value

        epoch_reports.append({"epoch": epoch, "train_policy_loss": avg_policy, "train_value_loss": avg_value, "train_total_loss": avg_total, **aux_avg, **val_metrics})
        if progress is not None:
            # 长训练需要可观测性：没有 per-epoch 输出时，"慢"和"卡死"无法区分。
            progress(epoch_reports[-1])

        if val_rows:
            val_total = val_metrics["val_policy_loss"] + cfg.value_loss_weight * val_metrics["val_value_loss"]
            if val_total < best_val:
                best_val = val_total
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_card_vocabulary(output_dir / "card_vocabulary.json", vocabulary)
    checkpoint_path = output_dir / "model.pt"
    torch.save({
        "schema": ENTITY_CHECKPOINT_SCHEMA,
        "version": ENTITY_CHECKPOINT_VERSION,
        "run_id": run_id,
        "backend": ENTITY_BACKEND,
        "architecture": cfg.architecture,
        "architecture_version": cfg.architecture_version,
        "vocab_size": vocabulary.vocab_size,
        "card_identity_kind": "base_card_id",
        "card_vocabulary_digest": vocabulary.digest,
        "state_scalar_width": state_width,
        "action_scalar_width": action_width,
        "entity_tensorization_schema": ENTITY_TENSORIZATION_SCHEMA,
        "entity_tensorization_version": ENTITY_TENSORIZATION_VERSION,
        "entity_layout_version": ENTITY_LAYOUT_VERSION,
        "config": cfg.to_jsonable(),
        "model_state_dict": model.state_dict(),
    }, checkpoint_path)

    metrics = {
        "run_id": run_id,
        "backend": ENTITY_BACKEND,
        "epochs": epoch_reports,
        "best_epoch": best_epoch,
        "early_stopped": patience_counter >= patience,
        "vocab_size": vocabulary.vocab_size,
        "card_vocabulary_digest": vocabulary.digest,
        "warm_start": warm_start,
        "state_scalar_width": state_width,
        "action_scalar_width": action_width,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "val_groups": {name: len(rows) for name, rows in val_groups.items()},
    }

    artifact_paths = _write_entity_artifact(
        output_dir=output_dir, run_id=run_id, cfg=cfg, vocabulary=vocabulary,
        state_width=state_width, action_width=action_width, metrics=metrics,
        checkpoint_path=checkpoint_path, warm_start=warm_start,
    )
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return EntityTrainingResult(checkpoint_path=checkpoint_path, train_metrics=metrics, artifact_paths=artifact_paths)


def evaluate_entity_model(
    model: Any,
    rows: Sequence[Any],
    *,
    batch_size: int = 256,
    device: str = "cpu",
    breakdown: bool = True,
) -> dict[str, float]:
    """独立评测入口（不训练）：加载 checkpoint 后单独调用，产出分桶诊断。

    ``tools/evaluate_entity_checkpoints.py`` 用它做多臂对比；分桶是必需的，
    因为候选动作数只有 1 的样本会把 top-1 抬高并稀释架构差距。
    """
    return _evaluate_model(model, rows, batch_size=batch_size, device=device, breakdown=breakdown)


def load_entity_checkpoint(checkpoint_path: Path, *, vocabulary: CardVocabulary, device: str = "cpu") -> Any:
    """strict load entity checkpoint，构建并加载模型。backend/architecture/vocab 不匹配拒绝。"""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required")
    from decision_engine.training.entity_policy_value import EntityModelConfig, CrossZoneEntityTransformer

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt.get("schema") != ENTITY_CHECKPOINT_SCHEMA:
        raise EntityCheckpointError(f"unexpected checkpoint schema: {ckpt.get('schema')!r}")
    if ckpt.get("backend") != ENTITY_BACKEND:
        raise EntityCheckpointError(f"backend mismatch: {ckpt.get('backend')!r} != {ENTITY_BACKEND}")
    if int(ckpt.get("vocab_size") or 0) != vocabulary.vocab_size:
        raise EntityCheckpointError("vocab_size mismatch")
    if ckpt.get("card_vocabulary_digest") != vocabulary.digest:
        raise EntityCheckpointError("card_vocabulary_digest mismatch")

    cfg_fields = EntityModelConfig.__dataclass_fields__
    stored_config = dict(ckpt.get("config") or {})
    # checkpoint 存的是保存时 dataclass 的全部字段。缺字段 = 该 checkpoint 来自更旧的
    # 代码；用当前默认值补齐可能**改变前向行为**（实测：一个新增的默认 True 开关让
    # 旧权重被施加了训练时不存在的零均值化，top-1 从 91.0% 变成 84.6%）。
    # 这里显式告警而不是静默补齐。
    missing_fields = sorted(set(cfg_fields) - set(stored_config))
    if missing_fields:
        warnings.warn(
            f"checkpoint config 缺少字段 {missing_fields}（{checkpoint_path}）；"
            "已用当前默认值补齐，若这些字段影响前向行为，指标会与训练时不一致",
            RuntimeWarning,
            stacklevel=2,
        )
    cfg = EntityModelConfig(**{k: v for k, v in stored_config.items() if k in cfg_fields})
    if cfg.architecture != ckpt.get("architecture") or cfg.architecture_version != ckpt.get("architecture_version"):
        raise EntityCheckpointError("architecture mismatch")

    model = CrossZoneEntityTransformer(
        vocab_size=vocabulary.vocab_size,
        state_scalar_width=int(ckpt["state_scalar_width"]),
        action_scalar_width=int(ckpt["action_scalar_width"]),
        config=cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _partial_warm_start(*, model: Any, initial_checkpoint: Path | None, vocabulary: CardVocabulary, device: str) -> dict[str, Any]:
    """从旧 structured checkpoint 复制 shape/语义一致的共享 embedding，产出 migration report。"""
    if initial_checkpoint is None:
        return {"enabled": False, "reason": "no_initial_checkpoint"}
    if not initial_checkpoint.exists():
        raise FileNotFoundError(f"initial checkpoint not found: {initial_checkpoint}")

    ckpt = torch.load(initial_checkpoint, map_location=device, weights_only=True)
    prev_state = ckpt.get("model_state_dict")
    if not isinstance(prev_state, Mapping):
        raise ValueError("warm-start checkpoint lacks model_state_dict")

    current = model.state_dict()
    copied: list[str] = []
    rejected: list[str] = []
    for src_key, dst_key in _WARM_START_EMBEDDING_KEYS.items():
        if src_key not in prev_state or dst_key not in current:
            rejected.append(src_key)
            continue
        src = prev_state[src_key]
        dst = current[dst_key]
        # append-only：旧 embedding 行数 <= 新行数，列维一致
        if src.shape[1:] != dst.shape[1:] or src.shape[0] > dst.shape[0]:
            rejected.append(src_key)
            continue
        dst[:src.shape[0]].copy_(src)
        copied.append(f"{src_key}->{dst_key}")

    model.load_state_dict(current)
    new_keys = [k for k in current.keys() if k not in {v for v in _WARM_START_EMBEDDING_KEYS.values()}]
    return {
        "enabled": True,
        "initial_checkpoint": str(initial_checkpoint),
        "copied_keys": copied,
        "rejected_keys": rejected,
        "new_key_count": len(new_keys),
        "vocabulary_digest": vocabulary.digest,
    }


def _write_entity_artifact(*, output_dir: Path, run_id: str, cfg: Any, vocabulary: CardVocabulary,
                           state_width: int, action_width: int, metrics: Mapping[str, Any],
                           checkpoint_path: Path, warm_start: Mapping[str, Any]) -> dict[str, str]:
    artifact = {
        "schema": "EntityPolicyValueModelArtifact",
        "version": 1,
        "run_id": run_id,
        "backend": ENTITY_BACKEND,
        "model_backend": ENTITY_BACKEND,
        "model_architecture_version": cfg.architecture_version,
        "architecture": cfg.architecture,
        "feature_schema_version": _feature_schema_version(),
        "tensorization_schema": ENTITY_TENSORIZATION_SCHEMA,
        "tensorization_version": ENTITY_TENSORIZATION_VERSION,
        "entity_layout_version": ENTITY_LAYOUT_VERSION,
        "card_vocabulary_digest": vocabulary.digest,
        "vocab_size": vocabulary.vocab_size,
        "state_vector_width": state_width,
        "action_vector_width": action_width,
        "model_config": cfg.to_jsonable(),
        "deck_feature_contract": {"own_remaining_deck": True, "opponent_remaining_deck": False, "status_required": "available"},
        # 对手手牌是隐藏信息：只暴露张数（state_vector 的 opponent.hand_count），
        # 不产出 hand entity token。旧值 "perfect_information_opponent_hand" 描述的是
        # 修复前的行为，不再是事实。
        "information_mode": "opponent_hand_hidden_count_only",
        "card_text_embedding": {
            "enabled": bool(getattr(cfg, "card_text_embedding_enabled", False)),
            "dim": int(getattr(cfg, "card_text_embedding_dim", 0)),
        },
        "effect_spec_encoder_enabled": cfg.effect_spec_encoder_enabled,
        "warm_start": dict(warm_start),
        "checkpoint": checkpoint_path.name,
    }
    manifest = {
        "schema": "EntityPolicyValueRunManifest",
        "version": 1,
        "run_id": run_id,
        "backend": ENTITY_BACKEND,
        "train_samples": metrics.get("train_samples"),
        "val_samples": metrics.get("val_samples"),
        "best_epoch": metrics.get("best_epoch"),
    }
    artifact_path = output_dir / "model_artifact.json"
    manifest_path = output_dir / "run_manifest.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model_artifact": str(artifact_path), "run_manifest": str(manifest_path), "checkpoint": str(checkpoint_path)}


def _feature_schema_version() -> int:
    from decision_engine.training.feature_schema import TRAINING_FEATURE_SCHEMA_VERSION
    return TRAINING_FEATURE_SCHEMA_VERSION
