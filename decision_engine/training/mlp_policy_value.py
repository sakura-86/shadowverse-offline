# -*- coding: utf-8 -*-
"""PyTorch MLP baseline for policy/value training."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

from decision_engine.training.model_protocol import PolicyValueModelArtifact
from decision_engine.training.tensor_dataset import flatten_tensor_dataset, summarize_tensor_dataset

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class MlpPolicyValueConfig:
    state_hidden: int = 256
    action_hidden: int = 128
    joint_hidden: int = 256
    value_hidden: int = 128
    seed: int = 1
    value_loss_weight: float = 1.0


@dataclass(frozen=True)
class MlpPolicyValueTrainingResult:
    report: Mapping[str, Any]
    train_metrics: Mapping[str, Any]
    eval_report: Mapping[str, Any]
    model_artifact: PolicyValueModelArtifact
    checkpoint_path: Path


if nn is not None:
    class MlpPolicyValueNet(nn.Module):
        def __init__(self, *, state_width: int, action_width: int, config: MlpPolicyValueConfig):
            super().__init__()
            self.state_encoder = nn.Sequential(
                nn.Linear(state_width, config.state_hidden),
                nn.ReLU(),
                nn.Linear(config.state_hidden, config.state_hidden),
                nn.ReLU(),
            )
            self.action_encoder = nn.Sequential(
                nn.Linear(action_width, config.action_hidden),
                nn.ReLU(),
                nn.Linear(config.action_hidden, config.action_hidden),
                nn.ReLU(),
            )
            self.policy_head = nn.Sequential(
                nn.Linear(config.state_hidden + config.action_hidden, config.joint_hidden),
                nn.ReLU(),
                nn.Linear(config.joint_hidden, 1),
            )
            self.value_head = nn.Sequential(
                nn.Linear(config.state_hidden, config.value_hidden),
                nn.ReLU(),
                nn.Linear(config.value_hidden, 1),
                nn.Tanh(),
            )

        def forward(self, state_batch, action_batch, action_mask):
            state_emb = self.state_encoder(state_batch)
            action_emb = self.action_encoder(action_batch)
            expanded_state = state_emb.unsqueeze(1).expand(-1, action_batch.shape[1], -1)
            logits = self.policy_head(torch.cat((expanded_state, action_emb), dim=-1)).squeeze(-1)
            logits = logits.masked_fill(~action_mask, -1.0e9)
            value = self.value_head(state_emb).squeeze(-1)
            return logits, value
else:
    class MlpPolicyValueNet:  # pragma: no cover
        pass


def train_mlp_policy_value_model(
    *,
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    run_id: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    config: MlpPolicyValueConfig | None = None,
) -> MlpPolicyValueTrainingResult:
    if torch is None or nn is None or F is None:
        raise RuntimeError(f"PyTorch is required for mlp_policy_value backend: {_TORCH_IMPORT_ERROR}")
    cfg = config or MlpPolicyValueConfig()
    _validate_training_args(run_id=run_id, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = list(dataset.get("train", ()))
    val_rows = list(dataset.get("val", ()))
    test_rows = list(dataset.get("test", ()))
    if not train_rows:
        raise ValueError("mlp_policy_value training requires at least one train sample")
    state_width, action_width = _infer_widths(train_rows + val_rows + test_rows)
    torch.manual_seed(cfg.seed)
    model = MlpPolicyValueNet(state_width=state_width, action_width=action_width, config=cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_reports: list[dict[str, Any]] = []
    rng = Random(cfg.seed)
    for epoch in range(1, max(epochs, 1) + 1):
        rng.shuffle(train_rows)
        model.train()
        batch_reports: list[dict[str, float]] = []
        for batch_rows in _batches(train_rows, batch_size):
            batch = _collate(batch_rows, device=device)
            optimizer.zero_grad()
            logits, values = model(batch["state"], batch["actions"], batch["mask"])
            losses = _losses(logits, values, batch, value_loss_weight=cfg.value_loss_weight)
            losses["total_loss"].backward()
            optimizer.step()
            batch_reports.append({key: float(value.detach().cpu().item()) for key, value in losses.items()})
        train_eval = _evaluate(model, train_rows, batch_size=batch_size, device=device, value_loss_weight=cfg.value_loss_weight)
        val_eval = _evaluate(model, val_rows, batch_size=batch_size, device=device, value_loss_weight=cfg.value_loss_weight) if val_rows else {}
        epoch_reports.append({
            "epoch": epoch,
            "batch_count": len(batch_reports),
            "sample_count": len(train_rows),
            "train": train_eval,
            "val": val_eval,
        })
    final_train = epoch_reports[-1]["train"] if epoch_reports else {}
    final_val = epoch_reports[-1]["val"] if epoch_reports and epoch_reports[-1]["val"] else {}
    final_metrics = final_val or final_train
    test_eval = _evaluate(model, test_rows, batch_size=batch_size, device=device, value_loss_weight=cfg.value_loss_weight) if test_rows else {}
    checkpoint_path = output_dir / "model.pt"
    torch.save({
        "schema": "MlpPolicyValueCheckpoint",
        "version": 1,
        "run_id": run_id,
        "backend": "torch_mlp",
        "state_vector_width": state_width,
        "action_vector_width": action_width,
        "config": cfg.__dict__,
        "model_state_dict": model.state_dict(),
    }, checkpoint_path)
    summary = summarize_tensor_dataset(dataset)
    rows = flatten_tensor_dataset(dataset)
    artifact = PolicyValueModelArtifact(
        run_id=run_id,
        backend="torch_mlp",
        model_kind="policy_value_dual_head_mlp",
        state_vector_width=state_width,
        action_vector_width=action_width,
        policy_head_width=action_width,
        metadata={
            "checkpoint": checkpoint_path.name,
            "tensor_dataset_schema": summary.get("schema"),
            "architecture": "state_mlp_action_mlp_per_action_policy_value_head",
            "state_hidden": cfg.state_hidden,
            "action_hidden": cfg.action_hidden,
            "joint_hidden": cfg.joint_hidden,
            "value_hidden": cfg.value_hidden,
            "value_loss_weight": cfg.value_loss_weight,
            "seed": cfg.seed,
        },
    )
    report = {
        "schema": "PolicyValueTrainingLoopReport",
        "version": 1,
        "run_id": run_id,
        "backend": "torch_mlp",
        "output_dir": str(output_dir),
        "tensor_dataset_summary": summary,
        "epoch_reports": epoch_reports,
        "final_metrics": final_metrics,
        "tensorized_samples_total": len(rows),
        "checkpoint_path": str(checkpoint_path),
        "non_goals": ["not_runtime_change", "not_action_legality", "not_state_transition"],
    }
    train_metrics = {
        "schema": "PolicyValueTrainingLoopMetrics",
        "version": 1,
        "run_id": run_id,
        "status": "completed",
        "backend": "torch_mlp",
        "epochs": epoch_reports,
        "policy_loss": final_metrics.get("policy_loss"),
        "value_loss": final_metrics.get("value_loss"),
        "total_loss": final_metrics.get("total_loss"),
        "sample_count": len(rows),
        "artifact_status": "trained",
        "checkpoint_path": str(checkpoint_path),
    }
    eval_report = {
        "schema": "PolicyValueTrainingLoopEvalReport",
        "version": 1,
        "run_id": run_id,
        "status": "completed" if val_rows or test_rows else "not_run",
        "backend": "torch_mlp",
        "metrics": {
            "val": final_val,
            "test": test_eval,
            "policy_loss_final": final_metrics.get("policy_loss"),
            "value_loss_final": final_metrics.get("value_loss"),
            "total_loss_final": final_metrics.get("total_loss"),
        },
        "artifact_status": "trained",
    }
    return MlpPolicyValueTrainingResult(
        report=report,
        train_metrics=train_metrics,
        eval_report=eval_report,
        model_artifact=artifact,
        checkpoint_path=checkpoint_path,
    )


def _losses(logits, values, batch: Mapping[str, Any], *, value_loss_weight: float) -> dict[str, Any]:
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(batch["policy"] * log_probs).sum(dim=-1).mean()
    value_mask = batch["value_mask"]
    if bool(value_mask.any().item()):
        value_loss = F.mse_loss(values[value_mask], batch["value"][value_mask])
    else:
        value_loss = values.sum() * 0.0
    total_loss = policy_loss + value_loss_weight * value_loss
    return {"policy_loss": policy_loss, "value_loss": value_loss, "total_loss": total_loss}


def _evaluate(model, rows: Sequence[Mapping[str, Any]], *, batch_size: int, device: str, value_loss_weight: float) -> dict[str, float]:
    if not rows:
        return {}
    model.eval()
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
    sample_count = 0
    with torch.no_grad():
        for batch_rows in _batches(list(rows), batch_size):
            batch = _collate(batch_rows, device=device)
            logits, values = model(batch["state"], batch["actions"], batch["mask"])
            losses = _losses(logits, values, batch, value_loss_weight=value_loss_weight)
            count = len(batch_rows)
            sample_count += count
            for key, value in losses.items():
                totals[key] += float(value.detach().cpu().item()) * count
    return {key: totals[key] / sample_count for key in totals} | {"sample_count": float(sample_count)}


def _collate(rows: Sequence[Mapping[str, Any]], *, device: str) -> dict[str, Any]:
    state_width, action_width = _infer_widths(rows)
    max_actions = max(len(_sequence(row.get("action_vectors"))) for row in rows)
    state = torch.zeros((len(rows), state_width), dtype=torch.float32, device=device)
    actions = torch.zeros((len(rows), max_actions, action_width), dtype=torch.float32, device=device)
    mask = torch.zeros((len(rows), max_actions), dtype=torch.bool, device=device)
    policy = torch.zeros((len(rows), max_actions), dtype=torch.float32, device=device)
    value = torch.zeros((len(rows),), dtype=torch.float32, device=device)
    value_mask = torch.zeros((len(rows),), dtype=torch.bool, device=device)
    for row_index, row in enumerate(rows):
        state_values = [float(item) for item in _sequence(row.get("state_vector"))]
        state[row_index, :len(state_values)] = torch.tensor(state_values, dtype=torch.float32, device=device)
        action_vectors = _sequence(row.get("action_vectors"))
        policy_target = [float(item) for item in _sequence(row.get("policy_target"))]
        if len(policy_target) != len(action_vectors):
            raise ValueError("policy_target width must match action_vectors width")
        for action_index, vector in enumerate(action_vectors):
            action_values = [float(item) for item in _sequence(vector)]
            actions[row_index, action_index, :len(action_values)] = torch.tensor(action_values, dtype=torch.float32, device=device)
            mask[row_index, action_index] = True
            policy[row_index, action_index] = policy_target[action_index]
        target = row.get("value_target")
        if target is not None:
            value[row_index] = float(target)
            value_mask[row_index] = True
    return {"state": state, "actions": actions, "mask": mask, "policy": policy, "value": value, "value_mask": value_mask}


def _infer_widths(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    state_widths = {len(_sequence(row.get("state_vector"))) for row in rows}
    action_widths = {len(_sequence(vector)) for row in rows for vector in _sequence(row.get("action_vectors"))}
    if not state_widths or 0 in state_widths:
        raise ValueError("tensor dataset requires non-empty state_vector")
    if not action_widths or 0 in action_widths:
        raise ValueError("tensor dataset requires non-empty action_vectors")
    if len(state_widths) != 1:
        raise ValueError(f"inconsistent state_vector widths: {sorted(state_widths)}")
    if len(action_widths) != 1:
        raise ValueError(f"inconsistent action_vector widths: {sorted(action_widths)}")
    return next(iter(state_widths)), next(iter(action_widths))


def _batches(rows: list[Mapping[str, Any]], batch_size: int) -> list[list[Mapping[str, Any]]]:
    size = max(batch_size, 1)
    return [rows[index:index + size] for index in range(0, len(rows), size)] or []


def _validate_training_args(*, run_id: str, epochs: int, batch_size: int, learning_rate: float) -> None:
    if not run_id:
        raise ValueError("run_id must be non-empty")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []
