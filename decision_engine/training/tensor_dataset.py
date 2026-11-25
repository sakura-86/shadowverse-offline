# -*- coding: utf-8 -*-
"""Tensorized training dataset loading and summary helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

TENSOR_DATASET_SCHEMA = "PolicyValueTensorDataset"
TENSOR_DATASET_VERSION = 1


def load_tensor_dataset_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"tensorized JSONL line {line_number} must be an object")
        rows.append(validate_tensorized_example(payload))
    return rows


def load_tensor_dataset_dir(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {name: load_tensor_dataset_jsonl(input_dir / f"{name}.jsonl") for name in ("train", "val", "test")}


def validate_tensorized_example(example: Mapping[str, Any]) -> dict[str, Any]:
    if example.get("schema") != "TrainingExampleTensorization":
        raise ValueError("tensorized example schema must be TrainingExampleTensorization")
    if example.get("version") != 1:
        raise ValueError("tensorized example version must be 1")
    sample_id = example.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("tensorized example sample_id must be a non-empty string")
    state_vector = _sequence(example.get("state_vector"))
    action_vectors = _sequence(example.get("action_vectors"))
    policy_target = _sequence(example.get("policy_target"))
    if not state_vector:
        raise ValueError("tensorized example requires state_vector")
    if not action_vectors:
        raise ValueError("tensorized example requires action_vectors")
    if len(policy_target) != len(action_vectors):
        raise ValueError("tensorized example policy_target width must match action_vectors length")
    labels = _mapping(example.get("labels"))
    if labels.get("policy_target_present") is not True:
        raise ValueError("tensorized example must preserve policy_target_present=true")
    if labels.get("value_target_present") not in (True, False):
        raise ValueError("tensorized example must preserve value_target_present as bool")
    _validate_numeric_sequence(state_vector, path="state_vector")
    for index, vector in enumerate(action_vectors):
        _validate_numeric_sequence(_sequence(vector), path=f"action_vectors[{index}]")
    _validate_numeric_sequence(policy_target, path="policy_target")
    if example.get("value_target") is not None and not isinstance(example.get("value_target"), int | float):
        raise ValueError("tensorized example value_target must be numeric or null")
    return dict(example)


def summarize_tensor_dataset(dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    split_summaries = {name: _summarize_split(items) for name, items in dataset.items()}
    return {
        "schema": TENSOR_DATASET_SCHEMA,
        "version": TENSOR_DATASET_VERSION,
        "splits": split_summaries,
        "total_samples": sum(summary["count"] for summary in split_summaries.values()),
    }


def flatten_tensor_dataset(dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, items in dataset.items():
        for item in items:
            payload = dict(item)
            payload["split"] = split_name
            rows.append(payload)
    return rows


def _summarize_split(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    policy_widths = Counter(str(len(_sequence(item.get("policy_target")))) for item in items)
    value_present = sum(1 for item in items if item.get("value_target") is not None)
    value_missing = len(items) - value_present
    state_widths = Counter(str(len(_sequence(item.get("state_vector")))) for item in items)
    action_widths = Counter(str(len(_sequence(item.get("action_vectors")))) for item in items)
    return {
        "count": len(items),
        "policy_target_width_counts": dict(sorted(policy_widths.items(), key=lambda item: int(item[0]))),
        "state_vector_width_counts": dict(sorted(state_widths.items(), key=lambda item: int(item[0]))),
        "action_vector_count_counts": dict(sorted(action_widths.items(), key=lambda item: int(item[0]))),
        "value_target_present_count": value_present,
        "value_target_missing_count": value_missing,
    }


def _validate_numeric_sequence(values: Sequence[Any], *, path: str) -> None:
    for index, value in enumerate(values):
        if not isinstance(value, int | float):
            raise ValueError(f"tensorized example {path}[{index}] must be numeric")


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
