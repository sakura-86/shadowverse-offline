# -*- coding: utf-8 -*-
"""JSONL helpers for TrainingExample records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from decision_engine.training.training_example import TrainingExample, validate_training_example


def write_training_examples_jsonl(path: Path, examples: Iterable[TrainingExample | Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = example.to_jsonable() if isinstance(example, TrainingExample) else dict(example)
            validate_training_example(payload)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_training_examples_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"training example JSONL line {line_number} must be an object")
        validate_training_example(payload)
        examples.append(payload)
    return examples
