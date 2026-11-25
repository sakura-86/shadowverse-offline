#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""磁盘分片行数据集：让训练内存占用与数据规模解耦。

背景：结构化训练样本张量化后每条约 60KB（Python 对象），~68000 条约 4.3GB，
与原始样本同时驻留会达 8GB+，在 16GB 机器上 OOM（RC 3221225477）。

方案：把张量化后的行分片写入磁盘（每片 N 行，jsonl），训练时按需逐片读取，
内存里只保留当前片（或当前 batch）。支持 epoch 间打乱：打乱分片顺序 + 片内打乱。

用法：
    ds = ShardedRowDataset.write(rows_iterable, shard_dir, rows_per_shard=2000)
    # 训练循环
    ds.shuffle(rng)
    for row in ds:            # 惰性逐行产出
        ...
    first = ds.first_row()    # 取一行用于推断特征宽度
    n = len(ds)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from random import Random
from typing import Any, Iterable, Iterator, Mapping


class ShardedRowDataset:
    """磁盘分片存储的行数据集，惰性迭代，内存占用与总量解耦。"""

    def __init__(self, shard_dir: Path, shard_paths: list[Path], length: int, rows_per_shard: int):
        self._shard_dir = shard_dir
        self._shard_paths = shard_paths
        self._length = length
        self._rows_per_shard = rows_per_shard
        # epoch 内打乱状态：分片顺序 + 每个分片是否片内打乱
        self._shard_order = list(range(len(shard_paths)))
        self._shuffle_within = False
        self._rng_seed: int | None = None

    # ------------------------------------------------------------------ #
    @classmethod
    def write(
        cls,
        rows: Iterable[Mapping[str, Any]],
        shard_dir: Path,
        *,
        rows_per_shard: int = 2000,
    ) -> "ShardedRowDataset":
        """把 rows 流式写入分片文件。rows 应为可迭代（消费后即可释放原对象）。"""
        shard_dir = Path(shard_dir)
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)

        shard_paths: list[Path] = []
        length = 0
        buffer: list[str] = []

        def flush() -> None:
            if not buffer:
                return
            idx = len(shard_paths)
            path = shard_dir / f"shard_{idx:05d}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write("\n".join(buffer))
                f.write("\n")
            shard_paths.append(path)
            buffer.clear()

        for row in rows:
            buffer.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            length += 1
            if len(buffer) >= rows_per_shard:
                flush()
        flush()

        return cls(shard_dir, shard_paths, length, rows_per_shard)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._length

    def shuffle(self, rng: Random) -> None:
        """为下一个 epoch 设置打乱：打乱分片读取顺序，并启用片内打乱。"""
        rng.shuffle(self._shard_order)
        self._shuffle_within = True
        self._rng_seed = rng.randint(0, 2**31 - 1)

    def _read_shard(self, shard_idx: int) -> list[dict]:
        path = self._shard_paths[shard_idx]
        out: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def __iter__(self) -> Iterator[dict]:
        within_rng = Random(self._rng_seed) if (self._shuffle_within and self._rng_seed is not None) else None
        for shard_idx in self._shard_order:
            rows = self._read_shard(shard_idx)
            if within_rng is not None:
                within_rng.shuffle(rows)
            yield from rows
            # 显式释放当前片，保证同一时刻只有一片驻留内存
            rows.clear()
            del rows

    def first_row(self) -> dict:
        """读取第一行（用于推断特征宽度），不加载整个数据集。"""
        if not self._shard_paths:
            raise ValueError("empty dataset")
        with self._shard_paths[0].open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    return json.loads(line)
        raise ValueError("empty dataset")

    def cleanup(self) -> None:
        """删除分片目录（训练结束后调用）。"""
        if self._shard_dir.exists():
            shutil.rmtree(self._shard_dir)
