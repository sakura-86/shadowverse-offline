# -*- coding: utf-8 -*-
"""UnifiedCardTemplate JSON 加载工具。

支持 pickle 缓存：首次加载后将构造好的 CardTemplateLibrary 对象序列化为
pickle 文件，后续直接反序列化跳过 _from_dict 反射开销（从 ~25s 降至 <1s）。
缓存基于源 JSON 文件内容的 MD5 hash 校验，源文件变化时自动重建。
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .templates import UnifiedCardTemplate

logger = logging.getLogger(__name__)

_memory_cache: dict[str, CardTemplateLibrary] = {}


@dataclass(frozen=True)
class CardTemplateLibrary:
    templates: tuple[UnifiedCardTemplate, ...]
    effect_specs_source: str | None = None


def load_card_template_library(path: str | Path) -> CardTemplateLibrary:
    """加载卡牌模板库，优先使用 pickle 缓存。"""
    path = Path(path)
    cache_key = str(path.resolve())
    if cache_key in _memory_cache:
        return _memory_cache[cache_key]

    source_bytes = path.read_bytes()
    source_hash = hashlib.md5(source_bytes).hexdigest()
    pickle_path = path.with_suffix(".pickle")
    library = _try_load_pickle_cache(pickle_path, source_hash)

    if library is None:
        start = time.perf_counter()
        payload = json.loads(source_bytes.decode("utf-8"))
        raw_templates = _raw_templates(payload, path)
        library = CardTemplateLibrary(
            templates=tuple(UnifiedCardTemplate.from_dict(item) for item in raw_templates),
            effect_specs_source=str(payload.get("effect_specs_source")) if isinstance(payload, dict) and payload.get("effect_specs_source") is not None else None,
        )
        logger.debug("card_template_library loaded from JSON in %.2fs: %s", time.perf_counter() - start, path)
        _write_pickle_cache(pickle_path, library, source_hash)

    _memory_cache[cache_key] = library
    return library


def load_card_templates(path: str | Path) -> tuple[UnifiedCardTemplate, ...]:
    return load_card_template_library(path).templates


def load_card_templates_by_base_card_id(path: str | Path) -> dict[int, UnifiedCardTemplate]:
    return templates_by_base_card_id(load_card_templates(path))


def templates_by_base_card_id(templates: tuple[UnifiedCardTemplate, ...]) -> dict[int, UnifiedCardTemplate]:
    result: dict[int, UnifiedCardTemplate] = {}
    duplicates: list[int] = []
    for template in templates:
        base_card_id = template.identity.base_card_id
        if base_card_id in result:
            duplicates.append(base_card_id)
            continue
        result[base_card_id] = template
    if duplicates:
        raise ValueError(f"duplicate card templates: {sorted(duplicates)}")
    return result


def invalidate_template_cache() -> None:
    """清除进程级内存缓存（测试用）。"""
    _memory_cache.clear()


_PICKLE_PROTOCOL = 5


def _try_load_pickle_cache(pickle_path: Path, source_hash: str) -> CardTemplateLibrary | None:
    """尝试从 pickle 缓存加载，hash 不匹配或加载失败返回 None。"""
    if not pickle_path.exists():
        return None
    try:
        start = time.perf_counter()
        with pickle_path.open("rb") as f:
            cached = pickle.load(f)
        if not isinstance(cached, dict) or cached.get("hash") != source_hash:
            logger.debug("pickle cache hash mismatch, will rebuild: %s", pickle_path)
            return None
        library = cached["library"]
        if not isinstance(library, CardTemplateLibrary):
            logger.debug("pickle cache contains unexpected type, will rebuild: %s", pickle_path)
            return None
        logger.debug("card_template_library loaded from pickle cache in %.3fs: %s", time.perf_counter() - start, pickle_path)
        return library
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, ModuleNotFoundError) as exc:
        logger.debug("pickle cache load failed (%s), will rebuild: %s", exc, pickle_path)
        return None


def _write_pickle_cache(pickle_path: Path, library: CardTemplateLibrary, source_hash: str) -> None:
    """将 CardTemplateLibrary 写入 pickle 缓存文件。"""
    try:
        start = time.perf_counter()
        data = {"hash": source_hash, "library": library}
        tmp_path = pickle_path.with_suffix(".pickle.tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(data, f, protocol=_PICKLE_PROTOCOL)
        tmp_path.replace(pickle_path)
        logger.debug("pickle cache written in %.3fs: %s", time.perf_counter() - start, pickle_path)
    except OSError as exc:
        logger.warning("failed to write pickle cache: %s: %s", pickle_path, exc)


def _raw_templates(payload: Any, path: str | Path) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw = payload.get("templates", payload.get("cards"))
    else:
        raw = payload
    if not isinstance(raw, list):
        raise ValueError(f"unsupported card template format: {path}")
    if not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"card template list contains non-object entries: {path}")
    return raw
