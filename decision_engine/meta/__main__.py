# -*- coding: utf-8 -*-
"""CLI 入口：python -m decision_engine.meta"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .data_source import MetaDataSource
from .analyzer import analyze
from .renderer import render


def main():
    parser = argparse.ArgumentParser(
        description="影之诗当前环境分析：从卡组数据库生成结构化环境报告"
    )
    parser.add_argument(
        "--deck-db", type=str, default=None,
        help="yokidou_decks.db 路径（默认 decision_engine/data/meta/yokidou_decks.db）"
    )
    parser.add_argument(
        "--card-db", type=str, default=None,
        help="sv_cards_cn.db 路径（默认 decision_engine/data/meta/sv_cards_cn.db）"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出 Markdown 路径（默认 docs/reference/current-meta.md）"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="输出 JSON 格式到 stdout（不写 Markdown 文件）"
    )
    args = parser.parse_args()

    # 确定输出路径
    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = Path(args.output) if args.output else project_root / "docs" / "reference" / "current-meta.md"

    # 加载数据
    try:
        source = MetaDataSource(deck_db_path=args.deck_db, card_db_path=args.card_db)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    decks = source.load_all_decks()
    if not decks:
        print("警告: 数据库中没有卡组数据", file=sys.stderr)
        sys.exit(1)

    # 分析
    report = analyze(decks)

    # 输出
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        md = render(report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")
        print(f"环境分析完成:")
        print(f"  数据范围: {report.data_range[0]} ~ {report.data_range[1]}")
        print(f"  总卡组数: {report.total_decks}")
        print(f"  指定环境: {report.standard.deck_count} 副")
        print(f"  无限环境: {report.unlimited.deck_count} 副")
        print(f"  输出文件: {output_path}")


if __name__ == "__main__":
    main()
