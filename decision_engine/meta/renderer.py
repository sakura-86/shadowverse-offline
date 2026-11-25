# -*- coding: utf-8 -*-
"""渲染层：将 MetaReport 渲染为 Markdown 文档。"""
from __future__ import annotations

from .models import ClassReport, DeckEntry, FormatReport, MetaReport, CardFreq


def render(report: MetaReport) -> str:
    lines: list[str] = []
    _header(lines, report)
    _data_source(lines, report)
    _overview(lines, report)
    _standard_detail(lines, report.standard)
    if report.unlimited.deck_count > 0:
        _unlimited_detail(lines, report.unlimited)
    _format_comparison(lines, report)
    _cross_class(lines, report.standard)
    _deck_lists(lines, report)
    _evaluator_advice(lines, report)
    _engineering(lines)
    return "\n".join(lines)


def _header(lines: list[str], r: MetaReport):
    lines.append("# 当前环境分析")
    lines.append("")
    lines.append(f"> 状态：current  ")
    lines.append(f"> 最后核对：{r.generated_at[:10]}  ")
    lines.append("> 适用阶段：Phase 5 Evaluator 设计  ")
    lines.append("> 文档性质：reference  ")
    lines.append(
        f"> 可信度：verified (基于 yokidou_decks.db {r.total_decks} 条真实对战卡组数据，"
        f"{r.data_range[0]} ~ {r.data_range[1]})  "
    )
    lines.append("> 当前入口：已登记于 `docs/reference/README.md`  ")
    lines.append("> 取代关系：无  ")
    lines.append("> 注意：本文档反映当前卡池环境下的实战卡组构筑和使用率。随新弹发布（约每 2 个月）将过时，需重新分析。  ")
    lines.append("> 失效触发器：新弹发布、卡牌平衡调整。")
    lines.append("")


def _data_source(lines: list[str], r: MetaReport):
    lines.append("## 数据源")
    lines.append("")
    lines.append(f"- **yokidou_decks.db**：{r.total_decks} 条高连胜卡组，涵盖 {len(r.standard.classes) or 7} 个职业。")
    lines.append(f"- **数据时间范围**：{r.data_range[0]} ~ {r.data_range[1]}。")
    lines.append(f"- **环境分类**：指定（Standard）{r.standard.deck_count} 条 / 无限（Unlimited）{r.unlimited.deck_count} 条。")
    lines.append('- **分类依据**：comment 中标注“无限”的为无限环境，其余为指定环境。')
    lines.append("- **自动生成**：由 `python -m decision_engine.meta` 生成。")
    lines.append("")
    lines.append("---")
    lines.append("")


def _overview(lines: list[str], r: MetaReport):
    lines.append("## 1. 环境总览")
    lines.append("")
    _format_overview_table(lines, "1.1 指定环境（Standard）", r.standard)
    lines.append("")
    if r.unlimited.deck_count > 0:
        _format_overview_table(lines, "1.2 无限环境（Unlimited）", r.unlimited)
    lines.append("")
    lines.append("---")
    lines.append("")


def _format_overview_table(lines: list[str], title: str, fmt: FormatReport):
    lines.append(f"### {title} — {fmt.deck_count} 副卡组")
    lines.append("")
    lines.append("| 职业 | 卡组数 | 占比 | 最高连胜 | 主力卡组类型 |")
    lines.append("|------|--------|------|----------|-------------|")
    for cr in fmt.classes:
        pct = f"{cr.deck_count / fmt.deck_count * 100:.1f}%" if fmt.deck_count else "0%"
        top_types = " / ".join(
            f"{t}({c}/{cr.deck_count})" for t, c in list(cr.deck_types.items())[:3]
        )
        lines.append(f"| {cr.class_cn} | {cr.deck_count} | {pct} | {cr.top_streak} | {top_types} |")


def _standard_detail(lines: list[str], fmt: FormatReport):
    lines.append("## 2. 指定环境各职业详解")
    lines.append("")
    for i, cr in enumerate(fmt.classes):
        _class_detail(lines, f"2.{i+1}", cr)


def _unlimited_detail(lines: list[str], fmt: FormatReport):
    lines.append("## 3. 无限环境各职业详解")
    lines.append("")
    for i, cr in enumerate(fmt.classes):
        _class_detail(lines, f"3.{i+1}", cr)


def _class_detail(lines: list[str], prefix: str, cr: ClassReport):
    top_type = next(iter(cr.deck_types), "?")
    lines.append(f"### {prefix} {cr.class_cn} — {top_type}（{cr.deck_count} 副）")
    lines.append("")

    # 核心卡（使用率 >= 80%）
    core = [c for c in cr.core_cards if c.rate >= 0.8]
    if core:
        lines.append("**核心卡（使用率 ≥80%）**：")
        lines.append("")
        lines.append("| 卡牌 | 费用 | 使用率 | 平均张数 |")
        lines.append("|------|------|--------|----------|")
        for c in core:
            lines.append(f"| {c.name} | {c.cost} | {c.frequency}/{c.total_decks} ({c.rate:.0%}) | {c.avg_count:.1f} |")
        lines.append("")

    # 高频卡（50%-80%）
    high = [c for c in cr.core_cards if 0.5 <= c.rate < 0.8]
    if high:
        lines.append("**高频卡（50%-80%）**：")
        for c in high:
            lines.append(f"- {c.name}({c.cost}费) {c.frequency}/{c.total_decks}({c.rate:.0%})")
        lines.append("")

    # 卡组类型分布
    if len(cr.deck_types) > 1:
        lines.append("**卡组类型分布**：" + " / ".join(f"{t}:{c}" for t, c in cr.deck_types.items()))
        lines.append("")

    lines.append("---")
    lines.append("")


def _format_comparison(lines: list[str], r: MetaReport):
    if r.unlimited.deck_count == 0:
        return
    lines.append("## 4. 指定 vs 无限环境对比")
    lines.append("")
    lines.append("| 维度 | 指定环境 | 无限环境 |")
    lines.append("|------|---------|---------|")
    lines.append(f"| 数据量 | {r.standard.deck_count} 副 | {r.unlimited.deck_count} 副 |")

    std_top = r.standard.classes[0] if r.standard.classes else None
    unl_top = r.unlimited.classes[0] if r.unlimited.classes else None
    if std_top:
        lines.append(f"| 指定 T1 | {std_top.class_cn}({std_top.deck_count}) | {unl_top.class_cn}({unl_top.deck_count}) |" if unl_top else "")

    lines.append("")
    lines.append("---")
    lines.append("")


def _cross_class(lines: list[str], fmt: FormatReport):
    if not fmt.cross_class_cards:
        return
    lines.append("## 5. 跨职业热门卡牌（指定环境）")
    lines.append("")
    lines.append("| 卡牌 | 费用 | 总出现次数 |")
    lines.append("|------|------|-----------|")
    for c in fmt.cross_class_cards[:15]:
        lines.append(f"| {c.name} | {c.cost} | {c.frequency} |")
    lines.append("")
    lines.append("---")
    lines.append("")


def _deck_lists(lines: list[str], r: MetaReport):
    lines.append("## 6. 代表卡组列表")
    lines.append("")
    lines.append("### 6.1 指定环境 Top 卡组")
    lines.append("")
    for cr in r.standard.classes:
        if cr.top_decks:
            _render_deck(lines, cr.class_cn, cr.top_decks[0])
    lines.append("")

    if r.unlimited.deck_count > 0:
        lines.append("### 6.2 无限环境 Top 卡组")
        lines.append("")
        for cr in r.unlimited.classes:
            if cr.top_decks:
                _render_deck(lines, cr.class_cn, cr.top_decks[0])
    lines.append("")
    lines.append("---")
    lines.append("")


def _render_deck(lines: list[str], cls_cn: str, d: DeckEntry):
    lines.append(f"#### {cls_cn} — {d.deck_type} {d.win_streak}连胜 ({d.player_name}, {d.article_date})")
    lines.append("")
    lines.append("```")
    sorted_cards = sorted(d.cards, key=lambda c: (c.cost, c.name))
    for c in sorted_cards:
        lines.append(f"{c.count}x {c.name} ({c.cost}费)")
    lines.append("```")
    lines.append("")


def _evaluator_advice(lines: list[str], r: MetaReport):
    lines.append("## 7. Evaluator 设计建议")
    lines.append("")
    lines.append("基于环境数据，各职业评估权重应侧重当前活跃体系的核心资源。")
    lines.append("详见 `class-archetypes.md` 中各职业的「当前环境活跃度」标注。")
    lines.append("")
    lines.append("---")
    lines.append("")


def _engineering(lines: list[str]):
    lines.append("## 8. 工程使用原则")
    lines.append("")
    lines.append("本文档可用于：")
    lines.append("- Evaluator 评分权重的职业差异化调参。")
    lines.append("- CardResolver 优先级排序的参考（优先处理环境高频卡）。")
    lines.append("")
    lines.append("本文档不应用于：")
    lines.append("- 硬编码特定卡组的出牌顺序。")
    lines.append("- 在搜索器中写卡名特判。")
    lines.append("")
    lines.append("更新方式：")
    lines.append("```bash")
    lines.append("# 拉取最新数据后重新生成")
    lines.append("scp ubuntu@43.140.225.148:/opt/qqbot/nonebot/data/yokidou_decks.db decision_engine/data/meta/")
    lines.append("python -m decision_engine.meta")
    lines.append("```")
