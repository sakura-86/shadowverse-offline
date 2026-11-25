# -*- coding: utf-8 -*-
"""动作级效果标签（action effect tags）。

## 为什么需要

`action_features` 目前只描述动作的**元数据**（kind、cost、target 数量），policy 头
吃的是 `[state_emb ; action_emb]`，看不到"打出这张牌会做什么"。schema v9 只补了
`evolve_desc` / `super_evolve_desc` 是否非空两个布尔位，粒度是"有没有额外效果"，
不是"这个效果是什么"。

实测证据（`legacy_svhook_..._0001.json` frame 41，坦忒拉&安古蕾雅局）：v9 特征让
`evolve 玛纳利亚密友` 的 prior 从 0.142 升到 0.264、`evolve 英灵` 从 0.378 降到
0.045，靠的就是"这两个动作里有一个会打 3"这一级别的信息。本模块把这种信息扩展
到全部 29 个效果类别。

## 标签来源

主通道是**效果文本**，因为覆盖 100%（1798 张卡）。`card_effect_specs.json` 只覆盖
334 张卡（19%），且部分 spec 只是骨架（`conditions` / 模式选项分离），不能当完备
ground truth；它用来做一致性校验（见 `tools/build_action_effect_tags.py --report`）。

## 槽位（slot）

一个随从的"打出"和"进化"是两种不同效果，必须分开：

    play_normal / play_enhance / play_accelerate / play_crystal
    evolve / super_evolve / activate

## 与 card 级特征的边界

已经由 `card_stat_vector` 表达的（cost / atk / life / 关键字 / 增幅进度 / 进化效果
存在性）不在这里重复。这里只补"动作会产生什么"。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

# ── 标签定义 ──────────────────────────────────────────────────────────

TAG_GROUPS: Mapping[str, tuple[str, ...]] = {
    "R_resource": (
        "heal", "draw", "to_hand", "pp_recover", "ep_recover", "pp_max_up", "spellboost",
    ),
    "B_board": (
        "summon", "buff", "grant_keyword", "gain_guard", "gain_storm", "gain_rush", "crest",
    ),
    "C_removal": (
        "damage", "damage_leader", "damage_aoe", "destroy", "banish", "bounce", "debuff",
    ),
    "K_cost": (
        "discard", "self_sacrifice",
    ),
    "M_mechanic": (
        "reanimate", "cost_change", "countdown_change",
    ),
    "U_uncertainty": (
        "conditional", "random", "mode_choice",
    ),
}

TAGS: tuple[str, ...] = tuple(tag for group in TAG_GROUPS.values() for tag in group)

SLOTS: tuple[str, ...] = (
    "play_normal", "play_enhance", "play_accelerate", "play_crystal",
    "evolve", "super_evolve", "activate",
)

# 文本段标记 -> 槽位
_SEGMENT_SLOT = {
    "入场曲": "play_normal",
    "爆能强化": "play_enhance",
    "激奏": "play_accelerate",
    "结晶": "play_crystal",
    "进化时": "evolve",
    "超进化时": "super_evolve",
    "启动": "activate",
}

SEGMENT_RE = re.compile(r"【(入场曲|爆能强化_?\d*|激奏_?\d*|结晶_?\d*|进化时|超进化时|启动|谢幕曲)】")
# 条件标记：效果仍属于本动作，只是有前置条件；剥掉标记保留正文
CONDITION_MARK_RE = re.compile(r"【(连击_?\d*|土之秘术_?\d*|觉醒|唤灵_?\d*|奥义|解放奥义|模式)】")
# "本随从进化时，" 与段标记等价
EVOLVE_BODY_RE = re.compile(r"本随从(超)?进化时[，,]")
# "本随从进入战场时，" 是【入场曲】的无标记写法
ENTER_FIELD_RE = re.compile(r"本随从进入战场时[，,]")
# 被动 / 定义性句子：不属于主动动作效果
_DROP_TOKENS = (
    "【谢幕曲】", "【吟唱", "【融合】", "【瞬念召唤】",
    "在牌组中发动", "X起始为", "回合结束时", "被破坏时", "进入战场时",
    "受到伤害", "使用财宝",
)

_TEXT_PATTERNS: Mapping[str, str] = {
    "heal": r"回复[^。]*生命值|回复至上限",
    "draw": r"抽取\s*(?:\d+|X)\s*张",
    "to_hand": r"加入[^。]{0,6}手牌",
    "pp_recover": r"回复[^。，,]{0,4}?(?:\d+|X)\s*点(?:能量点|PP)|(?:能量点|PP)回复至|能量点至上限",
    "ep_recover": r"进化点",
    "pp_max_up": r"(?:能量点|PP)最大值\+1",
    "spellboost": r"魔力增幅",
    "summon": r"召唤",
    "buff": r"\+\d+/\+\d+|\+X/\+\d+|\+X/\+X",
    "grant_keyword": r"获得【(?:守护|疾驰|突进|虹吸|必杀|潜伏|威慑|屏障|灵气|毁灭)",
    "gain_guard": r"获得【守护】",
    "gain_storm": r"获得【疾驰】",
    "gain_rush": r"获得【突进】",
    "crest": r"获得[^。]{0,8}纹章|纹章[^。]{0,4}\+\d",
    "damage": r"造成\s*(?:\d+|X)\s*点伤害|分配\s*(?:\d+|X)\s*点伤害|改为\s*(?:\d+|X)\s*点伤害",
    "destroy": r"破坏",
    "banish": r"(?<!本随从)(?<!本卡牌)消失",
    "bounce": r"返回(?:手牌|牌组)",
    "debuff": r"-\d+/-\d+|生命值变为|失去(?:所有)?(?:能力|效果|【)|无法攻击|使[^。]{0,10}变身",
    "discard": r"(?:舍弃|弃置|丢弃)",
    "self_sacrifice": r"破坏(?:这个|本)随从|破坏本卡牌",
    "reanimate": r"亡者召还|亡者复生",
    "cost_change": r"费用(?:[+-]\d|变为)",
    "conditional": r"如果|若为|若(?:自己|对手|本)|若本",
    "countdown_change": r"倒计数[+-]\d|倒计数变为",
    "random": r"随机",
}

_GOLD = Path(__file__).resolve().parent.parent / "data" / "public" / "card_database.json"


def _deduped_text(card: Mapping[str, Any]) -> str:
    """拼接该卡的全部效果文本，模式选项放在最前（使其归入 play 段）。"""
    pieces = [
        str(card.get("normal_desc") or ""),
        str(card.get("evolve_desc") or ""),
        str(card.get("super_evolve_desc") or ""),
    ]
    body = "\n".join(p for p in pieces if p.strip())
    if not body.strip():
        body = str(card.get("clean_skill_desc") or "")
    mode_bits = [
        str((choice or {}).get("clean_text") or "")
        for choice in (card.get("mode_choices") or ())
    ]
    head = "\n".join(b for b in mode_bits if b.strip())
    return f"{head}\n{body}" if head else body


def _split_segments(text: str) -> dict[str, list[str]]:
    text = EVOLVE_BODY_RE.sub(
        lambda m: "【超进化时】" if m.group(1) else "【进化时】", text or ""
    )
    text = ENTER_FIELD_RE.sub("【入场曲】", text)
    text = _INHERIT_FANFARE_RE.sub(_INHERIT_TOKEN, text)
    marks = list(SEGMENT_RE.finditer(text))
    out: dict[str, list[str]] = {}
    if not marks:
        return {"play_normal": [text]}
    head = text[: marks[0].start()].strip()
    if head:
        out.setdefault("play_normal", []).append(head)
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        key = re.sub(r"_?\d*$", "", mark.group(1))
        # 未登记段标记（目前只有【谢幕曲】）整段丢弃：它是被动收益，不属于主动动作效果
        slot = _SEGMENT_SLOT.get(key) or "__drop__"
        out.setdefault(slot, []).append(text[mark.end(): end])
    return out


# 整句只由关键字标签组成（"【守护】"、"【疾驰】和【突进】"）——那是卡牌自带属性，不是动作效果
_KEYWORD_ONLY_RE = re.compile(r"^(?:\s*【[^】]+】\s*(?:和|与|、)?\s*)+$")
# 句首的被动条目标记：该句只是被动触发/持续效果
_LEADING_PASSIVE_RE = re.compile(r"^(?:【(?:谢幕曲|魔力增幅时|吟唱_?\d*|融合|瞬念召唤|攻击时|毁灭|灵气|屏障|威慑|虹吸)】)")
# 被动句的延续：只有条件/结果连接词开头的句子仍属上一被动段
_CONTINUATION_RE = re.compile(r"^(?:若|则|之后|改为|并且|且)")
# "自己的回合结束时 / 被破坏时" 所辖的整段都属于被动，直到下一个段标记
_STRONG_PASSIVE_RE = re.compile(r"(?:回合结束|回合开始|被破坏|被舍弃)时")
# "破坏本卡牌"/"破坏这个随从" 是自毁代价，不是去除对手
_SELF_SACRIFICE_RE = re.compile(r"破坏(?:本卡牌|这个随从|本随从)")
# "选择自己的战场上的1张卡牌，破坏该卡牌" 同样是自毁（对手侧才会写"对手的战场上的"）
_SELF_SIDE_DESTROY_RE = re.compile(r"自己的[^。]{0,30}破坏")


def _is_passive_sentence(sentence: str) -> bool:
    if _KEYWORD_ONLY_RE.match(sentence):
        return True
    if _LEADING_PASSIVE_RE.match(sentence):
        return True
    if any(token in sentence for token in _DROP_TOKENS):
        return True
    # 被动触发句式："…时，" 开头的条件从句
    return "时，" in sentence


_INHERIT_FANFARE_RE = re.compile(r"发动与【入场曲】相同的能力")
# 该短语内部含一个【入场曲】引用，若不先换掉，段切分会在引用处再切一刀
_INHERIT_TOKEN = "〔继承入场曲〕"
# "不会被能力破坏" 是持续免疫，不是破坏效果
_DESTROY_IMMUNE_RE = re.compile(r"不会被[^。，,]{0,10}破坏|被破坏的")


def _tags_from_text(text: str) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    inherits_fanfare: set[str] = set()
    for slot, bodies in _split_segments(text).items():
        if slot == "__drop__":
            continue
        sentences: list[str] = []
        conditional = False
        mode_choice = False
        has_self_sacrifice = False
        for body in bodies:
            if _INHERIT_TOKEN in body:
                inherits_fanfare.add(slot)
            # 被动触发句之后的句子同样属于被动段（"被舍弃时，若费用为7，则…"）
            passive_run = False
            strong_run = False
            for sentence in re.split(r"(?<=。)|[\r\n]+", body):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if _is_passive_sentence(sentence):
                    if _STRONG_PASSIVE_RE.search(sentence):
                        strong_run = True
                    passive_run = True
                    continue
                if strong_run:
                    continue
                if passive_run and not _CONTINUATION_RE.match(sentence):
                    passive_run = False
                if passive_run:
                    continue
                if CONDITION_MARK_RE.search(sentence):
                    conditional = True
                if "【模式】" in sentence:
                    mode_choice = True
                sentence = _DESTROY_IMMUNE_RE.sub("", sentence)
                # 自毁：显式写法（破坏本卡牌/本随从/这个随从），或同句"自己的…破坏"且不含"对手"。
                self_side_destroy = (
                    _SELF_SIDE_DESTROY_RE.search(sentence) is not None
                    and "对手" not in sentence
                )
                if _SELF_SACRIFICE_RE.search(sentence) or self_side_destroy:
                    has_self_sacrifice = True
                    # 抹掉"破坏"二字，避免同一处又被计成去除对手的 destroy
                    sentence = re.sub(r"破坏", "", sentence)
                sentences.append(CONDITION_MARK_RE.sub("", sentence))
        if not sentences:
            continue
        tags: set[str] = set()
        for sentence in sentences:
            for tag, pattern in _TEXT_PATTERNS.items():
                if re.search(pattern, sentence):
                    tags.add(tag)
            # 打脸 / 群体伤害必须与"造成伤害"同句，避免"回复主战者"误判
            if re.search(_TEXT_PATTERNS["damage"], sentence):
                if "对手的主战者" in sentence:
                    tags.add("damage_leader")
                if re.search(r"(?:所有|全部)(?:随从|主战者)|分配", sentence):
                    tags.add("damage_aoe")
            if "亡者召还" in sentence:
                tags.add("reanimate")
        if conditional:
            tags.add("conditional")
        if mode_choice:
            tags.add("mode_choice")
        if has_self_sacrifice:
            tags.add("self_sacrifice")
        if tags:
            result[slot] = frozenset(tags)
    # 「发动与【入场曲】相同的能力」→ 该槽位继承 play_normal 的标签（进化时 / 超进化时都会写这句）
    inherited = result.get("play_normal")
    if inherited:
        for slot in inherits_fanfare:
            result[slot] = frozenset(result.get(slot, frozenset()) | inherited)
    return result


def _tags_from_related(card: Mapping[str, Any]) -> frozenset[str]:
    """激奏 / 结晶形态的效果落在**关联卡**上（名字带「激奏：」/「结晶：」前缀）。

    这类关联卡的效果文本不做"被动段"过滤：结晶形态的核心价值就是那张护符的
    【谢幕曲】（例如「结晶：渊底上校」= 吟唱 4 + 谢幕曲召唤渊底上校），把它当
    噪音丢掉会让"结晶"动作看起来什么都没做。
    """
    text = _deduped_text(card).replace("【谢幕曲】", "")
    parts = _tags_from_text(text)
    # 只取 play_normal：关联卡里的【启动】是护符放好之后的后续能力，不是这次出牌的效果
    return parts.get("play_normal", frozenset())


@lru_cache(maxsize=1)
def _card_table() -> dict[int, dict[str, frozenset[str]]]:
    """base_card_id -> {slot: tags}。首次调用读 card_database.json（约 6MB）。"""
    try:
        payload = json.loads(_GOLD.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    cards = payload.get("cards") if isinstance(payload, dict) else payload
    by_id = {
        int(card.get("base_card_id") or 0): card
        for card in (cards or ())
        if isinstance(card, dict) and card.get("base_card_id")
    }
    table: dict[int, dict[str, frozenset[str]]] = {}
    for base_card_id, card in by_id.items():
        table[base_card_id] = _tags_from_text(_deduped_text(card))

    # 第二遍：激奏 / 结晶。主卡自身没有这两种出牌的文本，效果在关联卡里。
    for base_card_id, card in by_id.items():
        for ref in card.get("specific_effect_card_ids") or ():
            related = by_id.get(int(ref))
            if related is None:
                continue
            name = str(related.get("name") or "")
            if name.startswith("激奏："):
                slot = "play_accelerate"
            elif name.startswith("结晶："):
                slot = "play_crystal"
            else:
                continue
            tags = _tags_from_related(related)
            if tags:
                table.setdefault(base_card_id, {})[slot] = tags
    return table


def card_effect_tags(base_card_id: int | None) -> Mapping[str, frozenset[str]]:
    """base_card_id -> {slot: frozenset[tag]}；未收录返回空 dict（不猜测）。"""
    if not base_card_id:
        return {}
    table = _card_table()
    return table.get(int(base_card_id), {})


def action_effect_slot(action_kind: str, play_kind: str | None = None) -> str | None:
    """(ActionKind, PlayKind) -> 槽位；无法对应主动效果时返回 None。"""
    kind = str(action_kind)
    if kind == "play":
        return {
            "enhance": "play_enhance",
            "accelerate": "play_accelerate",
            "crystal": "play_crystal",
        }.get(str(play_kind or "normal"), "play_normal")
    if kind in ("evolve", "super_evolve", "activate"):
        return kind
    return None


def action_effect_tags(
    base_card_id: int | None, *, action_kind: str, play_kind: str | None = None
) -> frozenset[str]:
    """某张卡在某个动作下会产生的效果标签。"""
    slot = action_effect_slot(action_kind, play_kind)
    if slot is None:
        return frozenset()
    return card_effect_tags(base_card_id).get(slot, frozenset())


def action_effect_vector(
    base_card_id: int | None, *, action_kind: str, play_kind: str | None = None
) -> tuple[float, ...]:
    """固定宽度 multi-hot；宽度 = len(TAGS)，顺序与 TAGS 一致。"""
    tags = action_effect_tags(base_card_id, action_kind=action_kind, play_kind=play_kind)
    if not tags:
        return (0.0,) * len(TAGS)
    return tuple(1.0 if tag in tags else 0.0 for tag in TAGS)


def describe_tags(tags: Iterable[str]) -> str:
    return "|".join(sorted(tags)) or "-"
