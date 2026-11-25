# -*- coding: utf-8 -*-
"""回合战术摘要：把「我方有多少攻击资源 / 对方要清多少血」算出来。

动机（``docs/plan/SEARCH_MYOPIA_DIAGNOSIS.md`` §5.4）：

S5 那类局面，人类不是搜索出来的，是**数资源做除法**：

- 对方 5 个随从，血量 9/9/6/6/2；
- 我方 4 次攻击（7/7 + 6/7，各有双击）+ 回合结束 AOE 2；
- 于是「7/7 打两个 9 血各一次、6/7 打两个 6 血各一次、2 血交给 AOE」。

而 PUCT 在 8 层、每层 6~20 个分支下要 64 万条路径里撞出唯一那条，512 sims 只覆盖 0.08%。

本模块只做**确定性算术**：给定快照，输出攻击次数、总攻击伤害、回合结束 AOE、对方血量，
以及「这回合能不能清场 / 能不能斩杀」的上界判断。它不改搜索，不改模型，也不替搜索做决策，
只把「目标是否可达」变成一个搜索和模型都能直接读到的标量。

用法：

    from decision_engine.planning.tactical_summary import summarize_turn
    summary = summarize_turn(snapshot, player="me")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CARD_DATABASE_PATH = ROOT / "decision_engine" / "data" / "public" / "card_database.json"

#: 「自己的回合结束时，……对对手的战场上的所有随从造成 N 点伤害」
_TURN_END_AOE_RE = re.compile(
    r"回合结束时[^。]*?对(?:对手|敌方)[^。]*?所有随从造成\s*(\d+)\s*点伤害"
)
#: 同一段文本里是否要求「超进化后的随从」
_SUPER_EVOLVE_CONDITION_RE = re.compile(r"超进化")


@lru_cache(maxsize=1)
def _card_database() -> dict[int, dict[str, Any]]:
    payload = json.loads(CARD_DATABASE_PATH.read_text(encoding="utf-8"))
    cards = payload.get("cards") if isinstance(payload, dict) else payload
    return {int(card["base_card_id"]): card for card in cards if "base_card_id" in card}


#: 「自己的回合结束时，若本随从为进化后，则使战场上的其他所有随从 -1/-1」
_TURN_END_DEBUFF_RE = re.compile(
    r"回合结束时[^。]*?使(?:战场上的)?(?:其他)?所有随从\s*-(\d+)/-(\d+)"
)
#: 同一张卡文本里是否要求「本随从为进化后」
_SELF_EVOLVED_CONDITION_RE = re.compile(r"本随从为进化后")
#: 「超进化后的随从攻击随从时……获得『1回合可以攻击2次』」（纹章 10864112）
_DOUBLE_ATTACK_RE = re.compile(r"超进化[^。]*?攻击随从[^。]*?攻击\s*2\s*次")


def double_attack_crest_from_card(base_card_id: int) -> dict[str, Any] | None:
    """识别「超进化随从攻击随从时获得双击」类纹章；识别不到返回 ``None``。"""
    card = _card_database().get(int(base_card_id))
    if card is None:
        return None
    text = str(card.get("clean_skill_desc") or card.get("normal_desc") or "")
    if _DOUBLE_ATTACK_RE.search(text) is None:
        return None
    return {"base_card_id": int(base_card_id), "text": text}


def has_double_attack_crest(crest_views) -> bool:
    for crest in crest_views or ():
        if double_attack_crest_from_card(int(crest.card_id)) is not None:
            return True
    return False


@lru_cache(maxsize=1)
def _crest_granters() -> dict[int, tuple[int, ...]]:
    """base_card_id -> 它能通过 ``add_crest`` 获得的纹章 id。

    用途：判断「本回合**可获取**的纹章」。已获取的纹章直接读 `crests` 区；但像
    10204122（回合结束 AOE）和 10864112（超进化随从双击）这类纹章，本回合可能
    还没拿到——它挂在某张手牌/场上卡的超进化时或入场曲上。想知道「这回合有没有
    可能用到」，就得往前看一步。

    这里只做静态索引，不判断触发条件是否满足：调用方按「乐观上界」使用。
    """
    path = ROOT / "decision_engine" / "data" / "public" / "card_templates.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[int, set[int]] = {}
    for template in payload.get("templates") or ():
        base_card_id = int((template.get("identity") or {}).get("base_card_id") or 0)
        if not base_card_id:
            continue
        effects = list(template.get("executable_effects") or ())
        for persistent in template.get("persistent_effects") or ():
            effects.extend(persistent.get("executable_effects") or ())
        for effect in effects:
            for operation in effect.get("operations") or ():
                if operation.get("op") != "add_crest":
                    continue
                for crest_id in (operation.get("params") or {}).get("crest_card_ids") or ():
                    out.setdefault(base_card_id, set()).add(int(crest_id))
    return {card_id: tuple(sorted(crest_ids)) for card_id, crest_ids in out.items()}


def pending_crest_ids(snapshot, *, player: str = "me") -> tuple[int, ...]:
    """本回合**可能获得**的纹章 id（我方手牌 + 场上卡的 add_crest 来源）。

    不判断触发条件（超进化、入场曲、连击……），只回答「盘面里有没有能给出这个纹章
    的卡」。搜索本身不需要它——只要枚举器能出那张牌，预测器就会把纹章加进 crest 区；
    这个函数是给诊断输出和乐观上界用的。
    """
    view = snapshot.me if player == "me" else snapshot.opponent
    cards = snapshot.cards_by_uid
    granters = _crest_granters()
    found: set[int] = set()
    for uid in (*view.hand, *view.field):
        card = cards.get(uid)
        if card is None:
            continue
        found.update(granters.get(int(card.base_card_id), ()))
    return tuple(sorted(found))


def pending_crest_summary(snapshot, *, player: str = "me") -> dict[str, Any]:
    """把「本回合可获取的纹章」折算成对战术判断有用的量。"""
    ids = pending_crest_ids(snapshot, player=player)
    aoe_total = 0
    aoe_evidence: list[dict[str, Any]] = []
    for crest_id in ids:
        info = turn_end_aoe_from_card(crest_id)
        if info is None:
            continue
        aoe_total += int(info["damage"])
        aoe_evidence.append({**info, "source": "pending"})
    return {
        "crest_ids": list(ids),
        "aoe_total": aoe_total,
        "aoe_evidence": aoe_evidence,
        "double_attack": any(double_attack_crest_from_card(crest_id) is not None for crest_id in ids),
    }


def turn_end_debuff_from_card(base_card_id: int) -> dict[str, Any] | None:
    """从卡牌文本提取「回合结束时对随从的 -N/-N」；识别不到返回 ``None``。

    与 :func:`turn_end_aoe_from_card` 分开：AOE 是「造成 N 点伤害」，
    debuff 是「-N/-N」，两者结算方式不同但都减少生命，可以叠加计入清场判断。
    """
    card = _card_database().get(int(base_card_id))
    if card is None:
        return None
    text = str(card.get("clean_skill_desc") or card.get("normal_desc") or "")
    match = _TURN_END_DEBUFF_RE.search(text)
    if match is None:
        return None
    return {
        "base_card_id": int(base_card_id),
        "life_delta": -int(match.group(2)),
        "requires_self_evolved": bool(_SELF_EVOLVED_CONDITION_RE.search(text)),
        "text": text,
    }


def turn_end_aoe_from_card(base_card_id: int) -> dict[str, Any] | None:
    """从卡牌文本提取「回合结束 AOE」；识别不到返回 ``None``。

    只解析文本里有明确伤害数值的形式，识别不到就明确返回 ``None``，不做猜测。
    """
    card = _card_database().get(int(base_card_id))
    if card is None:
        return None
    text = str(card.get("clean_skill_desc") or card.get("normal_desc") or "")
    match = _TURN_END_AOE_RE.search(text)
    if match is None:
        return None
    return {
        "base_card_id": int(base_card_id),
        "damage": int(match.group(1)),
        "requires_super_evolve_follower": bool(_SUPER_EVOLVE_CONDITION_RE.search(text)),
        "text": text,
    }


@dataclass(frozen=True)
class TurnResourceSummary:
    #: 本回合可用的攻击次数（含双击等「多次攻击」加成后的保守估计）
    attack_count: int
    #: 这些攻击合起来能造成的总伤害（不考虑目标血量上限）
    total_attack_damage: int
    #: 其中「攻击时不受反伤」的攻击次数（超进化随从在自己回合免疫反伤）
    safe_attack_count: int
    #: 其中「攻击时不受反伤」的总伤害
    safe_attack_damage: int
    #: 回合结束 AOE 伤害（没有则为 0）
    end_of_turn_aoe: int
    #: 该 AOE 的触发条件当前是否满足
    aoe_ready: bool
    #: 对方场上每个随从的当前血量
    enemy_field_hps: tuple[int, ...]
    enemy_field_total_hp: int
    #: 对方主战者生命
    enemy_life: int
    #: 严格口径：只用「免反伤」攻击就能清场（自己不死）
    can_clear_field: bool
    #: 宽松口径：允许普通随从换掉对方随从（自己会死）
    can_clear_field_at_cost: bool
    #: 粗略上界：这回合能不能直接打死对方主战者
    can_lethal: bool
    #: 诊断：参与计算的明细
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_count": self.attack_count,
            "total_attack_damage": self.total_attack_damage,
            "safe_attack_count": self.safe_attack_count,
            "safe_attack_damage": self.safe_attack_damage,
            "end_of_turn_aoe": self.end_of_turn_aoe,
            "aoe_ready": self.aoe_ready,
            "enemy_field_hps": list(self.enemy_field_hps),
            "enemy_field_total_hp": self.enemy_field_total_hp,
            "enemy_life": self.enemy_life,
            "can_clear_field": self.can_clear_field,
            "can_clear_field_at_cost": self.can_clear_field_at_cost,
            "can_lethal": self.can_lethal,
            "details": self.details,
        }


def _player_view(snapshot, player: str):
    if player == "me":
        return snapshot.me, snapshot.opponent
    if player == "opponent":
        return snapshot.opponent, snapshot.me
    raise ValueError(f"unsupported player: {player}")


def _summarize_crests(crest_views) -> dict[str, Any]:
    aoe_total = 0
    ready = False
    evidence: list[dict[str, Any]] = []
    for crest in crest_views or ():
        info = turn_end_aoe_from_card(int(crest.card_id))
        if info is None:
            continue
        aoe_total += int(info["damage"])
        evidence.append(info)
        # 条件满足与否由调用方按场面判断；这里先记条件类型
    return {"aoe_total": aoe_total, "evidence": evidence, "requires_super_evolve": any(e["requires_super_evolve_follower"] for e in evidence)}


def _summarize_field_end_of_turn(field_cards) -> dict[str, Any]:
    """扫描**我方场上随从**自己的「回合结束时」效果。

    典型：索菲娜「自己的回合结束时，若本随从为进化后，则使战场上的其他所有随从 -1/-1」。
    这类效果不看纹章，只在随从在场且满足自身条件时触发，所以必须单独扫一遍——
    只扫 crests 会漏掉它（S4 就是靠这条清场的）。
    """
    debuffs: list[dict[str, Any]] = []
    enemy_life_delta = 0
    ally_other_life_delta = 0
    for card in field_cards or ():
        info = turn_end_debuff_from_card(int(card.base_card_id))
        if info is None:
            continue
        if info["requires_self_evolved"] and int(getattr(card, "evolve_state", 0)) < 1:
            continue
        entry = {**info, "source_base_card_id": int(card.base_card_id)}
        debuffs.append(entry)
        # 「其他所有随从」= 敌方全部 + 我方除自己外的所有人
        enemy_life_delta += int(info["life_delta"])
        ally_other_life_delta += int(info["life_delta"])
    return {
        "debuffs": debuffs,
        "enemy_life_delta": enemy_life_delta,
        "ally_other_life_delta": ally_other_life_delta,
        "count": len(debuffs),
    }


def summarize_turn(snapshot, *, player: str = "me") -> TurnResourceSummary:
    """计算当前局面的回合资源摘要（纯算术，不搜索）。"""
    mine, theirs = _player_view(snapshot, player)
    cards = snapshot.cards_by_uid

    field_cards = [cards[uid] for uid in mine.field if uid in cards]

    # 攻击次数：attacks_left 已经包含双击/多次攻击的效果（如果有对应 buff）
    attack_count = sum(max(0, int(getattr(card, "attacks_left", 0))) for card in field_cards)
    total_attack_damage = sum(
        max(0, int(card.atk)) * max(0, int(getattr(card, "attacks_left", 0))) for card in field_cards
    )

    enemy_field_cards = [cards[uid] for uid in theirs.field if uid in cards]
    enemy_hps = tuple(max(0, int(card.life)) for card in enemy_field_cards)
    enemy_total_hp = sum(enemy_hps)

    crest_info = _summarize_crests(getattr(mine, "crests", ()))
    aoe_damage = int(crest_info["aoe_total"])
    super_evolved = [card for card in field_cards if int(getattr(card, "evolve_state", 0)) == 2]
    aoe_ready = bool(aoe_damage > 0 and (not crest_info["requires_super_evolve"] or super_evolved))
    field_eot = _summarize_field_end_of_turn(field_cards)

    # 清场上界：只统计「攻击时不受反伤」的随从（超进化随从在自己回合免疫反伤，
    # 见 rule_kernel.attack_follower）。用普通随从去换对方 13 攻随从是纯亏，
    # 不能算进「可以清场」——否则 S1/S3/S4 会被误判成能清。
    super_evolved = [card for card in field_cards if int(getattr(card, "evolve_state", 0)) == 2]
    safe_attack_count_base = sum(max(0, int(getattr(card, "attacks_left", 0))) for card in super_evolved)
    safe_attack_damage_base = sum(
        max(0, int(card.atk)) * max(0, int(getattr(card, "attacks_left", 0))) for card in super_evolved
    )

    # 双击纹章（10864112）：超进化随从「攻击随从时」额外获得一次攻击。
    # 它只在打随从时生效（打脸不加），所以 safe_* 翻倍、leader_damage 不变。
    # 不把这条算进去会让 S5 被剪成「伤害 13 < 剩余 22，无望」——实际能打 26。
    double_attack = has_double_attack_crest(getattr(mine, "crests", ()))
    if double_attack:
        extra_count = safe_attack_count_base
        extra_damage = safe_attack_damage_base
        safe_attack_count = safe_attack_count_base + extra_count
        safe_attack_damage = safe_attack_damage_base + extra_damage
        # 非超进化随从没有双击，只补超进化那部分的额外一击
        attack_count += extra_count
        total_attack_damage += extra_damage
    else:
        safe_attack_count = safe_attack_count_base
        safe_attack_damage = safe_attack_damage_base

    # 回合结束的伤害来源有两类，都会在 turn_end 时结算：
    #   1) 纹章 AOE（造成 N 点伤害）；
    #   2) 场上随从自己的「回合结束时 -N/-N」（例如索菲娜进化后对双方其他随从 -1/-1）。
    # 只算 (1) 会漏掉 S4——它没有纹章，清场完全靠索菲娜。
    end_of_turn_damage = aoe_damage + max(0, -int(field_eot["enemy_life_delta"]))
    end_of_turn_ready = bool(aoe_ready or field_eot["count"] > 0)
    if end_of_turn_ready:
        after_aoe = [max(0, hp - end_of_turn_damage) for hp in enemy_hps]
    else:
        after_aoe = list(enemy_hps)
    remaining_hp = sum(after_aoe)
    targets_left = sum(1 for hp in after_aoe if hp > 0)
    # 严格口径：只用「免反伤」攻击，自己不死
    can_clear_field = (
        remaining_hp <= safe_attack_damage and targets_left <= safe_attack_count
    ) if enemy_hps else True
    # 宽松口径：允许用普通随从换（Rush 随从打 13 攻随从必死，但场确实清了）
    can_clear_field_at_cost = (
        remaining_hp <= total_attack_damage and targets_left <= attack_count
    ) if enemy_hps else True

    # 斩杀只算「能攻击主战者」的随从：Rush 随从不能打脸，不能计入斩杀线。
    leader_capable = [card for card in field_cards if bool(getattr(card, "can_attack_leader", False))]
    leader_damage = sum(
        max(0, int(card.atk)) * max(0, int(getattr(card, "attacks_left", 0))) for card in leader_capable
    )
    # 击飞：超进化随从击杀对方随从时，对敌方主战者额外 1 点（rules §超进化）。
    # 每次「攻击机会」最多打掉一个随从，所以上限是 min(对方随从数, 攻击次数)。
    trample_damage = min(len(enemy_field_cards), safe_attack_count) if safe_attack_count > 0 else 0
    can_lethal = int(theirs.life) <= leader_damage + trample_damage

    return TurnResourceSummary(
        attack_count=attack_count,
        total_attack_damage=total_attack_damage,
        safe_attack_count=safe_attack_count,
        safe_attack_damage=safe_attack_damage,
        end_of_turn_aoe=end_of_turn_damage,
        aoe_ready=end_of_turn_ready,
        enemy_field_hps=enemy_hps,
        enemy_field_total_hp=enemy_total_hp,
        enemy_life=int(theirs.life),
        can_clear_field=can_clear_field,
        can_clear_field_at_cost=can_clear_field_at_cost,
        can_lethal=can_lethal,
        details={
            "player": player,
            "field_cards": [
                {"base_card_id": int(card.base_card_id), "atk": int(card.atk), "life": int(card.life), "attacks_left": int(getattr(card, "attacks_left", 0)), "evolve_state": int(getattr(card, "evolve_state", 0))}
                for card in field_cards
            ],
            "super_evolved_count": len(super_evolved),
            "safe_attack_count": safe_attack_count,
            "safe_attack_damage": safe_attack_damage,
            "crest_aoe": crest_info,
            "field_end_of_turn": field_eot,
            "end_of_turn_damage": end_of_turn_damage,
            "after_aoe_hps": after_aoe,
            "remaining_hp": remaining_hp,
            "targets_left": targets_left,
            "leader_capable_count": len(leader_capable),
            "leader_damage": leader_damage,
            "trample_damage": trample_damage,
            "double_attack_crest": double_attack,
            # 「本回合可获取」的纹章。不参与 can_lethal / can_clear_field（那些是
            # 严格口径，必须已经拿到），只作为诊断和乐观上界的输入。
            "pending_crests": pending_crest_summary(snapshot, player=player),
        },
    )
