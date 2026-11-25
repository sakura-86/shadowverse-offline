# -*- coding: utf-8 -*-
"""回合终局评分：先斩杀，再看净血差，最后看保下来的场面。

## 为什么需要这一层

`tactical_summary` 回答的是「这回合的攻击资源够不够清场 / 够不够斩杀」，是一个
**上界**判断。但真实决策要回答的是「这些攻击怎么分配」，而不同分配方式的收益
差别很大：

    我方 5/5 5/5（可进化、可打脸），对方 8/7，对方还有超进化点。

    A) 进化一只，全部打脸：打 12 点，但漏了 8/7。
       对方下回合反打 8 + 3 = 11 点，净血差 12 - 11 = 1。
    B) 进化的那只撞掉 8/7，另一只打脸：打 5 点，场上不留人。
       净血差 5 - 0 = 5。

B 更好，尽管它的「伤害」低得多。所以评分必须是**净血差**，不能是伤害。

净血差 = 我方本回合对敌方主战者造成的伤害 - 对方下回合的潜在伤害

对方的潜在伤害按「漏掉的随从全部打脸」估：

    对方潜在伤害 = 对方终局场上所有随从的攻击力之和 + 进化加成

进化加成只算一次（对方一回合只能用一次进化）：有超进化点 +3，只有进化点 +2。

## 评分是四级字典序

1. ``lethal``：能斩杀就斩杀，1 表示达成（这一级直接压过后面所有项）
2. ``net_life``：净血差
3. ``survivors``：我方存活随从数
4. ``bodies``：存活随从的身材，**按原随从的身材降序逐位比较**

第 4 级为什么要按「原随从」排序：10/10 和 5/5 面对 4/4 和 1/1 时，两种打法的
剩余身材总和都是 25，但让 10/10 去打 1/1（只掉 1 点生命）比让它去打 4/4（掉 4 点）
更好——数值越大越可能产生溢价，对方要在这只随从上多花一次攻击，群体伤害的解场
上限也被抬高。按终局身材排序会得出相反结论，所以必须锁定随从身份再比较。

## 进化加成的口径

`sep` / `ep` 是**剩余**点数。快照里可能没有这两个字段（谜题快照就没有），此时
按 0 处理并在 ``details`` 里记下 ``evolve_bonus_kind="unknown"``，不做猜测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: 超进化 +3/+3，普通进化 +2/+2（`docs/reference/shadowverse-rules.md` §进化）。
_SUPER_EVOLVE_ATTACK_BONUS = 3
_EVOLVE_ATTACK_BONUS = 2


@dataclass(frozen=True)
class TurnScore:
    """一个回合终局的可比较评分。"""

    #: 1 = 本回合打死了对方主战者
    lethal: int
    #: 我方造成的伤害 - 对方下回合的潜在伤害
    net_life: int
    #: 我方剩下的随从数
    survivors: int
    #: 存活随从的身材，按**原随从**身材降序排列
    bodies: tuple[tuple[int, int], ...]
    #: 明细，仅用于诊断
    our_damage: int = 0
    opponent_potential: int = 0
    details: dict[str, Any] | None = None

    def key(self) -> tuple:
        """字典序比较键；越大越好。"""
        return (self.lethal, self.net_life, self.survivors, self.bodies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lethal": self.lethal,
            "net_life": self.net_life,
            "survivors": self.survivors,
            "bodies": [list(body) for body in self.bodies],
            "our_damage": self.our_damage,
            "opponent_potential": self.opponent_potential,
            "details": self.details or {},
        }


def _views(snapshot, player: str):
    if player == "me":
        return snapshot.me, snapshot.opponent
    if player == "opponent":
        return snapshot.opponent, snapshot.me
    raise ValueError(f"unsupported player: {player}")


def opponent_evolve_bonus(opponent_view) -> tuple[int, str]:
    """对方下回合还能给一只随从加多少攻击力。

    返回 ``(bonus, kind)``，``kind`` 取值 ``super_evolve`` / ``evolve`` / ``none`` /
    ``unknown``。``unknown`` 表示快照里没有进化点字段，按 0 计但显式标记。
    """
    sep = getattr(opponent_view, "sep", None)
    ep = getattr(opponent_view, "ep", None)
    if sep is None and ep is None:
        max_sep = getattr(opponent_view, "max_sep", None)
        max_ep = getattr(opponent_view, "max_ep", None)
        if max_sep is None and max_ep is None:
            return 0, "unknown"
        sep, ep = max_sep, max_ep
    if sep is not None and int(sep) > 0:
        return _SUPER_EVOLVE_ATTACK_BONUS, "super_evolve"
    if ep is not None and int(ep) > 0:
        return _EVOLVE_ATTACK_BONUS, "evolve"
    return 0, "none"


def opponent_potential_damage(after_snapshot, player: str = "me") -> tuple[int, dict[str, Any]]:
    """对方下回合「全部打脸」的潜在伤害 + 明细。

    对方场上没有随从时没有进化对象，进化加成不成立——否则「清空对方场面」的路线
    会被凭空扣掉 2~3 点，输给一条放任对方随从活着的路线。
    """
    _, theirs = _views(after_snapshot, player)
    cards = after_snapshot.cards_by_uid
    field_cards = [cards[uid] for uid in theirs.field if uid in cards]
    field_atk = sum(max(0, int(card.atk)) for card in field_cards)
    if not field_cards:
        return 0, {
            "field_atk": 0,
            "evolve_bonus": 0,
            "evolve_bonus_kind": "no_follower",
            "field_card_count": 0,
        }
    bonus, bonus_kind = opponent_evolve_bonus(theirs)
    return field_atk + bonus, {
        "field_atk": field_atk,
        "evolve_bonus": bonus,
        "evolve_bonus_kind": bonus_kind,
        "field_card_count": len(field_cards),
    }


def _body_tuple(before_snapshot, after_snapshot, player: str = "me") -> tuple[tuple[int, int], ...]:
    """存活随从的身材，按**原随从**身材降序排列。

    先按 ``before`` 里每个随从的 (atk + life) 降序确定身份顺序，再取它在 ``after``
    里的身材；中途被打死的记 ``(0, 0)``（保留位置，这样「谁的尸体」也参与比较，
    多死一只就是多一个 ``(0, 0)``，字典序自然更差）。

    回合中新召唤的随从不在 ``before`` 里，按身材降序追加在后面。
    """
    mine_before, _ = _views(before_snapshot, player)
    mine_after, _ = _views(after_snapshot, player)
    before_cards = before_snapshot.cards_by_uid
    after_cards = after_snapshot.cards_by_uid

    original_uids = [uid for uid in mine_before.field if uid in before_cards]
    original_uids.sort(key=lambda uid: -(int(before_cards[uid].atk) + int(before_cards[uid].life)))

    bodies: list[tuple[int, int]] = []
    accounted: set[int] = set()
    for uid in original_uids:
        if uid in mine_after.field and uid in after_cards:
            card = after_cards[uid]
            bodies.append((int(card.atk), int(card.life)))
            accounted.add(uid)
        else:
            bodies.append((0, 0))

    newcomers = [uid for uid in mine_after.field if uid in after_cards and uid not in accounted and uid not in original_uids]
    newcomers.sort(
        key=lambda uid: (
            -(int(after_cards[uid].atk) + int(after_cards[uid].life)),
            -int(after_cards[uid].atk),
        )
    )
    for uid in newcomers:
        card = after_cards[uid]
        bodies.append((int(card.atk), int(card.life)))

    return tuple(bodies)


def score_terminal(before_snapshot, after_snapshot, *, player: str = "me") -> TurnScore:
    """给一个回合终局打分。

    ``before_snapshot`` 是回合开始时的快照（用来确定随从身份顺序和初始主战者生命），
    ``after_snapshot`` 是这条路线走完后的快照。
    """
    _, theirs_before = _views(before_snapshot, player)
    mine_after, theirs_after = _views(after_snapshot, player)

    lethal = 1 if int(theirs_after.life) <= 0 else 0
    our_damage = max(0, int(theirs_before.life) - int(theirs_after.life))
    if lethal:
        # 主战者已死，对方不会再行动，潜在伤害不存在。
        potential, potential_details = 0, {"field_atk": 0, "evolve_bonus": 0, "evolve_bonus_kind": "dead"}
    else:
        potential, potential_details = opponent_potential_damage(after_snapshot, player)

    return TurnScore(
        lethal=lethal,
        net_life=our_damage - potential,
        survivors=len(mine_after.field),
        bodies=_body_tuple(before_snapshot, after_snapshot, player),
        our_damage=our_damage,
        opponent_potential=potential,
        details={
            "before_enemy_life": int(theirs_before.life),
            "after_enemy_life": int(theirs_after.life),
            "after_enemy_field": potential_details,
            "after_own_field_count": len(mine_after.field),
            "after_enemy_field_count": len(theirs_after.field),
        },
    )


__all__ = [
    "TurnScore",
    "opponent_evolve_bonus",
    "opponent_potential_damage",
    "score_terminal",
    "state_fingerprint",
]


def state_fingerprint(snapshot, player: str = "me") -> tuple:
    """状态指纹，供搜索做环检测和记忆化。

    必须覆盖「已经做过什么」，否则同一个盘面会被当成同一个状态：

    - ``attacks_left``：这只随从这回合还能不能攻击；
    - ``base_card_id``：只看 (atk, life) 会把不同随从混为一谈；
    - ``stack`` / ``evolve_state``：土之印的层数决定【土之秘术_N】能否支付，
      进化状态决定击飞/双击/免反伤；
    - ``has_activation`` / ``activation_used_this_turn``：【启动】每张卡每回合一次，
      激活过的和刚打出/刚合并的同层数土之印后续能做的事不一样；
    - 手牌 ``(base_card_id, cost)``：手里还剩哪些牌；
    - 纹章集合：超进化前后的双击窗口不同。
    """
    mine, theirs = _views(snapshot, player)
    cards = snapshot.cards_by_uid

    def _field(pl):
        return tuple(
            sorted(
                (
                    int(cards[uid].base_card_id),
                    int(cards[uid].atk),
                    int(cards[uid].life),
                    int(getattr(cards[uid], "attacks_left", 0) or 0),
                    int(getattr(cards[uid], "stack", 0) or 0),
                    int(getattr(cards[uid], "evolve_state", 0) or 0),
                    int(bool(getattr(cards[uid], "has_activation", False))),
                    int(bool(getattr(cards[uid], "activation_used_this_turn", False))),
                )
                for uid in pl.field
                if uid in cards
            )
        )

    def _hand(pl):
        return tuple(
            sorted((int(cards[uid].base_card_id), int(cards[uid].cost)) for uid in pl.hand if uid in cards)
        )

    return (
        int(mine.life),
        int(theirs.life),
        _field(mine),
        _field(theirs),
        _hand(mine),
        int(getattr(mine, "pp", 0) or 0),
        tuple(sorted(int(crest.card_id) for crest in getattr(mine, "crests", ()) or ())),
    )
