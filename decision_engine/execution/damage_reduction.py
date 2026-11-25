# -*- coding: utf-8 -*-
"""伤害计算统一工具。

超进化随从在自己的回合内受到的所有伤害（包括战斗反击和效果伤害）皆变为 0。
受到的伤害为 0 仍视为一次受伤事件（可触发持续效果和纹章）。
参见 local-rule-spec.md §13.4。
"""

from __future__ import annotations

from decision_engine.state.snapshot import BattleSnapshot, CardView


def apply_damage_cut(target_card: CardView, raw_amount: int) -> int:
    """应用卡牌级伤害减免（DamageCut）。

    语义："受到的4点或以上的伤害变为3点"（如威严的星晶骑士·薇拉）。
    该减免既作用于战斗伤害，也作用于效果伤害，因此集中在此处，
    供 rule_kernel 战斗路径和 EffectInterpreter 效果伤害路径共用。

    参见 local-rule-spec.md §13.4 / shadowverse-rules.md（HasDamageCut）。
    """
    if target_card.has_damage_cut and raw_amount >= 4:
        return 3
    return raw_amount


def effective_damage_to_follower(
    snapshot: BattleSnapshot,
    target_card: CardView,
    target_owner: str,
    raw_amount: int,
) -> int:
    """计算随从实际受到的伤害。

    结算顺序：
    1. 超进化随从（evolve_state >= 2）在自己的回合中受到的伤害变为 0（最高优先级）。
    2. 否则应用卡牌级伤害减免（DamageCut，"4点及以上变为3点"）。

    该函数是随从受伤的统一入口，战斗伤害与效果伤害都应经过它，
    保证减免机制在两条路径上语义一致。

    Args:
        snapshot: 当前盘面（用于判断回合归属）。
        target_card: 受伤目标卡。
        target_owner: 目标所属侧 "me" / "opponent"。
        raw_amount: 原始伤害数值。

    Returns:
        实际伤害数值（可能为 0）。
    """
    if is_super_evolved_on_own_turn(snapshot, target_card, target_owner):
        return 0
    return apply_damage_cut(target_card, raw_amount)


def is_super_evolved_on_own_turn(
    snapshot: BattleSnapshot,
    card: CardView,
    owner: str,
) -> bool:
    """判断卡是否是超进化状态且在自己的回合中。

    「自己的回合中」包括本回合的**回合结束结算**：`CoreRuleKernel.end_turn` 只把
    `is_turn_ended` 置 True、不翻转 `is_ally_turn`，回合结束能力仍发生在这个回合内。
    因此这里只看 `is_ally_turn`，不能把 `is_turn_ended` 当作「回合已结束」而取消保护
    （否则回合结束的效果伤害会穿透超进化，见 replay_2882303761750246771 step60）。
    """
    if card.evolve_state < 2:
        return False
    if owner == "me":
        return bool(snapshot.is_ally_turn)
    if owner == "opponent":
        return not snapshot.is_ally_turn
    return False
