# -*- coding: utf-8 -*-
"""场面指纹（Board Fingerprint）。

本模块提供一个**独立的、只读的基础类型**，用于把一个 `BattleSnapshot` 双方
场上（field）的随从 / 护符卡的**动态关键属性**压缩成稳定、确定性的指纹。

设计边界（严格遵守 architecture-boundaries / prohibited-patterns）：

- **不含 uid**：局内 uid 是不可泛化、易泄漏的绝对标识，一律排除；卡牌只用
  base/card id + 动态属性描述。仅 uid 变化（其余属性不变）**不应**改变指纹。
- **只取动态关键属性**：atk / life / max_life / 关键词 / evolve 状态 /
  attacks_left / countdown / stack 等能影响战斗与后续回合走向的可观察字段。
- **stable / deterministic**：同一方的卡按属性元组排序，保证与 field 排列顺序、
  uid 分配无关；相同动态属性集合产出相同指纹。
- **不可声称状态等价**：BoardFingerprint 相等只说明被签名的场面属性维度相同，
  不代表整体局面等价（手牌、纹章、PP、附着的具体 buff 细节等均未纳入）。
  这是一个 fingerprint（识别用），不是 state-equality proof。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from decision_engine.state.snapshot import BattleSnapshot, CardView, PlayerView

# 视角说明：与 FutureResourceSignature 对齐；'perspective' 决定 ally/enemy 归属。
Perspective = Literal["me", "opponent"]


def _keyword_flags(card: CardView) -> tuple[str, ...]:
    """卡牌关键词标记的稳定排序元组。

    只取影响战斗 / 结算的可观察布尔关键词，不解释其效果 DSL。
    返回排序后的短名列表，保证确定性与可读性。
    """
    flags: list[str] = []
    # (属性名, 短名) 映射；只列可观察的战斗相关关键词。
    keyword_specs: tuple[tuple[str, str], ...] = (
        ("has_guard", "guard"),
        ("has_storm", "storm"),
        ("has_rush", "rush"),
        ("has_drain", "drain"),
        ("has_bane", "bane"),
        ("has_ambush", "ambush"),
        ("has_cant_attack", "cant_attack"),
        ("has_cant_be_attacked", "cant_be_attacked"),
        ("has_cant_select", "cant_select"),
        ("has_last_words", "last_words"),
        ("has_activation", "activation"),
        ("has_damage_cut", "damage_cut"),
        ("has_reduce_damage", "reduce_damage"),
        ("has_cant_destroy", "cant_destroy"),
        ("has_super_evolve_buff", "super_evolve_buff"),
        ("has_temp_shield", "temp_shield"),
        ("can_ignore_guard", "ignore_guard"),
        ("has_induction", "induction"),
        ("is_earth_sigil", "earth_sigil"),
    )
    for attr, short in keyword_specs:
        if bool(getattr(card, attr, False)):
            flags.append(short)
    return tuple(sorted(flags))


@dataclass(frozen=True)
class CardFingerprint:
    """单张场上卡的动态属性指纹（不含 uid）。

    Attributes:
        base_card_id: 卡牌基础 id（跨风格/皮肤稳定身份）。
        card_id: 卡牌具体 id（含风格变体）。
        card_type: 卡牌类型（follower / amulet / countdown_amulet 等）的值字符串。
        atk: 当前攻击力。
        life: 当前生命 / 耐久。
        max_life: 生命 / 耐久上限。
        evolve_state: 进化状态（0 未进化 / 进化 / 超进化，协议数值）。
        attacks_left: 本回合剩余攻击次数（含疾走/多次攻击语义的可观察结果）。
        countdown: 倒计时（护符/倒计时卡）。
        stack: 叠加数（协议 stack 字段）。
        keywords: 关键词短名排序元组。
    """

    base_card_id: int
    card_id: int
    card_type: str
    atk: int
    life: int
    max_life: int
    evolve_state: int
    attacks_left: int
    countdown: int
    stack: int
    keywords: tuple[str, ...]

    @staticmethod
    def from_card(card: CardView) -> "CardFingerprint":
        """从 CardView 提取动态属性指纹（刻意不含 uid）。"""
        card_type_value = (
            card.card_type.value
            if hasattr(card.card_type, "value")
            else str(card.card_type)
        )
        return CardFingerprint(
            base_card_id=int(card.base_card_id),
            card_id=int(card.card_id),
            card_type=str(card_type_value),
            atk=int(card.atk),
            life=int(card.life),
            max_life=int(card.max_life),
            evolve_state=int(card.evolve_state),
            attacks_left=int(card.attacks_left),
            countdown=int(card.countdown),
            stack=int(card.stack),
            keywords=_keyword_flags(card),
        )

    def _sort_key(self) -> tuple:
        """确定性排序键（用于同一方场面卡的稳定排序）。"""
        return (
            self.base_card_id,
            self.card_id,
            self.card_type,
            self.atk,
            self.life,
            self.max_life,
            self.evolve_state,
            self.attacks_left,
            self.countdown,
            self.stack,
            self.keywords,
        )


def _field_fingerprints(
    snapshot: BattleSnapshot, player: PlayerView
) -> tuple[CardFingerprint, ...]:
    """某玩家 field 上所有卡的指纹，按属性排序（与 field 顺序 / uid 无关）。

    field uid 无法在 cards_by_uid 中解析时跳过——这只发生在数据不完整场景，
    指纹本就只描述"可观察到的场面卡"，不臆造占位卡。
    """
    prints: list[CardFingerprint] = []
    for uid in player.field:
        card = snapshot.cards_by_uid.get(uid)
        if card is None:
            continue
        prints.append(CardFingerprint.from_card(card))
    prints.sort(key=lambda fp: fp._sort_key())
    return tuple(prints)


@dataclass(frozen=True)
class BoardFingerprint:
    """一个 BattleSnapshot 双方场面的指纹（不含 uid、视角明确）。

    - `ally` / `enemy` 分别为视角玩家与对手的场面卡指纹元组（均按属性排序）。
    - 场面卡是双方公开可见信息，因此两侧都完整暴露卡身份与动态属性
      （不存在对手隐藏泄漏问题——场面本就公开）。
    - frozen + 全 hashable 字段，可作为 dict key / set 元素，比较确定性。
    - **不可**声称为状态等价：只覆盖被签名的场面属性维度。仅 uid 变化不改变指纹。
    """

    perspective: Perspective
    ally: tuple[CardFingerprint, ...]
    enemy: tuple[CardFingerprint, ...]

    @staticmethod
    def from_snapshot(
        snapshot: BattleSnapshot,
        perspective: Perspective = "me",
    ) -> "BoardFingerprint":
        """从 BattleSnapshot 构造场面指纹。

        Args:
            snapshot: 局面状态视图。
            perspective: 视角玩家（'me' 或 'opponent'），决定 ally/enemy 归属。

        Returns:
            BoardFingerprint 实例。
        """
        if perspective == "me":
            ally_pv, enemy_pv = snapshot.me, snapshot.opponent
        elif perspective == "opponent":
            ally_pv, enemy_pv = snapshot.opponent, snapshot.me
        else:  # pragma: no cover - 防御性：非法视角显式报错，不静默 fallback
            raise ValueError(f"未知 perspective: {perspective!r}，应为 'me' 或 'opponent'")

        return BoardFingerprint(
            perspective=perspective,
            ally=_field_fingerprints(snapshot, ally_pv),
            enemy=_field_fingerprints(snapshot, enemy_pv),
        )
