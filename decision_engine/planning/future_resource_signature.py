# -*- coding: utf-8 -*-
"""未来资源签名（Future Resource Signature）。

本模块提供一个**独立的、只读的基础类型**，用于把一个 `BattleSnapshot`
中"影响后续回合资源走向"的可观察事实压缩成一个稳定、确定性的签名。

设计边界（严格遵守 architecture-boundaries / prohibited-patterns）：

- **只从本地可观察状态构造**：所有字段都来自 `BattleSnapshot` 已有的协议级视图
  （`PlayerView` / `CrestView` / `ExtraPpView`），不引入任何卡牌效果 DSL，
  不依赖协议 `Can*` / `Select*` 字段，不读取 after_context / runtime 路径 / 局内 uid 绝对值。
- **不泄漏对手隐藏手牌**：站在传入的 `perspective`（视角玩家）角度，
  只有"视角玩家自己"可以包含 known hand 的 base_card_id 多重集；
  对手一律只暴露 hand **count**（数量），绝不暴露对手手牌的任何 base/card id。
- **不猜 spent**：Extra PP lifecycle 复用 `macro_turn_search` 已有的
  NONE/SPENT/HOLD/EXPIRING/UNKNOWN 语义常量。但那套语义需要 turn-end after-state
  才能判定 SPENT/HOLD；在只有单个 snapshot、没有 TurnLine 历史时，
  本模块**不臆测** token 是否被消耗，而是表达"当前 snapshot 的 lifecycle 事实"
  （持有 / 未持有 / 先手无 / 未知），落在 NONE / HOLD / UNKNOWN 三种可从
  单帧安全推导的桶内，绝不写 SPENT。

本类型只做"状态签名"，**不可**被当作"状态等价"证明：
两个 snapshot 的 FutureResourceSignature 相等，只说明这些被签名的资源维度相同，
不代表整体局面等价（例如手牌顺序、隐藏牌具体身份、buff 细节均未纳入）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from decision_engine.state.snapshot import BattleSnapshot, PlayerView

# ---------------------------------------------------------------------------
# Extra PP lifecycle 语义常量（与 macro_turn_search 保持一致）
# ---------------------------------------------------------------------------
#
# 复用 macro_turn_search.extra_pp_future_signature 的语义命名，保证同一套词表：
# - NONE：先手方（不获得 extra_pp），或当前未持有 token → 无 extra_pp 未来差异。
# - SPENT：token 被实际花掉。**仅**能由 turn-end after-state 判定，本模块单帧不产出。
# - HOLD：当前持有 token（保留态）。单帧只能观察到"持有"，故归为 HOLD。
# - EXPIRING：HOLD 的特例——T5 后手持有的第一个 token 会在 T6 发放时被覆盖浪费。
# - UNKNOWN：extra_pp 视图缺失，无法判定持有情况 → 保守分离，不伪称与任何确定态相同。
#
# 单帧可安全推导的桶：{NONE, HOLD, EXPIRING, UNKNOWN}。SPENT 需要历史，绝不臆测。
EXTRA_PP_SIG_NONE = "NONE"
EXTRA_PP_SIG_SPENT = "SPENT"
EXTRA_PP_SIG_HOLD = "HOLD"
EXTRA_PP_SIG_EXPIRING = "EXPIRING"
EXTRA_PP_SIG_UNKNOWN = "UNKNOWN"

# T5 后手持有的第一个 extra_pp 在 T6 发放时被覆盖（与 macro_turn_search 一致）。
_FIRST_EXTRA_PP_EXPIRY_TURN = 5

# 视角说明：'me' / 'opponent' 指 snapshot 里的 me/opponent；
# 'perspective' 明确本签名以哪一方为"己方视角"，决定谁能暴露 known hand id。
Perspective = Literal["me", "opponent"]


def _extra_pp_snapshot_lifecycle(player: PlayerView) -> str:
    """从单个 PlayerView 推导 Extra PP 的 snapshot lifecycle 事实。

    只表达当前帧可观察到的事实，**不猜 spent**：

    - 先手方 → NONE（先手不获得 extra_pp）。
    - extra_pp 视图缺失 → UNKNOWN（无法判定，保守分离）。
    - 未持有 token（extra_pp<=0）→ NONE。
    - 持有 token → HOLD；若为 T5 后手则细分为 EXPIRING。

    注意：`is_used_extra_pp_this_turn` 是回合内瞬态，不代表 token 已从"未来资源"
    中消失（可能 activated-but-not-spent），因此不据此产出 SPENT。
    """
    # 先手方不获得 extra_pp，直接判定 NONE（即便协议给了 extra_pp=0 的对象）。
    if player.is_first_side:
        return EXTRA_PP_SIG_NONE

    extra = player.extra_pp
    if extra is None:
        # 没有 extra_pp 视图 → 无法从单帧判定持有情况，保守 UNKNOWN。
        return EXTRA_PP_SIG_UNKNOWN

    held = max(int(extra.extra_pp), 0)
    if held <= 0:
        return EXTRA_PP_SIG_NONE

    # 持有 token：单帧只能观察到"保留态"，归为 HOLD / EXPIRING。
    is_expiring = player.turn == _FIRST_EXTRA_PP_EXPIRY_TURN and not player.is_first_side
    return EXTRA_PP_SIG_EXPIRING if is_expiring else EXTRA_PP_SIG_HOLD


def _crest_identity(player: PlayerView) -> tuple[tuple[int, int, int], ...]:
    """某玩家纹章区的身份签名（不含 uid）。

    纹章不占场地位，是持续效果资源，会影响后续回合。这里只取影响资源走向的
    协议事实：card_id（身份）、countdown（倒计时剩余）、faith_value（信仰值）。
    排序保证与纹章排列顺序无关的稳定性。unique_id（局内 uid）刻意排除，避免泄漏
    不可泛化特征。
    """
    items: list[tuple[int, int, int]] = []
    for crest in player.crests:
        items.append((int(crest.card_id), int(crest.countdown), int(crest.faith_value)))
    return tuple(sorted(items))


def _known_hand_multiset(snapshot: BattleSnapshot, player: PlayerView) -> tuple[int, ...]:
    """视角玩家自己的 known hand base_card_id 多重集（排序元组）。

    只对"己方视角"调用。手牌 uid 若能在 cards_by_uid 中解析出 CardView，
    取其 base_card_id；无法解析（隐藏/未知牌）则用负 uid 占位以保证键稳定，
    但这只在己方自己身上使用，不涉及对手泄漏。
    """
    ids: list[int] = []
    for uid in player.hand:
        card = snapshot.cards_by_uid.get(uid)
        if card is not None:
            ids.append(int(card.base_card_id))
        else:
            ids.append(-int(uid))
    return tuple(sorted(ids))


@dataclass(frozen=True)
class PlayerResourceFacts:
    """单个玩家的未来资源事实（不含任何对手隐藏泄漏）。

    Attributes:
        life: 当前生命。
        max_life: 生命上限。
        max_pp: PP 上限（决定下回合可用 PP 规模）。
        ep: 进化点。
        sep: 超进化点。
        extra_pp_lifecycle: Extra PP 的 snapshot lifecycle 桶
            （NONE/HOLD/EXPIRING/UNKNOWN，单帧不产 SPENT）。
        deck_count: 牌库剩余张数（影响抽牌 / 牌库耗尽风险）。
        crest_identity: 纹章身份签名（(card_id, countdown, faith_value) 排序元组）。
        known_hand: 己方 known hand base_card_id 多重集；**仅视角玩家自己非空**，
            对手侧恒为 None（不泄漏隐藏手牌）。
        hand_count: 手牌数量。对手侧只暴露该 count，不暴露任何 id。
    """

    life: int
    max_life: int
    max_pp: int
    ep: int
    sep: int
    extra_pp_lifecycle: str
    deck_count: int
    crest_identity: tuple[tuple[int, int, int], ...]
    known_hand: tuple[int, ...] | None
    hand_count: int


@dataclass(frozen=True)
class FutureResourceSignature:
    """一个 BattleSnapshot 的未来资源签名（视角明确、无对手手牌泄漏）。

    - `perspective` 明确本签名以哪一方为己方视角；只有己方 `PlayerResourceFacts`
      带 `known_hand`，对手侧 `known_hand=None` 且只带 `hand_count`。
    - 该类型 frozen + 只含 hashable 字段，天然可作为 dict key / set 元素，
      且比较是确定性的。
    - **不可**声称为状态等价证明：仅覆盖被显式签名的资源维度。
    """

    perspective: Perspective
    ally: PlayerResourceFacts
    enemy: PlayerResourceFacts

    @staticmethod
    def from_snapshot(
        snapshot: BattleSnapshot,
        perspective: Perspective = "me",
    ) -> "FutureResourceSignature":
        """从 BattleSnapshot 构造未来资源签名。

        Args:
            snapshot: 局面状态视图。
            perspective: 视角玩家（'me' 或 'opponent'）。视角玩家是"己方"，
                会暴露 known hand id 多重集；另一方作为"对手"只暴露 hand count。

        Returns:
            FutureResourceSignature 实例。
        """
        if perspective == "me":
            ally_pv, enemy_pv = snapshot.me, snapshot.opponent
        elif perspective == "opponent":
            ally_pv, enemy_pv = snapshot.opponent, snapshot.me
        else:  # pragma: no cover - 防御性：非法视角显式报错，不静默 fallback
            raise ValueError(f"未知 perspective: {perspective!r}，应为 'me' 或 'opponent'")

        ally = PlayerResourceFacts(
            life=int(ally_pv.life),
            max_life=int(ally_pv.max_life),
            max_pp=int(ally_pv.max_pp),
            ep=int(ally_pv.ep),
            sep=int(ally_pv.sep),
            extra_pp_lifecycle=_extra_pp_snapshot_lifecycle(ally_pv),
            deck_count=int(ally_pv.deck_count),
            crest_identity=_crest_identity(ally_pv),
            # 己方可以暴露 known hand base-id 多重集。
            known_hand=_known_hand_multiset(snapshot, ally_pv),
            hand_count=len(ally_pv.hand),
        )
        enemy = PlayerResourceFacts(
            life=int(enemy_pv.life),
            max_life=int(enemy_pv.max_life),
            max_pp=int(enemy_pv.max_pp),
            ep=int(enemy_pv.ep),
            sep=int(enemy_pv.sep),
            extra_pp_lifecycle=_extra_pp_snapshot_lifecycle(enemy_pv),
            deck_count=int(enemy_pv.deck_count),
            crest_identity=_crest_identity(enemy_pv),
            # 对手绝不暴露 known hand id，仅暴露 count。
            known_hand=None,
            hand_count=len(enemy_pv.hand),
        )
        return FutureResourceSignature(perspective=perspective, ally=ally, enemy=enemy)
