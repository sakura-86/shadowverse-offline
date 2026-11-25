# -*- coding: utf-8 -*-
"""破坏（destroy）语义的统一事务构造。

「一张场上的卡被破坏」在本地执行器里由多条路径触发：显式 destroy 效果、
效果伤害致死、攻击结算清理、回合切换清理、倒计时归零。此前每处各写一份
zone / counter / cards 事务，新增路径容易漏掉 ``graveyard_count`` 或
``destroyed_card_ids``（``op_distributed_damage`` 就曾只移除场地不记账）。

这里只统一**事务部分**；事件部分由调用方按 cause 决定
（伤害致死发 ``destroyed_by_damage``，其余发 ``destroy`` + ``move_to_graveyard``），
因为不同 cause 的 source/target/payload 语义不同，强行合并会改变订阅者行为。
"""

from __future__ import annotations

from decision_engine.state.snapshot import CardIdCount, CardView

from .events import EventKind, ExecutionEvent
from .state_ops import ModifyPlayerCounter, RemoveCard, RemoveZoneUid, SetPlayerField, get_player
from .transactions import SnapshotTransaction


def vanishes_on_leave(card: CardView) -> bool:
    """这张卡是否「离场时，使本随从消失」（怨灵等）。

    这类卡的离场表现是**消失**：不进墓地、不计 ``destroyed_card_ids``，并且必须
    发 ``BANISH`` 事件，否则「自己的卡牌在战场上消失时」（``on_self_card_left_field``）
    的订阅者（如 11054110 诺克坦）永远不会响应。
    """
    return card.base_card_id in VANISH_ON_LEAVE_CARD_IDS


def vanish_on_leave_event(player: str, card: CardView, payload: dict[str, object] | None = None) -> ExecutionEvent:
    """「离场即消失」卡的离场事件（替代 destroy + move_to_graveyard）。"""
    return ExecutionEvent(
        EventKind.BANISH,
        player=player,
        source_uid=card.uid,
        target_uid=card.uid,
        payload=dict(payload or {}),
    )


def destroyed_card_transactions(
    snapshot,
    owner: str,
    card: CardView,
    *,
    view=None,
) -> tuple[SnapshotTransaction, ...]:
    """把一张场上的卡送入墓地的完整事务组。

    - 从 ``owner`` 的 field 移除 uid
    - 墓地计数 +1
    - 追加 ``destroyed_card_ids`` 历史（谢幕曲 / 复活类效果依赖）
    - 从 ``cards_by_uid`` 移除卡实例

    ``view`` 允许调用方传入「事务批量应用前的玩家视图」：``op_distributed_damage``
    在同一批事务里连续破坏多张卡，必须基于同一基线视图追加历史，逐张读 snapshot
    只会拿到应用前的状态（此处传 snapshot 即正确，保留参数供后续批量场景复用）。
    """
    player_view = view if view is not None else get_player(snapshot, owner)
    if card.base_card_id in VANISH_ON_LEAVE_CARD_IDS:
        # 「离场时，使本随从消失」：不进墓地、不计 destroyed_card_ids（怨灵等）。
        return (
            RemoveZoneUid(owner, "field", card.uid),
            RemoveCard(card.uid, card),
        )
    return (
        RemoveZoneUid(owner, "field", card.uid),
        ModifyPlayerCounter(owner, "graveyard_count", 1),
        SetPlayerField(
            owner,
            "destroyed_card_ids",
            (*player_view.destroyed_card_ids, CardIdCount(card_id=card.card_id, count=card.style_id)),
        ),
        RemoveCard(card.uid, card),
    )


def _vanish_on_leave_card_ids() -> frozenset[int]:
    """「离场时，使本随从消失」的卡：离场时不进墓地、不计 destroyed_card_ids。

    从卡面文本推导（不硬编码卡 id），模块加载时算一次。
    """
    import json as _json
    from pathlib import Path

    db_path = Path(__file__).resolve().parent.parent / "data/public/card_database.json"
    try:
        cards = _json.loads(db_path.read_text(encoding="utf-8"))["cards"]
    except Exception:
        return frozenset()
    ids: set[int] = set()
    for card in cards:
        desc = str(card.get("normal_desc") or "")
        if "离场时" in desc and "消失" in desc:
            try:
                ids.add(int(card["base_card_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    return frozenset(ids)


VANISH_ON_LEAVE_CARD_IDS = _vanish_on_leave_card_ids()
