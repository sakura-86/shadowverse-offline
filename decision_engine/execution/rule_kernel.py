# -*- coding: utf-8 -*-
"""Phase L4 核心规则内核。

只实现不依赖具体卡牌文本的基础规则；卡牌效果由后续 EventQueue / EffectSpec 接入。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from decision_engine.cards import CardResolver, MasterDataProvider
from decision_engine.simulation.deck_pool import materialize_draw_card, next_synthetic_uid
from decision_engine.state.snapshot import BattleSnapshot, CardIdCount, CardType, CardView, CrestView, ExtraPpView, PlayerView

from .damage_reduction import apply_damage_cut, is_super_evolved_on_own_turn
from .destruction import destroyed_card_transactions, vanish_on_leave_event, vanishes_on_leave
from .events import EventKind, ExecutionEvent
from .evidence import is_evidence_card
from .state_ops import (
    AddCard,
    InsertZoneUid,
    ModifyCardCounter,
    ModifyPlayerCounter,
    MoveZoneUid,
    RemoveCard,
    RemoveZoneUid,
    RemoveZoneUidAt,
    SetCardField,
    SetPlayerField,
    SetSnapshotField,
    get_player,
)
from .transactions import RollbackTransactionLog, SnapshotTransaction, TransactionGroup


MAX_FIELD_SIZE = 5
MAX_HAND_SIZE = 9


def hand_size(view: PlayerView) -> int:
    """手牌**张数**（含协议隐藏手牌）。

    `PlayerView.hand` 只含可观测 uid：replay 里对手手牌是 `unique_id=0` 占位卡，
    被 `_is_hidden_card_placeholder` 过滤，不进入 `hand`。手牌上限与爆牌判定按
    真实张数结算，所以优先用 `hand_count`。

    `hand_count` 由 `state_ops` 的手牌区事务增量维护（插入 +1、移出 -1），因此
    本地预测里「打出牌之后」的张数是准的；只有自由构造的 snapshot（self-play /
    沙盒）不带 `hand_count` 时才退回 `len(hand)`。
    """
    if view.hand_count is not None:
        return view.hand_count
    return len(view.hand)


BATTLE_RESULT_DECK_OUT_WIN = 103
BATTLE_RESULT_DECK_OUT_LOSE = 104
BATTLE_RESULT_VICTORY_CARD_WIN = 109
BATTLE_RESULT_VICTORY_CARD_LOSE = 110
SKYBOUND_ART_KEYS = frozenset(("skybound_art_count",))
EXTRA_PP_GRANT_TURNS = frozenset((1, 6))
EXTRA_PP_CAP = 1
_DEFAULT_CARD_DATABASE = Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"


@lru_cache(maxsize=1)
def _default_card_resolver() -> CardResolver:
    return CardResolver(MasterDataProvider.from_json(_DEFAULT_CARD_DATABASE))


@dataclass(frozen=True)
class DeckoutCard:
    """空牌库再次抽牌时显示的终局卡。

    反编译客户端将其作为独立 VFX prefab（而非可进入 hand/field 的 MasterData
    Card），因此只在 DECKOUT 事件中物化其结构化身份和胜负结果，避免伪造 card_id。
    """

    kind: str
    name: str
    player_result: int
    opponent_result: int

    def to_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "player_result": self.player_result,
            "opponent_result": self.opponent_result,
        }


def materialize_deckout_card(*, is_deck_out_win: bool) -> DeckoutCard:
    """按牌库耗尽胜利状态物化本次抽牌的终局卡与结果码。"""
    if is_deck_out_win:
        return DeckoutCard(
            kind="victory_card",
            name="胜利的卡牌",
            player_result=BATTLE_RESULT_VICTORY_CARD_WIN,
            opponent_result=BATTLE_RESULT_VICTORY_CARD_LOSE,
        )
    return DeckoutCard(
        kind="king_of_death",
        name="亡者之王",
        player_result=BATTLE_RESULT_DECK_OUT_LOSE,
        opponent_result=BATTLE_RESULT_DECK_OUT_WIN,
    )


def deckout_winner_from_results(me_result: int, opponent_result: int) -> str | None:
    """从 DeckOut / VictoryCard 结果码返回绝对胜者。"""
    me_wins = (BATTLE_RESULT_DECK_OUT_WIN, BATTLE_RESULT_VICTORY_CARD_WIN)
    me_loses = (BATTLE_RESULT_DECK_OUT_LOSE, BATTLE_RESULT_VICTORY_CARD_LOSE)
    if me_result in me_wins or opponent_result in me_loses:
        return "me"
    if opponent_result in me_wins or me_result in me_loses:
        return "opponent"
    return None


class RuleKernelError(RuntimeError):
    pass


class InsufficientResourceError(RuleKernelError):
    """资源（PP/EP 等）不足导致操作无法完成。

    与通用 RuleKernelError 区分：这不是执行 bug，而是正常游戏语义
    （如某效果"消耗 N 点能量点"，但当前 PP 不足）。effect 层捕获后
    应回滚整条效果并视为"效果未发动"，不应上报 predict_error。
    """

    pass


@dataclass(frozen=True)
class RuleExecutionResult:
    snapshot: BattleSnapshot
    transaction_log: RollbackTransactionLog
    events: tuple[ExecutionEvent, ...] = ()


# 「对手的回合结束时，破坏本卡牌」类临时卡：来自 MasterData 文本解析的 stable
# 例外表（悬丝傀儡及其变体、安的巨大英灵）。它们在卡主自己的回合结束时不销毁。
OPPONENT_TURN_END_CARD_IDS = frozenset({90071110, 90071111, 90071120, 90071140, 90071141, 90034130, 90034131})

# 「自己的回合结束时，使本随从消失」类卡：消失(banish)与破坏不同——不进墓地、
# 不计入 destroyed_card_ids、不触发谢幕曲（10574110 转动的《命运之轮》、
# 10673110 愚劣的兵器召唤出的临时随从）。
# 「回合结束时从场上消失」而不是「被破坏」的卡：不进墓地、不计 destroyed、
# 不触发谢幕曲。90051130 怨灵 / 90051131 是其变体（文本「离场时，使本随从消失」），
# 回合结束清理若按普通移除处理会多记一条 destroyed + 墓地 +1
# （replay_...1246399 st30）。这张集合是既有模式（10574110/10673110 同处理），
# 后续应收敛为「模板含 turn_end→banish 的卡自动加入」，避免继续堆卡号。
BANISH_AT_TURN_CHANGE_CARD_IDS = frozenset({10574110, 10673110, 90051130, 90051131})


class CoreRuleKernel:
    def _event(self, kind: EventKind, *, source_uid: int | None = None, target_uid: int | None = None, player: str | None = None, **payload: object) -> ExecutionEvent:
        return ExecutionEvent(kind=kind, source_uid=source_uid, target_uid=target_uid, player=player, payload=payload)

    @staticmethod
    def _field_card_ids_payload(snapshot: BattleSnapshot, *, exclude_uid: int | None = None) -> dict[str, object]:
        """记录「使用卡牌瞬间」双方战场的 card_id（**不含**本次使用的卡自身）。

        「自己使用其他卡牌时，若战场上有原始费用为 X 的其他卡牌」（10503210 大游戏世界）
        必须在**使用瞬间**取战场：打出卡的入场曲可能先击杀同费用随从（10914110、
        11072310），等 persistent 订阅段评估时战场已经变化。kernel 在发出 play_card /
        play_spell 事件时把使用瞬间的战场快照进 payload，条件求值优先使用它。

        排除自身必须按 **uid**：`card_id` 是实例卡号，同名的两张卡（replay_...114952
        st15 的两张 10503210）`card_id` 相同，按 card_id 排除会把两张都排掉。
        """
        ids: list[int] = []
        for uid in (*snapshot.me.field, *snapshot.opponent.field):
            if exclude_uid is not None and uid == exclude_uid:
                continue
            card = snapshot.cards_by_uid.get(uid)
            if card is not None:
                ids.append(int(card.card_id))
        return {"field_card_ids": tuple(ids)}

    def _deckout_event(self, snapshot: BattleSnapshot, player: str) -> ExecutionEvent:
        view = get_player(snapshot, player)
        terminal_card = materialize_deckout_card(is_deck_out_win=view.is_deck_out_win)
        return self._event(
            EventKind.DECKOUT,
            player=player,
            deckout_card=terminal_card.to_payload(),
        )

    def _leaving_card_payload(self, card: CardView) -> dict[str, object]:
        payload: dict[str, object] = {
            "card_id": card.card_id,
            "base_card_id": card.base_card_id,
            "name": card.name,
            "card_type": card.card_type.value,
        }
        # 只有存在动态赋予的标记时才附带 supplement_info，保持既有离开卡 payload
        # 的稳定形状（旧测试与 replay 契约按精确字段比较）。
        if card.supplement_info:
            payload["supplement_info"] = dict(card.supplement_info)
        return payload

    def _last_words_event(self, player: str, card: CardView) -> ExecutionEvent:
        return self._event(EventKind.LAST_WORDS, player=player, source_uid=card.uid, leaving_card=self._leaving_card_payload(card))

    def apply_transactions(self, snapshot: BattleSnapshot, transactions: tuple[SnapshotTransaction, ...], events: tuple[ExecutionEvent, ...] = ()) -> RuleExecutionResult:
        log = RollbackTransactionLog()
        try:
            after = log.apply_all(snapshot, transactions)
        except Exception as exc:
            try:
                if log.current is not None:
                    log.rollback()
            finally:
                raise RuleKernelError(str(exc)) from exc
        return RuleExecutionResult(snapshot=after, transaction_log=log, events=events)

    def _spellboost_hand_transactions(self, snapshot: BattleSnapshot, player: str, *, exclude_uid: int | None = None, amount: int = 1) -> tuple[tuple[SnapshotTransaction, ...], tuple[int, ...]]:
        """使用法术后的「使手牌发动 1 次魔力增幅」。

        返回 (transactions, 被增幅的 uid)。调用方要据此为每个 uid 发 SPELLBOOST
        事件 —— 【魔力增幅时】反应效果（10434110 的 +1/+1、10934120 的费用-1）
        订阅的是事件而不是计数器，只加计数器不事件会让它们完全不触发
        （replay_...992331 st26：使用法术后 uid 60 的 sb 涨了、atk/life 没涨）。
        """
        view = get_player(snapshot, player)
        transactions: list[SnapshotTransaction] = []
        boosted: list[int] = []
        for uid in view.hand:
            if uid == exclude_uid:
                continue
            card = snapshot.cards_by_uid[uid]
            if not card.has_spell_boost and card.variable_x <= 0:
                continue
            boosted.append(uid)
            transactions.append(ModifyCardCounter(uid, "spell_boost_count", amount))
            if card.spellboost_reduces_cost:
                transactions.append(SetCardField(uid, "cost", max(0, card.cost - amount)))
            elif card.variable_x > 0 or (card.has_spell_boost and not card.spellboost_reduces_cost):
                variable_x = card.variable_x + amount
                transactions.append(SetCardField(uid, "variable_x", variable_x))
                transactions.append(SetCardField(uid, "supplement_info", {**card.supplement_info, "variable_x": variable_x}))
        return tuple(transactions), tuple(boosted)

    def _record_played_card_transaction(self, snapshot: BattleSnapshot, player: str, uid: int, *, override_card_id: int | None = None) -> SnapshotTransaction | None:
        """记录本局使用过的卡牌到 played_card_ids（通用历史，协议兼容的聚合语义）。

        每次出牌将卡的 card_id 计数 +1；已存在则累加 count，否则追加新 entry。
        这是"本局使用过哪些卡牌"的通用记录来源，供依赖使用历史的条件/效果查询
        （如"使用卡费用包含1~8""使用过N张法术"等）。

        override_card_id：当以激奏/结晶形式使用时，实际使用的是激奏/结晶卡而非本体，
        传入该卡的 card_id 以正确记录玩家实际使用的卡牌。
        """
        record_card_id: int | None = override_card_id
        if record_card_id is None:
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                return None
            record_card_id = card.card_id
        view = get_player(snapshot, player)
        entries = list(view.played_card_ids)
        for index, item in enumerate(entries):
            if item.card_id == record_card_id:
                entries[index] = CardIdCount(card_id=item.card_id, count=item.count + 1)
                break
        else:
            entries.append(CardIdCount(card_id=record_card_id, count=1))
        return SetPlayerField(player, "played_card_ids", tuple(entries))

    def _record_entered_field_transaction(self, snapshot: BattleSnapshot, player: str, base_card_id: int) -> SnapshotTransaction | None:
        """记录本局进入战场的随从 base_card_id 计数到 entered_field_card_ids。

        对称 played_card_ids，用 base_card_id 作为键（同名判定按 base_card_id，
        token 的 card_id 可能因实例不同而变）。供"本局其他同名随从入场数>=N"
        等条件查询（如沉溺的实验体：本局其他实验体入场>=5 则 +3/+3）。
        """
        if not base_card_id:
            return None
        view = get_player(snapshot, player)
        entries = list(view.entered_field_card_ids)
        for index, item in enumerate(entries):
            if item.card_id == base_card_id:
                entries[index] = CardIdCount(card_id=item.card_id, count=item.count + 1)
                break
        else:
            entries.append(CardIdCount(card_id=base_card_id, count=1))
        return SetPlayerField(player, "entered_field_card_ids", tuple(entries))

    def _materialize_hand_card(self, snapshot: BattleSnapshot, uid: int, player: str) -> CardView:
        card = snapshot.cards_by_uid[uid]
        materialized = materialize_draw_card(card.card_id, snapshot, uid=uid, resolver=_default_card_resolver())
        if materialized is None:
            raise RuleKernelError(f"cannot materialize hand card: {card.card_id}")
        # 恢复奥义相关 supplement_info：如果原始卡有 skybound_art_count，
        # 回到手牌时初始化为当前回合数
        original_si = card.supplement_info
        if "skybound_art_count" in original_si or "super_skybound_art_count" in original_si:
            view = get_player(snapshot, player)
            turn = view.turn
            new_si = dict(materialized.supplement_info) if materialized.supplement_info else {}
            if "skybound_art_count" in original_si:
                new_si["skybound_art_count"] = turn
            if "super_skybound_art_count" in original_si:
                new_si["super_skybound_art_count"] = turn
            materialized = replace(materialized, supplement_info=new_si)
        return materialized

    def _extra_pp_after_turn_start(self, view: PlayerView, next_turn: int) -> ExtraPpView | None:
        if view.is_first_side:
            # 先手方不获得额外 PP，保持原值不变
            # （协议中先手方可能有 extra_pp=0 的对象，也可能是 None）
            return view.extra_pp
        current = view.extra_pp or ExtraPpView()
        held = min(max(current.extra_pp, 0), EXTRA_PP_CAP)
        if next_turn in EXTRA_PP_GRANT_TURNS:
            held = min(held + 1, EXTRA_PP_CAP)
        return replace(
            current,
            extra_pp=held,
            preparation_extra_pp=0,
            extra_pp_state=0,
            restore_extra_pp_turn=6,
            is_used_extra_pp_this_turn=False,
        )

    def consume_pp(self, snapshot: BattleSnapshot, player: str, amount: int) -> RuleExecutionResult:
        if amount < 0:
            raise RuleKernelError("pp cost must be non-negative")
        view = get_player(snapshot, player)
        if view.pp < amount:
            # 效果内的 pay_pp：PP 不足是正常游戏语义（效果不发动），非执行 bug。
            raise InsufficientResourceError("not enough pp")
        return self.apply_transactions(snapshot, tuple(self._pay_pp_transactions(view, player, amount)), tuple(self._pay_pp_events(view, player, amount)))

    def _pay_pp_transactions(self, view: PlayerView, player: str, amount: int) -> list[SnapshotTransaction]:
        transactions: list[SnapshotTransaction] = [ModifyPlayerCounter(player, "pp", -amount)]
        if self._uses_extra_pp(view, amount):
            extra_pp = view.extra_pp
            transactions.append(SetPlayerField(player, "extra_pp", replace(extra_pp, extra_pp=max(extra_pp.extra_pp - 1, 0), preparation_extra_pp=max(extra_pp.preparation_extra_pp - 1, 0), extra_pp_state=2, is_used_extra_pp_this_turn=True)))
        return transactions

    def _pay_pp_events(self, view: PlayerView, player: str, amount: int, source_uid: int | None = None) -> list[ExecutionEvent]:
        events = [self._event(EventKind.CONSUME_PP, player=player, source_uid=source_uid, amount=amount)]
        if self._uses_extra_pp(view, amount):
            events.append(self._event(EventKind.CONSUME_EXTRA_PP, player=player, source_uid=source_uid, amount=1))
        return events

    def _uses_extra_pp(self, view: PlayerView, amount: int) -> bool:
        extra_pp = view.extra_pp
        if extra_pp is None or extra_pp.preparation_extra_pp <= 0:
            return False
        if amount <= view.pp - extra_pp.preparation_extra_pp:
            return False
        return True

    def consume_ep(self, snapshot: BattleSnapshot, player: str, amount: int = 1) -> RuleExecutionResult:
        if amount < 0:
            raise RuleKernelError("ep cost must be non-negative")
        view = get_player(snapshot, player)
        if view.ep < amount:
            raise RuleKernelError("not enough ep")
        return self.apply_transactions(snapshot, (ModifyPlayerCounter(player, "ep", -amount),), (self._event(EventKind.CONSUME_EP, player=player, amount=amount),))

    def consume_sep(self, snapshot: BattleSnapshot, player: str, amount: int = 1) -> RuleExecutionResult:
        if amount < 0:
            raise RuleKernelError("sep cost must be non-negative")
        view = get_player(snapshot, player)
        if view.sep < amount:
            raise RuleKernelError("not enough sep")
        return self.apply_transactions(snapshot, (ModifyPlayerCounter(player, "sep", -amount),), (self._event(EventKind.CONSUME_SEP, player=player, amount=amount),))

    def play_card_from_hand(self, snapshot: BattleSnapshot, player: str, uid: int, cost: int) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if uid not in view.hand:
            raise RuleKernelError("card is not in hand")
        if len(view.field) >= MAX_FIELD_SIZE:
            raise RuleKernelError("field is full")
        if view.pp < cost:
            raise RuleKernelError("not enough pp")
        hand_index = view.hand.index(uid)
        card = snapshot.cards_by_uid.get(uid)
        transactions = [
            *self._pay_pp_transactions(view, player, cost),
            MoveZoneUid(player, "hand", hand_index, player, "field"),
            ModifyPlayerCounter(player, "play_count", 1),
        ]
        played_tx = self._record_played_card_transaction(snapshot, player, uid)
        if played_tx is not None:
            transactions.append(played_tx)
        if card is not None and card.card_type.is_follower:
            transactions.extend((ModifyPlayerCounter(player, "rally", 1), ModifyPlayerCounter(player, "followers_entered_battlefield", 1)))
            entered_tx = self._record_entered_field_transaction(snapshot, player, card.base_card_id)
            if entered_tx is not None:
                transactions.append(entered_tx)
        transactions.extend(self._earth_sigil_enter_transactions(snapshot, player, uid))
        return self.apply_transactions(
            snapshot,
            tuple(transactions),
            (
                *self._pay_pp_events(view, player, cost, source_uid=uid),
                self._event(EventKind.PLAY_CARD, player=player, source_uid=uid, **self._field_card_ids_payload(snapshot, exclude_uid=uid)),
                self._event(EventKind.ENTER_FIELD, player=player, source_uid=uid),
            ),
        )

    def play_spell_from_hand(self, snapshot: BattleSnapshot, player: str, uid: int, cost: int) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if uid not in view.hand:
            raise RuleKernelError("card is not in hand")
        if view.pp < cost:
            raise RuleKernelError("not enough pp")
        hand_index = view.hand.index(uid)
        spellboost_transactions, spellboost_uids = self._spellboost_hand_transactions(snapshot, player, exclude_uid=uid)
        transactions: tuple[SnapshotTransaction, ...] = (
            *self._pay_pp_transactions(view, player, cost),
            RemoveZoneUidAt(player, "hand", hand_index, uid),
            ModifyPlayerCounter(player, "play_count", 1),
            ModifyPlayerCounter(player, "graveyard_count", 1),
            RemoveCard(uid, snapshot.cards_by_uid[uid]),
            *spellboost_transactions,
        )
        played_tx = self._record_played_card_transaction(snapshot, player, uid)
        if played_tx is not None:
            transactions = (*transactions, played_tx)
        return self.apply_transactions(
            snapshot,
            transactions,
            (
                *self._pay_pp_events(view, player, cost, source_uid=uid),
                self._event(EventKind.PLAY_SPELL, player=player, source_uid=uid, source_base_card_id=snapshot.cards_by_uid[uid].base_card_id, **self._field_card_ids_payload(snapshot, exclude_uid=uid)),
                self._event(EventKind.MOVE_TO_GRAVEYARD, player=player, source_uid=uid),
                *(
                    self._event(EventKind.SPELLBOOST, player=player, source_uid=uid, target_uid=boosted_uid, amount=1)
                    for boosted_uid in spellboost_uids
                ),
            ),
        )

    def play_enhanced_from_hand(self, snapshot: BattleSnapshot, player: str, uid: int, cost: int) -> RuleExecutionResult:
        card = snapshot.cards_by_uid.get(uid)
        if card is None:
            raise RuleKernelError("card not found")
        if card.card_type.is_spell:
            result = self.play_spell_from_hand(snapshot, player, uid, cost)
        else:
            result = self.play_card_from_hand(snapshot, player, uid, cost)
        return RuleExecutionResult(
            snapshot=result.snapshot,
            transaction_log=result.transaction_log,
            events=(*result.events, self._event(EventKind.ENHANCE, player=player, source_uid=uid, cost=cost)),
        )

    def play_accelerated_from_hand(self, snapshot: BattleSnapshot, player: str, uid: int, cost: int, *, recorded_card_id: int | None = None) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if uid not in view.hand:
            raise RuleKernelError("card is not in hand")
        if view.pp < cost:
            raise RuleKernelError("not enough pp")
        hand_index = view.hand.index(uid)
        spellboost_transactions, spellboost_uids = self._spellboost_hand_transactions(snapshot, player, exclude_uid=uid)
        transactions: tuple[SnapshotTransaction, ...] = (
            *self._pay_pp_transactions(view, player, cost),
            RemoveZoneUidAt(player, "hand", hand_index, uid),
            ModifyPlayerCounter(player, "play_count", 1),
            ModifyPlayerCounter(player, "graveyard_count", 1),
            RemoveCard(uid, snapshot.cards_by_uid[uid]),
            *spellboost_transactions,
        )
        # 激奏以法术卡形式使用：本局使用历史记录激奏法术卡 id（而非本体随从 id）
        played_tx = self._record_played_card_transaction(snapshot, player, uid, override_card_id=recorded_card_id)
        if played_tx is not None:
            transactions = (*transactions, played_tx)
        return self.apply_transactions(
            snapshot,
            transactions,
            (
                *self._pay_pp_events(view, player, cost, source_uid=uid),
                self._event(EventKind.ACCELERATE, player=player, source_uid=uid, cost=cost),
                self._event(EventKind.PLAY_SPELL, player=player, source_uid=uid, source_base_card_id=snapshot.cards_by_uid[uid].base_card_id, **self._field_card_ids_payload(snapshot, exclude_uid=uid)),
                self._event(EventKind.MOVE_TO_GRAVEYARD, player=player, source_uid=uid),
                *(
                    self._event(EventKind.SPELLBOOST, player=player, source_uid=uid, target_uid=boosted_uid, amount=1)
                    for boosted_uid in spellboost_uids
                ),
            ),
        )

    def play_crystallized_from_hand(self, snapshot: BattleSnapshot, player: str, uid: int, cost: int, *, recorded_card_id: int | None = None, crystal_form_fields: "Mapping[str, Any] | None" = None, recorded_card: "CardView | None" = None) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if uid not in view.hand:
            raise RuleKernelError("card is not in hand")
        if len(view.field) >= MAX_FIELD_SIZE:
            raise RuleKernelError("field is full")
        if view.pp < cost:
            raise RuleKernelError("not enough pp")
        hand_index = view.hand.index(uid)
        # 结晶打出：打出的实例本身就是结晶护符（如 10962122），并非本体随从进场后变身。
        # crystal_form_fields 由 predictor 从结晶护符 template 提供（base_card_id/card_id/
        # name/countdown 等），使留场实例的后续谢幕曲/吟唱倒计数/启动效果按结晶护符结算。
        #
        # recorded_card：replay 里协议会给结晶出的护符分配**新的实例 uid**，
        # 并直接给出该实例的完整字段（replay_...1129510 st5：手牌 uid=60 打出后
        # 是 uid=85 base=10962122 cost=1）。那种情况下手牌实例要整个换掉，
        # 并以录制实例的字段为准（name / cost / countdown 都是协议侧的）。
        if recorded_card is not None and recorded_card.uid != uid:
            recorded_uid = recorded_card.uid
            new_card = recorded_card
            source_card = snapshot.cards_by_uid[uid]
            played_tx = self._record_played_card_transaction(snapshot, player, uid, override_card_id=recorded_card_id)
            transactions: tuple[SnapshotTransaction, ...] = (
                *self._pay_pp_transactions(view, player, cost),
                RemoveCard(uid, source_card),
                RemoveZoneUid(player, "hand", uid),
                AddCard(new_card),
                InsertZoneUid(player, "field", recorded_uid),
                ModifyPlayerCounter(player, "play_count", 1),
            )
            if played_tx is not None:
                transactions = (*transactions, played_tx)
            return self.apply_transactions(
                snapshot,
                transactions,
                (
                    *self._pay_pp_events(view, player, cost, source_uid=recorded_uid),
                    self._event(EventKind.CRYSTAL, player=player, source_uid=recorded_uid, cost=cost),
                    self._event(EventKind.PLAY_CARD, player=player, source_uid=recorded_uid),
                    self._event(EventKind.ENTER_FIELD, player=player, source_uid=recorded_uid),
                ),
            )
        form_transactions: tuple[SnapshotTransaction, ...] = ()
        if crystal_form_fields:
            form_transactions = tuple(
                SetCardField(uid, field_name, field_value)
                for field_name, field_value in crystal_form_fields.items()
            )
        transactions: tuple[SnapshotTransaction, ...] = (
            *self._pay_pp_transactions(view, player, cost),
            MoveZoneUid(player, "hand", hand_index, player, "field"),
            SetCardField(uid, "card_type", CardType.COUNTDOWN_AMULET),
            *form_transactions,
            ModifyPlayerCounter(player, "play_count", 1),
        )
        # 结晶以护符卡形式使用：本局使用历史记录结晶卡 id（而非本体随从 id）
        played_tx = self._record_played_card_transaction(snapshot, player, uid, override_card_id=recorded_card_id)
        if played_tx is not None:
            transactions = (*transactions, played_tx)
        return self.apply_transactions(
            snapshot,
            transactions,
            (
                *self._pay_pp_events(view, player, cost, source_uid=uid),
                self._event(EventKind.CRYSTAL, player=player, source_uid=uid, cost=cost),
                self._event(EventKind.PLAY_CARD, player=player, source_uid=uid),
                self._event(EventKind.ENTER_FIELD, player=player, source_uid=uid),
            ),
        )

    def summon_card(self, snapshot: BattleSnapshot, player: str, card: CardView) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if len(view.field) >= MAX_FIELD_SIZE:
            return RuleExecutionResult(
                snapshot=snapshot,
                transaction_log=RollbackTransactionLog(root=snapshot, current=snapshot),
                events=(self._event(EventKind.SUMMON_FAILED_FIELD_FULL, player=player, source_uid=card.uid),),
            )
        transactions: list[SnapshotTransaction] = [AddCard(card), InsertZoneUid(player, "field", card.uid)]
        if card.card_type == CardType.FOLLOWER:
            transactions.extend((ModifyPlayerCounter(player, "rally", 1), ModifyPlayerCounter(player, "followers_entered_battlefield", 1)))
            entered_tx = self._record_entered_field_transaction(snapshot, player, card.base_card_id)
            if entered_tx is not None:
                transactions.append(entered_tx)
        transactions.extend(self._earth_sigil_enter_transactions(snapshot, player, card.uid, entering_card=card))
        return self.apply_transactions(
            snapshot,
            tuple(transactions),
            (self._event(EventKind.SUMMON, player=player, source_uid=card.uid), self._event(EventKind.ENTER_FIELD, player=player, source_uid=card.uid)),
        )

    def _earth_sigil_enter_transactions(self, snapshot: BattleSnapshot, player: str, uid: int, *, entering_card: CardView | None = None) -> list[SnapshotTransaction]:
        card = entering_card or snapshot.cards_by_uid[uid]
        if not card.is_earth_sigil:
            return []
        view = get_player(snapshot, player)
        existing = [existing_uid for existing_uid in view.field if existing_uid != uid and snapshot.cards_by_uid[existing_uid].is_earth_sigil]
        if not existing:
            return []
        total_stack = max(card.stack, 1) + sum(max(snapshot.cards_by_uid[existing_uid].stack, 1) for existing_uid in existing)
        transactions: list[SnapshotTransaction] = []
        for existing_uid in existing:
            transactions.extend((RemoveZoneUid(player, "field", existing_uid), RemoveCard(existing_uid, snapshot.cards_by_uid[existing_uid])))
        transactions.append(SetCardField(uid, "stack", total_stack))
        return transactions

    def perform_special_action(self, snapshot: BattleSnapshot, player: str, uid: int) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if uid not in view.field:
            raise RuleKernelError("card is not on field")
        card = snapshot.cards_by_uid[uid]
        if not card.can_special_action:
            raise RuleKernelError("card has no special action")
        return self.apply_transactions(snapshot, (), (self._event(EventKind.SPECIAL_ACTION, player=player, source_uid=uid),))

    def end_turn(self, snapshot: BattleSnapshot, player: str, *, defer_removal_uids: frozenset[int] = frozenset()) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        transactions: list[SnapshotTransaction] = [SetSnapshotField("is_turn_ended", True)]
        if view.extra_pp is not None:
            # 「额外 PP」在回合结束时归还：activate_extra_pp 把 1 点 extra_pp 变成
            # 本回合的 +1 PP 并记 preparation_extra_pp=1；这部分 PP 若没用掉，回合
            # 结束要存回 extra_pp（录像证据：replay_...111334 st43 的
            # pp 1->0 / extra_pp 0->1 / extra_pp_state 1->0 / preparation 1->0，
            # 响应流里是 BattleResponseExtraPP；...9697186 st38 为 pp 6->5 同样 +1）。
            stored = min(max(view.pp, 0), max(view.extra_pp.preparation_extra_pp, 0))
            if stored > 0:
                transactions.extend((
                    ModifyPlayerCounter(player, "pp", -stored),
                    SetPlayerField(player, "extra_pp", replace(
                        view.extra_pp,
                        extra_pp=min(view.extra_pp.extra_pp + stored, EXTRA_PP_CAP),
                        preparation_extra_pp=0,
                        extra_pp_state=0,
                    )),
                ))
            else:
                transactions.append(SetPlayerField(player, "extra_pp", replace(view.extra_pp, preparation_extra_pp=0)))
        events: list[ExecutionEvent] = [self._event(EventKind.TURN_END, player=player)]
        destroyed_cards: dict[str, list[CardView]] = {"me": [], "opponent": []}
        # 回合结束离场（破坏或消失）的 uid：后续攻击次数清零不能再触碰这些卡。
        removed_uids: set[int] = set()
        for owner in ("me", "opponent"):
            for uid in get_player(snapshot, owner).field:
                card = snapshot.cards_by_uid[uid]
                # 「回合结束时离场」只在**持有者自己的回合结束**时发生；OPPONENT_TURN_END_CARD_IDS
                # 里的卡相反，只在对手回合结束时离场。
                # 录制证据（三处一致）：
                #   replay_...50009671 frame57 我方 10673110 在对手回合结束仍保留；
                #   replay_...50009671 frame66 同一张卡在我方回合结束才消失；
                #   replay_...463579 frame100 对手 10804110 在我方回合结束仍保留。
                # 原条件写成「任何一方回合结束都移除双方临时卡」，会误删另外一侧的卡。
                _turn_change_leaves = card.is_remove_field_at_turn_change and (
                    (owner == player and card.base_card_id not in OPPONENT_TURN_END_CARD_IDS)
                    or (owner != player and card.base_card_id in OPPONENT_TURN_END_CARD_IDS)
                )
                if uid in defer_removal_uids:
                    # 协议字段 `is_remove_field_at_turn_change` 只说明「回合结束时会离场」这个
                    # **结果**，不表示离场时机在 turn_end 事件之前。这类卡的离场其实是它自己的
                    # 「回合结束时破坏本卡牌」模板效果，必须和其它 turn_end 效果一起按登记顺序
                    # 结算（10903210 混沌监狱：纹章 10964112 的条件要先看到它还在场上，
                    # 否则 `field_max_original_cost>=6` 会塌，…692618 st156 少 1 点伤害+1 点回复）。
                    _turn_change_leaves = False
                if _turn_change_leaves:
                    if card.base_card_id in BANISH_AT_TURN_CHANGE_CARD_IDS:
                        # 「使本随从消失」：离场但不进墓地、不计 destroyed、不触发谢幕曲。
                        removed_uids.add(uid)
                        transactions.extend((
                            RemoveZoneUid(owner, "field", uid),
                            RemoveCard(uid, card),
                        ))
                        events.append(self._event(EventKind.BANISH, player=owner, source_uid=uid))
                        continue
                    destroyed_cards[owner].append(card)
                    removed_uids.add(uid)
                    transactions.extend((
                        RemoveZoneUid(owner, "field", uid),
                        ModifyPlayerCounter(owner, "graveyard_count", 1),
                        RemoveCard(uid, card),
                    ))
                    events.extend((
                        self._event(EventKind.DESTROY, player=owner, source_uid=uid),
                        self._event(EventKind.MOVE_TO_GRAVEYARD, player=owner, source_uid=uid),
                    ))
                    if card.has_last_words:
                        events.append(self._last_words_event(owner, card))
        for owner, cards in destroyed_cards.items():
            if not cards:
                continue
            history = get_player(snapshot, owner).destroyed_card_ids
            transactions.append(SetPlayerField(owner, "destroyed_card_ids", (*history, *(CardIdCount(card_id=card.card_id, count=card.style_id) for card in cards))))
        # 回合结束时，手牌上「本回合有效」的费用修正（buff.cost）回滚到
        # 本回合开始时的基线；跨回合保留的永久减费（如 10974120 的 -3）
        # 因为基线就是该值而保持不变。基线由 start_turn 写入，replay 由工具注入。
        for owner in ("me", "opponent"):
            for uid in get_player(snapshot, owner).hand:
                hand_card = snapshot.cards_by_uid.get(uid)
                if hand_card is None or hand_card.buff is None:
                    continue
                if "cost_buff_base" not in hand_card.supplement_info:
                    continue
                base = int(hand_card.supplement_info["cost_buff_base"])
                cleaned = {key: value for key, value in hand_card.supplement_info.items() if key != "cost_buff_base"}
                if base == hand_card.buff.cost:
                    transactions.append(SetCardField(uid, "supplement_info", cleaned))
                    continue
                restored_cost = hand_card.cost + base - hand_card.buff.cost
                if hand_card.cost == 0 and restored_cost > 0:
                    # 手牌费用已在下限 0：本回合的减费被 clamp 吞掉，回滚不应反过来
                    # 抬高费用（录像证据：0 费卡带 buff.cost=-1，回合结束仍是 0）。
                    restored_cost = 0
                transactions.extend((
                    SetCardField(uid, "cost", max(0, restored_cost)),
                    SetCardField(uid, "buff", replace(hand_card.buff, cost=base)),
                    SetCardField(uid, "supplement_info", cleaned),
                ))
        # 回合结束时回滚「回合结束前 +X/+Y」类临时身材修正（stat_buff duration=turn_end）。
        # 临时增量在 op_buff_stat 里以 supplement_info 标记记录，不能当成永久 buff，
        # 否则回合结束后的 after 快照会保留偏高的 atk/life。
        for owner in ("me", "opponent"):
            for uid in get_player(snapshot, owner).field:
                card = snapshot.cards_by_uid.get(uid)
                if card is None:
                    continue
                bonus_atk = int(card.supplement_info.get("temp_atk_bonus_until_turn_end", 0) or 0)
                bonus_life = int(card.supplement_info.get("temp_life_bonus_until_turn_end", 0) or 0)
                if not bonus_atk and not bonus_life:
                    continue
                cleaned = {
                    key: value for key, value in card.supplement_info.items()
                    if key not in {"temp_atk_bonus_until_turn_end", "temp_life_bonus_until_turn_end"}
                }
                if bonus_atk:
                    transactions.append(SetCardField(uid, "atk", max(0, card.atk - bonus_atk)))
                if bonus_life:
                    transactions.extend((
                        SetCardField(uid, "life", max(0, card.life - bonus_life)),
                        SetCardField(uid, "max_life", max(0, card.max_life - bonus_life)),
                    ))
                transactions.append(SetCardField(uid, "supplement_info", cleaned))
        # 回合结束时，当前玩家场上随从的攻击次数清零；下一次 start_turn 再恢复。        # 本次已被离场处理（is_remove_field_at_turn_change）的卡不再需要清零；
        # 但「对手回合结束时才移除」的卡（如悬丝傀儡 90071110）本回合仍留场，
        # 必须清零，否则 after 快照会保留可攻击次数。
        removed_uids |= {card.uid for cards in destroyed_cards.values() for card in cards}
        for uid in view.field:
            card = snapshot.cards_by_uid.get(uid)
            if card is None or not card.card_type.is_follower or uid in removed_uids:
                continue
            transactions.extend((
                SetCardField(uid, "attacks_left", 0),
                SetCardField(uid, "can_attack", False),
            ))
            if card.attack_limit > 1 and card.buff is not None and card.buff.attack_limit > 1:
                # 「本回合可攻击 N 次」在回合结束后回到 1。只有 buff.attack_limit
                # 提供额外次数时才重置：卡面自带「1回合可以攻击2/3次」的静态
                # attack_limit_N 落在 attack_limit 字段而 buff 为 0，录像里跨回合
                # 保留（10614110 2→2 / 10924120 3→3）。
                # 例外：由「获得『1回合可以攻击 N 次』」这类**关键字**赋予的，
                # 直到离场都有效（10834110 被 90034350 赋予 2 次、10654110 超进化
                # 获得 3 次，录像里 turn_end 前后都保持原值）。这类在赋予时由
                # `_attack_limit_transactions(permanent=True)` 写入 supplement_info
                # 标记；本回合临时加成没有该标记，仍然重置。
                permanent_limit = (card.supplement_info or {}).get("attack_limit_permanent")
                if not permanent_limit:
                    transactions.append(SetCardField(uid, "attack_limit", 1))
        # 牌组内持续效果窗口（对称 TURN_START_DECK）：处理"在牌组中发动，回合结束时瞬念召唤"类效果（如泽勒尔）。
        events.append(self._event(EventKind.TURN_END_DECK, player=player))
        return self.apply_transactions(snapshot, tuple(transactions), tuple(events))

    def _draw_card_transactions(self, snapshot: BattleSnapshot, player: str, card: CardView | None, *, turn_override: int | None = None) -> tuple[SnapshotTransaction, ...]:
        view = get_player(snapshot, player)
        if view.deck_count <= 0:
            opponent = "opponent" if player == "me" else "me"
            terminal_card = materialize_deckout_card(is_deck_out_win=view.is_deck_out_win)
            return (
                SetSnapshotField("is_turn_ended", True),
                SetPlayerField(player, "result", terminal_card.player_result),
                SetPlayerField(opponent, "result", terminal_card.opponent_result),
            )
        transactions: list[SnapshotTransaction] = [ModifyPlayerCounter(player, "deck_count", -1)]
        if card is None:
            if hand_size(view) >= MAX_HAND_SIZE:
                transactions.append(ModifyPlayerCounter(player, "graveyard_count", 1))
            return tuple(transactions)
        card_exists = card.uid in snapshot.cards_by_uid
        if hand_size(view) >= MAX_HAND_SIZE:
            if card_exists:
                transactions.extend((ModifyPlayerCounter(player, "graveyard_count", 1), RemoveCard(card.uid, card)))
            else:
                transactions.extend((AddCard(card), ModifyPlayerCounter(player, "graveyard_count", 1), RemoveCard(card.uid, card)))
            return tuple(transactions)
        if card_exists:
            transactions.append(InsertZoneUid(player, "hand", card.uid))
        else:
            transactions.extend((AddCard(card), InsertZoneUid(player, "hand", card.uid)))
        # 抽到的卡如果有奥义计数且还是初始值 0，设为当前回合数
        if "skybound_art_count" in card.supplement_info and card.supplement_info["skybound_art_count"] == 0:
            turn = turn_override if turn_override is not None else view.turn
            if turn > 0:
                new_si = {**card.supplement_info, "skybound_art_count": turn}
                transactions.append(SetCardField(card.uid, "supplement_info", new_si))
        return tuple(transactions)

    def start_turn(
        self,
        snapshot: BattleSnapshot,
        player: str,
        drawn_card: CardView | None = None,
        *,
        defer_draw: bool = False,
    ) -> RuleExecutionResult:
        opponent = "opponent" if player == "me" else "me"
        view = get_player(snapshot, player)
        next_turn = view.turn + 1
        next_max_pp = min(10, view.max_pp + 1)
        transactions: list[SnapshotTransaction] = [
            SetSnapshotField("is_ally_turn", player == "me"),
            SetSnapshotField("is_turn_ended", False),
            SetPlayerField(player, "turn", next_turn),
            SetPlayerField(player, "max_pp", next_max_pp),
            SetPlayerField(player, "pp", next_max_pp),
            SetPlayerField(player, "play_count", 0),
            SetPlayerField(player, "attack_count", 0),
            SetPlayerField(player, "manual_evolve_count", 0),
            SetPlayerField(player, "is_evolved_this_turn", False),
            SetPlayerField(player, "extra_pp", self._extra_pp_after_turn_start(view, next_turn)),
            # 归档"上一个己方回合随从是否攻击过对方主战者"：把上一回合累积的
            # this_turn 标记转存为 previous_turn，并清空 this_turn 供新回合累积。
            SetPlayerField(player, "follower_attacked_leader_previous_turn", view.follower_attacked_leader_this_turn),
            SetPlayerField(player, "follower_attacked_leader_this_turn", False),
            SetPlayerField(opponent, "play_count", 0),
            SetPlayerField(opponent, "attack_count", 0),
            SetPlayerField(opponent, "manual_evolve_count", 0),
            SetPlayerField(opponent, "is_evolved_this_turn", False),
        ]
        # 记录「回合开始前」的手牌费用修正基线：回合结束时（end_turn）回滚到该值，
        # 使「回合结束前费用-1」类临时减费失效、跨回合永久减费保留。
        for owner in ("me", "opponent"):
            for hand_uid in get_player(snapshot, owner).hand:
                hand_card = snapshot.cards_by_uid.get(hand_uid)
                if hand_card is None:
                    continue
                base = hand_card.buff.cost if hand_card.buff is not None else 0
                transactions.append(SetCardField(hand_uid, "supplement_info", {**hand_card.supplement_info, "cost_buff_base": base}))
        # 清零双方的 per_turn 效果触发计数
        if view.effect_trigger_counts:
            transactions.append(SetPlayerField(player, "effect_trigger_counts", {}))
        if get_player(snapshot, opponent).effect_trigger_counts:
            transactions.append(SetPlayerField(opponent, "effect_trigger_counts", {}))
        # 非活跃玩家的 extra_pp: 只重置 is_used_extra_pp_this_turn
        opponent_view = get_player(snapshot, opponent)
        if opponent_view.extra_pp is not None and opponent_view.extra_pp.is_used_extra_pp_this_turn:
            transactions.append(SetPlayerField(opponent, "extra_pp", replace(opponent_view.extra_pp, is_used_extra_pp_this_turn=False)))
        events: list[ExecutionEvent] = [self._event(EventKind.TURN_START, player=player), self._event(EventKind.REFILL_PP, player=player, amount=next_max_pp)]
        field_last_words: list[ExecutionEvent] = []
        crest_last_words: list[ExecutionEvent] = []
        destroyed_cards: dict[str, list[CardView]] = {"me": [], "opponent": []}
        for uid in view.field:
            card = snapshot.cards_by_uid[uid]
            # 判断 countdown 是否归零（即将销毁）
            will_destroy = card.countdown > 0 and card.countdown - 1 <= 0
            if not will_destroy:
                # 回合开始恢复场面随从攻击能力（仅对不销毁的卡）
                if card.card_type.is_follower:
                    base_attack_limit = (card.buff.attack_limit if card.buff is not None and card.buff.attack_limit else 0) or 1
                    attack_limit = base_attack_limit
                    transactions.extend((
                        SetCardField(uid, "attack_limit", base_attack_limit),
                        SetCardField(uid, "can_attack", not card.has_cant_attack),
                        SetCardField(uid, "can_attack_leader", not card.has_cant_attack),
                        SetCardField(uid, "attacks_left", attack_limit),
                    ))
                    if card.supplement_info.get("cant_attack_until_turn_end"):
                        # 11004112 等「回合结束前无法攻击随从或主战者」：回合开始时恢复。
                        cleaned = {key: value for key, value in card.supplement_info.items() if key != "cant_attack_until_turn_end"}
                        transactions.extend((
                            SetCardField(uid, "has_cant_attack", False),
                            SetCardField(uid, "can_attack", True),
                            SetCardField(uid, "can_attack_leader", True),
                            SetCardField(uid, "supplement_info", cleaned),
                        ))
                transactions.append(SetCardField(uid, "activation_used_this_turn", False))
            if card.countdown <= 0:
                continue
            next_countdown = card.countdown - 1
            events.append(self._event(EventKind.COUNTDOWN, player=player, source_uid=uid, amount=1, source_card=self._leaving_card_payload(card), next_countdown=next_countdown))
            if next_countdown <= 0:
                destroyed_cards[player].append(card)
                transactions.extend((RemoveZoneUid(player, "field", uid), ModifyPlayerCounter(player, "graveyard_count", 1), RemoveCard(uid, card)))
                # 倒计时归零离场在规则上算作"被破坏(destroyed)"：补发 DESTROY 事件，
                # 使依赖"自己护符被破坏时"(on_self_amulet_destroyed) 的反应式效果
                # (如护符教信仰纹章 10664122) 能在护符自然到期时触发。
                # 与正常破坏路径 (end_turn 的 is_remove_field_at_turn_change) 一致：
                # DESTROY 先于 MOVE_TO_GRAVEYARD；此处护符已被 RemoveCard 移除，
                # 故 DESTROY 必须带 leaving_card payload (含 card_type)，供
                # template_event_handler._destroyed_card_is_amulet 判定被破坏卡是否护符类。
                # target_uid=uid 作为 destroyed_uid，player 为拥有者使 subscriber_side 归属对齐。
                # 随从若也走此路径，其 DESTROY 不含 amulet card_type，scope 判定自然不触发
                # on_self_amulet_destroyed，故不会给随从误发护符破坏语义。
                events.append(self._event(
                    EventKind.DESTROY, player=player, source_uid=uid, target_uid=uid,
                    leaving_card=self._leaving_card_payload(card),
                ))
                events.append(self._event(EventKind.MOVE_TO_GRAVEYARD, player=player, source_uid=uid))
                last_words = self._last_words_event(player, card)
                field_last_words.append(replace(last_words, payload={**last_words.payload, "turn_start_phase": True}))
            else:
                transactions.append(SetCardField(uid, "countdown", next_countdown))
        # === 纹章 countdown 递减 ===
        remaining_crests: list[CrestView] = []
        for crest in view.crests:
            if crest.countdown > 0:
                next_cd = crest.countdown - 1
                if next_cd <= 0:
                    # 倒计时归零的纹章先离开纹章区；其谢幕曲在第二阶段的纹章窗口结算。
                    crest_last_words.append(self._event(EventKind.LAST_WORDS, player=player, source_uid=crest.unique_id or 0,
                                                        source_card={"base_card_id": crest.card_id, "card_id": crest.card_id, "name": "", "has_guard": False}, turn_start_phase=True))
                else:
                    remaining_crests.append(replace(crest, countdown=next_cd))
            else:
                remaining_crests.append(crest)
        if tuple(remaining_crests) != view.crests:
            transactions.append(SetPlayerField(player, "crests", tuple(remaining_crests)))
        for uid in view.hand:
            card = snapshot.cards_by_uid[uid]
            skybound_updates = {key: value + 1 for key, value in card.supplement_info.items() if key in SKYBOUND_ART_KEYS}
            supplement_info = {**card.supplement_info, **skybound_updates}
            if "fusion_used_this_turn" in supplement_info:
                supplement_info.pop("fusion_used_this_turn")
            if supplement_info != card.supplement_info:
                transactions.append(SetCardField(uid, "supplement_info", supplement_info))
            if skybound_updates:
                events.append(self._event(EventKind.SKYBOUND_ART, player=player, source_uid=uid, amount=1))
        for owner, cards in destroyed_cards.items():
            if not cards:
                continue
            history = get_player(snapshot, owner).destroyed_card_ids
            transactions.append(SetPlayerField(owner, "destroyed_card_ids", (*history, *(CardIdCount(card_id=card.card_id, count=card.style_id) for card in cards))))
        if defer_draw:
            events.extend((
                self._event(EventKind.TURN_START_DECK, player=player, field_last_words=tuple(field_last_words), crest_last_words=tuple(crest_last_words)),
            ))
        else:
            events.extend((*field_last_words, *crest_last_words))
        if not defer_draw:
            transactions.extend(self._draw_card_transactions(snapshot, player, drawn_card, turn_override=next_turn))
            if view.deck_count <= 0:
                events.append(self._deckout_event(snapshot, player))
            elif drawn_card is None:
                events.append(self._event(EventKind.DRAW_HIDDEN, player=player))
            elif hand_size(view) >= MAX_HAND_SIZE:
                events.extend((self._event(EventKind.DRAW, player=player, source_uid=drawn_card.uid), self._event(EventKind.OVERDRAW, player=player, source_uid=drawn_card.uid)))
            else:
                events.append(self._event(EventKind.DRAW, player=player, source_uid=drawn_card.uid))
        return self.apply_transactions(snapshot, tuple(transactions), tuple(events))

    def start_turn_draw(self, snapshot: BattleSnapshot, player: str, drawn_card: CardView | None = None) -> RuleExecutionResult:
        return self.draw_card(snapshot, player, drawn_card)

    def draw_card(self, snapshot: BattleSnapshot, player: str, card: CardView | None) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        deck_instance_transaction = None
        # 如果未提供具体卡牌，尝试从 deck_instances 自动取牌
        if card is None and view.deck_instances is not None and view.deck_instances.draw_pile:
            state = view.deck_instances
            instance_id = state.draw_pile[0]
            instance = state.by_instance_id()[instance_id]
            card = instance.card if instance.uid is None else replace(instance.card, uid=instance.uid)
            # deck_instances 状态更新通过 transaction 传播（确保 EffectInterpreter log.apply_all 能正确传递）
            moved = state.move(instance_id, "hand", uid=card.uid, card=card, visibility="visible")
            deck_instance_transaction = SetPlayerField(player, "deck_instances", moved)
        transactions = list(self._draw_card_transactions(snapshot, player, card))
        if deck_instance_transaction is not None:
            transactions.append(deck_instance_transaction)
        transactions = tuple(transactions)
        if view.deck_count <= 0:
            return self.apply_transactions(snapshot, transactions, (self._deckout_event(snapshot, player),))
        if card is None:
            return self.apply_transactions(snapshot, transactions, (self._event(EventKind.DRAW_HIDDEN, player=player),))
        if hand_size(view) >= MAX_HAND_SIZE:
            return self.apply_transactions(snapshot, transactions, (self._event(EventKind.DRAW, player=player, source_uid=card.uid), self._event(EventKind.OVERDRAW, player=player, source_uid=card.uid)))
        return self.apply_transactions(snapshot, transactions, (self._event(EventKind.DRAW, player=player, source_uid=card.uid),))

    def activate_extra_pp(self, snapshot: BattleSnapshot, player: str) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        extra_pp = view.extra_pp
        if extra_pp is None:
            raise RuleKernelError("extra pp state is missing")
        if extra_pp.extra_pp <= 0:
            raise RuleKernelError("no extra pp available")
        if extra_pp.is_used_extra_pp_this_turn:
            raise RuleKernelError("extra pp already used this turn")
        return self.apply_transactions(
            snapshot,
            (
                ModifyPlayerCounter(player, "pp", 1),
                SetPlayerField(player, "extra_pp", replace(extra_pp, preparation_extra_pp=1, is_used_extra_pp_this_turn=True)),
            ),
            (self._event(EventKind.EXTRA_PP, player=player, amount=1),),
        )

    def consume_extra_pp(self, snapshot: BattleSnapshot, player: str, amount: int = 1) -> RuleExecutionResult:
        if amount < 0:
            raise RuleKernelError("extra pp amount must be non-negative")
        view = get_player(snapshot, player)
        extra_pp = view.extra_pp
        if extra_pp is None:
            raise RuleKernelError("extra pp state is missing")
        if extra_pp.extra_pp < amount:
            raise RuleKernelError("not enough extra pp")
        return self.apply_transactions(
            snapshot,
            (SetPlayerField(player, "extra_pp", replace(extra_pp, extra_pp=extra_pp.extra_pp - amount)),),
            (self._event(EventKind.CONSUME_EXTRA_PP, player=player, amount=amount),),
        )

    def consume_necromancy(self, snapshot: BattleSnapshot, player: str, amount: int) -> RuleExecutionResult:
        if amount < 0:
            raise RuleKernelError("necromancy amount must be non-negative")
        view = get_player(snapshot, player)
        if view.graveyard_count < amount:
            # 墓地不足是正常游戏语义：该效果不发动（整条效果回滚），
            # 由 effect 层降级为 PARTIAL，不应报 predict_error 中断对局。
            raise InsufficientResourceError("not enough graveyard for necromancy")
        return self.apply_transactions(snapshot, (ModifyPlayerCounter(player, "graveyard_count", -amount),), (self._event(EventKind.NECROMANCY, player=player, amount=amount),))

    def consume_earth_sigil(self, snapshot: BattleSnapshot, player: str, amount: int = 1) -> RuleExecutionResult:
        if amount < 0:
            raise RuleKernelError("earth sigil amount must be non-negative")
        remaining = amount
        transactions: list[SnapshotTransaction] = []
        events: list[ExecutionEvent] = []
        destroyed_ids = list(get_player(snapshot, player).destroyed_card_ids)
        for uid in get_player(snapshot, player).field:
            if remaining <= 0:
                break
            card = snapshot.cards_by_uid[uid]
            if not card.is_earth_sigil:
                continue
            stack = max(card.stack, 1)
            consumed = min(stack, remaining)
            remaining -= consumed
            next_stack = stack - consumed
            events.append(self._event(EventKind.EARTH_RITE, player=player, source_uid=uid, amount=consumed))
            if next_stack <= 0:
                # 土之印被消耗到 0 时是**被破坏**：必须与 consume_evidence 同一口径，
                # 进墓地、记 destroyed_card_ids、发 DESTROY/MOVE_TO_GRAVEYARD 事件
                # （否则墓地数与 destroyed_card_ids 都会少一条，
                # replay_...1833407 st103：大地之魔片被土之秘术吃掉，本地墓地停在 14）。
                view = replace(get_player(snapshot, player), destroyed_card_ids=tuple(destroyed_ids))
                transactions.extend(destroyed_card_transactions(snapshot, player, card, view=view))
                destroyed_ids.append(CardIdCount(card_id=card.card_id, count=card.style_id))
                events.append(self._event(EventKind.DESTROY, player=player, source_uid=uid, leaving_card=self._leaving_card_payload(card), has_guard=card.has_guard))
                events.append(self._event(EventKind.MOVE_TO_GRAVEYARD, player=player, source_uid=uid))
                if card.has_last_words:
                    events.append(self._last_words_event(player, card))
            else:
                transactions.append(SetCardField(uid, "stack", next_stack))
        if remaining > 0:
            # 土之秘术资源不足是正常游戏语义：效果不发动（整条效果回滚），
            # 不应报 predict_error 中断对局。
            raise InsufficientResourceError("not enough earth sigil")
        return self.apply_transactions(snapshot, tuple(transactions), tuple(events))

    def consume_evidence(self, snapshot: BattleSnapshot, player: str, amount: int = 1) -> RuleExecutionResult:
        """「使自己的战场上的证据-N」：扣减自己战场上证据护符的证据值。

        证据不是「每张护符 1 点」，而是每张护符自带证据点数——协议字段
        `content_unknown`（实测 2 / 3 / 4）。卡面的「证据-N」按顺序从自己战场上的
        证据护符扣 N 点：

        * 扣到 0 的护符被破坏（进墓地、记 destroyed_card_ids）——
          replay_...777950 st160（2 点扣 2 → 移除）、...803112 st62（2 点扣 2 → 移除）；
        * 仍有剩余的写回剩余点数、留在场上——
          ...803112 st53（3 点扣 2 → 留场）、...1115031 st43（3 点扣 2 → 留场）；
        * 场上证据点数为 0 时整步不发动，但同一段落的其余操作照常结算
          （...803112 st101）。与【土之秘术】不同——后者不足会让整条效果不发动。
        """
        if amount < 0:
            raise RuleKernelError("evidence amount must be non-negative")
        player_view = get_player(snapshot, player)
        total_points = sum(
            max(int(card.content_unknown), 0)
            for uid in player_view.field
            if (card := snapshot.cards_by_uid.get(uid)) is not None and is_evidence_card(card)
        )
        if total_points < amount:
            # 证据是**整条效果的代价**：战场上的证据点数不够时这一步不发动，
            # 同段落后续操作也不结算（...803112 st101：场上 0 点证据时既没扣证据，
            # 也没有破坏对手的随从）。与【土之秘术】同一约定。
            raise InsufficientResourceError("not enough evidence")
        remaining = amount
        transactions: list[SnapshotTransaction] = []
        events: list[ExecutionEvent] = []
        destroyed_ids = list(player_view.destroyed_card_ids)
        for uid in player_view.field:
            if remaining <= 0:
                break
            card = snapshot.cards_by_uid.get(uid)
            if card is None or not is_evidence_card(card):
                continue
            available = max(int(card.content_unknown), 0)
            if available <= 0:
                # 这张护符已经没有证据点数：不参与支付，也不离场。
                continue
            taken = min(available, remaining)
            remaining -= taken
            if taken < available:
                transactions.append(SetCardField(uid, "content_unknown", available - taken))
                continue
            # 同一批事务里连续破坏多张时，后续每一张都要看到累积后的
            # destroyed_card_ids，否则后一张会覆盖前一张的历史。
            view = replace(player_view, destroyed_card_ids=tuple(destroyed_ids))
            transactions.extend(destroyed_card_transactions(snapshot, player, card, view=view))
            destroyed_ids.append(CardIdCount(card_id=card.card_id, count=card.style_id))
            events.append(self._event(EventKind.DESTROY, player=player, source_uid=uid, leaving_card=self._leaving_card_payload(card), has_guard=card.has_guard))
            events.append(self._event(EventKind.MOVE_TO_GRAVEYARD, player=player, source_uid=uid))
            if card.has_last_words:
                events.append(self._last_words_event(player, card))
        return self.apply_transactions(snapshot, tuple(transactions), tuple(events))

    def activate_card(self, snapshot: BattleSnapshot, player: str, uid: int) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if uid not in view.field:
            raise RuleKernelError("card is not on field")
        card = snapshot.cards_by_uid[uid]
        if not card.has_activation:
            raise RuleKernelError("card has no activation")
        evidence_txn = ()
        # 「本回合中自己启动过护符」类条件所需的本回合事件标记，复用
        # effect_trigger_counts（turn_start 已统一清零），避免新增状态字段。
        # 两个门槛分开记：11032310「启动过护符」不限族，11023310「启动过证据·护符」
        # 只认 tribe 21。
        counts = dict(view.effect_trigger_counts)
        counts["__amulet_activated_this_turn__"] = 1
        if 21 in (card.tribes or ()):
            counts["__evidence_amulet_activated_this_turn__"] = 1
        evidence_txn = (SetPlayerField(player, "effect_trigger_counts", counts),)
        if card.is_earth_sigil:
            if view.pp < 1:
                raise RuleKernelError("not enough pp")
            transactions = (
                *self._pay_pp_transactions(view, player, 1),
                ModifyCardCounter(uid, "stack", 1),
                SetCardField(uid, "activation_used_this_turn", True),
                SetCardField(uid, "has_activation", False),
                *evidence_txn,
            )
            return self.apply_transactions(snapshot, transactions, (*self._pay_pp_events(view, player, 1, source_uid=uid), self._event(EventKind.ACTIVATE, player=player, source_uid=uid)))
        return self.apply_transactions(snapshot, (SetCardField(uid, "activation_used_this_turn", True), SetCardField(uid, "has_activation", False), *evidence_txn), (self._event(EventKind.ACTIVATE, player=player, source_uid=uid),))

    def record_fusion(self, snapshot: BattleSnapshot, player: str, source_uid: int, material_uids: tuple[int, ...]) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if source_uid not in view.hand:
            raise RuleKernelError("fusion source is not in hand")
        if not material_uids:
            raise RuleKernelError("fusion requires materials")
        source_card = snapshot.cards_by_uid[source_uid]
        if source_card.supplement_info.get("fusion_used_this_turn"):
            raise RuleKernelError("fusion already used this turn")
        if len(set(material_uids)) != len(material_uids):
            raise RuleKernelError("fusion material duplicated")
        transactions: list[SnapshotTransaction] = []
        for material_uid in material_uids:
            if material_uid == source_uid:
                raise RuleKernelError("fusion source cannot be material")
            if material_uid not in view.hand:
                raise RuleKernelError("fusion material is not in hand")
            material_card = snapshot.cards_by_uid[material_uid]
            transactions.extend((RemoveZoneUid(player, "hand", material_uid), ModifyPlayerCounter(player, "graveyard_count", 1), RemoveCard(material_uid, material_card)))
        fusion_count = int(source_card.supplement_info.get("fusion_count", 0)) + len(material_uids)
        transactions.append(SetCardField(source_uid, "supplement_info", {**source_card.supplement_info, "fusion_count": fusion_count, "fusion_used_this_turn": 1}))
        return self.apply_transactions(snapshot, tuple(transactions), (self._event(EventKind.FUSION, player=player, source_uid=source_uid, material_uids=material_uids),))

    def manual_evolve(self, snapshot: BattleSnapshot, player: str, uid: int) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if view.ep <= 0:
            raise RuleKernelError("not enough ep")
        if uid not in view.field:
            raise RuleKernelError("card is not on field")
        card = snapshot.cards_by_uid[uid]
        skybound_transactions = self._skybound_art_increment_transactions(snapshot, player)
        transactions: tuple[SnapshotTransaction, ...] = (
            ModifyPlayerCounter(player, "ep", -1),
            ModifyPlayerCounter(player, "manual_evolve_count", 1),
            ModifyPlayerCounter(player, "evolve_count", 1),
            SetPlayerField(player, "is_evolved_this_turn", True),
            SetCardField(uid, "evolve_state", 1),
            *self._evolve_attack_transactions(card, uid),
            ModifyCardCounter(uid, "atk", 2),
            ModifyCardCounter(uid, "life", 2),
            ModifyCardCounter(uid, "max_life", 2 if card.max_life else 0),
            *skybound_transactions,
        )
        return self.apply_transactions(
            snapshot,
            transactions,
            (
                self._event(EventKind.MANUAL_EVOLVE, player=player, source_uid=uid),
                self._event(EventKind.BECOMES_EVOLVED, player=player, source_uid=uid),
                self._event(EventKind.SOURCE_BUFFED, player=player, source_uid=uid, target_uid=uid, attack_delta=2, life_delta=2),
            ),
        )

    def _skybound_art_increment_transactions(self, snapshot: BattleSnapshot, player: str) -> tuple[SnapshotTransaction, ...]:
        """进化时给同侧手牌中有奥义计数的卡 +1。"""
        view = get_player(snapshot, player)
        transactions: list[SnapshotTransaction] = []
        for uid in view.hand:
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            if "skybound_art_count" in card.supplement_info:
                new_si = {**card.supplement_info, "skybound_art_count": card.supplement_info["skybound_art_count"] + 1}
                transactions.append(SetCardField(uid, "supplement_info", new_si))
        return tuple(transactions)

    def _evolve_attack_transactions(self, card: CardView, uid: int) -> tuple[SnapshotTransaction, ...]:
        # 进化**解除召唤失调**：影之诗中进化后的随从当回合即可攻击，所以这里默认
        # 授予"可攻击随从"的窗口（`has_cant_attack` 除外）。进化本身不赋予打脸权，
        # 也不剥夺已有的打脸权；疾驰额外赋予打脸权。
        #
        # `not has_storm and not has_rush` 这一支**不是**笔误，它表达的正是
        # "没有疾驰/突进也能靠进化获得攻击窗口"。曾有一段改动把它当作
        # "任何随从都被判成能攻击"的 bug 而以 `card.can_attack` 替换，结果
        # 直接打掉了 `test_evolve_grants_follower_attack_window_without_leader_attack`
        # 与 `test_evolve_grants_rush_only_when_attack_count_remains` 覆盖的规则。
        # `has_rush` 改为从卡牌模板推导后，突进随从会走下面的 `if card.has_rush`
        # 分支，与本行结果一致，不需要调整本行。
        can_attack_follower = not card.has_cant_attack and (card.can_attack or not card.has_storm and not card.has_rush)
        can_attack_leader = card.can_attack_leader
        if card.has_storm:
            can_attack_follower = True
            can_attack_leader = True
        if card.has_rush:
            can_attack_follower = True
        attacks_left = card.attacks_left
        if attacks_left <= 0 and not card.can_attack and can_attack_follower:
            attacks_left = card.attack_limit or (card.buff.attack_limit if card.buff is not None else 0) or 1
        return (
            SetCardField(uid, "can_attack", can_attack_follower),
            SetCardField(uid, "can_attack_leader", can_attack_leader),
            SetCardField(uid, "attacks_left", attacks_left),
        )

    def auto_evolve(self, snapshot: BattleSnapshot, player: str, uid: int, *, grant_attack_window: bool = True) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if uid not in view.field:
            raise RuleKernelError("card is not on field")
        card = snapshot.cards_by_uid[uid]
        if card.evolve_state >= 1:
            raise RuleKernelError("card is already evolved")
        # 效果造成的进化不解除召唤失调：录像中刚登场随从被效果进化后
        # attacks_left 仍为 0（如 10844110 的入场曲进化）。
        attack_transactions = self._evolve_attack_transactions(card, uid) if grant_attack_window else ()
        skybound_transactions = self._skybound_art_increment_transactions(snapshot, player)
        transactions: tuple[SnapshotTransaction, ...] = (
            ModifyPlayerCounter(player, "evolve_count", 1),
            SetCardField(uid, "evolve_state", 1),
            *attack_transactions,
            ModifyCardCounter(uid, "atk", 2),
            ModifyCardCounter(uid, "life", 2),
            ModifyCardCounter(uid, "max_life", 2 if card.max_life else 0),
            *skybound_transactions,
        )
        return self.apply_transactions(
            snapshot,
            transactions,
            (
                self._event(EventKind.BECOMES_EVOLVED, player=player, source_uid=uid),
                self._event(EventKind.SOURCE_BUFFED, player=player, source_uid=uid, target_uid=uid, attack_delta=2, life_delta=2),
            ),
        )

    def manual_super_evolve(self, snapshot: BattleSnapshot, player: str, uid: int) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if view.sep <= 0:
            raise RuleKernelError("not enough sep")
        if uid not in view.field:
            raise RuleKernelError("card is not on field")
        card = snapshot.cards_by_uid[uid]
        skybound_transactions = self._skybound_art_increment_transactions(snapshot, player)
        transactions: tuple[SnapshotTransaction, ...] = (
            ModifyPlayerCounter(player, "sep", -1),
            ModifyPlayerCounter(player, "manual_evolve_count", 1),
            ModifyPlayerCounter(player, "evolve_count", 1),
            SetPlayerField(player, "is_evolved_this_turn", True),
            SetCardField(uid, "evolve_state", 2),
            SetCardField(uid, "has_super_evolve_buff", True),
            *self._evolve_attack_transactions(card, uid),
            ModifyCardCounter(uid, "atk", 3),
            ModifyCardCounter(uid, "life", 3),
            ModifyCardCounter(uid, "max_life", 3 if card.max_life else 0),
            *skybound_transactions,
        )
        return self.apply_transactions(
            snapshot,
            transactions,
            (
                self._event(EventKind.MANUAL_EVOLVE, player=player, source_uid=uid),
                self._event(EventKind.MANUAL_SUPER_EVOLVE, player=player, source_uid=uid),
                self._event(EventKind.BECOMES_EVOLVED, player=player, source_uid=uid),
                self._event(EventKind.BECOMES_SUPER_EVOLVED, player=player, source_uid=uid),
                self._event(EventKind.SOURCE_BUFFED, player=player, source_uid=uid, target_uid=uid, attack_delta=3, life_delta=3),
            ),
        )

    def auto_super_evolve(self, snapshot: BattleSnapshot, player: str, uid: int) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if uid not in view.field:
            raise RuleKernelError("card is not on field")
        card = snapshot.cards_by_uid[uid]
        if card.evolve_state >= 2:
            raise RuleKernelError("card is already super evolved")
        transactions: tuple[SnapshotTransaction, ...] = (
            ModifyPlayerCounter(player, "evolve_count", 1),
            SetCardField(uid, "evolve_state", 2),
            SetCardField(uid, "has_super_evolve_buff", True),
            *self._evolve_attack_transactions(card, uid),
            ModifyCardCounter(uid, "atk", 3),
            ModifyCardCounter(uid, "life", 3),
            ModifyCardCounter(uid, "max_life", 3 if card.max_life else 0),
        )
        return self.apply_transactions(
            snapshot,
            transactions,
            (
                self._event(EventKind.BECOMES_EVOLVED, player=player, source_uid=uid),
                self._event(EventKind.BECOMES_SUPER_EVOLVED, player=player, source_uid=uid),
                self._event(EventKind.SOURCE_BUFFED, player=player, source_uid=uid, target_uid=uid, attack_delta=3, life_delta=3),
            ),
        )

    def attack_leader(self, snapshot: BattleSnapshot, attacker_uid: int, defending_player: str) -> RuleExecutionResult:
        attacker = snapshot.cards_by_uid[attacker_uid]
        if attacker.attacks_left <= 0:
            raise RuleKernelError("attacker has no attacks left")
        damage = attacker.atk
        defending_view = get_player(snapshot, defending_player)
        # player buff increase_damage：主战者**受到**的伤害增加（如「本回合中，
        # 自己主战者受到的伤害+1」）。与 damage_cut 同属「对最终伤害的修正」，
        # 先增后减：卡面上「受到的伤害+X，受到的伤害-Y」按此顺序叠加。
        if defending_view.buff is not None and defending_view.buff.increase_damage > 0 and damage > 0:
            damage += defending_view.buff.increase_damage
        # player buff damage_cut: 减少对主战者的伤害
        if defending_view.buff is not None and defending_view.buff.damage_cut > 0 and damage > 0:
            damage = max(0, damage - defending_view.buff.damage_cut)
        transactions: list[SnapshotTransaction] = [SetCardField(attacker_uid, "attacks_left", max(attacker.attacks_left - 1, 0))]
        attacker_owner = self._owner_of_field_uid(snapshot, attacker_uid)
        transactions.append(ModifyPlayerCounter(attacker_owner or "me", "attack_count", 1))
        # 记录"本回合己方随从攻击过对方主战者"（供"上一回合随从攻击过主战者"类条件）。
        if attacker.card_type.is_follower and attacker_owner is not None:
            owner_view_for_flag = get_player(snapshot, attacker_owner)
            if not owner_view_for_flag.follower_attacked_leader_this_turn:
                transactions.append(SetPlayerField(attacker_owner, "follower_attacked_leader_this_turn", True))
        events: list[ExecutionEvent] = [self._event(EventKind.ATTACK_LEADER, source_uid=attacker_uid, player=attacker_owner)]
        if defending_view.buff is not None and defending_view.buff.temp_shield:
            damage = 0
            transactions.append(SetPlayerField(defending_player, "buff", replace(defending_view.buff, temp_shield=False)))
            events.append(self._event(EventKind.TEMP_SHIELD, source_uid=attacker_uid, player=defending_player))
        else:
            transactions.append(ModifyPlayerCounter(defending_player, "life", -damage))
        events.append(self._event(EventKind.DAMAGE, source_uid=attacker_uid, player=defending_player, amount=damage))
        if attacker.has_ambush:
            transactions.append(SetCardField(attacker_uid, "has_ambush", False))
        if attacker_owner is not None and attacker.has_drain and damage > 0:
            owner_view = get_player(snapshot, attacker_owner)
            transactions.append(SetPlayerField(attacker_owner, "life", min(owner_view.max_life, owner_view.life + damage)))
            events.append(self._event(EventKind.HEAL, source_uid=attacker_uid, player=attacker_owner, amount=damage))
        result = self.apply_transactions(snapshot, tuple(transactions), tuple(events))
        # 致死判定：对方 life <= 0 时设置 battle result
        after_view = get_player(result.snapshot, defending_player)
        if after_view.life <= 0:
            attacking_player = "me" if defending_player == "opponent" else "opponent"
            lethal_transactions = (
                SetPlayerField(attacking_player, "result", 101),
                SetPlayerField(defending_player, "result", 102),
            )
            result = self.apply_transactions(result.snapshot, lethal_transactions, result.events)
        return result

    def destroy_field_card(self, snapshot: BattleSnapshot, player: str, uid: int) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if uid not in view.field:
            raise RuleKernelError("card is not on field")
        card = snapshot.cards_by_uid[uid]
        if is_super_evolved_on_own_turn(snapshot, card, player):
            return self.apply_transactions(snapshot, (), ())
        if vanishes_on_leave(card):
            # 「离场时，使本随从消失」：退场按消失处理（不进墓地、不计 destroyed_card_ids），
            # 事件也必须是 BANISH，让 on_self_card_left_field 订阅者响应。
            return self.apply_transactions(
                snapshot,
                destroyed_card_transactions(snapshot, player, card),
                (vanish_on_leave_event(player, card, self._leaving_card_payload(card)),),
            )
        destroy_events = [
            self._event(EventKind.DESTROY, player=player, source_uid=uid, target_uid=uid, leaving_card=self._leaving_card_payload(card), has_guard=card.has_guard),
            self._event(EventKind.MOVE_TO_GRAVEYARD, player=player, source_uid=uid),
        ]
        if card.has_last_words:
            destroy_events.append(self._last_words_event(player, card))
        return self.apply_transactions(
            snapshot,
            destroyed_card_transactions(snapshot, player, card),
            tuple(destroy_events),
        )

    def banish_field_card(self, snapshot: BattleSnapshot, player: str, uid: int) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if uid not in view.field:
            raise RuleKernelError("card is not on field")
        return self.apply_transactions(
            snapshot,
            (RemoveZoneUid(player, "field", uid), RemoveCard(uid, snapshot.cards_by_uid[uid])),
            (self._event(EventKind.BANISH, player=player, source_uid=uid),),
        )

    def bounce_field_card(self, snapshot: BattleSnapshot, player: str, uid: int) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if uid not in view.field:
            raise RuleKernelError("card is not on field")
        card = snapshot.cards_by_uid[uid]
        if hand_size(view) >= MAX_HAND_SIZE:
            return self.apply_transactions(
                snapshot,
                (RemoveZoneUid(player, "field", uid), RemoveCard(uid, card)),
                (self._event(EventKind.FLOOD, player=player, source_uid=uid),),
            )
        field_index = view.field.index(uid)
        materialized = self._materialize_hand_card(snapshot, uid, player)
        return self.apply_transactions(
            snapshot,
            (
                RemoveZoneUid(player, "field", uid),
                RemoveCard(uid, card),
                AddCard(materialized),
                InsertZoneUid(player, "hand", uid),
            ),
            (self._event(EventKind.BOUNCE, player=player, source_uid=uid),),
        )

    def transform_field_card(self, snapshot: BattleSnapshot, player: str, uid: int, new_card: CardView) -> RuleExecutionResult:
        view = get_player(snapshot, player)
        if uid not in view.field:
            raise RuleKernelError("card is not on field")
        if new_card.uid == uid:
            raise RuleKernelError("transform replacement must use a new uid")
        if new_card.uid in snapshot.cards_by_uid:
            raise RuleKernelError("transform replacement uid already exists")
        field_index = view.field.index(uid)
        old_card = snapshot.cards_by_uid[uid]
        replacement = replace(new_card, attacks_left=0, evolve_state=0)
        return self.apply_transactions(
            snapshot,
            (
                RemoveZoneUidAt(player, "field", field_index, uid),
                RemoveCard(uid, old_card),
                AddCard(replacement),
                InsertZoneUid(player, "field", replacement.uid, field_index),
            ),
            (self._event(EventKind.TRANSFORM, player=player, source_uid=uid, target_uid=replacement.uid),),
        )

    def attack_follower(self, snapshot: BattleSnapshot, attacker_uid: int, defender_uid: int) -> RuleExecutionResult:
        attacker = snapshot.cards_by_uid[attacker_uid]
        defender = snapshot.cards_by_uid[defender_uid]
        if attacker.attacks_left <= 0:
            raise RuleKernelError("attacker has no attacks left")
        attacker_owner = self._owner_of_field_uid(snapshot, attacker_uid)
        defender_owner = self._owner_of_field_uid(snapshot, defender_uid)
        attacker_protected = is_super_evolved_on_own_turn(snapshot, attacker, attacker_owner or "")
        defender_protected = is_super_evolved_on_own_turn(snapshot, defender, defender_owner or "")
        attacker_damage = 0 if defender.has_temp_shield or defender_protected else attacker.atk
        defender_damage = 0 if attacker.has_temp_shield or attacker_protected else defender.atk
        # 卡牌级伤害减免（DamageCut，"受到的4点或以上的伤害变为3点"），
        # 与效果伤害共用 apply_damage_cut 保持语义一致。
        attacker_damage = apply_damage_cut(defender, attacker_damage)
        defender_damage = apply_damage_cut(attacker, defender_damage)
        # attack_follower / clash 事件带上防御者攻击前信息（leaving_card）。
        # 「超进化随从攻击随从时」等 on_attack 效果的 event_target 条件判定，
        # 语义时机是攻击发起的瞬间；但事件在攻击结算后（after_core）被消费，
        # 此时防御者若被这次攻击击杀已从场上移除，consumer 无法再取到 event_target。
        # 在产生端把防御者攻击前信息放进 payload，保证 conditions 能正确恢复目标类型。
        defender_leaving = self._leaving_card_payload(defender)
        # 事件源/目标分别携带各自攻击前状态：攻击者可能被这次攻击反击致死，
        # 消费端（条件求值）需要 source 的卡面信息；单一 leaving_card 语义是
        # 「事件源自身离场前状态」（destroy / last_words / countdown），攻击事件的
        # 防御者不是事件源，必须用独立的 leaving_target_card 表达。
        attacker_leaving = self._leaving_card_payload(attacker)
        transactions: list[SnapshotTransaction] = [SetCardField(attacker_uid, "attacks_left", max(attacker.attacks_left - 1, 0)), ModifyPlayerCounter(attacker_owner or "me", "attack_count", 1)]
        events: list[ExecutionEvent] = [
            self._event(EventKind.ATTACK_FOLLOWER, source_uid=attacker_uid, target_uid=defender_uid, player=attacker_owner, leaving_source_card=attacker_leaving, leaving_target_card=defender_leaving),
            self._event(EventKind.CLASH, source_uid=attacker_uid, target_uid=defender_uid, player=attacker_owner, leaving_source_card=attacker_leaving, leaving_target_card=defender_leaving),
        ]
        if attacker.has_ambush:
            transactions.append(SetCardField(attacker_uid, "has_ambush", False))
        if defender.has_temp_shield:
            transactions.append(SetCardField(defender_uid, "has_temp_shield", False))
            events.append(self._event(EventKind.TEMP_SHIELD, source_uid=attacker_uid, target_uid=defender_uid))
        else:
            transactions.append(ModifyCardCounter(defender_uid, "life", -attacker_damage))
        if attacker.has_temp_shield:
            transactions.append(SetCardField(attacker_uid, "has_temp_shield", False))
            events.append(self._event(EventKind.TEMP_SHIELD, source_uid=defender_uid, target_uid=attacker_uid))
        else:
            transactions.append(ModifyCardCounter(attacker_uid, "life", -defender_damage))
        if attacker.has_bane and not defender.has_cant_destroy and not defender_protected:
            transactions.append(SetCardField(defender_uid, "life", 0))
            events.append(self._event(EventKind.BANE, source_uid=attacker_uid, target_uid=defender_uid))
        if defender.has_bane and not attacker.has_cant_destroy and not attacker_protected:
            transactions.append(SetCardField(attacker_uid, "life", 0))
            events.append(self._event(EventKind.BANE, source_uid=defender_uid, target_uid=attacker_uid))
        events.extend((
            self._event(EventKind.DAMAGE, source_uid=attacker_uid, target_uid=defender_uid, amount=attacker_damage),
            self._event(EventKind.DAMAGE, source_uid=defender_uid, target_uid=attacker_uid, amount=defender_damage),
        ))
        if attacker_owner is not None and attacker.has_drain and attacker_damage > 0:
            view = get_player(snapshot, attacker_owner)
            transactions.append(SetPlayerField(attacker_owner, "life", min(view.max_life, view.life + attacker_damage)))
            events.append(self._event(EventKind.HEAL, source_uid=attacker_uid, player=attacker_owner, amount=attacker_damage))
        # 防御方随从的【虹吸】不触发：吸血只对「主动造成伤害」生效，被攻击时的
        # 反击伤害不回复主战者（录像证据：10614120 带虹吸被攻击，主战者生命不变）。
        after_damage = self.apply_transactions(snapshot, tuple(transactions), tuple(events))
        cleanup_transactions: list[SnapshotTransaction] = []
        cleanup_events: list[ExecutionEvent] = []
        after = after_damage.snapshot
        destroyed_defender = False
        for uid, owner in ((attacker_uid, self._owner_of_field_uid(after, attacker_uid)), (defender_uid, self._owner_of_field_uid(after, defender_uid))):
            if owner is None:
                continue
            if after.cards_by_uid[uid].life <= 0:
                if uid == defender_uid:
                    destroyed_defender = True
                card = after.cards_by_uid[uid]
                cleanup_transactions.extend(destroyed_card_transactions(after, owner, card))
                if vanishes_on_leave(card):
                    cleanup_events.append(vanish_on_leave_event(owner, card, self._leaving_card_payload(card)))
                    continue
                cleanup_events.extend((
                    self._event(EventKind.DESTROY, player=owner, source_uid=uid, leaving_card=self._leaving_card_payload(card), has_guard=card.has_guard),
                    self._event(EventKind.MOVE_TO_GRAVEYARD, player=owner, source_uid=uid),
                ))
                if card.has_last_words:
                    cleanup_events.append(self._last_words_event(owner, card))
        if destroyed_defender and attacker.evolve_state == 2 and defender_owner is not None:
            cleanup_transactions.append(ModifyPlayerCounter(defender_owner, "life", -1))
            events.append(self._event(EventKind.DAMAGE, source_uid=attacker_uid, player=defender_owner, amount=1, reason="super_evolve_destroy"))
        if not cleanup_transactions:
            return after_damage
        cleanup = RollbackTransactionLog(root=after_damage.transaction_log.root, current=after_damage.snapshot, applied=after_damage.transaction_log.applied)
        cleaned = cleanup.apply_all(after_damage.snapshot, cleanup_transactions)
        return RuleExecutionResult(snapshot=cleaned, transaction_log=cleanup, events=tuple(events + cleanup_events + [self._event(EventKind.DESTROYED_BY_DAMAGE)]))

    def _owner_of_field_uid(self, snapshot: BattleSnapshot, uid: int) -> str | None:
        if uid in snapshot.me.field:
            return "me"
        if uid in snapshot.opponent.field:
            return "opponent"
        return None
