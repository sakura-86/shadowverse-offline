# -*- coding: utf-8 -*-
"""Local self-play match setup."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path

from decision_engine.cards import BaseCardInfo, CardResolver, MasterDataProvider, load_card_templates_by_base_card_id
from decision_engine.execution.card_factory import build_card_view
from decision_engine.state.deck_instance import DeckInstance, DeckInstanceState
from decision_engine.state.snapshot import BattleSnapshot, CardView, CrestView, ExtraPpView, PlayerView


@dataclass(frozen=True)
class LocalDeckList:
    name: str
    class_name: str = ""
    card_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class LocalMatchSetupConfig:
    seed: int = 1
    first_player: str = "me"
    initial_life: int = 20
    initial_hand_first: int = 4
    initial_hand_second: int = 4
    initial_pp: int = 0
    initial_max_pp: int = 0
    first_ep: int = 2
    second_ep: int = 2
    first_sep: int = 2
    second_sep: int = 2
    source: str = "local_self_play"


@dataclass(frozen=True)
class LocalMatchSetupResult:
    snapshot: BattleSnapshot | None
    seed: int
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class LocalMatchSetup:
    def __init__(self, resolver: CardResolver | None = None, *, template_source: str | Path | None = None):
        if resolver is None:
            database_path = Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"
            resolver = CardResolver(MasterDataProvider.from_json(database_path))
        self.resolver = resolver
        template_path = Path(template_source) if template_source is not None else Path(__file__).resolve().parents[1] / "data" / "public" / "card_templates.json"
        self.templates_by_base_card_id = load_card_templates_by_base_card_id(template_path)

    def create(
        self,
        me_deck: LocalDeckList,
        opponent_deck: LocalDeckList,
        config: LocalMatchSetupConfig | None = None,
    ) -> LocalMatchSetupResult:
        config = config or LocalMatchSetupConfig()
        if config.first_player not in {"me", "opponent"}:
            return LocalMatchSetupResult(snapshot=None, seed=config.seed, errors=(f"unsupported first_player: {config.first_player}",))

        rng = random.Random(config.seed)
        cards_by_uid: dict[int, CardView] = {}
        errors: list[str] = []
        warnings: list[str] = []
        # 卡牌 uid 从 3 开始，避免与 player unique_id (1, 2) 冲突
        next_uid = 3

        me_cards, next_uid, me_errors = self._materialize_deck(me_deck, owner="me", uid_start=next_uid)
        cards_by_uid.update({card.uid: card for card in me_cards})
        errors.extend(me_errors)
        opponent_cards, next_uid, opponent_errors = self._materialize_deck(opponent_deck, owner="opponent", uid_start=next_uid)
        cards_by_uid.update({card.uid: card for card in opponent_cards})
        errors.extend(opponent_errors)
        if errors:
            return LocalMatchSetupResult(snapshot=None, seed=config.seed, warnings=tuple(warnings), errors=tuple(errors))

        me_instances = self._make_shuffled_instances(me_cards, owner="me", source=me_deck.name, rng=rng)
        opponent_instances = self._make_shuffled_instances(opponent_cards, owner="opponent", source=opponent_deck.name, rng=rng)
        me_hand_count = config.initial_hand_first if config.first_player == "me" else config.initial_hand_second
        opponent_hand_count = config.initial_hand_first if config.first_player == "opponent" else config.initial_hand_second
        me_instances, me_hand = self._draw_initial_hand(me_instances, me_hand_count)
        opponent_instances, opponent_hand = self._draw_initial_hand(opponent_instances, opponent_hand_count)

        me_first = config.first_player == "me"
        opponent_first = not me_first

        # 信仰初始化：检测卡组中是否有关联信仰卡的本体卡
        me_crests = self._detect_faith_crests(me_deck)
        opponent_crests = self._detect_faith_crests(opponent_deck)

        me = PlayerView(
            life=config.initial_life,
            max_life=config.initial_life,
            pp=config.initial_pp,
            max_pp=config.initial_max_pp,
            ep=config.first_ep if me_first else config.second_ep,
            sep=config.first_sep if me_first else config.second_sep,
            max_ep=config.first_ep if me_first else config.second_ep,
            max_sep=config.first_sep if me_first else config.second_sep,
            deck_count=len(me_instances.draw_pile),
            is_first_side=me_first,
            hand=me_hand,
            unique_id=1,
            extra_pp=ExtraPpView(),
            deck_instances=me_instances,
            evolve_turn=5 if me_first else 4,
            super_evolve_turn=7 if me_first else 6,
            crests=me_crests,
        )
        opponent = PlayerView(
            life=config.initial_life,
            max_life=config.initial_life,
            pp=config.initial_pp,
            max_pp=config.initial_max_pp,
            ep=config.first_ep if opponent_first else config.second_ep,
            sep=config.first_sep if opponent_first else config.second_sep,
            max_ep=config.first_ep if opponent_first else config.second_ep,
            max_sep=config.first_sep if opponent_first else config.second_sep,
            deck_count=len(opponent_instances.draw_pile),
            is_first_side=opponent_first,
            hand=opponent_hand,
            unique_id=2,
            extra_pp=ExtraPpView(),
            deck_instances=opponent_instances,
            evolve_turn=5 if opponent_first else 4,
            super_evolve_turn=7 if opponent_first else 6,
            crests=opponent_crests,
        )
        snapshot = BattleSnapshot(
            me=me,
            opponent=opponent,
            cards_by_uid=cards_by_uid,
            source=config.source,
            is_ally_turn=config.first_player == "me",
            is_turn_ended=True,
        )
        return LocalMatchSetupResult(snapshot=snapshot, seed=config.seed, warnings=tuple(warnings), errors=())

    def _detect_faith_crests(self, deck: LocalDeckList) -> tuple[CrestView, ...]:
        """检测卡组中是否有关联信仰卡的本体卡，返回需要初始化的信仰 CrestView。"""
        deck_base_card_ids = set()
        for card_id in deck.card_ids:
            base = self.resolver.get_base_by_card_id(card_id)
            if base:
                deck_base_card_ids.add(base.base_card_id)
        crests: list[CrestView] = []
        seen_faith_ids: set[int] = set()
        for base_card_id in deck_base_card_ids:
            base = self.resolver.get_base(base_card_id)
            if base is None:
                continue
            for related_id in base.related_base_card_ids:
                if related_id in seen_faith_ids:
                    continue
                related = self.resolver.get_base(related_id)
                if related is not None and related.type_id == 7:  # Faith type
                    crests.append(CrestView(card_id=related_id, faith_value=0))
                    seen_faith_ids.add(related_id)
        return tuple(crests)

    def _materialize_deck(self, deck: LocalDeckList, *, owner: str, uid_start: int) -> tuple[tuple[CardView, ...], int, tuple[str, ...]]:
        cards: list[CardView] = []
        errors: list[str] = []
        next_uid = uid_start
        for index, card_id in enumerate(deck.card_ids):
            base = self.resolver.get_base_by_card_id(card_id)
            if base is None:
                errors.append(f"{owner} deck {deck.name} card_id not found: {card_id}")
                continue
            cards.append(self._card_view(card_id, base, uid=next_uid, order=index))
            next_uid += 1
        return tuple(cards), next_uid, tuple(errors)

    def _card_view(self, card_id: int, base: BaseCardInfo, *, uid: int, order: int) -> CardView:
        """把主数据 + 模板落成 CardView。

        构造语义只有一份，见 execution/card_factory.py：沙盒、fixture 与 self-play
        建局必须得到同一张卡，否则「沙盒里能做的动作」与「训练/验收里能做的动作」会分叉。
        """
        template = self.templates_by_base_card_id.get(base.base_card_id)
        return build_card_view(
            uid=uid,
            base_card_id=base.base_card_id,
            card_id=card_id,
            template=template,
            info=base,
            supplement_info=self._initial_supplement_info(template, order, base=base),
        )

    def _initial_supplement_info(self, template, order: int, base: "BaseCardInfo | None" = None) -> dict[str, int]:
        """初始化卡的 supplement_info，包含 deck_order 和奥义相关字段。"""
        si: dict[str, int] = {"deck_order": order}
        # 从效果文本判断是否有奥义能力
        skill_desc = base.skill_desc if base else ""
        if "奥义" in (skill_desc or ""):
            si["skybound_art_count"] = 0
        if "解放奥义" in (skill_desc or ""):
            si["super_skybound_art_count"] = 0
        return si

    def _make_shuffled_instances(self, cards: tuple[CardView, ...], *, owner: str, source: str, rng: random.Random) -> DeckInstanceState:
        instances = tuple(
            DeckInstance(
                instance_id=index + 1,
                card=card,
                owner=owner,
                zone="deck",
                uid=card.uid,
                origin="deck",
                order_hint=index,
                visibility="hidden",
            )
            for index, card in enumerate(cards)
        )
        draw_pile = list(instance.instance_id for instance in instances)
        rng.shuffle(draw_pile)
        return DeckInstanceState(instances=instances, draw_pile=tuple(draw_pile), source=source, status="available")

    def _draw_initial_hand(self, state: DeckInstanceState, count: int) -> tuple[DeckInstanceState, tuple[int, ...]]:
        current = state
        hand_uids: list[int] = []
        for _ in range(max(count, 0)):
            if not current.draw_pile:
                break
            instance_id = current.draw_pile[0]
            instance = current.by_instance_id()[instance_id]
            if instance.uid is None:
                raise ValueError(f"deck instance has no uid in local setup: {instance_id}")
            current = current.move(instance_id, "hand", visibility="visible")
            hand_uids.append(instance.uid)
        return current, tuple(hand_uids)


def perform_mulligan(
    snapshot: BattleSnapshot,
    player: str,
    cards_to_change: tuple[int, ...] | list[int],
    *,
    rng: random.Random | None = None,
) -> BattleSnapshot:
    """执行换牌（mulligan）：将选中的手牌退回卡组洗入，再重新抽取同等数量的牌。

    Args:
        snapshot: 当前对局快照。
        player: "me" 或 "opponent"。
        cards_to_change: 要换掉的卡牌 uid 列表（可为空，表示全部保留）。
        rng: 可选随机数生成器，用于洗入后洗牌。不传时使用默认 Random()。

    Returns:
        更新后的 BattleSnapshot。
    """
    if not cards_to_change:
        return snapshot

    view = snapshot.me if player == "me" else snapshot.opponent if player == "opponent" else None
    if view is None:
        raise ValueError(f"unsupported player: {player}")
    state = view.deck_instances
    if state is None:
        raise ValueError(f"player {player} has no deck_instances, cannot perform mulligan")

    if rng is None:
        rng = random.Random()

    # 验证所有 cards_to_change 确实在手牌中
    hand_set = set(view.hand)
    for uid in cards_to_change:
        if uid not in hand_set:
            raise ValueError(f"uid {uid} not in {player} hand, cannot mulligan")

    # 1. 将选中的牌从 hand 移回 draw_pile
    by_uid = state.by_uid()
    current_state = state
    for uid in cards_to_change:
        instance = by_uid.get(uid)
        if instance is None:
            raise ValueError(f"uid {uid} has no deck instance in {player}")
        current_state = current_state.move(instance.instance_id, "deck", visibility="hidden")

    # 2. 重新洗 draw_pile
    draw_pile_list = list(current_state.draw_pile)
    rng.shuffle(draw_pile_list)
    current_state = replace(current_state, draw_pile=tuple(draw_pile_list))

    # 3. 从 draw_pile 顶部重新抽取同等数量的牌
    redraw_count = len(cards_to_change)
    new_hand_uids: list[int] = []
    for _ in range(redraw_count):
        if not current_state.draw_pile:
            break
        instance_id = current_state.draw_pile[0]
        instance = current_state.by_instance_id()[instance_id]
        if instance.uid is None:
            raise ValueError(f"deck instance has no uid during mulligan redraw: {instance_id}")
        current_state = current_state.move(instance_id, "hand", visibility="visible")
        new_hand_uids.append(instance.uid)

    # 4. 更新 PlayerView 的 hand：去掉换掉的 uid，加上新抽的 uid
    remaining_hand = tuple(uid for uid in view.hand if uid not in set(cards_to_change))
    new_hand = remaining_hand + tuple(new_hand_uids)
    new_deck_count = len(current_state.draw_pile)
    next_view = replace(view, hand=new_hand, deck_count=new_deck_count, deck_instances=current_state)

    if player == "me":
        return replace(snapshot, me=next_view)
    return replace(snapshot, opponent=next_view)


def draw_next_from_deck_instances(snapshot: BattleSnapshot, player: str) -> tuple[CardView | None, BattleSnapshot]:
    view = snapshot.me if player == "me" else snapshot.opponent if player == "opponent" else None
    if view is None:
        raise ValueError(f"unsupported player: {player}")
    state = view.deck_instances
    if state is None or not state.draw_pile:
        return None, snapshot
    instance_id = state.draw_pile[0]
    instance = state.by_instance_id()[instance_id]
    card = instance.card if instance.uid is None else replace(instance.card, uid=instance.uid)
    moved = state.move(instance_id, "hand", uid=card.uid, card=card, visibility="visible")
    next_view = replace(view, deck_instances=moved)
    if player == "me":
        return card, replace(snapshot, me=next_view)
    return card, replace(snapshot, opponent=next_view)
