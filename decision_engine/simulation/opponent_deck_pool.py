# -*- coding: utf-8 -*-
"""Opponent meta deck pool construction for hidden-knowledge branches."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from decision_engine.meta.data_source import MetaDataSource
from decision_engine.meta.models import CLASS_MAP, CardSlot, DeckEntry
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import CardType, DeckPoolCard, DeckPoolView


@dataclass(frozen=True)
class OpponentDeckPoolConfig:
    format: str = "standard"
    max_decks_per_class: int = 8
    min_win_streak: int = 0
    max_candidate_cards: int = 40
    require_exact_class: bool = True
    max_copies_per_deck: int = 3
    latest_card_set_id: int | None = None
    card_database_path: str | None = None
    unlimited_card_set_min: int = 10001


@dataclass(frozen=True)
class OpponentDeckPoolBuildResult:
    deck_pool: DeckPoolView | None
    certainty: PredictionCertainty
    assumptions: tuple[str, ...] = ()
    unresolved_effects: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class OpponentDeckPoolBuilder:
    def __init__(self, data_source: MetaDataSource | None = None):
        self.data_source = data_source

    def build_for_class(self, class_en: str, config: OpponentDeckPoolConfig | None = None) -> OpponentDeckPoolBuildResult:
        config = config or OpponentDeckPoolConfig()
        normalized_class = (class_en or "").strip()
        if not normalized_class:
            return self._error(
                "opponent_class_required_for_deck_pool",
                unresolved=("opponent_class_unknown",),
            )
        if config.require_exact_class and normalized_class not in CLASS_MAP:
            return self._error(
                f"opponent_class_not_supported:{normalized_class}",
                unresolved=("opponent_class_unknown",),
            )
        try:
            data_source = self.data_source or MetaDataSource()
            decks = data_source.load_all_decks()
        except FileNotFoundError as exc:
            return self._error(
                f"opponent_meta_deck_db_unavailable:{exc}",
                unresolved=("opponent_meta_deck_pool_unavailable",),
            )
        selected = self._select_decks(decks, normalized_class, config)
        return self.build_from_decks(
            tuple(selected),
            config,
            source=f"meta.yokidou_decks.db:class={normalized_class}",
            assumptions=("opponent_candidate_pool_from_meta", f"opponent_class_en={normalized_class}"),
        )

    def build_from_decks(
        self,
        decks: tuple[DeckEntry, ...],
        config: OpponentDeckPoolConfig | None = None,
        *,
        source: str = "opponent_deck_hypotheses",
        assumptions: tuple[str, ...] = ("opponent_candidate_pool_from_deck_hypotheses",),
    ) -> OpponentDeckPoolBuildResult:
        config = config or OpponentDeckPoolConfig()
        format_range = self._format_card_set_range(config)
        if format_range is None:
            return self._error(
                "opponent_meta_latest_card_set_unavailable",
                unresolved=("opponent_meta_format_range_unavailable",),
            )
        if not decks:
            return self._unknown(
                f"opponent_meta_deck_pool_empty:source={source}",
                unresolved=("opponent_candidate_pool_empty",),
            )

        cards, warnings = self._aggregate_cards(list(decks), config, format_range)
        if not cards:
            return self._unknown(
                f"opponent_meta_deck_pool_no_valid_cards:source={source}",
                unresolved=("opponent_candidate_pool_empty",),
                warnings=warnings,
            )

        total_weight = sum(card.count for card in cards)
        pool = DeckPoolView(
            source=f"{source}:format={config.format}:sets={format_range[0]}-{format_range[1]}:decks={len(decks)}:max_decks={config.max_decks_per_class}",
            cards=tuple(cards),
            observed_total=total_weight,
            remaining_count=total_weight,
            status="ambiguous",
            warnings=tuple(warnings),
        )
        return OpponentDeckPoolBuildResult(
            deck_pool=pool,
            certainty=PredictionCertainty.PARTIAL,
            assumptions=assumptions
            + (
                f"opponent_meta_format={config.format}",
                f"opponent_meta_card_set_range={format_range[0]}-{format_range[1]}",
                f"opponent_deck_hypothesis_count={len(decks)}",
                "opponent_deck_pool_is_ambiguous_meta_aggregate",
                "opponent_deck_pool_count_is_adoption_rate_percent",
            ),
            unresolved_effects=("opponent_actual_deck_unknown", "opponent_hidden_hand_full_state"),
            warnings=tuple(warnings),
        )

    def _select_decks(self, decks: list[DeckEntry], class_en: str, config: OpponentDeckPoolConfig) -> list[DeckEntry]:
        selected = [
            deck
            for deck in decks
            if deck.class_en == class_en
            and deck.format == config.format
            and deck.win_streak >= config.min_win_streak
        ]
        selected.sort(key=lambda deck: (deck.win_streak, deck.article_date), reverse=True)
        return selected[: max(config.max_decks_per_class, 0)]

    def _aggregate_cards(self, decks: list[DeckEntry], config: OpponentDeckPoolConfig, format_range: tuple[int, int]) -> tuple[list[DeckPoolCard], tuple[str, ...]]:
        copies: dict[int, int] = defaultdict(int)
        names: dict[int, str] = {}
        costs: dict[int, int] = {}
        types: dict[int, CardType | None] = {}
        warnings: list[str] = []
        for deck in decks:
            for slot in deck.cards:
                card_id = self._card_id(slot, warnings)
                if card_id <= 0:
                    continue
                if not self._is_card_in_format(slot, format_range):
                    warnings.append(f"opponent_candidate_card_out_of_format:{card_id}:set={slot.card_set_id}")
                    continue
                copies[card_id] += max(int(slot.count), 0)
                names.setdefault(card_id, slot.name)
                costs.setdefault(card_id, slot.cost)
                types.setdefault(card_id, self._card_type(slot))

        denominator = max(len(decks) * max(config.max_copies_per_deck, 1), 1)
        cards = [
            DeckPoolCard(
                card_id=card_id,
                count=min(100, round(total_copies * 100 / denominator)),
                name=names.get(card_id, ""),
                cost=costs.get(card_id, 0),
                card_type=types.get(card_id),
            )
            for card_id, total_copies in copies.items()
            if total_copies > 0
        ]
        cards.sort(key=lambda card: (-card.count, card.cost, card.card_id))
        return cards[: max(config.max_candidate_cards, 0)], tuple(dict.fromkeys(warnings))

    def _format_card_set_range(self, config: OpponentDeckPoolConfig) -> tuple[int, int] | None:
        latest = config.latest_card_set_id or self._latest_card_set_id(config)
        if latest is None:
            return None
        if config.format == "standard":
            return (max(config.unlimited_card_set_min, latest - 4), latest)
        if config.format == "unlimited":
            return (config.unlimited_card_set_min, latest)
        return (config.unlimited_card_set_min, latest)

    def _latest_card_set_id(self, config: OpponentDeckPoolConfig) -> int | None:
        if self.data_source is not None:
            latest = self.data_source.get_latest_set_id()
            if latest is not None:
                return int(latest)
        path = Path(config.card_database_path) if config.card_database_path else Path(__file__).resolve().parents[1] / "data" / "public" / "card_database.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        entries = raw.get("cards", raw) if isinstance(raw, dict) else raw
        card_set_ids = [int(entry.get("card_set_id", 0) or 0) for entry in entries if isinstance(entry, dict)]
        valid = [card_set_id for card_set_id in card_set_ids if card_set_id >= config.unlimited_card_set_min]
        return max(valid, default=None)

    def _is_card_in_format(self, slot: CardSlot, format_range: tuple[int, int]) -> bool:
        if slot.card_set_id is None:
            return True
        return format_range[0] <= int(slot.card_set_id) <= format_range[1]

    def _card_id(self, slot: CardSlot, warnings: list[str]) -> int:
        try:
            card_id = int(slot.cid)
        except (TypeError, ValueError):
            warnings.append(f"opponent_candidate_card_id_invalid:{slot.cid}")
            return 0
        if card_id <= 0:
            warnings.append(f"opponent_candidate_card_id_invalid:{slot.cid}")
        return card_id

    def _card_type(self, slot: CardSlot) -> CardType | None:
        normalized = (slot.type_name or "").strip().lower()
        if normalized in {"spell", "法术", "法術"}:
            return CardType.SPELL
        if normalized in {"amulet", "护符", "護符"}:
            return CardType.AMULET
        if normalized in {"follower", "随从", "從者", "從者/衍生物"}:
            return CardType.FOLLOWER
        return None

    def _error(
        self,
        description: str,
        *,
        unresolved: tuple[str, ...],
        warnings: tuple[str, ...] = (),
    ) -> OpponentDeckPoolBuildResult:
        return OpponentDeckPoolBuildResult(
            deck_pool=None,
            certainty=PredictionCertainty.PREDICT_ERROR,
            assumptions=(description,),
            unresolved_effects=unresolved,
            warnings=warnings,
        )

    def _unknown(
        self,
        description: str,
        *,
        unresolved: tuple[str, ...],
        warnings: tuple[str, ...] = (),
    ) -> OpponentDeckPoolBuildResult:
        return OpponentDeckPoolBuildResult(
            deck_pool=None,
            certainty=PredictionCertainty.UNKNOWN,
            assumptions=(description,),
            unresolved_effects=unresolved,
            warnings=warnings,
        )
