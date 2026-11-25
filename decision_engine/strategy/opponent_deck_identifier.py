# -*- coding: utf-8 -*-
"""Opponent deck hypothesis identification from observed cards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from decision_engine.meta.data_source import MetaDataSource
from decision_engine.meta.models import CLASS_MAP, CardSlot, DeckEntry
from decision_engine.state.snapshot import BattleSnapshot

from .opponent_observation import OpponentObservedCards, extract_opponent_observed_cards


@dataclass(frozen=True)
class OpponentDeckIdentifierConfig:
    format: str = "standard"
    max_decks_per_class: int = 30
    top_n: int = 5
    min_win_streak: int = 0
    match_weight: float = 10.0
    missing_seen_penalty: float = 30.0
    win_streak_weight: float = 0.2
    recency_weight: float = 0.05
    require_exact_class: bool = True


@dataclass(frozen=True)
class DeckHypothesis:
    deck: DeckEntry
    score: float
    matched_card_ids: tuple[int, ...]
    missing_seen_card_ids: tuple[int, ...]
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class OpponentDeckIdentifier:
    def __init__(self, data_source: MetaDataSource | None = None):
        self.data_source = data_source

    def identify(
        self,
        snapshot: BattleSnapshot,
        *,
        opponent_class_en: str,
        config: OpponentDeckIdentifierConfig | None = None,
    ) -> tuple[DeckHypothesis, ...]:
        config = config or OpponentDeckIdentifierConfig()
        class_en = (opponent_class_en or "").strip()
        if not class_en:
            return ()
        if config.require_exact_class and class_en not in CLASS_MAP:
            return ()
        data_source = self.data_source or MetaDataSource()
        decks = data_source.load_all_decks()
        candidates = self._select_decks(decks, class_en, config)
        observed = extract_opponent_observed_cards(snapshot)
        hypotheses = [self._score_deck(deck, observed, config) for deck in candidates]
        hypotheses.sort(key=lambda item: item.score, reverse=True)
        return tuple(hypotheses[: max(config.top_n, 0)])

    def _select_decks(self, decks: list[DeckEntry], class_en: str, config: OpponentDeckIdentifierConfig) -> list[DeckEntry]:
        selected = [
            deck
            for deck in decks
            if deck.class_en == class_en
            and deck.format == config.format
            and deck.win_streak >= config.min_win_streak
        ]
        selected.sort(key=lambda deck: (deck.win_streak, deck.article_date), reverse=True)
        return selected[: max(config.max_decks_per_class, 0)]

    def _score_deck(self, deck: DeckEntry, observed: OpponentObservedCards, config: OpponentDeckIdentifierConfig) -> DeckHypothesis:
        deck_card_ids = self._deck_card_ids(deck.cards)
        observed_card_ids = frozenset(observed.card_ids)
        matched = tuple(sorted(observed_card_ids & deck_card_ids))
        missing = tuple(sorted(observed_card_ids - deck_card_ids))
        assumptions = ["opponent_deck_hypothesis_from_observed_cards"]
        if not observed_card_ids:
            assumptions.append("opponent_deck_identification_without_seen_cards")
        score = len(matched) * config.match_weight
        score -= len(missing) * config.missing_seen_penalty
        score += deck.win_streak * config.win_streak_weight
        score += self._recency_score(deck.article_date) * config.recency_weight
        return DeckHypothesis(
            deck=deck,
            score=score,
            matched_card_ids=matched,
            missing_seen_card_ids=missing,
            assumptions=tuple(assumptions),
            warnings=observed.warnings,
        )

    def _deck_card_ids(self, cards: list[CardSlot]) -> frozenset[int]:
        ids: set[int] = set()
        for slot in cards:
            try:
                card_id = int(slot.cid)
            except (TypeError, ValueError):
                continue
            if card_id > 0:
                ids.add(card_id)
        return frozenset(ids)

    def _recency_score(self, value: str) -> float:
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            return 0.0
        return parsed.toordinal() / 1000.0
