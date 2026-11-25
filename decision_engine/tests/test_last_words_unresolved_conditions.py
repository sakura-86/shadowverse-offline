# -*- coding: utf-8 -*-
"""谢幕曲 unresolved 判定：条件不满足时是合法「效果不发动」。"""

from __future__ import annotations

import unittest

from decision_engine.cards.templates import (
    CardIdentity,
    ConditionSpec,
    ExecutableEffect,
    OperationSpec,
    PrintedStats,
    TriggerSpec,
    UnifiedCardTemplate,
)
from decision_engine.execution import LocalPredictor
from decision_engine.execution.events import ExecutionEvent
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView

BASE_ID = 9002


def _template(base_card_id: int, effect: ExecutableEffect) -> UnifiedCardTemplate:
    return UnifiedCardTemplate(
        template_version=1,
        identity=CardIdentity(
            base_card_id=base_card_id,
            name=f"card-{base_card_id}",
            type_id=1,
            type_name="follower",
            class_id=0,
            class_name="test",
            rarity=0,
            rarity_name="test",
        ),
        printed=PrintedStats(cost=1, atk=1, life=1),
        executable_effects=(effect,),
    )


def _snapshot(graveyard_count: int) -> BattleSnapshot:
    card = CardView(
        uid=5,
        card_id=BASE_ID,
        base_card_id=BASE_ID,
        name="墓地随从",
        atk=1,
        life=1,
        max_life=1,
        card_type=CardType.FOLLOWER,
        has_last_words=True,
    )
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=0, max_pp=0, graveyard_count=graveyard_count, field=(5,)),
        opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0),
        cards_by_uid={5: card},
        source="test",
    )


class LastWordsUnresolvedConditionsTest(unittest.TestCase):
    def setUp(self) -> None:
        effect = ExecutableEffect(
            effect_id=f"{BASE_ID}_last_words",
            trigger=TriggerSpec(event="last_words"),
            conditions=(ConditionSpec(kind="snapshot_threshold", subject="me.graveyard_count", operator=">=", value=2),),
            operations=(OperationSpec(op="draw", target_ref="self_player", amount=None),),
        )
        self.predictor = LocalPredictor(templates_by_base_card_id={BASE_ID: _template(BASE_ID, effect)})
        self.events = (ExecutionEvent("last_words", source_uid=5),)

    def test_condition_not_met_is_not_unresolved(self) -> None:
        unresolved = self.predictor._last_words_unresolved_from_events(_snapshot(0), self.events)
        self.assertEqual(unresolved, ())

    def test_condition_met_is_candidate(self) -> None:
        unresolved = self.predictor._last_words_unresolved_from_events(_snapshot(5), self.events)
        self.assertEqual(len(unresolved), 1)
        self.assertIn("last_words:uid=5", unresolved[0])


if __name__ == "__main__":
    unittest.main()
