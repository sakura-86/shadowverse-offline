# -*- coding: utf-8 -*-
"""资源不足导致效果未发动时的 certainty 语义。

资源不足是确定的「效果未发动」：整条效果已回滚，盘面合法，不是预测缺口。
应记 SPEC_ONLY，而不是 PARTIAL / predict_error。
"""

from __future__ import annotations

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.cards.templates import (
    AmountFormula,
    CardIdentity,
    ExecutableEffect,
    OperationSpec,
    PrintedStats,
    TriggerSpec,
    UnifiedCardTemplate,
)
from decision_engine.execution import LocalPredictor
from decision_engine.execution.player_local_predictor import PlayerLocalPredictor
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView


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


class InsufficientResourceCertaintyTest(unittest.TestCase):
    def test_necromancy_insufficient_is_spec_only(self) -> None:
        effect = ExecutableEffect(
            effect_id="9001_fanfare_necromancy",
            trigger=TriggerSpec(event="fanfare"),
            operations=(OperationSpec(op="consume_necromancy", amount=AmountFormula(kind="fixed", value=2)),),
        )
        card = CardView(
            uid=1,
            card_id=9001,
            base_card_id=9001,
            name="唤灵随从",
            cost=1,
            atk=1,
            life=1,
            max_life=1,
            card_type=CardType.FOLLOWER,
        )
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=5, max_pp=5, graveyard_count=0, hand=(1,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0),
            cards_by_uid={1: card},
            is_ally_turn=True,
            source="test",
        )
        predictor = PlayerLocalPredictor(
            LocalPredictor(templates_by_base_card_id={9001: _template(9001, effect)})
        )

        result = predictor.predict(snapshot, LegalAction(kind=ActionKind.PLAY, source_uid=1))

        self.assertEqual(result.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertIn("not activated (insufficient resource)", result.description)
        self.assertEqual(result.unresolved_effects, ())
        # 效果未发动：墓地计数不变。
        self.assertEqual(result.snapshot.me.graveyard_count, 0)


if __name__ == "__main__":
    unittest.main()
