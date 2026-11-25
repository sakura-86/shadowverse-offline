# -*- coding: utf-8 -*-
"""TrainingExample builder tests."""

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.state.snapshot import BattleSnapshot, CardView, PlayerView
from decision_engine.training.example_builder import build_training_example


class TrainingExampleBuilderTest(unittest.TestCase):
    def test_build_training_example_uses_selected_action_one_hot_policy(self) -> None:
        snapshot = _snapshot()
        actions = _actions()

        example = build_training_example(
            snapshot=snapshot,
            legal_actions=actions,
            selected_action=actions[0],
            sample_id="example-1",
            source={"kind": "unit_test"},
            trust={"level": "trusted"},
            generation_config={"kind": "policy_value_tensorization", "source": "unit_test"},
        )
        payload = example.to_jsonable()

        self.assertEqual(payload["sample_id"], "example-1")
        self.assertEqual(payload["selected_action_index"], 0)
        self.assertEqual(payload["policy_target"], [1.0, 0.0])
        self.assertEqual(payload["turn_index"], 3)
        self.assertEqual(payload["state_features"]["player"]["pp"], 3)
        self.assertEqual(payload["legal_action_features"][0]["action_kind"], "play")
        self.assertEqual(payload["generation_config"], {"kind": "policy_value_tensorization", "source": "unit_test"})

    def test_build_training_example_normalizes_visit_counts(self) -> None:
        snapshot = _snapshot()
        actions = _actions()

        example = build_training_example(
            snapshot=snapshot,
            legal_actions=actions,
            selected_action_index=1,
            visit_counts=(3, 1),
            sample_id="example-visits",
            source={"kind": "unit_test"},
            trust={"level": "self_play_search"},
            value_target=-1.0,
            generation_config={"kind": "self_play_search", "source": "unit_test"},
        )

        self.assertEqual(example.policy_target, (0.75, 0.25))
        self.assertEqual(example.selected_action_index, 1)
        self.assertEqual(example.value_target, -1.0)

    def test_build_training_example_allows_unselected_policy_target(self) -> None:
        snapshot = _snapshot()
        actions = _actions()

        example = build_training_example(
            snapshot=snapshot,
            legal_actions=actions,
            policy_target=(0.4, 0.6),
            sample_id="example-policy",
            source={"kind": "unit_test"},
            trust={"level": "search_distribution"},
            generation_config={"kind": "self_play_search", "source": "unit_test"},
        )

        self.assertIsNone(example.selected_action_index)
        self.assertEqual(example.policy_target, (0.4, 0.6))

    def test_build_training_example_rejects_missing_selected_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "selected_action is not present"):
            build_training_example(
                snapshot=_snapshot(),
                legal_actions=_actions(),
                selected_action=LegalAction(kind=ActionKind.PLAY, source_uid=99),
                sample_id="bad",
                source={},
                trust={},
                generation_config={"kind": "policy_value_tensorization", "source": "unit_test"},
            )

    def test_build_training_example_rejects_disagreeing_selected_action_index(self) -> None:
        actions = _actions()
        with self.assertRaisesRegex(ValueError, "disagree"):
            build_training_example(
                snapshot=_snapshot(),
                legal_actions=actions,
                selected_action=actions[0],
                selected_action_index=1,
                sample_id="bad",
                source={},
                trust={},
                generation_config={"kind": "policy_value_tensorization", "source": "unit_test"},
            )

    def test_build_training_example_rejects_bad_visit_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive total"):
            build_training_example(
                snapshot=_snapshot(),
                legal_actions=_actions(),
                visit_counts=(0, 0),
                sample_id="bad",
                source={},
                trust={},
                generation_config={"kind": "policy_value_tensorization", "source": "unit_test"},
            )


def _snapshot() -> BattleSnapshot:
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=3, max_pp=3, hand=(10,), field=(), turn=3),
        opponent=PlayerView(life=20, max_life=20, pp=2, max_pp=2, field=(20,), turn=3),
        cards_by_uid={
            10: CardView(uid=10, card_id=1001, base_card_id=1000, cost=3),
            20: CardView(uid=20, card_id=2001, base_card_id=2000, cost=2, atk=2, life=2, max_life=2),
        },
        is_ally_turn=True,
    )


def _actions() -> tuple[LegalAction, ...]:
    return (
        LegalAction(kind=ActionKind.PLAY, source_uid=10, cost=3),
        LegalAction(kind=ActionKind.TURN_END, source_uid=0),
    )


if __name__ == "__main__":
    unittest.main()
