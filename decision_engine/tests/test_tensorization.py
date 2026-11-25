# -*- coding: utf-8 -*-
"""Tensorization tests for policy/value training examples."""

from __future__ import annotations

import unittest

from decision_engine.training.tensorization import tensorize_training_example, tensorize_training_examples


class TensorizationTest(unittest.TestCase):
    def test_tensorize_training_example_returns_stable_vectors(self) -> None:
        result = tensorize_training_example(_example())

        self.assertEqual(result.schema, "TrainingExampleTensorization")
        self.assertEqual(result.feature_shapes["state_vector"], len(result.state_vector))
        self.assertEqual(result.feature_shapes["action_vectors"], [len(vector) for vector in result.action_vectors])
        self.assertEqual(result.labels["policy_target_width"], 2)
        self.assertEqual(result.value_target, 1.0)

    def test_tensorize_training_examples_exports_jsonable_rows(self) -> None:
        rows = tensorize_training_examples([_example()])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["schema"], "TrainingExampleTensorization")
        self.assertEqual(rows[0]["sample_id"], "sample-1")
        self.assertIn("state_vector", rows[0])
        self.assertIn("action_vectors", rows[0])


def _example() -> dict:
    return {
        "schema": "TrainingExample",
        "version": 1,
        "sample_id": "sample-1",
        "source": {"kind": "unit_test"},
        "trust": {"level": "self_play_search", "accepted": True},
        "player": "me",
        "turn_index": 3,
        "state_features": {
            "schema": "StateFeatures",
            "version": 1,
            "perspective": "me",
            "is_ally_turn": True,
            "is_turn_ended": False,
            "current_player": "me",
            "player": {
                "life": 20,
                "max_life": 20,
                "pp": 5,
                "max_pp": 5,
                "ep": 2,
                "sep": 1,
                "max_ep": 2,
                "max_sep": 1,
                "extra_pp": {"present": True, "extra_pp": 1, "preparation_extra_pp": 0, "extra_pp_state": 1, "restore_extra_pp_turn": 6, "is_used_extra_pp_this_turn": False},
                "turn": 3,
                "is_first_side": True,
                "is_kakusei": False,
                "is_evolved_this_turn": False,
                "manual_evolve_count": 0,
                "evolve_count": 0,
                "hand_count": 4,
                "field_count": 2,
                "deck_count": 20,
                "graveyard_count": 3,
                "rally": 5,
                "play_count": 2,
                "crest_count": 1,
                "extra_crest_count": 0,
                "boon_count": 0,
                "class_counters": {"rally": 5, "play_count": 2, "graveyard_count": 3, "is_kakusei": False, "spellboost_total_in_hand": 1, "spellboost_cards_in_hand": 1, "earth_sigil_stack_total": 0},
            },
            "opponent": {
                "life": 18,
                "max_life": 20,
                "pp": 4,
                "max_pp": 4,
                "ep": 0,
                "sep": 0,
                "hand_count": 3,
                "field_count": 1,
                "deck_count": 19,
                "graveyard_count": 4,
                "rally": 3,
                "play_count": 1,
                "crest_count": 0,
                "extra_crest_count": 0,
                "boon_count": 0,
            },
            "pressure": {"player_field_attack_total": 6, "opponent_field_attack_total": 3, "opponent_guard_count": 1},
            "legal_summary": {"present": True, "can_play_count": 2, "can_attack_leader_count": 1, "can_attack_field_count": 1, "can_evolve_count": 1, "can_super_evolve_count": 0, "can_activate_count": 0, "can_fusion_count": 0, "can_end_turn": True},
        },
        "legal_action_features": [
            {"schema": "ActionFeatures", "version": 1, "action_kind": "play", "cost": 5, "pp_before_action": 5, "unused_pp_after_action": 0, "uses_full_pp": True, "leaves_one_pp": False, "target_group_count": 0, "target_count": 0, "random_target_group_count": 0, "random_target_count": 0, "selected_index_count": 0, "material_count": 0, "has_target": False, "has_materials": False, "is_terminal_action": False, "is_resource_action": True, "is_combat_action": False, "is_evolve_action": False},
            {"schema": "ActionFeatures", "version": 1, "action_kind": "turn_end", "cost": 0, "pp_before_action": 5, "unused_pp_after_action": 5, "uses_full_pp": False, "leaves_one_pp": False, "target_group_count": 0, "target_count": 0, "random_target_group_count": 0, "random_target_count": 0, "selected_index_count": 0, "material_count": 0, "has_target": False, "has_materials": False, "is_terminal_action": True, "is_resource_action": False, "is_combat_action": False, "is_evolve_action": False},
        ],
        "selected_action_index": 0,
        "policy_target": [1.0, 0.0],
        "value_target": 1.0,
        "generation_config": {"kind": "policy_value_tensorization", "source": "unit_test"},
        "metadata": {"kind": "unit_test"},
    }


if __name__ == "__main__":
    unittest.main()
