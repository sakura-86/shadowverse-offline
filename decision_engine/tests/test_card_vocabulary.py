# -*- coding: utf-8 -*-
"""Regression tests for stable base-card identity embeddings."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.state.snapshot import BattleSnapshot, CardView, PlayerView
from decision_engine.training.card_vocabulary import CardVocabulary, build_card_vocabulary
from decision_engine.training.example_builder import build_training_example
from decision_engine.training.structured_tensorization import CARD_STAT_DIM, structured_tensorize_training_example
from decision_engine.training.tensorization import ACTION_VECTOR_FEATURE_ORDER


class CardVocabularyTest(unittest.TestCase):
    def test_append_only_extension_preserves_existing_indices(self) -> None:
        vocabulary = CardVocabulary({100: 2, 200: 3})
        extended = vocabulary.extend([50, 200, 300])
        self.assertEqual(extended.index_for_base_card_id(100), 2)
        self.assertEqual(extended.index_for_base_card_id(200), 3)
        self.assertEqual(extended.index_for_base_card_id(50), 4)
        self.assertEqual(extended.index_for_base_card_id(300), 5)

    def test_initial_build_is_independent_of_database_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cards.json"
            path.write_text(json.dumps({"cards": [{"base_card_id": 300}, {"base_card_id": 100}, {"base_card_id": 200}]}), encoding="utf-8")
            vocabulary = build_card_vocabulary(path)
        self.assertEqual(vocabulary.base_card_id_to_index, {100: 2, 200: 3, 300: 4})


class BaseCardIdentityTensorizationTest(unittest.TestCase):
    def test_different_card_ids_share_the_same_base_card_embedding_index(self) -> None:
        vocabulary = CardVocabulary({1000: 2, 2000: 3})
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=3, max_pp=3, hand=(10,), field=(11,), turn=3),
            opponent=PlayerView(life=20, max_life=20, pp=3, max_pp=3, field=(20,), turn=3),
            cards_by_uid={
                10: CardView(uid=10, card_id=1001, base_card_id=1000, cost=3),
                11: CardView(uid=11, card_id=1002, base_card_id=1000, cost=3, atk=4, life=4, max_life=4),
                20: CardView(uid=20, card_id=2001, base_card_id=2000, cost=2, atk=2, life=2, max_life=2),
            },
            is_ally_turn=True,
        )
        actions = (LegalAction(kind=ActionKind.PLAY, source_uid=10, cost=3), LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=11, target_groups=((20,),)))
        example = build_training_example(
            snapshot=snapshot, legal_actions=actions, sample_id="base-id", source={"kind": "unit_test"},
            trust={"level": "trusted"}, selected_action_index=0, value_target=1.0,
            generation_config={"kind": "pvp_real_extraction"},
        ).to_jsonable()
        tensor = structured_tensorize_training_example(example, vocabulary=vocabulary)
        self.assertEqual(tensor.hand_card_ids[0], 2)
        self.assertEqual(tensor.field_card_ids[0], 2)
        self.assertEqual(tensor.action_source_card_ids, (2, 2))
        self.assertEqual(tensor.opp_field_card_ids[0], 3)
        self.assertEqual(tensor.action_target_card_ids[1][0], 3)


class StructuredWarmStartTest(unittest.TestCase):
    def test_warm_start_preserves_old_embedding_rows_and_appends_new_card(self) -> None:
        import torch
        from decision_engine.training.structured_policy_value import StructuredModelConfig, train_structured_policy_value

        old_vocabulary = CardVocabulary({1000: 2})
        new_vocabulary = old_vocabulary.extend([2000])
        rows = [_tensor_row()]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = train_structured_policy_value(
                dataset={"train": rows, "val": rows}, output_dir=root / "old", run_id="old",
                vocab_size=old_vocabulary.vocab_size, vocabulary=old_vocabulary, epochs=1,
                batch_size=1, learning_rate=1e-3, device="cpu", config=StructuredModelConfig(),
            )
            old_checkpoint = torch.load(first.checkpoint_path, map_location="cpu", weights_only=True)
            second = train_structured_policy_value(
                dataset={"train": rows, "val": rows}, output_dir=root / "new", run_id="new",
                vocab_size=new_vocabulary.vocab_size, vocabulary=new_vocabulary, initial_checkpoint=first.checkpoint_path,
                epochs=1, batch_size=1, learning_rate=0.0, device="cpu", config=StructuredModelConfig(),
            )
            new_checkpoint = torch.load(second.checkpoint_path, map_location="cpu", weights_only=True)
        for key in ("card_encoder.embedding.weight", "crest_embedding.weight"):
            self.assertTrue(torch.equal(old_checkpoint["model_state_dict"][key], new_checkpoint["model_state_dict"][key][:old_vocabulary.vocab_size]))
        self.assertEqual(second.train_metrics["warm_start"]["added_base_cards"], 1)


def _tensor_row() -> dict:
    return {
        "state_vector": (0.0, 1.0),
        "hand_card_ids": (2,), "hand_card_stats": ((0.0,) * CARD_STAT_DIM,), "hand_card_mask": (True,),
        "field_card_ids": (0,), "field_card_stats": ((0.0,) * CARD_STAT_DIM,), "field_card_mask": (False,),
        "opp_hand_card_ids": (0,), "opp_hand_card_stats": ((0.0,) * CARD_STAT_DIM,), "opp_hand_card_mask": (False,),
        "opp_field_card_ids": (0,), "opp_field_card_stats": ((0.0,) * CARD_STAT_DIM,), "opp_field_card_mask": (False,),
        "crest_card_ids": (0,), "crest_stats": ((0.0,) * 3,), "crest_mask": (False,),
        "opp_crest_card_ids": (0,), "opp_crest_stats": ((0.0,) * 3,), "opp_crest_mask": (False,),
        "action_scalar_vectors": ((0.0,) * len(ACTION_VECTOR_FEATURE_ORDER),),
        "action_source_card_ids": (2,), "action_source_card_stats": ((0.0,) * CARD_STAT_DIM,),
        "action_target_card_ids": ((0,),), "action_target_card_stats": (((0.0,) * CARD_STAT_DIM,),), "action_target_mask": ((False,),),
        "history_state_vectors": ((0.0, 0.0),) * 4, "history_mask": (False,) * 4,
        "policy_target": (1.0,), "value_target": 1.0,
    }
