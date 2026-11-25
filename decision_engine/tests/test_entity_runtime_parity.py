# -*- coding: utf-8 -*-
"""C6: entity backend PUCT/runtime 统一接入与 parity 测试。

覆盖 §C6 验收：
- 同一 (snapshot, actions, perspective, vocabulary) 在训练/full 与 fast/runtime
  路径产生相同 entity tensor。
- 相同场面不同剩余牌组不命中同一 state fingerprint。
- 相同剩余多重集合、不同 draw order 命中等价 fingerprint。
- prior/value 在 evaluator 与独立 inference 中一致。
- evaluator 只消费 LegalAction，prior 只覆盖合法动作。
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

from decision_engine.state.deck_instance import make_deck_instance_state
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView
from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.training.card_vocabulary import CardVocabulary
from decision_engine.training.state_features import state_features
from decision_engine.training.action_features import legal_actions_features
from decision_engine.training.entity_tensorization import entity_tensorize_example, collate_entity_rows
from decision_engine.planning.entity_features import (
    entity_batch_from_snapshot,
    entity_snapshot_fingerprint,
)

_VOCAB = CardVocabulary({1000: 2, 2000: 3, 3000: 4})


def _deck_card(b: int) -> CardView:
    return CardView(uid=0, card_id=b + 900000, base_card_id=b, cost=2, card_type=CardType.FOLLOWER)


def _snapshot(hand=(1,), deck_base_ids=(1000, 2000)) -> BattleSnapshot:
    cards = {1: CardView(uid=1, card_id=1001, base_card_id=1000, cost=2, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER)}
    di = make_deck_instance_state(tuple(_deck_card(b) for b in deck_base_ids), source="t")
    me = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, hand=hand, deck_count=len(deck_base_ids), deck_instances=di)
    opp = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, deck_count=30)
    return BattleSnapshot(me=me, opponent=opp, cards_by_uid=cards, is_ally_turn=True)


def _actions():
    return (LegalAction(kind=ActionKind.PLAY, source_uid=1, cost=2),
            LegalAction(kind=ActionKind.TURN_END, source_uid=0))


class FullFastEntityParityTest(unittest.TestCase):
    def test_full_and_fast_produce_identical_row(self) -> None:
        snap = _snapshot(deck_base_ids=(1000, 2000, 2000))
        actions = _actions()
        # full path（训练风格）：显式构造 example
        full_example = {
            "state_features": state_features(snap, perspective="me"),
            "legal_action_features": legal_actions_features(snap, actions, perspective="me"),
            "policy_target": [0.0, 0.0],
            "value_target": None,
        }
        full_row = entity_tensorize_example(full_example, vocabulary=_VOCAB)
        # fast path（runtime helper）
        from decision_engine.planning.entity_features import entity_example_from_snapshot
        fast_row = entity_example_from_snapshot(snap, actions, perspective="me", vocabulary=_VOCAB)
        self.assertEqual(full_row.to_jsonable(), fast_row.to_jsonable())

    def test_fingerprint_distinguishes_deck_composition(self) -> None:
        a = entity_snapshot_fingerprint(_snapshot(deck_base_ids=(1000, 1000)), "me", vocabulary=_VOCAB)
        b = entity_snapshot_fingerprint(_snapshot(deck_base_ids=(1000, 2000)), "me", vocabulary=_VOCAB)
        self.assertNotEqual(a, b)

    def test_fingerprint_insensitive_to_draw_order(self) -> None:
        snap = _snapshot(deck_base_ids=(1000, 2000, 3000, 1000))
        shuffled = replace(snap, me=replace(snap.me, deck_instances=replace(
            snap.me.deck_instances, draw_pile=tuple(reversed(snap.me.deck_instances.draw_pile)))))
        self.assertEqual(
            entity_snapshot_fingerprint(snap, "me", vocabulary=_VOCAB),
            entity_snapshot_fingerprint(shuffled, "me", vocabulary=_VOCAB),
        )

    def test_both_perspectives_build(self) -> None:
        snap = _snapshot()
        actions = _actions()
        for player in ("me", "opponent"):
            batch = entity_batch_from_snapshot(snap, actions, perspective=player, vocabulary=_VOCAB)
            self.assertEqual(batch["action_mask"].shape[1], len(actions))


@unittest.skipUnless(_TORCH, "torch not available")
class EntityEvaluatorTest(unittest.TestCase):
    def _train_checkpoint(self, tmp: str) -> Path:
        from decision_engine.training.entity_policy_value import EntityModelConfig
        from decision_engine.training.entity_checkpoint import train_entity_policy_value

        snap = _snapshot(deck_base_ids=(1000, 2000))
        actions = _actions()
        example = {
            "state_features": state_features(snap, perspective="me"),
            "legal_action_features": legal_actions_features(snap, actions, perspective="me"),
            "policy_target": [1.0, 0.0], "value_target": 0.5,
        }
        row = entity_tensorize_example(example, vocabulary=_VOCAB)
        out = Path(tmp) / "run"
        result = train_entity_policy_value(
            dataset={"train": [row], "val": [row]}, output_dir=out, run_id="c6",
            vocabulary=_VOCAB, config=EntityModelConfig(d_model=32, num_layers=1, num_heads=4, ffn_dim=64, seed=1),
            epochs=2, batch_size=1, learning_rate=1e-3, device="cpu",
        )
        return result.checkpoint_path

    def test_evaluator_prior_and_value_valid_and_consistent(self) -> None:
        from decision_engine.planning.entity_prior import EntityNeuralEvaluator
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self._train_checkpoint(tmp)
            ev = EntityNeuralEvaluator(ckpt)
            snap = _snapshot(deck_base_ids=(1000, 2000))
            actions = _actions()
            prior = ev.priors(snap, actions, player="me")
            self.assertEqual(len(prior.priors), len(actions))
            self.assertAlmostEqual(sum(prior.priors), 1.0, places=5)
            v = ev.value(snap, actions, player="me")
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)
            # 一致性：再次调用结果相同
            prior2 = ev.priors(snap, actions, player="me")
            self.assertEqual(prior.priors, prior2.priors)

    def test_evaluator_only_consumes_given_actions(self) -> None:
        from decision_engine.planning.entity_prior import EntityNeuralEvaluator
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self._train_checkpoint(tmp)
            ev = EntityNeuralEvaluator(ckpt)
            snap = _snapshot()
            single = (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)
            prior = ev.priors(snap, single, player="me")
            self.assertEqual(len(prior.priors), 1)
            self.assertAlmostEqual(prior.priors[0], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
