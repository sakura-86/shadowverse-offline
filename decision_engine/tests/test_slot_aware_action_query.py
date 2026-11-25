# -*- coding: utf-8 -*-
"""目标槽（pointer 动作表示）接入 entity policy 的测试。

覆盖三件事：

1. ``target_group_sizes`` / ``target_group_specs`` 从 ``LegalAction`` 正确流出；
2. tensorization 的 row / batch 携带槽结构，旧数据（无 specs）落成「无槽信息」；
3. 模型在 ``slot_aware_query_enabled`` 打开时才用槽信息，关闭时行为与旧版一致。
"""

from __future__ import annotations

import unittest

try:
    import torch
    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False

from decision_engine.actions.model import ActionKind, LegalAction, PlayKind
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView
from decision_engine.training.action_features import action_features
from decision_engine.training.card_vocabulary import CardVocabulary
from decision_engine.training.entity_tensorization import (
    MAX_ACTION_SLOTS,
    collate_entity_rows,
    entity_tensorize_example,
)
from decision_engine.training.slot_spec_vocabulary import (
    BUILTIN_SLOT_SPECS,
    PAD_INDEX,
    UNKNOWN_INDEX,
    SlotSpecVocabulary,
    default_slot_spec_vocabulary,
)
from decision_engine.training.state_features import state_features

if _TORCH:
    from decision_engine.training.entity_policy_value import (
        CrossZoneEntityTransformer,
        EntityModelConfig,
    )

_VOCAB = CardVocabulary({1000: 2, 1001: 3, 2000: 4})


def _snapshot(hand=(), field=(), opp_field=()) -> BattleSnapshot:
    cards: dict[int, CardView] = {}
    for uid in hand + field + opp_field:
        cards.setdefault(
            uid,
            CardView(uid=uid, card_id=1000 + uid, base_card_id=1000, cost=2,
                     card_type=CardType.FOLLOWER, atk=2, life=2, max_life=2),
        )
    me = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, hand=hand, field=field)
    opp = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, field=opp_field)
    return BattleSnapshot(me=me, opponent=opp, cards_by_uid=cards, is_ally_turn=True)


class SlotSpecVocabularyTest(unittest.TestCase):
    def test_builtin_specs_have_stable_indices(self):
        vocab = default_slot_spec_vocabulary()

        for offset, spec in enumerate(BUILTIN_SLOT_SPECS):
            self.assertEqual(vocab.index_for_spec(spec), UNKNOWN_INDEX + 1 + offset)

    def test_pad_and_unknown(self):
        vocab = default_slot_spec_vocabulary()

        self.assertEqual(vocab.index_for_spec(None), PAD_INDEX)
        self.assertEqual(vocab.index_for_spec(""), PAD_INDEX)
        self.assertEqual(vocab.index_for_spec("no_such_spec"), UNKNOWN_INDEX)

    def test_jsonable_round_trip(self):
        vocab = default_slot_spec_vocabulary()

        restored = SlotSpecVocabulary.from_jsonable(vocab.to_jsonable())

        self.assertEqual(restored.vocab_size, vocab.vocab_size)
        self.assertEqual(restored.digest, vocab.digest)

    def test_extend_is_append_only(self):
        vocab = SlotSpecVocabulary(spec_to_index={"a": 2, "b": 3})

        extended = vocab.extend(["c", "a"])

        self.assertEqual(extended.index_for_spec("a"), 2)
        self.assertEqual(extended.index_for_spec("b"), 3)
        self.assertEqual(extended.index_for_spec("c"), 4)

    def test_non_contiguous_indices_rejected(self):
        with self.assertRaises(ValueError):
            SlotSpecVocabulary(spec_to_index={"a": 2, "b": 4})


class ActionFeaturesSlotTest(unittest.TestCase):
    def test_target_group_sizes_and_specs_are_emitted(self):
        snapshot = _snapshot(hand=(1,), opp_field=(5, 6))
        action = LegalAction(
            kind=ActionKind.PLAY,
            source_uid=1,
            play_kind=PlayKind.NORMAL,
            cost=2,
            target_groups=((5, 6),),
            target_group_specs=("vira_fanfare",),
        )

        features = action_features(snapshot, action)

        # count=2 是一个槽选 2 个，不是两个槽
        self.assertEqual(features["target_group_sizes"], [2])
        self.assertEqual(features["target_group_specs"], ["vira_fanfare"])
        self.assertEqual(features["target_group_count"], 1)
        self.assertEqual(len(features["targets"]), 2)

    def test_action_without_specs_emits_empty_specs_but_keeps_sizes(self):
        """旧数据（replay 来源的 LegalAction 无 specs）保持 sizes 但从不出 specs。"""
        snapshot = _snapshot(hand=(1,), opp_field=(5,))
        action = LegalAction(kind=ActionKind.PLAY, source_uid=1, play_kind=PlayKind.NORMAL, cost=2,
                             target_groups=((5,),))

        features = action_features(snapshot, action)

        self.assertEqual(features["target_group_sizes"], [1])
        self.assertEqual(features["target_group_specs"], [])


class EntityTensorizationSlotTest(unittest.TestCase):
    def _row(self, actions, *, slot_vocabulary=None):
        snapshot = _snapshot(hand=(1,), opp_field=(5, 6))
        example = {
            "state_features": state_features(snapshot, perspective="me"),
            "legal_action_features": [
                action_features(snapshot, action) for action in actions
            ],
            "policy_target": [1.0 / len(actions)] * len(actions),
            "value_target": 0.0,
        }
        kwargs = {"vocabulary": _VOCAB}
        if slot_vocabulary is not None:
            kwargs["slot_vocabulary"] = slot_vocabulary
        return entity_tensorize_example(example, **kwargs)

    def test_row_carries_slot_structure(self):
        vocab = default_slot_spec_vocabulary()
        action = LegalAction(kind=ActionKind.PLAY, source_uid=1, play_kind=PlayKind.NORMAL, cost=2,
                             target_groups=((5,), (6,)),
                             target_group_specs=("attack_follower", "local_select"))

        row = self._row([action], slot_vocabulary=vocab)

        self.assertEqual(row.action_target_slot_index[0][:2], (0, 1))
        self.assertEqual(
            row.action_slot_spec_ids[0][:2],
            (vocab.index_for_spec("attack_follower"), vocab.index_for_spec("local_select")),
        )
        self.assertEqual(row.action_slot_mask[0], (True, True, False, False))

    def test_row_without_specs_keeps_boundaries_but_has_no_slot_semantics(self):
        """无 specs 时仍能由 ``target_group_sizes`` 还原 group 边界，但没有槽语义。

        ``slot_index`` 来自边界（总是可用），``slot_mask`` 反映「有没有 spec」。
        模型侧用 ``slot_mask`` 判断是否启用 slot 池化。
        """
        action = LegalAction(kind=ActionKind.PLAY, source_uid=1, play_kind=PlayKind.NORMAL, cost=2,
                             target_groups=((5,),))

        row = self._row([action])

        self.assertEqual(row.action_target_slot_index[0][:1], (0,))
        self.assertEqual(row.action_slot_spec_ids[0], (0,) * MAX_ACTION_SLOTS)
        self.assertEqual(row.action_slot_mask[0], (False,) * MAX_ACTION_SLOTS)

    def test_row_without_group_sizes_has_unknown_boundaries(self):
        """旧 jsonl 的 ``legal_action_features`` 没有 ``target_group_sizes`` → slot_index=-1。"""
        snapshot = _snapshot(hand=(1,), opp_field=(5,))
        legacy = action_features(
            snapshot,
            LegalAction(kind=ActionKind.PLAY, source_uid=1, play_kind=PlayKind.NORMAL, cost=2, target_groups=((5,),)),
        )
        legacy.pop("target_group_sizes")
        legacy.pop("target_group_specs")
        example = {
            "state_features": state_features(snapshot, perspective="me"),
            "legal_action_features": [legacy],
            "policy_target": [1.0],
            "value_target": 0.0,
        }

        row = entity_tensorize_example(example, vocabulary=_VOCAB)

        self.assertEqual(row.action_target_slot_index[0][:1], (-1,))
        self.assertEqual(row.action_slot_mask[0], (False,) * MAX_ACTION_SLOTS)

    def test_too_many_slots_raises(self):
        action = LegalAction(
            kind=ActionKind.PLAY, source_uid=1, play_kind=PlayKind.NORMAL, cost=2,
            target_groups=((5,),) * (MAX_ACTION_SLOTS + 1),
            target_group_specs=("a",) * (MAX_ACTION_SLOTS + 1),
        )

        with self.assertRaises(ValueError):
            self._row([action])

    def test_batch_carries_slot_tensors(self):
        action = LegalAction(kind=ActionKind.PLAY, source_uid=1, play_kind=PlayKind.NORMAL, cost=2,
                             target_groups=((5,),), target_group_specs=("attack_leader",))

        batch = collate_entity_rows([self._row([action])])

        self.assertEqual(tuple(batch["action_slot_spec_ids"].shape[:2]), (1, 1))
        self.assertEqual(batch["action_slot_spec_ids"].shape[-1], MAX_ACTION_SLOTS)
        self.assertTrue(bool(batch["action_slot_mask"][0, 0, 0]))
        self.assertEqual(int(batch["action_target_slot_index"][0, 0, 0]), 0)


@unittest.skipUnless(_TORCH, "torch 不可用")
class SlotAwareModelTest(unittest.TestCase):
    def _batch(self, actions, *, slot_vocabulary=None):
        snapshot = _snapshot(hand=(1,), opp_field=(5, 6))
        example = {
            "state_features": state_features(snapshot, perspective="me"),
            "legal_action_features": [action_features(snapshot, a) for a in actions],
            "policy_target": [1.0 / len(actions)] * len(actions),
            "value_target": 0.0,
        }
        kwargs = {"vocabulary": _VOCAB}
        if slot_vocabulary is not None:
            kwargs["slot_vocabulary"] = slot_vocabulary
        return collate_entity_rows([entity_tensorize_example(example, **kwargs)])

    def _model(self, *, slot_aware: bool, state_width: int):
        vocab = default_slot_spec_vocabulary()
        config = EntityModelConfig(
            d_model=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
            attention_dropout=0.0,
            residual_dropout=0.0,
            slot_aware_query_enabled=slot_aware,
            max_action_slots=MAX_ACTION_SLOTS,
            slot_spec_vocab_size=vocab.vocab_size if slot_aware else 0,
        )
        torch.manual_seed(0)
        model = CrossZoneEntityTransformer(
            vocab_size=_VOCAB.vocab_size,
            state_scalar_width=state_width,
            action_scalar_width=52,
            config=config,
        )
        model.eval()
        return model

    @staticmethod
    def _state_width(batch) -> int:
        return int(batch["state_scalar"].shape[-1])

    def _forward(self, actions, *, slot_aware: bool, with_slot_data: bool = True):
        slot_vocab = default_slot_spec_vocabulary() if with_slot_data else None
        batch = self._batch(actions, slot_vocabulary=slot_vocab)
        model = self._model(slot_aware=slot_aware, state_width=self._state_width(batch))
        with torch.no_grad():
            return model(batch)

    def _two_actions_differing_only_in_spec(self):
        base = dict(kind=ActionKind.PLAY, source_uid=1, play_kind=PlayKind.NORMAL, cost=2,
                    target_groups=((5,),))
        left = LegalAction(**base, target_group_specs=("attack_follower",))
        right = LegalAction(**base, target_group_specs=("local_select",))
        return left, right

    def test_disabled_model_has_no_slot_parameters(self):
        model = self._model(slot_aware=False, state_width=55)
        names = [name for name, _ in model.named_parameters()]

        self.assertFalse(any(name.startswith("slot_") for name in names))

    def test_disabled_model_ignores_specs(self):
        """关闭 slot-aware 时，只有 specs 不同的两个动作必须得到相同 logit。"""
        left, right = self._two_actions_differing_only_in_spec()
        out = self._forward([left, right], slot_aware=False)

        self.assertAlmostEqual(float(out["policy_logits"][0, 0]), float(out["policy_logits"][0, 1]), places=6)

    def test_enabled_model_separates_specs(self):
        """打开 slot-aware 后，spec 不同会让 logit 分开——证明槽信息真的进了网络。"""
        left, right = self._two_actions_differing_only_in_spec()
        out = self._forward([left, right], slot_aware=True)

        self.assertNotAlmostEqual(float(out["policy_logits"][0, 0]), float(out["policy_logits"][0, 1]), places=6)

    def test_enabled_model_handles_actions_without_slots(self):
        """旧数据（无 specs）在启用 slot-aware 的模型上仍可前向，不产生 NaN。"""
        action = LegalAction(kind=ActionKind.PLAY, source_uid=1, play_kind=PlayKind.NORMAL, cost=2,
                             target_groups=((5,),))

        out = self._forward([action], slot_aware=True, with_slot_data=False)

        self.assertFalse(bool(torch.isnan(out["policy_logits"]).any()))
        self.assertFalse(bool(torch.isnan(out["value"]).any()))
        self.assertTrue(torch.isfinite(out["policy_logits"][0, 0]))

    def test_enabled_model_value_range_unchanged(self):
        action = LegalAction(kind=ActionKind.TURN_END, source_uid=0)

        out = self._forward([action], slot_aware=True)

        value = float(out["value"][0])
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
