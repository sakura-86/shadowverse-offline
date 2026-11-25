# -*- coding: utf-8 -*-
"""卡牌效果文本 embedding（`card_text_embedding_enabled`）的契约测试。

覆盖三件事：

1. 默认关闭时 state_dict 完全不变——旧 checkpoint 仍能 strict load。
2. 打开后多出 `card_text_embedding.weight` / `card_text_proj.weight` 两个参数组，
   形状与 vocab / dim 一致。
3. 初始化真的接上了：把某张卡的文本向量改成非零后，前向输出必须变化；形状不符必须
   显式报错而不是静默截断。
"""

from __future__ import annotations

import unittest

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView
from decision_engine.training.card_vocabulary import CardVocabulary
from decision_engine.training.entity_tensorization import collate_entity_rows, entity_tensorize_example
from decision_engine.training.state_features import state_features
from decision_engine.training.action_features import legal_actions_features

if _TORCH:
    from decision_engine.training.entity_policy_value import CrossZoneEntityTransformer, EntityModelConfig

_VOCAB = CardVocabulary({1000: 2, 1001: 3, 2000: 4, 3000: 5})
_TEXT_DIM = 8


def _snapshot() -> BattleSnapshot:
    cards = {
        11: CardView(uid=11, card_id=1011, base_card_id=1000, cost=2, card_type=CardType.FOLLOWER,
                     atk=2, life=2, max_life=2, can_attack=True),
    }
    me = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, hand=(11,), field=(), deck_count=10)
    opp = PlayerView(life=20, max_life=20, pp=3, max_pp=3, turn=3, field=(), deck_count=10)
    return BattleSnapshot(me=me, opponent=opp, cards_by_uid=cards, is_ally_turn=True)


def _batch():
    snapshot = _snapshot()
    actions = (
        LegalAction(kind=ActionKind.PLAY, source_uid=11),
        LegalAction(kind=ActionKind.TURN_END, source_uid=0),
    )
    example = {
        "state_features": state_features(snapshot, perspective="me"),
        "legal_action_features": legal_actions_features(snapshot, actions, perspective="me"),
        "policy_target": [0.5, 0.5],
        "value_target": 0.3,
    }
    row = entity_tensorize_example(example, vocabulary=_VOCAB)
    return collate_entity_rows([row])


def _model(*, text_enabled: bool) -> "CrossZoneEntityTransformer":
    prepared = _batch()
    state_width = int(prepared["state_scalar"].shape[-1])
    action_width = int(prepared["action_scalars"].shape[-1])
    torch.manual_seed(0)
    cfg = EntityModelConfig(
        d_model=32, num_layers=1, num_heads=4, ffn_dim=64,
        card_text_embedding_enabled=text_enabled, card_text_embedding_dim=_TEXT_DIM,
    )
    return CrossZoneEntityTransformer(
        vocab_size=_VOCAB.vocab_size,
        state_scalar_width=state_width,
        action_scalar_width=action_width,
        config=cfg,
    )


@unittest.skipUnless(_TORCH, "PyTorch 未安装")
class CardTextEmbeddingContractTest(unittest.TestCase):
    def test_disabled_keeps_state_dict_unchanged(self):
        model = _model(text_enabled=False)
        keys = set(model.state_dict().keys())
        self.assertFalse(any(key.startswith("card_text_embedding") for key in keys))
        self.assertFalse(any(key.startswith("card_text_proj") for key in keys))

    def test_enabled_adds_parameters_with_expected_shapes(self):
        model = _model(text_enabled=True)
        state = model.state_dict()
        self.assertIn("card_text_embedding.weight", state)
        self.assertEqual(tuple(state["card_text_embedding.weight"].shape), (_VOCAB.vocab_size, _TEXT_DIM))
        self.assertIn("card_text_proj.weight", state)
        self.assertEqual(tuple(state["card_text_proj.weight"].shape), (32, _TEXT_DIM))

    def test_initialize_changes_forward_output(self):
        batch = _batch()
        zero_model = _model(text_enabled=True)
        zero_model.initialize_card_text_embeddings(torch.zeros(_VOCAB.vocab_size, _TEXT_DIM))
        zero_out = zero_model(batch)["policy_logits"]

        given_model = _model(text_enabled=True)
        matrix = torch.zeros(_VOCAB.vocab_size, _TEXT_DIM)
        matrix[_VOCAB.index_for_base_card_id(1000)] = torch.ones(_TEXT_DIM)
        given_model.initialize_card_text_embeddings(matrix)
        given_out = given_model(batch)["policy_logits"]

        # 两张模型的其余参数由同一个 seed 初始化，差别只应来自文本向量。
        self.assertFalse(torch.allclose(zero_out, given_out, atol=1e-6))

    def test_initialize_rejects_wrong_shape(self):
        model = _model(text_enabled=True)
        with self.assertRaises(ValueError):
            model.initialize_card_text_embeddings(torch.zeros(_VOCAB.vocab_size, _TEXT_DIM + 1))
        with self.assertRaises(ValueError):
            model.initialize_card_text_embeddings(torch.zeros(_VOCAB.vocab_size + 1, _TEXT_DIM))

    def test_initialize_rejected_when_disabled(self):
        model = _model(text_enabled=False)
        with self.assertRaises(ValueError):
            model.initialize_card_text_embeddings(torch.zeros(_VOCAB.vocab_size, _TEXT_DIM))

    def test_padding_row_stays_zero(self):
        model = _model(text_enabled=True)
        matrix = torch.rand(_VOCAB.vocab_size, _TEXT_DIM)
        model.initialize_card_text_embeddings(matrix)
        # padding_idx=0 的行必须保持零，否则 [STATE] token 会被灌入无意义向量。
        self.assertTrue(torch.allclose(model.card_text_embedding.weight[0], torch.zeros(_TEXT_DIM)))


if __name__ == "__main__":
    unittest.main()
