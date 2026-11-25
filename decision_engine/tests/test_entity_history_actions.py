# -*- coding: utf-8 -*-
"""历史动作 token 的契约测试。

设计意图：让 policy 能看到"最近 K 步发生了什么动作"，从而学 combo / 顺序依赖
（影之诗有大量"本回合已打出 N 张""本回合已攻击过"类条件）。

锁住的行为：
1. 每个历史动作产出恰好一个 ``ENTITY_TYPE_HISTORY_ACTION`` token；
2. 超过 ``HISTORY_ACTION_LIMIT`` 时只保留**最近**的 K 个；
3. ``position_id`` 是步序（1 = 最近一步），用于表达顺序（Transformer 本身无位置编码）；
4. ``count`` 是动作 kind 的数值编码；
5. ``owner_id`` 反映动作属于我方还是对手（对手动作是公开信息，可以进序列）。
"""

from __future__ import annotations

import unittest

from decision_engine.execution.local_match_setup import (
    LocalDeckList,
    LocalMatchSetup,
    LocalMatchSetupConfig,
)
from decision_engine.training.entity_tensorization import (
    ACTION_KIND_IDS,
    ENTITY_TYPE_HISTORY_ACTION,
    HISTORY_ACTION_LIMIT,
    OWNER_OPPONENT,
    OWNER_SELF,
    ZONE_HISTORY,
    build_entity_tokens,
    default_card_vocabulary,
)
from decision_engine.training.state_features import state_features


def _payload():
    deck = LocalDeckList(name="hist", class_name="hist", card_ids=tuple([10161210] * 40))
    snapshot = LocalMatchSetup().create(deck, deck, LocalMatchSetupConfig(seed=11, first_player="me")).snapshot
    return state_features(snapshot, perspective="me")


def _action(kind: str, *, own: bool = True, card: int | None = 10161210) -> dict:
    return {"kind": kind, "source_base_card_id": card, "is_own": own}


class HistoryActionTokenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = _payload()
        cls.vocabulary = default_card_vocabulary()

    def _history_tokens(self, actions):
        tokens = build_entity_tokens(self.payload, vocabulary=self.vocabulary, history_actions=actions)
        return [t for t in tokens if t.type_id == ENTITY_TYPE_HISTORY_ACTION]

    def test_each_action_yields_one_token(self) -> None:
        history = self._history_tokens([_action("play"), _action("attack_leader")])
        self.assertEqual(len(history), 2)
        for token in history:
            self.assertEqual(token.zone_id, ZONE_HISTORY)

    def test_only_most_recent_k_are_kept(self) -> None:
        actions = [_action("play") for _ in range(HISTORY_ACTION_LIMIT + 3)]
        actions.append(_action("attack_leader"))
        history = self._history_tokens(actions)
        self.assertEqual(len(history), HISTORY_ACTION_LIMIT)
        # 最近一步（attack_leader）必须还在：position_id == 1
        self.assertEqual(history[0].count, float(ACTION_KIND_IDS["attack_leader"]))

    def test_position_id_encodes_recency(self) -> None:
        history = self._history_tokens([_action("play"), _action("evolve"), _action("attack_leader")])
        # reversed：最后一步是 position 1
        self.assertEqual([t.position_id for t in history], [1, 2, 3])

    def test_count_encodes_action_kind(self) -> None:
        history = self._history_tokens([_action("super_evolve")])
        self.assertEqual(history[0].count, float(ACTION_KIND_IDS["super_evolve"]))

    def test_owner_id_reflects_side(self) -> None:
        history = self._history_tokens([_action("play", own=True), _action("attack_follower", own=False)])
        # reversed：对手那步是最近一步
        self.assertEqual(history[0].owner_id, OWNER_OPPONENT)
        self.assertEqual(history[1].owner_id, OWNER_SELF)

    def test_empty_history_adds_no_tokens(self) -> None:
        self.assertEqual(self._history_tokens([]), [])

    def test_unknown_kind_is_zero_not_crash(self) -> None:
        history = self._history_tokens([_action("nonexistent_kind")])
        self.assertEqual(history[0].count, 0.0)


if __name__ == "__main__":
    unittest.main()
