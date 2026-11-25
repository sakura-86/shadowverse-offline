# -*- coding: utf-8 -*-
"""Tests for multi-step PUCT tree search."""

import unittest
from dataclasses import dataclass
from typing import Any

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.puct import PuctConfig, PuctSearch, PuctSearchResult
from decision_engine.planning.search_value import BoardAdvantageValue, TerminalValue, ValueResult
from decision_engine.state.snapshot import (
    BattleSnapshot,
    CardType,
    CardView,
    LegalView,
    PlayerView,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _action(kind: str, uid: int = 0, desc: str = "") -> LegalAction:
    return LegalAction(kind=ActionKind(kind), source_uid=uid, description=desc)


def _snapshot(
    *,
    my_life: int = 20,
    opp_life: int = 20,
    my_field: list[CardView] | None = None,
    opp_field: list[CardView] | None = None,
    is_turn_ended: bool = False,
) -> BattleSnapshot:
    my_field = my_field or []
    opp_field = opp_field or []
    cards: dict[int, CardView] = {}
    my_uids: list[int] = []
    opp_uids: list[int] = []
    for c in my_field:
        cards[c.uid] = c
        my_uids.append(c.uid)
    for c in opp_field:
        cards[c.uid] = c
        opp_uids.append(c.uid)
    return BattleSnapshot(
        me=PlayerView(life=my_life, max_life=20, pp=5, max_pp=5, field=tuple(my_uids)),
        opponent=PlayerView(life=opp_life, max_life=20, pp=5, max_pp=5, field=tuple(opp_uids)),
        cards_by_uid=cards,
        legal=LegalView(),
        is_turn_ended=is_turn_ended,
    )


@dataclass
class _MockPrediction:
    snapshot: BattleSnapshot
    certainty: Any = None
    unresolved_effects: tuple = ()


class _ScriptedPredictor:
    """按 (snapshot, action) 序列返回预设 prediction 的 mock predictor。"""

    def __init__(self, results: dict[str, BattleSnapshot]):
        """results: action_kind -> resulting snapshot"""
        self.results = results
        self.calls: list[str] = []

    def predict(self, snapshot: BattleSnapshot, action: LegalAction, *, player: str = "me"):
        kind = action.kind.value
        self.calls.append(kind)
        if kind in self.results:
            return _MockPrediction(snapshot=self.results[kind])
        # 默认返回原 snapshot（无变化）
        return _MockPrediction(snapshot=snapshot)


class _ScriptedEnumerator:
    """按 snapshot 状态返回预设合法动作的 mock enumerator。"""

    def __init__(self, actions_by_life: dict[int, list[LegalAction]] | None = None, default_actions: list[LegalAction] | None = None):
        self.actions_by_life = actions_by_life or {}
        self.default_actions = default_actions or [_action("turn_end")]

    def enumerate(self, snapshot: BattleSnapshot, *, player: str = "me"):
        key = snapshot.me.life if player == "me" else snapshot.opponent.life
        return self.actions_by_life.get(key, self.default_actions)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPuctMultiStep(unittest.TestCase):
    def test_depth1_backward_compat(self):
        """max_depth=1 时应和旧行为一致：每次 simulation 只走一步。"""
        play = _action("play", uid=1, desc="play_card")
        end = _action("turn_end", uid=0, desc="end")

        # play 后场面有 3/3 随从，turn_end 后无变化
        after_play = _snapshot(my_field=[CardView(uid=100, atk=3, life=3, card_type=CardType.FOLLOWER)])
        after_end = _snapshot()

        predictor = _ScriptedPredictor({"play": after_play, "turn_end": after_end})
        enumerator = _ScriptedEnumerator(default_actions=[play, end])

        search = PuctSearch(
            enumerator=enumerator,
            predictor=predictor,
            value=BoardAdvantageValue(),
            config=PuctConfig(simulations=32, max_depth=1, c_puct=1.5),
        )
        result = search.search(_snapshot(), player="me")

        self.assertEqual(len(result.actions), 2)
        self.assertIsNotNone(result.selected_action_index)
        # play 应该有正 value（场面加分），turn_end 应该是 0（空场）
        self.assertEqual(result.diagnostics["effective_max_depth"], 1)

    def test_multi_step_prefers_sequence(self):
        """depth>1 时，搜索能看到 play->turn_end 的完整序列价值。"""
        play = _action("play", uid=1, desc="play_follower")
        end = _action("turn_end", uid=0, desc="end_turn")

        # 初始：空场
        initial = _snapshot()
        # play 后：场上有 4/4
        after_play = _snapshot(my_field=[CardView(uid=100, atk=4, life=4, card_type=CardType.FOLLOWER)])

        # 第二层（after_play 状态）只有 turn_end
        predictor = _ScriptedPredictor({"play": after_play, "turn_end": after_play})
        enumerator = _ScriptedEnumerator(
            actions_by_life={
                20: [play, end],  # 初始 life=20 有 play 和 end
            },
            default_actions=[end],  # after_play 时只有 end
        )

        search = PuctSearch(
            enumerator=enumerator,
            predictor=predictor,
            value=BoardAdvantageValue(),
            config=PuctConfig(simulations=64, max_depth=3, c_puct=1.5),
        )
        result = search.search(initial, player="me")

        # play 的价值更高（4/4 场面），搜索应该选 play
        self.assertEqual(result.selected_action_index, 0)  # play is index 0
        # 深度应该 > 1
        self.assertGreaterEqual(result.diagnostics["effective_max_depth"], 1)

    def test_turn_end_terminates_simulation(self):
        """turn_end 动作应该终止 simulation，不继续展开。"""
        play = _action("play", uid=1)
        end = _action("turn_end", uid=0)

        after_play = _snapshot(my_field=[CardView(uid=100, atk=2, life=2, card_type=CardType.FOLLOWER)])
        after_end = _snapshot()

        predictor = _ScriptedPredictor({"play": after_play, "turn_end": after_end})
        enumerator = _ScriptedEnumerator(default_actions=[play, end])

        search = PuctSearch(
            enumerator=enumerator,
            predictor=predictor,
            value=BoardAdvantageValue(),
            config=PuctConfig(simulations=32, max_depth=5, c_puct=1.5),
        )
        result = search.search(_snapshot(), player="me")

        # 搜索应该正常完成
        self.assertIsNotNone(result.selected_action_index)
        self.assertEqual(result.diagnostics["executed_simulations"], 32)

    def test_terminal_detected(self):
        """对方 life<=0 的终局应返回 +1 并终止。"""
        attack = _action("attack_leader", uid=1)
        end = _action("turn_end", uid=0)

        # attack 直接击杀对方
        after_attack = _snapshot(opp_life=0)
        after_end = _snapshot()

        predictor = _ScriptedPredictor({"attack_leader": after_attack, "turn_end": after_end})
        enumerator = _ScriptedEnumerator(default_actions=[attack, end])

        search = PuctSearch(
            enumerator=enumerator,
            predictor=predictor,
            value=BoardAdvantageValue(),
            config=PuctConfig(simulations=32, max_depth=3, c_puct=1.5),
        )
        result = search.search(_snapshot(), player="me")

        # 攻击直接致死，应该全选 attack
        self.assertEqual(result.selected_action_index, 0)
        # root_value 应接近 1.0（击杀）
        self.assertGreater(result.root_value, 0.5)

    def test_predict_error_handled(self):
        """predict 失败不应崩溃。"""
        from decision_engine.simulation.predictor import PredictionCertainty

        play = _action("play", uid=1)
        end = _action("turn_end", uid=0)

        @dataclass
        class _FailPrediction:
            snapshot: BattleSnapshot = None
            certainty: Any = PredictionCertainty.PREDICT_ERROR
            unresolved_effects: tuple = ()

        class _FailPredictor:
            def predict(self, snapshot, action, *, player="me"):
                if action.kind == ActionKind.PLAY:
                    return _FailPrediction()
                return _MockPrediction(snapshot=snapshot)

        enumerator = _ScriptedEnumerator(default_actions=[play, end])
        search = PuctSearch(
            enumerator=enumerator,
            predictor=_FailPredictor(),
            value=BoardAdvantageValue(),
            config=PuctConfig(simulations=16, max_depth=2, c_puct=1.5),
        )
        result = search.search(_snapshot(), player="me")

        self.assertIsNotNone(result.selected_action_index)
        self.assertGreater(result.diagnostics["prediction_error_count"], 0)

    def test_policy_target_normalized(self):
        """policy_target 应归一化为概率分布。"""
        play = _action("play", uid=1)
        end = _action("turn_end", uid=0)
        after_play = _snapshot(my_field=[CardView(uid=100, atk=3, life=3, card_type=CardType.FOLLOWER)])

        predictor = _ScriptedPredictor({"play": after_play, "turn_end": _snapshot()})
        enumerator = _ScriptedEnumerator(default_actions=[play, end])

        search = PuctSearch(
            enumerator=enumerator,
            predictor=predictor,
            value=BoardAdvantageValue(),
            config=PuctConfig(simulations=32, max_depth=2, c_puct=1.5),
        )
        result = search.search(_snapshot(), player="me")

        self.assertAlmostEqual(sum(result.policy_target), 1.0, places=6)
        self.assertTrue(all(p >= 0.0 for p in result.policy_target))

    def test_search_scope_reports_multi_turn(self):
        """显式 max_turns=3 时 diagnostics 应报告 multi_turn_3。"""
        end = _action("turn_end", uid=0)
        predictor = _ScriptedPredictor({"turn_end": _snapshot()})
        enumerator = _ScriptedEnumerator(default_actions=[end])

        search = PuctSearch(
            enumerator=enumerator,
            predictor=predictor,
            value=BoardAdvantageValue(),
            config=PuctConfig(simulations=8, max_depth=3, max_turns=3, c_puct=1.5),
        )
        result = search.search(_snapshot(), player="me")
        self.assertEqual(result.diagnostics["search_scope"], "multi_turn_3")

    def test_search_scope_reports_intra_turn_when_max_turns_one(self):
        """max_turns=1 时 diagnostics 应报告 intra_turn_tree。"""
        end = _action("turn_end", uid=0)
        predictor = _ScriptedPredictor({"turn_end": _snapshot()})
        enumerator = _ScriptedEnumerator(default_actions=[end])

        search = PuctSearch(
            enumerator=enumerator,
            predictor=predictor,
            value=BoardAdvantageValue(),
            config=PuctConfig(simulations=8, max_depth=3, max_turns=1, c_puct=1.5),
        )
        result = search.search(_snapshot(), player="me")
        self.assertEqual(result.diagnostics["search_scope"], "intra_turn_tree")


if __name__ == "__main__":
    unittest.main()
