# -*- coding: utf-8 -*-
"""MacroTurnPlanArmSelector / Macro arm factory 契约测试。

覆盖：
- 完整 TurnLine 按序执行（逐步返回 action_sequence + 收尾 turn_end）。
- turn 切换后计划重置（新 turn 重新触发 MacroTurnSearch）。
- stale 计划拒绝（计划动作不在当前 legal actions → plan_stale，绝不 fallback）。
- 无 best_line → 结构化 no_selection。
- Macro arm 接入 StrictSearchAbRunner，成本诊断被消费。
- legacy / candidate 配置 manifest 可区分（config_hash 不同）。

全部用 in-memory fake，不依赖真实卡牌执行，也不使用任何 human 数据。
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field, replace
from typing import Any

from decision_engine.actions.model import ActionKind, LegalAction, PlayKind
from decision_engine.planning.macro_ab_selector import (
    MacroArmConfig,
    MacroTurnPlanArmSelector,
    action_match_key,
    build_macro_arm,
    lazy_macro_arm_config,
    legacy_macro_arm_config,
    static_macro_arm_config,
)
from decision_engine.planning.macro_turn_search import MacroResult, TurnLine
from decision_engine.planning.search_ab_match import (
    SearchAbConfig,
    StrictSearchAbRunner,
)
from decision_engine.simulation.predictor import PredictionCertainty


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def _play(uid: int) -> LegalAction:
    return LegalAction(kind=ActionKind.PLAY, source_uid=uid, play_kind=PlayKind.NORMAL)


def _turn_end() -> LegalAction:
    return LegalAction(kind=ActionKind.TURN_END, source_uid=0)


@dataclass
class _PlayerView:
    life: int
    turn: int = 1
    result: int = 0
    is_first_side: bool = True


@dataclass
class _Snapshot:
    me: _PlayerView
    opponent: _PlayerView
    is_ally_turn: bool = True
    is_turn_ended: bool = False
    cards_by_uid: dict = field(default_factory=dict)


class _StaticEnumerator:
    """总是返回固定 legal actions 的 enumerator（player-aware 签名兼容）。"""

    def __init__(self, actions):
        self._actions = list(actions)

    def enumerate(self, snapshot, *, player="me"):
        del snapshot, player
        return list(self._actions)


class _SequenceEnumerator:
    """按调用次数返回不同 legal actions 集合（用于制造 stale）。"""

    def __init__(self, sequences):
        self._sequences = [list(s) for s in sequences]
        self._i = 0

    def enumerate(self, snapshot, *, player="me"):
        del snapshot, player
        idx = min(self._i, len(self._sequences) - 1)
        self._i += 1
        return list(self._sequences[idx])


def _turn_line(action_sequence) -> TurnLine:
    return TurnLine(
        action_sequence=tuple(action_sequence),
        end_snapshot=None,  # 本测试不消费 end_snapshot
        remaining_hand_key=(),
        active_player="me",
        turn_number=1,
    )


class _FakeMacroSearch:
    """返回预设 best_line 的 MacroTurnSearch 替身；记录被注入的 predictor/seed。"""

    def __init__(self, best_line, *, diagnostics=None, captured=None, predictor=None, seed=None):
        self._best_line = best_line
        self._diag = diagnostics or {}
        self._captured = captured
        self._predictor = predictor
        self._seed = seed

    def search(self, snapshot, *, player="me"):
        if self._captured is not None:
            self._captured.setdefault("searches", []).append(
                {"player": player, "predictor": self._predictor, "seed": self._seed}
            )
        return MacroResult(
            backed_up_value=0.0,
            best_line=self._best_line,
            ranked_lines=() if self._best_line is None else (self._best_line,),
            line_values=() if self._best_line is None else (0.0,),
            diagnostics=self._diag,
        )


def _macro_factory(best_line, *, diagnostics=None, captured=None):
    def factory(shared_predictor, branch_seed):
        return _FakeMacroSearch(
            best_line, diagnostics=diagnostics, captured=captured,
            predictor=shared_predictor, seed=branch_seed,
        )
    return factory


# ---------------------------------------------------------------------------
# selector 单元测试
# ---------------------------------------------------------------------------


class MacroTurnPlanArmSelectorTest(unittest.TestCase):
    def _snapshot(self, *, turn=1):
        return _Snapshot(me=_PlayerView(life=20, turn=turn), opponent=_PlayerView(life=20))

    def test_full_line_executes_in_order_then_turn_end(self):
        # 计划 = [play 1, play 2]，legal actions 始终含这两张 + turn_end
        legal = [_play(1), _play(2), _turn_end()]
        line = _turn_line([_play(1), _play(2)])
        selector = MacroTurnPlanArmSelector(
            predictor=object(),
            search_factory=_macro_factory(line, diagnostics={"predictor_call_count": 7, "micro_executed_simulations": 12, "macro_turn_expansion_count": 3}),
            enumerator=_StaticEnumerator(legal),
        )
        snap = self._snapshot()

        # 第 1 步：play 1
        r1 = selector.select(snap, player="me", branch_seed=500)
        self.assertIsNotNone(r1.selected_action_index)
        self.assertEqual(r1.actions[r1.selected_action_index], _play(1))
        self.assertTrue(r1.diagnostics["macro_searched_this_step"])
        self.assertEqual(r1.diagnostics["predictor_call_count"], 7)
        self.assertEqual(r1.diagnostics["micro_simulations"], 12)
        self.assertEqual(r1.diagnostics["macro_node_costs"], 3)

        # 第 2 步：play 2（不重搜，成本记 0）
        r2 = selector.select(snap, player="me", branch_seed=500)
        self.assertEqual(r2.actions[r2.selected_action_index], _play(2))
        self.assertFalse(r2.diagnostics["macro_searched_this_step"])
        self.assertEqual(r2.diagnostics["predictor_call_count"], 0)
        self.assertEqual(r2.diagnostics["macro_node_costs"], 0)

        # 第 3 步：序列耗尽 → turn_end
        r3 = selector.select(snap, player="me", branch_seed=500)
        self.assertEqual(r3.actions[r3.selected_action_index], _turn_end())
        self.assertFalse(r3.diagnostics["macro_searched_this_step"])

    def test_empty_line_returns_turn_end_directly(self):
        # 空计划（不出牌直接 turn_end）：应第一步就返回 turn_end
        legal = [_turn_end()]
        selector = MacroTurnPlanArmSelector(
            predictor=object(),
            search_factory=_macro_factory(_turn_line([])),
            enumerator=_StaticEnumerator(legal),
        )
        r = selector.select(self._snapshot(), player="me", branch_seed=1)
        self.assertEqual(r.actions[r.selected_action_index], _turn_end())

    def test_plan_resets_on_turn_change(self):
        captured: dict[str, Any] = {}
        legal = [_play(1), _turn_end()]
        line = _turn_line([_play(1)])
        selector = MacroTurnPlanArmSelector(
            predictor="shared_pred",
            search_factory=_macro_factory(line, captured=captured),
            enumerator=_StaticEnumerator(legal),
        )
        snap_t1 = self._snapshot(turn=1)
        # turn1：play 1（触发 1 次搜索）
        selector.select(snap_t1, player="me", branch_seed=100)
        # turn1：turn_end（不搜索）
        selector.select(snap_t1, player="me", branch_seed=100)
        self.assertEqual(len(captured["searches"]), 1)

        # turn2：新 turn → 重新搜索
        snap_t2 = self._snapshot(turn=2)
        r = selector.select(snap_t2, player="me", branch_seed=200)
        self.assertEqual(len(captured["searches"]), 2)
        self.assertEqual(r.actions[r.selected_action_index], _play(1))
        # 注入的 predictor / seed 透传（同预测器 / 同 branch seed 契约）
        self.assertEqual(captured["searches"][0]["predictor"], "shared_pred")
        self.assertEqual(captured["searches"][0]["seed"], 100)
        self.assertEqual(captured["searches"][1]["seed"], 200)

    def test_stale_plan_is_structured_rejection_no_fallback(self):
        # 第 1 次 enumerate 返回含 play 5 的 legal（搜索计划 [play 5]）；
        # 但真正取步骤那次 enumerate 已不含 play 5 → plan_stale。
        line = _turn_line([_play(5)])
        enum = _SequenceEnumerator([[_play(9), _turn_end()]])  # 始终不含 play 5
        selector = MacroTurnPlanArmSelector(
            predictor=object(),
            search_factory=_macro_factory(line),
            enumerator=enum,
        )
        r = selector.select(self._snapshot(), player="me", branch_seed=1)
        self.assertIsNone(r.selected_action_index)  # 结构化拒绝
        self.assertIn("plan_stale", r.diagnostics["macro_rejection_reason"])
        self.assertEqual(r.diagnostics["planned_action"]["kind"], "play")
        self.assertEqual(r.diagnostics["planned_action"]["source_uid"], 5)

    def test_no_best_line_is_no_selection(self):
        selector = MacroTurnPlanArmSelector(
            predictor=object(),
            search_factory=_macro_factory(None, diagnostics={"predictor_call_count": 3}),
            enumerator=_StaticEnumerator([_turn_end()]),
        )
        r = selector.select(self._snapshot(), player="me", branch_seed=1)
        self.assertIsNone(r.selected_action_index)
        self.assertIn("no_selection", r.diagnostics["macro_rejection_reason"])

    def test_turn_end_not_legal_after_plan_exhausted_is_plan_stale(self):
        # 计划为空但 legal 里没有 turn_end → plan_stale:turn_end_not_legal
        selector = MacroTurnPlanArmSelector(
            predictor=object(),
            search_factory=_macro_factory(_turn_line([])),
            enumerator=_StaticEnumerator([_play(1)]),  # 无 turn_end
        )
        r = selector.select(self._snapshot(), player="me", branch_seed=1)
        self.assertIsNone(r.selected_action_index)
        self.assertIn("plan_stale:turn_end_not_legal", r.diagnostics["macro_rejection_reason"])

    def test_action_match_key_ignores_non_semantic_fields(self):
        a = LegalAction(kind=ActionKind.PLAY, source_uid=3, play_kind=PlayKind.NORMAL,
                        cost=2, legality_source="macro", confidence="approx", description="plan")
        b = LegalAction(kind=ActionKind.PLAY, source_uid=3, play_kind=PlayKind.NORMAL,
                        cost=5, legality_source="local", confidence="exact", description="legal")
        self.assertEqual(action_match_key(a), action_match_key(b))
        c = replace(b, source_uid=4)
        self.assertNotEqual(action_match_key(a), action_match_key(c))


# ---------------------------------------------------------------------------
# A/B runner 集成：Macro arm 成本诊断被消费 + 配置可区分
# ---------------------------------------------------------------------------


@dataclass
class _AbPlayerView:
    life: int
    turn: int = 0
    result: int = 0
    is_first_side: bool = True


@dataclass
class _AbSnapshot:
    me: _AbPlayerView
    opponent: _AbPlayerView
    is_ally_turn: bool = True
    is_turn_ended: bool = False
    cards_by_uid: dict = field(default_factory=dict)


@dataclass
class _AbPrediction:
    snapshot: _AbSnapshot
    certainty: PredictionCertainty = PredictionCertainty.EXACT
    unresolved_effects: tuple = ()
    description: str = ""


class _TurnEndDamagePredictor:
    """turn_start 递增行动方 turn；turn_end 对对方造成固定伤害。"""

    def __init__(self, *, me_damage, opp_damage):
        self.me_damage = me_damage
        self.opp_damage = opp_damage

    def predict_with_replay_context(self, snapshot, action, *, player="me", generated_cards=(), execution_context=None):
        del generated_cards, execution_context
        if action.kind is ActionKind.TURN_START:
            pv = snapshot.me if player == "me" else snapshot.opponent
            bumped = replace(pv, turn=pv.turn + 1)
            if player == "me":
                return _AbPrediction(replace(snapshot, me=bumped, is_turn_ended=False))
            return _AbPrediction(replace(snapshot, opponent=bumped, is_turn_ended=False))
        # turn_end：造成伤害
        damage = self.me_damage if player == "me" else self.opp_damage
        if player == "me":
            opp = replace(snapshot.opponent, life=snapshot.opponent.life - damage)
            return _AbPrediction(replace(snapshot, opponent=opp, is_turn_ended=True))
        me = replace(snapshot.me, life=snapshot.me.life - damage)
        return _AbPrediction(replace(snapshot, me=me, is_turn_ended=True))


def _ab_deck():
    from decision_engine.execution.local_match_setup import LocalDeckList

    return LocalDeckList(name="mirror", class_name="x", card_ids=(1, 2, 3))


class MacroArmAbIntegrationTest(unittest.TestCase):
    def setUp(self):
        import decision_engine.planning.search_ab_match as mod

        self._mod = mod
        self._orig_setup = mod.LocalMatchSetup
        self._orig_draw = mod.draw_next_from_deck_instances

        def _make_snapshot(config):
            first = config.first_player
            return _AbSnapshot(
                me=_AbPlayerView(life=20, is_first_side=(first == "me")),
                opponent=_AbPlayerView(life=20, is_first_side=(first == "opponent")),
                is_ally_turn=(first == "me"),
                is_turn_ended=True,
            )

        class _FakeSetupResult:
            def __init__(self, snapshot):
                self.snapshot = snapshot

        class _FakeSetup:
            def create(self, deck_a, deck_b, config):
                return _FakeSetupResult(_make_snapshot(config))

        def _fake_draw(snapshot, player):
            return None, snapshot

        mod.LocalMatchSetup = _FakeSetup
        mod.draw_next_from_deck_instances = _fake_draw

    def tearDown(self):
        self._mod.LocalMatchSetup = self._orig_setup
        self._mod.draw_next_from_deck_instances = self._orig_draw

    def _macro_arm(self, name, config: MacroArmConfig, *, best_line, diagnostics):
        # 每臂用 empty-line 计划（直接 turn_end），保证局能推进到 terminal。
        enum_factory = lambda pred: _StaticEnumerator([_turn_end()])
        search_factory = _macro_factory(best_line, diagnostics=diagnostics)
        return build_macro_arm(
            name, config, macro_search_factory=search_factory, enumerator_factory=enum_factory
        )

    def test_macro_arms_run_pair_and_consume_cost_diagnostics(self):
        predictor = _TurnEndDamagePredictor(me_damage=8, opp_damage=3)
        cand = self._macro_arm(
            "candidate", lazy_macro_arm_config(),
            best_line=_turn_line([]),
            diagnostics={"predictor_call_count": 5, "micro_executed_simulations": 9, "macro_turn_expansion_count": 2},
        )
        opp = self._macro_arm(
            "opponent", legacy_macro_arm_config(),
            best_line=_turn_line([]),
            diagnostics={"predictor_call_count": 4, "micro_executed_simulations": 6, "macro_turn_expansion_count": 1},
        )
        runner = StrictSearchAbRunner(
            predictor=predictor, candidate_arm=cand, opponent_arm=opp,
            config=SearchAbConfig(max_actions_per_turn=4),
        )
        pair = runner.run_pair(_ab_deck(), _ab_deck(), matchup_id="00:macro", seed=100000)
        self.assertTrue(pair.accepted, pair.rejection_reason)
        self.assertEqual(pair.outcome_label(), "win")  # candidate 每回合 8 伤更快
        arm_costs = pair.game_candidate_first.diagnostics["arm_costs"]
        # Macro 成本被 runner 消费（decisions/micro/predictor 累计 > 0）
        self.assertGreater(arm_costs["candidate"]["decisions"], 0)
        self.assertGreater(arm_costs["candidate"]["micro_simulations"], 0)
        self.assertGreater(arm_costs["candidate"]["predictor_call_count"], 0)
        self.assertGreater(arm_costs["candidate"]["macro_node_costs"], 0)

    def test_legacy_and_candidate_configs_are_distinguishable_in_manifest(self):
        legacy = legacy_macro_arm_config()
        lazy = lazy_macro_arm_config()
        static = static_macro_arm_config()
        arm_legacy = build_macro_arm("opponent", legacy, macro_search_factory=_macro_factory(_turn_line([])), enumerator_factory=lambda p: _StaticEnumerator([_turn_end()]))
        arm_lazy = build_macro_arm("candidate", lazy, macro_search_factory=_macro_factory(_turn_line([])), enumerator_factory=lambda p: _StaticEnumerator([_turn_end()]))
        arm_static = build_macro_arm("candidate", static, macro_search_factory=_macro_factory(_turn_line([])), enumerator_factory=lambda p: _StaticEnumerator([_turn_end()]))

        # config_hash 三者互异 → manifest 可区分 legacy / candidate
        hashes = {arm_legacy.config_hash(), arm_lazy.config_hash(), arm_static.config_hash()}
        self.assertEqual(len(hashes), 3)
        # 关键区分字段存在于可序列化 config
        self.assertEqual(arm_legacy.config["variant"], "legacy")
        self.assertFalse(arm_legacy.config["lazy_mode"])
        self.assertTrue(arm_lazy.config["lazy_mode"])
        self.assertTrue(arm_static.config["static_resource_mode"])
        self.assertEqual(arm_legacy.config["search"], "macro_turn_search")


if __name__ == "__main__":
    unittest.main()
