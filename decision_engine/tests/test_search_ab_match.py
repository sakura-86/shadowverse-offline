# -*- coding: utf-8 -*-
"""Strict paired search A/B harness contract tests.

只测配对 / strict 接纳 / 拒绝 / 汇总 / sign test 逻辑，用 in-memory fake predictor
与 fake selector，不依赖真实卡牌执行，也不使用任何 human 数据。
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field, replace
from typing import Any

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.search_ab_match import (
    STRICT_CERTAINTY_VALUES,
    AbSummary,
    PuctArmSelector,
    SearchArm,
    SearchAbConfig,
    SelectorResult,
    StrictSearchAbRunner,
    _sign_test_p_value,
    branch_seed,
    candidate_win_counts,
    candidate_win_rate,
    config_hash,
    summarize,
)
from decision_engine.simulation.predictor import PredictionCertainty


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _PlayerView:
    life: int
    result: int = 0
    is_first_side: bool = True


@dataclass
class _Snapshot:
    me: _PlayerView
    opponent: _PlayerView
    is_ally_turn: bool = True
    is_turn_ended: bool = False
    cards_by_uid: dict = field(default_factory=dict)


@dataclass
class _Prediction:
    snapshot: _Snapshot
    certainty: PredictionCertainty = PredictionCertainty.EXACT
    unresolved_effects: tuple[str, ...] = ()
    description: str = ""


class _DamageDealtOnTurnEndPredictor:
    """turn_end 时对对方造成固定伤害的确定性 predictor。

    每个决策 selector 只出 turn_end；turn_end 让对方掉 ``damage`` 点血。
    turn_start 原样返回（切换 is_turn_ended=False）。
    ``me_damage`` / ``opp_damage`` 控制 me/opponent 每回合造成的伤害，用于制造胜负。
    """

    def __init__(self, *, me_damage: int, opp_damage: int, certainty: PredictionCertainty = PredictionCertainty.EXACT, unresolved=()):
        self.me_damage = me_damage
        self.opp_damage = opp_damage
        self.certainty = certainty
        self.unresolved = tuple(unresolved)

    def predict_with_replay_context(self, snapshot, action, *, player="me", generated_cards=(), execution_context=None):
        del generated_cards, execution_context
        if action.kind is ActionKind.TURN_START:
            # 回合开始：清 is_turn_ended
            return _Prediction(replace(snapshot, is_turn_ended=False))
        # turn_end：对对方造成伤害并标记回合结束
        damage = self.me_damage if player == "me" else self.opp_damage
        if player == "me":
            opp = replace(snapshot.opponent, life=snapshot.opponent.life - damage)
            nxt = replace(snapshot, opponent=opp, is_turn_ended=True)
        else:
            me = replace(snapshot.me, life=snapshot.me.life - damage)
            nxt = replace(snapshot, me=me, is_turn_ended=True)
        return _Prediction(nxt, certainty=self.certainty, unresolved_effects=self.unresolved)


class _TurnEndSelector:
    """总是出 turn_end 的 selector；带可配置 cost 诊断。"""

    def __init__(self, *, micro=1, predictor_calls=2, macro=0):
        self._micro = micro
        self._predictor_calls = predictor_calls
        self._macro = macro

    def select(self, snapshot, *, player, branch_seed):
        del snapshot, player, branch_seed
        actions = (LegalAction(kind=ActionKind.TURN_END, source_uid=0),)
        return SelectorResult(
            actions=actions,
            selected_action_index=0,
            diagnostics={
                "micro_simulations": self._micro,
                "predictor_call_count": self._predictor_calls,
                "macro_node_costs": self._macro,
            },
        )


class _NoSelectionSelector:
    def select(self, snapshot, *, player, branch_seed):
        del snapshot, player, branch_seed
        return SelectorResult(actions=(), selected_action_index=None, diagnostics={})


def _monkeypatch_setup(runner: StrictSearchAbRunner, snapshot_factory):
    """替换 LocalMatchSetup.create 为返回给定 snapshot 的 fake，避免依赖真实牌组。"""
    import decision_engine.planning.search_ab_match as mod

    class _FakeSetupResult:
        def __init__(self, snapshot):
            self.snapshot = snapshot

    class _FakeSetup:
        def create(self, deck_a, deck_b, config):
            return _FakeSetupResult(snapshot_factory(config))

    mod.LocalMatchSetup = _FakeSetup  # type: ignore[assignment]

    def _fake_draw(snapshot, player):
        return None, snapshot

    mod.draw_next_from_deck_instances = _fake_draw  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class SearchAbContractTest(unittest.TestCase):
    def setUp(self) -> None:
        import decision_engine.planning.search_ab_match as mod
        self._mod = mod
        self._orig_setup = mod.LocalMatchSetup
        self._orig_draw = mod.draw_next_from_deck_instances

    def tearDown(self) -> None:
        self._mod.LocalMatchSetup = self._orig_setup
        self._mod.draw_next_from_deck_instances = self._orig_draw

    def _make_snapshot(self, config):
        first = config.first_player
        return _Snapshot(
            me=_PlayerView(life=20, is_first_side=(first == "me")),
            opponent=_PlayerView(life=20, is_first_side=(first == "opponent")),
            is_ally_turn=(first == "me"),
            is_turn_ended=True,
        )

    def _runner(self, *, me_damage, opp_damage, cand_selector=None, opp_selector=None, certainty=PredictionCertainty.EXACT, unresolved=()):
        predictor = _DamageDealtOnTurnEndPredictor(me_damage=me_damage, opp_damage=opp_damage, certainty=certainty, unresolved=unresolved)
        cand = SearchArm(name="candidate", factory=lambda p: cand_selector or _TurnEndSelector(), config={"kind": "cand"})
        opp = SearchArm(name="opponent", factory=lambda p: opp_selector or _TurnEndSelector(), config={"kind": "opp"})
        runner = StrictSearchAbRunner(predictor=predictor, candidate_arm=cand, opponent_arm=opp, config=SearchAbConfig(max_actions_per_turn=4))
        _monkeypatch_setup(runner, self._make_snapshot)
        return runner

    def test_branch_seed_matches_strict_self_play_formula(self) -> None:
        self.assertEqual(branch_seed(100000, 3, 2), 100000 * 100_000 + 3 * 100 + 2)
        self.assertEqual(branch_seed(5, 1, -1), 5 * 100_000 + 100 - 1)

    def test_strict_certainty_values_are_stable_protocol_values(self) -> None:
        self.assertEqual(STRICT_CERTAINTY_VALUES, {"exact", "spec_only"})

    def test_config_hash_is_stable_and_order_independent(self) -> None:
        h1 = config_hash({"a": 1, "b": [2, 3]})
        h2 = config_hash({"b": [2, 3], "a": 1})
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, config_hash({"a": 1, "b": [2, 4]}))
        self.assertEqual(SearchArm(name="x", factory=lambda p: None, config={"a": 1}).config_hash(), config_hash({"a": 1}))

    def test_candidate_faster_clock_wins_both_games(self) -> None:
        # candidate 控 me；me 每回合 8 伤，opponent 每回合 3 伤 → candidate 先手/后手都赢
        runner = self._runner(me_damage=8, opp_damage=3)
        pair = runner.run_pair(_deck(), _deck(), matchup_id="00:mirror", seed=100000)
        self.assertTrue(pair.accepted, pair.rejection_reason)
        self.assertTrue(pair.game_candidate_first.candidate_won)
        self.assertTrue(pair.game_candidate_second.candidate_won)
        self.assertEqual(pair.outcome_label(), "win")
        # cost 诊断被记录（两臂各自 decisions > 0）
        arm_costs = pair.game_candidate_first.diagnostics["arm_costs"]
        self.assertGreater(arm_costs["candidate"]["decisions"], 0)
        self.assertGreater(arm_costs["candidate"]["micro_simulations"], 0)
        self.assertGreater(pair.game_candidate_first.diagnostics["cost_totals"]["predictor_call_count"], 0)

    def test_candidate_slower_clock_loses_both_games(self) -> None:
        runner = self._runner(me_damage=3, opp_damage=8)
        pair = runner.run_pair(_deck(), _deck(), matchup_id="00:mirror", seed=100000)
        self.assertTrue(pair.accepted, pair.rejection_reason)
        self.assertEqual(pair.outcome_label(), "loss")

    def test_pair_rejected_when_execution_not_strict(self) -> None:
        runner = self._runner(me_damage=8, opp_damage=3, certainty=PredictionCertainty.PARTIAL, unresolved=("x",))
        pair = runner.run_pair(_deck(), _deck(), matchup_id="00:mirror", seed=100000)
        self.assertFalse(pair.accepted)
        self.assertIsNotNone(pair.rejection_reason)
        self.assertIn("executed_prediction", pair.rejection_reason)

    def test_pair_rejected_when_selector_returns_no_selection(self) -> None:
        runner = self._runner(me_damage=8, opp_damage=3, cand_selector=_NoSelectionSelector())
        pair = runner.run_pair(_deck(), _deck(), matchup_id="00:mirror", seed=100000)
        self.assertFalse(pair.accepted)
        self.assertIn("no_selection:candidate", pair.rejection_reason)

    def test_selector_exception_is_structured_rejection_not_silent_fallback(self) -> None:
        class _BoomSelector:
            def select(self, snapshot, *, player, branch_seed):
                raise RuntimeError("boom")

        runner = self._runner(me_damage=8, opp_damage=3, cand_selector=_BoomSelector())
        pair = runner.run_pair(_deck(), _deck(), matchup_id="00:mirror", seed=100000)
        self.assertFalse(pair.accepted)
        self.assertIn("select_error:candidate:RuntimeError", pair.rejection_reason)

    def test_summary_counts_and_sign_test(self) -> None:
        # 3 局 candidate 快 → win；2 局 candidate 慢 → loss
        wins = [self._runner(me_damage=8, opp_damage=3).run_pair(_deck(), _deck(), matchup_id="00", seed=s) for s in (100000, 100001, 100002)]
        losses = [self._runner(me_damage=3, opp_damage=8).run_pair(_deck(), _deck(), matchup_id="00", seed=s) for s in (200000, 200001)]
        summary = summarize(wins + losses)
        self.assertIsInstance(summary, AbSummary)
        self.assertEqual(summary.accepted_pairs, 5)
        self.assertEqual(summary.win, 3)
        self.assertEqual(summary.loss, 2)
        self.assertEqual(summary.sign_test["n_directional"], 5)
        self.assertFalse(summary.sign_test["is_proof"])
        self.assertIn("disclaimer", summary.to_jsonable())

    def test_summary_counts_rejections(self) -> None:
        good = self._runner(me_damage=8, opp_damage=3).run_pair(_deck(), _deck(), matchup_id="00", seed=100000)
        bad = self._runner(me_damage=8, opp_damage=3, certainty=PredictionCertainty.PARTIAL, unresolved=("x",)).run_pair(_deck(), _deck(), matchup_id="00", seed=100001)
        summary = summarize([good, bad])
        self.assertEqual(summary.accepted_pairs, 1)
        self.assertEqual(summary.rejected_pairs, 1)
        self.assertEqual(sum(summary.rejection_reasons.values()), 1)

    def test_sign_test_p_value_matches_exact_binomial(self) -> None:
        # 5 wins 0 losses → P(X>=5 | n=5, .5) = 1/32
        self.assertAlmostEqual(_sign_test_p_value(5, 0), 1 / 32)
        # 3 wins 3 losses → symmetric, tail P(X>=3|n=6) = (20+15+6+1)/64
        self.assertAlmostEqual(_sign_test_p_value(3, 3), (20 + 15 + 6 + 1) / 64)
        self.assertEqual(_sign_test_p_value(0, 0), 1.0)

    def test_mirror_pair_swaps_only_first_player(self) -> None:
        runner = self._runner(me_damage=8, opp_damage=3)
        pair = runner.run_pair(_deck(), _deck(), matchup_id="00:mirror", seed=100000)
        self.assertEqual(pair.game_candidate_first.first_player, "me")
        self.assertEqual(pair.game_candidate_second.first_player, "opponent")
        self.assertEqual(pair.game_candidate_first.candidate_side, "me")
        self.assertEqual(pair.game_candidate_second.candidate_side, "me")
        self.assertEqual(pair.seed, 100000)


class PuctArmSelectorAdapterTest(unittest.TestCase):
    def test_adapter_maps_puct_diagnostics_to_cost_keys(self) -> None:
        @dataclass
        class _Res:
            actions: tuple
            selected_action_index: int
            diagnostics: dict

        class _FakeSearch:
            def search(self, snapshot, *, player="me"):
                return _Res(
                    actions=(LegalAction(kind=ActionKind.TURN_END, source_uid=0),),
                    selected_action_index=0,
                    diagnostics={
                        "predictor_call_count": 42,
                        "executed_simulations": 64,
                        "turn_transition_count": 3,
                        "value_source": "terminal",
                        "search_name": "puct_two_player",
                    },
                )

        captured = {}

        def factory(predictor, seed):
            captured["seed"] = seed
            return _FakeSearch()

        selector = PuctArmSelector(predictor=object(), search_factory=factory)
        result = selector.select(object(), player="me", branch_seed=777)
        self.assertEqual(captured["seed"], 777)
        self.assertEqual(result.selected_action_index, 0)
        self.assertEqual(result.diagnostics["predictor_call_count"], 42)
        self.assertEqual(result.diagnostics["micro_simulations"], 64)
        self.assertEqual(result.diagnostics["macro_node_costs"], 3)
        self.assertEqual(result.diagnostics["search_value_source"], "terminal")


def _deck():
    from decision_engine.execution.local_match_setup import LocalDeckList

    return LocalDeckList(name="mirror", class_name="x", card_ids=(1, 2, 3))


class CandidateWinRateTest(unittest.TestCase):
    """candidate 胜率换算：2*win + split 除以 2*accepted_pairs。"""

    @staticmethod
    def _summary(*, win: int, loss: int, split: int, accepted_pairs: int) -> AbSummary:
        return AbSummary(
            accepted_pairs=accepted_pairs,
            rejected_pairs=0,
            win=win,
            loss=loss,
            split=split,
            equal=0,
            sign_test={},
            rejection_reasons={},
        )

    def test_counts_win_not_loss(self) -> None:
        # win=3 / loss=1 / split=2 / 5 对：胜局 = 2*3+2 = 8，总局 = 10。
        summary = self._summary(win=3, loss=1, split=2, accepted_pairs=5)
        self.assertEqual(candidate_win_counts(summary), (8, 10))
        self.assertAlmostEqual(candidate_win_rate(summary), 0.8, places=9)

    def test_regression_not_inverted_to_loss(self) -> None:
        # 回归：曾经把 2*loss+split 照抄进来，1/3/0 被算成 75% 而不是 25%。
        summary = self._summary(win=1, loss=3, split=0, accepted_pairs=4)
        self.assertEqual(candidate_win_counts(summary), (2, 8))
        self.assertAlmostEqual(candidate_win_rate(summary), 0.25, places=9)

    def test_split_counts_half(self) -> None:
        summary = self._summary(win=0, loss=0, split=4, accepted_pairs=4)
        self.assertAlmostEqual(candidate_win_rate(summary), 0.5, places=9)

    def test_no_accepted_pairs_returns_none(self) -> None:
        summary = self._summary(win=0, loss=0, split=0, accepted_pairs=0)
        self.assertEqual(candidate_win_counts(summary), (0, 0))
        self.assertIsNone(candidate_win_rate(summary))


if __name__ == "__main__":
    unittest.main()
