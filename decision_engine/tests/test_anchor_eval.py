# -*- coding: utf-8 -*-
"""锚点评测判定逻辑测试（纯统计，不跑真实对局）。"""

from __future__ import annotations

import unittest

from decision_engine.planning.anchor_eval import (
    CONFIRM_TIER,
    SCREEN_TIER,
    VERDICT_CONFIRMED_BETTER,
    VERDICT_CONFIRMED_NOT_BETTER,
    VERDICT_CONFIRMED_WORSE,
    VERDICT_INSUFFICIENT,
    VERDICT_SCREEN_INCONCLUSIVE,
    VERDICT_SCREEN_PROMOTE,
    VERDICT_SCREEN_REJECT,
    evaluate,
    required_games_per_arm,
    required_pairs,
    wilson_interval,
)
from decision_engine.planning.search_ab_match import GameOutcome, PairResult, summarize


def _game(candidate_won: bool | None, *, accepted: bool = True, reason: str = "ok") -> GameOutcome:
    return GameOutcome(
        game_id="g",
        seed=1,
        first_player="me",
        candidate_side="me",
        accepted=accepted,
        winner_side=(("me" if candidate_won else "opponent") if accepted and candidate_won is not None else None),
        candidate_won=(candidate_won if accepted else None),
        reason=reason,
        turns=10,
        steps=50,
        diagnostics={},
    )


def _pair(label: str, *, accepted: bool = True) -> PairResult:
    if not accepted:
        first = _game(None, accepted=False, reason="strict_reject")
        second = _game(None, accepted=False, reason="strict_reject")
    elif label == "win":
        first, second = _game(True), _game(True)
    elif label == "loss":
        first, second = _game(False), _game(False)
    elif label == "split":
        first, second = _game(True), _game(False)
    else:
        raise ValueError(label)
    return PairResult(
        pair_id="p",
        matchup_id="m",
        seed=1,
        game_candidate_first=first,
        game_candidate_second=second,
    )


def _pairs(*, win: int = 0, loss: int = 0, split: int = 0, rejected: int = 0) -> list[PairResult]:
    out = [_pair("win")] * win + [_pair("loss")] * loss + [_pair("split")] * split
    out += [_pair("win", accepted=False)] * rejected
    return out


class WilsonIntervalTest(unittest.TestCase):
    def test_empty_sample_is_uninformative(self) -> None:
        self.assertEqual(wilson_interval(0.0, 0), (0.0, 1.0))

    def test_all_wins_lower_bound_stays_below_one(self) -> None:
        low, high = wilson_interval(48.0, 48)
        # Wilson 在 p=1 时上界不严格等于 1，但应非常接近。
        self.assertGreater(high, 0.99)
        self.assertLess(low, 1.0)
        self.assertGreater(low, 0.90)

    def test_interval_contains_point_estimate(self) -> None:
        low, high = wilson_interval(30.0, 48)
        self.assertLess(low, 30 / 48)
        self.assertGreater(high, 30 / 48)

    def test_half_wins_is_centered_near_half(self) -> None:
        low, high = wilson_interval(200.0, 400)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)


class RequiredSampleTest(unittest.TestCase):
    def test_monotone_in_effect_size(self) -> None:
        self.assertLess(required_games_per_arm(20.0), required_games_per_arm(10.0))
        self.assertLess(required_games_per_arm(10.0), required_games_per_arm(5.0))

    def test_matches_hand_computed_order_of_magnitude(self) -> None:
        # 20pp ≈ 40 局/臂，5pp ≈ 620 局/臂（单侧 alpha=.05, power=.8）。
        self.assertTrue(35 <= required_games_per_arm(20.0) <= 45)
        self.assertTrue(600 <= required_games_per_arm(5.0) <= 650)

    def test_pairs_are_half_of_games(self) -> None:
        self.assertEqual(required_pairs(20.0), (required_games_per_arm(20.0) + 1) // 2)


class ScreenTierTest(unittest.TestCase):
    def test_below_minimum_pairs_is_insufficient(self) -> None:
        # screen 要求 48 局/臂 = 24 对；23 对不够。
        result = evaluate(_pairs(win=23), tier="screen")
        self.assertEqual(result.verdict, VERDICT_INSUFFICIENT)
        self.assertFalse(result.is_conclusive)

    def test_clear_win_promotes_but_never_confirms(self) -> None:
        result = evaluate(_pairs(win=20, loss=4), tier="screen")
        self.assertEqual(result.verdict, VERDICT_SCREEN_PROMOTE)
        # 关键：screen 档即使 48 局全胜也不得输出"确认更好"。
        self.assertFalse(result.is_confirmed)
        self.assertNotEqual(result.verdict, VERDICT_CONFIRMED_BETTER)

    def test_sweep_still_not_confirmed(self) -> None:
        result = evaluate(_pairs(win=24), tier="screen")
        self.assertEqual(result.verdict, VERDICT_SCREEN_PROMOTE)
        self.assertFalse(result.is_confirmed)

    def test_clear_loss_rejects(self) -> None:
        result = evaluate(_pairs(win=4, loss=20), tier="screen")
        self.assertEqual(result.verdict, VERDICT_SCREEN_REJECT)

    def test_even_is_inconclusive(self) -> None:
        result = evaluate(_pairs(win=12, loss=12), tier="screen")
        self.assertEqual(result.verdict, VERDICT_SCREEN_INCONCLUSIVE)

    def test_all_splits_count_as_half_wins(self) -> None:
        result = evaluate(_pairs(split=24), tier="screen")
        self.assertEqual(result.verdict, VERDICT_SCREEN_INCONCLUSIVE)
        self.assertAlmostEqual(result.win_rate, 0.5, places=6)


class ConfirmTierTest(unittest.TestCase):
    def test_below_minimum_pairs_is_insufficient(self) -> None:
        result = evaluate(_pairs(win=150, loss=49), tier="confirm")
        self.assertEqual(result.verdict, VERDICT_INSUFFICIENT)

    def test_clear_majority_confirms(self) -> None:
        result = evaluate(_pairs(win=130, loss=70), tier="confirm")
        self.assertEqual(result.verdict, VERDICT_CONFIRMED_BETTER)
        self.assertTrue(result.is_confirmed)
        self.assertGreater(result.ci_low, 0.5)

    def test_even_is_not_better(self) -> None:
        result = evaluate(_pairs(win=100, loss=100), tier="confirm")
        self.assertEqual(result.verdict, VERDICT_CONFIRMED_NOT_BETTER)
        self.assertFalse(result.is_confirmed)

    def test_clear_minority_is_worse(self) -> None:
        result = evaluate(_pairs(win=60, loss=140), tier="confirm")
        self.assertEqual(result.verdict, VERDICT_CONFIRMED_WORSE)

    def test_rejected_pairs_do_not_count_as_games(self) -> None:
        result = evaluate(_pairs(win=130, loss=70, rejected=30), tier="confirm")
        self.assertEqual(result.accepted_pairs, 200)
        self.assertEqual(result.rejected_pairs, 30)
        self.assertEqual(result.games, 400)


class ZeroSumTerminalTest(unittest.TestCase):
    """影之诗是零和博弈：任何"无胜负"都不得进入 win rate 的分子或分母。"""

    def test_pair_with_unknown_winner_raises_instead_of_split(self) -> None:
        # 回归：旧实现用 `w1 and w2` / `not w1 and not w2` 判断，而 None 是 falsy，
        # 于是 (True, None) 会落进 "split" 分支，被按 0.5 胜场计入分子。
        pair = PairResult(
            pair_id="p",
            matchup_id="m",
            seed=1,
            game_candidate_first=_game(True),
            game_candidate_second=_game(None),
        )
        self.assertTrue(pair.accepted)  # 两局都标了 accepted
        with self.assertRaises(AssertionError) as ctx:
            pair.outcome_label()
        self.assertIn("winner=None", str(ctx.exception))

    def test_summarize_refuses_to_count_unknown_winner(self) -> None:
        bad = PairResult(
            pair_id="p",
            matchup_id="m",
            seed=1,
            game_candidate_first=_game(False),
            game_candidate_second=_game(None),
        )
        with self.assertRaises(AssertionError):
            summarize([bad])

    def test_truncated_game_is_rejected_not_half_win(self) -> None:
        # 终局缺失 → accepted=False → 整对作废，既不进分子也不进分母。
        result = evaluate(_pairs(win=12, loss=12, rejected=12), tier="screen")
        self.assertEqual(result.accepted_pairs, 24)
        self.assertEqual(result.rejected_pairs, 12)
        self.assertEqual(result.games, 48)
        self.assertAlmostEqual(result.win_rate, 0.5, places=6)

    def test_split_is_one_win_one_loss_not_a_draw(self) -> None:
        result = evaluate(_pairs(split=24), tier="screen")
        self.assertEqual(result.split, 24)
        self.assertEqual(result.equal, 0)  # 结构性恒 0，不是"平局数"
        self.assertEqual(result.games, 48)  # 48 局都分出了胜负
        self.assertEqual(result.game_wins, 24.0)


class TierContractTest(unittest.TestCase):
    def test_tier_specs_are_disjoint_on_verdicts(self) -> None:
        self.assertNotIn(VERDICT_CONFIRMED_BETTER, SCREEN_TIER.allowed_verdicts)
        self.assertNotIn(VERDICT_SCREEN_PROMOTE, CONFIRM_TIER.allowed_verdicts)

    def test_confirm_requires_more_samples_than_screen(self) -> None:
        self.assertGreater(CONFIRM_TIER.min_games_per_arm, SCREEN_TIER.min_games_per_arm)

    def test_result_is_jsonable(self) -> None:
        payload = evaluate(_pairs(win=20, loss=4), tier="screen").to_jsonable()
        self.assertEqual(payload["tier"], "screen")
        self.assertIn("verdict", payload)
        self.assertFalse(payload["is_confirmed_better"])

    def test_unknown_tier_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(_pairs(win=1), tier="nope")


if __name__ == "__main__":
    unittest.main()
