# -*- coding: utf-8 -*-
"""换牌作为动作空间（方案 A）的契约测试。

覆盖四件事：

1. ``MulliganActionEnumerator`` 的候选生成（每张手牌一个 CHANGE + 一个 END）；
2. ``validate_changes`` 的结构校验（换牌不进 strict 验收，所以这层校验是唯一的防线）；
3. ``action_features`` 对换牌动作的编码——``source`` 复用 ``card_features``，
   且 **不改变** ``action_scalar`` 宽度（否则所有现有 checkpoint 都会维度不兼容）；
4. ``StrictSearchAbRunner`` 的换牌阶段：candidate 走换牌臂、对手走固定 selector、
   换牌决策进 ``step_recorder``（即进入 PPO 轨迹）。
"""

from __future__ import annotations

import json
import random
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.actions.mulligan_enumerator import (
    MULLIGAN_MAX_CHANGES,
    MulliganActionEnumerator,
)
from decision_engine.execution.local_match_setup import (
    LocalDeckList,
    LocalMatchSetup,
    LocalMatchSetupConfig,
)
from decision_engine.planning.search_ab_match import SearchAbConfig, SearchArm, SelectorResult, StrictSearchAbRunner
from decision_engine.training.action_features import action_features
from decision_engine.training.ppo_rollout import RolloutStep
from decision_engine.training.ppo_trainer import compute_gae
from decision_engine.training.tensorization import (
    ACTION_VECTOR_FEATURE_ORDER,
    action_vectors_from_action_features,
)

DECKS_DIR = Path(__file__).resolve().parents[1] / "data" / "decks" / "10010"
DECK_PATH = DECKS_DIR / "set10_royal_海贼皇_15207.json"


def _load_deck(path: Path) -> LocalDeckList:
    payload = json.loads(path.read_text(encoding="utf-8"))
    card_ids: list[int] = []
    for entry in payload.get("cards") or ():
        card_ids.extend([int(entry["card_id"])] * int(entry.get("count") or 1))
    if not card_ids:
        raise ValueError(f"牌组为空：{path}")
    return LocalDeckList(
        name=str(payload.get("archetype_id") or path.stem),
        class_name=str(payload.get("archetype_id") or ""),
        card_ids=tuple(card_ids),
    )


def _real_snapshot(seed: int = 12345):
    deck = _load_deck(DECK_PATH)
    setup = LocalMatchSetup().create(deck, deck, LocalMatchSetupConfig(seed=seed, first_player="me"))
    if setup.snapshot is None:
        raise AssertionError("LocalMatchSetup 没有产出 snapshot")
    return setup.snapshot


# -- fake snapshot：只用于枚举器 / 校验测试，不需要真实卡牌数据 -----------------


@dataclass(frozen=True)
class _View:
    hand: tuple[int, ...]


@dataclass(frozen=True)
class _Snapshot:
    me: _View
    opponent: _View
    cards_by_uid: dict = field(default_factory=dict)


def _snapshot(me_hand: tuple[int, ...], opp_hand: tuple[int, ...] = ()) -> _Snapshot:
    cards = {}
    for uid in me_hand + opp_hand:
        cards[uid] = object()
    return _Snapshot(me=_View(hand=me_hand), opponent=_View(hand=opp_hand), cards_by_uid=cards)


class MulliganActionEnumeratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.enum = MulliganActionEnumerator()

    def test_offers_one_change_per_card_plus_end(self) -> None:
        actions = self.enum.enumerate(_snapshot((11, 12, 13, 14)), player="me")
        kinds = [a.kind for a in actions]
        self.assertEqual(kinds, [ActionKind.MULLIGAN_CHANGE] * 4 + [ActionKind.MULLIGAN_END])
        self.assertEqual([a.source_uid for a in actions[:-1]], [11, 12, 13, 14])
        self.assertEqual(actions[-1].source_uid, 0)

    def test_supports_five_card_back_hand(self) -> None:
        actions = self.enum.enumerate(_snapshot((1, 2, 3, 4, 5)), player="me")
        changes = [a for a in actions if a.kind is ActionKind.MULLIGAN_CHANGE]
        self.assertEqual(len(changes), 5)

    def test_excludes_already_changed(self) -> None:
        actions = self.enum.enumerate(
            _snapshot((11, 12, 13, 14)), player="me", already_changed=(12, 13)
        )
        changes = [a.source_uid for a in actions if a.kind is ActionKind.MULLIGAN_CHANGE]
        self.assertEqual(changes, [11, 14])

    def test_only_end_at_limit(self) -> None:
        hand = tuple(range(1, 6))
        actions = self.enum.enumerate(
            _snapshot(hand), player="me", already_changed=hand[:MULLIGAN_MAX_CHANGES]
        )
        self.assertEqual([a.kind for a in actions], [ActionKind.MULLIGAN_END])

    def test_opponent_hand_is_used_for_opponent(self) -> None:
        snap = _snapshot((11, 12), opp_hand=(21, 22, 23))
        actions = self.enum.enumerate(snap, player="opponent")
        changes = [a.source_uid for a in actions if a.kind is ActionKind.MULLIGAN_CHANGE]
        self.assertEqual(changes, [21, 22, 23])

    def test_rejects_unknown_player(self) -> None:
        with self.assertRaises(ValueError):
            self.enum.enumerate(_snapshot((11,)), player="third")


class MulliganChangeValidationTest(unittest.TestCase):
    """换牌不进 strict 验收，``validate_changes`` 是唯一的合法性防线。"""

    def setUp(self) -> None:
        self.enum = MulliganActionEnumerator()
        self.snap = _snapshot((11, 12, 13, 14))

    def test_accepts_valid_changes(self) -> None:
        self.enum.validate_changes(self.snap, "me", (11, 13))

    def test_accepts_empty(self) -> None:
        self.enum.validate_changes(self.snap, "me", ())

    def test_rejects_too_many(self) -> None:
        with self.assertRaises(ValueError):
            self.enum.validate_changes(self.snap, "me", (1, 2, 3, 4, 5, 6))

    def test_rejects_duplicate(self) -> None:
        with self.assertRaises(ValueError):
            self.enum.validate_changes(self.snap, "me", (11, 11))

    def test_rejects_uid_outside_hand(self) -> None:
        with self.assertRaises(ValueError):
            self.enum.validate_changes(self.snap, "me", (99,))


class MulliganActionFeaturesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.enum = MulliganActionEnumerator()
        self.snap = _real_snapshot()
        self.actions = self.enum.enumerate(self.snap, player="me")

    def test_change_carries_source_card_features(self) -> None:
        features = action_features(self.snap, self.actions[0], perspective="me")
        self.assertEqual(features["action_kind"], "mulligan_change")
        self.assertTrue(features["is_mulligan_change"])
        self.assertFalse(features["is_mulligan_end"])
        self.assertTrue(features["source"]["known"])
        self.assertIsNotNone(features["source"]["cost"])

    def test_end_has_no_source_card(self) -> None:
        features = action_features(self.snap, self.actions[-1], perspective="me")
        self.assertEqual(features["action_kind"], "mulligan_end")
        self.assertFalse(features["source"]["known"])

    def test_scalar_width_matches_existing_actions(self) -> None:
        """换牌动作不能改变 action_scalar 宽度，否则现有 checkpoint 全部维度不兼容。"""
        vectors = action_vectors_from_action_features(
            [action_features(self.snap, a, perspective="me") for a in self.actions]
        )
        self.assertEqual(len({len(v) for v in vectors}), 1)
        self.assertEqual(len(vectors[0]), len(ACTION_VECTOR_FEATURE_ORDER))

    def test_no_uid_leak(self) -> None:
        features = action_features(self.snap, self.actions[0], perspective="me")
        self.assertNotIn("source_uid", features)
        self.assertNotIn("uid", features["source"])


class _FakeMulliganSelector:
    """按预设返回候选下标；``None`` 表示"没有可选项"（用于测拒绝路径）。"""

    def __init__(self, plan: list[int | None]) -> None:
        self._plan = list(plan)
        self.calls = 0

    def select_mulligan(self, snapshot, actions, *, player, branch_seed):
        index = self._plan[self.calls] if self.calls < len(self._plan) else len(actions) - 1
        self.calls += 1
        return SelectorResult(
            actions=tuple(actions),
            selected_action_index=index,
            diagnostics={"arm": "fake_mulligan"},
        )


def _runner(selector: _FakeMulliganSelector, recorder=None) -> StrictSearchAbRunner:
    arm = SearchArm(name="mulligan_arm", factory=lambda _p: selector, config={"arm": "fake"})
    return StrictSearchAbRunner(
        predictor=None,  # 换牌阶段不调用 predictor，fake factory 也不使用它
        candidate_arm=SearchArm(name="cand", factory=lambda _p: selector, config={}),
        opponent_arm=SearchArm(name="opp", factory=lambda _p: selector, config={}),
        config=SearchAbConfig(),
        step_recorder=recorder,
        candidate_mulligan_arm=arm,
    )


class RunnerMulliganPhaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.snap = _real_snapshot()

    def test_change_then_end_executes_mulligan_and_records_steps(self) -> None:
        recorded: list[dict] = []

        def recorder(**kwargs) -> None:
            recorded.append(kwargs)

        selector = _FakeMulliganSelector([0, 0, -1])  # 换 2 张，再 END（END 恒为最后一个候选）
        runner = _runner(selector, recorder)
        result = runner._run_candidate_mulligan(
            self.snap, "me", arm_name="mulligan_arm", is_candidate_side=True, game_id="g", seed=1, rng=random.Random(1), step_count=0
        )
        self.assertNotIsInstance(result, str)
        after, steps = result
        self.assertEqual(steps, 3)
        self.assertEqual(len(after.me.hand), len(self.snap.me.hand))
        self.assertEqual(len(recorded), 3)
        self.assertEqual([r["action"].kind for r in recorded],
                         [ActionKind.MULLIGAN_CHANGE, ActionKind.MULLIGAN_CHANGE, ActionKind.MULLIGAN_END])
        self.assertTrue(all(r["turn_index"] == 0 for r in recorded))

    def test_immediate_end_leaves_hand_untouched(self) -> None:
        selector = _FakeMulliganSelector([-1])  # 最后一个候选是 MULLIGAN_END
        runner = _runner(selector)
        result = runner._run_candidate_mulligan(
            self.snap, "me", arm_name="mulligan_arm", is_candidate_side=True, game_id="g", seed=1, rng=random.Random(1), step_count=0
        )
        after, steps = result
        self.assertEqual(steps, 1)
        self.assertEqual(tuple(after.me.hand), tuple(self.snap.me.hand))

    def test_no_selection_is_structured_rejection(self) -> None:
        selector = _FakeMulliganSelector([None])
        runner = _runner(selector)
        result = runner._run_candidate_mulligan(
            self.snap, "me", arm_name="mulligan_arm", is_candidate_side=True, game_id="g", seed=1, rng=random.Random(1), step_count=0
        )
        self.assertIsInstance(result, str)
        self.assertIn("mulligan_no_selection", result)

    def test_requires_select_mulligan_on_arm(self) -> None:
        class _BadSelector:
            def select(self, snapshot, *, player, branch_seed):  # pragma: no cover
                raise AssertionError("不应被调用")

        arm = SearchArm(name="bad", factory=lambda _p: _BadSelector(), config={})
        with self.assertRaises(TypeError):
            StrictSearchAbRunner(
                predictor=None,
                candidate_arm=arm,
                opponent_arm=arm,
                config=SearchAbConfig(),
                candidate_mulligan_arm=arm,
            )


def _rollout_step(*, value: float, reward: float, is_mulligan: bool = False) -> RolloutStep:
    return RolloutStep(
        sample_id="s",
        player="me",
        example={},
        selected_index=0,
        log_prob=0.0,
        value=value,
        executable_count=1,
        executable_indices=(0,),
        turn_index=0,
        step_index=0,
        reward=reward,
        is_mulligan=is_mulligan,
    )


class GaeMulliganTest(unittest.TestCase):
    """换牌步的 MC return 选项：只影响换牌步，其余步的 GAE 结果不变。"""

    def _episode(self):
        return [
            _rollout_step(value=0.5, reward=0.0, is_mulligan=True),
            _rollout_step(value=0.2, reward=0.0),
            _rollout_step(value=0.1, reward=1.0),
        ]

    def test_mc_return_uses_episode_total(self) -> None:
        gae = compute_gae([self._episode()], gamma=0.99, lam=0.95, mulligan_mc_return=True)
        self.assertAlmostEqual(gae["advantages"][0], 1.0 - 0.5, places=6)
        self.assertAlmostEqual(gae["returns"][0], 1.0, places=6)

    def test_default_gae_is_unchanged(self) -> None:
        a = compute_gae([self._episode()], gamma=0.99, lam=0.95, mulligan_mc_return=False)["advantages"][0]
        b = compute_gae([self._episode()], gamma=0.99, lam=0.95)["advantages"][0]
        self.assertAlmostEqual(a, b, places=9)
        # GAE 衰减：0.95^2 ≈ 0.90 量级的折扣后，明显小于 MC 的 0.5。
        self.assertLess(a, 1.0 - 0.5)

    def test_non_mulligan_steps_identical_under_both_modes(self) -> None:
        without = compute_gae([self._episode()], gamma=0.99, lam=0.95, mulligan_mc_return=False)
        with_mc = compute_gae([self._episode()], gamma=0.99, lam=0.95, mulligan_mc_return=True)
        for index in (1, 2):
            self.assertAlmostEqual(without["advantages"][index], with_mc["advantages"][index], places=9)
            self.assertAlmostEqual(without["returns"][index], with_mc["returns"][index], places=9)


if __name__ == "__main__":
    unittest.main()

