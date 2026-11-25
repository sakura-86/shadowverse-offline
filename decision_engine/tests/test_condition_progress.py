# -*- coding: utf-8 -*-
"""A 组：卡牌打出条件计划与进度物化测试。"""

import unittest
from dataclasses import replace

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.cards import CardIdentity, ExecutableEffect, PrintedStats, TriggerSpec, UnifiedCardTemplate
from decision_engine.cards.condition_plan import ConditionPlanCache, build_condition_plan
from decision_engine.cards.templates import CardSourceText, ConditionSpec
from decision_engine.execution import LocalPredictor
from decision_engine.execution.condition_progress import ConditionProgressProjector
from decision_engine.state.snapshot import (
    BattleSnapshot,
    CardType,
    CardView,
    ConditionProgress,
    ConditionState,
    PlayerView,
)
from decision_engine.training.state_features import card_features


def _template(base_card_id: int, *effects: ExecutableEffect, text: str = "") -> UnifiedCardTemplate:
    return UnifiedCardTemplate(
        template_version=1,
        identity=CardIdentity(
            base_card_id=base_card_id,
            name=f"card-{base_card_id}",
            type_id=1,
            type_name="follower",
            class_id=0,
            class_name="test",
            rarity=0,
            rarity_name="test",
        ),
        printed=PrintedStats(cost=3, atk=2, life=2),
        executable_effects=effects,
        source_text=CardSourceText(clean_skill_desc=text),
    )


def _effect(*conditions: ConditionSpec, event: str = "fanfare", play_modes: tuple[str, ...] = ()) -> ExecutableEffect:
    return ExecutableEffect(
        effect_id=f"effect_{event}_{len(conditions)}",
        trigger=TriggerSpec(event=event, play_modes=play_modes),
        conditions=conditions,
    )


def _threshold(subject: str, value: int, operator: str = ">=") -> ConditionSpec:
    return ConditionSpec(kind="snapshot_threshold", subject=subject, operator=operator, value=value)


def _snapshot(cards: dict[int, CardView], *, hand: tuple[int, ...] = (1,), field: tuple[int, ...] = ()) -> BattleSnapshot:
    me = PlayerView(life=20, max_life=20, pp=5, max_pp=5, turn=3, hand=hand, field=field)
    opponent = PlayerView(life=20, max_life=20, pp=0, max_pp=5, turn=3)
    return BattleSnapshot(me=me, opponent=opponent, cards_by_uid=dict(cards), source="test", is_ally_turn=True)


class ConditionPlanTest(unittest.TestCase):
    def test_fanfare_condition_is_collected(self) -> None:
        template = _template(100, _effect(_threshold("source.variable_x", 5)))
        plan = build_condition_plan(template, 100)
        self.assertEqual(plan.coverage, "modeled")
        self.assertEqual(len(plan.keys), 1)
        self.assertTrue(plan.has_conditional_effect)

    def test_non_play_event_condition_is_ignored(self) -> None:
        template = _template(100, _effect(_threshold("me.graveyard_count", 5), event="last_words"))
        plan = build_condition_plan(template, 100)
        self.assertEqual(plan.coverage, "modeled")
        self.assertEqual(plan.keys, ())
        self.assertFalse(plan.has_conditional_effect)

    def test_play_modes_make_effect_relevant(self) -> None:
        template = _template(100, _effect(_threshold("me.play_count", 3), event="unknown_event", play_modes=("enhance",)))
        plan = build_condition_plan(template, 100)
        self.assertEqual(len(plan.keys), 1)

    def test_runtime_and_action_dependent_conditions_are_excluded(self) -> None:
        template = _template(
            100,
            _effect(
                _threshold("runtime.is_enhance", 1),
                ConditionSpec(kind="selection_filter", subject="selected_card", operator="matches"),
            ),
        )
        plan = build_condition_plan(template, 100)
        self.assertEqual(plan.keys, ())
        self.assertEqual(plan.excluded_runtime, 1)
        self.assertEqual(plan.excluded_action_dependent, 1)

    def test_duplicate_conditions_are_deduped(self) -> None:
        condition = _threshold("source.variable_x", 5)
        template = _template(100, _effect(condition), _effect(condition))
        plan = build_condition_plan(template, 100)
        self.assertEqual(len(plan.keys), 1)

    def test_missing_template_is_unknown(self) -> None:
        plan = build_condition_plan(None, 999)
        self.assertEqual(plan.coverage, "unknown")
        self.assertEqual(plan.keys, ())


class KeywordConditionTest(unittest.TestCase):
    """【连击】【奥义】【解放奥义】【协作】【唤灵】【觉醒】等关键字本身就是条件。"""

    def test_combo_threshold_from_text(self) -> None:
        template = _template(100, _effect(), text="【连击】连击3以上：造成3点伤害。")
        plan = build_condition_plan(template, 100)
        self.assertEqual(
            [(c.kind, c.subject, c.operator, c.value) for c in plan.keys],
            [("snapshot_threshold", "me.play_count", ">=", 3)],
        )
        self.assertEqual(plan.keyword_conditions, 1)

    def test_ougi_and_super_ougi(self) -> None:
        plan = build_condition_plan(_template(100, _effect(), text="【奥义】造成5点伤害。"), 100)
        self.assertEqual([c.subject for c in plan.keys], ["ougi"])
        plan2 = build_condition_plan(_template(101, _effect(), text="【解放奥义】造成10点伤害。"), 101)
        self.assertEqual([c.subject for c in plan2.keys], ["super_ougi"])

    def test_kakusei(self) -> None:
        plan = build_condition_plan(_template(100, _effect(), text="【觉醒】获得+2/+2。"), 100)
        self.assertEqual([c.subject for c in plan.keys], ["me.is_kakusei"])

    def test_keyword_without_threshold_is_unresolved(self) -> None:
        plan = build_condition_plan(_template(100, _effect(), text="【魔力增幅】使伤害+1。"), 100)
        self.assertEqual(plan.keys[0].subject, "keyword_魔力增幅")
        self.assertEqual(plan.keyword_unresolved, 1)
        self.assertTrue(plan.has_conditional_effect)

    def test_structured_condition_not_duplicated_by_keyword(self) -> None:
        template = _template(100, _effect(_threshold("me.play_count", 3)), text="连击3以上：造成3点伤害。")
        plan = build_condition_plan(template, 100)
        self.assertEqual(len(plan.keys), 1)
        self.assertEqual(plan.keyword_conditions, 0)

    def test_play_mode_keywords_are_not_conditions(self) -> None:
        # 爆能强化/激奏/结晶是打出模式，不是状态条件，不应触发 has_conditional_effect。
        plan = build_condition_plan(
            _template(100, _effect(), text="【爆能强化】10：造成5点伤害。【激奏】2：抽1张牌。"),
            100,
        )
        self.assertEqual(plan.keys, ())

    def test_keyword_grant_is_not_condition(self) -> None:
        # 「使自己的【连击】+1」是增加连击数，不是条件。
        plan = build_condition_plan(_template(100, _effect(), text="【入场曲】使自己的【连击】+1。"), 100)
        self.assertEqual(plan.keys, ())


class PotentialValueTest(unittest.TestCase):
    """留牌潜在价值的另外两类来源：被舍弃时发动、在手牌中降费。"""

    def test_discard_payoff_from_text(self) -> None:
        plan = build_condition_plan(_template(100, _effect(), text="本卡牌被舍弃时，召唤1个『天刀』。"), 100)
        self.assertTrue(plan.has_discard_payoff)
        self.assertFalse(plan.has_hand_cost_reduction)

    def test_discard_payoff_from_trigger(self) -> None:
        template = _template(100, _effect(event="on_discard_self"), _effect())
        plan = build_condition_plan(template, 100)
        self.assertTrue(plan.has_discard_payoff)

    def test_hand_cost_reduction(self) -> None:
        plan = build_condition_plan(_template(100, _effect(), text="【魔力增幅时】使本卡牌的费用-1。"), 100)
        self.assertTrue(plan.has_hand_cost_reduction)
        self.assertFalse(plan.has_discard_payoff)

    def test_plain_card_has_no_potential_flags(self) -> None:
        plan = build_condition_plan(_template(100, _effect(), text="抽取1张卡牌。"), 100)
        self.assertFalse(plan.has_discard_payoff)
        self.assertFalse(plan.has_hand_cost_reduction)

    def test_progress_carries_potential_flags(self) -> None:
        template = _template(
            100,
            _effect(),
            text="本卡牌被舍弃时，召唤1个『天刀』。\n【魔力增幅时】使本卡牌的费用-1。",
        )
        card = CardView(uid=1, card_id=1001, base_card_id=100, card_type=CardType.SPELL)
        refreshed = ConditionProgressProjector(ConditionPlanCache({100: template})).refresh(_snapshot({1: card}))
        progress = refreshed.condition_progress_by_uid[1]
        self.assertTrue(progress.has_discard_payoff)
        self.assertTrue(progress.has_hand_cost_reduction)

        features = card_features(card, zone="hand", progress=progress)
        self.assertEqual(features["has_discard_payoff"], 1)
        self.assertEqual(features["has_hand_cost_reduction"], 1)


class ConditionProgressTest(unittest.TestCase):
    def _projector(self, templates: dict[int, UnifiedCardTemplate]) -> ConditionProgressProjector:
        return ConditionProgressProjector(ConditionPlanCache(templates))

    def test_unconditional_card_is_none(self) -> None:
        template = _template(100, _effect())
        snapshot = _snapshot({1: CardView(uid=1, card_id=1001, base_card_id=100, card_type=CardType.SPELL)})
        refreshed = self._projector({100: template}).refresh(snapshot)
        progress = refreshed.condition_progress_by_uid[1]
        self.assertEqual(progress.state, ConditionState.NONE)
        self.assertFalse(progress.has_conditional_effect)

    def test_satisfied_condition(self) -> None:
        template = _template(100, _effect(_threshold("source.variable_x", 5)))
        card = CardView(uid=1, card_id=1001, base_card_id=100, card_type=CardType.SPELL, variable_x=6)
        refreshed = self._projector({100: template}).refresh(_snapshot({1: card}))
        progress = refreshed.condition_progress_by_uid[1]
        self.assertEqual(progress.state, ConditionState.SATISFIED)
        self.assertTrue(progress.has_conditional_effect)
        self.assertEqual(progress.satisfied, 1)

    def test_unsatisfied_condition(self) -> None:
        template = _template(100, _effect(_threshold("source.variable_x", 5)))
        card = CardView(uid=1, card_id=1001, base_card_id=100, card_type=CardType.SPELL, variable_x=2)
        refreshed = self._projector({100: template}).refresh(_snapshot({1: card}))
        progress = refreshed.condition_progress_by_uid[1]
        self.assertEqual(progress.state, ConditionState.UNSATISFIED)
        self.assertEqual(progress.unsatisfied, 1)

    def test_unsatisfied_within_group_wins_over_unknown(self) -> None:
        # 同一 spec 的多个条件是 AND：组内任一 false 即该组未达成，不再计 unknown。
        template = _template(
            100,
            _effect(
                _threshold("source.variable_x", 5),
                ConditionSpec(kind="text_condition", subject="if", operator="requires_review", value=None),
            ),
        )
        card = CardView(uid=1, card_id=1001, base_card_id=100, card_type=CardType.SPELL, variable_x=1)
        refreshed = self._projector({100: template}).refresh(_snapshot({1: card}))
        progress = refreshed.condition_progress_by_uid[1]
        self.assertEqual(progress.state, ConditionState.UNSATISFIED)
        self.assertEqual(progress.unsatisfied, 1)
        self.assertEqual(progress.unknown, 0)

    def test_groups_are_ored(self) -> None:
        # 两个不同 spec：一组 false、一组 true → 至少一条强化分支可用 → SATISFIED。
        template = _template(
            100,
            _effect(_threshold("source.variable_x", 5)),
            _effect(_threshold("me.play_count", 1, "==")),
        )
        card = CardView(uid=1, card_id=1001, base_card_id=100, card_type=CardType.SPELL, variable_x=1)
        snapshot = _snapshot({1: card})
        snapshot = replace(snapshot, me=replace(snapshot.me, play_count=1))
        refreshed = self._projector({100: template}).refresh(snapshot)
        progress = refreshed.condition_progress_by_uid[1]
        self.assertEqual(progress.state, ConditionState.SATISFIED)
        self.assertEqual(progress.satisfied, 1)
        self.assertEqual(progress.unsatisfied, 1)

    def test_mutually_exclusive_branches_do_not_cancel(self) -> None:
        # if/else 两个 spec（==1 / ==0）不能相加成矛盾条件：总有一组成立。
        template = _template(
            100,
            _effect(_threshold("me.play_count", 1, "==")),
            _effect(_threshold("me.play_count", 0, "==")),
        )
        card = CardView(uid=1, card_id=1001, base_card_id=100, card_type=CardType.SPELL)
        snapshot = _snapshot({1: card})
        snapshot = replace(snapshot, me=replace(snapshot.me, play_count=0))
        progress = self._projector({100: template}).refresh(snapshot).condition_progress_by_uid[1]
        self.assertEqual(progress.state, ConditionState.SATISFIED)

    def test_unresolvable_condition_is_unknown(self) -> None:
        template = _template(
            100,
            _effect(ConditionSpec(kind="text_condition", subject="if", operator="requires_review", value=None)),
        )
        card = CardView(uid=1, card_id=1001, base_card_id=100, card_type=CardType.SPELL)
        refreshed = self._projector({100: template}).refresh(_snapshot({1: card}))
        progress = refreshed.condition_progress_by_uid[1]
        self.assertEqual(progress.state, ConditionState.UNKNOWN)
        self.assertTrue(progress.has_conditional_effect)

    def test_missing_template_card_is_unknown_without_condition(self) -> None:
        card = CardView(uid=1, card_id=9991, base_card_id=999, card_type=CardType.SPELL)
        refreshed = self._projector({100: _template(100)}).refresh(_snapshot({1: card}))
        progress = refreshed.condition_progress_by_uid[1]
        self.assertEqual(progress.state, ConditionState.UNKNOWN)
        self.assertFalse(progress.has_conditional_effect)

    def test_progress_is_identical_for_unchanged_snapshot(self) -> None:
        template = _template(100, _effect(_threshold("source.variable_x", 5)))
        card = CardView(uid=1, card_id=1001, base_card_id=100, card_type=CardType.SPELL, variable_x=0)
        projector = self._projector({100: template})
        once = projector.refresh(_snapshot({1: card}))
        twice = projector.refresh(once)
        # 幂等：状态未变时不重建 snapshot（返回同一对象）。
        self.assertIs(once, twice)

    def test_disabled_projector_without_templates(self) -> None:
        snapshot = _snapshot({1: CardView(uid=1, card_id=1001, base_card_id=100, card_type=CardType.SPELL)})
        refreshed = ConditionProgressProjector(ConditionPlanCache({})).refresh(snapshot)
        self.assertIs(refreshed, snapshot)
        self.assertEqual(refreshed.condition_progress_by_uid, {})


class ConditionFeatureTest(unittest.TestCase):
    def test_features_without_progress_are_unknown(self) -> None:
        card = CardView(uid=1, card_id=1001, base_card_id=100, card_type=CardType.SPELL)
        features = card_features(card, zone="hand")
        self.assertEqual(features["has_conditional_effect"], 0)
        self.assertEqual(features["conditional_effect_unknown"], 1)

    def test_features_reflect_condition_state(self) -> None:
        card = CardView(uid=1, card_id=1001, base_card_id=100, card_type=CardType.SPELL)
        satisfied = ConditionProgress(state=ConditionState.SATISFIED, satisfied=1, total=1, has_conditional_effect=True)
        features = card_features(card, zone="hand", progress=satisfied)
        self.assertEqual(features["has_conditional_effect"], 1)
        self.assertEqual(features["conditional_effect_satisfied"], 1)
        self.assertEqual(features["conditional_effect_unknown"], 0)

        plain = ConditionProgress(state=ConditionState.NONE)
        plain_features = card_features(card, zone="hand", progress=plain)
        self.assertEqual(plain_features["has_conditional_effect"], 0)
        self.assertEqual(plain_features["conditional_effect_unknown"], 0)


class ConditionPredictorIntegrationTest(unittest.TestCase):
    def test_predict_attaches_condition_progress(self) -> None:
        template = _template(100, _effect(_threshold("source.variable_x", 5)))
        snapshot = _snapshot(
            {1: CardView(uid=1, card_id=1001, base_card_id=100, cost=3, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER, variable_x=5)},
        )
        predictor = LocalPredictor(templates_by_base_card_id={100: template})
        result = predictor.predict(snapshot, LegalAction(kind=ActionKind.PLAY, source_uid=1, cost=3))

        progress = result.snapshot.condition_progress_by_uid.get(1)
        self.assertIsNotNone(progress)
        self.assertEqual(progress.state, ConditionState.SATISFIED)

    def test_predictor_without_templates_keeps_progress_empty(self) -> None:
        snapshot = _snapshot(
            {1: CardView(uid=1, card_id=1001, base_card_id=100, cost=3, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER)},
        )
        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.PLAY, source_uid=1, cost=3))
        self.assertEqual(result.snapshot.condition_progress_by_uid, {})


if __name__ == "__main__":
    unittest.main()
