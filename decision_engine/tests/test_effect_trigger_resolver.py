# -*- coding: utf-8 -*-
"""模板效果触发解析器测试。"""

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.cards.templates import CardIdentity, ExecutableEffect, PrintedStats, TriggerSpec, UnifiedCardTemplate
from decision_engine.execution.effect_trigger_resolver import resolve_template_effects_for_action, resolve_template_effects_for_event
from decision_engine.execution.events import EventKind, ExecutionEvent


class EffectTriggerResolverTest(unittest.TestCase):
    def test_play_matches_fanfare_and_mode_resolution_aliases(self):
        template = _template(
            _effect("choice_1", "mode_choice_resolution"),
            _effect("ability_1", "ability_resolution"),
            _effect("fanfare_1", "fanfare"),
            _effect("evolve_1", "manual_evolve"),
        )

        result = resolve_template_effects_for_action(template, LegalAction(kind=ActionKind.PLAY, source_uid=1))

        self.assertEqual(tuple(effect.effect_id for effect in result.matched_effects), ("choice_1", "ability_1", "fanfare_1"))
        self.assertEqual(result.skipped_effect_ids, ("evolve_1",))

    def test_super_evolve_matches_evolve_super_evolve_and_mode_resolution_events(self):
        template = _template(
            _effect("choice_1", "mode_choice_resolution"),
            _effect("ability_1", "ability_resolution"),
            _effect("evolve_1", "manual_evolve"),
            _effect("super_1", "manual_super_evolve"),
            _effect("becomes_1", "becomes_evolved"),
        )

        result = resolve_template_effects_for_action(template, LegalAction(kind=ActionKind.SUPER_EVOLVE, source_uid=1))

        self.assertEqual(tuple(effect.effect_id for effect in result.matched_effects), ("choice_1", "ability_1", "evolve_1", "super_1", "becomes_1"))

    def test_activate_matches_activate_and_activation_aliases(self):
        template = _template(_effect("activate_1", "activate"), _effect("activation_1", "activation"))

        result = resolve_template_effects_for_action(template, LegalAction(kind=ActionKind.ACTIVATE, source_uid=1))

        self.assertEqual(tuple(effect.effect_id for effect in result.matched_effects), ("activate_1", "activation_1"))

    def test_attack_follower_uses_clash_event_when_present(self):
        template = _template(_effect("attack_1", "on_attack"), _effect("clash_1", "on_clash"))

        result = resolve_template_effects_for_action(
            template,
            LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=1, target_groups=((2,),)),
            (ExecutionEvent(EventKind.CLASH, source_uid=1, target_uid=2),),
        )

        self.assertEqual(tuple(effect.effect_id for effect in result.matched_effects), ("attack_1", "clash_1"))

    def test_core_execution_events_match_template_trigger_names(self):
        template = _template(
            _effect("last_words_1", "last_words"),
            _effect("turn_end_1", "turn_end"),
            _effect("turn_start_1", "turn_start"),
            _effect("countdown_1", "countdown"),
        )

        result = resolve_template_effects_for_action(
            template,
            LegalAction(kind=ActionKind.TURN_END, source_uid=1),
            (
                ExecutionEvent(EventKind.LAST_WORDS, source_uid=2, player="me"),
                ExecutionEvent(EventKind.TURN_END, player="me"),
                ExecutionEvent(EventKind.TURN_START, player="opponent"),
                ExecutionEvent(EventKind.COUNTDOWN, source_uid=3, player="opponent"),
            ),
        )

        self.assertEqual(
            tuple(effect.effect_id for effect in result.matched_effects),
            ("last_words_1", "turn_end_1", "turn_start_1", "countdown_1"),
        )
        self.assertEqual(result.skipped_effect_ids, ())

    def test_event_resolution_matches_only_current_event_name(self):
        template = _template(_effect("turn_end_1", "turn_end"), _effect("damage_1", "damage"))

        result = resolve_template_effects_for_event(template, ExecutionEvent(EventKind.DAMAGE, source_uid=1, player="me"))

        self.assertEqual(tuple(effect.effect_id for effect in result.matched_effects), ("damage_1",))
        self.assertEqual(result.skipped_effect_ids, ("turn_end_1",))

    def test_last_words_event_matches_last_words_effect(self):
        template = _template(_effect("last_words_1", "last_words"))

        result = resolve_template_effects_for_event(template, ExecutionEvent(EventKind.LAST_WORDS, source_uid=2, player="me"))

        self.assertEqual(tuple(effect.effect_id for effect in result.matched_effects), ("last_words_1",))
        self.assertEqual(result.skipped_effect_ids, ())

    def test_source_buffed_event_matches_on_source_buffed_effect(self):
        template = _template(_effect("buffed_1", "on_source_buffed"))

        result = resolve_template_effects_for_event(template, ExecutionEvent(EventKind.SOURCE_BUFFED, source_uid=2, player="me"))

        self.assertEqual(tuple(effect.effect_id for effect in result.matched_effects), ("buffed_1",))
        self.assertEqual(result.skipped_effect_ids, ())


def _template(*effects):
    return UnifiedCardTemplate(
        template_version=1,
        identity=CardIdentity(base_card_id=1001, card_ids=(1001,), normal_card_id=1001, name="测试", class_id=0, class_name="neutral", type_id=1, type_name="follower", rarity="bronze", rarity_name="bronze"),
        printed=PrintedStats(cost=1, atk=1, life=1),
        executable_effects=effects,
    )


def _effect(effect_id, event):
    return ExecutableEffect(effect_id=effect_id, trigger=TriggerSpec(event=event))


if __name__ == "__main__":
    unittest.main()
