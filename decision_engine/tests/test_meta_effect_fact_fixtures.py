# -*- coding: utf-8 -*-
"""当前环境热门卡的 CardEffectFact 回归 fixture。"""

import unittest
from pathlib import Path

from decision_engine.cards.effect_facts import EffectFactKind, EffectFactTiming, EffectTargeting, EffectValueSource
from decision_engine.cards.master_data import MasterDataProvider
from decision_engine.cards.patch_loader import load_card_patches
from decision_engine.cards.resolver import CardResolver
from decision_engine.state.snapshot import CrestView


_DATABASE_PATH = Path("decision_engine/data/public/card_database.json")
_PATCH_PATH = Path("decision_engine/data/card_patches.json")


class MetaEffectFactFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = CardResolver(MasterDataProvider.from_json(_DATABASE_PATH))

    def test_ashgray_bandenait_and_negative_crest(self):
        card_facts = self.resolver.get_effect_facts(10744110)
        crest_facts = self.resolver.get_effect_facts(10744112)

        aoe = self._only_kind(card_facts, EffectFactKind.FIELD_DAMAGE)
        self.assertEqual(aoe.timing, EffectFactTiming.FANFARE)
        self.assertEqual(aoe.amount, 9)
        self.assertEqual(aoe.target_side, "opponent")
        self.assertEqual(aoe.target_zone, "field")
        self.assertEqual(aoe.targeting, EffectTargeting.ALL)
        self.assertFalse(aoe.safe_to_apply)
        self.assertTrue(aoe.requires_oracle)

        crest = self._only_kind(card_facts, EffectFactKind.CREST_OR_FAITH)
        self.assertEqual(crest.timing, EffectFactTiming.ON_SUPER_EVOLVE)
        self.assertEqual(crest.target_side, "opponent")
        self.assertEqual(crest.target_type, "crest")
        self.assertFalse(crest.safe_to_apply)

        damage_facts = [fact for fact in crest_facts if fact.kind is EffectFactKind.LEADER_DAMAGE]
        self.assertEqual(len(damage_facts), 2)
        turn_start = next(fact for fact in damage_facts if fact.timing is EffectFactTiming.TURN_START)
        continuous = next(fact for fact in damage_facts if fact.timing is EffectFactTiming.CONTINUOUS)
        self.assertEqual(turn_start.amount, 2)
        self.assertEqual(turn_start.target_side, "self")
        self.assertTrue(turn_start.safe_to_apply)
        self.assertFalse(turn_start.requires_oracle)
        self.assertEqual(continuous.amount, 1)
        self.assertEqual(continuous.value_source, EffectValueSource.COUNT_BASED)

    def test_gildaria_and_countdown_crest(self):
        card_facts = self.resolver.get_effect_facts(10724110)
        crest_facts = self.resolver.get_effect_facts(10724112)

        crest = self._only_kind(card_facts, EffectFactKind.CREST_OR_FAITH)
        self.assertEqual(crest.timing, EffectFactTiming.FANFARE)
        self.assertIn("rally", crest.conditions)
        self.assertFalse(crest.safe_to_apply)

        auto_evolve = self._find_fact(card_facts, EffectFactKind.AUTO_EVOLVE, timing=EffectFactTiming.FANFARE)
        self.assertIn("rally", auto_evolve.conditions)
        self.assertFalse(auto_evolve.safe_to_apply)

        keyword = self._find_fact(card_facts, EffectFactKind.KEYWORD_GAIN, timing=EffectFactTiming.CONTINUOUS)
        self.assertFalse(keyword.safe_to_apply)

        summon = self._only_kind(card_facts, EffectFactKind.SUMMON)
        self.assertEqual(summon.timing, EffectFactTiming.ON_EVOLVE)
        self.assertEqual(summon.count, 2)
        self.assertFalse(summon.safe_to_apply)

        crest_damage = self._only_kind(crest_facts, EffectFactKind.LEADER_DAMAGE)
        self.assertEqual(crest_damage.timing, EffectFactTiming.CONTINUOUS)
        self.assertEqual(crest_damage.amount, 1)
        self.assertEqual(crest_damage.target_side, "opponent")
        self.assertFalse(crest_damage.safe_to_apply)

    def test_sandalphon_and_healing_crest(self):
        card_facts = self.resolver.get_effect_facts(10404110)
        crest_facts = self.resolver.get_effect_facts(10404112)

        summon = self._find_fact(card_facts, EffectFactKind.SUMMON, timing=EffectFactTiming.TURN_START)
        self.assertIn("if", summon.conditions)
        self.assertFalse(summon.safe_to_apply)

        crest = self._only_kind(card_facts, EffectFactKind.CREST_OR_FAITH)
        self.assertEqual(crest.timing, EffectFactTiming.UNKNOWN)
        self.assertFalse(crest.safe_to_apply)

        bounce = self._only_kind(card_facts, EffectFactKind.BOUNCE)
        self.assertEqual(bounce.timing, EffectFactTiming.UNKNOWN)
        self.assertFalse(bounce.safe_to_apply)

        damage = self._only_kind(card_facts, EffectFactKind.FIELD_DAMAGE)
        self.assertEqual(damage.timing, EffectFactTiming.FANFARE)
        self.assertEqual(damage.amount, 2)
        self.assertEqual(damage.targeting, EffectTargeting.RANDOM)
        self.assertFalse(damage.safe_to_apply)

        heal = self._only_kind(crest_facts, EffectFactKind.HEAL)
        self.assertEqual(heal.timing, EffectFactTiming.TURN_END)
        self.assertEqual(heal.amount, 1)
        self.assertEqual(heal.target_side, "self")
        self.assertEqual(heal.targeting, EffectTargeting.ALL)
        self.assertFalse(heal.safe_to_apply)

    def test_windblast_spellboost_variable_damage(self):
        facts = self.resolver.get_effect_facts(10131320)

        damage = self._only_kind(facts, EffectFactKind.FIELD_DAMAGE)
        self.assertEqual(damage.timing, EffectFactTiming.ON_PLAY)
        self.assertIsNone(damage.amount)
        self.assertEqual(damage.targeting, EffectTargeting.SELECTED)
        self.assertEqual(damage.value_source, EffectValueSource.VARIABLE_X)
        self.assertFalse(damage.safe_to_apply)
        self.assertTrue(damage.requires_oracle)

    def test_baal_mode_random_options(self):
        facts = self.resolver.get_effect_facts(10452130)

        buff = self._only_kind(facts, EffectFactKind.STAT_BUFF)
        self.assertEqual(buff.timing, EffectFactTiming.FANFARE)
        self.assertEqual(buff.targeting, EffectTargeting.RANDOM)
        self.assertFalse(buff.safe_to_apply)
        self.assertIn("random", buff.unsupported_reason)

        mode = self._only_kind(facts, EffectFactKind.UNKNOWN_COMPLEX)
        self.assertIn("mode_choice", mode.unsupported_reason)

        damage = self._only_kind(facts, EffectFactKind.FIELD_DAMAGE)
        self.assertEqual(damage.timing, EffectFactTiming.FANFARE)
        self.assertEqual(damage.amount, 3)
        self.assertEqual(damage.targeting, EffectTargeting.RANDOM)
        self.assertFalse(damage.safe_to_apply)

    def test_miroku_mode_options_and_evolve_reference(self):
        facts = self.resolver.get_effect_facts(10514120)

        add_hand = self._only_kind(facts, EffectFactKind.ADD_TO_HAND)
        self.assertEqual(add_hand.timing, EffectFactTiming.FANFARE)
        self.assertFalse(add_hand.safe_to_apply)

        pp_change = self._only_kind(facts, EffectFactKind.PP_CHANGE)
        self.assertEqual(pp_change.timing, EffectFactTiming.FANFARE)
        self.assertFalse(pp_change.safe_to_apply)

        unknown_facts = [fact for fact in facts if fact.kind is EffectFactKind.UNKNOWN_COMPLEX]
        self.assertEqual(len(unknown_facts), 1)
        distributed_damage = self._find_fact_with_reason(
            facts,
            EffectFactKind.DISTRIBUTED_DAMAGE,
            timing=EffectFactTiming.FANFARE,
            reason="all_scope",
        )
        self.assertEqual(distributed_damage.targeting, EffectTargeting.ALL)
        self.assertEqual(distributed_damage.amount, 3)
        self.assertFalse(any(fact.timing is EffectFactTiming.ON_EVOLVE for fact in facts))

    def test_bunny_and_baron_rally_damage_and_evolve_resource(self):
        facts = self.resolver.get_effect_facts(10824110)

        summon = self._find_fact(facts, EffectFactKind.SUMMON, timing=EffectFactTiming.FANFARE)
        self.assertEqual(summon.count, 1)
        self.assertFalse(summon.safe_to_apply)

        damage = self._find_fact(facts, EffectFactKind.LEADER_DAMAGE, timing=EffectFactTiming.FANFARE)
        self.assertEqual(damage.amount, 4)
        self.assertEqual(damage.target_side, "opponent")
        self.assertIn("rally", damage.conditions)
        self.assertFalse(damage.safe_to_apply)
        self.assertTrue(damage.requires_oracle)

        add_hand = self._find_fact(facts, EffectFactKind.ADD_TO_HAND, timing=EffectFactTiming.ON_EVOLVE)
        self.assertFalse(add_hand.safe_to_apply)

    def test_scarlet_artifact_history_based_aoe(self):
        facts = self.resolver.get_effect_facts(10774110)

        damage = self._only_kind(facts, EffectFactKind.FIELD_DAMAGE)
        self.assertEqual(damage.timing, EffectFactTiming.FANFARE)
        self.assertIsNone(damage.amount)
        self.assertEqual(damage.target_side, "opponent")
        self.assertEqual(damage.target_zone, "field")
        self.assertEqual(damage.targeting, EffectTargeting.ALL)
        self.assertEqual(damage.value_source, EffectValueSource.VARIABLE_X)
        self.assertFalse(damage.safe_to_apply)
        self.assertTrue(damage.requires_oracle)

        variable = self._only_kind(facts, EffectFactKind.VARIABLE_DEFINE)
        self.assertEqual(variable.value_source, EffectValueSource.HISTORY_BASED)
        self.assertIn("history_dependent", variable.unsupported_reason)

    def _only_kind(self, facts, kind):
        matched = [fact for fact in facts if fact.kind is kind]
        self.assertEqual(len(matched), 1, facts)
        return matched[0]

    def _find_fact(self, facts, kind, *, timing):
        matched = [fact for fact in facts if fact.kind is kind and fact.timing is timing]
        self.assertEqual(len(matched), 1, facts)
        return matched[0]

    def _find_fact_with_reason(self, facts, kind, *, timing, reason):
        matched = [
            fact for fact in facts
            if fact.kind is kind and fact.timing is timing and reason in fact.unsupported_reason
        ]
        self.assertEqual(len(matched), 1, facts)
        return matched[0]


class MetaCrestFactFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = CardResolver(MasterDataProvider.from_json(_DATABASE_PATH))

    def test_meta_crests_are_visible_as_effect_facts(self):
        for crest_view in (CrestView(card_id=10744112), CrestView(card_id=10404112)):
            with self.subTest(card_id=crest_view.card_id):
                facts = self.resolver.get_effect_facts(crest_view.card_id)
                self.assertTrue(facts)
                self.assertTrue(any(fact.kind in {EffectFactKind.LEADER_DAMAGE, EffectFactKind.HEAL} for fact in facts))


class MetaCardPatchFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = CardResolver(
            MasterDataProvider.from_json(_DATABASE_PATH),
            patches=load_card_patches(_PATCH_PATH),
        )

    def test_sandalphon_patch_adds_oracle_only_crest_and_leader_damage_facts(self):
        facts = self.resolver.get_effect_facts(10404110)

        crest = self._only_raw_text(facts, "patch:sandalphon-invoke-crest")
        self.assertEqual(crest.kind, EffectFactKind.CREST_OR_FAITH)
        self.assertEqual(crest.target_type, "crest")
        self.assertTrue(crest.requires_oracle)

        damage = self._only_raw_text(facts, "patch:sandalphon-super-skybound-random-damage")
        self.assertEqual(damage.kind, EffectFactKind.LEADER_DAMAGE)
        self.assertEqual(damage.timing, EffectFactTiming.FANFARE)
        self.assertEqual(damage.amount, 2)
        self.assertEqual(damage.count, 5)
        self.assertEqual(damage.targeting, EffectTargeting.RANDOM)
        self.assertFalse(damage.safe_to_apply)

    def test_miroku_patch_adds_evolve_reference_and_distributed_damage_facts(self):
        facts = self.resolver.get_effect_facts(10514120)

        evolve = self._only_raw_text(facts, "patch:miroku-evolve-reuses-fanfare-mode")
        self.assertEqual(evolve.kind, EffectFactKind.UNKNOWN_COMPLEX)
        self.assertEqual(evolve.timing, EffectFactTiming.ON_EVOLVE)
        self.assertIn("reuses_fanfare", evolve.unsupported_reason)
        self.assertTrue(evolve.requires_oracle)

        damage = self._only_raw_text(facts, "patch:miroku-mode-distributed-field-damage")
        self.assertEqual(damage.kind, EffectFactKind.FIELD_DAMAGE)
        self.assertEqual(damage.timing, EffectFactTiming.FANFARE)
        self.assertEqual(damage.amount, 3)
        self.assertEqual(damage.targeting, EffectTargeting.ALL)
        self.assertIn("distributed_damage", damage.unsupported_reason)
        self.assertFalse(damage.safe_to_apply)

    def test_mars_patch_keeps_triggered_aura_oracle_only(self):
        facts = self.resolver.get_effect_facts(10824120)

        fanfare = self._find_fact(facts, EffectFactKind.SUMMON, timing=EffectFactTiming.FANFARE)
        self.assertEqual(fanfare.count, 3)
        super_evolve = self._find_fact(facts, EffectFactKind.SUMMON, timing=EffectFactTiming.ON_SUPER_EVOLVE)
        self.assertEqual(super_evolve.count, 1)

        soldier_buff = self._only_raw_text(facts, "patch:mars-soldier-entry-buff")
        self.assertEqual(soldier_buff.kind, EffectFactKind.STAT_BUFF)
        self.assertFalse(soldier_buff.safe_to_apply)
        self.assertTrue(soldier_buff.requires_oracle)
        self.assertIn("triggered_aura", soldier_buff.unsupported_reason)

        soldier_rush = self._only_raw_text(facts, "patch:mars-soldier-entry-rush")
        self.assertEqual(soldier_rush.kind, EffectFactKind.KEYWORD_GAIN)
        self.assertFalse(soldier_rush.safe_to_apply)
        self.assertFalse(any(fact.safe_to_apply for fact in facts if fact.kind is EffectFactKind.STAT_BUFF))
        self.assertFalse(any(fact.safe_to_apply for fact in facts if fact.kind is EffectFactKind.KEYWORD_GAIN))

    def test_magic_crystal_patch_filters_mode_safe_misreads(self):
        facts = self.resolver.get_effect_facts(10633310)

        mode = self._find_fact_with_reason(
            facts,
            EffectFactKind.UNKNOWN_COMPLEX,
            timing=EffectFactTiming.ON_PLAY,
            reason="mode_choice",
        )
        self.assertEqual(mode.targeting, EffectTargeting.SELECTED)

        mode_one = self._only_raw_text(facts, "patch:magic-crystal-mode-one-summon")
        self.assertEqual(mode_one.kind, EffectFactKind.SUMMON)
        self.assertEqual(mode_one.count, 1)
        mode_two = self._only_raw_text(facts, "patch:magic-crystal-mode-two-summon")
        self.assertEqual(mode_two.count, 2)
        storm = self._only_raw_text(facts, "patch:magic-crystal-mode-one-storm")
        self.assertEqual(storm.kind, EffectFactKind.KEYWORD_GAIN)
        self.assertFalse(storm.safe_to_apply)
        self.assertFalse(any(fact.safe_to_apply for fact in facts if fact.kind is EffectFactKind.STAT_BUFF))
        self.assertFalse(any(fact.safe_to_apply for fact in facts if fact.kind is EffectFactKind.KEYWORD_GAIN))

    def test_night_song_concert_patch_marks_distributed_damage(self):
        facts = self.resolver.get_effect_facts(10753310)

        damage = self._only_raw_text(facts, "patch:night-song-concert-distributed-field-damage")
        self.assertEqual(damage.kind, EffectFactKind.FIELD_DAMAGE)
        self.assertEqual(damage.timing, EffectFactTiming.ON_PLAY)
        self.assertEqual(damage.amount, 6)
        self.assertEqual(damage.target_side, "opponent")
        self.assertEqual(damage.target_zone, "field")
        self.assertEqual(damage.targeting, EffectTargeting.ALL)
        self.assertFalse(damage.safe_to_apply)
        self.assertTrue(damage.requires_oracle)
        self.assertIn("distributed_damage", damage.unsupported_reason)

        leader_damage = self._find_fact(facts, EffectFactKind.LEADER_DAMAGE, timing=EffectFactTiming.ON_PLAY)
        self.assertEqual(leader_damage.amount, 2)
        self.assertIn("necromancy", leader_damage.conditions)

    def test_kuon_patch_records_shikigami_package_and_super_evolve_burst(self):
        facts = self.resolver.get_effect_facts(70134111)

        summons = [
            fact for fact in facts
            if fact.kind is EffectFactKind.SUMMON
            and fact.timing is EffectFactTiming.FANFARE
            and fact.raw_text.startswith("patch:kuon-summon-")
        ]
        self.assertEqual({fact.target_type for fact in summons}, {"shikigami_tenko", "shikigami_aboki", "shikigami_paper"})
        tenko = next(fact for fact in summons if fact.target_type == "shikigami_tenko")
        self.assertIn("body:4/5", tenko.unsupported_reason)
        self.assertIn("keyword:guard", tenko.unsupported_reason)
        self.assertIn("last_words_spellboost:3", tenko.unsupported_reason)
        aboki = next(fact for fact in summons if fact.target_type == "shikigami_aboki")
        self.assertIn("keyword:rush", aboki.unsupported_reason)
        self.assertIn("last_words_spellboost:1", aboki.unsupported_reason)

        burst = self._only_raw_text(facts, "patch:kuon-super-evolve-grant-storm-to-shikigami-tenko-face-burst-4")
        self.assertEqual(burst.kind, EffectFactKind.KEYWORD_GAIN)
        self.assertEqual(burst.timing, EffectFactTiming.ON_SUPER_EVOLVE)
        self.assertEqual(burst.amount, 4)
        self.assertEqual(burst.target_type, "shikigami_follower")
        self.assertIn("face_burst:4", burst.unsupported_reason)
        self.assertFalse(burst.safe_to_apply)
        self.assertTrue(burst.requires_oracle)

    def _find_fact(self, facts, kind, *, timing):
        matched = [fact for fact in facts if fact.kind is kind and fact.timing is timing]
        self.assertEqual(len(matched), 1, facts)
        return matched[0]

    def _find_fact_with_reason(self, facts, kind, *, timing, reason):
        matched = [
            fact for fact in facts
            if fact.kind is kind and fact.timing is timing and reason in fact.unsupported_reason
        ]
        self.assertEqual(len(matched), 1, facts)
        return matched[0]

    def _only_raw_text(self, facts, raw_text):
        matched = [fact for fact in facts if fact.raw_text == raw_text]
        self.assertEqual(len(matched), 1, facts)
        return matched[0]


if __name__ == "__main__":
    unittest.main()
