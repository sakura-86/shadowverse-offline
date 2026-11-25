# -*- coding: utf-8 -*-
"""CardEffectFact 提取测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from decision_engine.cards.effect_facts import (
    CardEffectFact,
    EffectFactKind,
    EffectFactTiming,
    EffectTargeting,
    EffectValueSource,
    extract_effect_facts,
    is_fact_applicable,
    parse_effect_metadata,
)
from decision_engine.cards.master_data import BaseCardInfo, CardInfo, FaithAcquiredAbilityText, MasterDataProvider
from decision_engine.cards.patch_loader import load_card_patches
from decision_engine.cards.patches import CardPatch
from decision_engine.cards.resolver import CardResolver
from decision_engine.state.snapshot import CardView


class EffectFactsExtractionTest(unittest.TestCase):
    def test_fixed_fanfare_leader_damage_is_safe_fact(self):
        facts = extract_effect_facts("【入场曲】对对手的主战者造成6点伤害。")

        fact = self._only_kind(facts, EffectFactKind.LEADER_DAMAGE)
        self.assertEqual(fact.timing, EffectFactTiming.FANFARE)
        self.assertEqual(fact.amount, 6)
        self.assertEqual(fact.target_side, "opponent")
        self.assertEqual(fact.target_type, "leader")
        self.assertEqual(fact.value_source, EffectValueSource.FIXED)
        self.assertTrue(fact.safe_to_apply)
        self.assertFalse(fact.requires_oracle)

    def test_random_field_damage_requires_oracle(self):
        facts = extract_effect_facts("自己的妖精·随从进入战场时，对对手的战场上的随机1个随从造成1点伤害。")

        fact = self._only_kind(facts, EffectFactKind.FIELD_DAMAGE)
        self.assertEqual(fact.timing, EffectFactTiming.CONTINUOUS)
        self.assertEqual(fact.amount, 1)
        self.assertEqual(fact.targeting, EffectTargeting.RANDOM)
        self.assertFalse(fact.safe_to_apply)
        self.assertTrue(fact.requires_oracle)
        self.assertIn("random", fact.unsupported_reason)

    def test_turn_end_evolved_leader_damage_keeps_condition(self):
        facts = extract_effect_facts(
            "自己的回合结束时，若本随从为进化后，则对对手的主战者造成8点伤害。"
        )

        fact = self._only_kind(facts, EffectFactKind.LEADER_DAMAGE)
        self.assertEqual(fact.timing, EffectFactTiming.TURN_END)
        self.assertEqual(fact.amount, 8)
        self.assertEqual(fact.conditions, frozenset({"self_evolved", "if"}))
        self.assertFalse(fact.safe_to_apply)
        self.assertTrue(fact.requires_oracle)

    def test_unevolved_random_field_damage_and_heal_split_into_facts(self):
        facts = extract_effect_facts(
            "自己的回合结束时，若本随从为进化前，则对对手的战场上的随机2个随从造成8点伤害。回复自己的主战者8点生命值。"
        )

        kinds = [fact.kind for fact in facts]
        self.assertIn(EffectFactKind.FIELD_DAMAGE, kinds)
        self.assertIn(EffectFactKind.HEAL, kinds)
        damage = self._only_kind(facts, EffectFactKind.FIELD_DAMAGE)
        heal = self._only_kind(facts, EffectFactKind.HEAL)
        self.assertEqual(damage.targeting, EffectTargeting.RANDOM)
        self.assertFalse(damage.safe_to_apply)
        self.assertEqual(heal.amount, 8)
        self.assertEqual(heal.timing, EffectFactTiming.TURN_END)
        self.assertEqual(heal.conditions, frozenset({"self_unevolved", "if"}))
        self.assertFalse(heal.safe_to_apply)

    def test_following_branch_inherits_turn_end_timing(self):
        facts = extract_effect_facts(
            "自己的回合结束时，若本随从为进化前，则对对手的战场上的随机2个随从造成8点伤害。回复自己的主战者8点生命值。若为进化后，则对对手的主战者造成8点伤害。"
        )

        leader = self._only_kind(facts, EffectFactKind.LEADER_DAMAGE)
        self.assertEqual(leader.timing, EffectFactTiming.TURN_END)
        self.assertEqual(leader.amount, 8)
        self.assertEqual(leader.conditions, frozenset({"self_evolved", "if"}))
        self.assertFalse(leader.safe_to_apply)

    def test_blank_line_resets_inherited_context(self):
        facts = extract_effect_facts(
            "自己的回合结束时，若本随从为进化后，则对对手的主战者造成8点伤害。\n\n【进化时】本随从失去【守护】。本随从获得【威慑】。"
        )

        leader = self._only_kind(facts, EffectFactKind.LEADER_DAMAGE)
        remove = self._only_kind(facts, EffectFactKind.KEYWORD_REMOVE)
        gain = self._only_kind(facts, EffectFactKind.KEYWORD_GAIN)
        self.assertEqual(leader.conditions, frozenset({"self_evolved", "if"}))
        self.assertEqual(remove.timing, EffectFactTiming.ON_EVOLVE)
        self.assertEqual(gain.timing, EffectFactTiming.ON_EVOLVE)
        self.assertEqual(remove.conditions, frozenset())
        self.assertEqual(gain.conditions, frozenset())

    def test_unknown_timing_stays_unknown_without_default(self):
        facts = extract_effect_facts("对对手的主战者造成3点伤害。")

        fact = self._only_kind(facts, EffectFactKind.LEADER_DAMAGE)
        self.assertEqual(fact.timing, EffectFactTiming.UNKNOWN)

    def test_deduplicates_identical_facts(self):
        facts = extract_effect_facts("本随从获得【疾驰】。\n\n本随从获得【疾驰】。")

        keyword_facts = [fact for fact in facts if fact.kind is EffectFactKind.KEYWORD_GAIN]
        self.assertEqual(len(keyword_facts), 1)

    def test_variable_x_is_not_safe_to_apply(self):
        facts = extract_effect_facts("【入场曲】选择对手的战场上的1个随从，对其造成X点伤害。X为自己的战场上的土之印数量。")

        fact = self._only_kind(facts, EffectFactKind.FIELD_DAMAGE)
        self.assertEqual(fact.value_source, EffectValueSource.VARIABLE_X)
        self.assertFalse(fact.safe_to_apply)
        self.assertTrue(fact.requires_oracle)

    def test_turn_start_draw_is_unsafe_resource_fact(self):
        facts = extract_effect_facts("自己的回合开始时，抽取1张卡牌。")

        fact = self._only_kind(facts, EffectFactKind.DRAW)
        self.assertEqual(fact.timing, EffectFactTiming.TURN_START)
        self.assertEqual(fact.count, 1)
        self.assertEqual(fact.target_type, "card")
        self.assertFalse(fact.safe_to_apply)
        self.assertTrue(fact.requires_oracle)

    def test_special_result_is_oracle_only_risk_fact(self):
        facts = extract_effect_facts("自己的回合开始时，若本次对战中已被破坏的随从为10个以上，则获得胜利。")

        fact = self._only_kind(facts, EffectFactKind.SPECIAL_RESULT)
        self.assertEqual(fact.timing, EffectFactTiming.TURN_START)
        self.assertIn("if", fact.conditions)
        self.assertEqual(fact.value_source, EffectValueSource.HISTORY_BASED)
        self.assertFalse(fact.safe_to_apply)
        self.assertTrue(fact.requires_oracle)
        self.assertIn("history_dependent", fact.unsupported_reason)
        self.assertIn("special_result", fact.unsupported_reason)

    def test_crest_text_is_knowledge_fact_not_executable_effect(self):
        facts = extract_effect_facts("【入场曲】使自己的主战者获得1个纹章。")

        fact = self._only_kind(facts, EffectFactKind.CREST_OR_FAITH)
        self.assertEqual(fact.timing, EffectFactTiming.FANFARE)
        self.assertEqual(fact.target_side, "self")
        self.assertEqual(fact.target_type, "crest")
        self.assertFalse(fact.safe_to_apply)
        self.assertTrue(fact.requires_oracle)

    def test_variable_definition_is_structured_non_executable_fact(self):
        facts = extract_effect_facts("X为自己的战场上的土之印数量。")

        fact = self._only_kind(facts, EffectFactKind.VARIABLE_DEFINE)
        self.assertEqual(fact.value_source, EffectValueSource.VARIABLE_X)
        self.assertFalse(fact.safe_to_apply)

    def test_variable_initial_value_is_structured_non_executable_fact(self):
        facts = extract_effect_facts("X起始为2。")

        fact = self._only_kind(facts, EffectFactKind.VARIABLE_DEFINE)
        self.assertEqual(fact.value_source, EffectValueSource.VARIABLE_X)
        self.assertFalse(fact.safe_to_apply)

    def test_hand_spellboost_extracts_resource_fact(self):
        facts = extract_effect_facts("使自己的所有手牌发动2次魔力增幅。")

        fact = self._only_kind(facts, EffectFactKind.SPELLBOOST_HAND)
        self.assertEqual(fact.amount, 2)
        self.assertEqual(fact.target_zone, "hand")
        self.assertEqual(fact.target_side, "self")

    def test_distributed_damage_extracts_structured_fact(self):
        facts = extract_effect_facts("【入场曲】对对手的战场上的所有随从分配6点伤害。")

        fact = self._only_kind(facts, EffectFactKind.DISTRIBUTED_DAMAGE)
        self.assertEqual(fact.amount, 6)
        self.assertEqual(fact.target_side, "opponent")
        self.assertEqual(fact.target_zone, "field")
        self.assertEqual(fact.value_source, EffectValueSource.FIXED)

    def test_earth_sigil_delta_is_not_stat_buff(self):
        facts = extract_effect_facts("使自己的战场上的土之印+2。")

        fact = self._only_kind(facts, EffectFactKind.EARTH_SIGIL)
        self.assertEqual(fact.amount, 2)
        self.assertFalse(any(item.kind is EffectFactKind.STAT_BUFF for item in facts))

    def test_count_based_effect_is_not_safe_to_apply(self):
        facts = extract_effect_facts("【入场曲】对对手的主战者造成X点伤害。X为本次对战中已使用的卡牌数量。")

        fact = self._only_kind(facts, EffectFactKind.LEADER_DAMAGE)
        self.assertEqual(fact.value_source, EffectValueSource.VARIABLE_X)
        self.assertFalse(fact.safe_to_apply)
        self.assertTrue(fact.requires_oracle)
        self.assertIn("variable_x", fact.unsupported_reason)

    def test_snapshot_readable_conditions_get_precise_tags(self):
        cases = [
            ("【入场曲】若为【连击_3】，则对对手的主战者造成3点伤害。", "combo_threshold_3"),
            ("【入场曲】若为【协作_10】，则抽取2张卡牌。", "rally_threshold_10"),
            ("【入场曲】若自己的手牌数为5以上，则回复自己的主战者3点生命值。", "hand_count_threshold_5"),
            ("自己的回合开始时，若本次对战中已被破坏的卡牌数为10以上，则对对手的主战者造成5点伤害。", "history_destroyed_total_threshold_10"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                fact = extract_effect_facts(text)[0]
                self.assertIn(expected, fact.conditions)

    def _only_kind(self, facts, kind):
        matched = [fact for fact in facts if fact.kind is kind]
        self.assertEqual(len(matched), 1, facts)
        return matched[0]


class EffectMetadataTest(unittest.TestCase):
    def test_parse_effect_metadata_reads_patch_payloads(self):
        metadata = parse_effect_metadata(
            "hand_spellboost:3,body:5/5,keyword:rush,keyword:guard,last_words_spellboost:2,refill_to_max",
            "自己的所有手牌进行1次魔力增幅。",
        )

        self.assertEqual(metadata.hand_spellboost, 3)
        self.assertEqual(metadata.body, (5, 5))
        self.assertEqual(metadata.keywords, frozenset({"rush", "guard"}))
        self.assertTrue(metadata.refill_to_max)
        self.assertEqual(metadata.last_words_spellboost, 2)


class EffectFactApplicabilityTest(unittest.TestCase):
    def test_fact_without_conditions_is_applicable(self):
        fact = self._leader_fact("【入场曲】对对手的主战者造成3点伤害。")

        self.assertTrue(is_fact_applicable(fact))

    def test_self_evolved_condition_requires_evolved_source(self):
        fact = self._leader_fact("若为进化后，则对对手的主战者造成8点伤害。")

        self.assertFalse(is_fact_applicable(fact, CardView(uid=1, evolve_state=0)))
        self.assertTrue(is_fact_applicable(fact, CardView(uid=1, evolve_state=1)))
        self.assertTrue(is_fact_applicable(fact, CardView(uid=1, evolve_state=2)))

    def test_self_unevolved_condition_requires_unevolved_source(self):
        fact = self._heal_fact("若为进化前，则回复自己的主战者8点生命值。")

        self.assertTrue(is_fact_applicable(fact, CardView(uid=1, evolve_state=0)))
        self.assertFalse(is_fact_applicable(fact, CardView(uid=1, evolve_state=1)))

    def test_unknown_condition_is_not_applicable(self):
        fact = self._leader_fact("若为【连击_3】，则对对手的主战者造成3点伤害。")

        self.assertFalse(is_fact_applicable(fact, CardView(uid=1, evolve_state=0)))

    def test_conditional_fact_without_source_is_not_applicable(self):
        fact = self._leader_fact("若为进化后，则对对手的主战者造成8点伤害。")

        self.assertFalse(is_fact_applicable(fact))

    def _leader_fact(self, text):
        return next(fact for fact in extract_effect_facts(text) if fact.kind is EffectFactKind.LEADER_DAMAGE)

    def _heal_fact(self, text):
        return next(fact for fact in extract_effect_facts(text) if fact.kind is EffectFactKind.HEAL)


class EffectFactsResolverTest(unittest.TestCase):
    def test_card_resolver_returns_effect_facts_by_card_id(self):
        provider = MasterDataProvider(
            base_cards={
                100: BaseCardInfo(
                    base_card_id=100,
                    name="测试卡",
                    skill_desc="【入场曲】对对手的主战者造成3点伤害。",
                )
            },
            cards={101: CardInfo(card_id=101, base_card_id=100)},
        )
        facts = CardResolver(provider).get_effect_facts(101)

        fact = next(fact for fact in facts if fact.kind is EffectFactKind.LEADER_DAMAGE)
        self.assertEqual(fact.amount, 3)
        self.assertEqual(fact.timing, EffectFactTiming.FANFARE)

    def test_spell_without_explicit_timing_defaults_to_on_play(self):
        provider = MasterDataProvider(
            base_cards={
                100: BaseCardInfo(
                    base_card_id=100,
                    card_type="spell",
                    skill_desc="对对手的主战者造成3点伤害。",
                )
            },
            cards={101: CardInfo(card_id=101, base_card_id=100)},
        )
        facts = CardResolver(provider).get_effect_facts(101)

        fact = next(fact for fact in facts if fact.kind is EffectFactKind.LEADER_DAMAGE)
        self.assertEqual(fact.timing, EffectFactTiming.ON_PLAY)

    def test_unknown_card_returns_empty_facts(self):
        self.assertEqual(CardResolver(MasterDataProvider()).get_effect_facts(999), ())

    def test_resolver_extracts_facts_from_evolve_and_super_evolve_texts(self):
        provider = MasterDataProvider(
            base_cards={
                100: BaseCardInfo(
                    base_card_id=100,
                    evolve_desc="【进化时】回复自己的主战者3点生命值。",
                    super_evolve_desc="【超进化时】对对手的主战者造成2点伤害。",
                )
            },
            cards={101: CardInfo(card_id=101, base_card_id=100)},
        )
        facts = CardResolver(provider).get_effect_facts(101)

        heal = next(fact for fact in facts if fact.kind is EffectFactKind.HEAL)
        damage = next(fact for fact in facts if fact.kind is EffectFactKind.LEADER_DAMAGE)
        self.assertEqual(heal.timing, EffectFactTiming.ON_EVOLVE)
        self.assertEqual(heal.amount, 3)
        self.assertEqual(damage.timing, EffectFactTiming.ON_SUPER_EVOLVE)
        self.assertEqual(damage.amount, 2)

    def test_resolver_extracts_faith_acquired_ability_texts(self):
        provider = MasterDataProvider(
            base_cards={
                100: BaseCardInfo(
                    base_card_id=100,
                    skill_desc="【入场曲】使自己的主战者获得1个信仰。",
                    faith_acquired_ability_texts=(
                        FaithAcquiredAbilityText(
                            related_base_card_id=100,
                            clean_text="自己的回合结束时，对对手的主战者造成1点伤害。",
                        ),
                    ),
                )
            },
            cards={101: CardInfo(card_id=101, base_card_id=100)},
        )
        facts = CardResolver(provider).get_effect_facts(101)

        crest = next(fact for fact in facts if fact.kind is EffectFactKind.CREST_OR_FAITH)
        damage = next(fact for fact in facts if fact.kind is EffectFactKind.LEADER_DAMAGE)
        self.assertEqual(crest.timing, EffectFactTiming.FANFARE)
        self.assertFalse(crest.safe_to_apply)
        self.assertEqual(damage.timing, EffectFactTiming.TURN_END)
        self.assertEqual(damage.amount, 1)

    def test_patch_effect_facts_extend_auto_facts(self):
        provider = MasterDataProvider(
            base_cards={100: BaseCardInfo(base_card_id=100, skill_desc="【入场曲】抽取1张卡牌。")},
            cards={101: CardInfo(card_id=101, base_card_id=100)},
        )
        patch = CardPatch(
            base_card_id=100,
            effect_facts=(
                CardEffectFact(
                    timing=EffectFactTiming.TURN_END,
                    kind=EffectFactKind.LEADER_DAMAGE,
                    amount=2,
                    target_side="opponent",
                    target_type="leader",
                    safe_to_apply=False,
                    requires_oracle=True,
                    raw_text="manual patch: turn end leader damage",
                ),
            ),
        )
        facts = CardResolver(provider, patches={100: patch}).get_effect_facts(101)

        self.assertTrue(any(fact.kind is EffectFactKind.DRAW for fact in facts))
        patched = [fact for fact in facts if fact.raw_text == "manual patch: turn end leader damage"]
        self.assertEqual(len(patched), 1)
        self.assertEqual(patched[0].timing, EffectFactTiming.TURN_END)
        self.assertEqual(patched[0].amount, 2)

    def test_patch_can_filter_known_bad_auto_fact_by_raw_text(self):
        ignored_text = "【入场曲】抽取1张卡牌。"
        provider = MasterDataProvider(
            base_cards={100: BaseCardInfo(base_card_id=100, skill_desc=ignored_text)},
            cards={101: CardInfo(card_id=101, base_card_id=100)},
        )
        patch = CardPatch(base_card_id=100, ignored_auto_fact_raw_texts=(ignored_text,))

        facts = CardResolver(provider, patches={100: patch}).get_effect_facts(101)

        self.assertFalse(any(fact.kind is EffectFactKind.DRAW for fact in facts))

    def test_patch_loader_reads_effect_facts_and_ignored_auto_facts(self):
        payload = {
            "patches": [
                {
                    "base_card_id": 100,
                    "name": "测试补丁",
                    "ignored_auto_fact_raw_texts": ["bad auto text"],
                    "effect_facts": [
                        {
                            "timing": "turn_end",
                            "kind": "leader_damage",
                            "amount": 3,
                            "target_side": "opponent",
                            "target_type": "leader",
                            "requires_oracle": True,
                            "raw_text": "manual fact",
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "patches.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            patch = load_card_patches(path)[100]

        self.assertEqual(patch.ignored_auto_fact_raw_texts, ("bad auto text",))
        self.assertEqual(len(patch.effect_facts), 1)
        fact = patch.effect_facts[0]
        self.assertEqual(fact.timing, EffectFactTiming.TURN_END)
        self.assertEqual(fact.kind, EffectFactKind.LEADER_DAMAGE)
        self.assertEqual(fact.amount, 3)
        self.assertEqual(fact.raw_text, "manual fact")


if __name__ == "__main__":
    unittest.main()
