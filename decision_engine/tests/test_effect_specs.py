# -*- coding: utf-8 -*-
"""EffectSpec schema 与初始规格测试。"""

import unittest
from pathlib import Path

from decision_engine.cards import (
    CardEffectSpecBundle,
    AmountFormula,
    EffectSpec,
    EffectSpecCondition,
    EffectSpecOperation,
    EffectSpecPersistentEffect,
    EffectSpecTarget,
    EffectSpecTrigger,
    load_effect_spec_bundles,
    validate_effect_spec_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EFFECT_SPECS = REPO_ROOT / "decision_engine" / "data" / "public" / "card_effect_specs.json"


class EffectSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundles = load_effect_spec_bundles(EFFECT_SPECS)
        cls.by_id = {bundle.base_card_id: bundle for bundle in cls.bundles}

    def test_generated_effect_specs_pass_integrity_validation(self):
        import json

        payload = json.loads(EFFECT_SPECS.read_text(encoding="utf-8"))
        validated = validate_effect_spec_payload(payload)

        self.assertEqual(validated, self.bundles)
        self.assertEqual(payload["count"], len(validated))

    def test_integrity_validation_rejects_count_mismatch(self):
        payload = {"count": 2, "cards": [CardEffectSpecBundle(base_card_id=1, name="测试").to_dict()]}

        with self.assertRaisesRegex(ValueError, "payload.count=2 does not match cards=1"):
            validate_effect_spec_payload(payload)

    def test_integrity_validation_rejects_duplicate_bundle_and_spec_ids(self):
        first = CardEffectSpecBundle(
            base_card_id=1,
            name="一",
            specs=(EffectSpec(spec_id="duplicate", base_card_id=1, name="一", trigger=EffectSpecTrigger(event="fanfare")),),
        )
        second = CardEffectSpecBundle(
            base_card_id=1,
            name="二",
            specs=(EffectSpec(spec_id="duplicate", base_card_id=1, name="二", trigger=EffectSpecTrigger(event="fanfare")),),
        )

        with self.assertRaisesRegex(ValueError, "duplicate bundle base_card_id=1") as context:
            validate_effect_spec_payload({"count": 2, "cards": [first.to_dict(), second.to_dict()]})
        self.assertIn("duplicate spec_id=duplicate", str(context.exception))

    def test_integrity_validation_rejects_bundle_and_persistent_source_mismatches(self):
        nested = EffectSpec(spec_id="nested", base_card_id=2, name="测试", trigger=EffectSpecTrigger(event="turn_end"))
        persistent = EffectSpecPersistentEffect(
            effect_id="persistent",
            owner="self",
            source_zone="field",
            source_card_ref=2,
            executable_effects=(nested,),
        )
        bundle = CardEffectSpecBundle(
            base_card_id=1,
            name="测试",
            specs=(EffectSpec(spec_id="main", base_card_id=2, name="测试", trigger=EffectSpecTrigger(event="fanfare")),),
            persistent_specs=(persistent,),
        )

        with self.assertRaisesRegex(ValueError, "spec main base_card_id=2 does not match bundle") as context:
            validate_effect_spec_payload({"count": 1, "cards": [bundle.to_dict()]})
        self.assertIn("source_card_ref=2 does not match bundle", str(context.exception))
        self.assertIn("spec nested base_card_id=2 does not match bundle", str(context.exception))

    def test_integrity_validation_rejects_inconsistent_generated_card_references(self):
        spec = EffectSpec(
            spec_id="generated",
            base_card_id=1,
            name="测试",
            trigger=EffectSpecTrigger(event="fanfare"),
            operations=(
                EffectSpecOperation(
                    op="summon",
                    params={
                        "candidate_base_card_ids": [100],
                        "generated_card_refs": [{"kind": "exact", "base_card_ids": [200]}],
                    },
                ),
            ),
        )
        bundle = CardEffectSpecBundle(base_card_id=1, name="测试", specs=(spec,))

        with self.assertRaisesRegex(ValueError, "must be included in candidate_base_card_ids"):
            validate_effect_spec_payload({"count": 1, "cards": [bundle.to_dict()]})

    def test_integrity_validation_rejects_preferred_generated_card_outside_reference_pool(self):
        spec = EffectSpec(
            spec_id="generated_preferred",
            base_card_id=1,
            name="测试",
            trigger=EffectSpecTrigger(event="fanfare"),
            operations=(
                EffectSpecOperation(
                    op="summon",
                    params={
                        "candidate_base_card_ids": [100],
                        "generated_card_refs": [
                            {"kind": "candidate_pool", "base_card_ids": [100], "preferred_base_card_ids": [200]}
                        ],
                    },
                ),
            ),
        )
        bundle = CardEffectSpecBundle(base_card_id=1, name="测试", specs=(spec,))

        with self.assertRaisesRegex(ValueError, "preferred_base_card_ids must be included in base_card_ids"):
            validate_effect_spec_payload({"count": 1, "cards": [bundle.to_dict()]})

    def test_effect_spec_round_trip_preserves_nested_fields(self):
        spec = self.by_id[10134120].specs[1]
        restored = EffectSpec.from_dict(spec.to_dict())

        self.assertEqual(restored, spec)
        self.assertEqual(restored.trigger.event, "manual_evolve")
        self.assertEqual(restored.trigger.allowed_sources, ("manual_ep", "manual_sep"))
        self.assertEqual(restored.operations[0].amount.value, 3)

    def test_bundle_round_trip_preserves_status(self):
        bundle = self.by_id[10134110]
        restored = CardEffectSpecBundle.from_dict(bundle.to_dict())

        self.assertEqual(restored, bundle)
        self.assertEqual(
            tuple(spec.spec_id for spec in restored.specs),
            (
                "10134110_fanfare_summon_shikigami_package",
                "10134110_enhance_10_destroy_and_summon",
                "10134110_super_evolve_grant_storm_shikigami",
            ),
        )
        self.assertEqual(restored.persistent_specs, ())

    def test_snapshot_condition_and_amount_round_trip(self):
        spec = EffectSpec(
            spec_id="999_snapshot_amount",
            base_card_id=999,
            name="测试快照字段",
            trigger=EffectSpecTrigger(event="fanfare"),
            conditions=(
                EffectSpecCondition(
                    kind="snapshot_threshold",
                    subject="me.play_count",
                    operator=">=",
                    value=3,
                ),
            ),
            operations=(
                EffectSpecOperation(
                    op="draw",
                    target_ref="self_player",
                    amount=AmountFormula(
                        kind="snapshot_count",
                        variable="me.hand",
                        params={"count": "len"},
                    ),
                ),
            ),
        )
        restored = EffectSpec.from_dict(spec.to_dict())

        self.assertEqual(restored, spec)
        self.assertEqual(restored.conditions[0].subject, "me.play_count")
        self.assertEqual(restored.operations[0].amount.variable, "me.hand")

    def test_target_template_fields_round_trip(self):
        target = EffectSpecTarget(
            kind="selected_enemy_follower",
            side="enemy",
            zone="field",
            card_type="follower",
            count=1,
            selector_refs=("previous_choice",),
            relation="other_than_source",
            scope="selected",
            filters=({"max_attack": 4},),
        )
        restored = EffectSpecTarget.from_dict(target.to_dict()) if hasattr(EffectSpecTarget, "from_dict") else EffectSpecTarget(**target.to_dict())

        self.assertEqual(restored, target)
        self.assertEqual(restored.scope, "selected")
        self.assertEqual(restored.relation, "other_than_source")

    def test_persistent_effect_spec_round_trip(self):
        nested = EffectSpec(
            spec_id="999_persistent_draw",
            base_card_id=999,
            name="测试持续效果",
            trigger=EffectSpecTrigger(event="turn_end"),
            target=EffectSpecTarget(kind="self_player", side="self", scope="owner"),
            operations=(EffectSpecOperation(op="draw", target_ref="self_player"),),
        )
        persistent = EffectSpecPersistentEffect(
            effect_id="999_turn_end_draw",
            owner="self",
            source_zone="field",
            source_card_ref=999,
            trigger=EffectSpecTrigger(event="turn_end"),
            duration="while_on_field",
            executable_effects=(nested,),
        )
        bundle = CardEffectSpecBundle(base_card_id=999, name="测试", persistent_specs=(persistent,))
        restored = CardEffectSpecBundle.from_dict(bundle.to_dict())

        self.assertEqual(restored, bundle)
        self.assertEqual(restored.persistent_specs[0].duration, "while_on_field")
        self.assertEqual(restored.persistent_specs[0].executable_effects[0].target.scope, "owner")

    def test_homework_transform_is_hand_spellboost_not_fanfare(self):
        specs = {spec.spec_id: spec for spec in self.by_id[10133310].specs}

        self.assertEqual(set(specs), {"10133310_auto_001", "10133310_auto_002", "10133310_auto_003"})
        transform_spec = specs["10133310_auto_001"]
        self.assertEqual(transform_spec.trigger.event, "spellboost")
        self.assertEqual(transform_spec.trigger.source_zone, "hand")
        transform = next(operation for operation in transform_spec.operations if operation.op == "transform")
        self.assertEqual(transform.target_ref, "source_card")
        self.assertIn(90033310, transform.params["candidate_base_card_ids"])
        self.assertIn(90033310, transform.params["generated_card_refs"][0]["preferred_base_card_ids"])
        self.assertEqual(transform.params["condition"], {"subject": "source.variable_x", "operator": ">=", "value": 5})
        self.assertEqual(specs["10133310_auto_002"].trigger.event, "fanfare")
        self.assertEqual(specs["10133310_auto_002"].operations[0].op, "draw")
        self.assertEqual(specs["10133310_auto_003"].trigger.event, "spellboost")
        self.assertEqual(specs["10133310_auto_003"].operations[0].op, "modify_source_variable")

    def test_seed_specs_have_expected_ids(self):
        expected_seed_ids = {90033310, 90034340, 90034350, 90054130, 90064310, 10031310, 10042310, 10131320, 10133310, 10133320, 10134110, 10134120, 10134310, 10161210, 10162110, 10164120, 10201310, 10264110, 10264120, 10264122, 10304120, 10352110, 10352210, 10361120, 10362220, 10364120, 10444120, 10464120, 10531310, 10532120, 10534120, 10542120, 10564110, 10602210, 10802310, 10804120, 10831110, 10862110, 10832310, 10833310, 10834110, 10861110, 10863110, 10864110, 10864120}

        puzzle_include_ids = {10031210, 10062210, 10112210, 10204110, 10401120, 10414110, 90011310, 90021310, 90071210, 90071220, 90073120}
        self.assertTrue(expected_seed_ids <= set(self.by_id))
        self.assertTrue(puzzle_include_ids <= set(self.by_id))


    def test_marwynn_fanfare_adds_despair_torrent_token_to_hand(self):
        bundle = self.by_id[10364120]
        self.assertEqual(bundle.name, "绝望的显现·玛温")

        specs = {spec.spec_id: spec for spec in bundle.specs}
        self.assertEqual(set(specs), {"10364120_auto_001", "10364120_auto_002"})

        fanfare = specs["10364120_auto_001"]
        self.assertEqual(fanfare.trigger.event, "fanfare")
        self.assertEqual(fanfare.trigger.play_origin, "hand")
        self.assertEqual(fanfare.trigger.play_modes, ("normal_play",))
        self.assertEqual(len(fanfare.operations), 1)
        add_to_hand = fanfare.operations[0]
        self.assertEqual(add_to_hand.op, "add_to_hand")
        self.assertEqual(add_to_hand.target_ref, "self_hand")
        self.assertEqual(add_to_hand.params["candidate_base_card_ids"], [90064310])
        self.assertEqual(add_to_hand.params["generated_card_refs"][0]["base_card_ids"], [90064310])

        evolve = specs["10364120_auto_002"]
        self.assertEqual(evolve.trigger.event, "manual_evolve")
        self.assertEqual(evolve.operations[0].op, "add_crest")
        self.assertEqual(evolve.operations[0].params["crest_card_ids"], [10364122])


    def test_evolution_trigger_rules_are_encoded(self):
        anne_grea = self.by_id[10134120].specs[1]
        colette = self.by_id[10863110].specs[1]

        self.assertEqual(anne_grea.trigger, EffectSpecTrigger(event="manual_evolve", allowed_sources=("manual_ep", "manual_sep")))
        self.assertEqual(colette.trigger.event, "becomes_evolved")
        self.assertIn("fanfare_auto", colette.trigger.allowed_sources)
        self.assertNotEqual(colette.trigger.event, "manual_evolve")

    def test_confirmed_summon_policy_is_embedded_in_seed_specs(self):
        kuon = self.by_id[10134110].specs[0]
        summon_policy = kuon.operations[0].summon_policy

        self.assertIsNotNone(summon_policy)
        assert summon_policy is not None
        self.assertEqual(summon_policy.field_full_policy, "fail_to_enter_and_vanish")
        self.assertFalse(summon_policy.overflow_counts_as_destroyed)
        self.assertFalse(summon_policy.overflow_triggers_last_words)
        self.assertFalse(summon_policy.overflow_enters_graveyard)
        self.assertFalse(summon_policy.overflow_emits_enter_events)


if __name__ == "__main__":
    unittest.main()
