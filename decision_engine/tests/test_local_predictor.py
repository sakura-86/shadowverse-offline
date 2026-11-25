# -*- coding: utf-8 -*-
"""Phase L7 LocalPredictor 测试。"""

import unittest
from dataclasses import replace

from decision_engine.actions.model import ActionKind, LegalAction, PlayKind
from decision_engine.cards import CardIdentity, ExecutableEffect, PrintedStats, TriggerSpec, UnifiedCardTemplate
from decision_engine.execution import LocalPredictor
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, CrestView, ExtraPpView, PlayerBuffView, PlayerView


class LocalPredictorTest(unittest.TestCase):
    def make_snapshot(self):
        me = PlayerView(life=20, max_life=20, pp=7, max_pp=7, ep=1, sep=1, hand=(1, 2), field=(3,))
        opponent = PlayerView(life=18, max_life=20, pp=0, max_pp=7, field=(4,))
        cards = {
            1: CardView(uid=1, card_id=101, base_card_id=101, name="随从", cost=3, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER),
            2: CardView(uid=2, card_id=102, base_card_id=102, name="法术", cost=2, card_type=CardType.SPELL),
            3: CardView(uid=3, card_id=103, base_card_id=103, name="我方场上", atk=3, life=4, max_life=4, attacks_left=1),
            4: CardView(uid=4, card_id=104, base_card_id=104, name="敌方场上", atk=2, life=3, max_life=3, attacks_left=1),
        }
        return BattleSnapshot(me=me, opponent=opponent, cards_by_uid=cards, source="test", is_ally_turn=True)

    def test_play_follower_returns_partial_without_fallback(self):
        snapshot = self.make_snapshot()
        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.PLAY, source_uid=1, cost=3))

        self.assertEqual(result.certainty, PredictionCertainty.PARTIAL)
        self.assertEqual(result.snapshot.me.pp, 4)
        self.assertEqual(result.snapshot.me.hand, (2,))
        self.assertEqual(result.snapshot.me.field, (3, 1))
        self.assertEqual(result.unresolved_effects, ())
        self.assertEqual(snapshot.me.hand, (1, 2))

    def test_play_spell_returns_spec_only_and_unresolved_effects(self):
        snapshot = self.make_snapshot()
        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.PLAY, source_uid=2, cost=2))

        self.assertEqual(result.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(result.snapshot.me.pp, 5)
        self.assertEqual(result.snapshot.me.hand, (1,))
        self.assertNotIn(2, result.snapshot.cards_by_uid)
        self.assertEqual(result.unresolved_effects, ())

    def test_attack_and_evolve_predictions(self):
        snapshot = self.make_snapshot()
        predictor = LocalPredictor()

        leader = predictor.predict(snapshot, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=3))
        self.assertEqual(leader.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(leader.snapshot.opponent.life, 15)
        self.assertEqual(leader.assumptions, ("no_attack_effect_candidates",))
        self.assertEqual(leader.unresolved_effects, ())

        follower = predictor.predict(snapshot, LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=3, target_groups=((4,),)))
        self.assertEqual(follower.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertNotIn(4, follower.snapshot.cards_by_uid)
        self.assertEqual(follower.assumptions, ("no_attack_effect_candidates",))
        self.assertEqual(follower.unresolved_effects, ())

        evolved = predictor.predict(snapshot, LegalAction(kind=ActionKind.EVOLVE, source_uid=3))
        self.assertEqual(evolved.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(evolved.snapshot.me.ep, 0)
        self.assertEqual(evolved.snapshot.cards_by_uid[3].evolve_state, 1)
        self.assertEqual(evolved.unresolved_effects, ())

        super_evolved = predictor.predict(snapshot, LegalAction(kind=ActionKind.SUPER_EVOLVE, source_uid=3))
        self.assertEqual(super_evolved.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(super_evolved.snapshot.me.sep, 0)
        self.assertEqual(super_evolved.snapshot.cards_by_uid[3].evolve_state, 2)
        self.assertEqual(super_evolved.unresolved_effects, ())

    def test_attack_without_attacks_left_is_predict_error(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[3] = replace(cards[3], attacks_left=0)
        snapshot = replace(snapshot, cards_by_uid=cards)

        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=3))

        self.assertEqual(result.certainty, PredictionCertainty.PREDICT_ERROR)
        self.assertIn("attacker has no attacks left", result.description)
        self.assertEqual(result.snapshot.opponent.life, 18)

    def test_attack_reports_template_on_attack_candidate(self):
        snapshot = self.make_snapshot()
        template = _template(103, ExecutableEffect(effect_id="103_attack", trigger=TriggerSpec(event="on_attack")))

        result = LocalPredictor(templates_by_base_card_id={103: template}).predict(snapshot, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=3))

        # attack_leader/attack_follower 事件现已通过 _EVENT_TO_TRIGGER_ALIASES 映射到
        # on_attack trigger，由 event queue 实际解析【攻击时】效果（此处 template 无
        # operation，解析后为 known-noop），因此不再作为未解析候选上报。
        self.assertEqual(result.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(result.unresolved_effects, ())

    def test_attack_reports_damage_modifier_candidate_only_when_present(self):
        snapshot = self.make_snapshot()
        snapshot = replace(snapshot, opponent=replace(snapshot.opponent, buff=PlayerBuffView(damage_cut=1)))

        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=3))

        # player_buff.damage_cut is modeled by rule_kernel.attack_leader → SPEC_ONLY, no unresolved
        self.assertEqual(result.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(result.unresolved_effects, ())

    def test_attack_ignores_plain_crest_damage_modifier_noise(self):
        snapshot = self.make_snapshot()
        snapshot = replace(snapshot, me=replace(snapshot.me, crests=(CrestView(card_id=9001),)))

        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=3))

        self.assertEqual(result.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(result.unresolved_effects, ())

    def test_attack_reports_damage_modifier_crest_candidate_when_buff_mentions_damage(self):
        snapshot = self.make_snapshot()
        snapshot = replace(snapshot, me=replace(snapshot.me, crests=(CrestView(card_id=9001, buff={"damage_cut": 1}),)))

        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.ATTACK_LEADER, source_uid=3))

        self.assertEqual(result.certainty, PredictionCertainty.PARTIAL)
        self.assertEqual(result.unresolved_effects, ("damage_modifiers:crest_candidates_present",))

    def test_attack_follower_reports_keyword_and_last_words_candidates(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[4] = replace(cards[4], has_bane=True, has_last_words=True)
        snapshot = replace(snapshot, cards_by_uid=cards)

        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=3, target_groups=((4,),)))

        self.assertEqual(result.certainty, PredictionCertainty.PARTIAL)
        # bane is modeled by rule_kernel.attack_follower → not in unresolved
        self.assertNotIn("keywords:defender:uid=4:card_id=104:base_card_id=104:bane", result.unresolved_effects)
        # last_words remains unresolved (not modeled by kernel core rules)
        self.assertIn("last_words:uid=4:card_id=104:base_card_id=104", result.unresolved_effects)

    def test_turn_end_without_candidates_does_not_report_effect_unresolved(self):
        snapshot = self.make_snapshot()
        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.TURN_END, source_uid=0))

        self.assertEqual(result.certainty, PredictionCertainty.SPEC_ONLY)
        # turn_end 只执行回合结束核心规则，不追加 opponent turn_start
        self.assertTrue(result.snapshot.is_turn_ended)
        self.assertTrue(result.snapshot.is_ally_turn)
        self.assertEqual(result.assumptions, ("no_turn_end_effect_candidates",))
        self.assertEqual(result.unresolved_effects, ())

    def test_turn_end_reports_last_words_only_when_candidate_leaves(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[3] = replace(cards[3], has_last_words=True, is_remove_field_at_turn_change=True)
        snapshot = replace(snapshot, cards_by_uid=cards)

        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.TURN_END, source_uid=0))

        self.assertEqual(result.certainty, PredictionCertainty.PARTIAL)
        # turn_end 只执行回合结束核心规则，不追加 opponent turn_start
        self.assertTrue(result.snapshot.is_turn_ended)
        self.assertTrue(result.snapshot.is_ally_turn)
        self.assertIn("turn_end_effect_candidates_present", result.assumptions)
        self.assertNotIn("opponent_turn_start_simulated", result.assumptions)
        self.assertEqual(result.unresolved_effects, ("last_words:uid=3:card_id=103:base_card_id=103",))

    def test_turn_end_executes_template_effect_without_static_review_gate(self):
        snapshot = self.make_snapshot()
        template = _template(103, ExecutableEffect(effect_id="103_turn_end", trigger=TriggerSpec(event="turn_end")))

        result = LocalPredictor(templates_by_base_card_id={103: template}).predict(
            snapshot,
            LegalAction(kind=ActionKind.TURN_END, source_uid=0),
        )

        self.assertEqual(result.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertIn("event_queue_effects_enabled", result.assumptions)
        self.assertNotIn("draft_effect_blocked:103_turn_end", result.unresolved_effects)

    def test_turn_start_without_candidates_is_spec_only(self):
        snapshot = self.make_snapshot()

        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.TURN_START, source_uid=0))

        self.assertEqual(result.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(result.assumptions, ("no_turn_start_effect_candidates",))
        self.assertEqual(result.unresolved_effects, ())

    def test_turn_start_executes_template_effect_without_static_review_gate(self):
        snapshot = self.make_snapshot()
        template = _template(103, ExecutableEffect(effect_id="103_turn_start", trigger=TriggerSpec(event="turn_start")))

        result = LocalPredictor(templates_by_base_card_id={103: template}).predict(
            snapshot,
            LegalAction(kind=ActionKind.TURN_START, source_uid=0),
        )

        self.assertEqual(result.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertIn("event_queue_effects_enabled", result.assumptions)
        self.assertNotIn("draft_effect_blocked:103_turn_start", result.unresolved_effects)

    def test_extra_pp_returns_partial_without_fallback(self):
        snapshot = self.make_snapshot()
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=6, max_pp=6, is_first_side=False, extra_pp=ExtraPpView(extra_pp=1)),
            opponent=snapshot.opponent,
            cards_by_uid=snapshot.cards_by_uid,
            source="test",
            is_ally_turn=True,
        )

        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.EXTRA_PP, source_uid=0))

        self.assertEqual(result.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(result.snapshot.me.pp, 7)
        self.assertTrue(result.snapshot.me.extra_pp.is_used_extra_pp_this_turn)
        self.assertEqual(result.assumptions, ("extra_pp_activated",))
        self.assertEqual(result.unresolved_effects, ())

    def test_extra_pp_is_consumed_by_followup_play(self):
        cards = {1: CardView(uid=1, card_id=101, base_card_id=101, name="额外牌", cost=7, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER)}
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=6, max_pp=6, hand=(1,), is_first_side=False, extra_pp=ExtraPpView(extra_pp=1)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=6),
            cards_by_uid=cards,
            source="test",
            is_ally_turn=True,
        )
        predictor = LocalPredictor()

        activated = predictor.predict(snapshot, LegalAction(kind=ActionKind.EXTRA_PP, source_uid=0))
        played = predictor.predict(activated.snapshot, LegalAction(kind=ActionKind.PLAY, source_uid=1, cost=7))

        self.assertEqual(played.certainty, PredictionCertainty.PARTIAL)
        self.assertEqual(played.snapshot.me.pp, 0)
        self.assertEqual(played.snapshot.me.extra_pp.extra_pp, 0)
        self.assertEqual(played.snapshot.me.field, (1,))

    def test_activate_and_fusion_predictions_do_not_fallback(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[3] = replace(cards[3], has_activation=True)
        cards[5] = CardView(uid=5, card_id=105, base_card_id=105, name="融合源", cost=2)
        cards[6] = CardView(uid=6, card_id=106, base_card_id=106, name="素材", cost=1)
        snapshot = BattleSnapshot(
            me=replace(snapshot.me, hand=(1, 2, 5, 6), field=(3,)),
            opponent=snapshot.opponent,
            cards_by_uid=cards,
            source="test",
            is_ally_turn=True,
        )
        predictor = LocalPredictor()

        cards[3] = replace(cards[3], can_special_action=True)
        snapshot = BattleSnapshot(
            me=replace(snapshot.me, hand=(1, 2, 5, 6), field=(3,)),
            opponent=snapshot.opponent,
            cards_by_uid=cards,
            source="test",
            is_ally_turn=True,
        )

        activated = predictor.predict(snapshot, LegalAction(kind=ActionKind.ACTIVATE, source_uid=3))
        self.assertEqual(activated.certainty, PredictionCertainty.PARTIAL)
        # activate 后标记 activation_used_this_turn=True + has_activation=False
        self.assertTrue(activated.snapshot.cards_by_uid[3].activation_used_this_turn)
        self.assertFalse(activated.snapshot.cards_by_uid[3].has_activation)
        self.assertEqual(activated.unresolved_effects, ())

        special = predictor.predict(snapshot, LegalAction(kind=ActionKind.SPECIAL_ACTION, source_uid=3))
        self.assertEqual(special.certainty, PredictionCertainty.PARTIAL)
        # 登记顺序（`persistent_registration_order`）是「卡牌首次可见」的可观察历史：
        # 手工构造的 snapshot 没有这份历史，预测器会把它补全，这不动盘面。
        # 其余字段必须完全不变。
        def _without_registration_order(s):
            return replace(
                s,
                me=replace(s.me, persistent_registration_order=()),
                opponent=replace(s.opponent, persistent_registration_order=()),
            )

        self.assertEqual(_without_registration_order(special.snapshot), _without_registration_order(snapshot))
        self.assertEqual(special.unresolved_effects, ())

        fused = predictor.predict(snapshot, LegalAction(kind=ActionKind.FUSION, source_uid=5, material_uids=(6,)))
        # 无即时 fusion trigger 时，融合仅记录素材状态，CoreRuleKernel 可严格预测。
        self.assertEqual(fused.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(fused.snapshot.me.hand, (1, 2, 5))
        self.assertNotIn(6, fused.snapshot.cards_by_uid)
        self.assertEqual(fused.snapshot.cards_by_uid[5].supplement_info["fusion_count"], 1)
        self.assertEqual(fused.unresolved_effects, ())

    def test_play_cost_modes_predict_zone_and_cost_slices(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[5] = CardView(uid=5, card_id=105, base_card_id=105, name="费用形态", cost=7, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER)
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=7, max_pp=7, hand=(5,), field=()),
            opponent=snapshot.opponent,
            cards_by_uid=cards,
            source="test",
            is_ally_turn=True,
        )
        predictor = LocalPredictor()

        enhanced = predictor.predict(snapshot, LegalAction(kind=ActionKind.PLAY, source_uid=5, play_kind=PlayKind.ENHANCE, cost=6))
        accelerated = predictor.predict(snapshot, LegalAction(kind=ActionKind.PLAY, source_uid=5, play_kind=PlayKind.ACCELERATE, cost=1))
        crystallized = predictor.predict(snapshot, LegalAction(kind=ActionKind.PLAY, source_uid=5, play_kind=PlayKind.CRYSTAL, cost=2))

        self.assertEqual(enhanced.certainty, PredictionCertainty.PARTIAL)
        self.assertEqual(enhanced.snapshot.me.pp, 1)
        self.assertEqual(enhanced.snapshot.me.field, (5,))
        self.assertIn("enhance_effects", enhanced.unresolved_effects)
        self.assertEqual(accelerated.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(accelerated.snapshot.me.pp, 6)
        self.assertEqual(accelerated.snapshot.me.graveyard_count, 1)
        self.assertNotIn(5, accelerated.snapshot.cards_by_uid)
        self.assertIn("accelerate_effects", accelerated.unresolved_effects)
        self.assertEqual(crystallized.certainty, PredictionCertainty.SPEC_ONLY)
        self.assertEqual(crystallized.snapshot.me.pp, 5)
        self.assertEqual(crystallized.snapshot.me.field, (5,))
        self.assertEqual(crystallized.snapshot.cards_by_uid[5].card_type, CardType.COUNTDOWN_AMULET)
        self.assertIn("crystal_effects", crystallized.unresolved_effects)

    def test_invalid_action_returns_predict_error_without_mutating_root(self):
        snapshot = self.make_snapshot()
        result = LocalPredictor().predict(snapshot, LegalAction(kind=ActionKind.ATTACK_FOLLOWER, source_uid=3))

        self.assertEqual(result.certainty, PredictionCertainty.PREDICT_ERROR)
        self.assertEqual(result.snapshot, snapshot)
        self.assertEqual(snapshot.opponent.field, (4,))


def _template(base_card_id: int, *effects: ExecutableEffect) -> UnifiedCardTemplate:
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
        printed=PrintedStats(cost=1, atk=1, life=1),
        executable_effects=effects,
    )


if __name__ == "__main__":
    unittest.main()
