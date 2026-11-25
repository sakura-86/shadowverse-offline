# -*- coding: utf-8 -*-
"""LocalActionEnumerator 测试。"""

import unittest

from decision_engine.actions.local_enumerator import LocalActionEnumerator
from decision_engine.actions.model import ActionKind, PlayKind
from decision_engine.cards.effect_specs import EffectSpec, EffectSpecOperation, EffectSpecTarget, EffectSpecTrigger
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, ChoiceSkillInfo, ExtraPpView, LegalView, ModeSkillInfo, PlayerView, SelectFusionOption, SelectSkillOption


def _snapshot(*, me=None, opponent=None, cards=None, legal=None, is_ally_turn=True, is_turn_ended=False):
    return BattleSnapshot(
        me=me or PlayerView(life=20, max_life=20, pp=2, max_pp=2),
        opponent=opponent or PlayerView(life=20, max_life=20, pp=0, max_pp=2, unique_id=99),
        cards_by_uid=cards or {},
        legal=legal,
        is_ally_turn=is_ally_turn,
        is_turn_ended=is_turn_ended,
    )


class LocalActionEnumeratorTest(unittest.TestCase):
    def test_returns_empty_when_not_current_turn_or_ended(self):
        enumerator = LocalActionEnumerator()

        self.assertEqual(enumerator.enumerate(_snapshot(is_ally_turn=False)), [])
        self.assertEqual(enumerator.enumerate(_snapshot(is_turn_ended=True)), [])

    def test_enumerates_core_play_actions_without_protocol_legal_view(self):
        cards = {
            1: CardView(uid=1, cost=2, card_type=CardType.FOLLOWER),
            2: CardView(uid=2, cost=3, card_type=CardType.SPELL),
            3: CardView(uid=3, cost=1, card_type=CardType.SPELL),
        }
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=2, max_pp=2, hand=(1, 2, 3)), cards=cards)

        actions = LocalActionEnumerator().enumerate(snapshot)
        plays = [action for action in actions if action.kind is ActionKind.PLAY]

        self.assertEqual([(action.source_uid, action.play_kind, action.cost) for action in plays], [(1, PlayKind.NORMAL, 2), (3, PlayKind.NORMAL, 1)])
        self.assertTrue(all(action.legality_source == "local" for action in plays))
        self.assertTrue(all(action.confidence == "core_rules" for action in plays))

    def test_protocol_can_play_filters_local_core_play_actions(self):
        """协议 can_play 不再过滤本地枚举结果——本地推导优先。

        diff 检查会记录 warning 但不阻止枚举。两张卡本地均满足出牌条件
        （PP 够、非法术场满等），都应被枚举。
        """
        cards = {
            1: CardView(uid=1, cost=1, card_type=CardType.SPELL),
            2: CardView(uid=2, cost=1, card_type=CardType.SPELL),
        }
        legal = LegalView(can_play=frozenset({1}))
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=2, max_pp=2, hand=(1, 2)), cards=cards, legal=legal)

        plays = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.PLAY]

        # 本地推导：两张卡都可以出（cost=1, pp=2）
        self.assertEqual([(action.source_uid, action.play_kind) for action in plays], [(1, PlayKind.NORMAL), (2, PlayKind.NORMAL)])

    def test_enumerates_play_mode_costs_from_card_view(self):
        cards = {
            1: CardView(uid=1, cost=2, card_type=CardType.FOLLOWER, can_enhance=True, enhance_costs=(4, 6), accelerate_costs=(1,), crystal_costs=(3,)),
            2: CardView(uid=2, cost=5, card_type=CardType.SPELL, accelerate_costs=(2,)),
        }
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=4, max_pp=4, hand=(1, 2)), cards=cards)

        plays = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.PLAY]

        # enhance 优先规则：PP=4 >= enhance_cost=4 → 强制 enhance 打出（最高可负担的 enhance）
        # card2 cost=5 > PP=4 → 走 accelerate
        self.assertEqual(
            [(action.source_uid, action.play_kind, action.cost) for action in plays],
            [
                (1, PlayKind.ENHANCE, 4),
                (2, PlayKind.ACCELERATE, 2),
            ],
        )

    def test_enumerates_mode_and_choice_parameters_from_card_view(self):
        cards = {
            1: CardView(uid=1, cost=3, card_type=CardType.FOLLOWER, mode_skill_map={"play": ModeSkillInfo(mode_group=1, mode_count=3, select_count=1)}),
            2: CardView(uid=2, cost=3, card_type=CardType.FOLLOWER, choice_skill_map={"play": ChoiceSkillInfo(choice_count=1, choice_id_list=(10, 20, 30))}),
        }
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=3, max_pp=3, hand=(1, 2)), cards=cards)

        plays = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.PLAY]

        self.assertEqual(sorted(action.selected_indexes for action in plays if action.source_uid == 1), [(0,), (1,), (2,)])
        self.assertEqual(sorted(action.selected_indexes for action in plays if action.source_uid == 2), [(0,), (1,), (2,)])

    def test_effect_spec_required_target_blocks_play_when_candidates_missing(self):
        cards = {
            1: CardView(uid=1, base_card_id=10131320, cost=1, card_type=CardType.SPELL),
            2: CardView(uid=2, card_type=CardType.FOLLOWER),
        }
        empty = _snapshot(me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, hand=(1,)), opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=1, field=()), cards=cards)
        targeted = _snapshot(me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, hand=(1,)), opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=1, field=(2,)), cards=cards)

        empty_plays = [action for action in LocalActionEnumerator().enumerate(empty) if action.kind is ActionKind.PLAY]
        targeted_plays = [action for action in LocalActionEnumerator().enumerate(targeted) if action.kind is ActionKind.PLAY]

        self.assertEqual(empty_plays, [])
        self.assertEqual([(action.source_uid, action.target_groups) for action in targeted_plays], [(1, ((2,),))])

    def test_effect_spec_all_targets_do_not_become_selected_action_params(self):
        cards = {
            1: CardView(uid=1, base_card_id=10163110, cost=6, card_type=CardType.FOLLOWER),
            2: CardView(uid=2, card_type=CardType.FOLLOWER),
            3: CardView(uid=3, card_type=CardType.FOLLOWER),
        }
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=6, max_pp=6, hand=(1,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=1, field=(2, 3)),
            cards=cards,
        )

        plays = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.PLAY and action.source_uid == 1]

        self.assertEqual([(action.source_uid, action.target_groups) for action in plays], [(1, ())])

    def test_effect_spec_evolve_targets_are_enumerated_from_operation_params(self):
        cards = {
            1: CardView(uid=1, base_card_id=10304120, card_type=CardType.FOLLOWER, can_super_evolve=True),
            2: CardView(uid=2, card_type=CardType.FOLLOWER),
            3: CardView(uid=3, card_type=CardType.FOLLOWER),
        }
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=10, sep=1, field=(1, 2)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=10, field=(3,)),
            cards=cards,
        )

        super_evolves = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.SUPER_EVOLVE and action.source_uid == 1]

        self.assertEqual(sorted(action.target_groups for action in super_evolves), [((1,),), ((2,),), ((3,),)])

    def test_effect_spec_mode_options_are_enumerated_from_operation_params(self):
        cards = {
            1: CardView(uid=1, base_card_id=10564110, cost=4, card_type=CardType.FOLLOWER),
            2: CardView(uid=2, card_type=CardType.FOLLOWER, has_guard=True),
        }
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=4, max_pp=4, hand=(1,), field=(2,)), cards=cards)

        plays = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.PLAY and action.source_uid == 1]

        self.assertEqual(sorted(action.selected_indexes for action in plays), [(0,), (1,)])
        self.assertNotIn((), [action.selected_indexes for action in plays])

    def test_effect_spec_mode_options_use_legal_select_count(self):
        cards = {1: CardView(uid=1, base_card_id=10354120, cost=7, card_type=CardType.FOLLOWER)}
        legal = LegalView(can_play=frozenset({1}), mode_skill_map={1: ModeSkillInfo(mode_group=1, mode_count=2, select_count=2)})
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=7, max_pp=10, hand=(1,)), cards=cards, legal=legal)

        plays = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.PLAY and action.source_uid == 1]

        self.assertEqual([action.selected_indexes for action in plays], [(1, 2)])

    def test_effect_spec_mode_option_enumerated_even_without_random_candidate(self):
        # 模式选择不受该能力目标候选是否存在的影响：无守护随从时，
        # 「使随机守护进化前其他随从进化」这一能力仍可被选择入场(效果空转)。
        cards = {
            1: CardView(uid=1, base_card_id=10564110, cost=4, card_type=CardType.FOLLOWER),
            2: CardView(uid=2, card_type=CardType.FOLLOWER, has_guard=False),
        }
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=4, max_pp=4, hand=(1,), field=(2,)), cards=cards)

        plays = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.PLAY and action.source_uid == 1]

        self.assertEqual(sorted(action.selected_indexes for action in plays), [(0,), (1,)])

    def test_does_not_play_non_spell_when_field_full(self):
        cards = {uid: CardView(uid=uid, cost=1, card_type=CardType.FOLLOWER, crystal_costs=(1,)) for uid in range(1, 7)}
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, hand=(1,), field=(2, 3, 4, 5, 6)), cards=cards)

        actions = LocalActionEnumerator().enumerate(snapshot)

        self.assertNotIn(ActionKind.PLAY, {action.kind for action in actions})

    def test_activation_requires_snapshot_state_and_pp(self):
        ready = CardView(uid=1, base_card_id=10161210, card_id=10161210, cost=1, card_type=CardType.AMULET, has_activation=True)
        used = CardView(uid=2, base_card_id=10161210, card_id=10161210, cost=1, card_type=CardType.AMULET, has_activation=True, activation_used_this_turn=True)
        missing_state = CardView(uid=3, base_card_id=10161210, card_id=10161210, cost=1, card_type=CardType.AMULET, has_activation=False)

        actions = LocalActionEnumerator().enumerate(_snapshot(me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(1, 2, 3)), cards={1: ready, 2: used, 3: missing_state}))
        no_pp_actions = LocalActionEnumerator().enumerate(_snapshot(me=PlayerView(life=20, max_life=20, pp=0, max_pp=1, field=(1,)), cards={1: ready}))

        self.assertEqual([(action.kind, action.source_uid) for action in actions if action.kind is ActionKind.ACTIVATE], [(ActionKind.ACTIVATE, 1)])
        self.assertEqual([action for action in no_pp_actions if action.kind is ActionKind.ACTIVATE], [])

    def test_enumerates_attacks_with_guard_blocking_leader(self):
        cards = {
            1: CardView(uid=1, atk=3, life=3, attacks_left=1, can_attack=True, can_attack_leader=True),
            2: CardView(uid=2, atk=1, life=2, has_guard=True),
            3: CardView(uid=3, atk=4, life=4),
        }
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, field=(1,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, unique_id=99, field=(2, 3)),
            cards=cards,
        )

        attacks = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind in {ActionKind.ATTACK_LEADER, ActionKind.ATTACK_FOLLOWER}]

        self.assertEqual([(action.kind, action.source_uid, action.target_groups) for action in attacks], [(ActionKind.ATTACK_FOLLOWER, 1, ((2,),))])

    def test_attacks_only_target_followers_and_respect_guard(self):
        cards = {
            1: CardView(uid=1, atk=7, life=7, attacks_left=1, can_attack=True, can_attack_leader=True),
            2: CardView(uid=2, atk=7, life=3, has_guard=True, card_type=CardType.FOLLOWER),
            3: CardView(uid=3, atk=4, life=6, card_type=CardType.FOLLOWER),
            4: CardView(uid=4, card_type=CardType.AMULET),
        }
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, field=(1,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, unique_id=99, field=(2, 3, 4)),
            cards=cards,
        )

        attacks = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.ATTACK_FOLLOWER]

        self.assertEqual([(action.source_uid, action.target_groups) for action in attacks], [(1, ((2,),))])

    def test_unattackable_guard_still_blocks_non_guard_targets(self):
        cards = {
            1: CardView(uid=1, atk=7, life=7, attacks_left=1, can_attack=True, can_attack_leader=True),
            2: CardView(uid=2, atk=7, life=3, has_guard=True, has_cant_be_attacked=True, card_type=CardType.FOLLOWER),
            3: CardView(uid=3, atk=4, life=6, card_type=CardType.FOLLOWER),
        }
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, field=(1,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, unique_id=99, field=(2, 3)),
            cards=cards,
        )

        attacks = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind in {ActionKind.ATTACK_LEADER, ActionKind.ATTACK_FOLLOWER}]

        self.assertEqual(attacks, [])

    def test_ignore_guard_allows_attacking_leader_through_guard(self):
        cards = {
            1: CardView(uid=1, atk=3, life=3, attacks_left=1, can_attack=True, can_attack_leader=True, can_ignore_guard=True),
            2: CardView(uid=2, atk=1, life=2, has_guard=True),
        }
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, field=(1,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, unique_id=99, field=(2,)),
            cards=cards,
        )

        attacks = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind in {ActionKind.ATTACK_LEADER, ActionKind.ATTACK_FOLLOWER}]

        self.assertEqual(
            [(action.kind, action.source_uid, action.target_groups) for action in attacks],
            [(ActionKind.ATTACK_FOLLOWER, 1, ((2,),)), (ActionKind.ATTACK_LEADER, 1, ((99,),))],
        )

    def test_attack_filters_unattackable_targets_and_card_attack_targets(self):
        cards = {
            1: CardView(uid=1, atk=3, life=3, attacks_left=1, can_attack=True, can_attack_leader=True, attack_targets=(3, 4)),
            2: CardView(uid=2, atk=1, life=2, has_ambush=True),
            3: CardView(uid=3, atk=4, life=4),
            4: CardView(uid=4, atk=4, life=4, has_cant_be_attacked=True),
        }
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, field=(1,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, unique_id=99, field=(2, 3, 4)),
            cards=cards,
        )

        attacks = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind in {ActionKind.ATTACK_LEADER, ActionKind.ATTACK_FOLLOWER}]

        self.assertEqual([(action.kind, action.target_groups) for action in attacks], [(ActionKind.ATTACK_FOLLOWER, ((3,),)), (ActionKind.ATTACK_LEADER, ((99,),))])

    def test_enumerates_evolve_super_evolve_extra_pp_and_end_turn(self):
        cards = {
            1: CardView(uid=1, cost=3, card_type=CardType.FOLLOWER),
            2: CardView(uid=2, can_evolve=True, can_super_evolve=True),
        }
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=2, max_pp=2, ep=1, sep=1, hand=(1,), field=(2,), extra_pp=ExtraPpView(extra_pp=1)),
            cards=cards,
        )

        actions = LocalActionEnumerator().enumerate(snapshot)
        keys = {(action.kind, action.source_uid) for action in actions}

        self.assertIn((ActionKind.EVOLVE, 2), keys)
        self.assertIn((ActionKind.SUPER_EVOLVE, 2), keys)
        self.assertIn((ActionKind.EXTRA_PP, 0), keys)
        self.assertIn((ActionKind.TURN_END, 0), keys)
        self.assertNotIn(ActionKind.TURN_START, {action.kind for action in actions})

    def test_enumerates_evolve_from_protocol_legal_view(self):
        cards = {
            1: CardView(uid=1, cost=3, card_type=CardType.FOLLOWER),
            2: CardView(uid=2, can_evolve=True, can_super_evolve=True, evolve_state=0),
        }
        legal = LegalView(can_evolve=frozenset({2}), can_super_evolve=frozenset({2}))
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=2, max_pp=2, ep=1, sep=1, hand=(1,), field=(2,)),
            cards=cards,
            legal=legal,
        )

        actions = LocalActionEnumerator().enumerate(snapshot)
        keys = {(action.kind, action.source_uid) for action in actions}

        self.assertIn((ActionKind.EVOLVE, 2), keys)
        self.assertIn((ActionKind.SUPER_EVOLVE, 2), keys)

    def test_enumerates_activation_and_fusion_from_card_view(self):
        cards = {
            1: CardView(uid=1, has_activation=True),
            2: CardView(uid=2, card_type=CardType.SPELL, fusion_list=((3, 4), (5,))),
            3: CardView(uid=3),
            4: CardView(uid=4),
            5: CardView(uid=5),
        }
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, hand=(2, 3, 4, 5), field=(1,)), cards=cards)

        actions = LocalActionEnumerator().enumerate(snapshot)
        activations = [action for action in actions if action.kind is ActionKind.ACTIVATE]
        fusions = [action for action in actions if action.kind is ActionKind.FUSION]

        self.assertEqual([(action.source_uid, action.target_groups) for action in activations], [(1, ())])
        self.assertEqual([(action.source_uid, action.material_uids) for action in fusions], [(2, (3, 4)), (2, (5,))])

    def test_does_not_infer_activation_availability_from_effect_spec_trigger(self):
        cards = {
            1: CardView(uid=1, base_card_id=10763210, card_type=CardType.AMULET),
            2: CardView(uid=2, card_type=CardType.FOLLOWER),
        }
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, field=(1,)), opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, field=(2,)), cards=cards)

        activations = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.ACTIVATE]

        self.assertEqual(activations, [])

    def test_enumerates_template_fusion_for_any_other_hand_cards(self):
        cards = {
            1: CardView(uid=1, base_card_id=10934110),
            2: CardView(uid=2, base_card_id=10001110),
            3: CardView(uid=3, base_card_id=10002110),
        }
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=2, max_pp=2, hand=(1, 2, 3)), cards=cards)

        actions = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.FUSION]

        self.assertEqual([(action.source_uid, action.material_uids) for action in actions], [(1, (2,)), (1, (3,)), (1, (2, 3))])

    def test_does_not_enumerate_template_fusion_with_unaffordable_unconditional_pp_cost(self):
        cards = {
            1: CardView(uid=1, base_card_id=10934110),
            2: CardView(uid=2, base_card_id=10001110),
        }
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=1, max_pp=2, hand=(1, 2)), cards=cards)

        actions = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.FUSION]

        # 10934110 的融合无条件先支付 2 PP；1 PP 时不是合法动作。
        self.assertEqual(actions, [])

    def test_does_not_enumerate_fusion_used_this_turn(self):
        cards = {
            1: CardView(uid=1, base_card_id=10934110, supplement_info={"fusion_used_this_turn": 1}),
            2: CardView(uid=2, supplement_info={"fusion_used_this_turn": 1}),
            3: CardView(uid=3),
            4: CardView(uid=4),
        }
        legal = LegalView(
            can_fusion=frozenset({2, 3}),
            select_fusion_map={
                2: SelectFusionOption(select_material_count=1, material_uids=(4,)),
                3: SelectFusionOption(select_material_count=1, material_uids=(4,)),
            },
        )
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, hand=(1, 2, 3, 4)),
            cards=cards,
            legal=legal,
        )

        fusions = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.FUSION]

        self.assertEqual([(action.source_uid, action.material_uids) for action in fusions], [(3, (4,))])

    def test_enumerates_fusion_from_protocol_legal_view(self):
        cards = {
            1: CardView(uid=1, card_type=CardType.SPELL),
            2: CardView(uid=2),
            3: CardView(uid=3),
        }
        legal = LegalView(can_fusion=frozenset({1}), select_fusion_map={1: SelectFusionOption(select_material_count=2, material_uids=(2, 3))})
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=0, max_pp=2, hand=(1, 2, 3)), cards=cards, legal=legal)

        actions = [action for action in LocalActionEnumerator().enumerate(snapshot) if action.kind is ActionKind.FUSION]

        self.assertEqual([(action.source_uid, action.material_uids) for action in actions], [(1, (2, 3))])

    def test_enumerates_local_action_parameters_from_card_view(self):
        cards = {
            1: CardView(uid=1, cost=1, card_type=CardType.SPELL, select_skill_map={"play": (SelectSkillOption(select_count=1, target_uids=(7, 8)),)}),
            2: CardView(uid=2, can_evolve=True, mode_skill_map={"evolve": ModeSkillInfo(mode_count=3, select_count=2)}),
            3: CardView(uid=3, has_activation=True, choice_skill_map={"activation": ChoiceSkillInfo(choice_count=1, choice_id_list=(100, 101))}),
            4: CardView(uid=4, can_special_action=True, select_skill_map={"special_action": (SelectSkillOption(target_uids=(9,), selected_indexes=(2,)),)}),
            5: CardView(uid=5, select_fusion=SelectFusionOption(select_material_count=2, material_uids=(6, 7, 8))),
            6: CardView(uid=6),
            7: CardView(uid=7),
            8: CardView(uid=8),
        }
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=2, max_pp=2, ep=1, hand=(1, 5, 6, 7, 8), field=(2, 3, 4)), cards=cards)

        actions = LocalActionEnumerator().enumerate(snapshot)
        play_params = [(action.target_groups, action.selected_indexes) for action in actions if action.kind is ActionKind.PLAY and action.source_uid == 1]
        evolve_params = [action.selected_indexes for action in actions if action.kind is ActionKind.EVOLVE and action.source_uid == 2]
        activation_params = [action.selected_indexes for action in actions if action.kind is ActionKind.ACTIVATE and action.source_uid == 3]
        special_params = [(action.target_groups, action.selected_indexes) for action in actions if action.kind is ActionKind.SPECIAL_ACTION]
        fusion_params = [action.material_uids for action in actions if action.kind is ActionKind.FUSION]

        self.assertEqual(play_params, [(((7,),), ()), (((8,),), ())])
        self.assertEqual(evolve_params, [(0, 1), (0, 2), (1, 2)])
        self.assertEqual(activation_params, [(0,), (1,)])
        self.assertEqual(special_params, [(((9,),), (2,))])
        self.assertEqual(fusion_params, [(6, 7), (6, 8), (7, 8)])

    def test_extra_pp_checks_play_mode_costs(self):
        cards = {1: CardView(uid=1, cost=5, card_type=CardType.SPELL, accelerate_costs=(3,))}
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=2, max_pp=2, hand=(1,), extra_pp=ExtraPpView(extra_pp=1)), cards=cards)

        actions = LocalActionEnumerator().enumerate(snapshot)

        self.assertIn(ActionKind.EXTRA_PP, {action.kind for action in actions})

    def test_extra_pp_is_not_enumerated_when_already_used(self):
        cards = {1: CardView(uid=1, cost=3, card_type=CardType.FOLLOWER)}
        snapshot = _snapshot(me=PlayerView(life=20, max_life=20, pp=2, max_pp=2, hand=(1,), extra_pp=ExtraPpView(extra_pp=1, is_used_extra_pp_this_turn=True)), cards=cards)

        actions = LocalActionEnumerator().enumerate(snapshot)

        self.assertNotIn(ActionKind.EXTRA_PP, {action.kind for action in actions})

    def test_spec_target_count_none_is_internal_reference_not_selection(self):
        """spec 顶层 target 的 count=None 表示内部引用，不产生用户选择。

        回归：``11024120_auto_003`` / ``11064110_auto_004`` 的 target 是
        ``selected_target + card_type=crest + count=None``（超进化时给自己的纹章
        加倒计数），目标由 ``operation.params.crest_card_ids`` 自行解析。
        曾按默认 ``count=1`` 处理，把兜底池 ``opponent.field`` 的 5 个随从
        枚举成 5 个「选择对方随从」的非法动作。
        """
        cards = {uid: CardView(uid=uid, card_type=CardType.FOLLOWER, atk=1, life=1) for uid in range(1, 6)}
        cards[10] = CardView(uid=10, card_type=CardType.FOLLOWER, atk=2, life=2)
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=5, max_pp=5, field=(10,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, unique_id=99, field=(1, 2, 3, 4, 5)),
            cards=cards,
        )
        internal_reference = EffectSpecTarget(kind="selected_target", side="self", zone="field", card_type="crest", count=None, scope=None)

        expanded = LocalActionEnumerator()._expand_effect_target(snapshot, cards[10], internal_reference)

        self.assertEqual(expanded, [((), ())])

    def test_synthesized_implicit_selector_declares_count_one(self):
        """operations 合成的 implicit 选择必须显式 ``count=1``，与内部引用区分开。

        否则 ``_expand_effect_target`` 无法区分「spec 顶层 count=None 的内部引用」
        与「evolve/super_evolve 从 operation.target_ref 合成的真实选择」。
        """
        spec = EffectSpec(
            spec_id="t_implicit",
            base_card_id=1,
            name="t",
            trigger=EffectSpecTrigger(event="manual_evolve"),
            operations=(EffectSpecOperation(op="destroy", target_ref="selected_enemy_follower"),),
        )

        synthesized = LocalActionEnumerator()._synthesize_target_from_operations(spec, None)

        self.assertIsNotNone(synthesized)
        self.assertEqual(synthesized.kind, "selected_enemy_follower")
        self.assertEqual(synthesized.count, 1)

    def test_target_card_type_participates_in_candidate_filtering(self):
        """``target.card_type`` 必须参与候选过滤，不能被静默忽略。

        回归：``_target_pool`` 原来只处理 ``"follower"``，``"crest"`` 等取值被忽略，
        把不匹配的卡放进了候选池。候选过滤与物化器共用 ``CardType.matches``。
        """
        cards = {uid: CardView(uid=uid, card_type=CardType.FOLLOWER, atk=1, life=1) for uid in range(1, 6)}
        cards[10] = CardView(uid=10, card_type=CardType.FOLLOWER, atk=2, life=2)
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=5, max_pp=5, field=(10,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2, unique_id=99, field=(1, 2, 3, 4, 5)),
            cards=cards,
        )
        enumerator = LocalActionEnumerator()

        def expand(card_type):
            target = EffectSpecTarget(kind="selected_enemy_follower", side="enemy", zone="field", card_type=card_type, count=1, scope="selected")
            return enumerator._expand_effect_target(snapshot, cards[10], target)

        # crest 与场上的随从不匹配：候选为空，不产生任何目标
        self.assertEqual(expand("crest"), [])
        # follower 匹配：5 个随从各一个动作
        self.assertEqual(len(expand("follower")), 5)
        # card / any 表示不过滤
        self.assertEqual(len(expand("card")), 5)
        self.assertEqual(len(expand("any")), 5)
        # None 表示不过滤
        self.assertEqual(len(expand(None)), 5)


    def test_multi_slot_action_carries_spec_ids_per_group(self):
        """多目标动作的 ``target_group_specs`` 与 ``target_groups`` 等长且按 spec 对齐。

        ``10642310``（赤流）的 fanfare 有两个 ``selected_*`` spec：
        ``_auto_001`` = 选择自己的 1 张手牌，``_auto_002`` = 选择对方的 1 个随从。
        打出动作必须是 ``((手牌 uid,), (随从 uid,))``，specs 为
        ``("10642310_auto_001", "10642310_auto_002")``——下游不能靠位置推断语义。
        """
        cards = {
            1: CardView(uid=1, card_type=CardType.FOLLOWER, atk=1, life=1),
            2: CardView(uid=2, card_type=CardType.FOLLOWER, atk=1, life=1),
            10: CardView(uid=10, base_card_id=10642310, cost=3, card_type=CardType.FOLLOWER, atk=2, life=2),
            11: CardView(uid=11, cost=1, card_type=CardType.SPELL),
            12: CardView(uid=12, cost=1, card_type=CardType.SPELL),
        }
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=20, max_pp=20, hand=(10, 11, 12)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=20, unique_id=99, field=(1, 2)),
            cards=cards,
        )

        actions = [a for a in LocalActionEnumerator().enumerate(snapshot) if a.kind is ActionKind.PLAY and a.source_uid == 10]

        self.assertTrue(actions)
        for action in actions:
            self.assertEqual(len(action.target_group_specs), len(action.target_groups))
            self.assertEqual(action.target_group_specs, ("10642310_auto_001", "10642310_auto_002"))
        self.assertEqual(
            sorted(action.target_groups for action in actions),
            sorted([((11,), (1,)), ((11,), (2,)), ((12,), (1,)), ((12,), (2,))]),
        )

    def test_single_slot_action_carries_its_spec_id(self):
        """单目标动作也要带 spec_id，pointer 用它区分「打随从」与「弃手牌」。"""
        cards = {uid: CardView(uid=uid, card_type=CardType.FOLLOWER, atk=1, life=1) for uid in range(1, 6)}
        cards[10] = CardView(uid=10, base_card_id=11064110, cost=3, card_type=CardType.FOLLOWER, atk=2, life=2)
        snapshot = _snapshot(
            me=PlayerView(life=20, max_life=20, pp=10, max_pp=10, hand=(10,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=10, unique_id=99, field=(1, 2, 3, 4, 5)),
            cards=cards,
        )

        actions = [a for a in LocalActionEnumerator().enumerate(snapshot) if a.kind is ActionKind.PLAY and a.source_uid == 10]

        self.assertEqual([a.target_group_specs for a in actions], [("11064110_auto_001",)] * 5)


if __name__ == "__main__":
    unittest.main()
