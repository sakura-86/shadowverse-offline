# -*- coding: utf-8 -*-
"""ProtocolActionEnumerator 测试。"""

import unittest

from decision_engine.actions.model import ActionKind, PlayKind
from decision_engine.actions.protocol_enumerator import ProtocolActionEnumerator
from decision_engine.state.snapshot import (
    BattleSnapshot,
    CardType,
    CardView,
    ChoiceSkillInfo,
    LegalView,
    ModeSkillInfo,
    PlayerView,
    SelectFusionOption,
    SelectSkillOption,
)


def _snapshot(
    *,
    cards_by_uid=None,
    legal=None,
    opponent_unique_id=0,
    is_ally_turn=True,
    is_turn_ended=False,
    hand=None,
    field=None,
    opponent_field=None,
):
    if legal is not None:
        if hand is None:
            hand = tuple(sorted(
                legal.can_play
                | legal.can_enhance
                | legal.can_accelerate
                | legal.can_crystal
                | legal.can_fusion
                | legal.can_play_with_ex_pp
            ))
        if field is None:
            field = tuple(sorted(
                legal.can_attack_leader
                | legal.can_attack_field
                | legal.can_evolve
                | legal.can_super_evolve
                | legal.can_activate
                | legal.can_activate_with_ex_pp
                | legal.can_special_action
            ))
        # 自动推断 opponent.field：attack_targets 中除 opponent_unique_id 以外的 target uids
        if opponent_field is None and legal.attack_targets:
            all_targets: set[int] = set()
            for targets in legal.attack_targets.values():
                all_targets.update(targets)
            all_targets.discard(opponent_unique_id)
            opponent_field = tuple(sorted(all_targets))
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=10, max_pp=10, hand=hand or (), field=field or ()),
        opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0, unique_id=opponent_unique_id, field=opponent_field or ()),
        cards_by_uid=cards_by_uid or {},
        legal=legal,
        is_ally_turn=is_ally_turn,
        is_turn_ended=is_turn_ended,
    )


class ProtocolActionEnumeratorTest(unittest.TestCase):
    def test_returns_empty_without_legal_view(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(legal=None))

        self.assertEqual(actions, [])

    def test_returns_empty_when_not_ally_turn(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(can_play=frozenset({1})),
            is_ally_turn=False,
        ))

        self.assertEqual(actions, [])

    def test_returns_empty_when_turn_ended(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(can_play=frozenset({1})),
            is_turn_ended=True,
        ))

        self.assertEqual(actions, [])

    def test_skips_attack_leader_without_protocol_player_target(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(can_attack_leader=frozenset({11})),
        ))

        self.assertEqual(actions, [])

    def test_enumerates_attack_leader_with_protocol_player_target(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            opponent_unique_id=2,
            field=(20,),
            legal=LegalView(
                can_attack_leader=frozenset({20}),
                attack_targets={20: frozenset({2, 63})},
            ),
        ))

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, ActionKind.ATTACK_LEADER)
        self.assertEqual(actions[0].source_uid, 20)
        self.assertEqual(actions[0].target_groups, ((2,),))

    def test_does_not_enumerate_attack_leader_from_can_attack_leader_without_player_target(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            opponent_unique_id=2,
            field=(21,),
            legal=LegalView(
                can_attack_leader=frozenset({21}),
                can_attack_field=frozenset({21}),
                attack_targets={21: frozenset({51})},
            ),
        ))

        self.assertEqual([(action.kind, action.source_uid, action.target_groups) for action in actions], [
            (ActionKind.ATTACK_FOLLOWER, 21, ((51,),)),
        ])

    def test_does_not_enumerate_attack_leader_when_source_missing_from_target_map(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            opponent_unique_id=2,
            legal=LegalView(
                can_attack_leader=frozenset({21}),
                attack_targets={22: frozenset({2})},
            ),
        ))

        self.assertEqual(actions, [])

    def test_enumerates_attack_targets(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            field=(11,),
            legal=LegalView(
                can_attack_field=frozenset({11}),
                attack_targets={11: frozenset({101, 102})},
            ),
        ))

        follower_attacks = [action for action in actions if action.kind == ActionKind.ATTACK_FOLLOWER]
        self.assertEqual(
            [(action.source_uid, action.target_groups) for action in follower_attacks],
            [(11, ((101,),)), (11, ((102,),))],
        )
        self.assertTrue(all(action.legality_source == "protocol" for action in follower_attacks))
        self.assertTrue(all(action.confidence == "exact" for action in follower_attacks))

    def test_filters_actions_whose_source_is_not_in_current_zone(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            opponent_unique_id=2,
            hand=(),
            field=(),
            legal=LegalView(
                can_play=frozenset({10}),
                can_fusion=frozenset({11}),
                can_attack_leader=frozenset({20}),
                can_attack_field=frozenset({21}),
                can_evolve=frozenset({33}),
                can_super_evolve=frozenset({34}),
                can_super_evolve_with_skill=frozenset({35}),
                can_activate=frozenset({36}),
                can_special_action=frozenset({37}),
                attack_targets={20: frozenset({2}), 21: frozenset({51})},
            ),
        ))

        self.assertEqual(actions, [])

    def test_extra_pp_ignores_ex_pp_sources_not_in_current_zone(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            hand=(),
            field=(),
            legal=LegalView(
                can_play_with_ex_pp=frozenset({10}),
                can_activate_with_ex_pp=frozenset({20}),
            ),
        ))

        self.assertEqual(actions, [])

    def test_keeps_actions_whose_source_is_in_current_zone(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            opponent_unique_id=2,
            hand=(10, 11),
            field=(20, 21, 33, 34, 35, 36, 37),
            legal=LegalView(
                can_play=frozenset({10}),
                can_fusion=frozenset({11}),
                can_attack_leader=frozenset({20}),
                can_attack_field=frozenset({21}),
                can_evolve=frozenset({33}),
                can_super_evolve=frozenset({34}),
                can_super_evolve_with_skill=frozenset({35}),
                can_activate=frozenset({36}),
                can_special_action=frozenset({37}),
                attack_targets={20: frozenset({2}), 21: frozenset({51})},
            ),
        ))

        keys = {(action.kind, action.source_uid, action.variant) for action in actions}
        self.assertIn((ActionKind.PLAY, 10, None), keys)
        self.assertNotIn((ActionKind.FUSION, 11, None), keys)
        self.assertIn((ActionKind.ATTACK_LEADER, 20, None), keys)
        self.assertIn((ActionKind.ATTACK_FOLLOWER, 21, None), keys)
        self.assertIn((ActionKind.EVOLVE, 33, None), keys)
        self.assertIn((ActionKind.SUPER_EVOLVE, 34, None), keys)
        # can_super_evolve_with_skill 是 UI 提示，不是独立动作，不应被枚举
        self.assertNotIn((ActionKind.SUPER_EVOLVE, 35, "with_skill"), keys)
        self.assertIn((ActionKind.ACTIVATE, 36, None), keys)
        self.assertIn((ActionKind.SPECIAL_ACTION, 37, None), keys)

    def test_enumerates_play_kinds(self):
        cards_by_uid = {
            1: CardView(uid=1, name="普通", cost=2, card_type=CardType.FOLLOWER),
            2: CardView(uid=2, name="爆能", cost=2, enhance_costs=(6,), card_type=CardType.FOLLOWER),
            3: CardView(uid=3, name="激奏", cost=8, accelerate_costs=(1,), card_type=CardType.FOLLOWER),
            4: CardView(uid=4, name="结晶", cost=5, crystal_costs=(2,), card_type=CardType.AMULET),
        }
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            cards_by_uid=cards_by_uid,
            legal=LegalView(
                can_play=frozenset({1}),
                can_enhance=frozenset({2}),
                can_accelerate=frozenset({3}),
                can_crystal=frozenset({4}),
            ),
        ))

        plays = [action for action in actions if action.kind == ActionKind.PLAY]
        self.assertEqual(
            [(action.source_uid, action.play_kind, action.cost) for action in plays],
            [
                (1, PlayKind.NORMAL, 2),
                (2, PlayKind.ENHANCE, 6),
                (3, PlayKind.ACCELERATE, 1),
                (4, PlayKind.CRYSTAL, 2),
            ],
        )
        self.assertTrue(all(action.legality_source == "protocol" for action in plays))
        self.assertTrue(all(action.confidence == "exact" for action in plays))

    def test_enumerates_evolve_and_super_evolve(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(can_evolve=frozenset({11}), can_super_evolve=frozenset({12})),
        ))

        self.assertIn((ActionKind.EVOLVE, 11), [(action.kind, action.source_uid) for action in actions])
        self.assertIn((ActionKind.SUPER_EVOLVE, 12), [(action.kind, action.source_uid) for action in actions])
        self.assertTrue(all(action.legality_source == "protocol" for action in actions))
        self.assertTrue(all(action.confidence == "exact" for action in actions))

    def test_enumerates_end_turn_when_protocol_turn_facts_allow(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(legal=LegalView(can_end_turn=True)))

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, ActionKind.TURN_END)
        self.assertEqual(actions[0].source_uid, 0)
        self.assertEqual(actions[0].legality_source, "protocol")
        self.assertEqual(actions[0].confidence, "exact")

    def test_enumerates_activate_skeleton(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(can_activate=frozenset({21}), can_fusion=frozenset({31})),
        ))

        self.assertIn((ActionKind.ACTIVATE, 21), [(action.kind, action.source_uid) for action in actions])
        self.assertFalse(any(action.kind == ActionKind.FUSION for action in actions))

    def test_enumerates_select_skill_targets_for_play_evolve_super_evolve_and_activate(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(
                can_play=frozenset({101}),
                can_evolve=frozenset({201}),
                can_super_evolve=frozenset({251}),
                can_activate=frozenset({301}),
                select_skill_map={
                    "play": {101: (SelectSkillOption(target_uids=(401, 402), selected_indexes=(0,)),)},
                    "evolve": {201: (SelectSkillOption(target_uids=(501,), selected_indexes=(0,)),)},
                    "super_evolve": {251: (SelectSkillOption(target_uids=(551,)),)},
                    "activation": {301: (SelectSkillOption(target_uids=(601,), selected_indexes=(0,)),)},
                },
            ),
        ))

        action_map = {(action.kind, action.source_uid): action for action in actions}
        self.assertEqual(action_map[(ActionKind.PLAY, 101)].target_groups, ((401, 402),))
        self.assertEqual(action_map[(ActionKind.PLAY, 101)].selected_indexes, (0,))
        self.assertEqual(action_map[(ActionKind.EVOLVE, 201)].target_groups, ((501,),))
        self.assertEqual(action_map[(ActionKind.SUPER_EVOLVE, 251)].target_groups, ((551,),))
        self.assertEqual(action_map[(ActionKind.SUPER_EVOLVE, 251)].selected_indexes, ())
        self.assertEqual(action_map[(ActionKind.ACTIVATE, 301)].target_groups, ((601,),))

    def test_enumerates_fusion_material_combinations_and_groups(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(
                can_fusion=frozenset({701, 702}),
                select_fusion_map={
                    701: SelectFusionOption(select_material_count=2, material_uids=(801, 802, 803)),
                    702: SelectFusionOption(material_groups=((901, 902), (903, 904))),
                },
            ),
        ))

        fusion_materials = [action.material_uids for action in actions if action.kind is ActionKind.FUSION]
        self.assertEqual(fusion_materials, [(801, 802), (801, 803), (802, 803), (901, 902), (903, 904)])
        self.assertTrue(all(action.legality_source == "protocol" for action in actions))
        self.assertTrue(all(action.confidence == "exact" for action in actions))

    def test_skips_fusion_without_protocol_materials(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            hand=(701,),
            legal=LegalView(can_fusion=frozenset({701})),
        ))

        self.assertFalse(any(action.kind is ActionKind.FUSION for action in actions))

    def test_enumerates_special_action(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(can_special_action=frozenset({41})),
        ))

        special = [a for a in actions if a.kind is ActionKind.SPECIAL_ACTION]
        self.assertEqual(len(special), 1)
        self.assertEqual(special[0].source_uid, 41)
        self.assertEqual(special[0].legality_source, "protocol")
        self.assertEqual(special[0].confidence, "exact")
        self.assertIsNone(special[0].variant)

    def test_enumerates_special_action_with_select_skill_targets(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(
                can_special_action=frozenset({42}),
                select_skill_map={
                    "special_action": {42: (SelectSkillOption(target_uids=(501, 502), selected_indexes=(1,)),)},
                },
            ),
        ))

        special = [a for a in actions if a.kind is ActionKind.SPECIAL_ACTION]
        self.assertEqual(len(special), 1)
        self.assertEqual(special[0].source_uid, 42)
        self.assertEqual(special[0].target_groups, ((501, 502),))
        self.assertEqual(special[0].selected_indexes, (1,))

    def test_enumerates_extra_pp_when_ex_pp_only_cards_exist(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(can_play_with_ex_pp=frozenset({51})),
        ))

        extra_pp = [a for a in actions if a.kind is ActionKind.EXTRA_PP]
        self.assertEqual(len(extra_pp), 1)
        self.assertEqual(extra_pp[0].source_uid, 0)
        self.assertEqual(extra_pp[0].legality_source, "protocol")

    def test_no_extra_pp_when_all_ex_pp_cards_already_in_can_play(self):
        cards_by_uid = {51: CardView(uid=51, cost=3)}
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            cards_by_uid=cards_by_uid,
            legal=LegalView(can_play=frozenset({51}), can_play_with_ex_pp=frozenset({51})),
        ))

        extra_pp = [a for a in actions if a.kind is ActionKind.EXTRA_PP]
        self.assertEqual(len(extra_pp), 0)

    def test_enumerates_extra_pp_for_activate_with_ex_pp(self):
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(can_activate_with_ex_pp=frozenset({61})),
        ))

        extra_pp = [a for a in actions if a.kind is ActionKind.EXTRA_PP]
        self.assertEqual(len(extra_pp), 1)

    def test_can_super_evolve_with_skill_is_not_enumerated_as_action(self):
        """can_super_evolve_with_skill 是 UI 提示字段，不产出独立动作。"""
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(can_super_evolve_with_skill=frozenset({71})),
        ))

        with_skill = [a for a in actions if a.kind is ActionKind.SUPER_EVOLVE and a.variant == "with_skill"]
        self.assertEqual(len(with_skill), 0)

    def test_ex_pp_and_normal_play_dedup_when_both_present(self):
        """同一张卡同时在 can_play 和 can_play_with_ex_pp 中时，只产生 normal play。"""
        cards_by_uid = {
            81: CardView(uid=81, name="双重卡", cost=5),
        }
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            cards_by_uid=cards_by_uid,
            legal=LegalView(
                can_play=frozenset({81}),
                can_play_with_ex_pp=frozenset({81}),
            ),
        ))

        plays = [a for a in actions if a.kind is ActionKind.PLAY and a.source_uid == 81]
        self.assertEqual(len(plays), 1)
        self.assertIsNone(plays[0].variant)
        self.assertEqual(plays[0].cost, 5)

    def test_normal_actions_have_no_variant(self):
        """非变体动作的 variant 字段应为 None。"""
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(
                can_activate=frozenset({91}),
                can_super_evolve=frozenset({92}),
            ),
        ))

        activate = next(a for a in actions if a.kind is ActionKind.ACTIVATE)
        super_evolve = next(a for a in actions if a.kind is ActionKind.SUPER_EVOLVE)
        self.assertIsNone(activate.variant)
        self.assertIsNone(super_evolve.variant)

    def test_mode_skill_generates_selected_indexes_variants(self):
        """mode_skill_map 中的卡应生成每个 mode index 的独立 play 动作。"""
        cards_by_uid = {13: CardView(uid=13, cost=2)}
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            cards_by_uid=cards_by_uid,
            legal=LegalView(
                can_play=frozenset({13}),
                mode_skill_map={13: ModeSkillInfo(mode_group=1, mode_count=2, select_count=1)},
            ),
        ))

        plays = [a for a in actions if a.kind is ActionKind.PLAY and a.source_uid == 13]
        self.assertEqual(len(plays), 2)
        self.assertEqual(plays[0].selected_indexes, (0,))
        self.assertEqual(plays[1].selected_indexes, (1,))
        self.assertTrue(all(p.legality_source == "protocol" for p in plays))

    def test_mode_skill_with_candidate_indexes(self):
        """candidate_indexes 非空时只生成指定 index 的变体。"""
        cards_by_uid = {31: CardView(uid=31, cost=5)}
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            cards_by_uid=cards_by_uid,
            legal=LegalView(
                can_play=frozenset({31}),
                mode_skill_map={31: ModeSkillInfo(mode_group=1, mode_count=3, select_count=1, candidate_indexes=(0, 2))},
            ),
        ))

        plays = [a for a in actions if a.kind is ActionKind.PLAY and a.source_uid == 31]
        self.assertEqual(len(plays), 2)
        self.assertEqual(plays[0].selected_indexes, (0,))
        self.assertEqual(plays[1].selected_indexes, (2,))

    def test_select_skill_map_takes_precedence_over_mode_skill(self):
        """select_skill_map 有条目时优先使用，不走 mode_skill_map。"""
        cards_by_uid = {13: CardView(uid=13, cost=2)}
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            cards_by_uid=cards_by_uid,
            legal=LegalView(
                can_play=frozenset({13}),
                select_skill_map={"play": {13: (SelectSkillOption(target_uids=(99,)),)}},
                mode_skill_map={13: ModeSkillInfo(mode_group=1, mode_count=2, select_count=1)},
            ),
        ))

        plays = [a for a in actions if a.kind is ActionKind.PLAY and a.source_uid == 13]
        self.assertEqual(len(plays), 1)
        self.assertEqual(plays[0].target_groups, ((99,),))

    def test_ex_pp_play_produces_extra_pp_action_not_variant(self):
        """ex_pp only 卡不产生 variant play，而是产生 EXTRA_PP 动作。"""
        cards_by_uid = {
            81: CardView(uid=81, cost=5),
            82: CardView(uid=82, cost=3),
        }
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            cards_by_uid=cards_by_uid,
            legal=LegalView(
                can_play=frozenset({81}),
                can_play_with_ex_pp=frozenset({81, 82}),
            ),
        ))

        plays = [a for a in actions if a.kind is ActionKind.PLAY]
        uid81 = [a for a in plays if a.source_uid == 81]
        self.assertEqual(len(uid81), 1)
        self.assertIsNone(uid81[0].variant)
        uid82_plays = [a for a in plays if a.source_uid == 82]
        self.assertEqual(len(uid82_plays), 0)
        extra_pp = [a for a in actions if a.kind is ActionKind.EXTRA_PP]
        self.assertEqual(len(extra_pp), 1)

    def test_ex_pp_activate_produces_extra_pp_action_not_variant(self):
        """ex_pp only activate 不产生 variant，而是产生 EXTRA_PP 动作。"""
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(
                can_activate=frozenset({61}),
                can_activate_with_ex_pp=frozenset({61, 62}),
            ),
        ))

        activates = [a for a in actions if a.kind is ActionKind.ACTIVATE]
        uid61 = [a for a in activates if a.source_uid == 61]
        uid62 = [a for a in activates if a.source_uid == 62]
        self.assertEqual(len(uid61), 1)
        self.assertIsNone(uid61[0].variant)
        self.assertEqual(len(uid62), 0)
        extra_pp = [a for a in actions if a.kind is ActionKind.EXTRA_PP]
        self.assertEqual(len(extra_pp), 1)

    def test_choice_skill_generates_selected_indexes(self):
        """choice_skill_map 中 choice_count=1 时按候选数生成独立动作。"""
        cards_by_uid = {15: CardView(uid=15, cost=3)}
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            cards_by_uid=cards_by_uid,
            legal=LegalView(
                can_play=frozenset({15}),
                choice_skill_map={"play": {15: ChoiceSkillInfo(choice_count=1, choice_id_list=(100, 200, 300))}},
            ),
        ))

        plays = [a for a in actions if a.kind is ActionKind.PLAY and a.source_uid == 15]
        self.assertEqual(len(plays), 3)
        self.assertEqual(plays[0].selected_indexes, (0,))
        self.assertEqual(plays[1].selected_indexes, (1,))
        self.assertEqual(plays[2].selected_indexes, (2,))

    def test_choice_skill_multi_count_generates_combinations(self):
        """choice_count=2 时按组合枚举。"""
        cards_by_uid = {16: CardView(uid=16, cost=4)}
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            cards_by_uid=cards_by_uid,
            legal=LegalView(
                can_play=frozenset({16}),
                choice_skill_map={"play": {16: ChoiceSkillInfo(choice_count=2, choice_id_list=(100, 200, 300))}},
            ),
        ))

        plays = [a for a in actions if a.kind is ActionKind.PLAY and a.source_uid == 16]
        self.assertEqual(len(plays), 3)  # C(3,2)=3
        indexes = [p.selected_indexes for p in plays]
        self.assertEqual(indexes, [(0, 1), (0, 2), (1, 2)])

    def test_choice_skill_full_select_returns_all(self):
        """choice_count >= 候选数时全选。"""
        cards_by_uid = {17: CardView(uid=17, cost=2)}
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            cards_by_uid=cards_by_uid,
            legal=LegalView(
                can_play=frozenset({17}),
                choice_skill_map={"play": {17: ChoiceSkillInfo(choice_count=3, choice_id_list=(100, 200))}},
            ),
        ))

        plays = [a for a in actions if a.kind is ActionKind.PLAY and a.source_uid == 17]
        self.assertEqual(len(plays), 1)
        self.assertEqual(plays[0].selected_indexes, (0, 1))

    def test_select_skill_takes_precedence_over_choice_skill(self):
        """select_skill_map 有条目时优先使用，不走 choice_skill_map。"""
        cards_by_uid = {18: CardView(uid=18, cost=2)}
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            cards_by_uid=cards_by_uid,
            legal=LegalView(
                can_play=frozenset({18}),
                select_skill_map={"play": {18: (SelectSkillOption(target_uids=(99,)),)}},
                choice_skill_map={"play": {18: ChoiceSkillInfo(choice_count=1, choice_id_list=(100, 200))}},
            ),
        ))

        plays = [a for a in actions if a.kind is ActionKind.PLAY and a.source_uid == 18]
        self.assertEqual(len(plays), 1)
        self.assertEqual(plays[0].target_groups, ((99,),))

    def test_super_evolve_uses_super_evolve_mode_skill_map(self):
        """超进化使用 super_evolve_mode_skill_map 而非 mode_skill_map。"""
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(
                can_super_evolve=frozenset({25}),
                mode_skill_map={25: ModeSkillInfo(mode_count=2, select_count=1)},
                super_evolve_mode_skill_map={25: ModeSkillInfo(mode_count=3, select_count=1)},
            ),
        ))

        se = [a for a in actions if a.kind is ActionKind.SUPER_EVOLVE and a.source_uid == 25]
        self.assertEqual(len(se), 3)
        self.assertEqual(se[0].selected_indexes, (0,))
        self.assertEqual(se[1].selected_indexes, (1,))
        self.assertEqual(se[2].selected_indexes, (2,))

    def test_evolve_does_not_use_super_evolve_mode_skill_map(self):
        """普通进化不使用 super_evolve_mode_skill_map。"""
        actions = ProtocolActionEnumerator().enumerate(_snapshot(
            legal=LegalView(
                can_evolve=frozenset({26}),
                super_evolve_mode_skill_map={26: ModeSkillInfo(mode_count=3, select_count=1)},
            ),
        ))

        ev = [a for a in actions if a.kind is ActionKind.EVOLVE and a.source_uid == 26]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].selected_indexes, ())


if __name__ == "__main__":
    unittest.main()
