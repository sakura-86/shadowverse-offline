# -*- coding: utf-8 -*-
"""Phase L4 核心规则内核测试。"""

import json
import unittest
from dataclasses import replace

from decision_engine.execution import CoreRuleKernel, RuleKernelError
from decision_engine.execution.rule_kernel import (
    BATTLE_RESULT_DECK_OUT_LOSE,
    BATTLE_RESULT_DECK_OUT_WIN,
    BATTLE_RESULT_VICTORY_CARD_LOSE,
    BATTLE_RESULT_VICTORY_CARD_WIN,
)
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, ExtraPpView, PlayerBuffView, PlayerView


class CoreRuleKernelTest(unittest.TestCase):
    def make_snapshot(self):
        me = PlayerView(life=20, max_life=20, pp=7, max_pp=7, ep=1, sep=1, deck_count=3, hand=(1, 2), field=(3,))
        opponent = PlayerView(life=18, max_life=20, pp=5, max_pp=5, deck_count=3, field=(4,))
        cards = {
            1: CardView(uid=1, card_id=101, base_card_id=101, name="随从", cost=3, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER),
            2: CardView(uid=2, card_id=102, base_card_id=102, name="法术", cost=2, card_type=CardType.SPELL),
            5: CardView(uid=5, card_id=105, base_card_id=105, name="增幅", cost=5, card_type=CardType.SPELL, has_spell_boost=True, spellboost_reduces_cost=True, spell_boost_count=0),
            3: CardView(uid=3, card_id=103, base_card_id=103, name="我方场上", atk=3, life=4, max_life=4, attacks_left=1),
            4: CardView(uid=4, card_id=104, base_card_id=104, name="敌方场上", atk=2, life=3, max_life=3, attacks_left=1, has_last_words=True),
        }
        return BattleSnapshot(me=me, opponent=opponent, cards_by_uid=cards, source="test", is_ally_turn=True)

    def test_extra_pp_is_second_player_only_and_granted_on_turn_one_and_six(self):
        kernel = CoreRuleKernel()
        first = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=1, deck_count=1, turn=0, is_first_side=True, extra_pp=ExtraPpView(extra_pp=1)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=1),
            cards_by_uid={},
        )
        first_start = kernel.start_turn(first, "me")
        self.assertEqual(first_start.snapshot.me.turn, 1)
        # 先手方保持输入的 extra_pp 不变（不发放、不清零）
        self.assertEqual(first_start.snapshot.me.extra_pp, ExtraPpView(extra_pp=1))

        second = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=1, deck_count=1, turn=0, is_first_side=False, extra_pp=None),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=1),
            cards_by_uid={},
        )
        turn_one = kernel.start_turn(second, "me")
        self.assertEqual(turn_one.snapshot.me.turn, 1)
        self.assertEqual(turn_one.snapshot.me.extra_pp.extra_pp, 1)
        self.assertEqual(turn_one.snapshot.me.extra_pp.preparation_extra_pp, 0)
        self.assertFalse(turn_one.snapshot.me.extra_pp.is_used_extra_pp_this_turn)
        self.assertEqual(turn_one.snapshot.me.extra_pp.restore_extra_pp_turn, 6)

        held_until_six = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=6, deck_count=1, turn=5, is_first_side=False, extra_pp=ExtraPpView(extra_pp=1, preparation_extra_pp=1, is_used_extra_pp_this_turn=True)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=6),
            cards_by_uid={},
        )
        turn_six_held = kernel.start_turn(held_until_six, "me")
        self.assertEqual(turn_six_held.snapshot.me.turn, 6)
        self.assertEqual(turn_six_held.snapshot.me.extra_pp.extra_pp, 1)
        self.assertEqual(turn_six_held.snapshot.me.extra_pp.preparation_extra_pp, 0)
        self.assertFalse(turn_six_held.snapshot.me.extra_pp.is_used_extra_pp_this_turn)
        self.assertEqual(turn_six_held.snapshot.me.extra_pp.restore_extra_pp_turn, 6)

        spent_before_six = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=6, deck_count=1, turn=5, is_first_side=False, extra_pp=ExtraPpView(extra_pp=0)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=6),
            cards_by_uid={},
        )
        turn_six_refilled = kernel.start_turn(spent_before_six, "me")
        self.assertEqual(turn_six_refilled.snapshot.me.extra_pp.extra_pp, 1)

    def test_extra_pp_does_not_auto_restore_after_grant_turns(self):
        kernel = CoreRuleKernel()
        spent_midgame = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=5, deck_count=1, turn=4, is_first_side=False, extra_pp=ExtraPpView(extra_pp=0)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=5),
            cards_by_uid={},
        )
        turn_five = kernel.start_turn(spent_midgame, "me")
        self.assertEqual(turn_five.snapshot.me.turn, 5)
        self.assertEqual(turn_five.snapshot.me.extra_pp.extra_pp, 0)
        self.assertEqual(turn_five.snapshot.me.extra_pp.restore_extra_pp_turn, 6)

        held_late = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=10, deck_count=1, turn=9, is_first_side=False, extra_pp=ExtraPpView(extra_pp=1)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=10),
            cards_by_uid={},
        )
        turn_ten = kernel.start_turn(held_late, "me")
        self.assertEqual(turn_ten.snapshot.me.turn, 10)
        self.assertEqual(turn_ten.snapshot.me.extra_pp.extra_pp, 1)
        self.assertEqual(turn_ten.snapshot.me.extra_pp.restore_extra_pp_turn, 6)

    def test_extra_pp_activation_and_payment_consumption(self):
        kernel = CoreRuleKernel()
        cards = {1: CardView(uid=1, card_id=101, base_card_id=101, name="7费随从", cost=7, atk=1, life=1, max_life=1, card_type=CardType.FOLLOWER)}
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=6, max_pp=6, hand=(1,), is_first_side=False, extra_pp=ExtraPpView(extra_pp=1)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=6),
            cards_by_uid=cards,
        )

        activated = kernel.activate_extra_pp(snapshot, "me")
        self.assertEqual(activated.snapshot.me.pp, 7)
        self.assertEqual(activated.snapshot.me.extra_pp.extra_pp, 1)
        self.assertEqual(activated.snapshot.me.extra_pp.preparation_extra_pp, 1)
        self.assertTrue(activated.snapshot.me.extra_pp.is_used_extra_pp_this_turn)
        with self.assertRaisesRegex(RuleKernelError, "already used"):
            kernel.activate_extra_pp(activated.snapshot, "me")

        played = kernel.play_card_from_hand(activated.snapshot, "me", 1, cost=7)
        self.assertEqual(played.snapshot.me.pp, 0)
        self.assertEqual(played.snapshot.me.extra_pp.extra_pp, 0)
        self.assertEqual(played.snapshot.me.extra_pp.preparation_extra_pp, 0)
        self.assertEqual(played.snapshot.me.extra_pp.extra_pp_state, 2)
        self.assertTrue(played.snapshot.me.extra_pp.is_used_extra_pp_this_turn)

    def test_extra_pp_payment_uses_preparation_even_when_runtime_used_flag_is_false(self):
        kernel = CoreRuleKernel()
        cards = {1: CardView(uid=1, card_id=101, base_card_id=101, name="3费随从", cost=3, atk=1, life=1, max_life=1, card_type=CardType.FOLLOWER)}
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=3, max_pp=2, hand=(1,), is_first_side=False, extra_pp=ExtraPpView(extra_pp=0, preparation_extra_pp=1, extra_pp_state=1, restore_extra_pp_turn=6)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2),
            cards_by_uid=cards,
        )

        played = kernel.play_card_from_hand(snapshot, "me", 1, cost=3)

        self.assertEqual(played.snapshot.me.pp, 0)
        self.assertEqual(played.snapshot.me.extra_pp.extra_pp, 0)
        self.assertEqual(played.snapshot.me.extra_pp.preparation_extra_pp, 0)
        self.assertEqual(played.snapshot.me.extra_pp.extra_pp_state, 2)
        self.assertTrue(played.snapshot.me.extra_pp.is_used_extra_pp_this_turn)

    def test_extra_pp_activation_without_spending_keeps_consumable(self):
        kernel = CoreRuleKernel()
        cards = {1: CardView(uid=1, card_id=101, base_card_id=101, name="3费随从", cost=3, atk=1, life=1, max_life=1, card_type=CardType.FOLLOWER)}
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=6, max_pp=6, hand=(1,), is_first_side=False, extra_pp=ExtraPpView(extra_pp=1)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=6),
            cards_by_uid=cards,
        )

        activated = kernel.activate_extra_pp(snapshot, "me")
        played = kernel.play_card_from_hand(activated.snapshot, "me", 1, cost=3)
        self.assertEqual(played.snapshot.me.pp, 4)
        self.assertEqual(played.snapshot.me.extra_pp.extra_pp, 1)
        self.assertEqual(played.snapshot.me.extra_pp.preparation_extra_pp, 1)
        ended = kernel.end_turn(played.snapshot, "me")
        self.assertEqual(ended.snapshot.me.extra_pp.extra_pp, 1)
        self.assertEqual(ended.snapshot.me.extra_pp.preparation_extra_pp, 0)

    def test_consume_resources_and_rollback(self):
        snapshot = self.make_snapshot()
        kernel = CoreRuleKernel()

        result = kernel.consume_pp(snapshot, "me", 3)
        self.assertEqual(result.snapshot.me.pp, 4)
        self.assertEqual(result.events, ("consume_pp",))
        self.assertEqual(result.transaction_log.rollback(), snapshot)

        with self.assertRaises(RuleKernelError):
            kernel.consume_ep(snapshot, "me", 2)

    def test_play_follower_moves_hand_to_field(self):
        snapshot = self.make_snapshot()
        kernel = CoreRuleKernel()

        result = kernel.play_card_from_hand(snapshot, "me", 1, cost=3)

        self.assertEqual(result.snapshot.me.pp, 4)
        self.assertEqual(result.snapshot.me.hand, (2,))
        self.assertEqual(result.snapshot.me.field, (3, 1))
        self.assertEqual(result.snapshot.me.play_count, 1)
        self.assertEqual(result.snapshot.me.rally, 1)
        self.assertIn("enter_field", result.events)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_play_amulet_does_not_increment_rally(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[2] = replace(cards[2], card_type=CardType.AMULET)
        snapshot = BattleSnapshot(
            me=replace(snapshot.me, rally=3),
            opponent=snapshot.opponent,
            cards_by_uid=cards,
            source=snapshot.source,
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()

        result = kernel.play_card_from_hand(snapshot, "me", 2, cost=2)

        self.assertEqual(result.snapshot.me.play_count, 1)
        self.assertEqual(result.snapshot.me.rally, 3)
        self.assertEqual(result.snapshot.me.field, (3, 2))

    def test_play_spell_moves_to_graveyard_and_removes_card_instance(self):
        snapshot = self.make_snapshot()
        kernel = CoreRuleKernel()

        result = kernel.play_spell_from_hand(snapshot, "me", 2, cost=2)

        self.assertEqual(result.snapshot.me.pp, 5)
        self.assertEqual(result.snapshot.me.hand, (1,))
        self.assertEqual(result.snapshot.me.graveyard_count, 1)
        self.assertNotIn(2, result.snapshot.cards_by_uid)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_play_spell_triggers_hand_spellboost(self):
        snapshot = self.make_snapshot()
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=7, max_pp=7, hand=(2, 5)),
            opponent=snapshot.opponent,
            cards_by_uid={2: snapshot.cards_by_uid[2], 5: snapshot.cards_by_uid[5]},
            source="test",
        )
        kernel = CoreRuleKernel()

        result = kernel.play_spell_from_hand(snapshot, "me", 2, cost=2)

        self.assertEqual(result.snapshot.cards_by_uid[5].spell_boost_count, 1)
        self.assertEqual(result.snapshot.cards_by_uid[5].cost, 4)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_play_cost_modes_apply_zone_and_cost_slices(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[6] = CardView(uid=6, card_id=106, base_card_id=106, name="爆能随从", cost=2, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER)
        cards[7] = CardView(uid=7, card_id=107, base_card_id=107, name="激奏随从", cost=7, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER)
        cards[8] = CardView(uid=8, card_id=108, base_card_id=108, name="结晶随从", cost=7, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER)
        mode_snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=7, max_pp=7, hand=(6, 7, 8), field=()),
            opponent=snapshot.opponent,
            cards_by_uid=cards,
            source="test",
        )
        kernel = CoreRuleKernel()

        enhanced = kernel.play_enhanced_from_hand(mode_snapshot, "me", 6, cost=6)
        accelerated = kernel.play_accelerated_from_hand(mode_snapshot, "me", 7, cost=1)
        crystallized = kernel.play_crystallized_from_hand(mode_snapshot, "me", 8, cost=2)

        self.assertEqual(enhanced.snapshot.me.pp, 1)
        self.assertEqual(enhanced.snapshot.me.field, (6,))
        self.assertIn("enhance", enhanced.events)
        self.assertEqual(enhanced.transaction_log.rollback(), mode_snapshot)
        self.assertEqual(accelerated.snapshot.me.pp, 6)
        self.assertEqual(accelerated.snapshot.me.graveyard_count, 1)
        self.assertNotIn(7, accelerated.snapshot.cards_by_uid)
        self.assertIn("accelerate", accelerated.events)
        self.assertIn("play_spell", accelerated.events)
        self.assertEqual(crystallized.snapshot.me.pp, 5)
        self.assertEqual(crystallized.snapshot.me.field, (8,))
        self.assertEqual(crystallized.snapshot.cards_by_uid[8].card_type, CardType.COUNTDOWN_AMULET)
        self.assertIn("crystal", crystallized.events)

    def test_field_full_blocks_play_and_summon_vanishes_without_events(self):
        snapshot = self.make_snapshot()
        full_me = PlayerView(life=20, max_life=20, pp=7, max_pp=7, hand=(1,), field=(3, 5, 6, 7, 8))
        cards = dict(snapshot.cards_by_uid)
        for uid in (5, 6, 7, 8):
            cards[uid] = CardView(uid=uid, card_id=uid, base_card_id=uid, name=f"场上{uid}")
        full_snapshot = BattleSnapshot(me=full_me, opponent=snapshot.opponent, cards_by_uid=cards, source="test")
        kernel = CoreRuleKernel()

        with self.assertRaises(RuleKernelError):
            kernel.play_card_from_hand(full_snapshot, "me", 1, cost=3)

        summon_card = CardView(uid=9, card_id=109, base_card_id=109, name="召唤物")
        result = kernel.summon_card(full_snapshot, "me", summon_card)
        self.assertEqual(result.snapshot, full_snapshot)
        self.assertEqual(result.events, ("summon_failed_field_full",))
        self.assertNotIn(9, result.snapshot.cards_by_uid)

    def test_summon_amulet_does_not_increment_rally(self):
        snapshot = self.make_snapshot()
        snapshot = BattleSnapshot(
            me=replace(snapshot.me, rally=3),
            opponent=snapshot.opponent,
            cards_by_uid=snapshot.cards_by_uid,
            source=snapshot.source,
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()
        summon_card = CardView(uid=9, card_id=109, base_card_id=109, name="护符", card_type=CardType.AMULET)

        result = kernel.summon_card(snapshot, "me", summon_card)

        self.assertEqual(result.snapshot.me.rally, 3)
        self.assertEqual(result.snapshot.me.field, (3, 9))

    def test_draw_card_and_overdraw(self):
        snapshot = self.make_snapshot()
        kernel = CoreRuleKernel()
        drawn = CardView(uid=9, card_id=109, base_card_id=109, name="抽到的卡")

        result = kernel.draw_card(snapshot, "me", drawn)
        self.assertEqual(result.snapshot.me.deck_count, 2)
        self.assertEqual(result.snapshot.me.hand, (1, 2, 9))
        self.assertIn(9, result.snapshot.cards_by_uid)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

        full_hand = tuple(range(10, 19))
        full_cards = {uid: CardView(uid=uid, card_id=uid, base_card_id=uid, name=str(uid)) for uid in full_hand}
        full_snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, deck_count=1, hand=full_hand),
            opponent=snapshot.opponent,
            cards_by_uid=full_cards,
        )
        burned = CardView(uid=99, card_id=199, base_card_id=199, name="爆牌")
        overdraw = kernel.draw_card(full_snapshot, "me", burned)
        self.assertEqual(overdraw.snapshot.me.deck_count, 0)
        self.assertEqual(overdraw.snapshot.me.hand, full_hand)
        self.assertEqual(overdraw.snapshot.me.graveyard_count, 1)
        self.assertNotIn(99, overdraw.snapshot.cards_by_uid)
        self.assertIn("overdraw", overdraw.events)

        hidden_overdraw = kernel.draw_card(full_snapshot, "me", None)
        self.assertEqual(hidden_overdraw.snapshot.me.deck_count, 0)
        self.assertEqual(hidden_overdraw.snapshot.me.hand, full_hand)
        self.assertEqual(hidden_overdraw.snapshot.me.graveyard_count, 1)
        self.assertEqual(hidden_overdraw.events, ("draw_hidden",))

    def test_deckout_materializes_terminal_card_only_on_draw_after_deck_is_empty(self):
        snapshot = self.make_snapshot()
        kernel = CoreRuleKernel()
        final_card = CardView(uid=99, card_id=199, base_card_id=199, name="牌库最后一张", card_type=CardType.SPELL)
        one_card_left = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, deck_count=1),
            opponent=snapshot.opponent,
            cards_by_uid={},
        )

        last_draw = kernel.draw_card(one_card_left, "me", final_card)
        self.assertEqual(last_draw.snapshot.me.deck_count, 0)
        self.assertEqual(last_draw.snapshot.me.result, 0)
        self.assertEqual(last_draw.snapshot.opponent.result, 0)
        self.assertFalse(last_draw.snapshot.is_turn_ended)
        self.assertEqual(last_draw.events, ("draw",))

        king_of_death = kernel.draw_card(last_draw.snapshot, "me", None)
        self.assertTrue(king_of_death.snapshot.is_turn_ended)
        self.assertEqual(king_of_death.snapshot.me.result, BATTLE_RESULT_DECK_OUT_LOSE)
        self.assertEqual(king_of_death.snapshot.opponent.result, BATTLE_RESULT_DECK_OUT_WIN)
        self.assertEqual(king_of_death.events, ("deckout",))
        self.assertEqual(
            king_of_death.events[0].payload["deckout_card"],
            {
                "kind": "king_of_death",
                "name": "亡者之王",
                "player_result": BATTLE_RESULT_DECK_OUT_LOSE,
                "opponent_result": BATTLE_RESULT_DECK_OUT_WIN,
            },
        )
        self.assertEqual(king_of_death.transaction_log.rollback(), last_draw.snapshot)

    def test_deckout_win_materializes_victory_card_and_victory_result_codes(self):
        snapshot = self.make_snapshot()
        deckout_win = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, deck_count=0, is_deck_out_win=True),
            opponent=snapshot.opponent,
            cards_by_uid={},
        )

        result = CoreRuleKernel().draw_card(deckout_win, "me", None)
        self.assertTrue(result.snapshot.is_turn_ended)
        self.assertEqual(result.snapshot.me.result, BATTLE_RESULT_VICTORY_CARD_WIN)
        self.assertEqual(result.snapshot.opponent.result, BATTLE_RESULT_VICTORY_CARD_LOSE)
        self.assertEqual(
            result.events[0].payload["deckout_card"],
            {
                "kind": "victory_card",
                "name": "胜利的卡牌",
                "player_result": BATTLE_RESULT_VICTORY_CARD_WIN,
                "opponent_result": BATTLE_RESULT_VICTORY_CARD_LOSE,
            },
        )
        self.assertEqual(result.transaction_log.rollback(), deckout_win)

    def test_destroy_and_banish_field_card(self):
        snapshot = self.make_snapshot()
        kernel = CoreRuleKernel()

        destroyed = kernel.destroy_field_card(snapshot, "opponent", 4)
        self.assertNotIn(4, destroyed.snapshot.cards_by_uid)
        self.assertEqual(destroyed.snapshot.opponent.field, ())
        self.assertEqual(destroyed.snapshot.opponent.graveyard_count, 1)
        self.assertEqual(destroyed.snapshot.opponent.destroyed_card_ids[0].card_id, 104)
        self.assertEqual(destroyed.snapshot.opponent.destroyed_card_ids[0].count, 0)
        self.assertIn("last_words", destroyed.events)
        last_words = next(event for event in destroyed.events if event == "last_words")
        self.assertEqual(last_words.source_uid, 4)
        self.assertEqual(last_words.player, "opponent")
        self.assertEqual(
            last_words.payload["leaving_card"],
            {"card_id": 104, "base_card_id": 104, "name": "敌方场上", "card_type": "follower"},
        )
        json.dumps(last_words.payload)
        self.assertEqual(destroyed.transaction_log.rollback(), snapshot)

        banished = kernel.banish_field_card(snapshot, "opponent", 4)
        self.assertNotIn(4, banished.snapshot.cards_by_uid)
        self.assertEqual(banished.snapshot.opponent.field, ())
        self.assertEqual(banished.snapshot.opponent.graveyard_count, 0)
        self.assertEqual(banished.snapshot.opponent.destroyed_card_ids, ())
        self.assertNotIn("last_words", banished.events)
        self.assertEqual(banished.transaction_log.rollback(), snapshot)

    def test_bounce_field_card_and_full_hand_flood(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[3] = CardView(
            uid=3,
            card_id=10012310,
            base_card_id=10012310,
            name="昆虫的忠告",
            cost=1,
            atk=5,
            life=5,
            max_life=5,
            card_type=CardType.FOLLOWER,
            evolve_state=1,
            attacks_left=1,
        )
        snapshot = BattleSnapshot(me=replace(snapshot.me, hand=(1, 2), field=(3,)), opponent=snapshot.opponent, cards_by_uid=cards, source="test", is_ally_turn=True)
        kernel = CoreRuleKernel()

        bounced = kernel.bounce_field_card(snapshot, "me", 3)
        self.assertEqual(bounced.snapshot.me.field, ())
        self.assertEqual(bounced.snapshot.me.hand, (1, 2, 3))
        self.assertEqual(bounced.snapshot.cards_by_uid[3].atk, 0)
        self.assertEqual(bounced.snapshot.cards_by_uid[3].life, 0)
        self.assertEqual(bounced.snapshot.cards_by_uid[3].evolve_state, 0)
        self.assertEqual(bounced.snapshot.cards_by_uid[3].card_id, 10012310)
        self.assertEqual(bounced.snapshot.me.graveyard_count, 0)
        self.assertEqual(bounced.transaction_log.rollback(), snapshot)

        full_hand = tuple(range(10, 19))
        cards = {uid: CardView(uid=uid, card_id=uid, base_card_id=uid, name=str(uid)) for uid in full_hand}
        cards[3] = snapshot.cards_by_uid[3]
        full_snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, hand=full_hand, field=(3,)),
            opponent=snapshot.opponent,
            cards_by_uid=cards,
        )
        flooded = kernel.bounce_field_card(full_snapshot, "me", 3)
        self.assertEqual(flooded.snapshot.me.field, ())
        self.assertEqual(flooded.snapshot.me.hand, full_hand)
        self.assertEqual(flooded.snapshot.me.graveyard_count, 0)
        self.assertEqual(flooded.snapshot.me.destroyed_card_ids, ())
        self.assertNotIn(3, flooded.snapshot.cards_by_uid)
        self.assertIn("flood", flooded.events)
        self.assertEqual(flooded.transaction_log.rollback(), full_snapshot)

    def test_transform_field_card_replaces_uid_without_enter_field_or_rally(self):
        snapshot = self.make_snapshot()
        kernel = CoreRuleKernel()
        replacement = CardView(uid=9, card_id=109, base_card_id=109, name="变身后", atk=5, life=5, max_life=5, attacks_left=3, evolve_state=2)

        result = kernel.transform_field_card(snapshot, "me", 3, replacement)

        self.assertNotIn(3, result.snapshot.cards_by_uid)
        self.assertIn(9, result.snapshot.cards_by_uid)
        self.assertEqual(result.snapshot.me.field, (9,))
        self.assertEqual(result.snapshot.cards_by_uid[9].attacks_left, 0)
        self.assertEqual(result.snapshot.cards_by_uid[9].evolve_state, 0)
        self.assertEqual(result.snapshot.me.rally, 0)
        self.assertEqual(result.snapshot.me.destroyed_card_ids, ())
        self.assertNotIn("enter_field", result.events)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_end_turn_removes_temporary_cards_and_records_destroyed_history(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[6] = CardView(uid=6, card_id=106, base_card_id=106, name="临时我方", is_remove_field_at_turn_change=True, has_last_words=True)
        cards[7] = CardView(uid=7, card_id=107, base_card_id=107, name="临时敌方", is_remove_field_at_turn_change=True, has_last_words=True)
        turn_snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=7, field=(3, 6), destroyed_card_ids=()),
            opponent=PlayerView(life=18, max_life=20, pp=5, max_pp=5, field=(4, 7), destroyed_card_ids=()),
            cards_by_uid=cards,
            source="test",
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()

        result = kernel.end_turn(turn_snapshot, "me")

        self.assertTrue(result.snapshot.is_turn_ended)
        # 「回合结束时离场」只对**持有者自己的回合**生效：end_turn("me") 时只清理
        # 我方场上的临时卡 6；对手场上的 7 属于对手，要等对手自己的回合结束才离场。
        # 录制证据：replay_...50009671 frame57（我方 10673110 在对手回合结束仍保留）、
        # frame66（同一张卡在我方回合结束才消失）、replay_...463579 frame100
        # （对手 10804110 在我方回合结束仍保留）。
        self.assertEqual(result.snapshot.me.field, (3,))
        self.assertEqual(result.snapshot.opponent.field, (4, 7))
        self.assertEqual(result.snapshot.me.graveyard_count, 1)
        self.assertEqual(result.snapshot.opponent.graveyard_count, 0)
        self.assertEqual(result.snapshot.me.destroyed_card_ids[0].card_id, 106)
        self.assertEqual(result.snapshot.opponent.destroyed_card_ids, ())
        self.assertIn("turn_end", result.events)
        self.assertIn("last_words", result.events)
        last_words_payloads = [event.payload["leaving_card"] for event in result.events if event == "last_words"]
        self.assertEqual(
            last_words_payloads,
            [
                {"card_id": 106, "base_card_id": 106, "name": "临时我方", "card_type": "follower"},
            ],
        )
        self.assertEqual(result.transaction_log.rollback(), turn_snapshot)

    def test_start_turn_refills_pp_and_resets_turn_flags(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=7, turn=7, deck_count=1, play_count=3, manual_evolve_count=1, is_evolved_this_turn=True),
            opponent=PlayerView(life=18, max_life=20, pp=2, max_pp=5, play_count=2, manual_evolve_count=1, is_evolved_this_turn=True),
            cards_by_uid={},
            source="test",
            is_ally_turn=False,
            is_turn_ended=True,
        )
        kernel = CoreRuleKernel()
        drawn = CardView(uid=9, card_id=109, base_card_id=109, name="自然抽牌")

        result = kernel.start_turn(snapshot, "me", drawn)

        self.assertTrue(result.snapshot.is_ally_turn)
        self.assertFalse(result.snapshot.is_turn_ended)
        self.assertEqual(result.snapshot.me.pp, 8)  # max_pp=7 +1 = 8
        self.assertEqual(result.snapshot.me.max_pp, 8)
        self.assertEqual(result.snapshot.me.deck_count, 0)
        self.assertEqual(result.snapshot.me.hand, (9,))
        self.assertEqual(result.snapshot.me.play_count, 0)
        self.assertEqual(result.snapshot.me.manual_evolve_count, 0)
        self.assertFalse(result.snapshot.me.is_evolved_this_turn)
        self.assertEqual(result.snapshot.opponent.play_count, 0)
        self.assertEqual(result.events, ("turn_start", "refill_pp", "draw"))
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_start_turn_increments_existing_hand_skybound_art_only(self):
        cards = {
            1: CardView(uid=1, card_id=101, base_card_id=101, name="奥义手牌", supplement_info={"skybound_art_count": 3}),
            2: CardView(uid=2, card_id=102, base_card_id=102, name="普通手牌"),
        }
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=3, deck_count=1, hand=(1, 2)),
            opponent=PlayerView(life=18, max_life=20, pp=2, max_pp=5),
            cards_by_uid=cards,
            source="test",
            is_ally_turn=False,
            is_turn_ended=True,
        )
        kernel = CoreRuleKernel()
        drawn = CardView(uid=9, card_id=109, base_card_id=109, name="新抽奥义", supplement_info={"skybound_art_count": 7})

        result = kernel.start_turn(snapshot, "me", drawn)

        self.assertEqual(result.snapshot.cards_by_uid[1].supplement_info["skybound_art_count"], 4)
        self.assertEqual(result.snapshot.cards_by_uid[9].supplement_info["skybound_art_count"], 7)
        self.assertIn("skybound_art", result.events)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_start_turn_initializes_drawn_skybound_art_to_turn_number(self):
        """新抽到的奥义卡 skybound_art_count=0 时，被初始化为当前回合数。"""
        cards = {
            1: CardView(uid=1, card_id=101, base_card_id=101, name="普通手牌"),
        }
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=9, turn=9, deck_count=1, hand=(1,)),
            opponent=PlayerView(life=18, max_life=20, pp=2, max_pp=5),
            cards_by_uid=cards,
            source="test",
            is_ally_turn=False,
            is_turn_ended=True,
        )
        kernel = CoreRuleKernel()
        drawn = CardView(uid=9, card_id=109, base_card_id=109, name="新抽奥义", supplement_info={"skybound_art_count": 0, "deck_order": 5})

        result = kernel.start_turn(snapshot, "me", drawn)

        # next_turn = 10, 所以新抽的卡 skybound_art_count 应为 10
        self.assertEqual(result.snapshot.cards_by_uid[9].supplement_info["skybound_art_count"], 10)

    def test_start_turn_decrements_countdown_and_removes_zero_countdown(self):
        cards = {
            6: CardView(uid=6, card_id=106, base_card_id=106, name="倒计时2", countdown=2, card_type=CardType.AMULET),
            7: CardView(uid=7, card_id=107, base_card_id=107, name="倒计时1", countdown=1, card_type=CardType.AMULET),
        }
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=3, deck_count=1, field=(6, 7)),
            opponent=PlayerView(life=18, max_life=20, pp=2, max_pp=5),
            cards_by_uid=cards,
            source="test",
            is_ally_turn=False,
            is_turn_ended=True,
        )
        kernel = CoreRuleKernel()

        result = kernel.start_turn(snapshot, "me", None)

        self.assertEqual(result.snapshot.cards_by_uid[6].countdown, 1)
        self.assertNotIn(7, result.snapshot.cards_by_uid)
        self.assertEqual(result.snapshot.me.field, (6,))
        self.assertEqual(result.snapshot.me.graveyard_count, 1)
        self.assertEqual(result.snapshot.me.destroyed_card_ids[0].card_id, 107)
        self.assertIn("countdown", result.events)
        self.assertIn("last_words", result.events)
        last_words = next(event for event in result.events if event == "last_words")
        self.assertEqual(last_words.payload["leaving_card"], {"card_id": 107, "base_card_id": 107, "name": "倒计时1", "card_type": "amulet"})
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_consume_necromancy_and_earth_sigil(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[6] = CardView(uid=6, card_id=10031210, base_card_id=10031210, name="土印A", card_type=CardType.AMULET, is_earth_sigil=True, stack=2)
        cards[7] = CardView(uid=7, card_id=10031210, base_card_id=10031210, name="土印B", card_type=CardType.AMULET, is_earth_sigil=True, stack=1)
        resource_snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, graveyard_count=5, field=(6, 7)),
            opponent=snapshot.opponent,
            cards_by_uid=cards,
            source="test",
        )
        kernel = CoreRuleKernel()

        necromancy = kernel.consume_necromancy(resource_snapshot, "me", 3)
        self.assertEqual(necromancy.snapshot.me.graveyard_count, 2)
        self.assertEqual(necromancy.events, ("necromancy",))
        self.assertEqual(necromancy.transaction_log.rollback(), resource_snapshot)

        earth = kernel.consume_earth_sigil(resource_snapshot, "me", 3)
        self.assertEqual(earth.snapshot.me.field, ())
        # 土之印被消耗到 0 是「被破坏」：两张护符各进墓地一次（5 -> 7）。
        # 录像证据：replay_...1833407 st103 里大地之魔片被【土之秘术】吃掉后
        # 对手墓地 +1，此前实现只把它从场上移除、不记账。
        self.assertEqual(earth.snapshot.me.graveyard_count, 7)
        self.assertNotIn(6, earth.snapshot.cards_by_uid)
        self.assertNotIn(7, earth.snapshot.cards_by_uid)
        self.assertEqual(
            earth.events,
            ("earth_rite", "destroy", "move_to_graveyard", "earth_rite", "destroy", "move_to_graveyard"),
        )
        self.assertEqual(earth.transaction_log.rollback(), resource_snapshot)

        partial = kernel.consume_earth_sigil(resource_snapshot, "me", 1)
        self.assertEqual(partial.snapshot.cards_by_uid[6].stack, 1)
        self.assertEqual(partial.snapshot.me.field, (6, 7))

        with self.assertRaises(RuleKernelError):
            kernel.consume_necromancy(resource_snapshot, "me", 6)
        with self.assertRaises(RuleKernelError):
            kernel.consume_earth_sigil(resource_snapshot, "me", 4)

    def test_activate_earth_sigil_adds_stack_and_consumes_pp_once(self):
        card = CardView(uid=3, card_id=90031210, base_card_id=90031210, name="大地之魔片", card_type=CardType.AMULET, is_earth_sigil=True, stack=1, has_activation=True)
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=2, max_pp=2, field=(3,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=2),
            cards_by_uid={3: card},
            source="test",
        )
        kernel = CoreRuleKernel()

        activated = kernel.activate_card(snapshot, "me", 3)

        self.assertEqual(activated.snapshot.me.pp, 1)
        self.assertEqual(activated.snapshot.cards_by_uid[3].stack, 2)
        self.assertFalse(activated.snapshot.cards_by_uid[3].has_activation)
        self.assertEqual(activated.events, ("consume_pp", "activate"))
        self.assertEqual(activated.transaction_log.rollback(), snapshot)

    def test_entering_earth_sigil_absorbs_existing_stack(self):
        old = CardView(uid=3, card_id=90031210, base_card_id=90031210, name="旧土", card_type=CardType.AMULET, is_earth_sigil=True, stack=2)
        new = CardView(uid=4, card_id=10031210, base_card_id=10031210, name="新土", cost=1, card_type=CardType.AMULET, is_earth_sigil=True, stack=1, has_activation=True)
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=3, max_pp=3, hand=(4,), field=(3,)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=3),
            cards_by_uid={3: old, 4: new},
            source="test",
        )
        kernel = CoreRuleKernel()

        played = kernel.play_card_from_hand(snapshot, "me", 4, cost=1)

        self.assertEqual(played.snapshot.me.field, (4,))
        self.assertNotIn(3, played.snapshot.cards_by_uid)
        self.assertEqual(played.snapshot.cards_by_uid[4].stack, 3)
        self.assertTrue(played.snapshot.cards_by_uid[4].has_activation)
        self.assertEqual(played.transaction_log.rollback(), snapshot)

    def test_activate_and_consume_extra_pp(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=6, max_pp=6, extra_pp=ExtraPpView(extra_pp=1, preparation_extra_pp=0, extra_pp_state=1, restore_extra_pp_turn=8)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=6),
            cards_by_uid={},
            source="test",
        )
        kernel = CoreRuleKernel()

        activated = kernel.activate_extra_pp(snapshot, "me")
        self.assertEqual(activated.snapshot.me.pp, 7)
        self.assertEqual(activated.snapshot.me.extra_pp.extra_pp, 1)
        self.assertTrue(activated.snapshot.me.extra_pp.is_used_extra_pp_this_turn)
        self.assertEqual(activated.events, ("extra_pp",))
        self.assertEqual(activated.transaction_log.rollback(), snapshot)

        consumed = kernel.consume_extra_pp(activated.snapshot, "me")
        self.assertEqual(consumed.snapshot.me.extra_pp.extra_pp, 0)
        self.assertEqual(consumed.events, ("consume_extra_pp",))

        with self.assertRaises(RuleKernelError):
            kernel.activate_extra_pp(replace(snapshot, me=replace(snapshot.me, extra_pp=None)), "me")
        with self.assertRaises(RuleKernelError):
            kernel.activate_extra_pp(activated.snapshot, "me")

    def test_extra_pp_is_consumed_only_when_paid(self):
        cards = {
            1: CardView(uid=1, card_id=101, base_card_id=101, name="普通", cost=6, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER),
            2: CardView(uid=2, card_id=102, base_card_id=102, name="额外", cost=1, card_type=CardType.SPELL),
            3: CardView(uid=3, card_id=103, base_card_id=103, name="满费", cost=7, atk=2, life=2, max_life=2, card_type=CardType.FOLLOWER),
        }
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=6, max_pp=6, hand=(1, 2, 3), extra_pp=ExtraPpView(extra_pp=1)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=6),
            cards_by_uid=cards,
            source="test",
        )
        kernel = CoreRuleKernel()

        activated = kernel.activate_extra_pp(snapshot, "me")
        normal_paid = kernel.play_card_from_hand(activated.snapshot, "me", 1, cost=6)
        self.assertEqual(normal_paid.snapshot.me.pp, 1)
        self.assertEqual(normal_paid.snapshot.me.extra_pp.extra_pp, 1)
        self.assertEqual(normal_paid.snapshot.me.extra_pp.preparation_extra_pp, 1)
        self.assertNotIn("consume_extra_pp", normal_paid.events)
        self.assertEqual(normal_paid.transaction_log.rollback(), activated.snapshot)

        extra_paid = kernel.play_spell_from_hand(normal_paid.snapshot, "me", 2, cost=1)
        self.assertEqual(extra_paid.snapshot.me.pp, 0)
        self.assertEqual(extra_paid.snapshot.me.extra_pp.extra_pp, 0)
        self.assertEqual(extra_paid.snapshot.me.extra_pp.preparation_extra_pp, 0)
        self.assertIn("consume_extra_pp", extra_paid.events)

        full_cost = kernel.play_card_from_hand(activated.snapshot, "me", 3, cost=7)
        self.assertEqual(full_cost.snapshot.me.pp, 0)
        self.assertEqual(full_cost.snapshot.me.extra_pp.extra_pp, 0)
        self.assertIn("consume_extra_pp", full_cost.events)

    def test_start_turn_resets_extra_pp_used_flag(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=0, max_pp=6, deck_count=1, extra_pp=ExtraPpView(extra_pp=1, preparation_extra_pp=1, extra_pp_state=1, restore_extra_pp_turn=8, is_used_extra_pp_this_turn=True)),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=6),
            cards_by_uid={},
            source="test",
            is_turn_ended=True,
        )
        result = CoreRuleKernel().start_turn(snapshot, "me", None)

        self.assertFalse(result.snapshot.me.extra_pp.is_used_extra_pp_this_turn)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_activate_card_and_record_fusion(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[3] = replace(cards[3], has_activation=True)
        cards[6] = CardView(uid=6, card_id=106, base_card_id=106, name="融合源", cost=2)
        cards[7] = CardView(uid=7, card_id=107, base_card_id=107, name="素材A", cost=1)
        cards[8] = CardView(uid=8, card_id=108, base_card_id=108, name="素材B", cost=1)
        fusion_snapshot = BattleSnapshot(
            me=replace(snapshot.me, hand=(6, 7, 8), field=(3,)),
            opponent=snapshot.opponent,
            cards_by_uid=cards,
            source="test",
        )
        kernel = CoreRuleKernel()

        activated = kernel.activate_card(fusion_snapshot, "me", 3)
        self.assertEqual(activated.events, ("activate",))
        # activate 后标记 activation_used_this_turn=True + has_activation=False
        self.assertTrue(activated.snapshot.cards_by_uid[3].activation_used_this_turn)
        self.assertFalse(activated.snapshot.cards_by_uid[3].has_activation)
        self.assertEqual(activated.transaction_log.rollback(), fusion_snapshot)

        fused = kernel.record_fusion(fusion_snapshot, "me", 6, (7, 8))
        self.assertEqual(fused.snapshot.me.hand, (6,))
        self.assertNotIn(7, fused.snapshot.cards_by_uid)
        self.assertNotIn(8, fused.snapshot.cards_by_uid)
        self.assertEqual(fused.snapshot.me.graveyard_count, snapshot.me.graveyard_count + 2)
        self.assertEqual(fused.snapshot.cards_by_uid[6].supplement_info["fusion_count"], 2)
        self.assertEqual(fused.events, ("fusion",))
        self.assertEqual(fused.transaction_log.rollback(), fusion_snapshot)

        self.assertEqual(fused.snapshot.cards_by_uid[6].supplement_info["fusion_used_this_turn"], 1)
        with self.assertRaises(RuleKernelError):
            kernel.record_fusion(fusion_snapshot, "me", 6, ())
        with self.assertRaises(RuleKernelError):
            kernel.record_fusion(fusion_snapshot, "me", 6, (7, 7))
        with self.assertRaises(RuleKernelError):
            kernel.record_fusion(fused.snapshot, "me", 6, (7,))
        reset = kernel.start_turn(replace(fused.snapshot, is_turn_ended=True), "me", None)
        self.assertNotIn("fusion_used_this_turn", reset.snapshot.cards_by_uid[6].supplement_info)

    def test_special_action_emits_event_without_state_change(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[3] = replace(cards[3], can_special_action=True)
        snapshot = BattleSnapshot(me=snapshot.me, opponent=snapshot.opponent, cards_by_uid=cards, source="test")
        kernel = CoreRuleKernel()

        result = kernel.perform_special_action(snapshot, "me", 3)

        self.assertEqual(result.snapshot, snapshot)
        self.assertEqual(result.events, ("special_action",))
        with self.assertRaises(RuleKernelError):
            kernel.perform_special_action(snapshot, "me", 4)

    def test_manual_evolve_and_super_evolve_events(self):
        snapshot = self.make_snapshot()
        kernel = CoreRuleKernel()

        evolved = kernel.manual_evolve(snapshot, "me", 3)
        self.assertEqual(evolved.snapshot.me.ep, 0)
        self.assertEqual(evolved.snapshot.cards_by_uid[3].evolve_state, 1)
        self.assertEqual(evolved.snapshot.cards_by_uid[3].atk, 5)
        self.assertEqual(evolved.snapshot.cards_by_uid[3].life, 6)
        self.assertEqual(evolved.events, ("manual_evolve", "becomes_evolved", "on_source_buffed"))
        self.assertEqual(evolved.transaction_log.rollback(), snapshot)

        super_evolved = kernel.manual_super_evolve(snapshot, "me", 3)
        self.assertEqual(super_evolved.snapshot.me.sep, 0)
        self.assertEqual(super_evolved.snapshot.cards_by_uid[3].evolve_state, 2)
        self.assertEqual(super_evolved.snapshot.cards_by_uid[3].atk, 6)
        self.assertEqual(super_evolved.events, ("manual_evolve", "manual_super_evolve", "becomes_evolved", "becomes_super_evolved", "on_source_buffed"))
        self.assertEqual(super_evolved.transaction_log.rollback(), snapshot)

    def test_evolve_does_not_grant_attack_window_to_cant_attack_follower(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[3] = replace(cards[3], attacks_left=0, can_attack=False, can_attack_leader=False, has_cant_attack=True)
        snapshot = replace(snapshot, cards_by_uid=cards)
        kernel = CoreRuleKernel()

        evolved = kernel.manual_evolve(snapshot, "me", 3)
        super_evolved = kernel.manual_super_evolve(snapshot, "me", 3)

        self.assertEqual(evolved.snapshot.cards_by_uid[3].attacks_left, 0)
        self.assertFalse(evolved.snapshot.cards_by_uid[3].can_attack)
        self.assertFalse(evolved.snapshot.cards_by_uid[3].can_attack_leader)
        self.assertEqual(super_evolved.snapshot.cards_by_uid[3].attacks_left, 0)
        self.assertFalse(super_evolved.snapshot.cards_by_uid[3].can_attack)
        self.assertFalse(super_evolved.snapshot.cards_by_uid[3].can_attack_leader)

    def test_evolve_grants_rush_only_when_attack_count_remains(self):
        snapshot = self.make_snapshot()
        cards = dict(snapshot.cards_by_uid)
        cards[3] = replace(cards[3], attacks_left=1, can_attack=False, can_attack_leader=False, has_rush=False, has_storm=False)
        snapshot = replace(snapshot, cards_by_uid=cards)
        kernel = CoreRuleKernel()

        evolved = kernel.manual_evolve(snapshot, "me", 3)
        super_evolved = kernel.manual_super_evolve(snapshot, "me", 3)

        self.assertEqual(evolved.snapshot.cards_by_uid[3].attacks_left, 1)
        self.assertTrue(evolved.snapshot.cards_by_uid[3].can_attack)
        self.assertFalse(evolved.snapshot.cards_by_uid[3].can_attack_leader)
        self.assertEqual(super_evolved.snapshot.cards_by_uid[3].attacks_left, 1)
        self.assertTrue(super_evolved.snapshot.cards_by_uid[3].can_attack)
        self.assertFalse(super_evolved.snapshot.cards_by_uid[3].can_attack_leader)

    def test_attack_leader_and_follower(self):
        snapshot = self.make_snapshot()
        kernel = CoreRuleKernel()

        leader = kernel.attack_leader(snapshot, 3, "opponent")
        self.assertEqual(leader.snapshot.opponent.life, 15)
        self.assertEqual(leader.snapshot.cards_by_uid[3].attacks_left, 0)
        self.assertEqual(leader.transaction_log.rollback(), snapshot)

        follower = kernel.attack_follower(snapshot, 3, 4)
        self.assertEqual(follower.snapshot.cards_by_uid[3].life, 2)
        self.assertNotIn(4, follower.snapshot.cards_by_uid)
        self.assertEqual(follower.snapshot.opponent.field, ())
        self.assertEqual(follower.snapshot.opponent.graveyard_count, 1)
        self.assertEqual(follower.snapshot.opponent.destroyed_card_ids[0].card_id, 104)
        self.assertEqual(follower.snapshot.opponent.destroyed_card_ids[0].count, 0)
        self.assertIn("destroyed_by_damage", follower.events)
        last_words = next(event for event in follower.events if event == "last_words")
        self.assertEqual(last_words.payload["leaving_card"], {"card_id": 104, "base_card_id": 104, "name": "敌方场上", "card_type": "follower"})
        self.assertEqual(follower.transaction_log.rollback(), snapshot)

    def test_attack_leader_drain_and_ambush(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=10, max_life=20, pp=1, max_pp=1, field=(3,)),
            opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1),
            cards_by_uid={3: CardView(uid=3, card_id=103, base_card_id=103, name="吸血潜行", atk=4, life=4, max_life=4, attacks_left=1, has_drain=True, has_ambush=True)},
            source="test",
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()

        result = kernel.attack_leader(snapshot, 3, "opponent")

        self.assertEqual(result.snapshot.opponent.life, 16)
        self.assertEqual(result.snapshot.me.life, 14)
        self.assertFalse(result.snapshot.cards_by_uid[3].has_ambush)
        self.assertIn("heal", result.events)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_attack_follower_temp_shield_and_bane(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(3,)),
            opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(4,)),
            cards_by_uid={
                3: CardView(uid=3, card_id=103, base_card_id=103, name="毁灭", atk=2, life=4, max_life=4, attacks_left=1, has_bane=True),
                4: CardView(uid=4, card_id=104, base_card_id=104, name="屏障", atk=1, life=5, max_life=5, has_temp_shield=True),
            },
            source="test",
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()

        result = kernel.attack_follower(snapshot, 3, 4)

        self.assertNotIn(4, result.snapshot.cards_by_uid)
        self.assertEqual(result.snapshot.cards_by_uid[3].life, 3)
        self.assertEqual(result.snapshot.opponent.destroyed_card_ids[0].card_id, 104)
        self.assertIn("temp_shield", result.events)
        self.assertIn("bane", result.events)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_bane_does_not_destroy_cant_destroy_follower(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(3,)),
            opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(4,)),
            cards_by_uid={
                3: CardView(uid=3, card_id=103, base_card_id=103, name="毁灭", atk=1, life=4, max_life=4, attacks_left=1, has_bane=True),
                4: CardView(uid=4, card_id=104, base_card_id=104, name="不被能力破坏", atk=1, life=5, max_life=5, has_cant_destroy=True),
            },
            source="test",
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()

        result = kernel.attack_follower(snapshot, 3, 4)

        self.assertIn(4, result.snapshot.cards_by_uid)
        self.assertEqual(result.snapshot.cards_by_uid[4].life, 4)
        self.assertNotIn("bane", result.events)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_evolve_grants_follower_attack_window_without_leader_attack(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, ep=1, sep=1, field=(3, 4)),
            opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1),
            cards_by_uid={
                3: CardView(uid=3, card_id=103, base_card_id=103, name="进化", atk=2, life=3, max_life=3, attack_limit=1),
                4: CardView(uid=4, card_id=104, base_card_id=104, name="超进化", atk=2, life=3, max_life=3, attack_limit=1),
            },
            source="test",
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()

        evolved = kernel.manual_evolve(snapshot, "me", 3)
        super_evolved = kernel.manual_super_evolve(snapshot, "me", 4)

        self.assertEqual(evolved.snapshot.cards_by_uid[3].attacks_left, 1)
        self.assertTrue(evolved.snapshot.cards_by_uid[3].can_attack)
        self.assertFalse(evolved.snapshot.cards_by_uid[3].can_attack_leader)
        self.assertEqual(super_evolved.snapshot.cards_by_uid[4].attacks_left, 1)
        self.assertTrue(super_evolved.snapshot.cards_by_uid[4].can_attack)
        self.assertFalse(super_evolved.snapshot.cards_by_uid[4].can_attack_leader)

    def test_super_evolved_follower_takes_no_combat_damage_on_own_turn(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(3,)),
            opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(4,)),
            cards_by_uid={
                3: CardView(uid=3, card_id=103, base_card_id=103, name="超进化", atk=3, life=6, max_life=6, attacks_left=1, evolve_state=2),
                4: CardView(uid=4, card_id=104, base_card_id=104, name="目标", atk=9, life=3, max_life=3),
            },
            source="test",
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()

        result = kernel.attack_follower(snapshot, 3, 4)

        self.assertEqual(result.snapshot.cards_by_uid[3].life, 6)
        self.assertNotIn(4, result.snapshot.cards_by_uid)

    def test_combat_damage_respects_card_damage_cut(self):
        """回归：DamageCut（"受到的4点或以上的伤害变为3点"，如威严的星晶骑士·薇拉）
        在战斗伤害中减免 4 点及以上的攻击。"""
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(3,)),
            opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(4,)),
            cards_by_uid={
                3: CardView(uid=3, card_id=103, base_card_id=103, name="攻击者", atk=6, life=8, max_life=8, attacks_left=1),
                4: CardView(uid=4, card_id=10464120, base_card_id=10464120, name="薇拉", atk=1, life=8, max_life=8, has_damage_cut=True),
            },
            source="test",
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()

        result = kernel.attack_follower(snapshot, 3, 4)

        # 6 点攻击 >= 4，减为 3 点 -> 薇拉 8 - 3 = 5；薇拉 1 点反击 < 4 不减免
        self.assertEqual(result.snapshot.cards_by_uid[4].life, 5)
        self.assertEqual(result.snapshot.cards_by_uid[3].life, 7)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_super_evolved_follower_cannot_be_destroyed_by_ability_on_own_turn(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(3,)),
            opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1),
            cards_by_uid={3: CardView(uid=3, card_id=103, base_card_id=103, name="超进化", atk=3, life=6, max_life=6, evolve_state=2)},
            source="test",
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()

        result = kernel.destroy_field_card(snapshot, "me", 3)

        self.assertIn(3, result.snapshot.cards_by_uid)
        self.assertEqual(result.snapshot.me.field, (3,))
        self.assertEqual(result.events, ())

    def test_super_evolve_destroy_deals_one_to_leader(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(3,)),
            opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1, field=(4,)),
            cards_by_uid={
                3: CardView(uid=3, card_id=103, base_card_id=103, name="超进化", atk=3, life=6, max_life=6, attacks_left=1, evolve_state=2),
                4: CardView(uid=4, card_id=104, base_card_id=104, name="目标", atk=1, life=3, max_life=3),
            },
            source="test",
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()

        result = kernel.attack_follower(snapshot, 3, 4)

        self.assertEqual(result.snapshot.opponent.life, 19)
        self.assertNotIn(4, result.snapshot.cards_by_uid)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_leader_temp_shield_blocks_damage_and_drain(self):
        snapshot = BattleSnapshot(
            me=PlayerView(life=10, max_life=20, pp=1, max_pp=1, field=(3,)),
            opponent=PlayerView(life=20, max_life=20, pp=1, max_pp=1, buff=PlayerBuffView(temp_shield=True)),
            cards_by_uid={3: CardView(uid=3, card_id=103, base_card_id=103, name="吸血", atk=4, life=4, max_life=4, attacks_left=1, has_drain=True)},
            source="test",
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()

        result = kernel.attack_leader(snapshot, 3, "opponent")

        self.assertEqual(result.snapshot.opponent.life, 20)
        self.assertFalse(result.snapshot.opponent.buff.temp_shield)
        self.assertEqual(result.snapshot.me.life, 10)
        self.assertIn("temp_shield", result.events)
        self.assertNotIn("heal", result.events)
        self.assertEqual(result.transaction_log.rollback(), snapshot)


    def test_auto_evolve_increments_skybound_art_count(self):
        """auto_evolve 应当和 manual_evolve 一样递增手牌中奥义计数。"""
        cards = {
            3: CardView(uid=3, card_id=103, base_card_id=103, name="场上随从", atk=2, life=4, max_life=4, attacks_left=0),
            10: CardView(uid=10, card_id=110, base_card_id=110, name="奥义手牌", cost=4, supplement_info={"skybound_art_count": 7}),
        }
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=3, max_pp=6, ep=1, deck_count=5, hand=(10,), field=(3,)),
            opponent=PlayerView(life=20, max_life=20, pp=5, max_pp=5),
            cards_by_uid=cards,
            source="test",
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()
        result = kernel.auto_evolve(snapshot, "me", 3)
        self.assertEqual(result.snapshot.cards_by_uid[3].evolve_state, 1)
        self.assertEqual(result.snapshot.cards_by_uid[10].supplement_info["skybound_art_count"], 8)
        self.assertEqual(result.transaction_log.rollback(), snapshot)

    def test_manual_super_evolve_increments_skybound_art_count(self):
        """manual_super_evolve 应当递增手牌中奥义计数。"""
        cards = {
            3: CardView(uid=3, card_id=103, base_card_id=103, name="场上随从", atk=3, life=4, max_life=4, attacks_left=0),
            10: CardView(uid=10, card_id=110, base_card_id=110, name="奥义手牌", cost=4, supplement_info={"skybound_art_count": 5}),
        }
        snapshot = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=3, max_pp=6, sep=1, deck_count=5, hand=(10,), field=(3,)),
            opponent=PlayerView(life=20, max_life=20, pp=5, max_pp=5),
            cards_by_uid=cards,
            source="test",
            is_ally_turn=True,
        )
        kernel = CoreRuleKernel()
        result = kernel.manual_super_evolve(snapshot, "me", 3)
        self.assertEqual(result.snapshot.cards_by_uid[3].evolve_state, 2)
        self.assertEqual(result.snapshot.cards_by_uid[10].supplement_info["skybound_art_count"], 6)
        self.assertEqual(result.transaction_log.rollback(), snapshot)


class ConsumePpResourceTest(unittest.TestCase):
    """consume_pp（供效果内 pay_pp op 使用）的资源不足语义。

    背景：部分卡牌效果（如"融合时消耗 N 点能量点"）在触发时才付 PP。
    PP 不足是正常游戏语义（效果不发动），必须抛可识别的
    InsufficientResourceError（RuleKernelError 子类），供 effect 层回滚并
    降级为"效果未发动"，而不是上报 predict_error。
    """

    def _snapshot(self, pp: int) -> BattleSnapshot:
        return BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=pp, max_pp=7, deck_count=1),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=7),
            cards_by_uid={},
        )

    def test_consume_pp_enough_pays_and_updates_pp(self):
        from decision_engine.execution import CoreRuleKernel
        kernel = CoreRuleKernel()
        result = kernel.consume_pp(self._snapshot(pp=5), "me", 2)
        self.assertEqual(result.snapshot.me.pp, 3)

    def test_consume_pp_insufficient_raises_insufficient_resource_error(self):
        from decision_engine.execution import CoreRuleKernel, InsufficientResourceError, RuleKernelError
        kernel = CoreRuleKernel()
        with self.assertRaises(InsufficientResourceError):
            kernel.consume_pp(self._snapshot(pp=1), "me", 2)
        # 必须是 RuleKernelError 子类，保证既有捕获 RuleKernelError 的调用方行为不变
        self.assertTrue(issubclass(InsufficientResourceError, RuleKernelError))

    def test_consume_pp_negative_amount_is_generic_error_not_resource(self):
        from decision_engine.execution import CoreRuleKernel, InsufficientResourceError, RuleKernelError
        kernel = CoreRuleKernel()
        # 负数成本是编程错误，应是通用 RuleKernelError 而非资源不足语义
        with self.assertRaises(RuleKernelError) as ctx:
            kernel.consume_pp(self._snapshot(pp=5), "me", -1)
        self.assertNotIsInstance(ctx.exception, InsufficientResourceError)


if __name__ == "__main__":
    unittest.main()
