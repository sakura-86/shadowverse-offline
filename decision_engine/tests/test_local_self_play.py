# -*- coding: utf-8 -*-
"""Local self-play tests."""

import unittest
from dataclasses import replace

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.cards import BaseCardInfo, CardInfo, CardResolver, MasterDataProvider
from decision_engine.cards.templates import AmountFormula, CardIdentity, ExecutableEffect, OperationSpec, PrintedStats, TriggerSpec, UnifiedCardTemplate
from decision_engine.execution import (
    EndTurnSelector,
    LocalDeckList,
    LocalMatchSetup,
    LocalMatchSetupConfig,
    LocalPredictor,
    LocalSelfPlayConfig,
    LocalSelfPlayRunner,
    PlayerLocalPredictor,
)
from decision_engine.execution.local_match_setup import draw_next_from_deck_instances
from decision_engine.simulation.predictor import PredictionCertainty, PredictionResult
from decision_engine.state.snapshot import BattleSnapshot, CardType, CardView, PlayerView


class BadActionSelector:
    def select(self, _snapshot, _actions, *, player):
        return LegalAction(kind=ActionKind.FUSION, source_uid=999)


class TraceSelector:
    def __init__(self):
        self.selector = self
        self.last_selected_action_trace = None
        self.last_plan_trace = None
        self.last_selection_reason = ""

    def select(self, _snapshot, actions, *, player):
        action = actions[0]
        self.last_selected_action_trace = type("Trace", (), {"selection_rank": 1, "is_selected": True, "action_kind": action.kind.value})()
        depth = type("Depth", (), {"top_actions": (self.last_selected_action_trace,), "all_actions": (self.last_selected_action_trace,)})()
        self.last_plan_trace = type("Plan", (), {"depths": (depth,)})()
        self.last_selection_reason = "test trace selector"
        return action


class PlaySourceSelector:
    def __init__(self, source_uid: int):
        self.source_uid = source_uid

    def select(self, _snapshot, actions, *, player):
        for action in actions:
            if action.kind is ActionKind.PLAY and action.source_uid == self.source_uid:
                return action
        return None


def _template(base_card_id: int, effect: ExecutableEffect) -> UnifiedCardTemplate:
    return UnifiedCardTemplate(
        template_version=1,
        identity=CardIdentity(base_card_id=base_card_id, name=f"card-{base_card_id}", type_id=1, type_name="follower", class_id=0, class_name="test", rarity=0, rarity_name="test"),
        printed=PrintedStats(cost=1, atk=1, life=1),
        executable_effects=(effect,),
    )


class LocalSelfPlayTest(unittest.TestCase):
    def make_resolver(self):
        base_cards = {
            101: BaseCardInfo(base_card_id=101, name="一费随从", type_id=1, card_type="follower", cost=1, atk=1, life=1),
            102: BaseCardInfo(base_card_id=102, name="二费随从", type_id=1, card_type="follower", cost=2, atk=2, life=2),
            201: BaseCardInfo(base_card_id=201, name="一费法术", type_id=4, card_type="spell", cost=1),
            301: BaseCardInfo(base_card_id=301, name="护符", type_id=2, card_type="amulet", cost=1),
        }
        cards = {card_id: CardInfo(card_id=card_id, base_card_id=card_id) for card_id in base_cards}
        return CardResolver(MasterDataProvider(base_cards=base_cards, cards=cards))

    def setup(self, *, seed=1, first_player="me", initial_hand_first=1, initial_hand_second=1):
        return LocalMatchSetup(self.make_resolver()).create(
            LocalDeckList(name="A", class_name="test", card_ids=(101, 102, 201, 301, 101)),
            LocalDeckList(name="B", class_name="test", card_ids=(101, 102, 301, 201, 102)),
            LocalMatchSetupConfig(
                seed=seed,
                first_player=first_player,
                initial_hand_first=initial_hand_first,
                initial_hand_second=initial_hand_second,
            ),
        )

    def test_local_match_setup_creates_two_decks(self):
        result = self.setup()

        self.assertEqual(result.errors, ())
        self.assertIsNotNone(result.snapshot)
        snapshot = result.snapshot
        assert snapshot is not None
        self.assertEqual(snapshot.source, "local_self_play")
        self.assertEqual(len(snapshot.me.hand), 1)
        self.assertEqual(len(snapshot.opponent.hand), 1)
        self.assertEqual(snapshot.me.deck_count, 4)
        self.assertEqual(snapshot.opponent.deck_count, 4)
        self.assertIsNotNone(snapshot.me.deck_instances)
        self.assertIsNotNone(snapshot.opponent.deck_instances)

    def test_local_match_setup_assigns_unique_uids(self):
        result = self.setup(initial_hand_first=2, initial_hand_second=2)
        snapshot = result.snapshot
        assert snapshot is not None

        self.assertEqual(len(snapshot.cards_by_uid), 10)
        self.assertEqual(len(set(snapshot.cards_by_uid)), 10)
        self.assertTrue(set(snapshot.me.hand).isdisjoint(snapshot.opponent.hand))

    def test_local_match_setup_materializes_activation_from_templates(self):
        result = LocalMatchSetup().create(
            LocalDeckList(name="A", class_name="bishop", card_ids=(10161210,)),
            LocalDeckList(name="B", class_name="bishop", card_ids=(10161210,)),
            LocalMatchSetupConfig(seed=1, first_player="me", initial_hand_first=1, initial_hand_second=0),
        )
        snapshot = result.snapshot
        assert snapshot is not None

        card = snapshot.cards_by_uid[snapshot.me.hand[0]]
        self.assertEqual(card.base_card_id, 10161210)
        self.assertTrue(card.has_activation)
        self.assertTrue(card.has_last_words)
        self.assertEqual(card.countdown, 3)

    def test_local_match_setup_is_seed_deterministic(self):
        one = self.setup(seed=7).snapshot
        two = self.setup(seed=7).snapshot
        three = self.setup(seed=8).snapshot
        assert one is not None and two is not None and three is not None

        self.assertEqual(one.me.hand, two.me.hand)
        self.assertEqual(one.opponent.hand, two.opponent.hand)
        self.assertNotEqual((one.me.hand, one.opponent.hand), (three.me.hand, three.opponent.hand))

    def test_local_match_setup_errors_on_unknown_card(self):
        result = LocalMatchSetup(self.make_resolver()).create(
            LocalDeckList(name="A", card_ids=(999,)),
            LocalDeckList(name="B", card_ids=(101,)),
        )

        self.assertIsNone(result.snapshot)
        self.assertIn("me deck A card_id not found: 999", result.errors)

    def test_turn_start_draws_from_player_deck_instances(self):
        result = self.setup(initial_hand_first=0, initial_hand_second=0)
        snapshot = result.snapshot
        assert snapshot is not None
        before_pile = snapshot.me.deck_instances.draw_pile

        drawn, after = draw_next_from_deck_instances(snapshot, "me")

        self.assertIsNotNone(drawn)
        assert drawn is not None
        self.assertEqual(after.me.deck_instances.hand, (before_pile[0],))
        self.assertEqual(after.me.deck_instances.draw_pile, before_pile[1:])
        self.assertEqual(after.cards_by_uid[drawn.uid].card_id, drawn.card_id)

    def test_self_play_two_end_turn_selectors_runs_until_deckout(self):
        result = self.setup(initial_hand_first=0, initial_hand_second=0)
        snapshot = result.snapshot
        assert snapshot is not None
        runner = LocalSelfPlayRunner(predictor=PlayerLocalPredictor(LocalPredictor()), config=LocalSelfPlayConfig(max_actions_per_turn=3, stop_on_partial=False))

        played = runner.run(snapshot, me_selector=EndTurnSelector(), opponent_selector=EndTurnSelector())

        # 没有回合上限：双方都只结束回合，牌库抽空后由 deckout 结果码自然终局。
        self.assertTrue(played.reason.startswith("battle_result:"), played.reason)
        self.assertIsNotNone(played.winner)
        self.assertGreaterEqual(len([step for step in played.trace.steps if step.phase == "turn_start"]), 5)
        self.assertGreaterEqual(len([step for step in played.trace.steps if step.selected_action and step.selected_action.kind is ActionKind.TURN_END]), 5)
        self.assertIn("spec_only", played.trace.certainty_counts)

    def test_self_play_stops_on_predict_error(self):
        result = self.setup(initial_hand_first=1, initial_hand_second=0)
        snapshot = result.snapshot
        assert snapshot is not None
        runner = LocalSelfPlayRunner(predictor=PlayerLocalPredictor(LocalPredictor()), config=LocalSelfPlayConfig(max_actions_per_turn=3, stop_on_partial=False))

        played = runner.run(snapshot, me_selector=BadActionSelector(), opponent_selector=EndTurnSelector())

        self.assertIn("fusion source is not in hand", played.reason)
        self.assertEqual(played.trace.certainty_counts.get("predict_error"), 1)

    def test_self_play_records_spec_only_turn_progression(self):
        result = self.setup(initial_hand_first=0, initial_hand_second=0)
        snapshot = result.snapshot
        assert snapshot is not None
        runner = LocalSelfPlayRunner(predictor=PlayerLocalPredictor(LocalPredictor()), config=LocalSelfPlayConfig(max_actions_per_turn=2, stop_on_partial=False))

        played = runner.run(snapshot, me_selector=EndTurnSelector(), opponent_selector=EndTurnSelector())

        self.assertGreater(played.trace.certainty_counts.get("spec_only", 0), 0)
        self.assertEqual(played.trace.unresolved_counts.get("turn_end_effects", 0), 0)

    def test_self_play_records_selector_score_traces_when_available(self):
        result = self.setup(initial_hand_first=1, initial_hand_second=0)
        snapshot = result.snapshot
        assert snapshot is not None
        runner = LocalSelfPlayRunner(predictor=PlayerLocalPredictor(LocalPredictor()), config=LocalSelfPlayConfig(max_actions_per_turn=1, stop_on_partial=False))

        played = runner.run(snapshot, me_selector=TraceSelector(), opponent_selector=EndTurnSelector())

        action_step = next(step for step in played.trace.steps if step.phase == "action")
        self.assertIsNotNone(action_step.selected_action_score_trace)
        self.assertEqual(action_step.selected_action_rank, 1)
        self.assertEqual(len(action_step.planner_top_action_traces), 1)
        self.assertEqual(len(action_step.planner_all_action_traces), 1)
        self.assertEqual(action_step.selection_reason, "test trace selector")

    def test_self_play_random_source_pool_branch_expansion_is_opt_in(self):
        source = CardView(uid=1, card_id=1005, base_card_id=1005, name="随机召唤者", cost=1, atk=1, life=1, max_life=1, card_type=CardType.FOLLOWER)
        deck_cards = (
            CardView(uid=-1, card_id=106, base_card_id=1006, name="牌组随从A", cost=1, atk=1, life=1, max_life=1, card_type=CardType.FOLLOWER),
            CardView(uid=-2, card_id=107, base_card_id=1007, name="牌组随从B", cost=1, atk=1, life=1, max_life=1, card_type=CardType.FOLLOWER),
            CardView(uid=-3, card_id=108, base_card_id=1008, name="牌组随从C", cost=1, atk=1, life=1, max_life=1, card_type=CardType.FOLLOWER),
        )
        setup = self.setup(initial_hand_first=0, initial_hand_second=0)
        base = setup.snapshot
        assert base is not None
        deck_instances = base.me.deck_instances
        assert deck_instances is not None
        from decision_engine.state.deck_instance import make_deck_instance_state

        snapshot = BattleSnapshot(
            me=replace(base.me, pp=1, max_pp=1, hand=(1,), deck_instances=make_deck_instance_state(deck_cards, source="test_deck")),
            opponent=base.opponent,
            cards_by_uid={1: source},
            source="test",
            is_ally_turn=True,
        )
        effect = ExecutableEffect(
            effect_id="1005_random_summon_deck",
            trigger=TriggerSpec(event="fanfare"),
            operations=(OperationSpec(op="summon", target_ref="self_field", amount=AmountFormula(kind="fixed", value=1), params={"source_pool": {"selection": "random", "side": "self", "zone": "deck", "count": 1, "card_type": "follower"}}),),
        )
        predictor = PlayerLocalPredictor(LocalPredictor(templates_by_base_card_id={1005: _template(1005, effect)}))

        default = LocalSelfPlayRunner(predictor=predictor, config=LocalSelfPlayConfig(max_actions_per_turn=1, stop_on_partial=False)).run(
            snapshot,
            me_selector=PlaySourceSelector(1),
            opponent_selector=EndTurnSelector(),
        )
        branched = LocalSelfPlayRunner(predictor=predictor, config=LocalSelfPlayConfig(max_actions_per_turn=1, stop_on_partial=False, expand_random_source_pool_branches=True, branch_seed=7)).run(
            snapshot,
            me_selector=PlaySourceSelector(1),
            opponent_selector=EndTurnSelector(),
        )

        default_step = next(step for step in default.trace.steps if step.phase == "action")
        branch_step = next(step for step in branched.trace.steps if step.phase == "action")
        # 默认配置已开启 source_pool 随机分支展开
        self.assertEqual(default_step.prediction_certainty, "spec_only")
        self.assertTrue(any(item.startswith("random_branch:source_pool") for item in default_step.assumptions))
        self.assertEqual(branch_step.prediction_certainty, "spec_only")
        self.assertTrue(any(item.startswith("random_branch:source_pool") for item in branch_step.assumptions))


if __name__ == "__main__":
    unittest.main()
