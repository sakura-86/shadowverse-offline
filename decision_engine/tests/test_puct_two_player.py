# -*- coding: utf-8 -*-
"""Tests for two-player multi-turn PUCT tree search."""

import unittest
from dataclasses import dataclass, replace
from typing import Any

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.execution.rule_kernel import BATTLE_RESULT_DECK_OUT_LOSE, BATTLE_RESULT_DECK_OUT_WIN
from decision_engine.planning.puct import (
    DefaultTurnTransitioner,
    PuctConfig,
    PuctSearch,
    TurnTransitioner,
)
from decision_engine.planning.search_value import BoardAdvantageValue, ValueResult
from decision_engine.state.snapshot import (
    BattleSnapshot,
    CardType,
    CardView,
    LegalView,
    PlayerView,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _action(kind: str, uid: int = 0) -> LegalAction:
    return LegalAction(kind=ActionKind(kind), source_uid=uid)


def _follower(uid: int, atk: int, life: int, **kwargs) -> CardView:
    return CardView(uid=uid, card_id=uid * 100, base_card_id=uid * 100, atk=atk, life=life, max_life=life, card_type=CardType.FOLLOWER, **kwargs)


def _snapshot(
    *,
    my_life: int = 20,
    opp_life: int = 20,
    my_pp: int = 3,
    opp_pp: int = 3,
    my_field: tuple[CardView, ...] = (),
    opp_field: tuple[CardView, ...] = (),
    my_hand: tuple[int, ...] = (),
    my_deck: int = 10,
    opp_deck: int = 10,
    is_ally_turn: bool = True,
) -> BattleSnapshot:
    cards: dict[int, CardView] = {}
    my_uids: list[int] = []
    opp_uids: list[int] = []
    for c in my_field:
        cards[c.uid] = c
        my_uids.append(c.uid)
    for c in opp_field:
        cards[c.uid] = c
        opp_uids.append(c.uid)
    for uid in my_hand:
        if uid not in cards:
            cards[uid] = CardView(uid=uid, card_id=uid * 100, base_card_id=uid * 100, cost=2, card_type=CardType.SPELL)
    return BattleSnapshot(
        me=PlayerView(life=my_life, max_life=20, pp=my_pp, max_pp=my_pp, hand=my_hand, field=tuple(my_uids), deck_count=my_deck, turn=3),
        opponent=PlayerView(life=opp_life, max_life=20, pp=opp_pp, max_pp=opp_pp, field=tuple(opp_uids), deck_count=opp_deck, turn=2),
        cards_by_uid=cards,
        is_ally_turn=is_ally_turn,
        legal=LegalView(),
    )


@dataclass
class _MockPrediction:
    snapshot: BattleSnapshot
    certainty: Any = None
    unresolved_effects: tuple = ()


class _TwoPlayerPredictor:
    """支持双方动作预测的 mock predictor。

    attack_leader: 减对方 life（按 cards_by_uid[source_uid].atk）
    turn_end: 设 is_turn_ended=True
    play: 移入场上（简化）
    """

    def predict(self, snapshot: BattleSnapshot, action: LegalAction, *, player: str = "me"):
        kind = action.kind.value if hasattr(action.kind, "value") else str(action.kind)

        if kind == "turn_end":
            result = replace(snapshot, is_turn_ended=True)
            return _MockPrediction(snapshot=result)

        if kind == "attack_leader":
            card = snapshot.cards_by_uid.get(action.source_uid)
            damage = card.atk if card else 0
            # 攻击的是 opponent（从当前视角看）
            new_opp = replace(snapshot.opponent, life=snapshot.opponent.life - damage)
            # 减少 attacks_left
            if card:
                new_cards = {**snapshot.cards_by_uid, card.uid: replace(card, attacks_left=card.attacks_left - 1)}
            else:
                new_cards = snapshot.cards_by_uid
            result = replace(snapshot, opponent=new_opp, cards_by_uid=new_cards)
            return _MockPrediction(snapshot=result)

        if kind == "play":
            uid = action.source_uid
            card = snapshot.cards_by_uid.get(uid)
            if card:
                new_hand = tuple(u for u in snapshot.me.hand if u != uid)
                new_field = (*snapshot.me.field, uid)
                new_me = replace(snapshot.me, hand=new_hand, field=new_field, pp=snapshot.me.pp - (card.cost or 0))
                # 刚出的随从不能攻击（无疾驰）
                new_cards = {**snapshot.cards_by_uid, uid: replace(card, can_attack=False, attacks_left=0)}
                result = replace(snapshot, me=new_me, cards_by_uid=new_cards)
                return _MockPrediction(snapshot=result)
            return _MockPrediction(snapshot=snapshot)

        return _MockPrediction(snapshot=snapshot)


class _SimpleTurnTransitioner:
    """简化的回合切换：交换 is_ally_turn，恢复 PP，刷新场面随从攻击能力。"""

    def transition_turn(self, snapshot: BattleSnapshot, ending_player: str) -> BattleSnapshot:
        next_player = "opponent" if ending_player == "me" else "me"
        # 交换 active turn
        is_ally = (next_player == "me")

        # 刷新下一位玩家的场上随从 can_attack
        if is_ally:
            next_view = snapshot.me
        else:
            next_view = snapshot.opponent

        new_cards = dict(snapshot.cards_by_uid)
        for uid in next_view.field:
            card = new_cards.get(uid)
            if card and card.card_type is CardType.FOLLOWER:
                new_cards[uid] = replace(card, can_attack=True, can_attack_leader=True, attacks_left=1)

        # 恢复 PP
        next_pp = min(10, next_view.max_pp + 1)
        if is_ally:
            new_me = replace(snapshot.me, pp=next_pp, max_pp=next_pp, turn=snapshot.me.turn + 1)
            new_snapshot = replace(snapshot, me=new_me, is_ally_turn=True, is_turn_ended=False, cards_by_uid=new_cards)
        else:
            new_opp = replace(snapshot.opponent, pp=next_pp, max_pp=next_pp, turn=snapshot.opponent.turn + 1)
            new_snapshot = replace(snapshot, opponent=new_opp, is_ally_turn=False, is_turn_ended=False, cards_by_uid=new_cards)

        return new_snapshot


class _TwoPlayerEnumerator:
    """支持双方枚举的 mock enumerator。"""

    def enumerate(self, snapshot: BattleSnapshot, *, player: str = "me") -> list[LegalAction]:
        if player == "me":
            view = snapshot.me
        else:
            view = snapshot.opponent

        actions: list[LegalAction] = []
        cards = snapshot.cards_by_uid

        # 枚举攻击
        for uid in view.field:
            card = cards.get(uid)
            if card and card.can_attack and card.attacks_left > 0 and card.can_attack_leader:
                actions.append(_action("attack_leader", uid=uid))

        # 枚举出牌（简化：只枚举 pp 够的手牌）
        for uid in view.hand:
            card = cards.get(uid)
            if card and (card.cost or 0) <= view.pp:
                actions.append(LegalAction(kind=ActionKind.PLAY, source_uid=uid, cost=card.cost))

        # 总是有 turn_end
        actions.append(_action("turn_end"))
        return actions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPuctTwoPlayer(unittest.TestCase):

    def test_turn_end_expands_opponent_turn_with_transitioner(self):
        """有 turn_transitioner 时，turn_end 后应展开对方回合。"""
        # 我方场上有 3/3（能打脸），对方空场
        my_follower = _follower(10, atk=3, life=3, can_attack=True, can_attack_leader=True, attacks_left=1)
        opp_follower = _follower(20, atk=5, life=5, can_attack=False, can_attack_leader=False, attacks_left=0)
        initial = _snapshot(my_field=(my_follower,), opp_field=(opp_follower,))

        search = PuctSearch(
            enumerator=_TwoPlayerEnumerator(),
            predictor=_TwoPlayerPredictor(),
            value=BoardAdvantageValue(),
            turn_transitioner=_SimpleTurnTransitioner(),
            config=PuctConfig(simulations=128, max_depth=10, max_turns=3, c_puct=1.5),
        )
        result = search.search(initial, player="me")

        # 搜索应该观察到跨回合展开
        self.assertGreater(result.diagnostics["cross_turn_count"], 0)
        self.assertEqual(result.diagnostics["max_turns"], 3)
        self.assertIsNotNone(result.selected_action_index)

    def test_opponent_attack_lowers_my_value(self):
        """对方有强力随从时，turn_end 的估值应该很低（对方会打脸）。"""
        # 我方空场，对方有 7/7 能打脸
        opp_attacker = _follower(20, atk=7, life=7, can_attack=False, can_attack_leader=False, attacks_left=0)
        initial = _snapshot(opp_field=(opp_attacker,), my_pp=0)

        search = PuctSearch(
            enumerator=_TwoPlayerEnumerator(),
            predictor=_TwoPlayerPredictor(),
            value=BoardAdvantageValue(),
            turn_transitioner=_SimpleTurnTransitioner(),
            config=PuctConfig(simulations=64, max_depth=10, max_turns=2, c_puct=1.5),
        )
        result = search.search(initial, player="me")

        # 只有 turn_end 可选（无法攻击），但 value 应该是负的（对方会打我 7 点）
        self.assertLess(result.root_value, 0.0)

    def test_max_turns_one_degrades_to_leaf(self):
        """max_turns=1 时 turn_end 仍是叶节点（旧行为）。"""
        my_follower = _follower(10, atk=3, life=3, can_attack=True, can_attack_leader=True, attacks_left=1)
        initial = _snapshot(my_field=(my_follower,))

        search = PuctSearch(
            enumerator=_TwoPlayerEnumerator(),
            predictor=_TwoPlayerPredictor(),
            value=BoardAdvantageValue(),
            turn_transitioner=_SimpleTurnTransitioner(),
            config=PuctConfig(simulations=32, max_depth=5, max_turns=1, c_puct=1.5),
        )
        result = search.search(initial, player="me")

        # 没有跨回合展开
        self.assertEqual(result.diagnostics["cross_turn_count"], 0)

    def test_no_transitioner_degrades_to_leaf(self):
        """没有 turn_transitioner 时，turn_end 退化为叶节点。"""
        my_follower = _follower(10, atk=3, life=3, can_attack=True, can_attack_leader=True, attacks_left=1)
        initial = _snapshot(my_field=(my_follower,))

        search = PuctSearch(
            enumerator=_TwoPlayerEnumerator(),
            predictor=_TwoPlayerPredictor(),
            value=BoardAdvantageValue(),
            config=PuctConfig(simulations=32, max_depth=5, max_turns=3, c_puct=1.5),
        )
        result = search.search(initial, player="me")

        # 没有 transitioner，不能跨回合
        self.assertEqual(result.diagnostics["cross_turn_count"], 0)

    def test_opponent_selects_best_action_for_itself(self):
        """对方应选择对我方最不利的动作（打脸而非 turn_end）。"""
        # 对方有 5/5 可打脸随从，搜索后 turn_end 的 value 应反映被打 5 点
        opp_attacker = _follower(20, atk=5, life=5, can_attack=False, can_attack_leader=False, attacks_left=0)
        initial = _snapshot(opp_field=(opp_attacker,), my_pp=0)

        search = PuctSearch(
            enumerator=_TwoPlayerEnumerator(),
            predictor=_TwoPlayerPredictor(),
            value=BoardAdvantageValue(),
            turn_transitioner=_SimpleTurnTransitioner(),
            config=PuctConfig(simulations=128, max_depth=10, max_turns=2, c_puct=1.5),
        )
        result = search.search(initial, player="me")

        # 对方有强随从会打脸，所以我方的 root_value 必然为负
        self.assertLess(result.root_value, 0.0)

    def test_multi_turn_depth_four(self):
        """max_turns=4 可以覆盖 我方->对方->我方->对方。"""
        my_follower = _follower(10, atk=2, life=2, can_attack=True, can_attack_leader=True, attacks_left=1)
        opp_follower = _follower(20, atk=2, life=2, can_attack=False, can_attack_leader=False, attacks_left=0)
        initial = _snapshot(my_field=(my_follower,), opp_field=(opp_follower,))

        search = PuctSearch(
            enumerator=_TwoPlayerEnumerator(),
            predictor=_TwoPlayerPredictor(),
            value=BoardAdvantageValue(),
            turn_transitioner=_SimpleTurnTransitioner(),
            config=PuctConfig(simulations=256, max_depth=20, max_turns=4, c_puct=1.5),
        )
        result = search.search(initial, player="me")

        # 应该有多次跨回合展开
        self.assertGreater(result.diagnostics["cross_turn_count"], 0)
        self.assertEqual(result.diagnostics["max_turns"], 4)
        self.assertIsNotNone(result.selected_action_index)

    def test_terminal_in_opponent_turn_detected(self):
        """对方回合中如果击杀我方，应该被检测为终局。"""
        # 我方 life=3，对方有 5/5 随从
        opp_attacker = _follower(20, atk=5, life=5, can_attack=False, can_attack_leader=False, attacks_left=0)
        initial = _snapshot(my_life=3, opp_field=(opp_attacker,), my_pp=0)

        search = PuctSearch(
            enumerator=_TwoPlayerEnumerator(),
            predictor=_TwoPlayerPredictor(),
            value=BoardAdvantageValue(),
            turn_transitioner=_SimpleTurnTransitioner(),
            config=PuctConfig(simulations=64, max_depth=10, max_turns=2, c_puct=1.5),
        )
        result = search.search(initial, player="me")

        # 对方能击杀我方，root_value 应接近 -1（败局）
        self.assertLess(result.root_value, -0.5)

    def test_default_turn_transitioner_works(self):
        """DefaultTurnTransitioner 使用真实 CoreRuleKernel。"""
        transitioner = DefaultTurnTransitioner()
        my_follower = _follower(10, atk=3, life=3, can_attack=True, can_attack_leader=True, attacks_left=1)
        initial = _snapshot(my_field=(my_follower,), is_ally_turn=True)
        # 模拟 turn_end 后
        ended = replace(initial, is_turn_ended=True, is_ally_turn=True)

        result = transitioner.transition_turn(ended, "me")

        # 应切换到对方回合
        self.assertFalse(result.is_ally_turn)
        self.assertFalse(result.is_turn_ended)
        # 对方 PP 应该恢复
        self.assertGreater(result.opponent.pp, 0)
    def test_default_turn_transitioner_resolves_deckout_without_mutating_deck_count(self):
        transitioner = DefaultTurnTransitioner()
        ended = replace(_snapshot(opp_deck=0, is_ally_turn=True), is_turn_ended=True)

        result = transitioner.transition_turn(ended, "me")

        self.assertEqual(result.opponent.deck_count, 0)
        self.assertTrue(result.is_turn_ended)
        self.assertEqual(result.me.result, BATTLE_RESULT_DECK_OUT_WIN)
        self.assertEqual(result.opponent.result, BATTLE_RESULT_DECK_OUT_LOSE)
        terminal_value = BoardAdvantageValue().evaluate(result, player="me")
        self.assertTrue(terminal_value.terminal)
        self.assertEqual(terminal_value.value, 1.0)


if __name__ == "__main__":
    unittest.main()
