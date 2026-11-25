# -*- coding: utf-8 -*-
"""Tests for MicroTurnCompletionSession / CompletionResult（P1 可恢复完整回合完成器）。

覆盖 v17 待办的核心约束：
- session 两次 advance 不重置树/visit 累积（真正复用已展开子树）。
- 完整线提取：只含到达 turn_end 的 strict 路径。
- 未完成前缀（depth 耗尽）标记 PENDING，绝不生成 TurnLine、不进入 Macro backup/candidate。
- 完成后可被 Macro 包装消费（complete_lines 作为 turn_lines_fn 来源）。
- branch_seed 路径一致：session advance 与一次性 search 的展开语义相同。
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.puct import PuctConfig, PuctSearch
from decision_engine.planning.macro_turn_search import (
    CompletionResult,
    MacroTurnSearch,
    MicroTurnCompletionSession,
    PENDING_STATUS,
    PendingPrefix,
    TurnLine,
    extract_turn_lines,
)
from decision_engine.planning.search_value import BoardAdvantageValue
from decision_engine.state.snapshot import (
    BattleSnapshot,
    CardType,
    CardView,
    LegalView,
    PlayerView,
)


def _action(kind: str, uid: int = 0) -> LegalAction:
    return LegalAction(kind=ActionKind(kind), source_uid=uid)


def _snapshot(
    *,
    opp_life: int = 20,
    my_field: list[CardView] | None = None,
    my_hand: list[CardView] | None = None,
    is_turn_ended: bool = False,
) -> BattleSnapshot:
    my_field = my_field or []
    my_hand = my_hand or []
    cards: dict[int, CardView] = {}
    field_uids: list[int] = []
    hand_uids: list[int] = []
    for c in my_field:
        cards[c.uid] = c
        field_uids.append(c.uid)
    for c in my_hand:
        cards[c.uid] = c
        hand_uids.append(c.uid)
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=10, max_pp=10,
                      field=tuple(field_uids), hand=tuple(hand_uids)),
        opponent=PlayerView(life=opp_life, max_life=20, pp=10, max_pp=10),
        cards_by_uid=cards,
        legal=LegalView(),
        is_turn_ended=is_turn_ended,
    )


@dataclass
class _MockPrediction:
    snapshot: BattleSnapshot
    certainty: Any = None
    unresolved_effects: tuple = ()


class _ScriptedPredictor:
    """按 (marker_fn, action_kind) 返回预设 after-state。"""

    def __init__(self, transitions):
        self.transitions = transitions

    def predict(self, snapshot: BattleSnapshot, action: LegalAction, *, player: str = "me"):
        kind = action.kind.value
        for marker_fn, akind, to_snap in self.transitions:
            if akind == kind and marker_fn(snapshot):
                return _MockPrediction(snapshot=to_snap)
        return _MockPrediction(snapshot=snapshot)


class _ScriptedEnumerator:
    def __init__(self, rules):
        self.rules = rules

    def enumerate(self, snapshot: BattleSnapshot, *, player: str = "me"):
        for marker_fn, actions in self.rules:
            if marker_fn(snapshot):
                return actions
        return [_action("turn_end")]


# ---------------------------------------------------------------------------
# 场景 A：一张牌可以完整走到 turn_end（play -> turn_end）
# ---------------------------------------------------------------------------


def _card(uid, base, atk=3, life=3):
    return CardView(uid=uid, base_card_id=base, atk=atk, life=life, card_type=CardType.FOLLOWER)


def _build_completable_session(*, simulations=64, max_depth=4):
    """构造一个能走完 play->turn_end 的 session。"""
    card = _card(100, 777)
    initial = _snapshot(my_hand=[card])
    after_play = _snapshot(my_field=[_card(100, 777)])

    play = _action("play", uid=100)
    end = _action("turn_end", uid=0)

    def is_initial(s):
        return len(s.me.hand) == 1

    def is_after_play(s):
        return len(s.me.hand) == 0 and len(s.me.field) == 1

    predictor = _ScriptedPredictor([(is_initial, "play", after_play)])
    enumerator = _ScriptedEnumerator([
        (is_initial, [play, end]),
        (is_after_play, [end]),
    ])
    search = PuctSearch(
        enumerator=enumerator, child_enumerator=enumerator, predictor=predictor,
        value=BoardAdvantageValue(),
        config=PuctConfig(simulations=simulations, max_depth=max_depth, max_turns=1,
                          c_puct=1.5, dirichlet_epsilon=0.0),
    )
    return search, initial


class TestSessionAdvanceAccumulates(unittest.TestCase):
    def test_two_advances_do_not_reset_tree(self):
        """两次 advance：visit 累积、executed_simulations 累加，root 树对象不变。"""
        search, initial = _build_completable_session(simulations=16)
        session = MicroTurnCompletionSession(search, initial, player="me")

        root_before = session.root
        n1 = session.advance(16)
        visits_after_first = sum(root_before.visit_counts)
        exec_after_first = session.executed_simulations

        n2 = session.advance(16)
        visits_after_second = sum(session.root.visit_counts)
        exec_after_second = session.executed_simulations

        # root 对象在两次 advance 之间保持同一实例（复用已展开树）
        self.assertIs(session.root, root_before)
        # 执行数累加
        self.assertEqual(exec_after_first, 16)
        self.assertEqual(exec_after_second, 32)
        self.assertEqual(n1, 16)
        self.assertEqual(n2, 16)
        # root 总访问单调累积，不被第二次 advance 重置
        self.assertGreaterEqual(visits_after_second, visits_after_first)
        self.assertEqual(session.result().executed_simulations, 32)

    def test_advance_zero_is_noop(self):
        search, initial = _build_completable_session(simulations=8)
        session = MicroTurnCompletionSession(search, initial, player="me")
        session.advance(8)
        before = session.executed_simulations
        self.assertEqual(session.advance(0), 0)
        self.assertEqual(session.executed_simulations, before)


class TestCompleteLineExtraction(unittest.TestCase):
    def test_extracts_strict_complete_line(self):
        """走满预算后能提取到 play->turn_end 的 strict 完整线。"""
        search, initial = _build_completable_session(simulations=64, max_depth=4)
        session = MicroTurnCompletionSession(search, initial, player="me")
        session.advance(64)
        result = session.result()

        seqs = [tuple(a.kind.value for a in ln.action_sequence) for ln in result.complete_lines]
        self.assertIn(("play",), seqs, f"未提取到 play->turn_end 完整线: {seqs}")
        play_line = next(
            ln for ln in result.complete_lines
            if tuple(a.kind.value for a in ln.action_sequence) == ("play",)
        )
        # 完整线止于 turn_end 前：手牌空、场上 1 随从
        self.assertEqual(len(play_line.end_snapshot.me.hand), 0)
        self.assertEqual(len(play_line.end_snapshot.me.field), 1)
        # 完整回合可完成，无 pending
        self.assertFalse(result.has_pending(),
                         f"可完成回合不应有 pending: {result.pending_prefixes}")


# ---------------------------------------------------------------------------
# 场景 B：深度不足以走到 turn_end -> 前缀被截断为 PENDING
# ---------------------------------------------------------------------------


def _build_pending_session(*, simulations=64, max_depth):
    """构造一个需要连打 3 张牌才能 turn_end、但 max_depth 不足的 session。

    initial(hand=3) -play-> s1(hand=2) -play-> s2(hand=1) -play-> s3(hand=0) -turn_end->
    每一步都不暴露 turn_end，直到打完 3 张。max_depth 小于所需深度时，
    路径在中途被截断为 depth-leaf（未 turn_end）→ PENDING。
    """
    c1 = _card(101, 111)
    c2 = _card(102, 222)
    c3 = _card(103, 333)
    initial = _snapshot(my_hand=[c1, c2, c3])
    s1 = _snapshot(my_field=[_card(101, 111)], my_hand=[c2, c3])
    s2 = _snapshot(my_field=[_card(101, 111), _card(102, 222)], my_hand=[c3])
    s3 = _snapshot(my_field=[_card(101, 111), _card(102, 222), _card(103, 333)])

    play1 = _action("play", uid=101)
    play2 = _action("play", uid=102)
    play3 = _action("play", uid=103)
    end = _action("turn_end", uid=0)

    def is_init(s):
        return len(s.me.hand) == 3

    def is_s1(s):
        return len(s.me.hand) == 2

    def is_s2(s):
        return len(s.me.hand) == 1

    def is_s3(s):
        return len(s.me.hand) == 0 and len(s.me.field) == 3

    predictor = _ScriptedPredictor([
        (is_init, "play", s1),
        (is_s1, "play", s2),
        (is_s2, "play", s3),
    ])
    # 关键：只有打完 3 张（is_s3）才暴露 turn_end；中间步骤只有 play。
    # 这样必须走满 3 层 play 才能到达 turn_end。
    enumerator = _ScriptedEnumerator([
        (is_init, [play1]),
        (is_s1, [play2]),
        (is_s2, [play3]),
        (is_s3, [end]),
    ])
    search = PuctSearch(
        enumerator=enumerator, child_enumerator=enumerator, predictor=predictor,
        value=BoardAdvantageValue(),
        config=PuctConfig(simulations=simulations, max_depth=max_depth, max_turns=1,
                          c_puct=1.5, dirichlet_epsilon=0.0,
                          # 关闭 greedy rollout 之外的 phase，隔离 depth 截断行为
                          prior_deepening_threshold=0.0, prior_deepening_ratio=0.0),
    )
    return search, initial


class TestPendingPrefixNotComplete(unittest.TestCase):
    def test_depth_exhausted_prefix_is_pending_not_turnline(self):
        """max_depth 不足以到达 turn_end：不生成完整 TurnLine，前缀标记 PENDING。"""
        # 需要 3 层 play 才能 turn_end；max_depth=2 时永远到不了 turn_end。
        search, initial = _build_pending_session(simulations=32, max_depth=2)
        session = MicroTurnCompletionSession(search, initial, player="me")
        session.advance(32)
        result = session.result()

        # 没有任何到达 turn_end 的完整线
        self.assertEqual(len(result.complete_lines), 0,
                         f"深度不足时不应产生完整线: "
                         f"{[tuple(a.kind.value for a in l.action_sequence) for l in result.complete_lines]}")
        # extract_turn_lines 直接作用也应为空（未完成前缀不生成 TurnLine）
        self.assertEqual(len(extract_turn_lines(session.root)), 0)
        # 存在 PENDING 未完成前缀
        self.assertTrue(result.has_pending(), "深度耗尽的未完成回合应报告 pending 前缀")
        for p in result.pending_prefixes:
            self.assertIsInstance(p, PendingPrefix)
            self.assertEqual(p.status, PENDING_STATUS)
            # 前缀尚未 turn_end
            self.assertFalse(getattr(p.prefix_snapshot, "is_turn_ended", False))
            # 前缀不含 turn_end 动作
            self.assertNotIn("turn_end", [a.kind.value for a in p.action_sequence])

    def test_pending_prefix_carries_no_strategic_value(self):
        """PENDING 前缀是可恢复句柄，不携带 diff_score/TurnLine 战略价值。"""
        search, initial = _build_pending_session(simulations=16, max_depth=2)
        session = MicroTurnCompletionSession(search, initial, player="me")
        session.advance(16)
        result = session.result()
        self.assertTrue(result.pending_prefixes)
        # PendingPrefix 没有 diff_score 字段，不能被误当 TurnLine 消费
        for p in result.pending_prefixes:
            self.assertFalse(hasattr(p, "diff_score"))

    def test_deeper_budget_can_complete_previously_pending(self):
        """给足 max_depth 后，之前 PENDING 的回合可完成为 strict 完整线。"""
        # max_depth=4 足以走 3 层 play + turn_end
        search, initial = _build_pending_session(simulations=64, max_depth=4)
        session = MicroTurnCompletionSession(search, initial, player="me")
        session.advance(64)
        result = session.result()
        seqs = [tuple(a.kind.value for a in ln.action_sequence) for ln in result.complete_lines]
        self.assertIn(("play", "play", "play"), seqs,
                      f"深度足够时应完成 play,play,play->turn_end: {seqs}")


# ---------------------------------------------------------------------------
# 场景 C：完成后可被 Macro 包装消费
# ---------------------------------------------------------------------------


class TestCompletionConsumedByMacro(unittest.TestCase):
    def test_macro_wraps_only_complete_lines(self):
        """完成后的 complete_lines 作为 Macro 的 turn_lines_fn 来源，可跑完 minimax。

        用一个 max_depth 混合场景：session 完成一条 play->turn_end 完整线，
        Macro 用该 session 的 top_k_complete_lines 作为根层候选，正常 backup。
        """
        search, initial = _build_completable_session(simulations=64, max_depth=4)
        session = MicroTurnCompletionSession(search, initial, player="me")
        session.advance(64)
        result = session.result()
        self.assertTrue(result.complete_lines, "应有可供 Macro 消费的完整线")

        # Macro 包装：turn_lines_fn 只返回 session 的 strict 完整线（不含 PENDING）。
        complete = session.top_k_complete_lines(k=3)
        self.assertTrue(complete)
        # 断言 Macro 消费的候选全部是到达 turn_end 的完整线
        for ln in complete:
            self.assertIsInstance(ln, TurnLine)

        consumed = {"lines": None}

        def turn_lines_fn(micro_root, snapshot, player, k):
            # 根层直接消费 session 完整线；内层用占位单线终止递归
            if snapshot is initial:
                consumed["lines"] = complete
                return complete
            return [TurnLine(
                action_sequence=(_action("play"),),
                end_snapshot=_snapshot(),
                remaining_hand_key=(0,),
                active_player=player,
                turn_number=1,
                visit_count=1,
                diff_score=0.0,
            )]

        macro = MacroTurnSearch(
            micro_search_fn=lambda s, p: session.root,   # Macro 复用 session 已展开树
            transition_fn=lambda end, ep: _snapshot(opp_life=10),
            leaf_value_fn=lambda s, rp: 0.5,
            terminal_fn=lambda s, rp: None,
            turn_lines_fn=turn_lines_fn,
            macro_depth=2, k=3, root_k=5,
        )
        macro_result = macro.search(initial, player="me")

        # Macro 成功用 session 完整线作为 best_line
        self.assertIsNotNone(macro_result.best_line)
        self.assertIn(macro_result.best_line, complete)
        # 消费的候选正是 session 的 strict 完整线
        self.assertIs(consumed["lines"], complete)

    def test_pending_prefix_excluded_from_macro_candidates(self):
        """PENDING 前缀不进入 Macro 候选：session 的 top_k_complete_lines 不含未完成前缀。"""
        # 深度不足场景：只有 PENDING，无完整线
        search, initial = _build_pending_session(simulations=32, max_depth=2)
        session = MicroTurnCompletionSession(search, initial, player="me")
        session.advance(32)

        complete = session.top_k_complete_lines(k=5)
        # 无完整线 → Macro 候选为空（未完成前缀绝不作为候选）
        self.assertEqual(complete, [], "PENDING-only 回合不得向 Macro 暴露任何候选线")
        # 诊断标注 pending 已被排除
        self.assertTrue(session.result().diagnostics["pending_excluded_from_macro"])


# ---------------------------------------------------------------------------
# 场景 D：session 树复用与一次性 search 语义一致（branch seed 路径一致）
# ---------------------------------------------------------------------------


class TestSessionConsistentWithSearch(unittest.TestCase):
    def test_session_complete_lines_match_oneshot_search(self):
        """同配置下：session advance(N) 提取的完整线集合与一次性 search 的一致。

        两条路径共享同一 _run_ucb_simulation 与相同 branch_seed，因此展开的完整线
        （按动作序列种类）应一致。
        """
        # 一次性 search
        search1, initial1 = _build_completable_session(simulations=64, max_depth=4)
        search1.search(initial1, player="me")
        oneshot_seqs = sorted(
            tuple(a.kind.value for a in ln.action_sequence)
            for ln in extract_turn_lines(search1._last_root)
        )

        # session 增量推进（同配置、同 branch_seed 默认 0）
        search2, initial2 = _build_completable_session(simulations=64, max_depth=4)
        session = MicroTurnCompletionSession(search2, initial2, player="me")
        session.advance(64)
        session_seqs = sorted(
            tuple(a.kind.value for a in ln.action_sequence)
            for ln in session.complete_lines()
        )

        self.assertEqual(oneshot_seqs, session_seqs,
                         "session 复用树的完整线集合应与一次性 search 一致")


if __name__ == "__main__":
    unittest.main()
