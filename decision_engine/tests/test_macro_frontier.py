# -*- coding: utf-8 -*-
"""Tests for macro_frontier / MacroTurnSearch lazy mode（P3 资源感知惰性宏观前沿）。

覆盖合同要求：
1. legacy 不变：lazy_mode 关闭时行为与 P0/P1/P2 一致（现有测试已覆盖，此处再断言默认关闭）。
2. 资源差异均注册：FutureResourceSignature 不同的完整线全部注册为 edge，不被合并。
3. 两 board 代表 + pending：同资源桶初始最多两个 board-distinct 代表激活，其余 pending 不删除。
4. budget 停止而不静默删除：预算耗尽记录 pending reason，edge 保留。
5. successor 只展开一次：每次 macro selection 只展开一个 successor，不重跑已展开微观搜索。
6. 未完成 prefix 不进 candidate（P1 边界，lazy 只接受 strict 完整线）。
"""

from __future__ import annotations

import unittest

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.planning.macro_turn_search import (
    MacroCreditBudget,
    MacroTurnSearch,
    TurnLine,
    TurnTransitionOutcome,
)
from decision_engine.planning.macro_frontier import (
    DEFAULT_BUCKET_ACTIVE_QUOTA,
    EDGE_ACTIVE,
    EDGE_EXPANDED,
    EDGE_PENDING,
    ExpansionOutcome,
    LazyBudgetTracker,
    MacroFrontier,
    PENDING_BUCKET_QUOTA,
    PENDING_BUDGET_MACRO_NODES,
    PENDING_BUDGET_TURN_TRANSITIONS,
    PENDING_DUPLICATE_BOARD,
)
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
    my_life: int = 20,
    opp_life: int = 20,
    my_ep: int = 0,
    my_sep: int = 0,
    my_max_pp: int = 10,
    my_field: list[CardView] | None = None,
    my_hand: list[CardView] | None = None,
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
        me=PlayerView(life=my_life, max_life=20, pp=10, max_pp=my_max_pp,
                      ep=my_ep, sep=my_sep,
                      field=tuple(field_uids), hand=tuple(hand_uids)),
        opponent=PlayerView(life=opp_life, max_life=20, pp=10, max_pp=10),
        cards_by_uid=cards,
        legal=LegalView(),
    )


def _follower(uid: int, base: int, atk: int = 1, life: int = 1) -> CardView:
    return CardView(uid=uid, base_card_id=base, atk=atk, life=life,
                    card_type=CardType.FOLLOWER)


def _line(
    *,
    hand_key,
    diff_score,
    end_snapshot,
    seq=("play",),
    visit=1,
) -> TurnLine:
    return TurnLine(
        action_sequence=tuple(_action(k) for k in seq),
        end_snapshot=end_snapshot,
        remaining_hand_key=hand_key,
        active_player="me",
        turn_number=1,
        visit_count=visit,
        diff_score=diff_score,
    )


# ---------------------------------------------------------------------------
# LazyBudgetTracker：真实 enforce
# ---------------------------------------------------------------------------


class TestLazyBudgetTracker(unittest.TestCase):
    def test_none_limits_never_block(self):
        t = LazyBudgetTracker(MacroCreditBudget())
        for _ in range(1000):
            allowed, reason = t.check_macro_node(1)
            self.assertTrue(allowed)
            self.assertIsNone(reason)
            t.consume_macro_node(1)

    def test_macro_node_limit_blocks_and_records(self):
        t = LazyBudgetTracker(MacroCreditBudget(max_macro_nodes=1))
        allowed, _ = t.check_macro_node(1)
        self.assertTrue(allowed)
        t.consume_macro_node(1)
        blocked, reason = t.check_macro_node(1)
        self.assertFalse(blocked)
        self.assertEqual(reason, PENDING_BUDGET_MACRO_NODES)
        self.assertEqual(t.exhausted_events[PENDING_BUDGET_MACRO_NODES], 1)

    def test_turn_transition_and_predictor_limits(self):
        t = LazyBudgetTracker(
            MacroCreditBudget(max_turn_transitions=2, max_predictor_calls=3)
        )
        self.assertTrue(t.check_turn_transition(1)[0])
        t.consume_turn_transition(1)
        self.assertTrue(t.check_turn_transition(1)[0])
        t.consume_turn_transition(1)
        self.assertFalse(t.check_turn_transition(1)[0])
        self.assertTrue(t.check_predictor_calls(3)[0])
        t.consume_predictor_calls(3)
        self.assertFalse(t.check_predictor_calls(1)[0])


# ---------------------------------------------------------------------------
# MacroFrontier：注册 / 两 board 代表 / pending / 不删除
# ---------------------------------------------------------------------------


class TestMacroFrontierRegistration(unittest.TestCase):
    def test_distinct_resource_signatures_all_registered(self):
        """要点2：资源差异（不同 EP/SEP/max_pp）的完整线全部注册为独立 edge，不合并。"""
        tracker = LazyBudgetTracker(MacroCreditBudget())
        frontier = MacroFrontier(tracker, perspective="me")
        lines = [
            _line(hand_key=(1,), diff_score=0.9, end_snapshot=_snapshot(my_ep=0)),
            _line(hand_key=(1,), diff_score=0.8, end_snapshot=_snapshot(my_ep=1)),
            _line(hand_key=(1,), diff_score=0.7, end_snapshot=_snapshot(my_sep=1)),
            _line(hand_key=(1,), diff_score=0.6, end_snapshot=_snapshot(my_max_pp=9)),
        ]
        edges = frontier.register_lines(lines)
        # 每条 line 一条 edge，绝不因手牌相同而合并。
        self.assertEqual(len(edges), 4)
        self.assertEqual(len(frontier.edges), 4)
        # 4 个不同资源桶。
        sigs = {e.resource_signature for e in edges}
        self.assertEqual(len(sigs), 4)
        # 每桶单条 → 全部激活（board-distinct 各自成桶）。
        self.assertEqual(len(frontier.active_unexpanded_edges()), 4)
        self.assertEqual(len(frontier.pending_edges()), 0)

    def test_two_board_representatives_rest_pending(self):
        """要点3：同一资源桶内，board fingerprint 不同的最多两个代表激活，其余 pending。"""
        tracker = LazyBudgetTracker(MacroCreditBudget())
        frontier = MacroFrontier(tracker, perspective="me", bucket_active_quota=2)
        # 同资源桶（相同 ep/sep/max_pp/hand），但 4 个不同 board（不同 field 卡）。
        lines = [
            _line(hand_key=(), diff_score=0.9,
                  end_snapshot=_snapshot(my_field=[_follower(10, 100)])),
            _line(hand_key=(), diff_score=0.8,
                  end_snapshot=_snapshot(my_field=[_follower(11, 200)])),
            _line(hand_key=(), diff_score=0.7,
                  end_snapshot=_snapshot(my_field=[_follower(12, 300)])),
            _line(hand_key=(), diff_score=0.6,
                  end_snapshot=_snapshot(my_field=[_follower(13, 400)])),
        ]
        frontier.register_lines(lines)
        # 同一资源桶。
        self.assertEqual(len({e.resource_signature for e in frontier.edges}), 1)
        active = frontier.active_unexpanded_edges()
        pending = frontier.pending_edges()
        # 最多两个 board-distinct 代表激活。
        self.assertEqual(len(active), 2, "同资源桶初始最多两个 board-distinct 代表")
        self.assertEqual(len(pending), 2)
        # 激活的是分数最高的两条。
        self.assertEqual({round(e.diff_score, 1) for e in active}, {0.9, 0.8})
        # pending 原因为配额已满，且 edge 未被删除。
        self.assertTrue(all(p.pending_reason == PENDING_BUCKET_QUOTA for p in pending))
        self.assertEqual(len(frontier.edges), 4, "pending edge 绝不删除")

    def test_duplicate_board_does_not_consume_distinct_slot(self):
        """同资源桶内 board 重复的线不占 board-distinct 名额，标 DUPLICATE_BOARD pending。"""
        tracker = LazyBudgetTracker(MacroCreditBudget())
        frontier = MacroFrontier(tracker, perspective="me", bucket_active_quota=2)
        # board A（重复两次）+ board B + board C。
        board_a1 = _snapshot(my_field=[_follower(10, 100)])
        board_a2 = _snapshot(my_field=[_follower(20, 100)])  # 同 base/atk/life，仅 uid 不同 → 同 fingerprint
        board_b = _snapshot(my_field=[_follower(11, 200)])
        board_c = _snapshot(my_field=[_follower(12, 300)])
        lines = [
            _line(hand_key=(), diff_score=0.9, end_snapshot=board_a1),
            _line(hand_key=(), diff_score=0.85, end_snapshot=board_a2),
            _line(hand_key=(), diff_score=0.8, end_snapshot=board_b),
            _line(hand_key=(), diff_score=0.7, end_snapshot=board_c),
        ]
        frontier.register_lines(lines)
        active = frontier.active_unexpanded_edges()
        # board A（最高分代表）+ board B 激活；A 的重复项与 C 都 pending。
        self.assertEqual(len(active), 2)
        active_scores = {round(e.diff_score, 2) for e in active}
        self.assertEqual(active_scores, {0.9, 0.8})
        pending = frontier.pending_edges()
        reasons = {p.pending_reason for p in pending}
        self.assertIn(PENDING_DUPLICATE_BOARD, reasons)
        self.assertEqual(len(frontier.edges), 4)


class TestMacroFrontierExpansion(unittest.TestCase):
    def _make_frontier_two_active(self):
        tracker = LazyBudgetTracker(MacroCreditBudget())
        frontier = MacroFrontier(tracker, perspective="me", bucket_active_quota=2)
        lines = [
            _line(hand_key=(), diff_score=0.9,
                  end_snapshot=_snapshot(my_field=[_follower(10, 100)])),
            _line(hand_key=(), diff_score=0.8,
                  end_snapshot=_snapshot(my_field=[_follower(11, 200)])),
            _line(hand_key=(), diff_score=0.7,
                  end_snapshot=_snapshot(my_field=[_follower(12, 300)])),
        ]
        frontier.register_lines(lines)
        return tracker, frontier

    def test_expand_one_only_expands_single_edge(self):
        """要点5：每次 expand_one 只展开一条 edge。"""
        tracker, frontier = self._make_frontier_two_active()
        expand_calls = []

        def expander(edge):
            expand_calls.append(edge)
            return ExpansionOutcome(
                expanded=True, successor_snapshot=_snapshot(), backed_up_value=1.0
            )

        exp1 = frontier.expand_one(expander)
        self.assertIsNotNone(exp1.expanded_edge)
        self.assertEqual(len(expand_calls), 1, "一次调用只展开一条 edge")
        self.assertEqual(exp1.expanded_edge.status, EDGE_EXPANDED)
        # 已展开 edge 不再是 active。
        self.assertEqual(len(frontier.expanded_edges()), 1)

    def test_expanded_edge_not_reexpanded(self):
        """要点5：已展开 edge 不会被重复展开（不重跑微观搜索）。"""
        tracker, frontier = self._make_frontier_two_active()
        seen_ids = []

        def expander(edge):
            seen_ids.append(id(edge))
            return ExpansionOutcome(
                expanded=True, successor_snapshot=_snapshot(), backed_up_value=1.0
            )

        # 反复展开直到没有可展开 edge。
        while frontier.has_expandable():
            frontier.expand_one(expander)
        # 每条 edge 只展开一次（id 无重复）。
        self.assertEqual(len(seen_ids), len(set(seen_ids)))
        # 全部 edge 最终都展开（含被唤醒的 pending）。
        self.assertEqual(len(frontier.expanded_edges()), 3)

    def test_pending_woken_after_active_expanded(self):
        """active 展开完后，pending 被唤醒继续展开（progressive widening），不删除。"""
        tracker, frontier = self._make_frontier_two_active()
        # 初始 2 active + 1 pending。
        self.assertEqual(len(frontier.active_unexpanded_edges()), 2)
        self.assertEqual(len(frontier.pending_edges()), 1)

        def expander(edge):
            return ExpansionOutcome(
                expanded=True, successor_snapshot=_snapshot(), backed_up_value=1.0
            )

        frontier.expand_one(expander)
        frontier.expand_one(expander)
        # 现在 active 都展开了，pending 应可被唤醒。
        self.assertTrue(frontier.has_expandable())
        exp3 = frontier.expand_one(expander)
        self.assertIsNotNone(exp3.expanded_edge)
        self.assertEqual(len(frontier.expanded_edges()), 3)

    def test_budget_exhaustion_marks_pending_not_delete(self):
        """要点4：预算耗尽时把候选标 pending 并记录 reason，绝不删除。"""
        tracker = LazyBudgetTracker(MacroCreditBudget(max_macro_nodes=1))
        frontier = MacroFrontier(tracker, perspective="me", bucket_active_quota=2)
        lines = [
            _line(hand_key=(), diff_score=0.9,
                  end_snapshot=_snapshot(my_field=[_follower(10, 100)])),
            _line(hand_key=(), diff_score=0.8,
                  end_snapshot=_snapshot(my_field=[_follower(11, 200)])),
        ]
        frontier.register_lines(lines)

        def expander(edge):
            return ExpansionOutcome(
                expanded=True, successor_snapshot=_snapshot(), backed_up_value=1.0
            )

        # 第一次展开成功（消耗 macro_node 预算 1）。
        exp1 = frontier.expand_one(expander)
        self.assertIsNotNone(exp1.expanded_edge)
        # 第二次预算耗尽：候选标 pending，记录 reason，不删除。
        exp2 = frontier.expand_one(expander)
        self.assertIsNone(exp2.expanded_edge)
        self.assertTrue(exp2.exhausted)
        self.assertEqual(exp2.pending_reason, PENDING_BUDGET_MACRO_NODES)
        # edge 总数不变（保留），且有 pending。
        self.assertEqual(len(frontier.edges), 2)
        self.assertEqual(len(frontier.expanded_edges()), 1)
        self.assertGreaterEqual(len(frontier.pending_edges()), 1)

    def test_expander_internal_exhaustion_marks_pending(self):
        """expander 内部（transition/内层）预算耗尽也标 pending，不删除。"""
        tracker, frontier = self._make_frontier_two_active()

        def expander(edge):
            return ExpansionOutcome(
                expanded=False, pending_reason=PENDING_BUDGET_TURN_TRANSITIONS
            )

        exp = frontier.expand_one(expander)
        self.assertIsNone(exp.expanded_edge)
        self.assertEqual(exp.pending_reason, PENDING_BUDGET_TURN_TRANSITIONS)
        self.assertTrue(exp.exhausted)
        self.assertEqual(len(frontier.edges), 3, "不删除")


# ---------------------------------------------------------------------------
# MacroTurnSearch lazy 集成
# ---------------------------------------------------------------------------


def _tagged_snapshot(tag: int, *, my_ep: int = 0, my_field=None) -> BattleSnapshot:
    my_field = my_field or []
    cards = {c.uid: c for c in my_field}
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=10, max_pp=10, ep=my_ep,
                      field=tuple(c.uid for c in my_field)),
        opponent=PlayerView(life=tag, max_life=20, pp=10, max_pp=10),
        cards_by_uid=cards,
        legal=LegalView(),
    )


class TestMacroLazyModeIntegration(unittest.TestCase):
    def test_default_mode_is_legacy(self):
        """要点1：默认 lazy_mode=False，走 legacy 路径，诊断无 lazy_mode 键。"""
        def turn_lines_fn(micro_root, snapshot, player, k):
            tag = snapshot.opponent.life
            if tag == 100:
                return [
                    TurnLine(action_sequence=(_action("play"),),
                             end_snapshot=_tagged_snapshot(10),
                             remaining_hand_key=(1,), active_player="me",
                             turn_number=1, visit_count=1, diff_score=0.5),
                ]
            return [TurnLine(action_sequence=(_action("play"),),
                             end_snapshot=_tagged_snapshot(tag + 1),
                             remaining_hand_key=(0,), active_player="me",
                             turn_number=1, visit_count=1, diff_score=0.0)]

        search = MacroTurnSearch(
            micro_search_fn=lambda s, p: None,
            transition_fn=lambda e, ep: _tagged_snapshot(e.opponent.life + 1),
            leaf_value_fn=lambda s, rp: 0.3,
            terminal_fn=lambda s, rp: None,
            turn_lines_fn=turn_lines_fn,
            macro_depth=2, k=3, root_k=5,
        )
        self.assertFalse(search.lazy_mode)
        res = search.search(_tagged_snapshot(100), player="me")
        self.assertNotIn("lazy_mode", res.diagnostics)
        self.assertFalse(res.diagnostics["credit_budget_enforced"])

    def test_lazy_mode_enforces_budget_and_records_pending(self):
        """要点3/4/5：lazy 模式真实 enforce max_macro_nodes；耗尽记录 pending，不删除。

        根层 3 条同资源桶 board-distinct 线（初始 2 active + 1 pending）。
        max_macro_nodes=1 → 只能展开 1 条 successor；其余保留 pending，记录 reason。
        """
        # 根层 3 条：同资源桶（opponent.life/ep/hand 均相同），仅 board 不同。
        root_lines = [
            TurnLine(action_sequence=(_action("play"),),
                     end_snapshot=_tagged_snapshot(50, my_field=[_follower(1, 100)]),
                     remaining_hand_key=(), active_player="me", turn_number=1,
                     visit_count=1, diff_score=0.9),
            TurnLine(action_sequence=(_action("play"),),
                     end_snapshot=_tagged_snapshot(50, my_field=[_follower(2, 200)]),
                     remaining_hand_key=(), active_player="me", turn_number=1,
                     visit_count=1, diff_score=0.8),
            TurnLine(action_sequence=(_action("play"),),
                     end_snapshot=_tagged_snapshot(50, my_field=[_follower(3, 300)]),
                     remaining_hand_key=(), active_player="me", turn_number=1,
                     visit_count=1, diff_score=0.7),
        ]
        micro_calls = []

        def turn_lines_fn(micro_root, snapshot, player, k):
            if snapshot.opponent.life == 100:
                return root_lines
            return []  # 内层无线 → 叶结算

        def micro_search_fn(snapshot, player):
            micro_calls.append(snapshot.opponent.life)
            return None

        def transition_fn(end, ep):
            # 固定 successor（内层无线 → 叶结算）；只关心 frontier 预算行为。
            return TurnTransitionOutcome(
                snapshot=_tagged_snapshot(999),
                predictor_calls=1, turn_transitions=1,
            )

        search = MacroTurnSearch(
            micro_search_fn=micro_search_fn,
            transition_fn=transition_fn,
            leaf_value_fn=lambda s, rp: 0.5,
            terminal_fn=lambda s, rp: None,
            turn_lines_fn=turn_lines_fn,
            macro_depth=2, k=3, root_k=5,
            lazy_mode=True, bucket_active_quota=2,
            credit_budget=MacroCreditBudget(max_macro_nodes=1),
        )
        res = search.search(_tagged_snapshot(100), player="me")
        diag = res.diagnostics
        self.assertTrue(diag["lazy_mode"])
        self.assertTrue(diag["credit_budget_enforced"])
        # 只展开了 1 条 successor（预算 max_macro_nodes=1）。
        self.assertEqual(diag["lazy_successor_expansions"], 1)
        self.assertEqual(diag["frontier_expanded"], 1)
        # 其余 edge 全部保留（3 条注册，1 展开，2 pending，无删除）。
        self.assertEqual(diag["frontier_edge_count"], 3)
        self.assertGreaterEqual(diag["frontier_pending"], 2)
        # pending reason 里含 macro node 预算耗尽或桶配额。
        self.assertTrue(diag["frontier_pending_reason_counts"])
        # 只有已展开 edge 进入 candidate。
        self.assertEqual(len(res.ranked_lines), 1)
        self.assertIsNotNone(res.best_line)

    def test_lazy_mode_registers_all_distinct_resource_lines(self):
        """要点2：不同 FutureResourceSignature 的完整线全部注册为 edge，不合并。"""
        root_lines = [
            TurnLine(action_sequence=(_action("play"),),
                     end_snapshot=_tagged_snapshot(10, my_ep=0),
                     remaining_hand_key=(1,), active_player="me", turn_number=1,
                     visit_count=1, diff_score=0.9),
            TurnLine(action_sequence=(_action("play"),),
                     end_snapshot=_tagged_snapshot(11, my_ep=1),
                     remaining_hand_key=(1,), active_player="me", turn_number=1,
                     visit_count=1, diff_score=0.8),
            TurnLine(action_sequence=(_action("play"),),
                     end_snapshot=_tagged_snapshot(12, my_ep=2),
                     remaining_hand_key=(1,), active_player="me", turn_number=1,
                     visit_count=1, diff_score=0.7),
        ]

        def turn_lines_fn(micro_root, snapshot, player, k):
            if snapshot.opponent.life == 100:
                return root_lines
            return []

        search = MacroTurnSearch(
            micro_search_fn=lambda s, p: None,
            transition_fn=lambda e, ep: TurnTransitionOutcome(
                snapshot=_tagged_snapshot(e.opponent.life + 100)),
            leaf_value_fn=lambda s, rp: 0.2,
            terminal_fn=lambda s, rp: None,
            turn_lines_fn=turn_lines_fn,
            macro_depth=2, k=3, root_k=5,
            lazy_mode=True,
            credit_budget=MacroCreditBudget(),  # 无上限：全部展开
        )
        res = search.search(_tagged_snapshot(100), player="me")
        diag = res.diagnostics
        # 3 个不同资源桶，各自单条 → 全部激活并展开。
        self.assertEqual(diag["frontier_edge_count"], 3)
        self.assertEqual(diag["frontier_bucket_count"], 3)
        self.assertEqual(diag["frontier_expanded"], 3)
        self.assertEqual(diag["frontier_pending"], 0)
        self.assertEqual(len(res.ranked_lines), 3)

    def test_lazy_successor_expanded_exactly_once_each(self):
        """要点5：每个 successor 只展开一次，micro/transition 不因重复 selection 重跑。"""
        root_lines = [
            TurnLine(action_sequence=(_action("play"),),
                     end_snapshot=_tagged_snapshot(10, my_field=[_follower(1, 100)]),
                     remaining_hand_key=(), active_player="me", turn_number=1,
                     visit_count=1, diff_score=0.9),
            TurnLine(action_sequence=(_action("play"),),
                     end_snapshot=_tagged_snapshot(20, my_field=[_follower(2, 200)]),
                     remaining_hand_key=(), active_player="me", turn_number=1,
                     visit_count=1, diff_score=0.8),
        ]
        transition_calls = []

        def turn_lines_fn(micro_root, snapshot, player, k):
            if snapshot.opponent.life == 100:
                return root_lines
            return []

        def transition_fn(end, ep):
            transition_calls.append(end.opponent.life)
            return TurnTransitionOutcome(
                snapshot=_tagged_snapshot(end.opponent.life + 100),
                predictor_calls=1, turn_transitions=1,
            )

        search = MacroTurnSearch(
            micro_search_fn=lambda s, p: None,
            transition_fn=transition_fn,
            leaf_value_fn=lambda s, rp: 0.1,
            terminal_fn=lambda s, rp: None,
            turn_lines_fn=turn_lines_fn,
            macro_depth=2, k=3, root_k=5,
            lazy_mode=True,
            credit_budget=MacroCreditBudget(),
        )
        search.search(_tagged_snapshot(100), player="me")
        # 两条根 edge 各触发一次 transition（successor 展开），无重复。
        self.assertEqual(sorted(transition_calls), [10, 20])
        self.assertEqual(len(transition_calls), len(set(transition_calls)))

    def test_lazy_only_accepts_complete_lines_no_pending_prefix(self):
        """要点6：lazy 只消费 strict 完整 TurnLine；不存在把未完成 prefix 塞入 candidate。

        turn_lines_fn（= extract/top_k 语义）只返回完整线；此处断言 frontier edge
        数等于完整线数，未完成 prefix 不会以任何形式进入 candidate/ranked。
        """
        complete_lines = [
            TurnLine(action_sequence=(_action("play"), _action("turn_end")),
                     end_snapshot=_tagged_snapshot(10),
                     remaining_hand_key=(), active_player="me", turn_number=1,
                     visit_count=1, diff_score=0.5),
        ]

        def turn_lines_fn(micro_root, snapshot, player, k):
            if snapshot.opponent.life == 100:
                return complete_lines
            return []

        search = MacroTurnSearch(
            micro_search_fn=lambda s, p: None,
            transition_fn=lambda e, ep: TurnTransitionOutcome(
                snapshot=_tagged_snapshot(999)),
            leaf_value_fn=lambda s, rp: 0.0,
            terminal_fn=lambda s, rp: None,
            turn_lines_fn=turn_lines_fn,
            macro_depth=2, k=3, root_k=5,
            lazy_mode=True,
            credit_budget=MacroCreditBudget(),
        )
        res = search.search(_tagged_snapshot(100), player="me")
        self.assertEqual(res.diagnostics["frontier_edge_count"], 1)
        self.assertEqual(len(res.ranked_lines), 1)
        # candidate 只含完整线（其 action_sequence 来自完整 TurnLine）。
        self.assertEqual(res.best_line, complete_lines[0])

    def test_lazy_no_expandable_returns_leaf_value(self):
        """无候选完整线时，lazy 返回叶估值且不崩溃（结构化空结果）。"""
        def turn_lines_fn(micro_root, snapshot, player, k):
            return []

        search = MacroTurnSearch(
            micro_search_fn=lambda s, p: None,
            transition_fn=lambda e, ep: TurnTransitionOutcome(snapshot=_tagged_snapshot(1)),
            leaf_value_fn=lambda s, rp: 0.42,
            terminal_fn=lambda s, rp: None,
            turn_lines_fn=turn_lines_fn,
            macro_depth=2, k=3, root_k=5,
            lazy_mode=True,
        )
        res = search.search(_tagged_snapshot(100), player="me")
        self.assertAlmostEqual(res.backed_up_value, 0.42)
        self.assertIsNone(res.best_line)
        self.assertEqual(res.diagnostics["frontier_edge_count"], 0)


if __name__ == "__main__":
    unittest.main()
