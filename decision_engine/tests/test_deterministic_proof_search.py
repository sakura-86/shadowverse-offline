"""Phase N1.10 M0 deterministic proof-search tests."""

import unittest
from dataclasses import dataclass

from decision_engine.planning.deterministic_proof_search import (
    DeterministicProofSearch,
    ProofSchedulerContext,
    ProofSearchConfig,
    ProofStatus,
    ProofTransition,
    SharedProofGraph,
)


@dataclass(frozen=True)
class _Edge:
    action: str
    target: str
    context: ProofSchedulerContext


class _FakeGame:
    def __init__(self, *, edges, winners=(), context_winners=()):
        self.edges = dict(edges)
        self.winners = dict(winners)
        self.context_winners = dict(context_winners)

    def state_fingerprint(self, state):
        return state

    def active_player(self, _state, scheduler_context):
        return scheduler_context.current_player

    def terminal_winner(self, state, scheduler_context):
        return self.context_winners.get((state, scheduler_context.action_index), self.winners.get(state))

    def legal_actions(self, state, scheduler_context, *, player):
        return tuple(edge.action for edge in self.edges.get((state, scheduler_context, player), ()))

    def transition(self, state, action, scheduler_context, *, player):
        for edge in self.edges[(state, scheduler_context, player)]:
            if edge.action == action:
                return ProofTransition(state=edge.target, scheduler_context=edge.context)
        raise AssertionError(f"unknown fake edge {state=} {action=}")


def _ctx(turn, action, player):
    return ProofSchedulerContext(turn_index=turn, action_index=action, current_player=player)


class DeterministicProofSearchTest(unittest.TestCase):
    def test_hidden_opponent_refutation_overturns_winning_principal_variation(self):
        """PV A: pressure -> cooperate -> A赢；B的refute会推翻这个伪胜。"""
        root = _ctx(1, 0, "A")
        b_turn = _ctx(1, 1, "B")
        game = _FakeGame(
            edges={
                ("root", root, "A"): (_Edge("pressure", "b_reply", b_turn),),
                ("b_reply", b_turn, "B"): (
                    _Edge("cooperate", "a_win", _ctx(1, 2, "A")),
                    _Edge("refute", "b_win", _ctx(1, 2, "A")),
                ),
            },
            winners={"a_win": "A", "b_win": "B"},
        )

        result = DeterministicProofSearch(game).search("root", root_player="A", scheduler_context=root)

        self.assertEqual(result.root.status, ProofStatus.PROVEN_LOSS)
        reply = next(node for node in result.nodes.values() if node.key.state_fingerprint == "b_reply")
        self.assertEqual(reply.status, ProofStatus.PROVEN_LOSS)
        self.assertEqual(reply.refuted_actions, ("refute",))
        self.assertEqual(result.root.proof_number, float("inf"))
        self.assertEqual(result.root.disproof_number, 0.0)

    def test_forced_win_requires_all_opponent_replies_to_fail(self):
        root = _ctx(1, 0, "A")
        b_turn = _ctx(1, 1, "B")
        game = _FakeGame(
            edges={
                ("root", root, "A"): (
                    _Edge("force", "b_after_force", b_turn),
                    _Edge("blunder", "b_win", b_turn),
                ),
                ("b_after_force", b_turn, "B"): (
                    _Edge("defend_left", "a_win_1", _ctx(1, 2, "A")),
                    _Edge("defend_right", "a_win_2", _ctx(1, 2, "A")),
                ),
            },
            winners={"a_win_1": "A", "a_win_2": "A", "b_win": "B"},
        )

        result = DeterministicProofSearch(game).search("root", root_player="A", scheduler_context=root)

        self.assertEqual(result.root.status, ProofStatus.PROVEN_WIN)
        self.assertEqual(result.root.proven_actions, ("force",))
        forced_reply = next(node for node in result.nodes.values() if node.key.state_fingerprint == "b_after_force")
        self.assertEqual(forced_reply.status, ProofStatus.PROVEN_WIN)
        self.assertEqual(forced_reply.proven_actions, ("defend_left", "defend_right"))

    def test_forced_loss_when_every_root_action_is_refuted(self):
        root = _ctx(1, 0, "A")
        game = _FakeGame(
            edges={
                ("root", root, "A"): (
                    _Edge("line_1", "b_win_1", _ctx(1, 1, "B")),
                    _Edge("line_2", "b_win_2", _ctx(1, 1, "B")),
                ),
            },
            winners={"b_win_1": "B", "b_win_2": "B"},
        )

        result = DeterministicProofSearch(game).search("root", root_player="A", scheduler_context=root)

        self.assertEqual(result.root.status, ProofStatus.PROVEN_LOSS)
        self.assertEqual(result.root.refuted_actions, ("line_1", "line_2"))

    def test_budget_exhaustion_remains_unknown(self):
        root = _ctx(1, 0, "A")
        game = _FakeGame(edges={("root", root, "A"): (_Edge("move", "a_win", _ctx(1, 1, "B")),)}, winners={"a_win": "A"})

        result = DeterministicProofSearch(game, config=ProofSearchConfig(max_expansions=0)).search(
            "root", root_player="A", scheduler_context=root
        )

        self.assertEqual(result.root.status, ProofStatus.UNKNOWN_BUDGET)
        self.assertEqual(result.root.blocked_reason, "unknown_budget")
        self.assertEqual(result.trace.expanded_nodes, 0)
        self.assertEqual(result.trace.budget_cuts, 1)

    def test_depth_limit_remains_auditable_unknown_without_recursing_deeper(self):
        root = _ctx(1, 0, "A")
        child = _ctx(1, 1, "A")
        game = _FakeGame(
            edges={
                ("root", root, "A"): (_Edge("advance", "child", child),),
                ("child", child, "A"): (_Edge("finish", "a_win", _ctx(1, 2, "B")),),
            },
            winners={"a_win": "A"},
        )

        result = DeterministicProofSearch(game, config=ProofSearchConfig(max_depth=1)).search(
            "root", root_player="A", scheduler_context=root
        )

        self.assertEqual(result.root.status, ProofStatus.UNKNOWN_DEPTH)
        self.assertEqual(result.root.blocked_reason, "unknown_depth")
        self.assertEqual(result.trace.expanded_nodes, 1)
        self.assertEqual(result.trace.budget_cuts, 1)

    def test_cache_keeps_same_state_under_distinct_scheduler_contexts_separate(self):
        root = _ctx(1, 0, "A")
        left_context = _ctx(1, 1, "B")
        right_context = _ctx(1, 2, "B")
        game = _FakeGame(
            edges={
                ("root", root, "A"): (
                    _Edge("left", "shared", left_context),
                    _Edge("right", "shared", right_context),
                ),
            },
            context_winners={
                ("shared", 1): "A",
                ("shared", 2): "B",
            },
        )

        result = DeterministicProofSearch(game).search("root", root_player="A", scheduler_context=root)

        shared = [node for node in result.nodes.values() if node.key.state_fingerprint == "shared"]
        self.assertEqual(len(shared), 2)
        self.assertEqual({node.key.scheduler_context.action_index for node in shared}, {1, 2})
        self.assertEqual({node.status for node in shared}, {ProofStatus.PROVEN_WIN, ProofStatus.PROVEN_LOSS})
        self.assertEqual(result.root.status, ProofStatus.PROVEN_WIN)
        self.assertEqual(result.root.proven_actions, ("left",))

    def test_cycle_is_explicit_unknown_not_a_win_or_loss(self):
        root = _ctx(1, 0, "A")
        b_turn = _ctx(1, 1, "B")
        game = _FakeGame(
            edges={
                ("root", root, "A"): (_Edge("wait", "b_turn", b_turn),),
                ("b_turn", b_turn, "B"): (_Edge("loop", "root", root),),
            },
        )

        result = DeterministicProofSearch(game).search("root", root_player="A", scheduler_context=root)

        self.assertEqual(result.root.status, ProofStatus.UNKNOWN_CYCLE)
        self.assertEqual(result.trace.cycle_cuts, 1)
        self.assertFalse(result.root.status.is_proven)



    def test_shared_graph_reuses_proven_child_across_earlier_root(self):
        parent_context = _ctx(1, 1, "A")
        child_context = _ctx(2, 1, "A")
        terminal_context = _ctx(2, 2, "B")
        game = _FakeGame(
            edges={
                ("parent", parent_context, "A"): (_Edge("advance", "child", child_context),),
                ("child", child_context, "A"): (_Edge("finish", "a_win", terminal_context),),
            },
            winners={"a_win": "A"},
        )
        graph = SharedProofGraph()

        child = DeterministicProofSearch(
            game,
            config=ProofSearchConfig(max_expansions=1),
            shared_graph=graph,
        ).search("child", root_player="A", scheduler_context=child_context)
        self.assertEqual(child.root.status, ProofStatus.PROVEN_WIN)
        self.assertGreaterEqual(graph.certificate_count, 1)

        reused = DeterministicProofSearch(
            game,
            config=ProofSearchConfig(max_expansions=1),
            shared_graph=graph,
        ).search("parent", root_player="A", scheduler_context=parent_context)
        self.assertEqual(reused.root.status, ProofStatus.PROVEN_WIN)
        self.assertEqual(reused.trace.expanded_nodes, 1)
        self.assertEqual(reused.trace.certificate_hits, 1)

        isolated = DeterministicProofSearch(
            game,
            config=ProofSearchConfig(max_expansions=1),
        ).search("parent", root_player="A", scheduler_context=parent_context)
        self.assertEqual(isolated.root.status, ProofStatus.UNKNOWN_BUDGET)


if __name__ == "__main__":
    unittest.main()
