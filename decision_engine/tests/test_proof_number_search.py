"""PNS proof-number scheduling tests."""

from dataclasses import dataclass
import unittest

from decision_engine.planning.deterministic_proof_search import ProofSchedulerContext, ProofSearchConfig, ProofStatus, ProofTransition
from decision_engine.planning.proof_number_search import ProofNumberSearch


@dataclass(frozen=True)
class _Edge:
    action: str
    target: str
    context: ProofSchedulerContext


class _Game:
    def __init__(self, edges, winners):
        self.edges = edges
        self.winners = winners

    def state_fingerprint(self, state): return state
    def active_player(self, _state, context): return context.current_player
    def terminal_winner(self, state, _context): return self.winners.get(state)
    def legal_actions(self, state, context, *, player):
        return tuple(item.action for item in self.edges.get((state, context, player), ()))
    def transition(self, state, action, context, *, player):
        for item in self.edges[(state, context, player)]:
            if item.action == action:
                return ProofTransition(item.target, item.context)
        raise AssertionError(action)


def _ctx(turn, action, player):
    return ProofSchedulerContext(turn, action, player)


class ProofNumberSearchTest(unittest.TestCase):
    def test_or_node_stops_after_first_proven_winning_child(self):
        root = _ctx(1, 1, "A")
        game = _Game(
            {
                ("root", root, "A"): (
                    _Edge("win_now", "a_win", _ctx(1, 2, "B")),
                    _Edge("huge_unneeded_branch", "later", _ctx(1, 2, "B")),
                ),
            },
            {"a_win": "A"},
        )
        result = ProofNumberSearch(game, config=ProofSearchConfig(max_expansions=1)).search(
            "root", root_player="A", scheduler_context=root
        )
        self.assertEqual(result.root.status, ProofStatus.PROVEN_WIN)
        self.assertEqual(result.trace.expanded_nodes, 1)
        self.assertEqual(result.root.proven_actions, ("win_now",))

    def test_and_node_stops_after_first_refuting_opponent_child(self):
        root = _ctx(1, 1, "A")
        opponent = _ctx(1, 2, "B")
        game = _Game(
            {
                ("root", root, "A"): (_Edge("advance", "opponent_node", opponent),),
                ("opponent_node", opponent, "B"): (
                    _Edge("refute", "b_win", _ctx(1, 3, "A")),
                    _Edge("irrelevant", "later", _ctx(1, 3, "A")),
                ),
            },
            {"b_win": "B"},
        )
        result = ProofNumberSearch(game, config=ProofSearchConfig(max_expansions=2)).search(
            "root", root_player="A", scheduler_context=root
        )
        self.assertEqual(result.root.status, ProofStatus.PROVEN_LOSS)
        self.assertEqual(result.trace.expanded_nodes, 2)
        node = next(value for value in result.nodes.values() if value.key.state_fingerprint == "opponent_node")
        self.assertEqual(node.refuted_actions, ("refute",))

    def test_budget_exhaustion_is_unknown(self):
        root = _ctx(1, 1, "A")
        game = _Game({("root", root, "A"): (_Edge("later", "later", _ctx(1, 2, "B")),)}, {})
        result = ProofNumberSearch(game, config=ProofSearchConfig(max_expansions=1)).search(
            "root", root_player="A", scheduler_context=root
        )
        self.assertEqual(result.root.status, ProofStatus.UNKNOWN_BUDGET)
        self.assertEqual(result.trace.budget_cuts, 1)


if __name__ == "__main__":
    unittest.main()
