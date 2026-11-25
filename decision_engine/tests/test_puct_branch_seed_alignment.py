# -*- coding: utf-8 -*-
"""Branch seed 对齐契约测试。

验证 Macro PUCT 完整 TurnLine 的微观搜索与 strict self-play / search_ab_match
执行侧使用**同一** branch_seed 派生：

- 搜索树内从回合根起算的第 N（1-based）个回合内动作，预测 branch_seed = base + N；
- 执行侧第 N 个动作用 branch_seed(seed, turn, N) = seed*100000 + turn*100 + N；
- 当把 PUCT 的 config.branch_seed 设为 base = branch_seed(seed, turn, 0) 时，两者在
  同一 (snapshot, action_sequence) 下完全一致，Macro TurnLine 的随机/生成物后状态
  不再偏离，MacroTurnPlan selector 不再因该类偏差 plan_stale。

本测试不依赖真实卡牌 / 规则引擎，用受控 spy predictor 直接观测注入的 branch_seed，
从而把「seed 派生契约」与「规则执行语义」解耦验证。
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.execution.execution_context import LocalExecutionContext
from decision_engine.planning.puct import PuctConfig, PuctSearch
from decision_engine.planning.search_ab_match import branch_seed as ab_branch_seed
from decision_engine.planning.strict_self_play import _action_seed
from decision_engine.simulation.predictor import PredictionCertainty
from decision_engine.state.snapshot import BattleSnapshot, PlayerView


# ---------------------------------------------------------------------------
# 受控 fixture：线性回合，逐层推进到 turn_end
# ---------------------------------------------------------------------------


@dataclass
class _Prediction:
    snapshot: BattleSnapshot
    certainty: PredictionCertainty = PredictionCertainty.EXACT
    unresolved_effects: tuple[str, ...] = ()


def _snapshot(depth: int, *, turn: int = 3, is_ally: bool = True, turn_ended: bool = False) -> BattleSnapshot:
    """深度标记 snapshot：me.pp 编码「本回合已打出的动作数」。

    通过 me.turn=turn 让 selector/strict 的 base 可复现（turn 固定）。
    """
    return BattleSnapshot(
        me=PlayerView(life=20, max_life=20, pp=10 - depth, max_pp=10, turn=turn),
        opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=0, turn=turn),
        cards_by_uid={},
        is_ally_turn=is_ally,
        is_turn_ended=turn_ended,
    )


class _LinearEnumerator:
    """每个局面返回 [推进动作, turn_end]。

    推进动作用 source_uid=100+depth 区分层次（depth 从 me.pp 推出）；一直可推进，
    直到 depth 达到 _max_actions 后只剩 turn_end，强制回合在固定长度内收束。
    """

    def __init__(self, max_actions: int) -> None:
        self._max_actions = max_actions

    def enumerate(self, snapshot: BattleSnapshot, *, player: str = "me") -> list[LegalAction]:
        depth = 10 - int(snapshot.me.pp)
        turn_end = LegalAction(kind=ActionKind.TURN_END, source_uid=0)
        if depth >= self._max_actions:
            return [turn_end]
        advance = LegalAction(kind=ActionKind.PLAY, source_uid=100 + depth)
        # 推进动作在前，保证高 prior / greedy 展开优先走推进边再到 turn_end。
        return [advance, turn_end]


class _SeedRecordingPredictor:
    """记录每次 predict 注入的 branch_seed 与 (source_uid)。

    推进动作 → 返回 depth+1 的 snapshot（me.pp 递减），使搜索能继续展开成链。
    turn_end → 返回 is_turn_ended=True 的终态 snapshot（本回合叶）。
    """

    def __init__(self) -> None:
        # [(source_uid, action_kind, branch_seed)]
        self.calls: list[tuple[int, str, int | None]] = []

    def predict_with_replay_context(
        self,
        snapshot: BattleSnapshot,
        action: LegalAction,
        *,
        player: str = "me",
        generated_cards: tuple = (),
        execution_context: LocalExecutionContext | None = None,
    ) -> _Prediction:
        del generated_cards, player
        seed = execution_context.branch_seed if execution_context is not None else None
        self.calls.append((action.source_uid, action.kind.value, seed))
        depth = 10 - int(snapshot.me.pp)
        if action.kind is ActionKind.TURN_END:
            return _Prediction(snapshot=_snapshot(depth, turn_ended=True))
        return _Prediction(snapshot=_snapshot(depth + 1))


class BranchSeedAlignmentTest(unittest.TestCase):
    def test_branch_seed_formula_matches_execution_side(self) -> None:
        """_branch_seed_for 与 strict / search_ab 执行侧公式逐项一致。"""
        seed, turn = 7, 3
        base = ab_branch_seed(seed, turn, 0)
        # strict_self_play._action_seed 与 search_ab_match.branch_seed 必须一致。
        self.assertEqual(base, _action_seed(seed, turn, 0))

        search = PuctSearch(
            predictor=_SeedRecordingPredictor(),
            config=PuctConfig(branch_seed=base),
        )
        action = LegalAction(kind=ActionKind.PLAY, source_uid=555)
        for action_index in range(1, 13):
            derived = search._branch_seed_for(action, action_index)
            self.assertEqual(derived, base + action_index)
            # 与执行侧两个入口都一致。
            self.assertEqual(derived, ab_branch_seed(seed, turn, action_index))
            self.assertEqual(derived, _action_seed(seed, turn, action_index))

    def test_single_step_api_backward_compatible(self) -> None:
        """action_index=None（普通单步调用）保持旧 hash(source_uid) 语义，API 兼容。"""
        base = 999
        search = PuctSearch(
            predictor=_SeedRecordingPredictor(),
            config=PuctConfig(branch_seed=base),
        )
        action = LegalAction(kind=ActionKind.PLAY, source_uid=12345)
        expected = base + hash(action.source_uid) % 10000
        self.assertEqual(search._branch_seed_for(action, None), expected)
        # 提供 action_index 时不再使用 hash，而是 base + index。
        self.assertNotEqual(search._branch_seed_for(action, 1), expected)

    def test_search_tree_depth_derives_execution_aligned_seed(self) -> None:
        """搜索树内第 N 层回合内动作预测用 base + N，与执行侧第 N 个动作一致。

        构造线性回合（推进动作逐层递进 → turn_end），用 forced deepening + UCB 展开，
        收集每层预测注入的 branch_seed，断言：第 k 层推进动作（source_uid=100+(k-1)）
        的 seed == base + k == 执行侧 branch_seed(seed, turn, k)。
        """
        seed, turn = 11, 3
        base = ab_branch_seed(seed, turn, 0)
        max_actions = 4
        predictor = _SeedRecordingPredictor()
        enumerator = _LinearEnumerator(max_actions)
        search = PuctSearch(
            enumerator=enumerator,
            child_enumerator=enumerator,
            predictor=predictor,
            config=PuctConfig(
                simulations=8,
                max_depth=10,
                max_turns=1,
                branch_seed=base,
                # 关闭 Dirichlet 噪声保证确定性线性展开。
                dirichlet_epsilon=0.0,
            ),
        )
        search.search(_snapshot(0, turn=turn), player="me")

        # 收集推进动作（source_uid=100+depth）首次出现时注入的 seed。
        # depth d 的推进动作是回合内第 (d+1) 个动作（1-based）。
        seed_by_depth: dict[int, int] = {}
        for source_uid, kind, bseed in predictor.calls:
            if kind != ActionKind.PLAY.value:
                continue
            d = source_uid - 100
            if d not in seed_by_depth and bseed is not None:
                seed_by_depth[d] = bseed

        self.assertTrue(seed_by_depth, "搜索应至少预测过一个推进动作")
        for d, bseed in seed_by_depth.items():
            action_index = d + 1  # depth d 的推进动作 = 回合内第 d+1 个动作
            self.assertEqual(
                bseed, base + action_index,
                f"depth={d} 推进动作 seed 应为 base+{action_index}",
            )
            # 与执行侧 strict / search_ab 第 action_index 个动作用同一 seed。
            self.assertEqual(bseed, ab_branch_seed(seed, turn, action_index))
            self.assertEqual(bseed, _action_seed(seed, turn, action_index))

    def test_turn_end_edge_uses_turn_internal_action_index(self) -> None:
        """turn_end 边也按回合内 1-based action_index 派生 seed，与执行侧 TURN_END 对齐。

        执行侧对一个回合内第 N 个动作（含最终的 TURN_END）用 base+N；搜索树里 turn_end
        作为某节点的边，其 action_index = 该节点 turn_action_depth + 1，应等于执行侧对
        应 TURN_END 的 action_index。
        """
        seed, turn = 5, 4
        base = ab_branch_seed(seed, turn, 0)
        max_actions = 2
        predictor = _SeedRecordingPredictor()
        enumerator = _LinearEnumerator(max_actions)
        search = PuctSearch(
            enumerator=enumerator,
            child_enumerator=enumerator,
            predictor=predictor,
            config=PuctConfig(
                simulations=8,
                max_depth=10,
                max_turns=1,
                branch_seed=base,
                dirichlet_epsilon=0.0,
            ),
        )
        search.search(_snapshot(0, turn=turn), player="me")

        # 收集 turn_end 边注入的所有 seed。turn_end 可能出现在不同回合内深度（root 后
        # 直接 turn_end、或推进若干步后 turn_end），每个都应是 base + (该处回合内深度+1)。
        turn_end_seeds = [b for (uid, kind, b) in predictor.calls
                          if kind == ActionKind.TURN_END.value and b is not None]
        self.assertTrue(turn_end_seeds, "搜索应至少预测过一次 turn_end 边")
        # 所有 turn_end seed 必须落在 [base+1, base+max_actions+1] 且等于某个
        # 执行侧 action_index 的派生值（回合内合法深度范围）。
        for b in turn_end_seeds:
            action_index = b - base
            self.assertGreaterEqual(action_index, 1)
            self.assertLessEqual(action_index, max_actions + 1)
            self.assertEqual(b, ab_branch_seed(seed, turn, action_index))


if __name__ == "__main__":
    unittest.main()
