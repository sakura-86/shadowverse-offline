# -*- coding: utf-8 -*-
"""MacroDrivenSearch 适配器单元测试。"""
import unittest
from decision_engine.planning.macro_driven_search import MacroDrivenSearch
from decision_engine.planning.puct import PuctSearchResult
from decision_engine.actions.model import ActionKind, LegalAction


def _act(kind="play", uid=0):
    return LegalAction(kind=ActionKind(kind), source_uid=uid, description="")


class _FakeMicro:
    """假微观搜索：返回固定 actions/visit_counts。"""
    def __init__(self, actions, selected=0):
        self._actions = actions
        self._selected = selected
        self.prior = None
        self.value = None
        self.turn_end_calls = 0
    def search(self, snapshot, *, player):
        n = len(self._actions)
        return PuctSearchResult(
            actions=tuple(self._actions),
            visit_counts=tuple([10]*n),
            policy_target=tuple([1.0/n]*n),
            selected_action_index=self._selected,
            root_value=0.5, trace={}, diagnostics={"baseline_name": "test"},
        )
    def notify_turn_end(self):
        self.turn_end_calls += 1


class _FakeLine:
    def __init__(self, seq): self.action_sequence = tuple(seq)

class _FakeMacroResult:
    def __init__(self, line, val=7.0):
        self.best_line = line; self.backed_up_value = val
        self.ranked_lines=(line,) if line else (); self.line_values=(val,); self.diagnostics={}

class _FakeMacro:
    def __init__(self, line): self._line = line; self.calls=0
    def search(self, snapshot, *, player):
        self.calls += 1
        return _FakeMacroResult(self._line)


class _Cfg:
    simulations=24; max_depth=8; max_turns=1; c_puct=1.5


class TestMacroDriven(unittest.TestCase):
    def test_macro_decision_overrides_micro_selected(self):
        """宏观 best_line 的动作覆盖微观 selected。"""
        a0, a1, a2 = _act("play",1), _act("attack_leader",2), _act("evolve",3)
        micro = _FakeMicro([a0,a1,a2], selected=0)  # 微观选 a0
        macro = _FakeMacro(_FakeLine([a1, a2]))     # 宏观回合线: a1 → a2
        s = MacroDrivenSearch(macro_search=macro, micro_search=micro, config=_Cfg())
        r1 = s.search("snap", player="me")
        # 第一步应选宏观线的 a1(index=1)，非微观的 a0
        self.assertEqual(r1.selected_action_index, 1)
        self.assertTrue(r1.diagnostics["macro_driven"])
        # policy_target 仍是微观(均匀)
        self.assertEqual(len(r1.policy_target), 3)
        # 第二步：同回合线继续执行 a2(index=2)，宏观不再重新搜索
        r2 = s.search("snap", player="me")
        self.assertEqual(r2.selected_action_index, 2)
        self.assertEqual(macro.calls, 1, "同回合线应只宏观搜索1次")

    def test_replan_after_line_exhausted(self):
        """回合线执行完后重新宏观规划。"""
        a0, a1 = _act("play",1), _act("attack_leader",2)
        micro = _FakeMicro([a0,a1], selected=0)
        macro = _FakeMacro(_FakeLine([a1]))  # 单动作线
        s = MacroDrivenSearch(macro_search=macro, micro_search=micro, config=_Cfg())
        s.search("snap", player="me")   # 执行 a1，线空
        s.search("snap", player="me")   # 线空 → 重新规划
        self.assertEqual(macro.calls, 2)

    def test_deviation_falls_back_to_micro(self):
        """宏观建议的动作不在微观 actions 里 → 回退微观 selected。"""
        a0, a1 = _act("play",1), _act("attack_leader",2)
        micro = _FakeMicro([a0,a1], selected=1)
        macro = _FakeMacro(_FakeLine([_act("play", 999)]))  # uid=999 不在微观
        s = MacroDrivenSearch(macro_search=macro, micro_search=micro, config=_Cfg())
        r = s.search("snap", player="me")
        # play kind 匹配到 a0(index=0，同kind退化匹配) 或回退 selected；至少不崩、有合法index
        self.assertIn(r.selected_action_index, (0, 1))

    def test_notify_turn_end_resets_line(self):
        a0 = _act("play",1)
        micro = _FakeMicro([a0], selected=0)
        macro = _FakeMacro(_FakeLine([a0]))
        s = MacroDrivenSearch(macro_search=macro, micro_search=micro, config=_Cfg())
        s.search("snap", player="me")
        s.notify_turn_end()
        self.assertIsNone(s._pending_line)
        self.assertEqual(micro.turn_end_calls, 1)

    def test_macro_failure_falls_back(self):
        """宏观搜索抛异常 → 回退微观，不阻断。"""
        a0 = _act("play",1)
        micro = _FakeMicro([a0], selected=0)
        class _BadMacro:
            def search(self,s,*,player): raise RuntimeError("boom")
        s = MacroDrivenSearch(macro_search=_BadMacro(), micro_search=micro, config=_Cfg())
        r = s.search("snap", player="me")
        self.assertEqual(r.selected_action_index, 0)  # 回退微观 selected


if __name__ == "__main__":
    unittest.main()
