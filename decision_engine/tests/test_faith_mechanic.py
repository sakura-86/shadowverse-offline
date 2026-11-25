# -*- coding: utf-8 -*-
"""护符教信仰(faith)机制回归测试。

覆盖 bug: 古旧天书·莲妥丝(10664120)回合结束无条件信仰-10 导致 faith_value 变负
(0/-10/-20/...), 且信仰纹章(10664122)"护符被破坏时+1"累积语义缺失导致信仰涨不上去。

权威规则:
1. faith_value 是非负累积量, 绝不为负。
2. 10664120 的"信仰-10 + 加天书深渊入手"仅当 me.faith_value>=10 时发动(同一条件门控)。
3. 10664122 响应 on_self_amulet_destroyed 使信仰+1 累积。
"""

import unittest

from decision_engine.cards.templates import (
    AmountFormula,
    ConditionSpec,
    ExecutableEffect,
    OperationSpec,
    TriggerSpec,
)
from decision_engine.execution import (
    EffectInterpretationContext,
    EffectInterpreter,
)
from decision_engine.execution.condition_evaluator import (
    ConditionEvaluationContext,
    ConditionEvaluator,
)
from decision_engine.cards.template_loader import (
    invalidate_template_cache,
    load_card_templates_by_base_card_id,
)
from decision_engine.actions.model import ActionKind, LegalAction
from decision_engine.state.snapshot import (
    BattleSnapshot,
    CardType,
    CardView,
    CrestView,
    PlayerView,
)

TEMPLATES_PATH = "decision_engine/data/public/card_templates.json"
FAITH_CREST_ID = 10664122


def _snapshot_with_faith(faith: int) -> BattleSnapshot:
    me = PlayerView(
        life=20,
        max_life=20,
        pp=10,
        max_pp=10,
        crests=(CrestView(card_id=FAITH_CREST_ID, unique_id=1, faith_value=faith),),
    )
    opponent = PlayerView(life=20, max_life=20, pp=0, max_pp=10)
    return BattleSnapshot(me=me, opponent=opponent, cards_by_uid={}, source="test")


def _legal_action() -> LegalAction:
    return LegalAction(kind=ActionKind.TURN_END, source_uid=0)


class FaithGateConditionTest(unittest.TestCase):
    """信仰>=10 条件门控: 扣减与生成同受此条件约束。"""

    def _eval_gate(self, faith: int) -> bool:
        snapshot = _snapshot_with_faith(faith)
        effect = ExecutableEffect(
            effect_id="test_faith_gate",
            trigger=TriggerSpec(event="turn_end"),
            conditions=(
                ConditionSpec(kind="snapshot_threshold", subject="me.faith_value", operator=">=", value=10),
            ),
        )
        evaluator = ConditionEvaluator()
        ctx = ConditionEvaluationContext(action=_legal_action())
        return evaluator.evaluate_all(snapshot, effect, effect.conditions, ctx).matched

    def test_faith_below_10_gate_blocks(self):
        # (a) 信仰<10: 门控不满足 -> 扣减与生成都不发动
        for faith in (0, 5, 9):
            self.assertFalse(self._eval_gate(faith), f"faith={faith} 应当门控失败")

    def test_faith_at_or_above_10_gate_opens(self):
        # (b) 信仰>=10: 门控满足 -> 扣减与生成都发动
        for faith in (10, 11, 20):
            self.assertTrue(self._eval_gate(faith), f"faith={faith} 应当门控通过")


class FaithNonNegativeTest(unittest.TestCase):
    """modify_crest_value 对 faith_value 的非负约束。"""

    def _modify(self, start: int, amount: int) -> int:
        snapshot = _snapshot_with_faith(start)
        interpreter = EffectInterpreter()
        result = interpreter.apply_operations(
            snapshot,
            (
                OperationSpec(
                    op="modify_crest_value",
                    target_ref="self_player",
                    amount=AmountFormula(kind="fixed", value=amount),
                    params={"crest_card_ids": [FAITH_CREST_ID]},
                ),
            ),
            EffectInterpretationContext(source_uid=0),
        )
        return result.snapshot.me.crests[0].faith_value

    def test_subtract_below_zero_clamps(self):
        # (c) faith=0 扣10 -> 仍为0, 不变负
        self.assertEqual(self._modify(0, -10), 0)
        # faith=5 扣10 -> 0
        self.assertEqual(self._modify(5, -10), 0)
        # faith=10 扣10 -> 0
        self.assertEqual(self._modify(10, -10), 0)

    def test_positive_accumulation_preserved(self):
        # 正常累积不受 clamp 影响
        self.assertEqual(self._modify(3, 1), 4)
        self.assertEqual(self._modify(9, 1), 10)
        self.assertEqual(self._modify(12, -10), 2)

    def test_accumulation_never_goes_negative(self):
        # (c) 模拟累积过程: 多次+1 到>=10, 再-10, 全程非负
        faith = 0
        snapshot = _snapshot_with_faith(faith)
        interpreter = EffectInterpreter()
        values = [faith]
        # 连续 +1 共 12 次
        for _ in range(12):
            res = interpreter.apply_operations(
                snapshot,
                (OperationSpec(op="modify_crest_value", target_ref="self_player",
                               amount=AmountFormula(kind="fixed", value=1),
                               params={"crest_card_ids": [FAITH_CREST_ID]}),),
                EffectInterpretationContext(source_uid=0),
            )
            snapshot = res.snapshot
            values.append(snapshot.me.crests[0].faith_value)
        # -10 一次
        res = interpreter.apply_operations(
            snapshot,
            (OperationSpec(op="modify_crest_value", target_ref="self_player",
                           amount=AmountFormula(kind="fixed", value=-10),
                           params={"crest_card_ids": [FAITH_CREST_ID]}),),
            EffectInterpretationContext(source_uid=0),
        )
        values.append(res.snapshot.me.crests[0].faith_value)
        self.assertEqual(values[-2], 12)   # 累积到 12
        self.assertEqual(values[-1], 2)    # 12-10=2
        self.assertTrue(all(v >= 0 for v in values), f"信仰出现负值: {values}")


class FaithTemplateModelingTest(unittest.TestCase):
    """锁定数据修复: 模板中门控条件与累积效果必须存在。"""

    @classmethod
    def setUpClass(cls):
        invalidate_template_cache()
        cls.tm = load_card_templates_by_base_card_id(TEMPLATES_PATH)

    def test_10664120_faith_effects_gated(self):
        t = self.tm[10664120]
        gated = {}
        for pe in t.persistent_effects:
            for ee in pe.executable_effects:
                gated[ee.effect_id] = any(
                    c.subject == "me.faith_value" and c.operator == ">=" and int(c.value) == 10
                    for c in ee.conditions
                )
        # 扣减(auto_003)与生成天书深渊(auto_002)都必须被 faith>=10 门控
        self.assertTrue(gated.get("10664120_auto_002"), "加天书深渊入手未门控")
        self.assertTrue(gated.get("10664120_auto_003"), "信仰-10 未门控")

    def test_10664122_faith_accumulation_modeled(self):
        f = self.tm[FAITH_CREST_ID]
        found = False
        for pe in f.persistent_effects:
            if pe.trigger is not None and pe.trigger.event == "on_self_amulet_destroyed":
                for ee in pe.executable_effects:
                    for op in ee.operations:
                        if op.op == "modify_crest_value" and op.amount is not None and op.amount.value == 1:
                            found = True
        self.assertTrue(found, "10664122 缺少'护符被破坏时信仰+1'累积效果")


class FaithTemplateExecutionTest(unittest.TestCase):
    """用真实模板中的 operations 端到端执行, 验证累积+门控行为。"""

    @classmethod
    def setUpClass(cls):
        invalidate_template_cache()
        cls.tm = load_card_templates_by_base_card_id(TEMPLATES_PATH)

    def _faith_gain_ops(self):
        f = self.tm[FAITH_CREST_ID]
        for pe in f.persistent_effects:
            if pe.trigger is not None and pe.trigger.event == "on_self_amulet_destroyed":
                return pe.executable_effects[0].operations
        raise AssertionError("未找到信仰累积 effect")

    def _steal_ops(self):
        """返回 (信仰-10 op, 其门控 condition)。"""
        t = self.tm[10664120]
        for pe in t.persistent_effects:
            for ee in pe.executable_effects:
                if ee.effect_id == "10664120_auto_003":
                    return ee
        raise AssertionError("未找到信仰-10 effect")

    def test_real_template_accumulation_from_zero_is_nonnegative(self):
        # 用真实模板 +1 op 累积, 从 0 开始多次破坏护符, faith 正常上涨且非负
        ops = self._faith_gain_ops()
        snapshot = _snapshot_with_faith(0)
        interpreter = EffectInterpreter()
        for expected in range(1, 11):
            res = interpreter.apply_operations(snapshot, ops, EffectInterpretationContext(source_uid=0))
            snapshot = res.snapshot
            self.assertEqual(snapshot.me.crests[0].faith_value, expected)
        self.assertTrue(snapshot.me.crests[0].faith_value >= 0)

    def test_real_template_steal_gated_and_nonnegative(self):
        # 真实模板 -10 op: 配合门控 condition 判定
        ee = self._steal_ops()
        self.assertTrue(
            any(c.subject == "me.faith_value" and c.operator == ">=" and int(c.value) == 10 for c in ee.conditions),
            "信仰-10 effect 未被 faith>=10 门控",
        )
        evaluator = ConditionEvaluator()
        ctx = ConditionEvaluationContext(action=_legal_action())
        # faith=5: 门控阻断, 不应扣减
        self.assertFalse(evaluator.evaluate_all(_snapshot_with_faith(5), ee, ee.conditions, ctx).matched)
        # faith=10: 门控通过
        self.assertTrue(evaluator.evaluate_all(_snapshot_with_faith(10), ee, ee.conditions, ctx).matched)
        # 即便执行 -10 op, 结果也被 op 层 clamp 保持非负
        interpreter = EffectInterpreter()
        res = interpreter.apply_operations(_snapshot_with_faith(10), ee.operations, EffectInterpretationContext(source_uid=0))
        self.assertEqual(res.snapshot.me.crests[0].faith_value, 0)


class FaithCrestDestroyEventTest(unittest.TestCase):
    """端到端: 纹章作为 on_self_amulet_destroyed 订阅者, 护符被破坏后 faith+1。

    回归 bug: _source_cards 的 destroy 分支未枚举 crests + _card_owner_side 不检查
    crests, 导致信仰纹章永远不响应护符破坏事件, 端到端 faith 恒为 0。
    """

    @classmethod
    def setUpClass(cls):
        from decision_engine.cards.template_loader import (
            invalidate_template_cache,
            load_card_templates_by_base_card_id,
        )
        invalidate_template_cache()
        cls.tm = load_card_templates_by_base_card_id(TEMPLATES_PATH)

    def _build_handler(self):
        from decision_engine.execution.template_event_handler import (
            TemplateEventHandler,
            EventEffectState,
        )
        return TemplateEventHandler(
            templates_by_base_card_id=self.tm,
            effect_interpreter=EffectInterpreter(),
            action=_legal_action(),
            generated_cards=(),
            state=EventEffectState(consumed=[], unresolved=[]),
            effect_unresolved_reasons=lambda *a, **k: (),
        )

    def test_card_owner_side_resolves_crest(self):
        from decision_engine.execution.template_event_handler import TemplateEventHandler
        amulet_uid = 101
        amulet = CardView(uid=amulet_uid, card_id=90001, base_card_id=90001,
                          name="amulet", card_type=CardType.AMULET)
        me = PlayerView(life=20, max_life=20, pp=10, max_pp=10, field=(amulet_uid,),
                        crests=(CrestView(card_id=FAITH_CREST_ID, unique_id=1, faith_value=0),),
                        unique_id=1)
        snap = BattleSnapshot(me=me, opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=10, unique_id=2),
                              cards_by_uid={amulet_uid: amulet}, source="test")
        # 场上护符 uid -> 'me'; 纹章 unique_id -> 'me' (修复前返回 None)
        self.assertEqual(TemplateEventHandler._card_owner_side(snap, amulet_uid), "me")
        self.assertEqual(TemplateEventHandler._card_owner_side(snap, 1), "me")
        # 对手侧纹章
        opp = PlayerView(life=20, max_life=20, pp=0, max_pp=10, unique_id=2,
                         crests=(CrestView(card_id=FAITH_CREST_ID, unique_id=5, faith_value=0),))
        snap2 = BattleSnapshot(me=me, opponent=opp, cards_by_uid={amulet_uid: amulet}, source="test")
        self.assertEqual(TemplateEventHandler._card_owner_side(snap2, 5), "opponent")

    def _destroy_amulet_and_get_faith(self, start_faith: int) -> int:
        from decision_engine.execution.events import ExecutionEvent, EventQueue
        amulet_uid = 101
        # 破坏后的 snapshot: 护符已离场, 纹章仍在
        snap_after = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=10, max_pp=10, field=(),
                          crests=(CrestView(card_id=FAITH_CREST_ID, unique_id=1, faith_value=start_faith),),
                          unique_id=1),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=10, unique_id=2),
            cards_by_uid={}, source="test",
        )
        event = ExecutionEvent(
            "destroy", source_uid=amulet_uid, target_uid=amulet_uid, player="me",
            payload={"leaving_card": {"base_card_id": 90001, "card_id": 90001,
                                      "name": "amulet", "card_type": "amulet", "has_guard": False}},
        )
        dispatch = EventQueue([event]).dispatch(snap_after, (self._build_handler(),))
        return dispatch.snapshot.me.crests[0].faith_value

    def test_amulet_destroy_accumulates_faith(self):
        # 核心验收: 破坏自己的护符后 faith +1
        self.assertEqual(self._destroy_amulet_and_get_faith(0), 1)
        self.assertEqual(self._destroy_amulet_and_get_faith(4), 5)
        self.assertEqual(self._destroy_amulet_and_get_faith(9), 10)

    def test_opponent_amulet_destroy_does_not_accumulate_my_faith(self):
        # 不破坏既有触发: 对手护符被破坏 (event.player='opponent') 不应给我方纹章 +1
        from decision_engine.execution.events import ExecutionEvent, EventQueue
        amulet_uid = 201
        snap_after = BattleSnapshot(
            me=PlayerView(life=20, max_life=20, pp=10, max_pp=10, field=(),
                          crests=(CrestView(card_id=FAITH_CREST_ID, unique_id=1, faith_value=0),),
                          unique_id=1),
            opponent=PlayerView(life=20, max_life=20, pp=0, max_pp=10, unique_id=2),
            cards_by_uid={}, source="test",
        )
        event = ExecutionEvent(
            "destroy", source_uid=amulet_uid, target_uid=amulet_uid, player="opponent",
            payload={"leaving_card": {"base_card_id": 90001, "card_id": 90001,
                                      "name": "amulet", "card_type": "amulet", "has_guard": False}},
        )
        dispatch = EventQueue([event]).dispatch(snap_after, (self._build_handler(),))
        # 我方纹章 subscriber_side='me' != event.player='opponent' -> scope 不匹配
        self.assertEqual(dispatch.snapshot.me.crests[0].faith_value, 0)


class FaithCountdownAmuletExpireEventTest(unittest.TestCase):
    """端到端(走真实 start_turn): 倒计时护符倒计时归零而离场, 算作"被破坏",
    应触发 on_self_amulet_destroyed -> 信仰纹章 faith +1。

    回归 bug: rule_kernel.start_turn 的倒计时结算在 next_countdown<=0 时只发
    MOVE_TO_GRAVEYARD + last_words, 不发 DESTROY, 导致护符自然到期永不触发
    on_self_amulet_destroyed, 护符教信仰无法在护符到期时累积。
    修复: 补发带 leaving_card payload(含 card_type) 的 DESTROY 事件。
    """

    COUNTDOWN_AMULET_ID = 10663210

    @classmethod
    def setUpClass(cls):
        invalidate_template_cache()
        cls.tm = load_card_templates_by_base_card_id(TEMPLATES_PATH)

    def _build_handler(self):
        from decision_engine.execution.template_event_handler import (
            TemplateEventHandler,
            EventEffectState,
        )
        return TemplateEventHandler(
            templates_by_base_card_id=self.tm,
            effect_interpreter=EffectInterpreter(),
            action=_legal_action(),
            generated_cards=(),
            state=EventEffectState(consumed=[], unresolved=[]),
            effect_unresolved_reasons=lambda *a, **k: (),
        )

    def _run_start_turn(self, start_faith, *, countdown=1, n_amulets=1, player="me",
                        with_my_crest_when_opponent=False):
        """构造 countdown 倒计时护符 + 信仰纹章, 走真实 start_turn + 事件分发,
        返回 (faith_of_owner_crest 或 my_faith, owner_field_left)。"""
        from decision_engine.execution.rule_kernel import CoreRuleKernel
        from decision_engine.execution.events import EventQueue
        uids = [101 + i for i in range(n_amulets)]
        cards = {
            u: CardView(uid=u, card_id=self.COUNTDOWN_AMULET_ID,
                        base_card_id=self.COUNTDOWN_AMULET_ID, name="countdown_amulet",
                        card_type=CardType.COUNTDOWN_AMULET, countdown=countdown)
            for u in uids
        }
        owner = PlayerView(
            life=20, max_life=20, pp=10, max_pp=10, turn=1, field=tuple(uids),
            crests=(CrestView(card_id=FAITH_CREST_ID, unique_id=1, faith_value=start_faith),),
            unique_id=1, deck_count=10,
        )
        if player == "me":
            me, opp = owner, PlayerView(life=20, max_life=20, pp=0, max_pp=10,
                                        unique_id=2, deck_count=10)
        else:
            # 对手的倒计时护符到期; 我方可带自己的信仰纹章以验证 side 不串
            my_crests = (CrestView(card_id=FAITH_CREST_ID, unique_id=9, faith_value=0),) \
                if with_my_crest_when_opponent else ()
            me = PlayerView(life=20, max_life=20, pp=0, max_pp=10, unique_id=2,
                            deck_count=10, crests=my_crests)
            opp = owner
        snap = BattleSnapshot(me=me, opponent=opp, cards_by_uid=cards, source="test")
        kernel = CoreRuleKernel()
        # defer_draw=True 复刻 PlayerLocalPredictor 的分阶段结算路径
        result = kernel.start_turn(snap, player, defer_draw=True)
        dispatch = EventQueue(list(result.events)).dispatch(result.snapshot, (self._build_handler(),))
        final = dispatch.snapshot
        owner_side = final.me if player == "me" else final.opponent
        return final, owner_side

    def test_single_countdown_amulet_expire_accumulates_faith(self):
        # 核心验收: countdown=1 护符到期 -> 护符离场 且 faith N -> N+1
        for start in (0, 3, 9):
            final, owner_side = self._run_start_turn(start, countdown=1, n_amulets=1, player="me")
            self.assertEqual(len(owner_side.field), 0, "护符应已到期离场")
            self.assertEqual(owner_side.crests[0].faith_value, start + 1,
                             f"start_faith={start} 到期后应 +1")

    def test_multiple_countdown_amulets_expire_same_turn(self):
        # 多个倒计时护符同回合到期 -> faith 增加对应次数
        for n in (2, 3):
            final, owner_side = self._run_start_turn(0, countdown=1, n_amulets=n, player="me")
            self.assertEqual(len(owner_side.field), 0)
            self.assertEqual(owner_side.crests[0].faith_value, n,
                             f"{n} 个护符同回合到期应 +{n}")

    def test_non_expiring_countdown_amulet_does_not_accumulate(self):
        # countdown=2 未归零: 护符留场, faith 不变
        final, owner_side = self._run_start_turn(3, countdown=2, n_amulets=1, player="me")
        self.assertEqual(len(owner_side.field), 1, "countdown=2 护符不应离场")
        self.assertEqual(owner_side.crests[0].faith_value, 3, "未到期不应 +1")

    def test_opponent_countdown_amulet_expire_does_not_touch_my_faith(self):
        # 对手回合、对手倒计时护符到期: 我方信仰纹章不应 +1 (side 正确)
        final, _ = self._run_start_turn(0, countdown=1, n_amulets=1, player="opponent",
                                        with_my_crest_when_opponent=True)
        self.assertEqual(final.me.crests[0].faith_value, 0,
                         "对手护符到期不应给我方纹章 +1")
        # 对手自己的纹章则 +1 (其 subscriber_side='opponent'==event.player)
        self.assertEqual(final.opponent.crests[0].faith_value, 1,
                         "对手护符到期应给对手自己纹章 +1")


if __name__ == "__main__":
    unittest.main()
