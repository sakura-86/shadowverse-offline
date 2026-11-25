"""effect_ops 包基础设施：错误类型、共享上下文、operation 注册表。

本模块是 effect_ops 子包的最底层，不依赖包内其他模块，避免循环导入。
拆分方案见 docs/EFFECT_INTERPRETER_REFACTOR_PLAN.md。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..events import EventKind, ExecutionEvent

if TYPE_CHECKING:
    from decision_engine.cards.templates import OperationSpec
    from decision_engine.state.snapshot import BattleSnapshot
    from ..effect_interpreter import (
        EffectInterpretationContext,
        EffectInterpretationResult,
        EffectInterpreter,
    )
    from ..transactions import SnapshotTransaction


class EffectInterpreterError(RuntimeError):
    """效果解释器错误。

    历史上定义在 effect_interpreter.py，为消除 resolution/op 模块与解释器之间的
    循环依赖而下沉到 base.py。effect_interpreter.py 仍 re-export 该符号以保持
    对外 API 不变。
    """


@dataclass
class OpContext:
    """op 执行器的共享上下文。

    渐进式重构（strangler）期间，OpContext 持有 EffectInterpreter 引用并把解析/执行
    能力委托回去。待所有 op 执行器迁出、interpreter 只剩编排后，再把解析实现整体
    移入 resolution 模块。这样 B1 阶段不搬运任何实现代码，风险最低。
    拆分方案见 docs/EFFECT_INTERPRETER_REFACTOR_PLAN.md。
    """

    interpreter: "EffectInterpreter"

    @property
    def kernel(self):
        return self.interpreter.kernel

    @property
    def card_resolver(self):
        return self.interpreter.card_resolver

    def run(
        self,
        snapshot: "BattleSnapshot",
        transactions: tuple["SnapshotTransaction", ...],
        events: tuple["ExecutionEvent", ...],
    ) -> "EffectInterpretationResult":
        return self.interpreter._run(snapshot, transactions, events)

    def dispatch(
        self,
        snapshot: "BattleSnapshot",
        operation: "OperationSpec",
        context: "EffectInterpretationContext",
    ) -> "EffectInterpretationResult":
        """供 repeat 等需要回调 apply_operation 的 op 使用。"""
        return self.interpreter.apply_operation(snapshot, operation, context)

    def operation_player(self, operation: "OperationSpec", context: "EffectInterpretationContext") -> str:
        return self.interpreter._operation_player(operation, context)

    def amount(self, operation: "OperationSpec", *, default: int = 0, snapshot=None, context=None) -> int:
        return self.interpreter._amount(operation, default=default, snapshot=snapshot, context=context)

    def target_uids(self, snapshot, operation, context) -> tuple:
        return self.interpreter._target_uids(snapshot, operation, context)

    def resolve_amount_value(self, value, *, snapshot=None, context=None) -> int:
        return self.interpreter._resolve_amount_value(value, snapshot=snapshot, context=context)

    def attack_limit_transactions(self, snapshot, target_uids, amount, **kwargs) -> tuple:
        return self.interpreter._attack_limit_transactions(snapshot, target_uids, amount, **kwargs)

    def owner_of_field_uid(self, snapshot, uid):
        return self.interpreter._owner_of_field_uid(snapshot, uid)

    def field_death_transactions(self, snapshot, damaged) -> list:
        return self.interpreter._field_death_transactions(snapshot, damaged)

    def field_destroy_events(self, snapshot, damaged, *, source_uid: int | None = None) -> tuple:
        """属性归零致死时生成 DESTROY 事件。

        ``stat_buff`` / ``set_stat`` 只改属性，不像 ``damage`` 那样天然带破坏事件；
        统一在这里生成，保证不同 op 的死亡事件口径一致，谢幕曲不再因走哪条 op
        而丢失。
        """
        events: list[ExecutionEvent] = []
        for uid, life_after in damaged:
            if life_after > 0:
                continue
            card = snapshot.cards_by_uid.get(uid)
            if card is None:
                continue
            owner = self.owner_of_field_uid(snapshot, uid)
            if owner is None:
                continue
            events.append(ExecutionEvent(
                EventKind.DESTROY,
                source_uid=source_uid,
                target_uid=uid,
                player=owner,
                payload={"leaving_card": {
                    "base_card_id": card.base_card_id,
                    "card_id": card.card_id,
                    "name": card.name,
                    "has_guard": card.has_guard,
                    "card_type": card.card_type.value,
                }},
            ))
        return tuple(events)

    def validate_field_targets(self, snapshot, target_uids, player, target_ref, *, allow_missing: bool = True) -> tuple:
        return self.interpreter._validate_field_targets(snapshot, target_uids, player, target_ref, allow_missing=allow_missing)

    def random_target_candidates(self, snapshot, target_ref, *, player="me", opponent="opponent", source_uid=None) -> tuple:
        return self.interpreter.random_target_candidates(snapshot, target_ref, player=player, opponent=opponent, source_uid=source_uid)

    def apply_field_card_operation(self, snapshot, operation, context, kind):
        return self.interpreter._apply_field_card_operation(snapshot, operation, context, kind)


# op handler 签名：(snapshot, operation, context, op_context) -> EffectInterpretationResult
OpHandler = Callable[
    ["BattleSnapshot", "OperationSpec", "EffectInterpretationContext", "OpContext"],
    "EffectInterpretationResult",
]


OP_REGISTRY: dict[str, OpHandler] = {}


def register_op(*names: str) -> Callable[[OpHandler], OpHandler]:
    """把一个 op handler 注册到全局 OP_REGISTRY，可注册多个别名 key。"""

    def _decorator(handler: OpHandler) -> OpHandler:
        for name in names:
            if name in OP_REGISTRY:
                raise EffectInterpreterError(f"duplicate op handler registration: {name}")
            OP_REGISTRY[name] = handler
        return handler

    return _decorator
