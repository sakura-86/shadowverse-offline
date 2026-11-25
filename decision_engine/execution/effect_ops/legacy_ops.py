"""legacy_ops：尚未物理迁出到领域模块的 op，以委托方式注册到 OP_REGISTRY。

目的：让 apply_operation 完全由注册表驱动（移除 130 行 if-elif），同时保持这些 op
的实现暂留在 EffectInterpreter 私有方法上，行为零变化。后续可把它们逐组物理迁移到
combat_ops / resource_ops / zone_ops / hand_ops / generated_ops / control_ops，
迁移时只需把注册从这里移走，无需再动 dispatch。
拆分方案见 docs/EFFECT_INTERPRETER_REFACTOR_PLAN.md。
"""

from __future__ import annotations

from .base import register_op

# op 名 -> EffectInterpreter 私有方法名
_DELEGATIONS: dict[str, str] = {
    # control: 已迁移到 control_ops.py
    # combat: 已迁移到 combat_ops.py
    # resource: 已迁移到 resource_ops.py（draw 仍在 legacy，属 hand/generated 复杂组）
    "draw": "_draw",
    # zone
    "summon": "_summon",
    "invoke_self": "_invoke_self",
    "deck_ability_activation": "_deck_ability_activation",
    "bounce": "_bounce",
    "return_to_deck": "_return_to_deck",
    "transform": "_transform",
    # hand / generated cards
    "put_token_to_hand": "_put_token_to_hand",
    "add_to_hand": "_add_to_hand",
    "replace_hand_card": "_replace_hand_card",
    "copy_card": "_copy_card",
    "reanimate": "_reanimate",
    "history_same_name_pool": "_history_same_name_pool",
    "copy_from_opponent_hand": "_copy_from_opponent_hand",
    "deck_operation": "_deck_operation",
}

# 别名（多 op key 指向同一方法）
_ALIASES: dict[str, str] = {
    "spellboost": "_spellboost",
    "spellboost_hand": "_spellboost",
    "discard": "_discard_hand_card",
    "discard_hand_card": "_discard_hand_card",
}


def _make_delegate(method_name: str):
    def _delegate(snapshot, operation, context, ctx):
        return getattr(ctx.interpreter, method_name)(snapshot, operation, context)

    return _delegate


for _op_name, _method in {**_DELEGATIONS, **_ALIASES}.items():
    register_op(_op_name)(_make_delegate(_method))
