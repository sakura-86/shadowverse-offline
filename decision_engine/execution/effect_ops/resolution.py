"""effect_ops 共享解析工具：filter/tribes/keyword/compare 等无状态纯函数。

B0 阶段先迁入模块级纯函数。后续批次将把 EffectInterpreter 里的共享解析方法
（amount/target/filtered_targets/validate_field_targets/condition 等）迁入本模块。
拆分方案见 docs/EFFECT_INTERPRETER_REFACTOR_PLAN.md。
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .base import EffectInterpreterError

if TYPE_CHECKING:
    from decision_engine.cards.templates import OperationSpec
    from decision_engine.state.snapshot import CardView


def filter_value(filters: tuple[dict[str, Any], ...], field: str) -> Any:
    for item in filters:
        if item.get("field") == field:
            return item.get("value")
    return None


# 卡面「创造物」= tribe 14（复仇者 token 类别）。`card_database.json` 中 90071130/
# 90071140/90071150/90071160 等创造物卡片的 tribes 均为 [14]，共 21 张。
# 「自己的创造物·随从进入战场时…」这类条件按 tribe 判定，不按卡名后缀匹配。
ARTIFACT_TRIBE = 14


def effective_tribes(card: "CardView") -> tuple[int, ...]:
    values = [tribe for tribe in card.tribes if tribe != 0]
    if card.buff is not None:
        values.extend(tribe for tribe in card.buff.add_tribe if tribe != 0)
    if not values:
        # 协议帧不下发 tribes 时回退到卡面模板（与 target_selector_materializer 的同名
        # 函数一致）。否则「破坏自己场上 tribe 21（证据护符）的随从/护符」这类 filter
        # 永远匹配不到（replay_...468408 st92：11073310 的 tribe21 过滤返回空集）。
        from ..generated_card_materializer import default_card_resolver

        base = default_card_resolver().get_base_by_card_id(card.base_card_id)
        if base is not None:
            values = [tribe for tribe in (base.tribes or ()) if tribe != 0]
    return tuple(dict.fromkeys(values))


def keyword_values(card: "CardView") -> tuple[str, ...]:
    values = []
    if card.has_guard:
        values.append("guard")
    if card.has_storm:
        values.append("storm")
    if card.has_rush:
        values.append("rush")
    if card.has_bane:
        values.append("bane")
    if card.has_drain:
        values.append("drain")
    if card.has_cant_be_attacked:
        values.append("intimidate")
    if card.has_ambush:
        values.append("ambush")
    if card.has_temp_shield:
        values.append("barrier")
    if card.has_last_words:
        values.append("last_words")
    return tuple(values)


def explicit_random_target_repeat(operation: "OperationSpec") -> int | None:
    """随机目标类 operation 在 params 里显式声明的「取几个目标」。

    spec 里这三者语义等价、可以互换出现：`repeat`（重复次数）、
    `count`（目标数量）、`random_count`（随机目标数量）。例如
    `10964120_auto_005`「对对手战场上的随机 2 个随从造成 3 点伤害」写的是
    `{"count": 2, "random_count": 2}`，`10401110_auto_001` 同样只给 `count`。

    返回 ``None`` 表示 params 没写数量，此时执行端退回 ``operation.amount``
    —— 那是「随机 N 个随从」的另一套写法：`destroy random_enemy_followers`
    用 `AmountFormula(kind="snapshot_count_diff")`（敌我场面差）表达 N，
    见 ``test_random_enemy_followers_uses_snapshot_count_diff_amount``。
    两种写法不能混：`damage random_enemy_followers` 的 amount 是**伤害值**，
    把 fixed 3 当成「取 3 个目标」会多做两次伤害。

    执行端（`EffectInterpreter._target_uids`）与验收端
    （`replay_validator._random_target_candidate_groups` 枚举随机分支）必须
    用同一个解析，否则验收端会按 1 个目标枚举全部候选组合，永远碰不到
    「打 2 个」的那一支。
    """
    params = operation.params or {}
    for key in ("repeat", "count", "random_count"):
        value = params.get(key)
        if value is None:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def compare_value(current: Any, operator: str, expected: Any) -> bool:
    if operator == "==":
        return current == expected
    if operator == "<=":
        return current <= expected
    if operator == ">=":
        return current >= expected
    if operator == "in":
        return current in expected
    if operator == "contains":
        return expected in current
    if operator == "not_contains":
        return expected not in current
    raise EffectInterpreterError(f"unsupported filter operator: {operator}")


def unsupported_operation_message(operation: "OperationSpec") -> str:
    return (
        f"unsupported effect operation: op={operation.op} "
        f"target_ref={operation.target_ref or ''} "
        f"params={','.join(sorted(str(key) for key in operation.params))}"
    )
