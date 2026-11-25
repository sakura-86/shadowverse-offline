# -*- coding: utf-8 -*-
"""动作 pointer 表示：把 ``LegalAction`` 转成「动作头 + 源实体 + 目标槽」。

这是 pointer 动作表示的**接口层**，只做 ``LegalAction <-> PointerAction`` 的双向
转换与候选池辅助。不含网络结构——网络侧（entity transformer）消费 ``PointerAction``
的字段，而动作合法性仍完全由 ``LocalActionEnumerator`` 决定。

设计要点（见 ``docs/plan/ENTITY_TRANSFORMER_DESIGN.md`` §3）：

- **动作头**：``ActionKind`` 归并成少量 ``ActionHead``。``ATTACK_LEADER`` 与
  ``ATTACK_FOLLOWER`` 合并成 ``ATTACK``——主战者在目标池里用哨兵 ``uid=0`` 区分，
  否则网络要在两个独立 softmax 之间比较「打脸还是打随从」，结构上做不到。
- **目标槽**：一个动作可以有 0..n 次目标选择，每次由 ``LegalAction.target_groups``
  的一个 group 表示，语义由等长的 ``target_group_specs`` 标识。
  ``count=N`` 是**一个槽选 N 个实体**（arity=N），不是 N 个槽。
- **反查**：``resolve_pointer_action`` 把 pointer 解码结果映射回唯一
  ``LegalAction``；匹配失败显式抛错，不退回默认动作。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from decision_engine.actions.model import ActionKind, LegalAction, PlayKind


class ActionHead(Enum):
    """pointer 的动作头（比 ``ActionKind`` 粗，攻击合并成一个头）。"""

    PLAY = "play"
    ATTACK = "attack"
    EVOLVE = "evolve"
    SUPER_EVOLVE = "super_evolve"
    ACTIVATE = "activate"
    FUSION = "fusion"
    SPECIAL_ACTION = "special_action"
    EXTRA_PP = "extra_pp"
    TURN_END = "turn_end"
    TURN_START = "turn_start"
    MULLIGAN_CHANGE = "mulligan_change"
    MULLIGAN_END = "mulligan_end"


#: ``ActionKind`` → ``ActionHead``。``ATTACK_LEADER`` / ``ATTACK_FOLLOWER``
#: 归并到同一个 ``ATTACK``。
HEAD_BY_KIND: dict[ActionKind, ActionHead] = {
    ActionKind.PLAY: ActionHead.PLAY,
    ActionKind.ATTACK_LEADER: ActionHead.ATTACK,
    ActionKind.ATTACK_FOLLOWER: ActionHead.ATTACK,
    ActionKind.EVOLVE: ActionHead.EVOLVE,
    ActionKind.SUPER_EVOLVE: ActionHead.SUPER_EVOLVE,
    ActionKind.ACTIVATE: ActionHead.ACTIVATE,
    ActionKind.FUSION: ActionHead.FUSION,
    ActionKind.SPECIAL_ACTION: ActionHead.SPECIAL_ACTION,
    ActionKind.EXTRA_PP: ActionHead.EXTRA_PP,
    ActionKind.TURN_END: ActionHead.TURN_END,
    ActionKind.TURN_START: ActionHead.TURN_START,
    ActionKind.MULLIGAN_CHANGE: ActionHead.MULLIGAN_CHANGE,
    ActionKind.MULLIGAN_END: ActionHead.MULLIGAN_END,
}


class PointerActionError(ValueError):
    """pointer 动作无法表达或无法对应回合法动作集合。"""


@dataclass(frozen=True)
class ActionSlot:
    """一个目标槽：一次实体选择。

    ``spec_id`` 指明这次选择的语义（来自 ``LegalAction.target_group_specs``），
    ``arity`` 是它要选几个实体（= 对应 group 的长度）。
    """

    spec_id: str
    arity: int


@dataclass(frozen=True)
class PointerAction:
    """pointer 表示的动作：动作头 + 源实体 + 有序目标槽。

    ``targets[i]`` 是第 i 个槽选中的实体 uid；``slots[i].arity == len(targets[i])``。

    ``kind`` / ``cost`` / ``variant`` 保留原始 ``ActionKind`` 与派生量，使
    ``to_pointer_action`` 与 ``resolve_pointer_action`` 构成严格往返。
    """

    head: ActionHead
    kind: ActionKind
    source_uid: int
    play_kind: PlayKind | None
    targets: tuple[tuple[int, ...], ...]
    slots: tuple[ActionSlot, ...]
    selected_indexes: tuple[int, ...] = ()
    material_uids: tuple[int, ...] = ()
    cost: int | None = None
    variant: str | None = None

    def __post_init__(self) -> None:
        if len(self.targets) != len(self.slots):
            raise PointerActionError(
                f"targets({len(self.targets)}) 与 slots({len(self.slots)}) 数量不一致: "
                f"kind={self.kind} source_uid={self.source_uid}"
            )
        for index, (group, slot) in enumerate(zip(self.targets, self.slots)):
            if len(group) != slot.arity:
                raise PointerActionError(
                    f"slot[{index}] arity 不匹配: spec_id={slot.spec_id} "
                    f"arity={slot.arity} 实际={len(group)}"
                )

    @property
    def pointer_calls(self) -> int:
        """完成这个动作需要的 pointer 次数（各槽 arity 之和）。"""
        return sum(slot.arity for slot in self.slots)

    @property
    def target_specs(self) -> tuple[str, ...]:
        """各槽的 spec_id 序列。"""
        return tuple(slot.spec_id for slot in self.slots)


def to_pointer_action(action: LegalAction) -> PointerAction:
    """把 ``LegalAction`` 转成 ``PointerAction``。

    ``target_group_specs`` 必须与 ``target_groups`` 等长；不等长说明枚举器漏填，
    属于数据缺口，显式失败而不是按位置猜一个 spec。
    """
    head = HEAD_BY_KIND.get(action.kind)
    if head is None:
        raise PointerActionError(f"pointer 不支持的动作类型: {action.kind}")
    if len(action.target_group_specs) != len(action.target_groups):
        raise PointerActionError(
            f"target_group_specs({len(action.target_group_specs)}) 与 "
            f"target_groups({len(action.target_groups)}) 长度不一致: "
            f"kind={action.kind.value} source_uid={action.source_uid}"
        )
    slots = tuple(
        ActionSlot(spec_id=spec_id, arity=len(group))
        for spec_id, group in zip(action.target_group_specs, action.target_groups)
    )
    return PointerAction(
        head=head,
        kind=action.kind,
        source_uid=action.source_uid,
        play_kind=action.play_kind,
        targets=action.target_groups,
        slots=slots,
        selected_indexes=action.selected_indexes,
        material_uids=action.material_uids,
        cost=action.cost,
        variant=action.variant,
    )


def _action_identity(action: LegalAction) -> tuple[object, ...]:
    """动作身份：``description`` / ``legality_source`` / ``confidence`` 不参与。"""
    return (
        action.kind,
        action.source_uid,
        action.play_kind,
        action.cost,
        action.variant,
        action.target_groups,
        action.selected_indexes,
        action.material_uids,
        action.target_group_specs,
    )


def _pointer_identity(pointer: PointerAction) -> tuple[object, ...]:
    return (
        pointer.kind,
        pointer.source_uid,
        pointer.play_kind,
        pointer.cost,
        pointer.variant,
        pointer.targets,
        pointer.selected_indexes,
        pointer.material_uids,
        pointer.target_specs,
    )


def resolve_pointer_action(pointer: PointerAction, legal_actions: Sequence[LegalAction]) -> LegalAction:
    """把 pointer 解码结果反查回唯一的 ``LegalAction``。

    匹配键是动作身份（kind / source / play_kind / targets / selected_indexes /
    material_uids / specs），不含「打脸还是打随从」这类由 targets 表达的差别。

    匹配失败说明 pointer 指向了枚举器不认可的动作（mask 泄漏、解码越界、或
    pointer 与枚举器不同步），必须显式失败——不退回默认动作、不取第一个候选。

    若合法动作集合里存在身份完全相同的重复项（枚举器未去重的等价动作），返回
    第一个；它们在语义上等价。
    """
    identity = _pointer_identity(pointer)
    for action in legal_actions:
        if _action_identity(action) == identity:
            return action
    raise PointerActionError(
        "pointer 动作在合法动作集合中不存在: "
        f"head={pointer.head.value} kind={pointer.kind.value} "
        f"source_uid={pointer.source_uid} targets={pointer.targets} "
        f"specs={pointer.target_specs} selected_indexes={pointer.selected_indexes}"
    )


def group_by_head(actions: Sequence[LegalAction]) -> dict[ActionHead, tuple[LegalAction, ...]]:
    """按动作头分组，供 pointer 在 head 维度做 softmax。"""
    grouped: dict[ActionHead, list[LegalAction]] = {}
    for action in actions:
        head = HEAD_BY_KIND.get(action.kind)
        if head is None:
            raise PointerActionError(f"pointer 不支持的动作类型: {action.kind}")
        grouped.setdefault(head, []).append(action)
    return {head: tuple(items) for head, items in grouped.items()}


def source_candidates(actions: Sequence[LegalAction]) -> tuple[int, ...]:
    """所有候选动作的源实体，去重且保持首次出现顺序。"""
    return tuple(dict.fromkeys(action.source_uid for action in actions))


def slot_candidates(actions: Sequence[LegalAction], slot_index: int) -> tuple[int, ...]:
    """第 ``slot_index`` 个目标槽在所有候选动作里出现过的实体（pointer 的候选池）。

    槽数不足的动作被跳过；它们本来就没有这个槽。
    """
    seen: dict[int, None] = {}
    for action in actions:
        if slot_index >= len(action.target_groups):
            continue
        for uid in action.target_groups[slot_index]:
            seen.setdefault(uid, None)
    return tuple(seen)


def slot_specs(actions: Sequence[LegalAction], slot_index: int) -> tuple[str, ...]:
    """第 ``slot_index`` 个槽可能对应的 spec_id 集合（用于 slot token 的语义编码）。"""
    seen: dict[str, None] = {}
    for action in actions:
        if slot_index >= len(action.target_group_specs):
            continue
        seen.setdefault(action.target_group_specs[slot_index], None)
    return tuple(seen)


def max_slot_count(actions: Sequence[LegalAction]) -> int:
    """候选动作集合里最大的槽数（变长 slot 序列的长度上界）。"""
    return max((len(action.target_groups) for action in actions), default=0)


def policy_logits_from_pointer(
    *,
    head_logits: object,
    source_logits: object,
    slot_logits: object,
    action_head_indices: object,
    action_source_positions: object,
    action_target_slot_index: object,
    action_target_positions: object,
    action_target_mask: object,
    action_mask: object,
) -> object:
    """把生成式 pointer 的三个分布折算成每个候选动作的 logit。

    因子化：``logit(a) = head[h(a)] + source[s(a)] + Σ_t slot[k(a,t), p(a,t)]``。
    三项都在 log 空间相加，等价于概率相乘（head 决定动作类型、source 决定打哪张、
    各槽决定每次选择指向谁）。

    这是**兼容层**：pointer 模式本身不需要候选动作，但旧训练管线（PPO/BC）消费
    ``policy_logits [B,A]``，用它把 pointer 输出接回原管线。

    输入张量约定（torch）：

    - ``head_logits [B,H]``、``source_logits [B,P]`` 或 ``[B,H,P]``（按动作头条件化）
      、``slot_logits [B,K,P]``
    - ``action_head_indices [B,A]``（-1 = 无法映射）
    - ``action_source_positions [B,A]``（-1 = 无坐标）
    - ``action_target_slot_index [B,A,T]``（-1 = pad）
    - ``action_target_positions [B,A,T]``（-1 = pad）
    - ``action_target_mask [B,A,T]``、``action_mask [B,A]``

    无坐标或无法映射的动作被置为 ``-inf``——显式失败，而不是选到位置 0。
    """
    import torch

    B, A = action_source_positions.shape
    T = action_target_positions.shape[-1]
    P = source_logits.shape[-1]
    neg_inf = torch.finfo(head_logits.dtype).min

    head_invalid = action_head_indices < 0
    source_invalid = action_source_positions < 0

    head_index = action_head_indices.clamp(min=0).long()                             # [B,A]
    source_index = action_source_positions.clamp(min=0).long()                       # [B,A]

    head_term = head_logits.gather(1, head_index)                                    # [B,A]
    if source_logits.dim() == 3:
        # 条件化：每个候选动作取自己 head 的那一片 source 分布
        batch_index = torch.arange(B, device=head_index.device).unsqueeze(-1)        # [B,1]
        source_term = source_logits[batch_index, head_index, source_index]           # [B,A]
    else:
        source_term = source_logits.gather(1, source_index)                          # [B,A]

    slot_index = action_target_slot_index.clamp(min=0).long()                        # [B,A,T]
    target_pos = action_target_positions.clamp(min=0).long()                         # [B,A,T]
    K = slot_logits.shape[-2]
    if slot_logits.dim() == 4:
        # slot 也按 head 条件化：``[B,H,K,P]``，索引 ``(h*K + k)*P + p``
        flat_slots = slot_logits.reshape(B, -1)
        combined = head_index.unsqueeze(-1) * (K * P) + slot_index * P + target_pos
        flat_index = combined.reshape(B, -1)                                         # [B,A*T]
    else:
        flat_slots = slot_logits.reshape(B, -1)                                      # [B,K*P]
        flat_index = (slot_index * P + target_pos).reshape(B, -1)                    # [B,A*T]
    slot_term = flat_slots.gather(1, flat_index).reshape(B, A, T)                    # [B,A,T]
    slot_term = torch.where(action_target_mask, slot_term, torch.zeros_like(slot_term)).sum(-1)

    logits = head_term + source_term + slot_term
    logits = logits.masked_fill(head_invalid | source_invalid, neg_inf)
    return logits.masked_fill(~action_mask, neg_inf)
