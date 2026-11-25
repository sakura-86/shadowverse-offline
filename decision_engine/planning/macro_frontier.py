# -*- coding: utf-8 -*-
"""Lazy Macro frontier（P3：资源感知惰性宏观前沿）。

本模块把「已完成的 TurnLine」组织成一个**惰性展开的 Macro frontier**：

- 每条**完整** TurnLine 按其结束局面的
  `FutureResourceSignature` + `BoardFingerprint` 注册为一条 edge。
- **签名不是状态等价证明**：两条 edge 的 resource signature 或 board fingerprint 相等，
  只说明被签名的维度相同，绝不据此声称局面等价、也绝不据此删除任何 edge。
- 同一「资源桶」（FutureResourceSignature）内，初始最多**激活两个 board-distinct
  代表**（board fingerprint 不同的最高分线），其余 edge 一律标记 `pending`（保留、不删除）。
- 全局 `MacroCreditBudget` 在 lazy 模式下**真实 enforce**：
  `max_predictor_calls` / `max_micro_simulations` / `max_macro_nodes` /
  `max_turn_transitions`。预算耗尽时把待展开 edge / successor 标记 pending 并记录
  `pending_reason`，绝不静默删除，也绝不静默改走别的逻辑。
- 每一次 macro selection 只展开**一个**待展开 edge / successor，不重跑已展开的微观搜索。

设计边界（严格遵守 architecture-boundaries / prohibited-patterns）：
- 只使用本地可观察状态（TurnLine.end_snapshot + 已计算的 diff_score）。不读取协议
  Can*/Select*、after_context、runtime 路径或模型输出。
- 未完成 prefix（PENDING）绝不进入 frontier / candidate / backup —— 那是 P1
  MicroTurnCompletionSession 的职责边界，本模块只接受 strict 完整 TurnLine。
- 预算耗尽、无法展开时**显式记录结构化 pending reason**，不伪装成成功，也不静默 fallback。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from decision_engine.planning.board_fingerprint import BoardFingerprint
from decision_engine.planning.future_resource_signature import FutureResourceSignature

if TYPE_CHECKING:
    from decision_engine.planning.macro_turn_search import MacroCreditBudget, TurnLine


# ---------------------------------------------------------------------------
# Pending reason 常量（结构化，不静默）
# ---------------------------------------------------------------------------

# 同资源桶初始名额已满（>2 个 board-distinct 代表）：保留为 pending，不删除。
PENDING_BUCKET_QUOTA = "bucket_representative_quota"
# 同资源桶内 board fingerprint 与已激活代表重复：不占 board-distinct 名额，保留为 pending。
PENDING_DUPLICATE_BOARD = "duplicate_board_fingerprint_in_bucket"
# 全局预算耗尽：无法再展开，保留为 pending 并记录具体耗尽维度。
PENDING_BUDGET_PREDICTOR_CALLS = "budget_exhausted_predictor_calls"
PENDING_BUDGET_MICRO_SIMULATIONS = "budget_exhausted_micro_simulations"
PENDING_BUDGET_MACRO_NODES = "budget_exhausted_macro_nodes"
PENDING_BUDGET_TURN_TRANSITIONS = "budget_exhausted_turn_transitions"


# ---------------------------------------------------------------------------
# 预算 tracker：lazy 模式真实 enforce（对照 P0 observe-only）
# ---------------------------------------------------------------------------


@dataclass
class LazyBudgetTracker:
    """lazy 模式下对 MacroCreditBudget 的真实 enforcement。

    与 P0 `_MacroCreditLedger` 的 observe-only 不同：本 tracker 在**每次将要消耗**
    某类预算前显式检查上限；超限时返回 False 让调用方把候选标记 pending，绝不透支、
    也绝不静默删除。None 上限表示该维度不设限。

    account 的四个维度与 MacroCreditBudget 对齐：
    - predictor_calls  ↔ max_predictor_calls
    - micro_simulations ↔ max_micro_simulations
    - macro_nodes      ↔ max_macro_nodes（宏观 successor 节点展开数）
    - turn_transitions ↔ max_turn_transitions
    """

    budget: "MacroCreditBudget"
    predictor_calls: int = 0
    micro_simulations: int = 0
    macro_nodes: int = 0
    turn_transitions: int = 0
    # 结构化耗尽记录（维度 -> 命中次数），供诊断，不静默。
    exhausted_events: dict[str, int] = field(default_factory=dict)

    def _would_exceed(self, current: int, amount: int, limit: int | None) -> bool:
        if limit is None:
            return False
        return current + amount > limit

    def _record_exhausted(self, reason: str) -> None:
        self.exhausted_events[reason] = self.exhausted_events.get(reason, 0) + 1

    # -- 预检查（不消耗）：返回 (allowed, pending_reason_or_None) --

    def check_macro_node(self, amount: int = 1) -> tuple[bool, str | None]:
        if self._would_exceed(self.macro_nodes, amount, self.budget.max_macro_nodes):
            self._record_exhausted(PENDING_BUDGET_MACRO_NODES)
            return False, PENDING_BUDGET_MACRO_NODES
        return True, None

    def check_turn_transition(self, amount: int = 1) -> tuple[bool, str | None]:
        if self._would_exceed(
            self.turn_transitions, amount, self.budget.max_turn_transitions
        ):
            self._record_exhausted(PENDING_BUDGET_TURN_TRANSITIONS)
            return False, PENDING_BUDGET_TURN_TRANSITIONS
        return True, None

    def check_micro_simulations(self, amount: int) -> tuple[bool, str | None]:
        if self._would_exceed(
            self.micro_simulations, amount, self.budget.max_micro_simulations
        ):
            self._record_exhausted(PENDING_BUDGET_MICRO_SIMULATIONS)
            return False, PENDING_BUDGET_MICRO_SIMULATIONS
        return True, None

    def check_predictor_calls(self, amount: int) -> tuple[bool, str | None]:
        if self._would_exceed(
            self.predictor_calls, amount, self.budget.max_predictor_calls
        ):
            self._record_exhausted(PENDING_BUDGET_PREDICTOR_CALLS)
            return False, PENDING_BUDGET_PREDICTOR_CALLS
        return True, None

    # -- 实际消耗（预检查通过后调用） --

    def consume_macro_node(self, amount: int = 1) -> None:
        self.macro_nodes += amount

    def consume_turn_transition(self, amount: int = 1) -> None:
        self.turn_transitions += amount

    def consume_micro_simulations(self, amount: int) -> None:
        self.micro_simulations += max(0, amount)

    def consume_predictor_calls(self, amount: int) -> None:
        self.predictor_calls += max(0, amount)

    def to_diagnostics(self) -> dict[str, object]:
        values: dict[str, object] = {
            "credit_schema": "MacroLazyCredits/v1",
            "credit_mode": "enforced",
            "credit_budget_enforced": True,
            "lazy_predictor_calls": self.predictor_calls,
            "lazy_micro_simulations": self.micro_simulations,
            "lazy_macro_nodes": self.macro_nodes,
            "lazy_turn_transitions": self.turn_transitions,
            "lazy_budget_exhausted_events": dict(self.exhausted_events),
        }
        values.update(self.budget.to_diagnostics())
        return values


# ---------------------------------------------------------------------------
# Frontier edge
# ---------------------------------------------------------------------------

# edge 展开状态
EDGE_ACTIVE = "active"      # 已获激活名额，可被 selection 展开（尚未展开）。
EDGE_PENDING = "pending"    # 保留但未激活 / 未展开；带 pending_reason。绝不删除。
EDGE_EXPANDED = "expanded"  # 已被展开一次（微观 successor 展开完成），不重跑。


@dataclass
class FrontierEdge:
    """一条完整 TurnLine 在 Macro frontier 中的 edge。

    resource_signature / board_fingerprint 只作为**识别键**（注册与分桶用），
    **不是状态等价证明**：绝不据此把两条 edge 合并或删除。

    Attributes:
        line: 该 edge 对应的 strict 完整 TurnLine。
        resource_signature: 结束局面的 FutureResourceSignature（资源桶键）。
        board_fingerprint: 结束局面的 BoardFingerprint（board-distinct 键）。
        status: EDGE_ACTIVE / EDGE_PENDING / EDGE_EXPANDED。
        pending_reason: status==PENDING 时的结构化原因（配额 / 重复 board / 预算耗尽）。
        successor_snapshot: 该 edge 展开出的对手回合起始局面（展开后物化）。
        backed_up_value: 该 edge 展开后回传的 root-relative 价值（展开后物化）。
        insertion_index: 稳定输入序号（tie-break 用，保证确定性）。
    """

    line: "TurnLine"
    resource_signature: FutureResourceSignature
    board_fingerprint: BoardFingerprint
    status: str = EDGE_PENDING
    pending_reason: str | None = None
    successor_snapshot: object = None
    backed_up_value: float | None = None
    insertion_index: int = 0

    @property
    def diff_score(self) -> float:
        return float(self.line.diff_score)

    @property
    def is_expanded(self) -> bool:
        return self.status == EDGE_EXPANDED

    @property
    def is_active_unexpanded(self) -> bool:
        return self.status == EDGE_ACTIVE


# ---------------------------------------------------------------------------
# Macro frontier
# ---------------------------------------------------------------------------

# 同资源桶初始激活的 board-distinct 代表上限（合同要求：最多两个）。
DEFAULT_BUCKET_ACTIVE_QUOTA = 2


class MacroFrontier:
    """资源感知惰性宏观前沿。

    注册阶段：把所有 strict 完整 TurnLine 按 FutureResourceSignature 分桶；
    同桶内按 diff_score 降序，激活最多 `bucket_active_quota` 个 **board-distinct**
    （BoardFingerprint 不同）代表，其余标记 pending（配额已满 / board 重复）。

    展开阶段：`expand_one` 每次只激活/展开一条 edge —— 优先在预算内展开一个已激活但
    未展开的 edge；若已激活 edge 全部展开完，再从 pending 中「唤醒」一个（例如给同桶第
    三个 board-distinct 代表名额）。预算耗尽时把候选标记 pending 并记录 reason，绝不删除。

    **frontier 绝不删除任何 edge**：pending 是保留态，可被后续预算唤醒。
    """

    def __init__(
        self,
        budget_tracker: LazyBudgetTracker,
        *,
        perspective: str = "me",
        bucket_active_quota: int = DEFAULT_BUCKET_ACTIVE_QUOTA,
    ):
        self._tracker = budget_tracker
        self._perspective = perspective
        self._bucket_active_quota = max(0, int(bucket_active_quota))
        self._edges: list[FrontierEdge] = []
        # resource_signature -> 该桶已激活的 board fingerprint 集合（board-distinct 计数）
        self._active_boards_by_bucket: dict[FutureResourceSignature, set] = {}
        self._insertion_counter = 0

    # -- 注册 --

    def register_lines(self, lines) -> list[FrontierEdge]:
        """把一批 strict 完整 TurnLine 注册为 frontier edge。

        每条 line 独立成一条 edge（resource 或 board signature 相同**不**合并，
        绝不声称状态等价）。注册后立即按桶配额决定初始 active/pending。
        返回本次新增的 edge 列表（保序）。
        """
        new_edges: list[FrontierEdge] = []
        for line in lines:
            edge = self._build_edge(line)
            self._edges.append(edge)
            new_edges.append(edge)
        # 注册完成后统一按桶激活初始代表（保证同桶内按分数与 board-distinct 选取稳定）。
        self._assign_initial_activation()
        return new_edges

    def _build_edge(self, line) -> FrontierEdge:
        end = line.end_snapshot
        sig = FutureResourceSignature.from_snapshot(end, self._perspective)
        board = BoardFingerprint.from_snapshot(end, self._perspective)
        edge = FrontierEdge(
            line=line,
            resource_signature=sig,
            board_fingerprint=board,
            status=EDGE_PENDING,
            pending_reason=None,
            insertion_index=self._insertion_counter,
        )
        self._insertion_counter += 1
        return edge

    def _bucket_edges(self, sig: FutureResourceSignature) -> list[FrontierEdge]:
        return [e for e in self._edges if e.resource_signature == sig]

    def _assign_initial_activation(self) -> None:
        """按资源桶决定初始 active/pending（最多两个 board-distinct 代表）。

        只影响仍处于 PENDING 且未展开的 edge —— 已展开 / 已被唤醒激活的 edge 不回退。
        同桶内按 (diff_score desc, insertion_index asc) 稳定排序，依次尝试激活；
        board fingerprint 与本桶已激活代表重复的不占 board-distinct 名额（标 pending，
        原因 DUPLICATE_BOARD）；名额用满后剩余标 pending（原因 BUCKET_QUOTA）。
        """
        buckets: dict[FutureResourceSignature, list[FrontierEdge]] = {}
        for edge in self._edges:
            buckets.setdefault(edge.resource_signature, []).append(edge)

        for sig, edges in buckets.items():
            active_boards = self._active_boards_by_bucket.setdefault(sig, set())
            ordered = sorted(
                edges, key=lambda e: (-e.diff_score, e.insertion_index)
            )
            for edge in ordered:
                if edge.status in (EDGE_ACTIVE, EDGE_EXPANDED):
                    # 已激活或已展开的不动。
                    continue
                board = edge.board_fingerprint
                if board in active_boards:
                    # 该 board 已有激活代表：不占 board-distinct 名额，保留 pending。
                    edge.status = EDGE_PENDING
                    edge.pending_reason = PENDING_DUPLICATE_BOARD
                    continue
                if len(active_boards) >= self._bucket_active_quota:
                    # board-distinct 名额用满：保留 pending（配额）。
                    edge.status = EDGE_PENDING
                    edge.pending_reason = PENDING_BUCKET_QUOTA
                    continue
                edge.status = EDGE_ACTIVE
                edge.pending_reason = None
                active_boards.add(board)

    # -- 查询 --

    @property
    def edges(self) -> list[FrontierEdge]:
        return list(self._edges)

    def active_unexpanded_edges(self) -> list[FrontierEdge]:
        return [e for e in self._edges if e.status == EDGE_ACTIVE]

    def expanded_edges(self) -> list[FrontierEdge]:
        return [e for e in self._edges if e.status == EDGE_EXPANDED]

    def pending_edges(self) -> list[FrontierEdge]:
        return [e for e in self._edges if e.status == EDGE_PENDING]

    def _next_active_edge(self) -> FrontierEdge | None:
        """下一条待展开（已激活未展开）edge：按 (diff_score desc, insertion asc) 稳定选取。"""
        candidates = self.active_unexpanded_edges()
        if not candidates:
            return None
        candidates.sort(key=lambda e: (-e.diff_score, e.insertion_index))
        return candidates[0]

    def _wake_one_pending(self) -> FrontierEdge | None:
        """在已激活 edge 全部展开后，从 pending 中唤醒一条（progressive widening）。

        唤醒优先级：先按桶 diversity（board-distinct 名额未满且非重复 board 的桶优先），
        再按 (diff_score desc, insertion asc)。仅把 status 置回 ACTIVE 并占用 board 名额；
        不消耗任何预算（真正的预算在 expand_one 展开阶段检查）。唤醒不删除任何 edge。
        """
        # 优先唤醒能贡献新 board-distinct 代表的 edge（BUCKET_QUOTA 而非重复 board）。
        def priority(edge: FrontierEdge) -> tuple:
            duplicate = edge.pending_reason == PENDING_DUPLICATE_BOARD
            return (duplicate, -edge.diff_score, edge.insertion_index)

        pendings = [
            e for e in self._edges
            if e.status == EDGE_PENDING
            and e.pending_reason in (PENDING_BUCKET_QUOTA, PENDING_DUPLICATE_BOARD)
        ]
        if not pendings:
            return None
        pendings.sort(key=priority)
        edge = pendings[0]
        edge.status = EDGE_ACTIVE
        edge.pending_reason = None
        self._active_boards_by_bucket.setdefault(
            edge.resource_signature, set()
        ).add(edge.board_fingerprint)
        return edge

    def has_expandable(self) -> bool:
        """是否还有可展开的 edge（已激活未展开，或可唤醒的 pending）。"""
        if self.active_unexpanded_edges():
            return True
        return any(
            e.status == EDGE_PENDING
            and e.pending_reason in (PENDING_BUCKET_QUOTA, PENDING_DUPLICATE_BOARD)
            for e in self._edges
        )

    # -- 展开：每次只展开一个 --

    def expand_one(self, expander) -> "FrontierExpansion":
        """展开且仅展开一条 edge。

        expander(edge) -> ExpansionOutcome：由调用方（MacroTurnSearch lazy 路径）提供，
        负责实际执行 transition + 内层 minimax，并汇报真实成本（predictor calls /
        turn transitions / micro simulations）供预算 enforcement。expander 必须在实际
        消耗前不假设预算充足；本方法在调用 expander 前先做宏观节点预算预检查。

        返回 FrontierExpansion：
        - expanded_edge：本次成功展开的 edge（status→EXPANDED），或 None。
        - pending_reason：无法展开时的结构化原因（预算耗尽 / 无可展开 edge）。

        约束：
        - 每次调用最多展开一条 edge（不重跑已展开）。
        - 预算耗尽时把选中的 edge 标记 pending 并记录 reason，绝不删除。
        """
        edge = self._next_active_edge()
        if edge is None:
            edge = self._wake_one_pending()
        if edge is None:
            return FrontierExpansion(expanded_edge=None, pending_reason=None, exhausted=False)

        # 宏观 successor 节点预算预检查（展开一条 edge = 新增一个宏观 successor 节点）。
        allowed, reason = self._tracker.check_macro_node(1)
        if not allowed:
            edge.status = EDGE_PENDING
            edge.pending_reason = reason
            return FrontierExpansion(expanded_edge=None, pending_reason=reason, exhausted=True)

        outcome = expander(edge)
        if not outcome.expanded:
            # expander 内部（transition / 内层）预算耗尽：标记 pending，不删除。
            edge.status = EDGE_PENDING
            edge.pending_reason = outcome.pending_reason
            return FrontierExpansion(
                expanded_edge=None, pending_reason=outcome.pending_reason, exhausted=True
            )

        # 展开成功：消耗宏观节点预算并物化 edge 结果。
        self._tracker.consume_macro_node(1)
        edge.status = EDGE_EXPANDED
        edge.pending_reason = None
        edge.successor_snapshot = outcome.successor_snapshot
        edge.backed_up_value = outcome.backed_up_value
        return FrontierExpansion(expanded_edge=edge, pending_reason=None, exhausted=False)

    def to_diagnostics(self) -> dict[str, object]:
        pending_reason_counts: dict[str, int] = {}
        for e in self._edges:
            if e.status == EDGE_PENDING and e.pending_reason:
                pending_reason_counts[e.pending_reason] = (
                    pending_reason_counts.get(e.pending_reason, 0) + 1
                )
        return {
            "frontier_edge_count": len(self._edges),
            "frontier_active_unexpanded": len(self.active_unexpanded_edges()),
            "frontier_expanded": len(self.expanded_edges()),
            "frontier_pending": len(self.pending_edges()),
            "frontier_pending_reason_counts": pending_reason_counts,
            "frontier_bucket_count": len(self._active_boards_by_bucket),
        }


# ---------------------------------------------------------------------------
# 展开结果类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpansionOutcome:
    """expander 回报的一次 edge 展开结果。

    expanded=False 表示（内层/transition）预算耗尽或无法展开；必须带 pending_reason。
    """

    expanded: bool
    successor_snapshot: object = None
    backed_up_value: float | None = None
    pending_reason: str | None = None


@dataclass(frozen=True)
class FrontierExpansion:
    """MacroFrontier.expand_one 的一次调用结果。

    expanded_edge：本次成功展开的 edge（None 表示未展开）。
    pending_reason：未展开时的结构化原因（None 表示本轮根本没有可展开 edge，非错误）。
    exhausted：是否因预算耗尽而停止（用于上层决定是否终止 widening 循环）。
    """

    expanded_edge: FrontierEdge | None
    pending_reason: str | None
    exhausted: bool


__all__ = [
    "PENDING_BUCKET_QUOTA",
    "PENDING_DUPLICATE_BOARD",
    "PENDING_BUDGET_PREDICTOR_CALLS",
    "PENDING_BUDGET_MICRO_SIMULATIONS",
    "PENDING_BUDGET_MACRO_NODES",
    "PENDING_BUDGET_TURN_TRANSITIONS",
    "LazyBudgetTracker",
    "EDGE_ACTIVE",
    "EDGE_PENDING",
    "EDGE_EXPANDED",
    "FrontierEdge",
    "DEFAULT_BUCKET_ACTIVE_QUOTA",
    "MacroFrontier",
    "ExpansionOutcome",
    "FrontierExpansion",
]
