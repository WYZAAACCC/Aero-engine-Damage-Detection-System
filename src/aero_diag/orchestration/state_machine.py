"""任务状态机——强制的工作流阶段迁移与前置条件校验。

遵循文档第 5.2 节设计：13 个任务状态，
每次状态迁移必须满足前置条件并写入审计事件。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from aero_diag.domain.task import TaskState


# 状态迁移表——定义每个状态的合法下一状态
_ALLOWED_TRANSITIONS: dict[TaskState, list[TaskState]] = {
    TaskState.RECEIVED: [
        TaskState.DATA_VALIDATION,
    ],
    TaskState.DATA_VALIDATION: [
        TaskState.NEED_MORE_DATA,
        TaskState.PLAN_PROPOSAL,
    ],
    TaskState.NEED_MORE_DATA: [
        TaskState.DATA_VALIDATION,
    ],
    TaskState.PLAN_PROPOSAL: [
        TaskState.PLAN_COMPILE,
    ],
    TaskState.PLAN_COMPILE: [
        TaskState.PLAN_REVIEW_REQUIRED,
        TaskState.DETECTION_EXECUTION,
    ],
    TaskState.PLAN_REVIEW_REQUIRED: [
        TaskState.DETECTION_EXECUTION,
        TaskState.NEED_MORE_DATA,
    ],
    TaskState.DETECTION_EXECUTION: [
        TaskState.CHARACTERIZATION,
        TaskState.NEED_MORE_DATA,
    ],
    TaskState.CHARACTERIZATION: [
        TaskState.RELIABILITY_ASSESSMENT,
        TaskState.EVIDENCE_FUSION,
        TaskState.NEED_MORE_DATA,
    ],
    TaskState.RELIABILITY_ASSESSMENT: [
        TaskState.EVIDENCE_FUSION,
    ],
    TaskState.EVIDENCE_FUSION: [
        TaskState.DECISION_DRAFT,
        TaskState.NEED_MORE_DATA,
    ],
    TaskState.DECISION_DRAFT: [
        TaskState.EXPERT_REVIEW,
    ],
    TaskState.EXPERT_REVIEW: [
        TaskState.APPROVED,
        TaskState.REJECTED,
        TaskState.REWORK,
    ],
    TaskState.APPROVED: [
        TaskState.ARCHIVED,
    ],
    TaskState.REJECTED: [
        TaskState.ARCHIVED,
    ],
    TaskState.REWORK: [
        TaskState.PLAN_PROPOSAL,
        TaskState.DETECTION_EXECUTION,
    ],
    TaskState.ARCHIVED: [],
}


# 状态迁移前置条件
_TRANSITION_GATES: dict[tuple[TaskState, TaskState], str] = {
    (TaskState.RECEIVED, TaskState.DATA_VALIDATION):
        "任务必须包含至少一个 input_artifact 引用",

    (TaskState.DATA_VALIDATION, TaskState.PLAN_PROPOSAL):
        "数据质量门必须通过（DataQualityReport.overall_status != FAIL）",

    (TaskState.PLAN_COMPILE, TaskState.DETECTION_EXECUTION):
        "ExecutionPlan 必须已冻结（plan_digest 不为空）",

    (TaskState.DETECTION_EXECUTION, TaskState.CHARACTERIZATION):
        "必须至少存在一个 DetectionFinding",

    (TaskState.CHARACTERIZATION, TaskState.RELIABILITY_ASSESSMENT):
        "必须至少存在一个 DamageCharacterization；需要载荷/材料数据或明确的缺失声明",

    (TaskState.EVIDENCE_FUSION, TaskState.DECISION_DRAFT):
        "证据融合必须已执行（EvidenceGraph 不为空）",

    (TaskState.DECISION_DRAFT, TaskState.EXPERT_REVIEW):
        "决策草案必须标记 requires_review=True",

    (TaskState.EXPERT_REVIEW, TaskState.APPROVED):
        "必须存在专家签名和最终证据包",

    (TaskState.APPROVED, TaskState.ARCHIVED):
        "所有证据必须已打包；审核签名已确认",
}


@dataclass
class StateTransition:
    """状态迁移事件记录。"""
    from_state: TaskState
    to_state: TaskState
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    actor: str = ""  # "system" / agent_role / human_reviewer
    preconditions_met: list[str] = field(default_factory=list)
    preconditions_failed: list[str] = field(default_factory=list)


class InvalidTransitionError(ValueError):
    """非法的状态迁移。"""
    pass


class TaskStateMachine:
    """任务状态机——管控诊断任务的完整生命周期。

    每次迁移：
    1. 检查是否为合法迁移
    2. 检查前置条件
    3. 记录迁移事件
    """

    def __init__(self):
        self._current_state: TaskState = TaskState.RECEIVED
        self._history: list[StateTransition] = []
        # 外部注入的前置条件检查器
        self._gate_checkers: dict[tuple[TaskState, TaskState], Callable[[Any], bool]] = {}

    @property
    def current_state(self) -> TaskState:
        return self._current_state

    @property
    def is_terminal(self) -> bool:
        return self._current_state == TaskState.ARCHIVED

    @property
    def history(self) -> list[StateTransition]:
        return list(self._history)

    def allowed_transitions(self, from_state: TaskState | None = None) -> list[TaskState]:
        """返回当前状态的合法下一状态列表。"""
        source = from_state or self._current_state
        return _ALLOWED_TRANSITIONS.get(source, [])

    def transition_gate(self, from_state: TaskState, to_state: TaskState) -> str:
        """返回状态迁移的前置条件描述。"""
        return _TRANSITION_GATES.get((from_state, to_state), "")

    def transition(
        self,
        to_state: TaskState,
        *,
        reason: str = "",
        actor: str = "system",
        context: Any = None,
    ) -> StateTransition:
        """执行状态迁移。

        Args:
            to_state: 目标状态
            reason: 迁移原因
            actor: 迁移发起者
            context: 领域上下文（用于前置条件检查）

        Returns:
            StateTransition 记录

        Raises:
            InvalidTransitionError: 迁移被拒绝
        """
        from_state = self._current_state

        # 1. 合法性检查
        allowed = self.allowed_transitions(from_state)
        if to_state not in allowed:
            raise InvalidTransitionError(
                f"Transition from {from_state.value} to {to_state.value} is not allowed. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        # 2. 前置条件检查
        gate_description = self.transition_gate(from_state, to_state)
        preconditions_met: list[str] = [gate_description] if gate_description else []
        preconditions_failed: list[str] = []

        # 执行自定义检查器
        custom_checker = self._gate_checkers.get((from_state, to_state))
        if custom_checker is not None:
            try:
                ok = custom_checker(context)
                if ok:
                    preconditions_met.append("custom_check_passed")
                else:
                    preconditions_failed.append("custom_check_failed")
            except Exception as e:
                preconditions_failed.append(f"custom_check_error: {e}")

        if preconditions_failed:
            raise InvalidTransitionError(
                f"Precondition check failed for {from_state.value} -> {to_state.value}: "
                f"{preconditions_failed}"
            )

        # 3. 执行迁移
        self._current_state = to_state
        record = StateTransition(
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            actor=actor,
            preconditions_met=preconditions_met,
            preconditions_failed=preconditions_failed,
        )
        self._history.append(record)
        return record

    def register_gate_checker(
        self,
        from_state: TaskState,
        to_state: TaskState,
        checker: Callable[[Any], bool],
    ) -> None:
        """注册自定义前置条件检查器。"""
        self._gate_checkers[(from_state, to_state)] = checker

    def reset(self) -> None:
        """重置状态机到初始状态。"""
        self._current_state = TaskState.RECEIVED
        self._history.clear()

    def is_at_or_after(self, state: TaskState) -> bool:
        """检查当前状态是否已经达到或超过指定状态。"""
        order = list(_ALLOWED_TRANSITIONS.keys())
        try:
            return order.index(self._current_state) >= order.index(state)
        except ValueError:
            return False
