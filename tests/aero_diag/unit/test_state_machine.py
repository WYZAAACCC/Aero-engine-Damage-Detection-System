"""状态机单元测试 — 门检查、非法迁移、禁止自动审批"""
import pytest
import sys
sys.path.insert(0, 'src')

from aero_diag.domain.task import TaskState
from aero_diag.orchestration.state_machine import (
    TaskStateMachine, InvalidTransitionError,
)


class TestTaskStateMachine:
    """TaskStateMachine 门检查与安全测试"""

    def test_required_gate_blocked_without_checker(self):
        """REQUIRED_GATES 迁移无 checker 时应被阻断 (AER-003)"""
        tsm = TaskStateMachine()
        tsm._current_state = TaskState.RECEIVED
        with pytest.raises(InvalidTransitionError, match="GATE_BLOCKED"):
            tsm.transition(TaskState.DATA_VALIDATION, actor='system')

    def test_required_gate_passes_with_checker(self):
        """注册 checker 后 REQUIRED_GATES 迁移应通过"""
        tsm = TaskStateMachine()
        tsm.register_gate_checker(
            TaskState.RECEIVED, TaskState.DATA_VALIDATION,
            lambda ctx: True,
        )
        record = tsm.transition(TaskState.DATA_VALIDATION, actor='system')
        assert record.to_state == TaskState.DATA_VALIDATION
        assert tsm.current_state == TaskState.DATA_VALIDATION

    def test_required_gate_fails_with_false_checker(self):
        """checker 返回 False 时迁移应被阻断"""
        tsm = TaskStateMachine()
        tsm.register_gate_checker(
            TaskState.RECEIVED, TaskState.DATA_VALIDATION,
            lambda ctx: False,
        )
        with pytest.raises(InvalidTransitionError, match="custom_check_failed"):
            tsm.transition(TaskState.DATA_VALIDATION, actor='system')

    def test_auto_approve_blocked(self):
        """系统不能自动 APPROVE (P0-5, FORBIDDEN_AUTO_TRANSITIONS)"""
        tsm = TaskStateMachine()
        tsm._current_state = TaskState.EXPERT_REVIEW
        with pytest.raises(InvalidTransitionError, match="REQUIRES authenticated human reviewer"):
            tsm.transition(TaskState.APPROVED, actor='system')

    def test_illegal_transition_rejected(self):
        """不允许的状态迁移应被拒绝"""
        tsm = TaskStateMachine()
        tsm._current_state = TaskState.RECEIVED
        with pytest.raises(InvalidTransitionError, match="not allowed"):
            tsm.transition(TaskState.ARCHIVED, actor='system')

    def test_transition_history_recorded(self):
        """状态迁移历史应被记录"""
        tsm = TaskStateMachine()
        tsm.register_gate_checker(
            TaskState.RECEIVED, TaskState.DATA_VALIDATION,
            lambda ctx: True,
        )
        tsm.transition(TaskState.DATA_VALIDATION, actor='system')
        assert len(tsm.history) == 1
        assert tsm.history[0].from_state == TaskState.RECEIVED
        assert tsm.history[0].to_state == TaskState.DATA_VALIDATION

    def test_full_path_to_expert_review(self):
        """完整路径可走通到 EXPERT_REVIEW（不含自动审批）"""
        tsm = TaskStateMachine()
        _pass = lambda ctx: True
        for src, dst in [
            (TaskState.RECEIVED, TaskState.DATA_VALIDATION),
            (TaskState.DATA_VALIDATION, TaskState.PLAN_PROPOSAL),
            (TaskState.PLAN_COMPILE, TaskState.DETECTION_EXECUTION),
            (TaskState.DETECTION_EXECUTION, TaskState.CHARACTERIZATION),
            (TaskState.EVIDENCE_FUSION, TaskState.DECISION_DRAFT),
            (TaskState.DECISION_DRAFT, TaskState.EXPERT_REVIEW),
        ]:
            tsm.register_gate_checker(src, dst, _pass)

        tsm.transition(TaskState.DATA_VALIDATION, actor='system')
        tsm.transition(TaskState.PLAN_PROPOSAL, actor='system')
        tsm.transition(TaskState.PLAN_COMPILE, actor='system')
        tsm.transition(TaskState.DETECTION_EXECUTION, actor='system')
        tsm.transition(TaskState.CHARACTERIZATION, actor='system')
        tsm._current_state = TaskState.EVIDENCE_FUSION
        tsm.transition(TaskState.DECISION_DRAFT, actor='system')
        tsm.transition(TaskState.EXPERT_REVIEW, actor='system')

        assert tsm.current_state == TaskState.EXPERT_REVIEW
        # 不能再自动到 APPROVED
        with pytest.raises(InvalidTransitionError):
            tsm.transition(TaskState.APPROVED, actor='system')
