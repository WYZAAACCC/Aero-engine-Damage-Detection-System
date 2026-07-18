"""任务服务——诊断任务的创建、查询和状态管理。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from aero_diag.domain.task import InspectionTask, TaskState


class TaskService:
    """诊断任务的创建和管理服务。"""

    def __init__(self) -> None:
        self._tasks: dict[str, InspectionTask] = {}

    def create_task(
        self,
        *,
        title: str,
        description: str,
        objective: str,
        engine_type: str = "",
        engine_serial: str = "",
        component: str = "",
        component_location: str = "",
        operating_mode: str = "",
        constraints: dict[str, Any] | None = None,
        created_by: str = "",
    ) -> InspectionTask:
        """创建诊断任务。"""
        from aero_diag.domain.task import EngineInfo, OperatingCondition

        task = InspectionTask(
            task_id=uuid.uuid4().hex[:12],
            title=title,
            description=description,
            objective=objective,
            engine=EngineInfo(
                engine_type=engine_type,
                engine_serial=engine_serial,
                component=component,
                component_location=component_location,
            ),
            operating_conditions=OperatingCondition(mode=operating_mode),
            state=TaskState.RECEIVED,
            constraints=constraints or {},
            created_by=created_by,
        )
        self._tasks[task.task_id] = task
        return task

    def get_task(self, task_id: str) -> InspectionTask | None:
        """获取任务。"""
        return self._tasks.get(task_id)

    def update_constraints(
        self, task_id: str, constraints: dict[str, Any],
    ) -> InspectionTask | None:
        """更新任务约束。"""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.constraints.update(constraints)
        task.updated_at = datetime.now(timezone.utc)
        return task

    def add_input_artifact(
        self, task_id: str, artifact_ref: Any,  # ArtifactRef
    ) -> InspectionTask | None:
        """关联输入数据。"""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.input_artifacts.append(artifact_ref)
        task.updated_at = datetime.now(timezone.utc)
        return task

    def transition_state(
        self,
        task_id: str,
        to_state: TaskState,
        *,
        actor: str = "system",
        reason: str = "",
    ) -> InspectionTask:
        """执行任务状态迁移。"""
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")

        from aero_diag.orchestration.state_machine import TaskStateMachine

        tsm = TaskStateMachine()
        tsm._current_state = task.state
        tsm.transition(to_state, actor=actor, reason=reason)
        task.state = to_state
        task.updated_at = datetime.now(timezone.utc)
        return task

    def list_tasks(self, state: TaskState | None = None) -> list[InspectionTask]:
        """列出任务，可选按状态过滤。"""
        tasks = list(self._tasks.values())
        if state is not None:
            tasks = [t for t in tasks if t.state == state]
        return tasks
