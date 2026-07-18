"""任务 API 路由——创建、查询、状态迁移。"""

from fastapi import APIRouter, Depends, HTTPException

from aero_diag.api.dependencies import get_task_service
from aero_diag.domain.task import InspectionTask, TaskState
from aero_diag.services.task_service import TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=InspectionTask, status_code=201)
def create_task(
    task_data: dict,
    service: TaskService = Depends(get_task_service),
) -> InspectionTask:
    """创建诊断任务。"""
    try:
        return service.create_task(
            title=task_data.get("title", ""),
            description=task_data.get("description", ""),
            objective=task_data.get("objective", ""),
            engine_type=task_data.get("engine_type", ""),
            engine_serial=task_data.get("engine_serial", ""),
            component=task_data.get("component", ""),
            component_location=task_data.get("component_location", ""),
            operating_mode=task_data.get("operating_mode", ""),
            constraints=task_data.get("constraints"),
            created_by=task_data.get("created_by", "api"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{task_id}", response_model=InspectionTask)
def get_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
) -> InspectionTask:
    """获取诊断任务详情。"""
    task = service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return task


@router.get("", response_model=list[InspectionTask])
def list_tasks(
    state: str = "",
    service: TaskService = Depends(get_task_service),
) -> list[InspectionTask]:
    """列出任务，可选按状态过滤。"""
    task_state = TaskState(state) if state else None
    return service.list_tasks(state=task_state)


@router.post("/{task_id}/transition")
def transition_task(
    task_id: str,
    transition_data: dict,
    service: TaskService = Depends(get_task_service),
) -> InspectionTask:
    """执行任务状态迁移。"""
    try:
        to_state = TaskState(transition_data["to_state"])
        return service.transition_state(
            task_id=task_id,
            to_state=to_state,
            actor=transition_data.get("actor", "api"),
            reason=transition_data.get("reason", ""),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{task_id}/artifacts")
def add_artifact(
    task_id: str,
    artifact_data: dict,
    service: TaskService = Depends(get_task_service),
) -> InspectionTask:
    """关联输入数据到任务。"""
    task = service.add_input_artifact(task_id, artifact_data)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return task
