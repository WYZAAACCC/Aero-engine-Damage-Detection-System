"""运行 API 路由——执行与结果查询。"""

from fastapi import APIRouter, Depends, HTTPException

from aero_diag.api.dependencies import get_task_service
from aero_diag.services.task_service import TaskService

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("")
def submit_run(run_data: dict) -> dict:
    """提交执行计划运行。"""
    # P0 阶段：返回占位符，P1 接入完整执行引擎
    return {
        "run_id": "placeholder",
        "status": "not_implemented",
        "message": "Execution engine — pending P1 implementation",
    }


@router.get("/{run_id}")
def get_run(run_id: str) -> dict:
    """查询运行状态和结果。"""
    return {
        "run_id": run_id,
        "status": "not_implemented",
        "message": "Run query — pending P1 implementation",
    }
