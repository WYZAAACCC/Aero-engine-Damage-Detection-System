"""运行 API 路由——执行计划提交、状态查询与恢复。

审计修复 (AER-001): 不再返回 not_implemented 占位符。
接入 PlanExecutor + RunStore 提供真实的执行和状态追踪。
"""

from fastapi import APIRouter, Depends, HTTPException

from aero_diag.api.dependencies import get_asset_registry
from aero_diag.orchestration.run_store import RunStore
from aero_diag.registries.asset_registry import AssetRegistry

router = APIRouter(prefix="/runs", tags=["runs"])

# ── 全局 RunStore 单例（生产环境应注入）──
_run_store = RunStore()
_executor_cache: dict[str, object] = {}  # PlanExecutor lazy init


def _get_executor():
    """懒初始化 PlanExecutor——需要 AssetRegistry。"""
    key = "default"
    if key not in _executor_cache:
        from aero_diag.orchestration.executor import PlanExecutor
        from aero_diag.plugins.official.asset_runner import AssetRunner, _build_default_impl_registry
        reg = get_asset_registry()
        runner = AssetRunner(reg, _build_default_impl_registry())
        _executor_cache[key] = PlanExecutor(runner, _run_store)
    return _executor_cache[key]


@router.post("")
def submit_run(run_data: dict) -> dict:
    """提交执行计划运行。

    请求体:
        {"task_id": "...", "plan": {...ExecutionPlan...}}

    返回运行 ID 和初始状态。实际执行可同步或异步触发。
    """
    task_id = run_data.get("task_id", "")
    plan_data = run_data.get("plan")

    if not task_id:
        raise HTTPException(400, "task_id is required")
    if not plan_data:
        raise HTTPException(400, "plan is required (frozen ExecutionPlan)")

    try:
        from aero_diag.orchestration.plan import ExecutionPlan
        plan = ExecutionPlan.model_validate(plan_data)
    except Exception as e:
        raise HTTPException(400, f"Invalid plan: {e}")

    executor = _get_executor()
    record = executor.submit(plan, task_id=task_id)

    return {
        "run_id": record.run_id,
        "task_id": record.task_id,
        "status": record.status,
        "plan_digest": record.plan_digest[:16],
        "node_count": len(record.node_statuses),
        "started_at": record.started_at,
        "message": f"Run created with {len(record.node_statuses)} nodes. Call POST /runs/{record.run_id}/execute to start.",
    }


@router.post("/{run_id}/execute")
def execute_run(run_id: str, async_mode: bool = False) -> dict:
    """执行已提交的运行。

    GET /runs/{run_id}/execute?async=true — 异步执行（马上返回）
    GET /runs/{run_id}/execute — 同步执行（等待完成）
    """
    executor = _get_executor()

    if async_mode:
        import threading
        thread = threading.Thread(target=executor.execute, args=(run_id,), daemon=True)
        thread.start()
        return {
            "run_id": run_id,
            "status": "executing_async",
            "message": "Execution started in background. Poll GET /runs/{run_id} for status.",
        }

    record = executor.execute(run_id)
    if record is None:
        raise HTTPException(404, f"Run not found: {run_id}")

    summary = _run_store.get_run_summary(run_id)
    return {
        "run_id": run_id,
        "status": record.status,
        "summary": summary,
    }


@router.get("/{run_id}")
def get_run(run_id: str) -> dict:
    """查询运行状态和结果。"""
    record = _run_store.get_run(run_id)
    if record is None:
        raise HTTPException(404, f"Run not found: {run_id}")

    summary = _run_store.get_run_summary(run_id)
    return {
        "run_id": run_id,
        "task_id": record.task_id,
        "status": record.status,
        "plan_digest": record.plan_digest[:16] if record.plan_digest else "",
        "summary": summary,
        "errors": record.errors,
        "resumable": _run_store.is_resumable(run_id),
    }


@router.post("/{run_id}/resume")
def resume_run(run_id: str) -> dict:
    """从 checkpoint 恢复执行。"""
    executor = _get_executor()
    record = executor.resume(run_id)
    if record is None:
        raise HTTPException(404, f"Run not found: {run_id}")
    return {
        "run_id": run_id,
        "status": record.status,
        "message": "Resumed execution" if record.status == "completed" else f"Execution finished: {record.status}",
    }


@router.post("/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    """取消正在执行的运行。"""
    executor = _get_executor()
    record = executor.cancel(run_id)
    if record is None:
        raise HTTPException(404, f"Run not found: {run_id}")
    return {"run_id": run_id, "status": record.status, "message": "Run cancelled"}


@router.get("")
def list_runs(task_id: str | None = None) -> dict:
    """列出所有运行记录。"""
    runs = _run_store.list_runs(task_id=task_id)
    return {"runs": runs, "count": len(runs)}
