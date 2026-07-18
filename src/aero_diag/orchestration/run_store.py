"""RunStore — 执行运行的状态持久化与溯源。

审计修复 (AER-001): 替代原来"未实现"的占位符。
支持运行创建、节点状态追踪、checkpoint 和恢复。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class NodeRunStatus:
    """单个节点的执行状态。"""
    node_id: str
    node_name: str = ""
    asset_id: str = ""
    status: str = "pending"  # pending/running/completed/failed/skipped
    started_at: str = ""
    completed_at: str = ""
    elapsed_ms: int = 0
    output_artifact_ids: list[str] = field(default_factory=list)
    evidence_items: list[str] = field(default_factory=list)
    retry_count: int = 0
    error: str | None = None
    result_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    """一次完整的诊断运行。"""
    run_id: str
    task_id: str
    plan_digest: str = ""
    status: str = "created"  # created/pending/running/completed/failed/cancelled
    node_statuses: dict[str, NodeRunStatus] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""
    total_elapsed_ms: int = 0
    current_node_id: str = ""
    errors: list[str] = field(default_factory=list)
    # checkpoint
    completed_node_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RunStore:
    """运行状态持久化存储。

    当前为内存实现（原型），生产环境需替换为 PostgreSQL。
    """

    def __init__(self):
        self._runs: dict[str, RunRecord] = {}

    def create_run(self, run_id: str, task_id: str, plan_digest: str = "",
                   metadata: dict[str, Any] | None = None) -> RunRecord:
        """创建一个新的执行运行。"""
        record = RunRecord(
            run_id=run_id,
            task_id=task_id,
            plan_digest=plan_digest,
            status="created",
            started_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        self._runs[run_id] = record
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        """获取运行状态。"""
        return self._runs.get(run_id)

    def update_run_status(self, run_id: str, status: str) -> RunRecord | None:
        """更新运行整体状态。"""
        record = self._runs.get(run_id)
        if record:
            record.status = status
            if status in ("completed", "failed", "cancelled"):
                record.completed_at = datetime.now(timezone.utc).isoformat()
        return record

    def init_node(self, run_id: str, node_id: str, node_name: str,
                  asset_id: str) -> NodeRunStatus:
        """初始化节点执行状态。"""
        record = self._runs.get(run_id)
        if not record:
            raise KeyError(f"Run not found: {run_id}")
        ns = NodeRunStatus(
            node_id=node_id,
            node_name=node_name,
            asset_id=asset_id,
            status="pending",
        )
        record.node_statuses[node_id] = ns
        return ns

    def start_node(self, run_id: str, node_id: str) -> NodeRunStatus | None:
        """标记节点开始执行。"""
        record = self._runs.get(run_id)
        if not record:
            return None
        ns = record.node_statuses.get(node_id)
        if ns:
            ns.status = "running"
            ns.started_at = datetime.now(timezone.utc).isoformat()
            record.current_node_id = node_id
        return ns

    def complete_node(self, run_id: str, node_id: str,
                      output_artifact_ids: list[str] | None = None,
                      evidence_items: list[str] | None = None,
                      elapsed_ms: int = 0, result_summary: dict | None = None) -> NodeRunStatus | None:
        """标记节点执行完成。"""
        record = self._runs.get(run_id)
        if not record:
            return None
        ns = record.node_statuses.get(node_id)
        if ns:
            ns.status = "completed"
            ns.completed_at = datetime.now(timezone.utc).isoformat()
            ns.elapsed_ms = elapsed_ms
            ns.output_artifact_ids = output_artifact_ids or []
            ns.evidence_items = evidence_items or []
            ns.result_summary = result_summary or {}
            record.completed_node_ids.append(node_id)
        return ns

    def fail_node(self, run_id: str, node_id: str, error: str,
                  retry_count: int = 0) -> NodeRunStatus | None:
        """标记节点执行失败。"""
        record = self._runs.get(run_id)
        if not record:
            return None
        ns = record.node_statuses.get(node_id)
        if ns:
            ns.status = "failed"
            ns.error = error
            ns.retry_count = retry_count
            ns.completed_at = datetime.now(timezone.utc).isoformat()
        return ns

    def skip_node(self, run_id: str, node_id: str, reason: str) -> NodeRunStatus | None:
        """跳过节点（如上游失败或审批未通过）。"""
        record = self._runs.get(run_id)
        if not record:
            return None
        ns = record.node_statuses.get(node_id)
        if ns:
            ns.status = "skipped"
            ns.error = reason
        return ns

    def get_run_summary(self, run_id: str) -> dict[str, Any]:
        """获取运行摘要。"""
        record = self._runs.get(run_id)
        if not record:
            return {"run_id": run_id, "status": "not_found"}

        node_summaries = {}
        completed = 0; failed = 0; pending = 0; running = 0; skipped = 0
        for nid, ns in record.node_statuses.items():
            node_summaries[nid] = {
                "name": ns.node_name, "status": ns.status,
                "elapsed_ms": ns.elapsed_ms, "retries": ns.retry_count,
                "error": ns.error,
            }
            if ns.status == "completed": completed += 1
            elif ns.status == "failed": failed += 1
            elif ns.status == "pending": pending += 1
            elif ns.status == "running": running += 1
            elif ns.status == "skipped": skipped += 1

        return {
            "run_id": record.run_id,
            "task_id": record.task_id,
            "plan_digest": record.plan_digest,
            "status": record.status,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
            "total_elapsed_ms": record.total_elapsed_ms,
            "node_count": len(record.node_statuses),
            "nodes": node_summaries,
            "counts": {"completed": completed, "failed": failed,
                       "pending": pending, "running": running, "skipped": skipped},
            "current_node": record.current_node_id,
            "errors": record.errors,
        }

    def list_runs(self, task_id: str | None = None) -> list[dict]:
        """列出运行记录，可选按任务过滤。"""
        runs = []
        for rid, record in self._runs.items():
            if task_id and record.task_id != task_id:
                continue
            runs.append({
                "run_id": rid, "task_id": record.task_id,
                "status": record.status, "started_at": record.started_at,
                "plan_digest": record.plan_digest,
            })
        return runs

    def is_resumable(self, run_id: str) -> bool:
        """检查运行是否可从 checkpoint 恢复。"""
        record = self._runs.get(run_id)
        if not record:
            return False
        return record.status in ("created", "pending", "running")
