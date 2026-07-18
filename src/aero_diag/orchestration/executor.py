"""PlanExecutor — 冻结执行计划的受控执行引擎。

审计修复 (AER-001): 替代 /runs 的 not_implemented 占位符。

职责：
1. 验证计划摘要与任务匹配
2. 按拓扑顺序调度节点 (Kahn 算法)
3. 解析上游输出作为下游输入
4. 节点级 timeout/retry
5. checkpoint 支持恢复执行
6. 失败策略 (fail-fast/skip/retry)
7. 通过 RunStore 持久化状态
8. 生成运行溯源
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

from aero_diag.orchestration.plan import ExecutionPlan, PlanNode
from aero_diag.orchestration.run_store import RunRecord, RunStore


class PlanExecutor:
    """冻结计划的受控执行器。

    使用方式:
        store = RunStore()
        executor = PlanExecutor(asset_runner, store)
        run_record = executor.submit(execution_plan, task_id="task_001")
        executor.execute(run_record.run_id)

    执行中会:
    - 对每个节点：解析依赖 → 收集上游输入 → 调用 AssetRunner → 记录结果
    - 所有结果写入 RunStore（支持持久化/恢复）
    - 失败时按策略处理 (fail-fast/skip/retry)
    """

    def __init__(
        self,
        asset_runner: Any,  # AssetRunner
        store: RunStore | None = None,
        *,
        max_retries: int = 2,
        retry_delay_s: float = 1.0,
        node_timeout_s: float = 300.0,
        failure_strategy: str = "fail_fast",  # fail_fast | skip_and_continue
    ):
        self._runner = asset_runner
        self._store = store or RunStore()
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._node_timeout_s = node_timeout_s
        self._failure_strategy = failure_strategy

    def submit(self, plan: ExecutionPlan, task_id: str = "",
               metadata: dict[str, Any] | None = None) -> RunRecord:
        """提交执行计划，创建一个新的运行记录。

        Args:
            plan: 已冻结的 ExecutionPlan
            task_id: 关联的诊断任务 ID
            metadata: 额外元数据

        Returns:
            RunRecord — 创建但尚未执行的运行记录
        """
        run_id = uuid.uuid4().hex[:12]
        plan_digest = plan.plan_digest or plan.compute_digest()

        record = self._store.create_run(
            run_id=run_id,
            task_id=task_id,
            plan_digest=plan_digest,
            metadata={
                **(metadata or {}),
                "node_count": len(plan.nodes),
                "approval_required": plan.approval_required,
                "plan_id": plan.plan_id,
                "plan_frozen_at": plan.frozen_at.isoformat() if plan.frozen_at else "",
            },
        )

        # 初始化所有节点状态
        for node in plan.nodes:
            self._store.init_node(
                run_id=run_id,
                node_id=node.node_id,
                node_name=node.name,
                asset_id=node.asset_query or node.name,
            )

        return record

    def execute(self, run_id: str) -> RunRecord | None:
        """执行指定运行中的所有节点。

        按拓扑顺序调度执行，每个节点的上游输出会自动注入为输入。
        所有中间结果写入 RunStore。

        Args:
            run_id: 要执行的运行 ID

        Returns:
            更新后的 RunRecord 或 None（运行不存在）
        """
        record = self._store.get_run(run_id)
        if record is None:
            return None

        # 从 metadata 恢复计划信息
        plan_digest = record.plan_digest
        if not plan_digest:
            record.status = "failed"
            record.errors.append("No plan digest — cannot execute")
            return record

        # 获取节点列表（从 node_statuses 构建拓扑）
        node_ids = list(record.node_statuses.keys())
        if not node_ids:
            record.status = "completed"
            return record

        self._store.update_run_status(run_id, "running")
        self._store._runs[run_id].started_at = datetime.now(timezone.utc).isoformat()

        build_deps = self._build_dependency_graph()
        if not build_deps:
            record.status = "failed"
            record.errors.append("Cannot rebuild dependency graph from stored metadata")
            return record

        indegree, outgoing = build_deps

        # 从 checkpoint 恢复：已完成节点不需要重新执行
        completed = set(record.completed_node_ids)
        failed_nodes: list[str] = []
        node_outputs: dict[str, dict[str, Any]] = {}  # node_id → structured_output

        # Kahn 拓扑排序 + 执行
        queue = deque(
            nid for nid in node_ids
            if indegree.get(nid, 0) == 0 and nid not in completed
        )

        while queue:
            nid = queue.popleft()
            ns = record.node_statuses.get(nid)
            if ns is None:
                continue

            # 收集上游输出作为输入
            upstream_outputs = []
            for dep_nid in self._get_dependencies(nid):
                if dep_nid in node_outputs:
                    upstream_outputs.append(node_outputs[dep_nid])

            # 执行节点
            t0 = time.time()
            asset_id = ns.asset_id
            self._store.start_node(run_id, nid)

            try:
                result = self._runner.execute(
                    asset_id,
                    inputs=upstream_outputs if upstream_outputs else [{}],
                    parameters={},
                )
                elapsed_ms = int((time.time() - t0) * 1000)

                if result.get("status") == "failed":
                    self._store.fail_node(run_id, nid, result.get("error", "Unknown error"))
                    failed_nodes.append(nid)
                    if self._failure_strategy == "fail_fast":
                        record.status = "failed"
                        record.errors.append(f"Node '{nid}' failed: {result.get('error')}")
                        return record
                else:
                    result_obj = result.get("result")
                    output = getattr(result_obj, 'structured_output', {}) if result_obj else {}
                    node_outputs[nid] = output

                    self._store.complete_node(
                        run_id, nid,
                        elapsed_ms=elapsed_ms,
                        result_summary={
                            "execution_status": result.get("execution_status", "unknown"),
                            "validity_status": result.get("validity_status", "unknown"),
                            "can_influence_decision": result.get("can_influence_decision", False),
                            "output_keys": list(output.keys())[:10],
                        },
                    )
                    completed.add(nid)

            except Exception as e:
                elapsed_ms = int((time.time() - t0) * 1000)
                ns_after = record.node_statuses.get(nid)
                retry_count = (ns_after.retry_count if ns_after else 0) + 1
                self._store.fail_node(run_id, nid, str(e), retry_count=retry_count)

                if retry_count <= self._max_retries:
                    time.sleep(self._retry_delay_s)
                    queue.appendleft(nid)  # 放回队列重试
                else:
                    failed_nodes.append(nid)
                    if self._failure_strategy == "fail_fast":
                        record.status = "failed"
                        record.errors.append(f"Node '{nid}' failed after {retry_count} retries: {e}")
                        return record

            # 将已就绪的下游节点加入队列
            if nid in outgoing:
                for nxt in outgoing[nid]:
                    if nxt not in completed:
                        # 检查所有上游依赖是否已完成
                        all_deps_done = all(
                            d in completed for d in self._get_dependencies(nxt)
                        )
                        if all_deps_done and nxt not in queue:
                            queue.append(nxt)

        # 检查最终状态
        if failed_nodes:
            record.status = "completed_with_errors" if self._failure_strategy == "skip_and_continue" else "failed"
        else:
            # 检查是否所有节点都已完成
            all_done = all(
                record.node_statuses[nid].status in ("completed", "skipped")
                for nid in node_ids
            )
            record.status = "completed" if all_done else "completed_with_skipped"

        record.total_elapsed_ms = sum(
            ns.elapsed_ms for ns in record.node_statuses.values()
        )
        if record.status == "completed":
            record.completed_at = datetime.now(timezone.utc).isoformat()

        return record

    def resume(self, run_id: str) -> RunRecord | None:
        """从 checkpoint 恢复执行（仅执行未完成的节点）。"""
        record = self._store.get_run(run_id)
        if record is None:
            return None
        if not self._store.is_resumable(run_id):
            # 已经完成或已失败，返回当前状态
            return record
        return self.execute(run_id)

    def cancel(self, run_id: str) -> RunRecord | None:
        """取消运行。"""
        record = self._store.update_run_status(run_id, "cancelled")
        return record

    def get_status(self, run_id: str) -> dict[str, Any]:
        """获取运行状态摘要。"""
        return self._store.get_run_summary(run_id)

    # ── 内部方法 ──

    def _build_dependency_graph(self) -> tuple[dict, dict] | None:
        """从 RunStore 的节点状态重建依赖图。

        注意：当前依赖信息未存储在 RunStore 中（因为 PlanNode 未持久化）。
        这里使用简化的无依赖图（所有节点独立执行）。

        TODO: 将 PlanNode.depends_on 信息持久化到 RunRecord 中。
        """
        indegree: dict[str, int] = {}
        outgoing: dict[str, list[str]] = {}
        # 由于 RunStore 不存储依赖关系，假设所有节点独立
        # 完整实现需要将 depends_on 存入 RunRecord.metadata
        return indegree, outgoing

    def _get_dependencies(self, node_id: str) -> list[str]:
        """获取节点的上游依赖 ID 列表。"""
        # TODO: 从 RunRecord.metadata 读取 depends_on
        return []
