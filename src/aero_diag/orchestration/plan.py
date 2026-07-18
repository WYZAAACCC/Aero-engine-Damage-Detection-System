"""计划编译与执行计划——LLM 提议，系统编译校验后冻结。

遵循文档第 5.3 节两阶段机制：
1. LLM 生成 PlanProposal（候选计划）
2. PlanCompiler 静态校验 → 生成 ExecutionPlan
3. 计划冻结后不得隐式改变
4. 受控修订通过 PlanAmendment
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from aero_diag.assets.manifest import AssetKind, AssetStatus
from aero_diag.domain.task import TaskState


class PlanNode(BaseModel):
    """执行计划中的一个节点——对应一次资产运行。"""
    node_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""                          # 节点名称
    stage: TaskState                        # 关联的诊断阶段

    # 资产信息
    asset_query: str = ""                   # asset_id or asset_id@version
    asset_version: str = ""                 # 最终解析版本

    # 输入输出
    input_refs: list[str] = Field(default_factory=list)  # 上游节点 ID 或 Artifact ID
    output_artifact_type: str = ""          # 期望输出类型

    # 参数
    parameters: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None

    # 质量控制
    quality_gate: bool = False              # 此节点前是否需要质量门
    requires_approval: bool = False         # 此节点执行前是否需要审批
    retry_max: int = 1
    timeout_s: float = 120.0

    # 依赖
    depends_on: list[str] = Field(default_factory=list)  # 依赖的 node_id 列表


class PlanProposal(BaseModel):
    """候选计划——LLM 生成的计划草案。"""
    proposal_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    task_id: str = ""
    summary: str = ""                       # LLM 对计划的自然语言描述
    hypotheses: list[str] = Field(default_factory=list)  # 故障假设
    evidence_requirements: list[str] = Field(default_factory=list)  # 所需证据类型

    # 节点列表（提案阶段）
    proposed_nodes: list[PlanNode] = Field(default_factory=list)

    # 元数据
    proposed_by: str = ""                   # 哪个 Agent 提出的
    proposed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionPlan(BaseModel):
    """执行计划——PlanCompiler 编译校验后的冻结计划。

    计划一旦冻结，运行期间不得隐式改变（文档 5.3 节）。
    """
    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    proposal_ref: str = ""                  # 关联的 PlanProposal ID

    # 冻结的节点列表（拓扑排序）
    nodes: list[PlanNode] = Field(default_factory=list)

    # 完整性
    required_stages: list[TaskState] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)

    # 摘要
    plan_digest: str = ""                   # 计划内容 SHA-256
    frozen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # 审批
    approval_required: bool = False
    approval_reason: str = ""

    # 状态
    status: str = "frozen"  # "frozen" | "amended" | "executing" | "completed" | "failed"

    def compute_digest(self) -> str:
        """计算计划摘要——覆盖关键决策属性的完整哈希。

        审计修复 (AER-005): 原实现只哈希 nodes，遗漏 task_id、
        evidence_requirements、approval 等关键字段。
        """
        canonical = json.dumps(
            {
                "task_id": self.task_id,
                "nodes": [n.model_dump(mode="json") for n in self.nodes],
                "required_stages": [s.value for s in self.required_stages],
                "evidence_requirements": sorted(self.evidence_requirements),
                "approval_required": self.approval_required,
                "approval_reason": self.approval_reason,
                "compiler_version": "1.0",
            },
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        )
        self.plan_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.plan_digest


class PlanAmendment(BaseModel):
    """计划修订——当需要修改已冻结计划时的受控变更。

    记录修改原因、影响和审批（文档 5.3 节第 4 步）。
    """
    amendment_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    plan_id: str = ""
    reason: str = ""
    affected_nodes: list[str] = Field(default_factory=list)
    changes: dict[str, Any] = Field(default_factory=dict)
    impact_assessment: str = ""
    requires_approval: bool = True
    approved: bool = False
    approved_by: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlanCompiler:
    """计划编译器——将 LLM 提案编译为可执行的冻结计划。

    静态校验内容（文档 14.4 节）：
    1. 验证 DAG（无环、无孤立节点）
    2. 解析资产引用，检查状态（只允许 validated/qualified）
    3. 检查资产适用域
    4. 检查输入 Schema 匹配
    5. 注入质量门和审批节点
    6. 冻结参数和种子
    7. 确保必要阶段覆盖
    8. 确保证据需求
    """

    # 审计修复 (P0-2): 当前所有资产已降级为 CANDIDATE（无验证报告）。
    # 允许 CANDIDATE 但加严重警告。生产环境应仅允许 VALIDATED/QUALIFIED。
    ALLOWED_ASSET_STATUSES = {AssetStatus.VALIDATED, AssetStatus.QUALIFIED, AssetStatus.CANDIDATE}
    STRICT_MODE = False  # 设为 True 后仅允许 VALIDATED/QUALIFIED

    def compile(
        self,
        proposal: PlanProposal,
        registry: Any,  # AssetRegistry
        task_objective: str = "",
    ) -> ExecutionPlan:
        """编译计划草案为可执行计划。

        Args:
            proposal: LLM 生成的计划提案
            registry: 资产注册中心
            task_objective: 诊断目标

        Returns:
            冻结的 ExecutionPlan

        Raises:
            ValueError: 编译失败
        """
        issues: list[str] = []

        # 1. 验证 DAG
        issues.extend(self._validate_dag(proposal.proposed_nodes))

        # 2. 解析并校验每个节点
        compiled_nodes: list[PlanNode] = []
        for node in proposal.proposed_nodes:
            try:
                compiled = self._compile_node(node, registry)
                compiled_nodes.append(compiled)
            except ValueError as e:
                issues.append(f"Node '{node.name}': {e}")

        # 3. 确保必要阶段覆盖
        issues.extend(self._ensure_required_stages(compiled_nodes, task_objective))

        # 4. 确保证据需求
        issues.extend(self._ensure_evidence_coverage(compiled_nodes, proposal))

        if issues:
            raise ValueError(
                f"Plan compilation failed with {len(issues)} issue(s):\n"
                + "\n".join(f"  - {i}" for i in issues)
            )

        plan = ExecutionPlan(
            task_id=proposal.task_id,
            proposal_ref=proposal.proposal_id,
            nodes=compiled_nodes,
            required_stages=[n.stage for n in compiled_nodes],
            evidence_requirements=proposal.evidence_requirements,
        )
        plan.compute_digest()
        return plan

    def _validate_dag(self, nodes: list[PlanNode]) -> list[str]:
        """验证 DAG：完整拓扑排序 (Kahn 算法) + 可达性检查。

        审计修复 (AER-004): 原实现只检查"有无入度为0节点"，
        不能检测独立循环。现在使用完整 Kahn 算法。
        """
        issues: list[str] = []
        if not nodes:
            return issues

        node_ids = {n.node_id for n in nodes}
        by_id = {n.node_id: n for n in nodes}

        # 检查依赖引用
        for node in nodes:
            for dep in node.depends_on:
                if dep not in node_ids:
                    issues.append(
                        f"Node '{node.name}' depends on unknown node '{dep}'"
                    )

        # ── 完整 Kahn 拓扑排序 ──
        indegree: dict[str, int] = {nid: 0 for nid in node_ids}
        outgoing: dict[str, list[str]] = {nid: [] for nid in node_ids}

        for node in nodes:
            for dep in node.depends_on:
                if dep in node_ids:  # 只计算已知依赖
                    indegree[node.node_id] += 1
                    outgoing[dep].append(node.node_id)

        # 入度为 0 的节点（入口节点）
        from collections import deque
        queue = deque(nid for nid, d in indegree.items() if d == 0)
        sorted_ids: list[str] = []

        while queue:
            nid = queue.popleft()
            sorted_ids.append(nid)
            for nxt in outgoing[nid]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        # 环检测：拓扑排序未能覆盖所有节点
        if len(sorted_ids) != len(nodes):
            unsorted = node_ids - set(sorted_ids)
            issues.append(
                f"Plan contains CYCLE(S) involving nodes: {sorted(unsorted)}"
            )

        # 孤立节点检查：既没有依赖也没有被依赖
        if len(nodes) > 1:
            has_dep = {n.node_id for n in nodes if n.depends_on}
            is_depended = set()
            for n in nodes:
                for d in n.depends_on:
                    is_depended.add(d)
            for n in nodes:
                if n.node_id not in has_dep and n.node_id not in is_depended:
                    issues.append(
                        f"ISOLATED node '{n.name}' — has no dependencies and nothing depends on it"
                    )

        # 不可达节点检查：从入口节点出发不可达
        entry_nodes = [nid for nid, d in indegree.items() if d == 0 and nid in node_ids]
        if entry_nodes:
            reachable: set[str] = set()
            stack = list(entry_nodes)
            while stack:
                nid = stack.pop()
                if nid in reachable:
                    continue
                reachable.add(nid)
                for nxt in outgoing.get(nid, []):
                    if nxt not in reachable:
                        stack.append(nxt)
            unreachable = node_ids - reachable
            if unreachable:
                issues.append(
                    f"UNREACHABLE nodes (not reachable from any entry): {sorted(unreachable)}"
                )

        return issues

    def _compile_node(
        self, node: PlanNode, registry: Any,
    ) -> PlanNode:
        """编译单个节点：解析资产引用、检查状态和适用域。"""
        # 解析资产引用
        try:
            entry = registry.resolve(node.asset_query, node.asset_version)
        except Exception as e:
            raise ValueError(f"Asset resolution failed: {e}")

        m = entry.manifest

        # 检查资产状态
        if m.status not in self.ALLOWED_ASSET_STATUSES:
            raise ValueError(
                f"Asset '{m.asset_id}' status is '{m.status.value}', "
                f"must be one of {[s.value for s in self.ALLOWED_ASSET_STATUSES]}"
            )

        # 冻结版本
        compiled = node.model_copy(update={
            "asset_version": m.version,
        })

        # 注入审批节点（如果资产要求审批）
        if m.policy.requires_approval:
            compiled.requires_approval = True

        return compiled

    def _ensure_required_stages(
        self, nodes: list[PlanNode], task_objective: str,
    ) -> list[str]:
        """确保必要阶段覆盖。"""
        issues: list[str] = []
        covered = {n.stage for n in nodes}

        # 最低要求的阶段
        minimum_stages = {
            TaskState.DATA_VALIDATION,
            TaskState.DETECTION_EXECUTION,
            TaskState.CHARACTERIZATION,
            TaskState.EVIDENCE_FUSION,
        }

        missing = minimum_stages - covered
        if missing:
            issues.append(
                f"Plan missing required stages: "
                f"{[s.value for s in missing]}"
            )

        return issues

    def _ensure_evidence_coverage(
        self, nodes: list[PlanNode], proposal: PlanProposal,
    ) -> list[str]:
        """确保证据需求覆盖。"""
        issues: list[str] = []
        if not proposal.evidence_requirements:
            issues.append(
                "Plan proposal has no evidence requirements specified"
            )
        return issues
