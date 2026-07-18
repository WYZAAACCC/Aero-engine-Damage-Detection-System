"""证据服务——证据追加、关联、查询和打包。"""

from __future__ import annotations

import uuid
from typing import Any

from aero_diag.domain.evidence import (
    Claim,
    EvidenceGraph,
    EvidenceItem,
    EvidenceRelation,
    EvidenceStrength,
    EvidenceType,
)


class EvidenceService:
    """证据管理服务——构建和查询证据图。"""

    def __init__(self) -> None:
        self._graphs: dict[str, EvidenceGraph] = {}  # task_id -> EvidenceGraph

    def get_or_create_graph(self, task_id: str) -> EvidenceGraph:
        """获取或创建任务的证据图。"""
        if task_id not in self._graphs:
            self._graphs[task_id] = EvidenceGraph(task_id=task_id)
        return self._graphs[task_id]

    def append_evidence(
        self,
        task_id: str,
        *,
        claim: str,
        evidence_type: EvidenceType,
        artifact_ref: str = "",
        producer_asset_ref: str = "",
        strength: EvidenceStrength = EvidenceStrength.MODERATE,
        limitations: list[str] | None = None,
    ) -> EvidenceItem:
        """追加一条证据。"""
        item = EvidenceItem(
            claim=claim,
            evidence_type=evidence_type,
            artifact_ref=artifact_ref,
            producer_asset_ref=producer_asset_ref,
            strength=strength,
            limitations=limitations or [],
        )
        graph = self.get_or_create_graph(task_id)
        graph.add_evidence(item)
        return item

    def link(
        self,
        task_id: str,
        source_id: str,
        target_id: str,
        relation: EvidenceRelation,
        description: str = "",
    ) -> None:
        """在证据图中添加关联。"""
        graph = self.get_or_create_graph(task_id)
        graph.add_relation(source_id, target_id, relation, description)

    def query_evidence(
        self, task_id: str, evidence_type: EvidenceType | None = None,
    ) -> list[EvidenceItem]:
        """查询证据条目，可选按类型过滤。"""
        graph = self._graphs.get(task_id)
        if graph is None:
            return []
        if evidence_type is None:
            return list(graph.evidence_items.values())
        return [
            e for e in graph.evidence_items.values()
            if e.evidence_type == evidence_type
        ]

    def check_conflicts(self, task_id: str) -> list[tuple[str, str]]:
        """检测证据冲突。"""
        graph = self._graphs.get(task_id)
        if graph is None:
            return []
        return graph.check_conflicts()

    def package(self, task_id: str) -> dict[str, Any]:
        """打包证据图用于审计和归档。"""
        graph = self._graphs.get(task_id)
        if graph is None:
            return {"task_id": task_id, "evidence_count": 0, "items": [], "relations": []}
        return {
            "graph_id": graph.graph_id,
            "task_id": graph.task_id,
            "evidence_count": len(graph.evidence_items),
            "claim_count": len(graph.claims),
            "relation_count": len(graph.relations),
            "conflict_count": graph.conflict_count,
            "missing_evidence_types": graph.missing_evidence_types,
            "items": [e.model_dump(mode="json") for e in graph.evidence_items.values()],
            "claims": [c.model_dump(mode="json") for c in graph.claims.values()],
            "relations": [r.model_dump(mode="json") for r in graph.relations],
        }
