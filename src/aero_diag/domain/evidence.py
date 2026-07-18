"""证据对象与证据图——全链路证据管理。

遵循文档第 11.1 节设计：EvidenceGraph 将 Claim、Evidence、Method、
Artifact、Assumption、Rule、Decision 连接成可追溯的有向图。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    """证据类型。"""
    DATA = "data"                # 原始或衍生数据
    ALGORITHM_OUTPUT = "algorithm_output"  # 算法输出
    KNOWLEDGE_REFERENCE = "knowledge_reference"  # 知识引用
    RULE_APPLICATION = "rule_application"  # 规则应用
    EXPERT_JUDGMENT = "expert_judgment"    # 专家判断
    MODEL_RESULT = "model_result"          # 模型结果
    ASSUMPTION = "assumption"              # 假设


class EvidenceStrength(str, Enum):
    """证据强度。"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INCONCLUSIVE = "inconclusive"


class EvidenceRelation(str, Enum):
    """证据关系类型（文档 11.1 节）。"""
    PRODUCED_BY = "produced_by"          # X produced_by Y（Y 产生了 X）
    DERIVED_FROM = "derived_from"        # X derived_from Y（X 派生自 Y）
    SUPPORTS = "supports"                # 支持
    CONTRADICTS = "contradicts"          # 冲突
    ASSUMES = "assumes"                  # 假设
    APPLIES_RULE = "applies_rule"        # 应用规则
    REVIEWED_BY = "reviewed_by"          # 由谁审核


class EvidenceItem(BaseModel):
    """单条证据条目。"""
    evidence_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    claim: str = ""                     # 证据主张
    evidence_type: EvidenceType = EvidenceType.ALGORITHM_OUTPUT
    artifact_ref: str = ""              # 关联的 Artifact ID
    producer_asset_ref: str = ""        # 产生此证据的资产 ID@version
    strength: EvidenceStrength = EvidenceStrength.MODERATE
    limitations: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Claim(BaseModel):
    """主张——证据图中需要被证明或反驳的陈述。"""
    claim_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    statement: str = ""                 # 陈述内容
    status: str = ""                    # "supported" / "contradicted" / "unresolved"
    supporting_evidence: list[str] = Field(default_factory=list)  # EvidenceItem ID
    contradicting_evidence: list[str] = Field(default_factory=list)


class EvidenceRelationRecord(BaseModel):
    """证据图中的一条边。"""
    source_id: str = ""
    target_id: str = ""
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS
    description: str = ""


class EvidenceGraph(BaseModel):
    """证据图——节点 + 边的有向图。

    最终报告中的每个重要陈述都应能定位到图中的节点（文档 11.1 节）。
    """
    graph_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    evidence_items: dict[str, EvidenceItem] = Field(default_factory=dict)
    claims: dict[str, Claim] = Field(default_factory=dict)
    relations: list[EvidenceRelationRecord] = Field(default_factory=list)
    conflict_count: int = 0
    missing_evidence_types: list[str] = Field(default_factory=list)

    def add_evidence(self, item: EvidenceItem) -> None:
        self.evidence_items[item.evidence_id] = item

    def add_claim(self, claim: Claim) -> None:
        self.claims[claim.claim_id] = claim

    def add_relation(
        self, source_id: str, target_id: str,
        relation: EvidenceRelation, description: str = "",
    ) -> None:
        self.relations.append(EvidenceRelationRecord(
            source_id=source_id, target_id=target_id,
            relation=relation, description=description,
        ))

    def check_conflicts(self) -> list[tuple[str, str]]:
        """检测证据图中的冲突（两个节点之间存在 supports 和 contradicts 边）。"""
        conflicts: list[tuple[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for r in self.relations:
            key = (r.source_id, r.target_id, r.relation.value)
            if key in seen:
                conflicts.append((r.source_id, r.target_id))
            seen.add(key)
        self.conflict_count = len(conflicts)
        return conflicts
