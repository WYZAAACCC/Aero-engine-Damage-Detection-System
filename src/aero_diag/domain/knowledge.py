"""专家知识与历史案例——知识条目和案例记录。

遵循文档第 10 节设计：知识分层（术语/机理/规则/文献/经验/案例），
每条知识必须有来源、版本、适用范围、审核人和失效日期。
案例检索返回"相似案例证据"，不得直接复制历史处置（文档 10.3 节）。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeType(str, Enum):
    """知识类型——文档 10.1 节知识分层。"""
    TERMINOLOGY = "terminology"      # 术语与本体
    MECHANISM = "mechanism"          # 故障机理
    ENGINEERING_RULE = "engineering_rule"  # 工程规则（阈值/检查流程）
    STANDARD_CLAUSE = "standard_clause"    # 文献/标准条款
    EXPERT_EXPERIENCE = "expert_experience"  # 专家经验
    HISTORICAL_CASE = "historical_case"    # 历史案例


class EvidenceLevel(str, Enum):
    """证据等级。"""
    LEVEL_A = "A"  # 金标准案例确认
    LEVEL_B = "B"  # 受控实验验证
    LEVEL_C = "C"  # 专家共识
    LEVEL_D = "D"  # 单一专家意见
    LEVEL_E = "E"  # 未验证


class KnowledgeItem(BaseModel):
    """知识条目——遵循文档 10.2 节的 YAML 规范。"""
    knowledge_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    knowledge_type: KnowledgeType = KnowledgeType.EXPERT_EXPERIENCE
    title: str = ""
    content: str = ""
    source_ref: str = ""           # 来源引用：文献/标准/专家
    source_location: str = ""      # 页码/条款/章节
    applicability: dict[str, Any] = Field(default_factory=dict)  # 适用范围
    evidence_level: EvidenceLevel = EvidenceLevel.LEVEL_D
    author_or_expert: str = ""
    reviewer: str = ""
    effective_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    version: str = "1.0"
    supersedes: str | None = None
    confidentiality: str = "internal"
    related_entities: list[str] = Field(default_factory=list)
    machine_executable_rule_ref: str | None = None  # 对应的可执行规则

    # 可被检索的嵌入向量（可选）
    embedding: list[float] | None = None


class CaseStatus(str, Enum):
    """案例确认状态。"""
    CONFIRMED = "confirmed"      # 有后验结果确认
    PRESUMED = "presumed"        # 推测结论
    UNRESOLVED = "unresolved"    # 尚未有后验结果


class CaseRecord(BaseModel):
    """历史案例记录——文档 10.3 节。

    每个案例保存：任务、输入数据摘要、数据质量、执行计划、资产版本、
    Finding、表征、寿命结果、专家结论、实际处置和后验结果。
    """
    case_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    case_status: CaseStatus = CaseStatus.CONFIRMED

    # 案例摘要
    title: str = ""
    engine_type: str = ""
    component: str = ""
    damage_types: list[str] = Field(default_factory=list)

    # 任务输入摘要
    task_summary: dict[str, Any] = Field(default_factory=dict)
    data_quality_summary: dict[str, Any] = Field(default_factory=dict)

    # 过程记录
    plan_digest: str = ""
    asset_versions: dict[str, str] = Field(default_factory=dict)  # asset_id -> version

    # 结果
    findings_summary: list[str] = Field(default_factory=list)
    characterization_summary: dict[str, Any] = Field(default_factory=dict)
    reliability_summary: dict[str, Any] = Field(default_factory=dict)

    # 专家结论
    expert_conclusion: str = ""
    actual_disposition: str = ""  # 实际处置
    posterior_outcome: str = ""   # 后验结果（是否确认）

    # 脱敏
    anonymized: bool = True
    date: str = ""
    source_organization: str = ""

    # 检索
    keywords: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
