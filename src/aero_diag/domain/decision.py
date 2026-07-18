"""决策草案与专家复核——安全关键结论的人机协同。

遵循文档第 11.3 节"三段式输出"和第 12 节"人机协同与审批机制"设计：
- 事实层 / 推断层 / 建议层 分离
- 最终安全关键结论必须由授权专家签署
- 任何修改产生新版本，不能覆盖原决定
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """风险等级。"""
    NEGLIGIBLE = "negligible"  # 可忽略
    LOW = "low"                # 低
    MEDIUM = "medium"          # 中
    HIGH = "high"              # 高
    CRITICAL = "critical"      # 危急
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    """候选处置动作类型。"""
    CONTINUE_OPERATION = "continue_operation"    # 继续服役
    INCREASED_MONITORING = "increased_monitoring"  # 加强监控
    REINSPECTION = "reinspection"               # 复检
    REPAIR = "repair"                           # 维修
    REPLACE = "replace"                         # 更换
    DERATE = "derate"                           # 降功率运行
    IMMEDIATE_SHUTDOWN = "immediate_shutdown"   # 立即停飞
    FURTHER_INVESTIGATION = "further_investigation"  # 需要进一步调查


class CandidateAction(BaseModel):
    """候选处置动作。"""
    action_type: ActionType
    description: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    conditions: list[str] = Field(default_factory=list)  # 适用条件
    rule_refs: list[str] = Field(default_factory=list)   # 引用的规则条款
    urgency: str = ""  # "immediate" / "before_next_flight" / "next_inspection" / "scheduled"


class DecisionDraft(BaseModel):
    """决策草案——三段式输出结构。

    事实层 + 推断层 + 建议层（文档 11.3 节），LLM 组织和解释，
    但风险等级和候选处置由版本化规则、模型结果和审批策略生成。
    """
    draft_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""

    # 事实层
    facts: dict[str, Any] = Field(default_factory=dict)
    fact_summary: str = ""  # 自然语言摘要

    # 推断层
    supported_hypotheses: list[str] = Field(default_factory=list)
    refuted_hypotheses: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)

    # 建议层
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    candidate_actions: list[CandidateAction] = Field(default_factory=list)
    rule_hits: list[str] = Field(default_factory=list)

    # 审核
    requires_review: bool = True
    review_triggers: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    # 元数据
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""  # LLM or agent role


class ReviewDecision(BaseModel):
    """专家复核决定——必须包含审核人身份、决定、时间和签名。

    任何修改都产生新版本，不能覆盖原决定（文档第 12 节）。
    """
    review_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    draft_id: str = ""
    task_id: str = ""

    # 审核人
    reviewer: str = ""          # 审核人姓名
    reviewer_role: str = ""     # "certifying_engineer" / "reliability_engineer" / "chief_inspector"
    reviewer_credentials: str = ""  # 资质/认证编号

    # 决定
    decision: str = ""  # "approved" / "rejected" / "conditional_approval" / "needs_more_evidence"
    comments: str = ""
    conditions: list[str] = Field(default_factory=list)  # 附条件批准的附加条件

    # 证据
    evidence_package_id: str = ""

    # 签名
    signed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signature_hash: str = ""  # 数字签名 / HMAC

    # 版本
    version: int = 1
    supersedes_review_id: str | None = None
