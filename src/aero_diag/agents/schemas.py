"""Agent 结构化输出 Schema — 强制 LLM 输出可验证的 Pydantic 对象。

审计修复 (AER-002): Agent 不能输出自然语言当计划执行。
所有规划输出必须通过 Pydantic 校验。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    """故障假设——Agent 提出的候选诊断假设。"""
    hypothesis_id: str = Field(default="")
    description: str = ""                             # 假设描述
    supporting_evidence_types: list[str] = Field(default_factory=list)  # 所需证据类型
    prior_likelihood: str = ""                        # "high" / "medium" / "low" / "unknown"
    rationale: str = ""                               # 假设依据
    assets_suggested: list[str] = Field(default_factory=list)  # 建议使用的资产 ID


class MissingInformation(BaseModel):
    """缺失信息——Agent 识别的数据/知识缺口。"""
    field: str = ""                                   # 缺失字段名
    description: str = ""
    severity: str = "medium"                          # "blocking" / "high" / "medium" / "low"
    suggested_action: str = ""                        # 建议补数动作


class ProposedNode(BaseModel):
    """Agent 提议的执行计划节点。"""
    node_id: str = ""
    name: str = ""
    stage: str = ""                                   # TaskState 值
    asset_query: str = ""                             # 资产 ID
    depends_on: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""                               # 为什么选这个资产


class EvidenceRequirement(BaseModel):
    """证据需求——Agent 提出的证据收集要求。"""
    evidence_type: str = ""                           # EvidenceType 值
    description: str = ""
    required_for_claim: str = ""                      # 关联的主张
    priority: str = "required"                        # "required" / "recommended" / "optional"


class StopCondition(BaseModel):
    """停止条件——Agent 识别的终止场景。"""
    condition: str = ""
    type: str = "success"                             # "success" / "data_insufficient" / "conflict_unresolvable"
    description: str = ""


class PlannerOutput(BaseModel):
    """结构化规划输出——LLM 的 PlanProposal 必须符合此 Schema。

    文档 5.3 节 + 审计 18.3 节要求：
    - LLM 不能输出自然语言作为计划
    - 必须通过 Pydantic 校验
    - 解析失败必须重试或终止
    """
    summary: str = ""                                 # 自然语言摘要（仅用于人类阅读）
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    proposed_nodes: list[ProposedNode] = Field(default_factory=list)
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list)
    stop_conditions: list[StopCondition] = Field(default_factory=list)
    confidence: str = "medium"                        # "high" / "medium" / "low" — 规划置信度
    notes: str = ""                                   # 附加说明


class DiagnosticReport(BaseModel):
    """最终诊断报告——Agent 的结构化输出。"""
    task_id: str = ""
    engine_info: dict[str, Any] = Field(default_factory=dict)

    # 发现
    detection_findings: list[dict[str, Any]] = Field(default_factory=list)
    damage_characterizations: list[dict[str, Any]] = Field(default_factory=list)

    # 评估
    risk_level: str = "unknown"
    severity: str = "unknown"
    rul_cycles: float | None = None

    # 证据
    evidence_references: list[str] = Field(default_factory=list)  # evidence_ids
    knowledge_references: list[str] = Field(default_factory=list)  # knowledge_ids
    tool_calls_made: list[dict[str, Any]] = Field(default_factory=list)  # 审计追踪

    # 建议
    candidate_actions: list[str] = Field(default_factory=list)
    requires_review: bool = True
    open_questions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    # 元数据
    agent_model: str = ""
    agent_role: str = ""
    total_cost_cny: float = 0.0
    generated_at: str = ""
