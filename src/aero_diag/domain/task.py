"""诊断任务定义——系统的入口与状态承载对象。

遵循文档第 5.2 节状态机设计：13 个任务状态，
每次状态迁移必须满足前置条件并写入审计事件。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .artifacts import ArtifactRef, DataClassification


class TaskState(str, Enum):
    """诊断任务状态——文档 5.2 节定义的完整状态机。

    迁移路径:
    RECEIVED → DATA_VALIDATION → PLAN_PROPOSAL → PLAN_COMPILE
    → DETECTION_EXECUTION → CHARACTERIZATION → RELIABILITY_ASSESSMENT
    → EVIDENCE_FUSION → DECISION_DRAFT → EXPERT_REVIEW → APPROVED/REJECTED/REWORK → ARCHIVED

    可从多个状态回退到 NEED_MORE_DATA 或数据质量门失败。
    """
    RECEIVED = "received"                        # 任务已接收
    DATA_VALIDATION = "data_validation"          # 数据质量门
    NEED_MORE_DATA = "need_more_data"            # 需要补充数据
    PLAN_PROPOSAL = "plan_proposal"              # 计划提案
    PLAN_COMPILE = "plan_compile"                # 计划编译校验
    PLAN_REVIEW_REQUIRED = "plan_review_required"  # 计划需审批（高风险）
    DETECTION_EXECUTION = "detection_execution"   # 检测执行
    CHARACTERIZATION = "characterization"         # 损伤表征
    RELIABILITY_ASSESSMENT = "reliability_assessment"  # 可靠性/寿命评估
    EVIDENCE_FUSION = "evidence_fusion"           # 证据融合
    DECISION_DRAFT = "decision_draft"             # 决策草案
    EXPERT_REVIEW = "expert_review"              # 专家复核
    APPROVED = "approved"                        # 已批准
    REJECTED = "rejected"                        # 已驳回
    REWORK = "rework"                            # 需返工
    ARCHIVED = "archived"                        # 已归档


class OperatingCondition(BaseModel):
    """运行工况——作为一等对象。"""
    mode: str = ""  # "runup", "steady", "rundown", "transient"
    speed_rpm: tuple[float, float] | None = None  # (min, max)
    load: tuple[float, float] | None = None
    ambient_temp_c: float | None = None
    altitude_m: float | None = None
    duration_s: float | None = None
    segment_label: str = ""


class EngineInfo(BaseModel):
    """发动机基本信息。"""
    engine_type: str = ""      # 型号，如 "PW4000", "CFM56"
    engine_serial: str = ""    # 发动机序列号
    component: str = ""        # 目标部件，如 "HPT Blade Stage 1"
    component_location: str = ""  # 部件位置描述
    total_hours: float | None = None  # 总运行小时
    total_cycles: int | None = None   # 总循环数
    last_overhaul_hours: float | None = None
    last_overhaul_date: str = ""


class InspectionTask(BaseModel):
    """诊断任务——系统入口的核心领域对象。"""
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    description: str = ""  # 自然语言任务描述（供 LLM 理解）
    objective: str = ""    # 诊断目标

    # 发动机信息
    engine: EngineInfo = Field(default_factory=EngineInfo)
    operating_conditions: OperatingCondition = Field(default_factory=OperatingCondition)

    # 状态
    state: TaskState = TaskState.RECEIVED

    # 输入数据引用
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)

    # 约束
    constraints: dict[str, Any] = Field(default_factory=dict)  # {"max_cost": 0.5, "priority": "normal"}
    required_evidence_types: list[str] = Field(default_factory=list)

    # 审计
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # 安全
    data_classification: DataClassification = "internal"
    confidentiality_note: str = ""

    # 关联
    parent_task_id: str | None = None
    batch_id: str | None = None
