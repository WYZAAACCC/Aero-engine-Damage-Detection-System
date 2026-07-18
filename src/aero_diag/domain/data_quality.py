"""数据质量报告——数据质量门的结构化输出。

遵循文档第 8.2 节设计：数据质量门由可测试的 DataQualityService 产生，
Agent 负责解释报告并提出补数建议。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class QualityStatus(str, Enum):
    """数据质量检查状态。"""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_CHECKED = "not_checked"


class QualityFlag(BaseModel):
    """单条质量检查标记。"""
    category: str = ""       # "structural" / "numerical" / "temporal" / "unit" / "source" / "fitness"
    field: str = ""          # 被检查的字段或通道
    status: QualityStatus = QualityStatus.NOT_CHECKED
    message: str = ""
    metric_name: str = ""    # 指标名
    metric_value: float | None = None
    threshold: float | None = None


class DataQualityReport(BaseModel):
    """数据质量报告——由 DataQualityService 生成。

    质量门至少包括 7 个维度（文档 8.2 节）：
    结构完整性、数值有效性、时间一致性、单位和量纲、来源与标定、任务适用性、隐私与保密。
    """
    report_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    artifact_refs: list[str] = Field(default_factory=list)  # 被检查的 Artifact ID

    # 七个质量维度
    structural: list[QualityFlag] = Field(default_factory=list)    # 结构完整性
    numerical: list[QualityFlag] = Field(default_factory=list)     # 数值有效性
    temporal: list[QualityFlag] = Field(default_factory=list)      # 时间一致性
    unit_dimensional: list[QualityFlag] = Field(default_factory=list)  # 单位和量纲
    source_calibration: list[QualityFlag] = Field(default_factory=list)  # 来源与标定
    fitness: list[QualityFlag] = Field(default_factory=list)       # 任务适用性
    privacy: list[QualityFlag] = Field(default_factory=list)       # 隐私与保密

    # 汇总
    overall_status: QualityStatus = QualityStatus.NOT_CHECKED
    passed_count: int = 0
    warn_count: int = 0
    fail_count: int = 0
    recommendation: str = ""  # "proceed" / "needs_supplement" / "blocked"

    # 审计
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checked_by: str = ""      # DataQualityService 版本

    @property
    def is_blocking(self) -> bool:
        return self.overall_status == QualityStatus.FAIL

    @property
    def all_flags(self) -> list[QualityFlag]:
        """返回所有质量标记的合并列表。"""
        return (
            self.structural + self.numerical + self.temporal
            + self.unit_dimensional + self.source_calibration
            + self.fitness + self.privacy
        )
