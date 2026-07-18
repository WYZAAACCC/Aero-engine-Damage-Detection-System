"""可靠性评估与剩余寿命——模型选择、参数映射、不确定性传播。

遵循文档第 9.3 节设计：
- 模型选择依据部件、损伤模式、材料、载荷、温度
- 参数映射通过 ParameterBinding 显式记录来源
- 适用域检查在运行前执行
- 输出分位数而非单一值
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .uncertainty import UncertaintyDescriptor


class ModelFamily(str, Enum):
    """可靠性/寿命模型族。"""
    CRACK_GROWTH = "crack_growth"      # 裂纹扩展：Paris Law, NASGRO
    FATIGUE = "fatigue"                # 疲劳：S-N, ε-N, 多轴疲劳
    CREEP = "creep"                    # 蠕变：Larson-Miller, Theta Projection
    WEAR = "wear"                      # 磨损：Archard
    PROBABILISTIC = "probabilistic"    # 概率方法：Monte Carlo, FORM/SORM
    SURROGATE = "surrogate"            # 代理模型
    HYBRID = "hybrid"                  # 混合方法
    EMPIRICAL = "empirical"            # 经验公式


class ApplicabilityStatus(str, Enum):
    """模型适用域检查结果。"""
    IN_DOMAIN = "in_domain"
    EXTRAPOLATION = "extrapolation"    # 外推——需审批
    NOT_APPLICABLE = "not_applicable"


class ParameterBinding(BaseModel):
    """模型参数绑定——显式记录每个参数来自哪个数据源/知识条目。"""
    parameter_name: str
    value: float | None = None
    source_type: str = ""  # "artifact", "knowledge_item", "expert_input", "material_database"
    source_ref: str = ""   # artifact_id / knowledge_id / expert_name
    uncertainty: UncertaintyDescriptor | None = None


class ApplicabilityCheck(BaseModel):
    """模型适用域检查结果。"""
    status: ApplicabilityStatus = ApplicabilityStatus.IN_DOMAIN
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    requires_approval: bool = False


class RULDistribution(BaseModel):
    """剩余寿命分布。"""
    mean_hours: float | None = None
    median_hours: float | None = None
    percentile_2_5: float | None = None
    percentile_97_5: float | None = None
    distribution_family: str = ""
    distribution_params: dict[str, float] = Field(default_factory=dict)
    samples_uri: str | None = None


class ReliabilityAssessment(BaseModel):
    """可靠性/寿命评估结果。"""
    assessment_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])

    # 模型信息
    model_asset_ref: str = ""  # 使用的可靠性模型资产 ID@version
    model_family: ModelFamily = ModelFamily.EMPIRICAL
    model_assumptions: list[str] = Field(default_factory=list)
    applicability: ApplicabilityCheck = Field(default_factory=ApplicabilityCheck)

    # 参数映射
    parameter_bindings: list[ParameterBinding] = Field(default_factory=list)

    # 输入引用
    characterization_refs: list[str] = Field(default_factory=list)  # DamageCharacterization ID
    input_artifact_ids: list[str] = Field(default_factory=list)

    # 结果
    failure_probability: float | None = None
    rul: RULDistribution = Field(default_factory=RULDistribution)
    sensitivity: dict[str, float] = Field(default_factory=dict)  # 参数敏感性

    # 不确定性
    uncertainty: UncertaintyDescriptor | None = None

    # 限制
    limitations: list[str] = Field(default_factory=list)
    open_evidence_gaps: list[str] = Field(default_factory=list)

    # 执行
    run_id: str = ""
    seed: int | None = None
    elapsed_ms: int = 0
