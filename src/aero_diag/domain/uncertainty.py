"""不确定性统一表示——贯穿诊断全链路。

遵循文档第 11.2 节设计：5 种不确定性表示形式，
每种包含明确的来源、方法和适用说明。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UncertaintyType(str, Enum):
    """不确定性表示类型。"""
    POINT_ESTIMATE = "point_estimate"    # 点估计 + 标准差
    CONFIDENCE_INTERVAL = "confidence_interval"  # 置信/可信区间
    DISTRIBUTION = "distribution"        # 概率分布 / 样本
    SET_BASED = "set_based"              # 集合 / 模糊等级
    UNKNOWN = "unknown"                  # 未知 / 不可量化


class PointEstimate(BaseModel):
    """点估计 + 标准差。"""
    value: float
    std: float
    method: str = ""  # 如 "bootstrap", "analytical", "calibration"


class ConfidenceInterval(BaseModel):
    """置信/可信区间。"""
    lower: float
    upper: float
    level: float = 0.95  # 置信水平
    interpretation: str = ""  # "frequentist_confidence" | "bayesian_credible"


class DistributionDescriptor(BaseModel):
    """概率分布描述。"""
    family: str = ""  # "normal", "weibull", "lognormal", "empirical"
    parameters: dict[str, float] = Field(default_factory=dict)
    samples_uri: str | None = None  # 指向样本数据文件
    seed: int | None = None
    correlation: dict[str, float] = Field(default_factory=dict)


class SetDescriptor(BaseModel):
    """集合 / 模糊等级。"""
    candidates: list[tuple[str, float]] = Field(default_factory=list)  # (标签, 隶属度/权重)
    source: str = ""  # "expert_judgment", "rule_set"


class UnknownUncertainty(BaseModel):
    """未知 / 不可量化不确定性。"""
    reason: str = ""  # 为何不可量化
    impact: str = ""  # 对下游的影响
    required_evidence: list[str] = Field(default_factory=list)  # 需要什么证据才能量化


class UncertaintyDescriptor(BaseModel):
    """统一不确定性描述符。"""
    kind: UncertaintyType
    point: PointEstimate | None = None
    interval: ConfidenceInterval | None = None
    distribution: DistributionDescriptor | None = None
    set_based: SetDescriptor | None = None
    unknown: UnknownUncertainty | None = None
    notes: str = ""
