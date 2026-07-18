"""检测发现——描述检测到的现象，不等同于最终损伤诊断。

遵循文档第 9.1 节设计：DetectionFinding 以统一 Schema 输出，
无论算法来源（Python/MATLAB/C++/外部服务）。
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from .uncertainty import UncertaintyDescriptor


class ScoreSemantics(str, Enum):
    """分数语义——不同算法的 score 含义不同，禁止直接平均。"""
    PROBABILITY = "probability"
    ANOMALY_SCORE = "anomaly_score"
    SIMILARITY = "similarity"
    RULE_STRENGTH = "rule_strength"
    CONFIDENCE = "confidence"


class OODStatus(str, Enum):
    """分布外状态。"""
    IN_DOMAIN = "in_domain"
    WARNING = "warning"
    OUT_OF_DOMAIN = "out_of_domain"
    UNKNOWN = "unknown"


class SpatialLocation(BaseModel):
    """空间位置——发动机部件坐标系下的定位。"""
    component: str = ""            # "HPT Blade Stage 1"
    subregion: str = ""           # "pressure_side" / "suction_side"
    axial_position: str = ""      # "leading_edge" / "mid_chord" / "trailing_edge"
    radial_position: str = ""     # "tip" / "mid_span" / "root"
    coordinate_system: str = ""   # "engine_axial_radial_circumferential"
    coordinates: dict[str, float] = Field(default_factory=dict)


class TemporalLocation(BaseModel):
    """时间位置。"""
    time_s: float | None = None
    time_range: tuple[float, float] | None = None
    frame_index: int | None = None
    operating_segment: str = ""


class DetectionFinding(BaseModel):
    """检测发现——描述检测到的现象。

    "Finding 描述检测到的现象，不直接等同于最终损伤"（文档 9.1 节）
    """
    finding_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    target: str = ""              # 检测目标：部件/区域
    phenomenon: str = ""          # 检测到的现象："异常振动阶次" / "视觉异常区域"
    location: SpatialLocation = Field(default_factory=SpatialLocation)
    temporal: TemporalLocation = Field(default_factory=TemporalLocation)

    score: float | None = None
    score_semantics: ScoreSemantics = ScoreSemantics.ANOMALY_SCORE
    threshold: float | None = None

    method_asset_ref: str = ""    # 使用的检测资产 ID@version
    input_artifact_ids: list[str] = Field(default_factory=list)
    operating_condition_ref: str = ""

    ood_status: OODStatus = OODStatus.UNKNOWN
    uncertainty: UncertaintyDescriptor | None = None

    supporting_artifacts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    # 元数据
    detection_time_ms: int = 0
