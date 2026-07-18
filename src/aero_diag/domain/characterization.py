"""损伤表征——损伤类型、位置、几何量、严重度和不确定性。

遵循文档第 9.2 节设计：表征对象与部件坐标系绑定，
输出区分"观测损伤""推定损伤"和"待确认损伤"。
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .detection import SpatialLocation
from .uncertainty import UncertaintyDescriptor


class DamageType(str, Enum):
    """常见航空发动机损伤类型（从故障机理知识库扩展）。"""
    CRACK = "crack"                      # 裂纹
    COATING_SPALLATION = "coating_spallation"  # 涂层剥落
    EROSION = "erosion"                  # 冲蚀
    CORROSION = "corrosion"              # 腐蚀
    BURN_MARK = "burn_mark"              # 烧蚀
    DENT = "dent"                        # 凹陷
    FOD = "foreign_object_damage"        # 外物损伤
    WEAR = "wear"                        # 磨损
    DEFORMATION = "deformation"          # 变形
    RUB = "rub"                          # 碰摩痕迹
    UNKNOWN = "unknown"                  # 未知类型


class DamageConfidence(str, Enum):
    """损伤确认状态。"""
    OBSERVED = "observed"        # 观测损伤：目视或算法确认
    INFERRED = "inferred"        # 推定损伤：间接证据推断
    SUSPECTED = "suspected"      # 待确认损伤：弱信号，需复检


class GeometryDescriptor(BaseModel):
    """损伤几何量描述。

    几何量必须记录测量方法、比例尺、像素到物理量转换、重复测量误差和可见性。
    没有标尺或标定时，只允许输出像素/相对尺度或区间，不得伪造毫米值（文档 9.2 节）。
    """
    length_mm: float | None = None
    width_mm: float | None = None
    depth_mm: float | None = None
    area_mm2: float | None = None
    relative_size: str = ""      # 相对尺度描述
    pixel_extent: tuple[int, int] | None = None  # 像素范围

    measurement_method: str = ""  # "stereo_photogrammetry", "laser_triangulation", "scale_reference"
    scale_source: str = ""        # 比例尺来源：标尺类型、标定方法
    calibration_method: str = ""
    pixel_to_physical: dict[str, Any] = Field(default_factory=dict)
    viewpoint_distortion: str = ""
    repeat_measurement_error: dict[str, float] = Field(default_factory=dict)


class Severity(BaseModel):
    """严重度评估。"""
    level: str = ""  # "minor" / "moderate" / "severe" / "critical"
    criteria: str = ""  # 分级依据
    rule_ref: str | None = None  # 引用的规则条目


class DamageCharacterization(BaseModel):
    """损伤表征——对检测发现进行定量描述和分类。"""
    characterization_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])

    # 损伤标识
    damage_type: DamageType = DamageType.UNKNOWN
    damage_confidence: DamageConfidence = DamageConfidence.OBSERVED

    # 位置
    component_location: SpatialLocation = Field(default_factory=SpatialLocation)

    # 几何
    geometry: GeometryDescriptor = Field(default_factory=GeometryDescriptor)

    # 严重度
    severity: Severity = Field(default_factory=Severity)

    # 不确定性
    uncertainty: UncertaintyDescriptor | None = None

    # 证据溯源
    finding_refs: list[str] = Field(default_factory=list)  # 关联的 DetectionFinding ID
    characterization_method_ref: str = ""  # 使用的表征资产 ID@version
    input_artifact_ids: list[str] = Field(default_factory=list)

    # 审核
    reviewer: str = ""
    reviewed_at: str = ""
    notes: str = ""
