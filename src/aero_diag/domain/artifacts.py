"""统一数据对象（Artifact）——所有阶段间传递的结构化数据信封。

遵循文档第 6 节设计：ArtifactEnvelope 作为所有领域数据的基类，
二进制/大数据保存在对象存储中，消息中只传 URI、哈希、Schema 和摘要。
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    """数据对象类型枚举。"""
    RAW_TIMESERIES = "raw_timeseries"          # 原始时序：振动/温度/转速/EGT
    RAW_IMAGE = "raw_image"                    # 原始图像：孔探照片
    RAW_VIDEO = "raw_video"                    # 原始视频：孔探视频
    RAW_TEXT = "raw_text"                      # 原始文本：维护记录/检查报告
    RAW_STRUCTURED = "raw_structured"          # 原始结构化：载荷谱/材料参数
    DERIVED_TIMESERIES = "derived_timeseries"  # 衍生时序：滤波/重采样后
    DERIVED_IMAGE = "derived_image"            # 衍生图像：校正/裁剪后
    DETECTION_FINDING = "detection_finding"    # 检测发现
    CHARACTERIZATION = "damage_characterization"  # 损伤表征
    RELIABILITY_ASSESSMENT = "reliability_assessment"  # 可靠性/寿命评估
    QUALITY_REPORT = "data_quality_report"     # 数据质量报告
    EVIDENCE_PACKAGE = "evidence_package"      # 证据包
    DECISION_DRAFT = "decision_draft"          # 决策草案
    REVIEW_DECISION = "review_decision"        # 复核决定
    REPORT = "report"                          # 可读报告


DataClassification = Literal[
    "public", "internal", "confidential", "restricted"
]


class ArtifactRef(BaseModel):
    """对 Artifact 的轻量引用，不包含实际数据。"""
    artifact_id: str
    artifact_type: ArtifactType
    uri: str | None = None
    sha256: str = ""
    schema_version: str = "aero.artifact.v1"


class ArtifactEnvelope(BaseModel):
    """统一数据信封——所有领域对象的基础容器。

    二进制或大数据保存在对象存储（uri），消息中仅传此信封。
    """
    artifact_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    artifact_type: ArtifactType
    schema_version: str = "aero.artifact.v1"
    uri: str | None = None               # 对象存储路径
    sha256: str = ""                     # 原始数据 SHA-256
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    producer_asset_id: str | None = None  # 产生此数据的资产 ID
    producer_version: str | None = None   # 资产版本
    run_id: str = ""
    parent_artifact_ids: list[str] = Field(default_factory=list)
    data_classification: DataClassification = "internal"
    units: dict[str, str] = Field(default_factory=dict)  # {"length": "mm", "frequency": "Hz"}
    quality_flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # 结构化载荷
    payload: dict[str, Any] | None = None

    def compute_hash(self, content: bytes) -> str:
        self.sha256 = hashlib.sha256(content).hexdigest()
        return self.sha256

    def to_ref(self) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=self.artifact_id,
            artifact_type=self.artifact_type,
            uri=self.uri,
            sha256=self.sha256,
            schema_version=self.schema_version,
        )


class TimeSeriesBundle(BaseModel):
    """时序数据束——振动/温度/转速/性能参数等。"""
    channels: list[str] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)
    sample_rate: float | None = None
    units: dict[str, str] = Field(default_factory=dict)
    operating_segments: list[dict[str, Any]] = Field(default_factory=list)
    sensor_info: dict[str, dict[str, Any]] = Field(default_factory=dict)
    timezone: str = "UTC"
    gap_markers: list[dict[str, Any]] = Field(default_factory=list)


class ImageCollection(BaseModel):
    """图像/视频集合——孔探图像与视频。"""
    image_uris: list[str] = Field(default_factory=list)
    video_uri: str | None = None
    camera_params: dict[str, Any] = Field(default_factory=dict)  # 内参/外参
    borescope_params: dict[str, Any] = Field(default_factory=dict)  # 镜头/视角
    scale_info: dict[str, Any] = Field(default_factory=dict)  # 比例尺来源
    viewpoint: str = ""  # 拍摄方位
    frame_indices: list[int] = Field(default_factory=list)
