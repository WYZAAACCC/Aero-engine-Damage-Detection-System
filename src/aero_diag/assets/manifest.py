"""工程资产清单——EngineeringAssetManifest。

遵循文档第 7.2 节设计：在 SeekFlow ToolManifest 外层新增工程语义描述字段。
包含身份、输入输出、适用域、方法、验证、不确定性、资源、安全策略 9 组必填字段。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AssetKind(str, Enum):
    """资产分类——文档 7.1 节。"""
    DATA_ADAPTER = "data_adapter"          # 数据适配器
    PREPROCESSOR = "preprocessor"          # 预处理工具
    DETECTOR = "detector"                  # 检测算法
    CHARACTERIZER = "characterizer"        # 损伤表征工具
    FUSION = "fusion"                      # 证据融合
    RELIABILITY_MODEL = "reliability_model"  # 可靠性/寿命模型
    KNOWLEDGE_SOURCE = "knowledge_source"   # 专家知识库
    CASE_SOURCE = "case_source"            # 历史案例库
    DECISION_RULE = "decision_rule"        # 决策规则
    MONITOR = "monitor"                    # 监控插件


class AssetStatus(str, Enum):
    """资产状态——文档 16.1 节状态机。

    draft → candidate → validated → qualified → deprecated → retired
                   ↘ rejected
    """
    DRAFT = "draft"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    QUALIFIED = "qualified"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class InputSpec(BaseModel):
    """资产输入规范。"""
    artifact_type: str = ""          # 期望的 ArtifactType
    required_fields: list[str] = Field(default_factory=list)
    required_channels: list[str] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    allowed_file_types: list[str] = Field(default_factory=list)
    allow_missing: list[str] = Field(default_factory=list)


class OutputSpec(BaseModel):
    """资产输出规范。"""
    artifact_type: str = ""
    schema_ref: str = ""  # JSON Schema 文件路径
    description: str = ""


class ApplicabilitySpec(BaseModel):
    """适用域规范——防止模型外推。"""
    components: list[str] = Field(default_factory=list)
    damage_types: list[str] = Field(default_factory=list)
    operating_modes: list[str] = Field(default_factory=list)
    speed_range_rpm: tuple[float, float] | None = None
    temperature_range_c: tuple[float, float] | None = None
    sensor_types: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)  # 明确不适用条件


class MethodSpec(BaseModel):
    """方法规范。"""
    family: str = ""               # 方法族
    deterministic: bool = True     # 是否确定性的
    assumptions: list[str] = Field(default_factory=list)
    parameters_schema: str = ""    # 参数 JSON Schema 路径
    default_parameters: dict[str, Any] = Field(default_factory=dict)


class VerificationSpec(BaseModel):
    """验证信息规范。"""
    validation_dataset_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    baseline: dict[str, float] = Field(default_factory=dict)
    reviewer: str = ""
    reviewed_at: str = ""
    report_uri: str = ""
    valid_until: str = ""


class UncertaintySpec(BaseModel):
    """不确定性规范。"""
    output_representation: str = ""  # "interval" / "distribution" / "set" / "none"
    calibration_method: str = ""
    ood_checks: list[str] = Field(default_factory=list)
    error_sources: list[str] = Field(default_factory=list)


class ResourceSpec(BaseModel):
    """资源需求规范。"""
    cpu: int = 1
    memory_mb: int = 512
    gpu: bool = False
    gpu_memory_mb: int = 0
    timeout_s: float = 60.0
    image: str = ""                # 容器镜像
    parallel_safe: bool = False
    dependencies: list[str] = Field(default_factory=list)


class PolicySpec(BaseModel):
    """安全策略规范。"""
    risk: str = "read"
    capabilities: list[str] = Field(default_factory=list)
    requires_approval: bool = False
    allowed_network_domains: list[str] = Field(default_factory=list)
    workspace_required: bool = False
    data_classification: str = "internal"


class EngineeringAssetManifest(BaseModel):
    """工程资产清单——完整描述一个可注册执行的工程资产。

    包含文档第 7.3 节定义的 9 组必填字段：身份/版本、输入输出、
    适用域、方法/参数、验证、不确定性、资源/执行、安全策略、可观测性。
    """
    schema_version: str = "aero.asset.v1"

    # 1. 身份与版本
    asset_id: str = ""             # 唯一标识，如 "detector.vibration.order_tracking"
    name: str = ""
    version: str = "1.0.0"
    asset_kind: AssetKind = AssetKind.DETECTOR
    publisher: str = ""
    status: AssetStatus = AssetStatus.DRAFT
    description: str = ""
    digest: str = ""               # Manifest 自身的 SHA-256

    # 平台工具入口
    entrypoint_tool: str = "run_engineering_asset"
    implementation_ref: str = ""   # SeekFlow ToolManifest 引用

    # 2. 输入输出
    inputs: list[InputSpec] = Field(default_factory=list)
    outputs: list[OutputSpec] = Field(default_factory=list)

    # 3. 适用域
    applicability: ApplicabilitySpec = Field(default_factory=ApplicabilitySpec)

    # 4. 方法与参数
    method: MethodSpec = Field(default_factory=MethodSpec)

    # 5. 验证
    verification: VerificationSpec = Field(default_factory=VerificationSpec)

    # 6. 不确定性
    uncertainty: UncertaintySpec = Field(default_factory=UncertaintySpec)

    # 7. 资源与执行
    resources: ResourceSpec = Field(default_factory=ResourceSpec)

    # 8. 安全策略
    policy: PolicySpec = Field(default_factory=PolicySpec)

    # 9. 可观测性
    metrics_keys: list[str] = Field(default_factory=list)
    health_check_endpoint: str = ""
    known_failure_modes: list[str] = Field(default_factory=list)

    # 10. 资产集成元数据（扩展字段）
    intro_level: str = ""   # L1_CORE / L2_RECOMMENDED / L3_OPTIONAL / L4_EXPERIMENTAL
    priority: str = ""      # P0_CRITICAL / P1_HIGH / P2_MEDIUM / P3_LOW / P4_DEFERRED
    impl_method: str = ""   # direct_pip / git_clone / huggingface / scikit_builtin / ...
    impl_source: str = ""   # GitHub URL / HuggingFace model ID / PyPI package name
    impl_notes: str = ""    # 实现注意事项

    # 风险与限制
    risk_notes: str = ""    # 使用风险和注意事项
    limitation_notes: str = ""  # 功能限制

    def compute_digest(self) -> str:
        """计算 Manifest 的 SHA-256 摘要。"""
        canonical = json.dumps(
            self.model_dump(mode="json", exclude={"digest"}),
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        )
        self.digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return self.digest
