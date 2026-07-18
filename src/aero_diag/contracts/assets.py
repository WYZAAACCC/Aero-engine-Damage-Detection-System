"""工程资产接口协议——定义所有资产插件必须实现的接口。

遵循文档第 14.3 节设计：EngineeringAsset Protocol 确保
统一数据适配器、预处理工具、检测算法等资产的可替换性。
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from aero_diag.domain.artifacts import ArtifactRef


class ValidationReport(BaseModel):
    """资产输入验证报告。"""
    ok: bool = True
    issues: list[str] = []
    warnings: list[str] = []


class HealthReport(BaseModel):
    """资产健康检查报告。"""
    healthy: bool = True
    details: dict[str, Any] = {}
    last_checked: str = ""


class ProvenanceRecord(BaseModel):
    """溯源记录。"""
    asset_id: str = ""
    asset_version: str = ""
    parameters_snapshot: dict[str, Any] = {}
    input_hashes: list[str] = []
    run_id: str = ""
    seed: int | None = None
    environment_digest: str = ""
    start_time: str = ""
    end_time: str = ""
    elapsed_ms: int = 0


class AssetRunResult(BaseModel):
    """资产运行结果。"""
    status: Literal["success", "partial", "failed", "needs_review"]
    output_artifacts: list[ArtifactRef] = []
    structured_output: dict[str, Any] = {}
    evidence_items: list = []  # EvidenceItem
    uncertainty: Any | None = None
    warnings: list[str] = []
    metrics: dict[str, float] = {}
    provenance: ProvenanceRecord | None = None


@runtime_checkable
class EngineeringAsset(Protocol):
    """工程资产协议——所有插件资产必须实现的方法。

    对应文档 14.3 节 EngineeringAsset 接口。
    """
    manifest: Any  # EngineeringAssetManifest

    def validate_inputs(
        self,
        inputs: list[ArtifactRef],
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> ValidationReport:
        """校验输入数据是否满足资产要求。"""
        ...

    def run(
        self,
        inputs: list[ArtifactRef],
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> AssetRunResult:
        """执行资产并返回结构化结果。"""
        ...

    def health_check(self) -> HealthReport:
        """健康检查。"""
        ...
