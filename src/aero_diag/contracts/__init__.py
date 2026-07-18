"""契约/协议层——所有插件和服务必须实现的接口 Protocol。"""

from .assets import (
    AssetRunResult,
    EngineeringAsset,
    HealthReport,
    ProvenanceRecord,
    ValidationReport,
)

__all__ = [
    "AssetRunResult",
    "EngineeringAsset",
    "HealthReport",
    "ProvenanceRecord",
    "ValidationReport",
]
