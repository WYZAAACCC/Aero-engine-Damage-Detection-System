"""FastAPI 依赖注入——提供各服务单例。"""

from aero_diag.artifacts.store import ArtifactStore
from aero_diag.registries.asset_registry import AssetRegistry
from aero_diag.services.evidence_service import EvidenceService
from aero_diag.services.monitoring_service import MonitoringService
from aero_diag.services.review_service import ReviewService
from aero_diag.services.task_service import TaskService

# 全局单例
_task_service: TaskService | None = None
_asset_registry: AssetRegistry | None = None
_artifact_store: ArtifactStore | None = None
_evidence_service: EvidenceService | None = None
_review_service: ReviewService | None = None
_monitoring_service: MonitoringService | None = None


def get_task_service() -> TaskService:
    global _task_service
    if _task_service is None:
        _task_service = TaskService()
    return _task_service


def get_asset_registry() -> AssetRegistry:
    global _asset_registry
    if _asset_registry is None:
        _asset_registry = AssetRegistry()
    return _asset_registry


def get_artifact_store() -> ArtifactStore:
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store


def get_evidence_service() -> EvidenceService:
    global _evidence_service
    if _evidence_service is None:
        _evidence_service = EvidenceService()
    return _evidence_service


def get_review_service() -> ReviewService:
    global _review_service
    if _review_service is None:
        _review_service = ReviewService()
    return _review_service


def get_monitoring_service() -> MonitoringService:
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service
