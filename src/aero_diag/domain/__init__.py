"""领域对象层——所有阶段间传递的结构化 Pydantic 对象。

统一导出遵循文档第 6 节的领域数据与证据对象体系。
"""

from .artifacts import (
    ArtifactEnvelope,
    ArtifactRef,
    ArtifactType,
    DataClassification,
    ImageCollection,
    TimeSeriesBundle,
)
from .characterization import (
    DamageCharacterization,
    DamageConfidence,
    DamageType,
    GeometryDescriptor,
    Severity,
)
from .data_quality import DataQualityReport, QualityFlag, QualityStatus
from .decision import (
    ActionType,
    CandidateAction,
    DecisionDraft,
    ReviewDecision,
    RiskLevel,
)
from .detection import (
    DetectionFinding,
    OODStatus,
    ScoreSemantics,
    SpatialLocation,
    TemporalLocation,
)
from .evidence import (
    Claim,
    EvidenceGraph,
    EvidenceItem,
    EvidenceRelation,
    EvidenceRelationRecord,
    EvidenceStrength,
    EvidenceType,
)
from .knowledge import (
    CaseRecord,
    CaseStatus,
    EvidenceLevel,
    KnowledgeItem,
    KnowledgeType,
)
from .reliability import (
    ApplicabilityCheck,
    ApplicabilityStatus,
    ModelFamily,
    ParameterBinding,
    ReliabilityAssessment,
    RULDistribution,
)
from .task import (
    EngineInfo,
    InspectionTask,
    OperatingCondition,
    TaskState,
)
from .uncertainty import (
    ConfidenceInterval,
    DistributionDescriptor,
    PointEstimate,
    SetDescriptor,
    UncertaintyDescriptor,
    UncertaintyType,
    UnknownUncertainty,
)

__all__ = [
    # Artifacts
    "ArtifactEnvelope",
    "ArtifactRef",
    "ArtifactType",
    "DataClassification",
    "ImageCollection",
    "TimeSeriesBundle",
    # Task
    "EngineInfo",
    "InspectionTask",
    "OperatingCondition",
    "TaskState",
    # Detection
    "DetectionFinding",
    "OODStatus",
    "ScoreSemantics",
    "SpatialLocation",
    "TemporalLocation",
    # Characterization
    "DamageCharacterization",
    "DamageConfidence",
    "DamageType",
    "GeometryDescriptor",
    "Severity",
    # Reliability
    "ApplicabilityCheck",
    "ApplicabilityStatus",
    "ModelFamily",
    "ParameterBinding",
    "ReliabilityAssessment",
    "RULDistribution",
    # Evidence
    "Claim",
    "EvidenceGraph",
    "EvidenceItem",
    "EvidenceRelation",
    "EvidenceRelationRecord",
    "EvidenceStrength",
    "EvidenceType",
    # Decision
    "ActionType",
    "CandidateAction",
    "DecisionDraft",
    "ReviewDecision",
    "RiskLevel",
    # Data Quality
    "DataQualityReport",
    "QualityFlag",
    "QualityStatus",
    # Knowledge
    "CaseRecord",
    "CaseStatus",
    "EvidenceLevel",
    "KnowledgeItem",
    "KnowledgeType",
    # Uncertainty
    "ConfidenceInterval",
    "DistributionDescriptor",
    "PointEstimate",
    "SetDescriptor",
    "UncertaintyDescriptor",
    "UncertaintyType",
    "UnknownUncertainty",
]
