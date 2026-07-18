"""Aero-Engine Damage Detection System — 航空发动机损伤诊断智能体平台。

基于 SeekFlow v0.3.7 可信执行内核，构建面向航空发动机多源检测、
损伤表征、可靠性评估和工程决策的可扩展智能体系统。

遵循《航空发动机损伤诊断智能体系统工程指导文档 V1.2》。
"""

__version__ = "0.1.0"

# ── 领域对象 ──
from aero_diag.domain import (
    # Artifacts
    ArtifactEnvelope,
    ArtifactRef,
    ArtifactType,
    DataClassification,
    ImageCollection,
    TimeSeriesBundle,
    # Task
    EngineInfo,
    InspectionTask,
    OperatingCondition,
    TaskState,
    # Detection
    DetectionFinding,
    OODStatus,
    ScoreSemantics,
    SpatialLocation,
    TemporalLocation,
    # Characterization
    DamageCharacterization,
    DamageConfidence,
    DamageType,
    GeometryDescriptor,
    Severity,
    # Reliability
    ApplicabilityCheck,
    ApplicabilityStatus,
    ModelFamily,
    ParameterBinding,
    ReliabilityAssessment,
    RULDistribution,
    # Evidence
    Claim,
    EvidenceGraph,
    EvidenceItem,
    EvidenceRelation,
    EvidenceStrength,
    EvidenceType,
    # Decision
    ActionType,
    CandidateAction,
    DecisionDraft,
    ReviewDecision,
    RiskLevel,
    # Data Quality
    DataQualityReport,
    QualityFlag,
    QualityStatus,
    # Knowledge
    CaseRecord,
    CaseStatus,
    EvidenceLevel,
    KnowledgeItem,
    KnowledgeType,
    # Uncertainty
    UncertaintyDescriptor,
    UncertaintyType,
)

# ── 工程资产 ──
from aero_diag.assets import (
    AssetKind,
    AssetStatus,
    EngineeringAssetManifest,
)

# ── 注册中心 ──
from aero_diag.registries import (
    AssetRegistry,
    AssetNotFoundError,
)

# ── 数据管理 ──
from aero_diag.artifacts import (
    ArtifactStore,
)

# ── 编排 ──
from aero_diag.orchestration import (
    ExecutionPlan,
    PlanAmendment,
    PlanCompiler,
    PlanNode,
    PlanProposal,
    TaskStateMachine,
    InvalidTransitionError,
)

# ── Agent ──
from aero_diag.agents import (
    AgentRole,
    ALL_ROLES,
    get_role,
    PLANNER_ROLE,
    DATA_QUALITY_ROLE,
    DETECTION_ROLE,
    CHARACTERIZATION_ROLE,
    RELIABILITY_ROLE,
    DECISION_ROLE,
    MONITOR_ROLE,
)

# ── 服务 ──
from aero_diag.services import (
    TaskService,
    EvidenceService,
    ReviewService,
    MonitoringService,
)

# ── API ──
from aero_diag.api import app, create_app

# ── 配置 ──
from aero_diag.infrastructure import config

__all__ = [
    "__version__",
    # Domain
    "ArtifactEnvelope", "ArtifactRef", "ArtifactType", "DataClassification",
    "ImageCollection", "TimeSeriesBundle",
    "EngineInfo", "InspectionTask", "OperatingCondition", "TaskState",
    "DetectionFinding", "OODStatus", "ScoreSemantics", "SpatialLocation", "TemporalLocation",
    "DamageCharacterization", "DamageConfidence", "DamageType", "GeometryDescriptor", "Severity",
    "ApplicabilityCheck", "ApplicabilityStatus", "ModelFamily", "ParameterBinding",
    "ReliabilityAssessment", "RULDistribution",
    "Claim", "EvidenceGraph", "EvidenceItem", "EvidenceRelation", "EvidenceStrength", "EvidenceType",
    "ActionType", "CandidateAction", "DecisionDraft", "ReviewDecision", "RiskLevel",
    "DataQualityReport", "QualityFlag", "QualityStatus",
    "CaseRecord", "CaseStatus", "EvidenceLevel", "KnowledgeItem", "KnowledgeType",
    "UncertaintyDescriptor", "UncertaintyType",
    # Assets
    "AssetKind", "AssetStatus", "EngineeringAssetManifest",
    # Registry
    "AssetRegistry", "AssetNotFoundError",
    # Artifacts
    "ArtifactStore",
    # Orchestration
    "ExecutionPlan", "PlanAmendment", "PlanCompiler", "PlanNode", "PlanProposal",
    "TaskStateMachine", "InvalidTransitionError",
    # Agents
    "AgentRole", "ALL_ROLES", "get_role",
    "PLANNER_ROLE", "DATA_QUALITY_ROLE", "DETECTION_ROLE",
    "CHARACTERIZATION_ROLE", "RELIABILITY_ROLE", "DECISION_ROLE", "MONITOR_ROLE",
    # Services
    "TaskService", "EvidenceService", "ReviewService", "MonitoringService",
    # API
    "app", "create_app",
    # Config
    "config",
]
