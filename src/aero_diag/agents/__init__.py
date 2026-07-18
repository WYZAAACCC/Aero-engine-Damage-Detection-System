"""Agent 角色定义、控制器、工具绑定与结构化 Schema。"""

from .roles import (
    ALL_ROLES,
    CHARACTERIZATION_ROLE,
    DATA_QUALITY_ROLE,
    DECISION_ROLE,
    DETECTION_ROLE,
    MONITOR_ROLE,
    PLANNER_ROLE,
    RELIABILITY_ROLE,
    AgentRole,
    get_role,
)
from .controller import DomainAgentController
from .tool_bindings import (
    PlatformTool,
    get_tools_for_role,
    get_all_platform_tools,
    SEARCH_ASSETS_TOOL,
    RETRIEVE_KNOWLEDGE_TOOL,
    PROPOSE_PLAN_TOOL,
    REQUEST_REVIEW_TOOL,
)
from .schemas import (
    PlannerOutput,
    DiagnosticReport,
    Hypothesis,
    MissingInformation,
    ProposedNode,
    EvidenceRequirement,
    StopCondition,
)

__all__ = [
    # Roles
    "ALL_ROLES",
    "AgentRole",
    "CHARACTERIZATION_ROLE", "DATA_QUALITY_ROLE", "DECISION_ROLE",
    "DETECTION_ROLE", "MONITOR_ROLE", "PLANNER_ROLE", "RELIABILITY_ROLE",
    "get_role",
    # Controller
    "DomainAgentController",
    # Tool bindings
    "PlatformTool", "get_tools_for_role", "get_all_platform_tools",
    "SEARCH_ASSETS_TOOL", "RETRIEVE_KNOWLEDGE_TOOL",
    "PROPOSE_PLAN_TOOL", "REQUEST_REVIEW_TOOL",
    # Schemas
    "PlannerOutput", "DiagnosticReport",
    "Hypothesis", "MissingInformation", "ProposedNode",
    "EvidenceRequirement", "StopCondition",
]
