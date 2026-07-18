"""Agent 角色定义——受约束的职责视图与受限工具集。"""

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

__all__ = [
    "ALL_ROLES",
    "AgentRole",
    "CHARACTERIZATION_ROLE",
    "DATA_QUALITY_ROLE",
    "DECISION_ROLE",
    "DETECTION_ROLE",
    "MONITOR_ROLE",
    "PLANNER_ROLE",
    "RELIABILITY_ROLE",
    "get_role",
]
