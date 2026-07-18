"""监控服务——指标收集、告警和干预策略。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AlertSeverity(str, Enum):
    """告警严重度。"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class InterventionAction(str, Enum):
    """干预动作（仅白名单中的低风险动作可自动执行）。"""
    RETRY_NODE = "retry_node"
    SWITCH_BACKUP_ASSET = "switch_backup_asset"
    REQUEST_ADDITIONAL_DATA = "request_additional_data"
    FLAG_FOR_REVIEW = "flag_for_review"
    SUSPEND_TASK = "suspend_task"
    NOTIFY_OPERATOR = "notify_operator"


@dataclass
class MonitoringEvent:
    """监控事件。"""
    event_type: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    source: str = ""                # 事件来源：data_quality / tool / model / workflow / security
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    task_id: str = ""
    node_id: str = ""


@dataclass
class InterventionProposal:
    """干预建议——监控 Agent 提出的行动方案。"""
    event_ref: str = ""
    action: InterventionAction = InterventionAction.FLAG_FOR_REVIEW
    reason: str = ""
    requires_approval: bool = True
    details: dict[str, Any] = field(default_factory=dict)


class MonitoringService:
    """运行监控服务——收集指标、触发告警、管理干预策略。

    遵循文档第 13 节设计：监控 6 个域（数据/工具/模型/流程/安全/基础设施）。
    """

    AUTO_APPROVED_ACTIONS: set[InterventionAction] = {
        InterventionAction.RETRY_NODE,
        InterventionAction.NOTIFY_OPERATOR,
        InterventionAction.FLAG_FOR_REVIEW,
    }

    def __init__(self) -> None:
        self._events: list[MonitoringEvent] = []
        self._interventions: list[InterventionProposal] = []
        self._alerts: dict[str, list[MonitoringEvent]] = {}  # task_id -> events

    def record_event(
        self,
        *,
        event_type: str,
        severity: AlertSeverity,
        source: str,
        message: str,
        metrics: dict[str, Any] | None = None,
        task_id: str = "",
        node_id: str = "",
    ) -> MonitoringEvent:
        """记录一个监控事件。"""
        event = MonitoringEvent(
            event_type=event_type,
            severity=severity,
            source=source,
            message=message,
            metrics=metrics or {},
            task_id=task_id,
            node_id=node_id,
        )
        self._events.append(event)
        if task_id:
            if task_id not in self._alerts:
                self._alerts[task_id] = []
            self._alerts[task_id].append(event)
        return event

    def propose_intervention(
        self,
        event: MonitoringEvent,
        action: InterventionAction,
        reason: str = "",
        details: dict[str, Any] | None = None,
    ) -> InterventionProposal:
        """提出干预建议。"""
        requires_approval = action not in self.AUTO_APPROVED_ACTIONS
        proposal = InterventionProposal(
            event_ref=event.event_type,
            action=action,
            reason=reason or event.message,
            requires_approval=requires_approval,
            details=details or {},
        )
        self._interventions.append(proposal)
        return proposal

    def get_events(
        self, task_id: str | None = None, severity: AlertSeverity | None = None,
    ) -> list[MonitoringEvent]:
        """查询事件，可选按任务和严重度过滤。"""
        events = self._alerts.get(task_id, self._events) if task_id else self._events
        if severity is not None:
            events = [e for e in events if e.severity == severity]
        return events

    def get_metrics_summary(self, task_id: str) -> dict[str, Any]:
        """获取任务的指标摘要。"""
        events = self._alerts.get(task_id, [])
        return {
            "total_events": len(events),
            "by_severity": {
                s.value: len([e for e in events if e.severity == s])
                for s in AlertSeverity
            },
            "by_source": {
                src: len([e for e in events if e.source == src])
                for src in {"data_quality", "tool", "model", "workflow", "security", "infrastructure"}
            },
        }
