# agentapi/models/general_models.py
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
from datetime import datetime
from enum import Enum

@dataclass
class APIResponse:
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result
        
    @staticmethod
    def success_response(data: Any = None, message: str = "Operation successful") -> Dict[str, Any]:
        return APIResponse(success=True, message=message, data=data).to_dict()
        
    @staticmethod
    def error_response(message: str, code: int = 400, details: Any = None) -> Dict[str, Any]:
        return APIResponse(success=False, message=message, data={"error_code": code, "details": details}).to_dict()

class AuditEventType(Enum):
    AGENT_CREATED = "AGENT_CREATED"
    AGENT_DELETED = "AGENT_DELETED"
    AGENT_ACTION = "AGENT_ACTION"
    MESSAGE_SENT = "MESSAGE_SENT"
    MESSAGE_RECEIVED = "MESSAGE_RECEIVED"
    RESOURCE_CREATED = "RESOURCE_CREATED"
    RESOURCE_ACCESSED = "RESOURCE_ACCESSED"
    RESOURCE_MODIFIED = "RESOURCE_MODIFIED"
    RESOURCE_DELETED = "RESOURCE_DELETED"
    AUTH_SUCCESS = "AUTH_SUCCESS"
    AUTH_FAILED = "AUTH_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    BACKUP_SUCCESS = "BACKUP_SUCCESS"
    BACKUP_FAILED = "BACKUP_FAILED"
    RESTORE_SUCCESS = "RESTORE_SUCCESS"
    RESTORE_FAILED = "RESTORE_FAILED"
    ALERT_RAISED = "ALERT_RAISED"
    ALERT_RESOLVED = "ALERT_RESOLVED"
    PLUGIN_LOADED = "PLUGIN_LOADED"
    PLUGIN_FAILED = "PLUGIN_FAILED"

@dataclass
class AuditEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: AuditEventType
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    action: str
    resource_id: Optional[str] = None
    result: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        event_dict = asdict(self)
        event_dict["timestamp"] = self.timestamp.isoformat()
        event_dict["event_type"] = self.event_type.value
        return event_dict

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"

class AlertStatus(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"

@dataclass
class Metric:
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        metric_dict = asdict(self)
        metric_dict["timestamp"] = self.timestamp.isoformat()
        return metric_dict

@dataclass
class Alert:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: AlertStatus = AlertStatus.ACTIVE
    details: Dict[str, Any] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        alert_dict = asdict(self)
        alert_dict["timestamp"] = self.timestamp.isoformat()
        alert_dict["severity"] = self.severity.value
        alert_dict["status"] = self.status.value
        if self.resolved_at:
            alert_dict["resolved_at"] = self.resolved_at.isoformat()
        return alert_dict