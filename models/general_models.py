# agentapi/models/general_models.py

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
from datetime import datetime
from enum import Enum

# ================================\
# GENERAL MODELS FOR API RESPONSES, ERRORS, ETC.
# ================================

@dataclass
class APIResponse:
    """Respuesta estándar de la API"""
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


@dataclass
class ErrorResponse:
    """Modelo para respuestas de error"""
    message: str
    code: int
    details: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"error": {"message": self.message, "code": self.code, "details": self.details}}

# Modelos para Auditoría (podrían ir en security_system.py, pero si es global, aquí)
class AuditEventType(Enum):
    """Tipos de eventos de auditoría"""
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
    """Representa un evento auditable en el sistema"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: AuditEventType
    user_id: Optional[str] = None # Para usuarios externos (CLI/API)
    agent_id: Optional[str] = None # Para acciones de agentes
    action: str # Descripción corta de la acción (e.g., "login", "create_agent", "execute_capability")
    resource_id: Optional[str] = None # ID del recurso afectado (ej. ID de agente, ID de recurso)
    result: str # "SUCCESS" o "FAILED"
    details: Dict[str, Any] = field(default_factory=dict) # Detalles adicionales en JSON

    def to_dict(self) -> Dict[str, Any]:
        event_dict = asdict(self)
        event_dict["timestamp"] = self.timestamp.isoformat()
        event_dict["event_type"] = self.event_type.value
        return event_dict

# Modelos para Monitoreo y Alertas (podrían ir en monitoring_system.py, pero si son genéricos, aquí)
class AlertSeverity(Enum):
    """Severidad de alertas"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"

class AlertStatus(Enum):
    """Estado de alertas"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"

@dataclass
class Metric:
    """Métrica del sistema"""
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
    """Alerta del sistema"""
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