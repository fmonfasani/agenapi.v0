# agentapi/models/agent_models.py

import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

class AgentStatus(Enum):
    """Estados de los agentes"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"

class MessageType(Enum):
    """Tipos de mensajes entre agentes"""
    COMMAND = "command"
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    BROADCAST = "broadcast" # Mensaje para todos los agentes o un grupo

class ResourceType(Enum):
    """Tipos de recursos"""
    CODE = "code"
    INFRA = "infra"
    WORKFLOW = "workflow"
    UI = "ui"
    DATA = "data"
    TEST = "test"
    SECURITY = "security"
    RELEASE = "release"
    DOCUMENT = "document"
    CONFIG = "config"
    LOG = "log"

@dataclass
class AgentMessage:
    """Mensaje entre agentes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = "" # Puede ser un ID de agente o un "broadcast"
    message_type: MessageType = MessageType.COMMAND
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    trace_id: str = "" # Para seguimiento de conversaciones/solicitudes
    # Posibles campos adicionales para contexto de seguridad, firma, etc.

@dataclass
class AgentCapability:
    """Capacidad que un agente puede realizar"""
    name: str
    namespace: str # Ej: "agent.planning.strategist.define_strategy"
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict) # JSON Schema
    output_schema: Dict[str, Any] = field(default_factory=dict) # JSON Schema
    handler: Optional[Callable[..., Any]] = None # Función que maneja la capacidad
    parameters: List[Dict[str, Any]] = field(default_factory=list) # para descripción detallada de OpenAPI
    is_private: bool = False # Si la capacidad es interna (solo para otros agentes con permiso) o pública (expuesta vía API)
    requires_permission: Optional[str] = None # Permiso de seguridad necesario para ejecutar esta capacidad

@dataclass
class AgentResource:
    """Representa un recurso gestionado por el framework"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ResourceType
    name: str
    namespace: str # Ej: "code.backend.user_service"
    data: Any # Contenido real del recurso (ej. código, JSON de configuración)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    owner_agent_id: Optional[str] = None # Agente que creó o es responsable del recurso
    version: str = "1.0.0"
    checksum: Optional[str] = None # Hash para verificar integridad
    tags: Dict[str, str] = field(default_factory=dict)
    is_locked: bool = False # Indica si el recurso está actualmente en uso/bloqueado
    locked_by: Optional[str] = None # ID del agente que lo bloqueó
    lock_timestamp: Optional[datetime] = None