"""
core/models.py - Definiciones de modelos de datos fundamentales para el framework.
Contiene enumeraciones y dataclasses compartidas por múltiples componentes.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

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

class ResourceType(Enum):
    """Tipos de recursos gestionados por el framework."""
    CODE = "code"
    INFRA = "infra"
    WORKFLOW = "workflow"
    UI = "ui"
    DATA = "data"
    TEST = "test"
    SECURITY = "security"
    RELEASE = "release"

@dataclass
class AgentMessage:
    """Mensaje entre agentes."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.COMMAND
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None # Para correlacionar requests/responses
    status: str = "sent" # e.g., "sent", "delivered", "read", "failed"
    error: Optional[str] = None

@dataclass
class AgentResource:
    """Representa un recurso gestionado por los agentes."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ResourceType = ResourceType.DATA
    name: str = ""
    namespace: str = "" # e.g., "resource.database.user_data"
    version: str = "1.0.0"
    data: Any = None # Contenido real del recurso (e.g., código, datos JSON)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    owner_agent_id: Optional[str] = None
    access_permissions: List[Dict[str, Any]] = field(default_factory=list) # e.g., [{"agent_id": "xyz", "permission": "read"}]
    checksum: Optional[str] = None # Para integridad del recurso

@dataclass
class AgentCapability:
    """Define una capacidad (acción) que un agente puede realizar."""
    name: str # Nombre de la acción (ej. "create.workflow", "generate.code")
    namespace: str # Namespace completo de la capacidad (ej. "agent.design.workflow.create")
    description: str
    input_schema: Dict[str, Any] # Esquema JSON para la validación de entrada
    output_schema: Dict[str, Any] # Esquema JSON para la validación de salida
    handler: Callable[[Dict[str, Any]], Any] # Función asíncrona que implementa la lógica

@dataclass
class AgentInfo:
    """Información básica de un agente para el registro."""
    id: str
    name: str
    namespace: str
    status: AgentStatus
    last_heartbeat: datetime = field(default_factory=datetime.now)
    # Considerar añadir una referencia débil al agente real aquí si es necesario
    # agent_ref: Optional[weakref.ReferenceType['BaseAgent']] = None