# agentapi/models/agent_models.py
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

class AgentStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"

class MessageType(Enum):
    COMMAND = "command"
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    BROADCAST = "broadcast"

class ResourceType(Enum):
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
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.COMMAND
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    trace_id: str = ""
    status: str = "sent"
    error: Optional[str] = None

@dataclass
class AgentResource:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ResourceType = ResourceType.DATA
    name: str = ""
    namespace: str = ""
    version: str = "1.0.0"
    data: Any = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    owner_agent_id: Optional[str] = None
    access_permissions: List[Dict[str, Any]] = field(default_factory=list)
    checksum: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    is_locked: bool = False
    locked_by: Optional[str] = None
    lock_timestamp: Optional[datetime] = None

@dataclass
class AgentCapability:
    name: str
    namespace: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable[..., Any]] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    is_private: bool = False
    requires_permission: Optional[str] = None

@dataclass
class AgentInfo:
    id: str
    name: str
    namespace: str
    status: AgentStatus
    last_heartbeat: datetime = field(default_factory=datetime.now)