# agentapi/models/security_models.py
import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class SecurityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class Permission(Enum):
    READ_AGENTS = "read_agents"
    WRITE_AGENTS = "write_agents"
    CREATE_AGENTS = "create_agents"
    DELETE_AGENTS = "delete_agents"
    EXECUTE_ACTIONS = "execute_actions"
    READ_RESOURCES = "read_resources"
    WRITE_RESOURCES = "write_resources"
    DELETE_RESOURCES = "delete_resources"
    READ_MESSAGES = "read_messages"
    SEND_MESSAGES = "send_messages"
    ADMIN_ACCESS = "admin_access"
    MONITOR_SYSTEM = "monitor_system"
    DEPLOY_SYSTEM = "deploy_system"
    MANAGE_PLUGINS = "manage_plugins"
    MANAGE_BACKUPS = "manage_backups"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"

class AuthenticationMethod(Enum):
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    AGENT_CERTIFICATE = "agent_certificate"
    USER_PASSWORD = "user_password"

@dataclass
class SecurityRole:
    name: str
    permissions: Set[Permission] = field(default_factory=set)

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions or Permission.ADMIN_ACCESS in self.permissions

@dataclass
class UserRole(SecurityRole):
    pass

@dataclass
class AgentRole(SecurityRole):
    pass

@dataclass
class SecurityToken:
    token: str
    token_type: AuthenticationMethod
    expiry_time: datetime
    issued_at: datetime = field(default_factory=datetime.now)
    subject_id: str
    claims: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        return datetime.now() > self.expiry_time

@dataclass
class User:
    id: str
    username: str
    hashed_password: str
    roles: List[str] = field(default_factory=list)
    api_keys: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None

@dataclass
class AgentAuthenticationEntry:
    agent_id: str
    token_hash: str
    security_level: SecurityLevel
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None