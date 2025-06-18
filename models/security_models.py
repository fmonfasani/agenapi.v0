# agentapi/models/security_models.py

import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class SecurityLevel(Enum):
    """Niveles de seguridad para recursos o interacciones."""
    PUBLIC = "public" # Acceso general
    INTERNAL = "internal" # Acceso solo para componentes internos del framework
    CONFIDENTIAL = "confidential" # Acceso restringido a roles/agentes específicos
    RESTRICTED = "restricted" # Acceso muy limitado, solo administradores o agentes autorizados

class Permission(Enum):
    """Permisos granulares dentro del sistema."""
    READ_AGENTS = "read_agents"
    WRITE_AGENTS = "write_agents"
    CREATE_AGENTS = "create_agents"
    DELETE_AGENTS = "delete_agents"
    EXECUTE_ACTIONS = "execute_actions" # Permiso para que un agente ejecute acciones en otro
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
    """Métodos de autenticación soportados."""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    AGENT_CERTIFICATE = "agent_certificate"
    USER_PASSWORD = "user_password" # Para autenticación de usuarios CLI/Web

@dataclass
class SecurityRole:
    """Define un rol de seguridad con un conjunto de permisos."""
    name: str
    permissions: Set[Permission] = field(default_factory=set) # Conjunto de objetos Permission

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions or Permission.ADMIN_ACCESS in self.permissions

@dataclass
class UserRole(SecurityRole):
    """Rol específico para usuarios humanos."""
    pass

@dataclass
class AgentRole(SecurityRole):
    """Rol específico para agentes autónomos."""
    pass

@dataclass
class SecurityToken:
    """Representa un token de seguridad generado (ej. JWT)."""
    token: str
    token_type: AuthenticationMethod
    expiry_time: datetime
    issued_at: datetime = field(default_factory=datetime.now)
    subject_id: str # ID del usuario o agente al que pertenece el token
    claims: Dict[str, Any] = field(default_factory=dict) # Permisos o roles codificados en el token

    def is_expired(self) -> bool:
        return datetime.now() > self.expiry_time

@dataclass
class User:
    """Modelo básico para un usuario humano del sistema."""
    id: str
    username: str
    hashed_password: str
    roles: List[str] = field(default_factory=list) # Nombres de roles
    api_keys: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None

@dataclass
class AgentAuthenticationEntry:
    """Entrada de autenticación para un agente."""
    agent_id: str
    token_hash: str # Hash del token o secreto del agente
    security_level: SecurityLevel
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None