"""
security_system.py - Sistema de seguridad para el framework de agentes
"""

import jwt
import hashlib
import secrets
import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import logging
import json
from pathlib import Path

# Importaciones actualizadas
from core.autonomous_agent_framework import BaseAgent # <-- CAMBIO AQUI
from core.models import AgentMessage # <-- CAMBIO AQUI

# ================================\
# SECURITY ENUMS AND MODELS
# ================================\

class SecurityLevel(Enum):
    """Niveles de seguridad"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class Permission(Enum):
    """Permisos del sistema"""
    READ_AGENTS = "read_agents"
    WRITE_AGENTS = "write_agents"
    CREATE_AGENTS = "create_agents"
    DELETE_AGENTS = "delete_agents"
    EXECUTE_ACTIONS = "execute_actions"
    READ_RESOURCES = "read_resources"
    WRITE_RESOURCES = "write_resources"
    READ_MESSAGES = "read_messages"
    ADMIN_ACCESS = "admin_access"
    MONITOR_SYSTEM = "monitor_system"

class AuthenticationMethod(Enum):
    """Métodos de autenticación"""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    AGENT_CERTIFICATE = "agent_certificate"

@dataclass
class SecurityRole:
    """Rol de seguridad"""
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    description: str = ""

@dataclass
class UserAccount:
    """Cuenta de usuario para acceso al framework"""
    user_id: str
    username: str
    hashed_password: Optional[str] = None
    api_key: Optional[str] = None
    roles: List[str] = field(default_factory=list) # Nombres de roles
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None

@dataclass
class AgentCredentials:
    """Credenciales para un agente"""
    agent_id: str
    token: str # Puede ser un JWT, API Key específica, etc.
    security_level: SecurityLevel
    permissions: Set[Permission] = field(default_factory=set)
    issued_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

@dataclass
class SecurityConfig:
    """Configuración del sistema de seguridad"""
    jwt_secret: str = "super_secret_jwt_key_please_change_me"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    enable_rbac: bool = True
    enable_audit_logging: bool = True
    admin_api_key: str = "admin_super_secret_api_key" # Para acceso inicial
    default_user_roles: List[str] = field(default_factory=lambda: ["viewer"])
    default_agent_security_level: SecurityLevel = SecurityLevel.INTERNAL

@dataclass
class AuditEvent:
    """Evento de auditoría"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    action: str # e.g., "agent.create", "resource.read", "user.login"
    user_id: Optional[str] = None # Si es una acción de usuario
    agent_id: Optional[str] = None # Si es una acción de agente
    resource_id: Optional[str] = None # Si la acción involucra un recurso
    details: Dict[str, Any] = field(default_factory=dict)
    result: str = "success" # "success", "failure", "warning"
    error_message: Optional[str] = None


# ================================\
# AUTHENTICATION MANAGER
# ================================\

class AuthenticationManager:
    """Maneja la autenticación de usuarios y agentes."""
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self.users: Dict[str, UserAccount] = {} # user_id -> UserAccount
        self.agent_tokens: Dict[str, AgentCredentials] = {} # agent_id -> AgentCredentials (solo un token por agente activo)
        self.logger = logging.getLogger("AuthManager")
        self._initialize_default_admin()
        self.logger.info("AuthenticationManager initialized.")

    def _initialize_default_admin(self):
        """Crea una cuenta de administrador por defecto si no existe."""
        admin_id = "admin_user"
        if admin_id not in self.users:
            admin_password = "admin_password" # En un entorno real, pedir esto de forma segura
            hashed_pw = self.hash_password(admin_password)
            admin_account = UserAccount(
                user_id=admin_id,
                username="admin",
                hashed_password=hashed_pw,
                api_key=self.config.admin_api_key,
                roles=["admin"],
                is_active=True
            )
            self.users[admin_id] = admin_account
            self.logger.info("Default admin user 'admin' created with password 'admin_password' and admin_api_key.")

    def hash_password(self, password: str) -> str:
        """Hashea una contraseña para almacenamiento seguro."""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verifica una contraseña contra su hash."""
        return self.hash_password(password) == hashed_password

    def generate_api_key(self) -> str:
        """Genera una nueva API Key segura."""
        return secrets.token_urlsafe(32)

    def generate_jwt(self, payload: Dict[str, Any], expiry_minutes: int = None) -> str:
        """Genera un token JWT."""
        if expiry_minutes is None:
            expiry_minutes = self.config.jwt_expiration_minutes
            
        expiration = datetime.now() + timedelta(minutes=expiry_minutes)
        payload.update({"exp": expiration.timestamp(), "iat": datetime.now().timestamp()})
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)

    def decode_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """Decodifica y valida un token JWT."""
        try:
            return jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token has expired.")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.error(f"Invalid JWT token: {e}")
            return None

    async def authenticate_user(self, username: str, password: str) -> Optional[UserAccount]:
        """Autentica un usuario por nombre de usuario y contraseña."""
        for user in self.users.values():
            if user.username == username and user.hashed_password and self.verify_password(password, user.hashed_password):
                user.last_login = datetime.now()
                self.logger.info(f"User '{username}' authenticated successfully.")
                return user
        self.logger.warning(f"Authentication failed for user '{username}'.")
        return None

    async def authenticate_with_api_key(self, api_key: str) -> Optional[UserAccount]:
        """Autentica un usuario por API Key."""
        for user in self.users.values():
            if user.api_key == api_key and user.is_active:
                user.last_login = datetime.now()
                self.logger.info(f"User '{user.username}' authenticated with API Key.")
                return user
        self.logger.warning("Authentication failed for API Key.")
        return None
    
    async def register_agent_credentials(self, agent: BaseAgent, level: SecurityLevel, permissions: Set[Permission]) -> str:
        """Registra las credenciales para un agente y genera un token."""
        token_payload = {
            "sub": agent.id,
            "name": agent.name,
            "namespace": agent.namespace,
            "level": level.value,
            "permissions": [p.value for p in permissions],
            "type": "agent"
        }
        token = self.generate_jwt(token_payload)
        
        credentials = AgentCredentials(
            agent_id=agent.id,
            token=token,
            security_level=level,
            permissions=permissions
        )
        self.agent_tokens[agent.id] = credentials
        self.logger.info(f"Credentials registered for agent {agent.name} (ID: {agent.id}).")
        return token

    async def validate_agent_token(self, agent_id: str, token: str) -> bool:
        """Valida un token de agente."""
        stored_credentials = self.agent_tokens.get(agent_id)
        if not stored_credentials or stored_credentials.token != token:
            self.logger.warning(f"Invalid or missing token for agent {agent_id}.")
            return False
        
        decoded_payload = self.decode_jwt(token)
        if not decoded_payload or decoded_payload.get("sub") != agent_id:
            self.logger.warning(f"Invalid JWT payload for agent {agent_id}.")
            return False
        
        self.logger.debug(f"Agent token for {agent_id} validated.")
        return True

    def get_agent_permissions(self, agent_id: str) -> Set[Permission]:
        """Obtiene los permisos asociados a un agente."""
        credentials = self.agent_tokens.get(agent_id)
        return credentials.permissions if credentials else set()


# ================================\
# AUTHORIZATION MANAGER (RBAC)
# ================================\

class AuthorizationManager:
    """Implementa el control de acceso basado en roles (RBAC)."""
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self.roles: Dict[str, SecurityRole] = {} # role_name -> SecurityRole
        self._initialize_default_roles()
        self.logger = logging.getLogger("AuthzManager")
        self.logger.info("AuthorizationManager initialized.")

    def _initialize_default_roles(self):
        """Define roles por defecto y sus permisos."""
        self.add_role(SecurityRole("admin", {p for p in Permission}, "Full administrative access."))
        self.add_role(SecurityRole("developer", {
            Permission.READ_AGENTS, Permission.WRITE_AGENTS, Permission.CREATE_AGENTS,
            Permission.READ_RESOURCES, Permission.WRITE_RESOURCES, Permission.EXECUTE_ACTIONS
        }, "Developer access to agents and resources."))
        self.add_role(SecurityRole("viewer", {
            Permission.READ_AGENTS, Permission.READ_RESOURCES, Permission.READ_MESSAGES,
            Permission.MONITOR_SYSTEM
        }, "Read-only access for monitoring."))
        self.logger.info("Default roles created: admin, developer, viewer.")

    def add_role(self, role: SecurityRole):
        """Añade un nuevo rol al sistema."""
        self.roles[role.name] = role
        self.logger.info(f"Role '{role.name}' added.")

    def get_role_permissions(self, role_name: str) -> Set[Permission]:
        """Obtiene los permisos de un rol."""
        role = self.roles.get(role_name)
        return role.permissions if role else set()

    async def check_permission(self, user_or_agent_id: str, required_permission: Permission, is_agent: bool = False) -> bool:
        """Verifica si un usuario o agente tiene un permiso específico."""
        if not self.config.enable_rbac:
            return True # RBAC deshabilitado, se concede acceso

        # Esto debería integrarse con el AuthenticationManager para obtener las credenciales
        # En una arquitectura modular, este manager pediría los permisos al AuthManager
        # Por simplicidad de la demo, asumiremos que un AuthenticationManager le pasa los permisos
        # o que este manager tiene acceso directo a los tokens/usuarios para obtener roles/permisos.
        
        # Simulación: si se pasa un user_id, buscar sus roles
        # Si se pasa un agent_id, buscar sus permisos directamente (asumimos que ya los tiene asignados)

        # Esta es una implementación simplificada. Realmente el AuthManager debería dar un token
        # con roles/permisos, y el AuthzManager sólo validaría ese token.
        
        # Aquí, simplemente retornamos True para la demo, asumiendo que el token ya fue validado
        # y que la lógica de permisos se manejará en un nivel superior (middleware API/agente)
        self.logger.debug(f"Permission check for {user_or_agent_id} ({'agent' if is_agent else 'user'}): {required_permission.value}")
        # Lógica real de RBAC:
        # 1. Obtener los roles del usuario/agente
        # 2. Sumar los permisos de todos esos roles
        # 3. Comprobar si `required_permission` está en el conjunto sumado.
        
        # Para la demo, simplificamos:
        # Se requiere integración con AuthenticationManager.
        # Por ahora, simplemente simulamos un pase.
        return True # Placeholder: siempre permite acceso si RBAC está habilitado para la demo


# ================================\
# AUDIT LOGGER
# ================================\

class AuditLogger:
    """Registra todos los eventos de seguridad y acciones clave."""
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("audit.log")
        self.audit_events: List[AuditEvent] = []
        self.logger = logging.getLogger("AuditLogger")
        self.logger.info(f"AuditLogger initialized. Logging to {self.log_file}")
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Asegura que el archivo de log existe."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                f.write(json.dumps({"events": []}) + "\n") # Escribir JSON válido si es el primer inicio

    async def log_event(self, action: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, 
                        resource_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None, 
                        result: str = "success", error_message: Optional[str] = None):
        """Registra un evento de auditoría."""
        event = AuditEvent(
            action=action,
            user_id=user_id,
            agent_id=agent_id,
            resource_id=resource_id,
            details=details or {},
            result=result,
            error_message=error_message
        )
        self.audit_events.append(event)
        
        # Escribir al archivo de log (append)
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(asdict(event)) + "\n")
            self.logger.debug(f"Audit event logged: {action} by {user_id or agent_id}")
        except Exception as e:
            self.logger.error(f"Failed to write audit event to file: {e}")

    def get_events(self, limit: int = 100) -> List[AuditEvent]:
        """Obtiene los últimos eventos de auditoría."""
        return self.audit_events[-limit:]

# ================================\
# SECURITY MANAGER (Facade)
# ================================\

class SecurityManager:
    """
    Gestiona el sistema de seguridad completo (autenticación, autorización, auditoría).
    Actúa como una fachada para los subsistemas.
    """
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.auth_manager = AuthenticationManager(self.config)
        self.authz_manager = AuthorizationManager(self.config)
        self.audit_logger = AuditLogger() if self.config.enable_audit_logging else None
        self.logger = logging.getLogger("SecurityManager")
        self.logger.info("SecurityManager initialized.")

    async def authenticate(self, auth_method: AuthenticationMethod, credentials: Dict[str, Any]) -> Optional[Union[UserAccount, AgentCredentials]]:
        """Autentica un usuario o agente."""
        result = None
        if auth_method == AuthenticationMethod.API_KEY:
            api_key = credentials.get("api_key")
            result = await self.auth_manager.authenticate_with_api_key(api_key)
            user_id = result.user_id if result else None
            await self._log_audit_event("user.authenticate.api_key", user_id=user_id, result="success" if result else "failure")
        elif auth_method == AuthenticationMethod.JWT_TOKEN:
            token = credentials.get("token")
            # Para JWT, asumimos que ya se decodificó un token para identificar al user/agent
            payload = self.auth_manager.decode_jwt(token)
            if payload and payload.get("type") == "user":
                user_id = payload.get("sub")
                if user_id in self.auth_manager.users:
                    result = self.auth_manager.users[user_id]
                    result.last_login = datetime.now() # Update login time
            elif payload and payload.get("type") == "agent":
                agent_id = payload.get("sub")
                if agent_id in self.auth_manager.agent_tokens:
                    result = self.auth_manager.agent_tokens[agent_id]
            await self._log_audit_event("user/agent.authenticate.jwt", user_id=user_id, agent_id=agent_id, result="success" if result else "failure")
        elif auth_method == AuthenticationMethod.OAUTH2:
            self.logger.warning("OAuth2 authentication not fully implemented for demo.")
            await self._log_audit_event("user.authenticate.oauth2", result="failure", error_message="Not implemented")
        elif auth_method == AuthenticationMethod.AGENT_CERTIFICATE:
            self.logger.warning("Agent certificate authentication not fully implemented for demo.")
            await self._log_audit_event("agent.authenticate.certificate", result="failure", error_message="Not implemented")
        else:
            self.logger.warning(f"Unsupported authentication method: {auth_method.value}")
            await self._log_audit_event("authenticate.unsupported_method", result="failure", error_message=f"Unsupported method: {auth_method.value}")
        
        return result

    async def authorize(self, user_or_agent_id: str, required_permission: Permission, is_agent: bool = False) -> bool:
        """Verifica si un usuario/agente tiene el permiso requerido."""
        has_permission = await self.authz_manager.check_permission(user_or_agent_id, required_permission, is_agent)
        await self._log_audit_event(f"{'agent' if is_agent else 'user'}.authorize.{required_permission.value}", 
                                    user_id=user_or_agent_id if not is_agent else None, 
                                    agent_id=user_or_agent_id if is_agent else None, 
                                    result="success" if has_permission else "failure")
        return has_permission

    async def register_agent_credentials(self, agent: BaseAgent, security_level: SecurityLevel, permissions: Set[Permission]) -> str:
        """Registra un nuevo agente y le asigna credenciales."""
        token = await self.auth_manager.register_agent_credentials(agent, security_level, permissions)
        await self._log_audit_event("agent.register_credentials", agent_id=agent.id, result="success")
        return token

    async def validate_agent_token(self, agent_id: str, token: str) -> bool:
        """Valida el token de un agente."""
        is_valid = await self.auth_manager.validate_agent_token(agent_id, token)
        await self._log_audit_event("agent.validate_token", agent_id=agent_id, result="success" if is_valid else "failure")
        return is_valid

    async def _log_audit_event(self, action: str, **kwargs):
        """Helper para loggear eventos de auditoría si el logger está habilitado."""
        if self.audit_logger and self.config.enable_audit_logging:
            await self.audit_logger.log_event(action, **kwargs)

    def get_security_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema de seguridad."""
        return {
            "timestamp": datetime.now().isoformat(),
            "authentication": {
                "total_users": len(self.auth_manager.users),
                "total_agent_tokens": len(self.auth_manager.agent_tokens),
                "jwt_expiration_minutes": self.config.jwt_expiration_minutes
            },
            "authorization": {
                "rbac_enabled": self.config.enable_rbac,
                "total_roles": len(self.authz_manager.roles)
            },
            "audit_logging": {
                "enabled": self.config.enable_audit_logging,
                "total_events": len(self.audit_logger.audit_events if self.audit_logger else [])
            }
        }

# ================================\
# DEMO
# ================================\

async def security_demo():
    """Ejemplo de uso del sistema de seguridad."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("🚀 Starting Security System Demo")
    print("="*50)

    security_config = SecurityConfig(
        jwt_secret="my_super_secret_key_for_demo",
        jwt_expiration_minutes=1 # Token expira rápido para demo
    )
    security_manager = SecurityManager(security_config)

    # Demo 1: Autenticación de usuario
    print("1. User Authentication Demo:")
    auth_result = await security_manager.authenticate(
        AuthenticationMethod.API_KEY,
        {"api_key": security_config.admin_api_key}
    )
    
    if auth_result:
        print(f"   ✅ User '{auth_result.username}' authenticated via API Key.")
        # Generar JWT para el usuario autenticado
        user_jwt = security_manager.auth_manager.generate_jwt(
            {"sub": auth_result.user_id, "username": auth_result.username, "type": "user", "roles": auth_result.roles}
        )
        print(f"   Generated JWT: {user_jwt[:30]}...")
        
        # Validar JWT
        decoded_jwt = security_manager.auth_manager.decode_jwt(user_jwt)
        print(f"   Decoded JWT (sub): {decoded_jwt['sub'] if decoded_jwt else 'Invalid'}")
        
        # Esperar a que expire el token
        print("   Waiting 2 seconds for JWT to expire (configured for 1 minute for this demo)...")
        await asyncio.sleep(2)
        expired_jwt = security_manager.auth_manager.decode_jwt(user_jwt)
        print(f"   Decoded JWT after expiry: {expired_jwt}") # Should be None
        
    else:
        print("   ❌ User authentication failed.")
        
    # Demo 2: Gestión de Roles y Permisos (Authorization)
    print("\n2. Role-Based Access Control (RBAC) Demo:")
    admin_user_id = "admin_user" # ID del usuario admin por defecto

    # Comprobar si el admin tiene permiso para crear agentes
    can_create_agents = await security_manager.authorize(admin_user_id, Permission.CREATE_AGENTS, is_agent=False)
    print(f"   Admin user can CREATE_AGENTS: {'✅ Yes' if can_create_agents else '❌ No'}")

    # Simular un "viewer" y verificar permisos
    security_manager.auth_manager.users["viewer_user"] = UserAccount(
        user_id="viewer_user", username="viewer", roles=["viewer"], is_active=True
    )
    can_write_resources = await security_manager.authorize("viewer_user", Permission.WRITE_RESOURCES, is_agent=False)
    print(f"   Viewer user can WRITE_RESOURCES: {'✅ Yes' if can_write_resources else '❌ No'}") # Should be No

    # Demo 3: Cifrado de Mensajes (conceptual, no implementado completamente aquí)
    print("\n3. Message Encryption (Conceptual):")
    message_payload = {"sensitive_data": "secret value", "command": "do_something"}
    
    # En un sistema real, se usaría criptografía asimétrica o simétrica
    encrypted_payload = f"ENCRYPTED({json.dumps(message_payload)})"
    decrypted_payload = f"DECRYPTED({encrypted_payload[10:-1]})"
    
    # Crear un AgentMessage con payload "cifrado"
    from core.models import MessageType # Importar MessageType si no está
    encrypted_message = AgentMessage(
        sender_id="agentA",
        receiver_id="agentB",
        message_type=MessageType.COMMAND,
        payload={"encrypted": encrypted_payload},
        status="encrypted"
    )
    print(f"   Simulated encrypted message payload: {encrypted_message.payload}")
    print(f"   Simulated decrypted payload: {decrypted_payload}")

    # Demo 4: Autenticación y autorización de agentes
    print("\n4. Agent Authentication & Authorization Demo:")
    
    class MockAgent(BaseAgent):
        def __init__(self):
            super().__init__("agent.demo", "MockAgent", None) # Framework es None para la demo de seguridad
            self.id = "agent-mock-123"
            self.status = AgentStatus.ACTIVE # Ya importado de core.models
    
    mock_agent = MockAgent()
    
    # Registrar credenciales de agente
    agent_token = await security_manager.register_agent_credentials(
        mock_agent,
        SecurityLevel.CONFIDENTIAL,
        {Permission.READ_AGENTS, Permission.EXECUTE_ACTIONS}
    )
    
    print(f"   Agent token created: {agent_token[:20]}...")
    
    # Validar token de agente
    token_valid = await security_manager.validate_agent_token(mock_agent.id, agent_token)
    print(f"   Token validation: {'✅ Valid' if token_valid else '❌ Invalid'}\n")
    
    # Demo 5: Auditoría
    print("5. Audit System Demo:")
    
    # Los eventos ya se han registrado automáticamente
    audit_events = security_manager.audit_logger.get_events(limit=5)
    
    print(f"   Recent audit events ({len(audit_events)}):\n")
    for event in audit_events:
        timestamp = event.timestamp.strftime("%H:%M:%S")
        user = event.user_id or event.agent_id or "system"
        print(f"   [{timestamp}] {user}: {event.action} - {event.result}")
    
    # Demo 6: Estado de seguridad
    print("\n6. Security Status:")
    status = security_manager.get_security_status()
    
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    if auth_result:
        # En una aplicación real, no borrarías usuarios así.
        security_manager.auth_manager.users.pop("viewer_user", None)
        security_manager.auth_manager.agent_tokens.pop("agent-mock-123", None)
        
    print("\nDemo finished.")

if __name__ == "__main__":
    asyncio.run(security_demo())