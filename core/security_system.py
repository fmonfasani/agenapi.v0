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

from core.autonomous_agent_framework import BaseAgent, AgentMessage

# ================================
# SECURITY ENUMS AND MODELS
# ================================

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
    permissions: Set[Permission]
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    description: str = ""
    
@dataclass
class UserAccount:
    """Cuenta de usuario"""
    id: str
    username: str
    email: str
    password_hash: str
    roles: List[str] = field(default_factory=list)
    api_keys: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentCredentials:
    """Credenciales de agente"""
    agent_id: str
    agent_token: str
    security_level: SecurityLevel
    allowed_namespaces: List[str] = field(default_factory=list)
    permissions: Set[Permission] = field(default_factory=set)
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

# ================================
# AUTHENTICATION PROVIDERS
# ================================

class AuthenticationProvider(ABC):
    """Interfaz base para proveedores de autenticación"""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Autenticar usuario/agente"""
        pass
        
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validar token"""
        pass

class JWTAuthProvider(AuthenticationProvider):
    """Proveedor de autenticación JWT"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Autenticar con username/password"""
        username = credentials.get("username")
        password = credentials.get("password")
        
        # En implementación real, verificarías contra base de datos
        # Por ahora, simulamos usuarios válidos
        valid_users = {
            "admin": "admin_password_hash",
            "operator": "operator_password_hash"
        }
        
        if username in valid_users:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if password_hash == valid_users[username]:
                return await self._create_jwt_token({
                    "username": username,
                    "roles": ["admin"] if username == "admin" else ["operator"]
                })
                
        return None
        
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validar token JWT"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Verificar expiración
            if datetime.fromtimestamp(payload["exp"]) < datetime.now():
                return None
                
            return payload
            
        except (jwt.InvalidTokenError, KeyError):
            return None
            
    async def _create_jwt_token(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Crear token JWT"""
        # Añadir claims estándar
        payload.update({
            "iat": datetime.now(),
            "exp": datetime.now() + timedelta(hours=24),
            "iss": "agent_framework"
        })
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": 86400,  # 24 hours
            "user_info": {
                "username": payload["username"],
                "roles": payload["roles"]
            }
        }

class APIKeyAuthProvider(AuthenticationProvider):
    """Proveedor de autenticación por API Key"""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Autenticar con API key"""
        api_key = credentials.get("api_key")
        
        if api_key in self.api_keys:
            key_info = self.api_keys[api_key]
            
            # Verificar si la key está activa
            if key_info.get("is_active", True):
                return key_info
                
        return None
        
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validar API key como token"""
        return await self.authenticate({"api_key": token})
        
    def create_api_key(self, user_id: str, permissions: List[Permission], 
                      description: str = "") -> str:
        """Crear nueva API key"""
        api_key = secrets.token_urlsafe(32)
        
        self.api_keys[api_key] = {
            "user_id": user_id,
            "permissions": [p.value for p in permissions],
            "description": description,
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
        
        return api_key
        
    def revoke_api_key(self, api_key: str) -> bool:
        """Revocar API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]["is_active"] = False
            return True
        return False

# ================================
# AUTHORIZATION SYSTEM
# ================================

class AuthorizationManager:
    """Gestor de autorización y permisos"""
    
    def __init__(self):
        self.roles: Dict[str, SecurityRole] = {}
        self.user_permissions_cache: Dict[str, Set[Permission]] = {}
        self._setup_default_roles()
        
    def _setup_default_roles(self):
        """Configurar roles por defecto"""
        # Rol de administrador
        self.roles["admin"] = SecurityRole(
            name="admin",
            permissions={
                Permission.READ_AGENTS, Permission.WRITE_AGENTS, 
                Permission.CREATE_AGENTS, Permission.DELETE_AGENTS,
                Permission.EXECUTE_ACTIONS, Permission.READ_RESOURCES,
                Permission.WRITE_RESOURCES, Permission.READ_MESSAGES,
                Permission.ADMIN_ACCESS, Permission.MONITOR_SYSTEM
            },
            security_level=SecurityLevel.RESTRICTED,
            description="Full system administrator"
        )
        
        # Rol de operador
        self.roles["operator"] = SecurityRole(
            name="operator",
            permissions={
                Permission.READ_AGENTS, Permission.EXECUTE_ACTIONS,
                Permission.READ_RESOURCES, Permission.MONITOR_SYSTEM
            },
            security_level=SecurityLevel.INTERNAL,
            description="System operator with read and execute permissions"
        )
        
        # Rol de visor
        self.roles["viewer"] = SecurityRole(
            name="viewer",
            permissions={
                Permission.READ_AGENTS, Permission.READ_RESOURCES,
                Permission.MONITOR_SYSTEM
            },
            security_level=SecurityLevel.INTERNAL,
            description="Read-only access to system information"
        )
        
        # Rol de agente
        self.roles["agent"] = SecurityRole(
            name="agent",
            permissions={
                Permission.READ_AGENTS, Permission.EXECUTE_ACTIONS,
                Permission.READ_RESOURCES, Permission.WRITE_RESOURCES
            },
            security_level=SecurityLevel.CONFIDENTIAL,
            description="Agent-to-agent communication permissions"
        )
        
    def add_role(self, role: SecurityRole):
        """Añadir nuevo rol"""
        self.roles[role.name] = role
        
    def get_user_permissions(self, user_roles: List[str]) -> Set[Permission]:
        """Obtener permisos de usuario basado en roles"""
        if not user_roles:
            return set()
            
        # Usar caché si está disponible
        cache_key = ",".join(sorted(user_roles))
        if cache_key in self.user_permissions_cache:
            return self.user_permissions_cache[cache_key]
            
        # Calcular permisos
        permissions = set()
        for role_name in user_roles:
            if role_name in self.roles:
                permissions.update(self.roles[role_name].permissions)
                
        # Cachear resultado
        self.user_permissions_cache[cache_key] = permissions
        return permissions
        
    def check_permission(self, user_roles: List[str], required_permission: Permission) -> bool:
        """Verificar si el usuario tiene el permiso requerido"""
        user_permissions = self.get_user_permissions(user_roles)
        return required_permission in user_permissions
        
    def check_security_level(self, user_roles: List[str], required_level: SecurityLevel) -> bool:
        """Verificar si el usuario tiene el nivel de seguridad requerido"""
        user_level = SecurityLevel.PUBLIC
        
        for role_name in user_roles:
            if role_name in self.roles:
                role_level = self.roles[role_name].security_level
                # Tomar el nivel más alto
                if role_level.value in ["restricted", "confidential", "internal", "public"]:
                    levels = ["public", "internal", "confidential", "restricted"]
                    if levels.index(role_level.value) > levels.index(user_level.value):
                        user_level = role_level
                        
        # Verificar si el nivel del usuario es suficiente
        levels = ["public", "internal", "confidential", "restricted"]
        return levels.index(user_level.value) >= levels.index(required_level.value)

# ================================
# AUDIT SYSTEM
# ================================

@dataclass
class AuditEvent:
    """Evento de auditoría"""
    id: str
    timestamp: datetime
    user_id: Optional[str]
    agent_id: Optional[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    result: str  # success, failure, unauthorized
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class AuditLogger:
    """Sistema de auditoría y logging de seguridad"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = Path(log_file)
        self.events: List[AuditEvent] = []
        self.max_events_in_memory = 1000
        
    async def log_event(self, user_id: Optional[str], agent_id: Optional[str],
                       action: str, resource_type: str, resource_id: Optional[str],
                       result: str, details: Dict[str, Any] = None,
                       ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Registrar evento de auditoría"""
        
        event = AuditEvent(
            id=secrets.token_hex(16),
            timestamp=datetime.now(),
            user_id=user_id,
            agent_id=agent_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            result=result,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Añadir a memoria
        self.events.append(event)
        
        # Mantener límite en memoria
        if len(self.events) > self.max_events_in_memory:
            self.events.pop(0)
            
        # Escribir a archivo
        await self._write_to_file(event)
        
        # Log crítico para eventos importantes
        if result == "unauthorized" or "admin" in action.lower():
            logging.warning(f"Security event: {action} by {user_id or agent_id} - {result}")
            
    async def _write_to_file(self, event: AuditEvent):
        """Escribir evento a archivo de log"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                event_data = {
                    "id": event.id,
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": event.user_id,
                    "agent_id": event.agent_id,
                    "action": event.action,
                    "resource_type": event.resource_type,
                    "resource_id": event.resource_id,
                    "result": event.result,
                    "details": event.details,
                    "ip_address": event.ip_address,
                    "user_agent": event.user_agent
                }
                f.write(json.dumps(event_data) + "\n")
        except Exception as e:
            logging.error(f"Failed to write audit log: {e}")
            
    def get_events(self, filter_criteria: Dict[str, Any] = None, 
                  limit: int = 100) -> List[AuditEvent]:
        """Obtener eventos de auditoría con filtros"""
        filtered_events = self.events
        
        if filter_criteria:
            if "user_id" in filter_criteria:
                filtered_events = [e for e in filtered_events if e.user_id == filter_criteria["user_id"]]
            if "action" in filter_criteria:
                filtered_events = [e for e in filtered_events if filter_criteria["action"] in e.action]
            if "result" in filter_criteria:
                filtered_events = [e for e in filtered_events if e.result == filter_criteria["result"]]
                
        # Ordenar por timestamp descendente y limitar
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_events[:limit]

# ================================
# SECURE MESSAGE WRAPPER
# ================================

class SecureMessageWrapper:
    """Wrapper para mensajes seguros entre agentes"""
    
    def __init__(self, encryption_key: str):
        self.encryption_key = encryption_key
        
    def encrypt_message(self, message: AgentMessage) -> AgentMessage:
        """Encriptar mensaje (simulado)"""
        # En implementación real, usarías criptografía real
        encrypted_payload = self._simple_encrypt(json.dumps(message.payload))
        
        secure_message = AgentMessage(
            id=message.id,
            sender_id=message.sender_id,
            receiver_id=message.receiver_id,
            message_type=message.message_type,
            action=message.action,
            payload={"encrypted": True, "data": encrypted_payload},
            timestamp=message.timestamp,
            correlation_id=message.correlation_id,
            response_required=message.response_required
        )
        
        return secure_message
        
    def decrypt_message(self, encrypted_message: AgentMessage) -> AgentMessage:
        """Desencriptar mensaje (simulado)"""
        if not encrypted_message.payload.get("encrypted"):
            return encrypted_message
            
        decrypted_data = self._simple_decrypt(encrypted_message.payload["data"])
        original_payload = json.loads(decrypted_data)
        
        decrypted_message = AgentMessage(
            id=encrypted_message.id,
            sender_id=encrypted_message.sender_id,
            receiver_id=encrypted_message.receiver_id,
            message_type=encrypted_message.message_type,
            action=encrypted_message.action,
            payload=original_payload,
            timestamp=encrypted_message.timestamp,
            correlation_id=encrypted_message.correlation_id,
            response_required=encrypted_message.response_required
        )
        
        return decrypted_message
        
    def _simple_encrypt(self, data: str) -> str:
        """Encriptación simple (NO USAR EN PRODUCCIÓN)"""
        # Esto es solo para demo - usa una librería de criptografía real
        import base64
        return base64.b64encode(data.encode()).decode()
        
    def _simple_decrypt(self, encrypted_data: str) -> str:
        """Desencriptación simple (NO USAR EN PRODUCCIÓN)"""
        import base64
        return base64.b64decode(encrypted_data.encode()).decode()

# ================================
# SECURITY MANAGER
# ================================

class SecurityManager:
    """Gestor principal de seguridad del framework"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Inicializar componentes
        self.auth_providers: Dict[str, AuthenticationProvider] = {}
        self.authorization_manager = AuthorizationManager()
        self.audit_logger = AuditLogger(self.config.get("audit_log_file", "security_audit.log"))
        self.message_wrapper = SecureMessageWrapper(self.config.get("encryption_key", "default_key"))
        
        # Sesiones activas
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.agent_credentials: Dict[str, AgentCredentials] = {}
        
        # Configurar proveedores por defecto
        self._setup_default_providers()
        
    def _setup_default_providers(self):
        """Configurar proveedores de autenticación por defecto"""
        # JWT Provider
        jwt_secret = self.config.get("jwt_secret", "default_jwt_secret_change_in_production")
        self.auth_providers["jwt"] = JWTAuthProvider(jwt_secret)
        
        # API Key Provider
        self.auth_providers["api_key"] = APIKeyAuthProvider()
        
    async def authenticate_user(self, method: AuthenticationMethod, 
                              credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Autenticar usuario"""
        provider_name = method.value.split("_")[0]  # jwt, api, oauth2
        
        if provider_name in self.auth_providers:
            auth_result = await self.auth_providers[provider_name].authenticate(credentials)
            
            if auth_result:
                # Crear sesión
                session_id = secrets.token_urlsafe(32)
                self.active_sessions[session_id] = {
                    "user_info": auth_result.get("user_info", {}),
                    "created_at": datetime.now(),
                    "last_activity": datetime.now(),
                    "auth_method": method.value
                }
                
                # Log evento
                await self.audit_logger.log_event(
                    user_id=auth_result.get("user_info", {}).get("username"),
                    agent_id=None,
                    action="user_authentication",
                    resource_type="session",
                    resource_id=session_id,
                    result="success",
                    details={"auth_method": method.value}
                )
                
                auth_result["session_id"] = session_id
                return auth_result
                
        # Log fallo de autenticación
        await self.audit_logger.log_event(
            user_id=credentials.get("username"),
            agent_id=None,
            action="user_authentication",
            resource_type="session",
            resource_id=None,
            result="failure",
            details={"auth_method": method.value}
        )
        
        return None
        
    async def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validar sesión activa"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Verificar expiración (24 horas por defecto)
            max_age = timedelta(hours=self.config.get("session_max_hours", 24))
            if datetime.now() - session["created_at"] > max_age:
                del self.active_sessions[session_id]
                return None
                
            # Actualizar última actividad
            session["last_activity"] = datetime.now()
            return session
            
        return None
        
    async def authorize_action(self, session_id: str, required_permission: Permission,
                             resource_type: str = "general", 
                             security_level: SecurityLevel = SecurityLevel.INTERNAL) -> bool:
        """Autorizar acción del usuario"""
        session = await self.validate_session(session_id)
        if not session:
            return False
            
        user_info = session.get("user_info", {})
        user_roles = user_info.get("roles", [])
        
        # Verificar permiso
        has_permission = self.authorization_manager.check_permission(user_roles, required_permission)
        
        # Verificar nivel de seguridad
        has_security_level = self.authorization_manager.check_security_level(user_roles, security_level)
        
        result = has_permission and has_security_level
        
        # Log evento de autorización
        await self.audit_logger.log_event(
            user_id=user_info.get("username"),
            agent_id=None,
            action=f"authorize_{required_permission.value}",
            resource_type=resource_type,
            resource_id=None,
            result="success" if result else "unauthorized",
            details={
                "required_permission": required_permission.value,
                "security_level": security_level.value,
                "user_roles": user_roles
            }
        )
        
        return result
        
    async def register_agent_credentials(self, agent: BaseAgent, 
                                       security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL,
                                       permissions: Set[Permission] = None) -> str:
        """Registrar credenciales de agente"""
        agent_token = secrets.token_urlsafe(32)
        
        credentials = AgentCredentials(
            agent_id=agent.id,
            agent_token=agent_token,
            security_level=security_level,
            allowed_namespaces=[agent.namespace],
            permissions=permissions or {Permission.READ_AGENTS, Permission.EXECUTE_ACTIONS}
        )
        
        self.agent_credentials[agent.id] = credentials
        
        await self.audit_logger.log_event(
            user_id=None,
            agent_id=agent.id,
            action="agent_credentials_registered",
            resource_type="agent",
            resource_id=agent.id,
            result="success",
            details={"security_level": security_level.value}
        )
        
        return agent_token
        
    async def validate_agent_token(self, agent_id: str, token: str) -> bool:
        """Validar token de agente"""
        if agent_id in self.agent_credentials:
            credentials = self.agent_credentials[agent_id]
            
            # Verificar token
            if credentials.agent_token == token:
                # Verificar expiración si está configurada
                if credentials.expires_at and datetime.now() > credentials.expires_at:
                    return False
                    
                return True
                
        return False
        
    async def secure_agent_message(self, message: AgentMessage) -> AgentMessage:
        """Asegurar mensaje entre agentes"""
        # Verificar credenciales del remitente
        sender_creds = self.agent_credentials.get(message.sender_id)
        if not sender_creds:
            raise PermissionError(f"Agent {message.sender_id} not authenticated")
            
        # Encriptar si es necesario
        if sender_creds.security_level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED]:
            return self.message_wrapper.encrypt_message(message)
            
        return message
        
    async def logout_session(self, session_id: str) -> bool:
        """Cerrar sesión"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            user_info = session.get("user_info", {})
            
            del self.active_sessions[session_id]
            
            await self.audit_logger.log_event(
                user_id=user_info.get("username"),
                agent_id=None,
                action="user_logout",
                resource_type="session",
                resource_id=session_id,
                result="success"
            )
            
            return True
            
        return False
        
    def create_api_key(self, user_id: str, permissions: List[Permission],
                      description: str = "") -> str:
        """Crear API key para usuario"""
        if "api_key" in self.auth_providers:
            provider = self.auth_providers["api_key"]
            return provider.create_api_key(user_id, permissions, description)
        else:
            raise RuntimeError("API Key provider not available")
            
    def get_security_status(self) -> Dict[str, Any]:
        """Obtener estado de seguridad del sistema"""
        return {
            "active_sessions": len(self.active_sessions),
            "registered_agents": len(self.agent_credentials),
            "auth_providers": list(self.auth_providers.keys()),
            "audit_events_in_memory": len(self.audit_logger.events),
            "default_roles": list(self.authorization_manager.roles.keys())
        }

# ================================
# SECURITY DECORATORS
# ================================

def require_permission(permission: Permission, security_level: SecurityLevel = SecurityLevel.INTERNAL):
    """Decorador para requerir permisos específicos"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Asumiendo que el primer argumento contiene información de sesión
            if args and hasattr(args[0], 'security_manager'):
                security_manager = args[0].security_manager
                session_id = kwargs.get('session_id') or getattr(args[0], 'current_session_id', None)
                
                if session_id:
                    authorized = await security_manager.authorize_action(
                        session_id, permission, func.__name__, security_level
                    )
                    
                    if not authorized:
                        raise PermissionError(f"Insufficient permissions: {permission.value} required")
                        
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# ================================
# EXAMPLE USAGE
# ================================

async def security_demo():
    """Demo del sistema de seguridad"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🔒 Security System Demo")
    print("="*50)
    
    # Crear gestor de seguridad
    security_config = {
        "jwt_secret": "demo_secret_key_change_in_production",
        "encryption_key": "demo_encryption_key",
        "session_max_hours": 24,
        "audit_log_file": "demo_security_audit.log"
    }
    
    security_manager = SecurityManager(security_config)
    
    # Demo 1: Autenticación de usuario
    print("\n1. User Authentication Demo:")
    
    # Autenticación exitosa
    auth_result = await security_manager.authenticate_user(
        AuthenticationMethod.JWT_TOKEN,
        {"username": "admin", "password": "admin_password"}  # Password será hasheado
    )
    
    if auth_result:
        session_id = auth_result["session_id"]
        print(f"✅ Authentication successful")
        print(f"   Session ID: {session_id}")
        print(f"   User: {auth_result['user_info']['username']}")
        print(f"   Token: {auth_result['access_token'][:20]}...")
        
        # Demo 2: Autorización
        print("\n2. Authorization Demo:")
        
        # Verificar diferentes permisos
        permissions_to_test = [
            Permission.READ_AGENTS,
            Permission.DELETE_AGENTS, 
            Permission.ADMIN_ACCESS
        ]
        
        for perm in permissions_to_test:
            authorized = await security_manager.authorize_action(
                session_id, perm, "demo_resource"
            )
            status = "✅ AUTHORIZED" if authorized else "❌ DENIED"
            print(f"   {perm.value}: {status}")
            
    # Demo 3: API Key
    print("\n3. API Key Demo:")
    api_key = security_manager.create_api_key(
        "demo_user",
        [Permission.READ_AGENTS, Permission.MONITOR_SYSTEM],
        "Demo API key"
    )
    print(f"   Created API Key: {api_key}")
    
    # Validar API key
    api_auth = await security_manager.authenticate_user(
        AuthenticationMethod.API_KEY,
        {"api_key": api_key}
    )
    
    if api_auth:
        print("   ✅ API Key validation successful")
    
    # Demo 4: Agent Credentials
    print("\n4. Agent Security Demo:")
    
    # Crear agente mock
    class MockAgent:
        def __init__(self):
            self.id = "demo_agent_001"
            self.namespace = "agent.demo"
    
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
    print(f"   Token validation: {'✅ Valid' if token_valid else '❌ Invalid'}")
    
    # Demo 5: Auditoría
    print("\n5. Audit System Demo:")
    
    # Los eventos ya se han registrado automáticamente
    audit_events = security_manager.audit_logger.get_events(limit=5)
    
    print(f"   Recent audit events ({len(audit_events)}):")
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
        await security_manager.logout_session(session_id)
        print(f"\n🔓 Session logged out: {session_id}")
    
    print("\n✅ Security demo completed")

if __name__ == "__main__":
    asyncio.run(security_demo())