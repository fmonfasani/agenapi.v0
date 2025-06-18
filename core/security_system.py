"""
security_system.py - Sistema de seguridad refactorizado para el framework de agentes
"""

import jwt
import hashlib
import secrets
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Type, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

# Importar desde el framework principal (asumiendo que está disponible)
# from core.autonomous_agent_framework import BaseAgent, AgentMessage # Descomentar si estas clases existen

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================
# EXCEPCIONES PERSONALIZADAS
# ================================

class SecurityError(Exception):
    """Excepción base para errores de seguridad."""
    pass

class AuthenticationError(SecurityError):
    """Excepción para fallos de autenticación."""
    pass

class AuthorizationError(SecurityError):
    """Excepción para fallos de autorización (permisos insuficientes)."""
    pass

class InvalidTokenError(SecurityError):
    """Excepción para tokens inválidos o expirados."""
    pass

# ================================
# SEGURIDAD - ENUMS Y MODELOS DE DATOS
# ================================

class SecurityLevel(Enum):
    """Niveles de seguridad para recursos y acciones."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class Permission(Enum):
    """Permisos del sistema para granularidad de acceso."""
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
    """Métodos de autenticación soportados."""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    AGENT_CERTIFICATE = "agent_certificate"

@dataclass
class SecurityRole:
    """Define un rol de seguridad con un conjunto de permisos y un nivel de seguridad asociado."""
    name: str
    permissions: Set[Permission]
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    description: str = ""

@dataclass
class UserAccount:
    """Representa una cuenta de usuario en el sistema."""
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
    """Almacena las credenciales y permisos de un agente."""
    agent_id: str
    agent_token: str
    security_level: SecurityLevel
    allowed_namespaces: List[str] = field(default_factory=list)
    permissions: Set[Permission] = field(default_factory=set)
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AuthResult:
    """Resultado de una operación de autenticación o validación de token."""
    success: bool
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: Set[Permission] = field(default_factory=set)
    access_token: Optional[str] = None
    token_type: Optional[str] = None
    expires_in: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

@dataclass
class UserSession:
    """Representa una sesión de usuario activa."""
    session_id: str
    user_id: str
    username: str
    roles: List[str]
    created_at: datetime
    last_activity: datetime
    auth_method: str
    permissions: Set[Permission] = field(default_factory=set) # Permisos calculados al inicio de la sesión

@dataclass
class SecurityConfig:
    """Clase de configuración para el sistema de seguridad."""
    jwt_secret: str = "default_jwt_secret_change_in_production"
    encryption_key: str = "default_encryption_key"
    session_max_hours: int = 24
    audit_log_file: str = "security_audit.log"
    # Otras configuraciones de seguridad pueden ir aquí (ej. password policies)

# ================================
# PROVEEDORES DE AUTENTICACIÓN
# ================================

class AuthenticationProvider(ABC):
    """Interfaz base para proveedores de autenticación."""

    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """
        Autentica un usuario/agente con las credenciales proporcionadas.
        Retorna un AuthResult indicando éxito o fallo.
        """
        pass

    @abstractmethod
    async def validate_token(self, token: str) -> AuthResult:
        """
        Valida un token existente.
        Retorna un AuthResult indicando éxito o fallo y los datos del token.
        """
        pass

class JWTAuthProvider(AuthenticationProvider):
    """Proveedor de autenticación JWT."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        logger.info(f"JWTAuthProvider inicializado con algoritmo: {self.algorithm}")

    async def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Autenticar con username/password y generar JWT."""
        username = credentials.get("username")
        password = credentials.get("password")

        # En implementación real, verificarías contra base de datos de usuarios
        # Por ahora, simulamos usuarios válidos y un hash de contraseña simple
        valid_users = {
            "admin": hashlib.sha256("admin_password".encode()).hexdigest(),
            "operator": hashlib.sha256("operator_password".encode()).hemdigest()
        }

        if username in valid_users:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if password_hash == valid_users[username]:
                roles = ["admin"] if username == "admin" else ["operator"]
                return await self._create_jwt_token(username, roles)
        logger.warning(f"Intento de autenticación JWT fallido para usuario: {username}")
        return AuthResult(success=False, error_message="Credenciales inválidas.")

    async def validate_token(self, token: str) -> AuthResult:
        """Valida un token JWT."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            if datetime.fromtimestamp(payload["exp"]) < datetime.now():
                logger.warning("Token JWT expirado.")
                raise InvalidTokenError("Token expirado.")

            user_id = payload.get("user_id") or payload.get("username") # Adaptar si hay 'user_id' en el payload real
            roles = payload.get("roles", [])
            permissions = {Permission(p) for p in payload.get("permissions", [])} # Suponiendo que permisos también están en el token

            return AuthResult(
                success=True,
                user_id=user_id,
                roles=roles,
                permissions=permissions,
                details={"token_payload": payload}
            )

        except (jwt.InvalidTokenError, KeyError) as e:
            logger.error(f"Fallo en la validación del token JWT: {e}")
            raise InvalidTokenError(f"Token JWT inválido: {e}")

    async def _create_jwt_token(self, username: str, roles: List[str]) -> AuthResult:
        """Crea un token JWT con la información de usuario y roles."""
        expires_at = datetime.now() + timedelta(hours=24)
        payload = {
            "username": username,
            "roles": roles,
            "iat": datetime.now(),
            "exp": expires_at,
            "iss": "agent_framework_security"
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        return AuthResult(
            success=True,
            user_id=username, # Usar username como user_id para la demo
            roles=roles,
            access_token=token,
            token_type="bearer",
            expires_in=int((expires_at - datetime.now()).total_seconds()),
            details={"message": "Token JWT generado exitosamente."}
        )

class APIKeyAuthProvider(AuthenticationProvider):
    """Proveedor de autenticación por API Key."""

    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        logger.info("APIKeyAuthProvider inicializado.")

    async def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Autenticar con API key."""
        api_key = credentials.get("api_key")
        key_info = self.api_keys.get(api_key)

        if key_info and key_info.get("is_active", True):
            user_id = key_info.get("user_id")
            permissions = {Permission(p) for p in key_info.get("permissions", [])}
            logger.info(f"API Key autenticada exitosamente para usuario: {user_id}")
            return AuthResult(success=True, user_id=user_id, permissions=permissions, details=key_info)
        logger.warning(f"Intento de autenticación API Key fallido.")
        return AuthResult(success=False, error_message="API Key inválida o inactiva.")

    async def validate_token(self, token: str) -> AuthResult:
        """Valida una API key como token."""
        # Para API Key, validate_token es lo mismo que authenticate
        return await self.authenticate({"api_key": token})

    def create_api_key(self, user_id: str, permissions: List[Permission],
                       description: str = "") -> str:
        """Crea una nueva API key para un usuario."""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            "user_id": user_id,
            "permissions": [p.value for p in permissions],
            "description": description,
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
        logger.info(f"API Key creada para usuario: {user_id}")
        return api_key

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoca una API key, marcándola como inactiva."""
        if api_key in self.api_keys:
            self.api_keys[api_key]["is_active"] = False
            logger.info(f"API Key revocada: {api_key[:10]}...")
            return True
        logger.warning(f"Intento de revocar API Key no existente: {api_key[:10]}...")
        return False

# ================================
# SISTEMA DE AUTORIZACIÓN
# ================================

class AuthorizationManager:
    """Gestor de autorización y permisos basado en roles."""

    def __init__(self):
        self.roles: Dict[str, SecurityRole] = {}
        self.user_permissions_cache: Dict[str, Set[Permission]] = {}
        self._setup_default_roles()
        logger.info("AuthorizationManager inicializado con roles por defecto.")

    def _setup_default_roles(self):
        """Configura los roles por defecto del sistema."""
        self.add_role(SecurityRole(
            name="admin",
            permissions={p for p in Permission}, # Todos los permisos
            security_level=SecurityLevel.RESTRICTED,
            description="Full system administrator"
        ))
        self.add_role(SecurityRole(
            name="operator",
            permissions={
                Permission.READ_AGENTS, Permission.EXECUTE_ACTIONS,
                Permission.READ_RESOURCES, Permission.MONITOR_SYSTEM
            },
            security_level=SecurityLevel.INTERNAL,
            description="System operator with read and execute permissions"
        ))
        self.add_role(SecurityRole(
            name="viewer",
            permissions={
                Permission.READ_AGENTS, Permission.READ_RESOURCES,
                Permission.MONITOR_SYSTEM
            },
            security_level=SecurityLevel.INTERNAL,
            description="Read-only access to system information"
        ))
        self.add_role(SecurityRole(
            name="agent",
            permissions={
                Permission.READ_AGENTS, Permission.EXECUTE_ACTIONS,
                Permission.READ_RESOURCES, Permission.WRITE_RESOURCES
            },
            security_level=SecurityLevel.CONFIDENTIAL,
            description="Agent-to-agent communication permissions"
        ))

    def add_role(self, role: SecurityRole):
        """Añade un nuevo rol al sistema."""
        if role.name in self.roles:
            logger.warning(f"Sobrescribiendo rol existente: {role.name}")
        self.roles[role.name] = role

    def get_user_permissions(self, user_roles: List[str]) -> Set[Permission]:
        """Obtiene el conjunto total de permisos de un usuario basado en sus roles."""
        if not user_roles:
            return set()

        cache_key = ",".join(sorted(user_roles))
        if cache_key in self.user_permissions_cache:
            return self.user_permissions_cache[cache_key]

        permissions = set()
        for role_name in user_roles:
            if role_name in self.roles:
                permissions.update(self.roles[role_name].permissions)
            else:
                logger.warning(f"Rol '{role_name}' no encontrado para calcular permisos.")

        self.user_permissions_cache[cache_key] = permissions
        return permissions

    def check_permission(self, user_permissions: Set[Permission], required_permission: Permission) -> bool:
        """Verifica si el usuario tiene un permiso específico."""
        return required_permission in user_permissions

    def get_effective_security_level(self, user_roles: List[str]) -> SecurityLevel:
        """Calcula el nivel de seguridad efectivo de un usuario basado en sus roles."""
        effective_level = SecurityLevel.PUBLIC # Nivel más bajo por defecto
        levels_order = [SecurityLevel.PUBLIC, SecurityLevel.INTERNAL,
                        SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED]

        for role_name in user_roles:
            if role_name in self.roles:
                role_level = self.roles[role_name].security_level
                if levels_order.index(role_level) > levels_order.index(effective_level):
                    effective_level = role_level
        return effective_level

    def check_security_level(self, user_roles: List[str], required_level: SecurityLevel) -> bool:
        """Verifica si el nivel de seguridad del usuario es suficiente para el nivel requerido."""
        user_level = self.get_effective_security_level(user_roles)
        levels_order = [SecurityLevel.PUBLIC, SecurityLevel.INTERNAL,
                        SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED]
        return levels_order.index(user_level) >= levels_order.index(required_level)

# ================================
# SISTEMA DE AUDITORÍA
# ================================

@dataclass
class AuditEvent:
    """Define un evento de auditoría para registrar actividades de seguridad."""
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
    """Sistema de auditoría y logging de seguridad."""

    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = Path(log_file)
        self.events: List[AuditEvent] = []
        self.max_events_in_memory = 1000
        logger.info(f"AuditLogger inicializado, log_file: {self.log_file}")

    async def log_event(self, user_id: Optional[str], agent_id: Optional[str],
                        action: str, resource_type: str, resource_id: Optional[str],
                        result: str, details: Dict[str, Any] = None,
                        ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Registra un evento de auditoría."""
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

        self.events.append(event)
        if len(self.events) > self.max_events_in_memory:
            self.events.pop(0) # Eliminar el evento más antiguo si se excede el límite

        await self._write_to_file(event)

        # Log de advertencia para eventos de seguridad críticos
        if result == "unauthorized" or "admin" in action.lower() or result == "failure":
            logger.warning(f"Evento de seguridad crítico: {action} por {user_id or agent_id or 'system'} - {result}")
        else:
            logger.info(f"Evento de auditoría: {action} por {user_id or agent_id or 'system'} - {result}")


    async def _write_to_file(self, event: AuditEvent):
        """Escribe un evento de auditoría a un archivo de log."""
        try:
            # Crear el directorio si no existe
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                event_data = asdict(event) # Usar asdict para convertir dataclass a dict
                event_data["timestamp"] = event_data["timestamp"].isoformat()
                f.write(json.dumps(event_data) + "\n")
        except Exception as e:
            logger.error(f"Fallo al escribir el log de auditoría en archivo {self.log_file}: {e}")

    def get_events(self, filter_criteria: Optional[Dict[str, Any]] = None,
                   limit: int = 100) -> List[AuditEvent]:
        """Obtiene eventos de auditoría, aplicando filtros y límites."""
        filtered_events = self.events

        if filter_criteria:
            if "user_id" in filter_criteria:
                filtered_events = [e for e in filtered_events if e.user_id == filter_criteria["user_id"]]
            if "agent_id" in filter_criteria:
                filtered_events = [e for e in filtered_events if e.agent_id == filter_criteria["agent_id"]]
            if "action" in filter_criteria:
                filtered_events = [e for e in filtered_events if filter_criteria["action"] in e.action]
            if "result" in filter_criteria:
                filtered_events = [e for e in filtered_events if e.result == filter_criteria["result"]]
            if "resource_type" in filter_criteria:
                filtered_events = [e for e in filtered_events if e.resource_type == filter_criteria["resource_type"]]

        # Ordenar por timestamp descendente y limitar
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_events[:limit]

# ================================
# WRAPPER DE MENSAJES SEGUROS
# ================================

class SecureMessageWrapper:
    """Wrapper para la encriptación/desencriptación simulada de mensajes entre agentes."""

    def __init__(self, encryption_key: str):
        self.encryption_key = encryption_key
        logger.info("SecureMessageWrapper inicializado.")

    def encrypt_message(self, message: 'AgentMessage') -> 'AgentMessage':
        """
        Simula la encriptación de un mensaje de agente.
        (ADVERTENCIA: NO USAR ESTA ENCRIPTACIÓN EN PRODUCCIÓN. ES SOLO PARA DEMO).
        """
        # Asegúrate de que AgentMessage.payload sea serializable a JSON
        try:
            payload_str = json.dumps(message.payload)
            encrypted_payload = self._simple_encrypt(payload_str)

            # Crear una copia modificada del mensaje
            # Requiere que AgentMessage sea un dataclass o similar con un constructor que acepte campos
            # Si no, se debería construir un nuevo objeto AgentMessage manualmente con los campos existentes
            # Para la demo, asumo que AgentMessage es un dataclass y puede ser instanciado con campos.
            # Si 'AgentMessage' no es un dataclass, esta parte requeriría adaptar.
            # Para fines de esta refactorización, lo mantengo genérico.
            # Si 'AgentMessage' no es un dataclass con campos accesibles, necesitaríamos su definición completa.
            # Asumo que es algo como: AgentMessage(id=..., sender_id=..., payload=..., etc.)
            
            # Placeholder para simular la estructura de AgentMessage si no se importa
            if hasattr(message, 'id'): # Check if it's an AgentMessage-like object
                return type(message)(
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
            else:
                logger.error("El objeto message no parece ser una instancia de AgentMessage con atributos esperados.")
                raise TypeError("El objeto message no es un AgentMessage válido.")

        except Exception as e:
            logger.error(f"Fallo en la encriptación del mensaje: {e}")
            raise

    def decrypt_message(self, encrypted_message: 'AgentMessage') -> 'AgentMessage':
        """
        Simula la desencriptación de un mensaje de agente.
        (ADVERTENCIA: NO USAR ESTA ENCRIPTACIÓN EN PRODUCCIÓN. ES SOLO PARA DEMO).
        """
        if not encrypted_message.payload.get("encrypted"):
            return encrypted_message # No está encriptado, devolver original

        try:
            encrypted_data = encrypted_message.payload["data"]
            decrypted_data_str = self._simple_decrypt(encrypted_data)
            original_payload = json.loads(decrypted_data_str)

            if hasattr(encrypted_message, 'id'):
                return type(encrypted_message)(
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
            else:
                logger.error("El objeto encrypted_message no parece ser una instancia de AgentMessage con atributos esperados.")
                raise TypeError("El objeto encrypted_message no es un AgentMessage válido.")

        except Exception as e:
            logger.error(f"Fallo en la desencriptación del mensaje: {e}")
            raise

    def _simple_encrypt(self, data: str) -> str:
        """Encriptación base64 simple (NO SEGURA PARA PRODUCCIÓN)."""
        import base64
        return base64.b64encode(data.encode()).decode()

    def _simple_decrypt(self, encrypted_data: str) -> str:
        """Desencriptación base64 simple (NO SEGURA PARA PRODUCCIÓN)."""
        import base64
        return base64.b64decode(encrypted_data.encode()).decode()

# ================================
# GESTOR PRINCIPAL DE SEGURIDAD
# ================================

class SecurityManager:
    """
    Gestor principal de seguridad del framework de agentes,
    coordinando autenticación, autorización, auditoría y seguridad de mensajes.
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig() # Usar configuración por defecto si no se provee
        self.auth_providers: Dict[AuthenticationMethod, AuthenticationProvider] = {}
        self.authorization_manager = AuthorizationManager()
        self.audit_logger = AuditLogger(self.config.audit_log_file)
        self.message_wrapper = SecureMessageWrapper(self.config.encryption_key)

        self.active_sessions: Dict[str, UserSession] = {}
        self.agent_credentials: Dict[str, AgentCredentials] = {}

        self._setup_default_providers()
        logger.info("SecurityManager inicializado.")

    def _setup_default_providers(self):
        """Configura los proveedores de autenticación por defecto."""
        self.auth_providers[AuthenticationMethod.JWT_TOKEN] = JWTAuthProvider(self.config.jwt_secret)
        self.auth_providers[AuthenticationMethod.API_KEY] = APIKeyAuthProvider()
        # Se pueden añadir otros proveedores aquí

    async def authenticate_user(self, method: AuthenticationMethod,
                                credentials: Dict[str, Any]) -> UserSession:
        """
        Autentica un usuario usando el método especificado.
        Levanta AuthenticationError si la autenticación falla.
        """
        provider = self.auth_providers.get(method)
        if not provider:
            raise AuthenticationError(f"Método de autenticación no soportado: {method.value}")

        auth_result = await provider.authenticate(credentials)

        if auth_result.success:
            session_id = secrets.token_urlsafe(32)
            
            # Calcular permisos una vez al inicio de la sesión
            user_permissions = self.authorization_manager.get_user_permissions(auth_result.roles)

            session = UserSession(
                session_id=session_id,
                user_id=auth_result.user_id,
                username=auth_result.user_id, # Usar user_id como username para simplicidad
                roles=auth_result.roles,
                permissions=user_permissions,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                auth_method=method.value
            )
            self.active_sessions[session_id] = session

            await self.audit_logger.log_event(
                user_id=session.user_id,
                agent_id=None,
                action="user_authentication",
                resource_type="session",
                resource_id=session_id,
                result="success",
                details={"auth_method": method.value, "roles": session.roles}
            )
            logger.info(f"Usuario {session.user_id} autenticado exitosamente. Sesión ID: {session_id[:10]}...")
            return session
        else:
            await self.audit_logger.log_event(
                user_id=credentials.get("username"),
                agent_id=None,
                action="user_authentication",
                resource_type="session",
                resource_id=None,
                result="failure",
                details={"auth_method": method.value, "error": auth_result.error_message}
            )
            raise AuthenticationError(auth_result.error_message or "Credenciales inválidas.")

    async def validate_session(self, session_id: str) -> UserSession:
        """
        Valida una sesión de usuario activa.
        Levanta AuthenticationError si la sesión es inválida o expirada.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            logger.warning(f"Intento de validar sesión no existente: {session_id[:10]}...")
            raise AuthenticationError("Sesión no encontrada.")

        max_age = timedelta(hours=self.config.session_max_hours)
        if datetime.now() - session.created_at > max_age:
            del self.active_sessions[session_id]
            await self.audit_logger.log_event(
                user_id=session.user_id,
                agent_id=None,
                action="session_expiration",
                resource_type="session",
                resource_id=session_id,
                result="failure",
                details={"reason": "expired"}
            )
            logger.warning(f"Sesión expirada para usuario {session.user_id}: {session_id[:10]}...")
            raise AuthenticationError("Sesión expirada.")

        session.last_activity = datetime.now()
        logger.debug(f"Sesión validada para usuario {session.user_id}: {session_id[:10]}...")
        return session

    async def authorize_action(self, session_id: str, required_permission: Permission,
                               resource_type: str = "general",
                               resource_id: Optional[str] = None,
                               security_level: SecurityLevel = SecurityLevel.INTERNAL) -> bool:
        """
        Autoriza una acción para un usuario en una sesión activa.
        Levanta AuthorizationError si el usuario no tiene permisos suficientes.
        """
        try:
            session = await self.validate_session(session_id)
            user_roles = session.roles
            user_permissions = session.permissions # Usar permisos ya calculados en la sesión

            has_permission = self.authorization_manager.check_permission(user_permissions, required_permission)
            has_security_level = self.authorization_manager.check_security_level(user_roles, security_level)

            result = has_permission and has_security_level

            audit_result_status = "success" if result else "unauthorized"
            if not result:
                logger.warning(
                    f"Autorización denegada para {session.user_id}. Permiso requerido: {required_permission.value}, "
                    f"Nivel de seguridad requerido: {security_level.value}. Roles: {user_roles}"
                )

            await self.audit_logger.log_event(
                user_id=session.user_id,
                agent_id=None,
                action=f"authorize_{required_permission.value}",
                resource_type=resource_type,
                resource_id=resource_id,
                result=audit_result_status,
                details={
                    "required_permission": required_permission.value,
                    "required_security_level": security_level.value,
                    "user_roles": user_roles,
                    "has_permission": has_permission,
                    "has_security_level": has_security_level
                }
            )

            if not result:
                raise AuthorizationError(f"Permisos insuficientes para '{required_permission.value}' o nivel de seguridad '{security_level.value}'.")
            
            logger.info(f"Autorización concedida para {session.user_id} para {required_permission.value}")
            return True

        except AuthenticationError as e:
            # Re-lanzar errores de autenticación si la sesión no es válida
            logger.error(f"Fallo de autorización por sesión inválida: {e}")
            raise AuthorizationError(f"Fallo de autorización: {e}")
        except AuthorizationError:
            # Re-lanzar errores de autorización específicos
            raise
        except Exception as e:
            logger.error(f"Error inesperado durante la autorización: {e}")
            await self.audit_logger.log_event(
                user_id=session_id, # Usar session_id si user_id no está disponible
                agent_id=None,
                action=f"authorize_{required_permission.value}",
                resource_type=resource_type,
                resource_id=resource_id,
                result="error",
                details={"error": str(e)}
            )
            raise SecurityError(f"Error interno del sistema de seguridad durante la autorización: {e}")


    # Mock BaseAgent and AgentMessage for demonstration purposes if not imported
    # REMOVE these if you are importing them from core.autonomous_agent_framework
    class MockBaseAgent:
        def __init__(self, id: str, namespace: str = "agent.mock", name: str = "MockAgent"):
            self.id = id
            self.namespace = namespace
            self.name = name
            
    @dataclass
    class MockAgentMessage:
        id: str
        sender_id: str
        receiver_id: str
        message_type: Enum
        action: str
        payload: Dict[str, Any]
        timestamp: datetime
        correlation_id: Optional[str] = None
        response_required: bool = False

    async def register_agent_credentials(self, agent: Union['BaseAgent', MockBaseAgent],
                                         security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL,
                                         permissions: Optional[Set[Permission]] = None) -> str:
        """
        Registra las credenciales para un agente, generando un token único.
        """
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
            details={"security_level": security_level.value, "permissions": [p.value for p in credentials.permissions]}
        )
        logger.info(f"Credenciales de agente registradas para {agent.id}")
        return agent_token

    async def validate_agent_token(self, agent_id: str, token: str) -> bool:
        """Valida el token de autenticación de un agente."""
        credentials = self.agent_credentials.get(agent_id)
        if not credentials:
            logger.warning(f"Intento de validar token para agente no registrado: {agent_id}")
            return False

        if credentials.agent_token == token:
            if credentials.expires_at and datetime.now() > credentials.expires_at:
                logger.warning(f"Token de agente expirado para: {agent_id}")
                return False
            logger.info(f"Token de agente válido para: {agent_id}")
            return True
        logger.warning(f"Token inválido para agente: {agent_id}")
        return False

    async def secure_agent_message(self, message: Union['AgentMessage', MockAgentMessage]) -> Union['AgentMessage', MockAgentMessage]:
        """
        Asegura un mensaje entre agentes, encriptándolo si el nivel de seguridad lo requiere.
        """
        sender_creds = self.agent_credentials.get(message.sender_id)
        if not sender_creds:
            await self.audit_logger.log_event(
                user_id=None,
                agent_id=message.sender_id,
                action="secure_message_attempt",
                resource_type="message",
                resource_id=message.id,
                result="failure",
                details={"reason": "Sender agent not authenticated"}
            )
            raise AuthenticationError(f"Agente remitente {message.sender_id} no autenticado.")

        if sender_creds.security_level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED]:
            try:
                secured_message = self.message_wrapper.encrypt_message(message)
                logger.debug(f"Mensaje de agente encriptado de {message.sender_id} a {message.receiver_id}")
                return secured_message
            except Exception as e:
                await self.audit_logger.log_event(
                    user_id=None,
                    agent_id=message.sender_id,
                    action="secure_message_attempt",
                    resource_type="message",
                    resource_id=message.id,
                    result="error",
                    details={"reason": f"Encryption failed: {e}"}
                )
                raise SecurityError(f"Fallo al asegurar el mensaje: {e}")
        logger.debug(f"Mensaje de agente no encriptado (nivel de seguridad no lo requiere) de {message.sender_id}")
        return message

    async def logout_session(self, session_id: str) -> bool:
        """Cierra una sesión de usuario activa."""
        session = self.active_sessions.pop(session_id, None)
        if session:
            await self.audit_logger.log_event(
                user_id=session.user_id,
                agent_id=None,
                action="user_logout",
                resource_type="session",
                resource_id=session_id,
                result="success"
            )
            logger.info(f"Sesión cerrada para usuario {session.user_id}: {session_id[:10]}...")
            return True
        logger.warning(f"Intento de cerrar sesión no existente: {session_id[:10]}...")
        return False

    def create_api_key(self, user_id: str, permissions: List[Permission],
                       description: str = "") -> str:
        """Crea una API key para un usuario a través del proveedor de API Key."""
        api_key_provider = self.auth_providers.get(AuthenticationMethod.API_KEY)
        if isinstance(api_key_provider, APIKeyAuthProvider):
            return api_key_provider.create_api_key(user_id, permissions, description)
        else:
            raise RuntimeError("El proveedor de API Key no está disponible o no es del tipo esperado.")

    def get_security_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual de seguridad del sistema."""
        return {
            "active_user_sessions": len(self.active_sessions),
            "registered_agent_credentials": len(self.agent_credentials),
            "authentication_providers_active": [m.value for m in self.auth_providers.keys()],
            "audit_events_in_memory": len(self.audit_logger.events),
            "default_roles_configured": list(self.authorization_manager.roles.keys()),
            "session_max_hours": self.config.session_max_hours,
            "audit_log_file": str(self.config.audit_log_file)
        }

# ================================
# DECORADORES DE SEGURIDAD
# ================================

def require_permission(permission: Permission, security_level: SecurityLevel = SecurityLevel.INTERNAL):
    """
    Decorador para requerir permisos específicos y un nivel de seguridad para un método.
    Asume que el primer argumento del método decorado es un objeto con un 'security_manager'
    y 'current_session_id' (o que 'session_id' se pasa como un kwargs).
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            security_manager: Optional[SecurityManager] = None
            session_id: Optional[str] = None

            # Intentar obtener security_manager y session_id del primer argumento (ej. 'self') o kwargs
            if args and hasattr(args[0], 'security_manager') and isinstance(args[0].security_manager, SecurityManager):
                security_manager = args[0].security_manager
                session_id = kwargs.get('session_id') or getattr(args[0], 'current_session_id', None)
            elif 'security_manager' in kwargs and isinstance(kwargs['security_manager'], SecurityManager):
                security_manager = kwargs['security_manager']
                session_id = kwargs.get('session_id')

            if not security_manager:
                raise RuntimeError("El decorador 'require_permission' requiere un SecurityManager accesible.")
            if not session_id:
                raise AuthorizationError("ID de sesión no proporcionado para la autorización.")

            await security_manager.authorize_action(
                session_id, permission, func.__name__, security_level=security_level
            )
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# ================================
# EJEMPLO DE USO (Demo refactorizada)
# ================================

async def security_demo():
    """Demo del sistema de seguridad refactorizado."""

    logger.info("🔒 Security System Demo (Refactorizado)")
    logger.info("="*50)

    # Configuración de seguridad
    security_config = SecurityConfig(
        jwt_secret="super_secret_key_for_demo_only_replace_in_prod",
        encryption_key="another_secret_key_for_encryption",
        session_max_hours=1, # Sesiones más cortas para demo
        audit_log_file="refactored_security_audit.log"
    )

    security_manager = SecurityManager(security_config)
    session_id: Optional[str] = None

    try:
        # Demo 1: Autenticación de usuario exitosa
        logger.info("\n1. Demo de Autenticación de Usuario:")
        try:
            user_session = await security_manager.authenticate_user(
                AuthenticationMethod.JWT_TOKEN,
                {"username": "admin", "password": "admin_password"}
            )
            session_id = user_session.session_id
            logger.info(f"✅ Autenticación exitosa. Sesión ID: {session_id[:10]}...")
            logger.info(f"   Usuario: {user_session.username}, Roles: {user_session.roles}")
            logger.info(f"   Permisos de sesión: {[p.value for p in user_session.permissions]}")

            # Demo 2: Autorización con sesión válida
            logger.info("\n2. Demo de Autorización (con sesión válida):")
            permissions_to_test = [
                Permission.READ_AGENTS,
                Permission.DELETE_AGENTS, # Admin debería tener este
                Permission.CREATE_AGENTS,
                Permission.ADMIN_ACCESS
            ]

            for perm in permissions_to_test:
                try:
                    authorized = await security_manager.authorize_action(
                        session_id, perm, "demo_feature"
                    )
                    status = "✅ AUTORIZADO" if authorized else "❌ DENEGADO" # Este no debería verse si levanta excepción
                    logger.info(f"   Acción '{perm.value}': {status}")
                except AuthorizationError as e:
                    logger.info(f"   Acción '{perm.value}': ❌ DENEGADO - {e}")
            
            # Intento con un usuario sin permisos de administrador
            logger.info("\n2b. Demo de Autorización (como operador):")
            operator_session = await security_manager.authenticate_user(
                AuthenticationMethod.JWT_TOKEN,
                {"username": "operator", "password": "operator_password"}
            )
            operator_session_id = operator_session.session_id
            logger.info(f"   Operador autenticado. Sesión ID: {operator_session_id[:10]}...")

            try:
                await security_manager.authorize_action(operator_session_id, Permission.DELETE_AGENTS, "agent_management")
                logger.info("   Operador: ✅ Autorizado para DELETE_AGENTS (¡Esto no debería pasar!)")
            except AuthorizationError as e:
                logger.info(f"   Operador: ❌ DENEGADO para DELETE_AGENTS - {e}")
            finally:
                await security_manager.logout_session(operator_session_id)


        except AuthenticationError as e:
            logger.error(f"❌ Fallo de autenticación inicial: {e}")

        # Demo 3: API Key
        logger.info("\n3. Demo de API Key:")
        try:
            api_key = security_manager.create_api_key(
                "demo_user_api",
                [Permission.READ_RESOURCES, Permission.MONITOR_SYSTEM],
                "API Key para monitoreo"
            )
            logger.info(f"   API Key creada: {api_key[:20]}...")

            api_auth_session = await security_manager.authenticate_user(
                AuthenticationMethod.API_KEY,
                {"api_key": api_key}
            )
            logger.info(f"   ✅ API Key validada. Sesión ID: {api_auth_session.session_id[:10]}...")
            logger.info(f"   Usuario API Key: {api_auth_session.user_id}, Permisos: {[p.value for p in api_auth_session.permissions]}")
            
            # Revocar API Key
            if security_manager.auth_providers[AuthenticationMethod.API_KEY].revoke_api_key(api_key):
                logger.info("   ✅ API Key revocada.")
            
            # Intentar usar la API Key revocada
            try:
                await security_manager.authenticate_user(
                    AuthenticationMethod.API_KEY, {"api_key": api_key}
                )
                logger.info("   ❌ API Key revocada aún válida (¡Esto no debería pasar!)")
            except AuthenticationError as e:
                logger.info(f"   ✅ Intento de autenticación con API Key revocada falló: {e}")

        except Exception as e:
            logger.error(f"❌ Fallo en demo de API Key: {e}")

        # Demo 4: Seguridad de Agentes y Mensajes
        logger.info("\n4. Demo de Seguridad de Agentes y Mensajes:")

        # Crear agente mock (usando la clase MockBaseAgent definida en SecurityManager)
        mock_agent = security_manager.MockBaseAgent(id="sensor_agent_001", namespace="agent.sensor", name="SensorAgent")
        agent_token = await security_manager.register_agent_credentials(
            mock_agent,
            SecurityLevel.CONFIDENTIAL,
            {Permission.READ_AGENTS, Permission.EXECUTE_ACTIONS}
        )
        logger.info(f"   Token de agente creado: {agent_token[:20]}...")

        token_valid = await security_manager.validate_agent_token(mock_agent.id, agent_token)
        logger.info(f"   Validación de token de agente: {'✅ Válido' if token_valid else '❌ Inválido'}")

        # Simular envío de mensaje seguro
        from enum import Enum as _Enum # Usar alias para evitar conflicto si AgentMessage no se importa
        class MockMessageType(_Enum):
            COMMAND = "command"
            RESPONSE = "response"

        mock_message = security_manager.MockAgentMessage(
            id="msg_123",
            sender_id=mock_agent.id,
            receiver_id="control_agent_001",
            message_type=MockMessageType.COMMAND,
            action="report_status",
            payload={"cpu_usage": 75.5, "memory_usage": 80.2},
            timestamp=datetime.now()
        )

        try:
            secure_msg = await security_manager.secure_agent_message(mock_message)
            logger.info(f"   Mensaje de agente procesado por seguridad. Encriptado: {secure_msg.payload.get('encrypted', False)}")
            
            # Desencriptar para verificar (solo para demo)
            if secure_msg.payload.get('encrypted'):
                decrypted_msg = security_manager.message_wrapper.decrypt_message(secure_msg)
                logger.info(f"   Mensaje desencriptado: {decrypted_msg.payload}")

        except SecurityError as e:
            logger.error(f"   ❌ Fallo al asegurar/procesar mensaje de agente: {e}")


        # Demo 5: Auditoría
        logger.info("\n5. Demo del Sistema de Auditoría:")
        audit_events = security_manager.audit_logger.get_events(limit=5)
        logger.info(f"   Eventos de auditoría recientes ({len(audit_events)}):")
        for event in audit_events:
            timestamp_str = event.timestamp.strftime("%H:%M:%S")
            actor = event.user_id or event.agent_id or "system"
            logger.info(f"   [{timestamp_str}] {actor}: {event.action} - {event.result} (Tipo: {event.resource_type})")

        # Demo 6: Estado de seguridad
        logger.info("\n6. Estado del Sistema de Seguridad:")
        status = security_manager.get_security_status()
        for key, value in status.items():
            logger.info(f"   {key}: {value}")

    finally:
        # Cleanup: Cerrar sesión si existe
        if session_id:
            await security_manager.logout_session(session_id)
            logger.info(f"\n🔓 Sesión de admin cerrada: {session_id[:10]}...")

    logger.info("\n✅ Demo de seguridad completada.")

if __name__ == "__main__":
    asyncio.run(security_demo())