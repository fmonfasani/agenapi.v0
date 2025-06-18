# agentapi/interfaces/security_interfaces.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from agentapi.models.framework_models import SecurityConfig
from agentapi.models.general_models import AuditEvent
from agentapi.models.security_models import Permission, SecurityLevel, UserRole, AgentRole, AuthenticationMethod, SecurityToken

class ISecurityManager(ABC):
    """
    Interfaz abstracta para el sistema de seguridad.
    Define las operaciones de autenticación, autorización y auditoría.
    """

    @abstractmethod
    async def initialize(self, config: SecurityConfig) -> bool:
        """Inicializa el sistema de seguridad con la configuración dada."""
        pass

    @abstractmethod
    async def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Autentica un usuario y devuelve un token si es exitoso."""
        pass

    @abstractmethod
    async def validate_user_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Valida un token de usuario y devuelve el payload si es válido."""
        pass

    @abstractmethod
    async def register_agent_credentials(self, agent_id: str, security_level: SecurityLevel, permissions: Set[Permission]) -> str:
        """Registra credenciales para un agente y devuelve un token de agente."""
        pass
    
    @abstractmethod
    async def validate_agent_token(self, agent_id: str, token: str) -> bool:
        """Valida un token de agente."""
        pass

    @abstractmethod
    async def check_permission(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, permission: Permission = Permission.ADMIN_ACCESS) -> bool:
        """Verifica si un usuario o agente tiene un permiso específico."""
        pass
    
    @abstractmethod
    async def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """Asigna un rol de seguridad a un usuario."""
        pass

    @abstractmethod
    async def assign_role_to_agent(self, agent_id: str, role_name: str) -> bool:
        """Asigna un rol de seguridad a un agente."""
        pass

    @property
    @abstractmethod
    def audit_logger(self) -> Any: # Retorna una interfaz de AuditLogger
        """Obtiene la instancia del sistema de registro de auditoría."""
        pass

    @abstractmethod
    async def get_security_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema de seguridad."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Realiza tareas de cierre para el sistema de seguridad."""
        pass

class IAuditLogger(ABC):
    """Interfaz abstracta para el sistema de registro de auditoría."""

    @abstractmethod
    async def log_event(self, event_type: Any, action: str, user_id: Optional[str] = None,
                        agent_id: Optional[str] = None, resource_id: Optional[str] = None,
                        result: str = "SUCCESS", details: Optional[Dict[str, Any]] = None) -> None:
        """Registra un evento de auditoría."""
        pass

    @abstractmethod
    def get_events(self, limit: int = 100, event_type: Optional[Any] = None) -> List[AuditEvent]:
        """Obtiene eventos de auditoría."""
        pass

    @abstractmethod
    async def cleanup_old_events(self, older_than_days: int) -> int:
        """Elimina eventos de auditoría antiguos."""
        pass