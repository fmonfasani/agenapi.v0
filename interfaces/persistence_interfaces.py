# agentapi/interfaces/persistence_interfaces.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from agentapi.models.framework_models import PersistenceConfig
from agentapi.models.agent_models import AgentMessage, AgentResource, AgentStatus

class IPersistenceManager(ABC):
    """
    Interfaz abstracta para el gestor de persistencia.
    Define las operaciones básicas que cualquier implementación de persistencia debe soportar.
    """

    @abstractmethod
    async def initialize(self, config: PersistenceConfig) -> bool:
        """Inicializa el backend de persistencia con la configuración dada."""
        pass

    @abstractmethod
    async def save_agent_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """Guarda el estado interno de un agente."""
        pass

    @abstractmethod
    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Carga el estado interno de un agente."""
        pass

    @abstractmethod
    async def save_message(self, message: AgentMessage) -> bool:
        """Guarda un mensaje en el historial."""
        pass

    @abstractmethod
    async def load_messages(self, agent_id: str, limit: int = 100) -> List[AgentMessage]:
        """Carga los mensajes enviados/recibidos por un agente."""
        pass

    @abstractmethod
    async def save_resource(self, resource: AgentResource) -> bool:
        """Guarda un recurso del sistema."""
        pass

    @abstractmethod
    async def load_resource(self, resource_id: str) -> Optional[AgentResource]:
        """Carga un recurso por su ID."""
        pass

    @abstractmethod
    async def delete_resource(self, resource_id: str) -> bool:
        """Elimina un recurso por su ID."""
        pass

    @abstractmethod
    async def save_full_state(self, framework_state: Dict[str, Any]) -> bool:
        """
        Guarda el estado completo del framework (agentes, recursos, etc.).
        Esta es una operación de 'snapshot'.
        """
        pass
    
    @abstractmethod
    async def load_full_state(self) -> Optional[Dict[str, Any]]:
        """
        Carga el estado completo del framework.
        """
        pass

    @abstractmethod
    async def cleanup(self, older_than_days: int) -> Dict[str, Any]:
        """Realiza tareas de limpieza de datos antiguos."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Cierra cualquier conexión o recurso abierto por el gestor de persistencia."""
        pass