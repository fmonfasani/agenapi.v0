"""
core/registry.py - Módulo para la gestión del registro de agentes.
"""

import uuid
import logging
import weakref
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
from dataclasses import dataclass, field

# Importaciones desde el nuevo módulo de modelos
from core.models import AgentStatus, AgentInfo # <--- CAMBIO CLAVE

class AgentRegistry:
    """
    Gestiona el registro y ciclo de vida de los agentes.
    Utiliza weakref para evitar referencias circulares y permitir la recolección de agentes inactivos.
    """
    def __init__(self):
        logging.info("Initializing AgentRegistry...")
        # Usamos WeakValueDictionary para que los agentes puedan ser recolectados
        # por el garbage collector si no hay otras referencias fuertes a ellos.
        self._agents: Dict[str, weakref.ReferenceType[Any]] = weakref.WeakValueDictionary()
        logging.info("AgentRegistry initialized.")

    async def register_agent(self, agent: Any) -> bool:
        """
        Registra un agente en el sistema.
        Se espera que 'agent' tenga atributos como 'id', 'name', 'namespace', 'status'.
        """
        if not hasattr(agent, 'id') or not hasattr(agent, 'name') or not hasattr(agent, 'namespace') or not hasattr(agent, 'status'):
            logging.error(f"Attempted to register object without required agent attributes: {type(agent)}")
            return False

        if agent.id in self._agents and self._agents[agent.id]() is not None:
            logging.warning(f"Agent with ID {agent.id} already registered. Updating status.")
            # Si ya existe, actualizamos su información. No debería pasar si el ID es único.
        
        self._agents[agent.id] = weakref.ref(agent)
        logging.info(f"Agent '{agent.namespace}.{agent.name}' (ID: {agent.id}) registered.")
        return True

    async def unregister_agent(self, agent_id: str) -> bool:
        """Da de baja a un agente del registro."""
        if agent_id in self._agents:
            agent_ref = self._agents.pop(agent_id)
            agent = agent_ref()
            if agent:
                logging.info(f"Agent '{agent.namespace}.{agent.name}' (ID: {agent_id}) unregistered.")
            else:
                logging.info(f"Agent (ID: {agent_id}) unregistered (was already garbage collected).")
            return True
        logging.warning(f"Attempted to unregister unknown agent with ID: {agent_id}.")
        return False

    def get_agent(self, agent_id: str) -> Optional[Any]:
        """Obtiene una referencia a un agente por su ID."""
        agent_ref = self._agents.get(agent_id)
        if agent_ref:
            agent = agent_ref()
            if agent:
                return agent
            else:
                # La referencia débil apunta a un objeto que ya fue recolectado. Limpiar.
                self._agents.pop(agent_id, None)
                logging.warning(f"Agent with ID {agent_id} was garbage collected but still in registry. Removed.")
        logging.debug(f"Agent with ID {agent_id} not found or no longer active.")
        return None

    def list_all_agents(self) -> List[Any]:
        """Lista todos los agentes activos en el registro."""
        active_agents = []
        # weakref.WeakValueDictionary itera sobre referencias fuertes activas
        for agent in self._agents.values():
            if agent: # Asegurarse de que la referencia débil no sea None
                active_agents.append(agent)
        return active_agents

    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Obtiene el estado actual de un agente."""
        agent = self.get_agent(agent_id)
        return agent.status if agent else None
    
    def update_agent_status(self, agent_id: str, new_status: AgentStatus) -> bool:
        """Actualiza el estado de un agente."""
        agent = self.get_agent(agent_id)
        if agent:
            agent.status = new_status
            logging.debug(f"Agent {agent_id} status updated to {new_status.value}.")
            return True
        logging.warning(f"Cannot update status for unknown agent: {agent_id}")
        return False
    
    def count_agents_by_status(self) -> Dict[AgentStatus, int]:
        """Cuenta agentes por su estado."""
        counts = {status: 0 for status in AgentStatus}
        for agent in self._agents.values():
            if agent and agent.status in counts:
                counts[agent.status] += 1
        return counts

    def get_agent_info_list(self) -> List[AgentInfo]:
        """Retorna una lista de objetos AgentInfo para todos los agentes registrados."""
        info_list = []
        for agent in self._agents.values():
            if agent:
                info_list.append(
                    AgentInfo(
                        id=agent.id,
                        name=agent.name,
                        namespace=agent.namespace,
                        status=agent.status,
                        last_heartbeat=agent.last_heartbeat if hasattr(agent, 'last_heartbeat') else datetime.now()
                    )
                )
        return info_list