# agentapi/interfaces/agent_interfaces.py

import uuid
import asyncio
import logging
import weakref
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable, Union
from datetime import datetime

from agentapi.models.agent_models import AgentStatus, AgentCapability, AgentMessage, MessageType, AgentInfo, AgentResource, ResourceType # Ensure all necessary models are imported

class BaseAgent(ABC):
    """
    Clase base abstracta para todos los agentes.
    Define la interfaz mínima y la lógica común para un agente autónomo.
    """
    def __init__(self, namespace: str, name: str, framework):
        self.id = str(uuid.uuid4())
        self.namespace = namespace # Ej: "agent.planning", "agent.development.backend"
        self.name = name
        self._framework_ref = weakref.ref(framework) # Referencia débil al framework
        self.status = AgentStatus.INITIALIZING
        self.capabilities: List[AgentCapability] = []
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self._message_listener_task: Optional[asyncio.Task] = None
        self._heartbeat_sender_task: Optional[asyncio.Task] = None # Added heartbeat task
        self.created_at = datetime.now()
        self.last_heartbeat = datetime.now()
        self.logger = logging.getLogger(f"agent.{namespace}.{name}")
        self._stop_event = asyncio.Event() # Internal stop event

    @property
    def framework(self):
        """Acceso a la referencia débil del framework."""
        fw = self._framework_ref()
        if fw is None:
            self.logger.error(f"Framework reference is no longer valid for agent {self.id}")
            # Optionally, trigger self-termination or report error
        return fw

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Método asíncrono para inicializar el agente.
        Debe ser implementado por subclases.
        Retorna True si la inicialización fue exitosa, False en caso contrario.
        """
        self.logger.info(f"Agent {self.id} ({self.name}) initializing...")
        return True

    @abstractmethod
    async def process_message(self, message: AgentMessage):
        """
        Método asíncrono para procesar un mensaje entrante.
        Debe ser implementado por subclases.
        """
        self.logger.info(f"Agent {self.id} ({self.name}) received message: {message.id}")
        pass

    async def _message_listener(self):
        """Tarea de fondo para escuchar mensajes en la cola del agente."""
        self.logger.info(f"Agent {self.id} message listener started.")
        while not self._stop_event.is_set():
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                if message:
                    await self.process_message(message)
                self.message_queue.task_done()
            except asyncio.TimeoutError:
                # No messages for a while, just continue loop to check stop event
                pass
            except asyncio.CancelledError:
                self.logger.info(f"Agent {self.id} message listener cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error processing message for agent {self.id}: {e}", exc_info=True)
                # Optionally, send an error message back or raise an alert
        self.logger.info(f"Agent {self.id} message listener stopped.")

    async def _heartbeat_sender(self):
        """Tarea de fondo para enviar latidos al framework."""
        self.logger.info(f"Agent {self.id} heartbeat sender started.")
        while not self._stop_event.is_set():
            try:
                self.last_heartbeat = datetime.now()
                # Inform the framework's registry about the status
                if self.framework and self.framework.registry:
                    await self.framework.registry.update_agent_status(self.id, self.status, self.last_heartbeat)
                
                # Sleep for the configured heartbeat interval, or default
                await asyncio.sleep(self.framework.config.heartbeat_interval if self.framework else 30)
            except asyncio.CancelledError:
                self.logger.info(f"Agent {self.id} heartbeat sender cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in agent {self.id} heartbeat: {e}", exc_info=True)
        self.logger.info(f"Agent {self.id} heartbeat sender stopped.")

    async def start(self):
        """Inicia las tareas de fondo del agente y lo registra."""
        if self.status not in [AgentStatus.ACTIVE, AgentStatus.BUSY]:
            self.logger.info(f"Starting agent {self.id} ({self.name})...")
            self._stop_event.clear()
            # These tasks are now started by AgentFactory within the wrapper
            # so they are not started here directly.
            self.status = AgentStatus.ACTIVE
            self.last_heartbeat = datetime.now()
            self.logger.info(f"Agent {self.id} ({self.name}) is now ACTIVE.")

    async def stop(self):
        """Detiene el agente, cancelando sus tareas de fondo y actualizando su estado."""
        if self.status == AgentStatus.TERMINATED:
            self.logger.info(f"Agent {self.id} ({self.name}) is already terminated.")
            return

        self.logger.info(f"Stopping agent {self.id} ({self.name})...")
        self.status = AgentStatus.TERMINATED
        self._stop_event.set() # Signal background tasks to stop

        tasks_to_cancel = []
        if self._message_listener_task and not self._message_listener_task.done():
            self._message_listener_task.cancel()
            tasks_to_cancel.append(self._message_listener_task)
        if self._heartbeat_sender_task and not self._heartbeat_sender_task.done():
            self._heartbeat_sender_task.cancel()
            tasks_to_cancel.append(self._heartbeat_sender_task)

        if tasks_to_cancel:
            self.logger.debug(f"Waiting for agent {self.id} background tasks to finish...")
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True) # Await cancellation
            self.logger.debug(f"Agent {self.id} background tasks finished.")
        
        # Unregister from framework components if still registered (framework will also do this)
        if self.framework and self.framework.registry:
            await self.framework.registry.unregister_agent(self.id)
        if self.framework and self.framework.message_bus:
             await self.framework.message_bus.unregister_agent_queue(self.id)

        self.logger.info(f"Agent {self.id} ({self.name}) terminated.")

    async def send_message(self, receiver_id: str, message_type: MessageType, content: Dict[str, Any], trace_id: Optional[str] = None) -> str:
        """Envía un mensaje a otro agente a través del bus de mensajes."""
        if not self.framework or not self.framework.message_bus:
            self.logger.error(f"Cannot send message: Framework or MessageBus not available for agent {self.id}.")
            raise RuntimeError("Framework MessageBus not initialized.")

        message = AgentMessage(
            sender_id=self.id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            trace_id=trace_id
        )
        await self.framework.message_bus.send_message(message)
        return message.id

    async def publish_resource(self, resource_name: str, resource_type: ResourceType, namespace: str, data: Any) -> Optional[AgentResource]:
        """Publica un recurso en el registro de recursos del framework."""
        if not self.framework or not self.framework.resource_manager:
            self.logger.error(f"Cannot publish resource: Framework or ResourceManager not available for agent {self.id}.")
            return None
        
        resource = AgentResource(
            name=resource_name,
            type=resource_type,
            namespace=namespace,
            data=data,
            owner_agent_id=self.id
        )
        await self.framework.resource_manager.add_resource(resource)
        self.logger.info(f"Agent {self.id} published resource: {resource.name} ({resource.id})")
        return resource

    async def get_resource(self, resource_id: Optional[str] = None, resource_name: Optional[str] = None, namespace: Optional[str] = None) -> Optional[AgentResource]:
        """Obtiene un recurso del registro de recursos del framework."""
        if not self.framework or not self.framework.resource_manager:
            self.logger.error(f"Cannot get resource: Framework or ResourceManager not available for agent {self.id}.")
            return None
        
        if resource_id:
            return await self.framework.resource_manager.get_resource(resource_id)
        elif resource_name and namespace:
            return await self.framework.resource_manager.get_resource_by_name(resource_name, namespace)
        return None

    async def create_agent(self, namespace: str, name: str, agent_class: Type['BaseAgent']) -> Optional['BaseAgent']:
        """Permite que un agente cree otro agente usando la factoría del framework."""
        if not self.framework or not self.framework.agent_factory:
            self.logger.error(f"Cannot create agent: Framework or AgentFactory not available for agent {self.id}.")
            return None

        # La verificación de permisos ahora se delega a AgentFactory y SecurityManager
        return await self.framework.agent_factory.create_agent(
            namespace=namespace,
            name=name,
            agent_class=agent_class,
            creator_agent_id=self.id
        )

    def __repr__(self):
        return f"<BaseAgent {self.namespace}.{self.name} (ID: {self.id[:8]}..., Status: {self.status.value})>"