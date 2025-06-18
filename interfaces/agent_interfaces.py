# agentapi/interfaces/agent_interfaces.py

import uuid
import asyncio
import logging
import weakref
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable, Union
from datetime import datetime

from agentapi.models.agent_models import AgentStatus, AgentCapability, AgentMessage, MessageType

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
        self.created_at = datetime.now()
        self.last_heartbeat = datetime.now()
        self.logger = logging.getLogger(f"agent.{namespace}.{name}")

    @property
    def framework(self):
        # Acceso a la referencia débil del framework
        fw = self._framework_ref()
        if fw is None:
            self.logger.error(f"Framework reference for agent {self.id} is no longer valid.")
            # Podrías cambiar el estado del agente a ERROR o TERMINATED aquí
        return fw

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Inicializa el estado del agente, registra capacidades, etc.
        Debe ser implementado por subclases. Retorna True si la inicialización fue exitosa.
        """
        pass

    @abstractmethod
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta una acción específica con parámetros dados.
        Debe ser implementado por subclases.
        """
        pass

    async def start(self):
        """Inicia el agente y su procesamiento de mensajes."""
        if self.status == AgentStatus.INITIALIZING:
            self.logger.info(f"Agent {self.namespace}.{self.name} ({self.id}) initializing...")
            if await self.initialize():
                self.status = AgentStatus.ACTIVE
                self._message_listener_task = asyncio.create_task(self._message_listener())
                self.logger.info(f"Agent {self.namespace}.{self.name} ({self.id}) started and active.")
                
                # Registrarse con el framework
                if self.framework and self.framework.registry:
                    self.framework.registry.register_agent(self)
                    self.framework.message_bus.subscribe(self.id, self.message_queue)
                    # Registrar capacidades con el framework
                    for cap in self.capabilities:
                        self.framework.registry.register_capability(cap, self.id)
                else:
                    self.logger.warning(f"Framework registry or message bus not available for agent {self.id}.")
            else:
                self.status = AgentStatus.ERROR
                self.logger.error(f"Agent {self.namespace}.{self.name} ({self.id}) failed to initialize.")
        else:
            self.logger.warning(f"Attempted to start agent {self.id} which is already in status {self.status.value}.")

    async def stop(self):
        """Detiene el agente y su procesamiento de mensajes."""
        self.logger.info(f"Agent {self.namespace}.{self.name} ({self.id}) stopping...")
        self.status = AgentStatus.TERMINATED
        if self._message_listener_task:
            self._message_listener_task.cancel()
            try:
                await self._message_listener_task
            except asyncio.CancelledError:
                pass
        
        if self.framework and self.framework.registry:
            self.framework.registry.deregister_agent(self.id)
            self.framework.message_bus.unsubscribe(self.id)
            # Desregistrar capacidades del framework
            for cap in self.capabilities:
                self.framework.registry.deregister_capability(cap.namespace, self.id)
        
        self.logger.info(f"Agent {self.namespace}.{self.name} ({self.id}) stopped.")

    async def _message_listener(self):
        """Tarea que escucha y procesa mensajes entrantes."""
        while self.status != AgentStatus.TERMINATED:
            try:
                message: AgentMessage = await self.message_queue.get()
                self.logger.info(f"Agent {self.id} received message: {message.id} ({message.message_type.value}) from {message.sender_id}")
                await self._process_message(message)
            except asyncio.CancelledError:
                self.logger.info(f"Message listener for agent {self.id} cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error processing message for agent {self.id}: {e}", exc_info=True)
                # Considerar reportar al MonitoringSystem
            finally:
                self.message_queue.task_done()

    async def _process_message(self, message: AgentMessage):
        """Procesa un mensaje recibido. Este método puede ser extendido por subclases."""
        if message.message_type == MessageType.COMMAND or message.message_type == MessageType.REQUEST:
            action = message.content.get("action")
            params = message.content.get("params", {})
            response_content: Dict[str, Any] = {}
            try:
                # Verificar si el agente tiene la capacidad de ejecutar la acción
                found_capability = None
                for cap in self.capabilities:
                    if cap.name == action or cap.namespace == action:
                        found_capability = cap
                        break

                if found_capability and found_capability.handler:
                    # Opcional: Verificar permisos específicos de la capacidad
                    if found_capability.requires_permission and self.framework.security_manager:
                        from agentapi.models.security_models import Permission # Importación local para evitar circular
                        if not await self.framework.security_manager.check_permission(
                            agent_id=message.sender_id,
                            permission=Permission(found_capability.requires_permission)
                        ):
                            error_msg = f"Agent {message.sender_id} denied permission to execute capability '{action}' on agent {self.id}."
                            self.logger.warning(error_msg)
                            response_content = {"error": error_msg, "status": "permission_denied"}
                            if message.message_type == MessageType.REQUEST:
                                await self._send_response(message, response_content)
                            return # Salir si no hay permiso

                    self.status = AgentStatus.BUSY # Marcar agente como ocupado
                    self.logger.info(f"Agent {self.id} executing action '{action}' from message {message.id}")
                    # Aquí se podría añadir validación de esquema de entrada (input_schema)
                    execution_result = await found_capability.handler(params)
                    response_content = {"result": execution_result, "status": "success"}
                else:
                    error_msg = f"Action '{action}' not found or no handler for agent {self.id}."
                    self.logger.warning(error_msg)
                    response_content = {"error": error_msg, "status": "failed"}

            except Exception as e:
                error_msg = f"Error executing action '{action}' for agent {self.id}: {e}"
                self.logger.error(error_msg, exc_info=True)
                response_content = {"error": error_msg, "status": "failed", "traceback": traceback.format_exc()}
            finally:
                self.status = AgentStatus.ACTIVE # Volver a activo/idle
                # Enviar respuesta si el mensaje era un REQUEST
                if message.message_type == MessageType.REQUEST:
                    await self._send_response(message, response_content)
        elif message.message_type == MessageType.HEARTBEAT:
            self.last_heartbeat = datetime.now()
            # Podrías responder con un HEARTBEAT_ACK si es necesario
        elif message.message_type == MessageType.ERROR:
            self.logger.error(f"Agent {self.id} received an ERROR message from {message.sender_id}: {message.content.get('error_message')}")
        else:
            self.logger.warning(f"Agent {self.id} received unhandled message type: {message.message_type.value}")

    async def _send_response(self, original_message: AgentMessage, content: Dict[str, Any]):
        """Helper para enviar mensajes de respuesta."""
        response_message = AgentMessage(
            sender_id=self.id,
            receiver_id=original_message.sender_id,
            message_type=MessageType.RESPONSE,
            content=content,
            trace_id=original_message.id # Para correlacionar respuesta con solicitud
        )
        await self.framework.message_bus.send_message(response_message)


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
        return f"<Agent {self.name} ({self.id[:8]}...) [{self.status.value}]>"