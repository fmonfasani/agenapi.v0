"""
Framework para Agentes Autónomos Interoperables
Versión: 0.0.1
"""

import asyncio
import uuid
import json
import logging
from typing import Dict, List, Any, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
import weakref # Todavía puede ser necesario si BaseAgent usa weakref internamente
from contextlib import asynccontextmanager
import traceback

# Importar las nuevas clases/módulos de `core`
from core.models import AgentStatus, MessageType, ResourceType, AgentMessage, AgentResource, AgentCapability # <--- CAMBIO CLAVE
from core.registry import AgentRegistry # <--- CAMBIO CLAVE

# ================================\
# ABSTRACT BASE CLASSES (ABCs)
# ================================\

class BaseAgent(ABC):
    """
    Clase base abstracta para todos los agentes.
    Define la interfaz mínima que debe implementar cualquier agente.
    """
    def __init__(self, namespace: str, name: str, framework: 'AgentFramework'):
        self.id = str(uuid.uuid4())
        self.namespace = namespace # e.g., "agent.planning", "agent.development.code"
        self.name = name # Nombre único dentro del namespace
        self.status = AgentStatus.INITIALIZING
        self.framework: 'AgentFramework' = weakref.proxy(framework) # Usar weakref para evitar ciclo de referencia
        self.capabilities: List[AgentCapability] = []
        self.message_queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.task: Optional[asyncio.Task] = None
        self.last_heartbeat = datetime.now()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f"{self.namespace}.{self.name}")
        self.logger.info(f"Agent {self.namespace}.{self.name} (ID: {self.id}) initialized.")

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Inicializa el agente, configura sus capacidades y carga estados.
        Debe ser implementado por subclases.
        """
        pass

    async def start(self):
        """Inicia el bucle de procesamiento del agente."""
        if self.status not in [AgentStatus.ACTIVE, AgentStatus.BUSY]:
            await self.framework.registry.register_agent(self)
            self.status = AgentStatus.ACTIVE
            self.task = asyncio.create_task(self._run())
            self.logger.info(f"Agent {self.name} started.")
        else:
            self.logger.warning(f"Agent {self.name} is already running or busy.")

    async def stop(self):
        """Detiene el bucle de procesamiento del agente."""
        if self.status != AgentStatus.TERMINATED:
            self.stop_event.set()
            await self.framework.registry.unregister_agent(self.id)
            self.status = AgentStatus.TERMINATED
            if self.task:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    self.logger.info(f"Agent {self.name} task cancelled.")
                except Exception as e:
                    self.logger.error(f"Error stopping agent {self.name} task: {e}")
            self.logger.info(f"Agent {self.name} stopped.")

    async def _run(self):
        """Bucle principal de procesamiento de mensajes del agente."""
        self.logger.info(f"Agent {self.name} message processing loop started.")
        try:
            while not self.stop_event.is_set():
                self.last_heartbeat = datetime.now()
                self.framework.registry.update_agent_status(self.id, AgentStatus.ACTIVE) # Keep status updated
                try:
                    # Espera mensajes o el evento de parada con un timeout
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                    await self.process_message(message)
                except asyncio.TimeoutError:
                    # No hay mensajes, solo actualiza el estado o realiza tareas en segundo plano
                    # self.logger.debug(f"Agent {self.name} is idle.")
                    pass
                except Exception as e:
                    self.logger.error(f"Error in agent {self.name} _run loop: {e}", exc_info=True)
                    self.status = AgentStatus.ERROR
                await asyncio.sleep(0.1) # Pequeña pausa para no monopolizar el CPU
        except asyncio.CancelledError:
            self.logger.info(f"Agent {self.name} processing loop cancelled.")
        except Exception as e:
            self.logger.critical(f"Agent {self.name} critical error in _run: {e}", exc_info=True)
            self.status = AgentStatus.ERROR
        finally:
            self.logger.info(f"Agent {self.name} processing loop terminated.")
            self.status = AgentStatus.TERMINATED # Asegurar estado final

    async def process_message(self, message: AgentMessage):
        """Procesa un mensaje entrante."""
        self.logger.info(f"Agent {self.name} received message type: {message.message_type.value} from {message.sender_id}")
        self.status = AgentStatus.BUSY
        try:
            # Lógica para manejar diferentes tipos de mensajes
            if message.message_type == MessageType.COMMAND or message.message_type == MessageType.REQUEST:
                # Disparar la acción correspondiente a la capacidad
                action = message.payload.get("action")
                params = message.payload.get("params", {})
                
                response_payload = {"status": "failed", "error": "Action not found or not handled."}
                
                # Buscar y ejecutar la capacidad
                found_capability = False
                for capability in self.capabilities:
                    if capability.name == action or capability.namespace == action:
                        self.logger.info(f"Executing capability '{capability.name}' for agent {self.name}.")
                        response_payload = await capability.handler(params)
                        found_capability = True
                        break
                
                if not found_capability:
                    self.logger.warning(f"Agent {self.name} received unknown action: {action}")
                    response_payload = {"status": "error", "message": f"Unknown action: {action}"}

                # Si es una REQUEST, enviar una RESPONSE
                if message.message_type == MessageType.REQUEST:
                    response_message = AgentMessage(
                        sender_id=self.id,
                        receiver_id=message.sender_id,
                        message_type=MessageType.RESPONSE,
                        payload=response_payload,
                        correlation_id=message.id # Correlacionar con el mensaje original
                    )
                    await self.framework.message_bus.send_message(response_message)
                    self.logger.info(f"Sent response to {message.sender_id} for request {message.id}")

            elif message.message_type == MessageType.EVENT:
                # Los agentes pueden reaccionar a eventos globales o específicos
                await self.handle_event(message.payload)

            elif message.message_type == MessageType.RESPONSE:
                # Manejar respuestas a requests que este agente envió previamente
                await self.handle_response(message.payload, message.correlation_id)
            
            elif message.message_type == MessageType.HEARTBEAT:
                self.logger.debug(f"Received heartbeat from {message.sender_id}")
                # No se necesita una acción compleja, el loop ya actualiza last_heartbeat.
                pass
            
            elif message.message_type == MessageType.ERROR:
                self.logger.error(f"Received error message from {message.sender_id}: {message.payload.get('error')}")

        except Exception as e:
            self.logger.error(f"Error processing message for agent {self.name}: {e}", exc_info=True)
            # Podría enviar un mensaje de error al remitente si es una solicitud
            if message.message_type == MessageType.REQUEST:
                error_response = AgentMessage(
                    sender_id=self.id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.ERROR,
                    payload={"error": str(e), "original_message": message.payload},
                    correlation_id=message.id
                )
                await self.framework.message_bus.send_message(error_response)
        finally:
            self.status = AgentStatus.ACTIVE

    async def send_message(self, receiver_id: str, action: str, params: Dict[str, Any], message_type: MessageType = MessageType.REQUEST) -> str:
        """
        Envía un mensaje a otro agente a través del MessageBus.
        Retorna el ID del mensaje enviado.
        """
        message = AgentMessage(
            sender_id=self.id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload={"action": action, "params": params}
        )
        await self.framework.message_bus.send_message(message)
        self.logger.info(f"Agent {self.name} sent {message_type.value} '{action}' to {receiver_id} (Msg ID: {message.id}).")
        return message.id

    async def create_agent(self, namespace: str, name: str, agent_class: Type['BaseAgent'], initial_params: Optional[Dict[str, Any]] = None) -> Optional['BaseAgent']:
        """
        Solicita al framework la creación de un nuevo agente.
        Esto debería ser manejado por un agente de orquestación o el propio framework.
        """
        self.logger.info(f"Agent {self.name} requesting creation of new agent: {namespace}.{name}")
        # En una arquitectura real, esto sería un mensaje al Framework o a un AgentFactory
        # para que el Framework gestione la creación y el ciclo de vida.
        # Por ahora, lo implementamos directamente aquí para simplificar la demo.
        try:
            new_agent = await self.framework.agent_factory.create_agent_instance(namespace, name, agent_class, self.framework, initial_params)
            if new_agent:
                self.logger.info(f"Agent {self.name} successfully created new agent: {new_agent.name} (ID: {new_agent.id})")
                return new_agent
        except Exception as e:
            self.logger.error(f"Agent {self.name} failed to create new agent {namespace}.{name}: {e}")
        return None

    async def publish_event(self, event_type: str, data: Dict[str, Any]):
        """Publica un evento para que otros agentes puedan reaccianar."""
        event_message = AgentMessage(
            sender_id=self.id,
            message_type=MessageType.EVENT,
            receiver_id="all", # O un topic específico si implementamos topics
            payload={"event_type": event_type, "data": data}
        )
        await self.framework.message_bus.send_message(event_message)
        self.logger.info(f"Agent {self.name} published event: {event_type}")

    async def handle_event(self, event_payload: Dict[str, Any]):
        """Método para que los agentes manejen eventos entrantes. Puede ser sobreescrito."""
        event_type = event_payload.get("event_type")
        self.logger.info(f"Agent {self.name} handling event: {event_type}")
        # Implementar lógica de manejo de eventos aquí
        pass

    async def handle_response(self, response_payload: Dict[str, Any], correlation_id: Optional[str]):
        """Método para que los agentes manejen respuestas a sus requests. Puede ser sobreescrito."""
        self.logger.info(f"Agent {self.name} handling response for correlation ID: {correlation_id}")
        # Implementar lógica de manejo de respuestas aquí, e.g., usando un diccionario de callbacks
        pass

    def __str__(self):
        return f"{self.namespace}.{self.name} (ID: {self.id[:8]}...)"


# ================================\
# FRAMEWORK CORE COMPONENTS (placeholder for now)
# These will be extracted to their own files in subsequent steps
# ================================\

class MessageBus:
    """
    Sistema de comunicación central para que los agentes intercambien mensajes.
    (Placeholder - se extraerá a core/message_bus.py)
    """
    def __init__(self, framework: 'AgentFramework'):
        self.framework = weakref.proxy(framework)
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.logger = logging.getLogger("MessageBus")
        self._listener_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self.logger.info("MessageBus initialized.")

    async def start(self):
        """Inicia el bus de mensajes."""
        if not self._listener_task:
            self._stop_event.clear()
            self._listener_task = asyncio.create_task(self._listen_for_messages())
            self.logger.info("MessageBus started listening for messages.")

    async def stop(self):
        """Detiene el bus de mensajes."""
        if self._listener_task:
            self._stop_event.set()
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                self.logger.info("MessageBus listener task cancelled.")
            self._listener_task = None
            self.logger.info("MessageBus stopped.")

    async def send_message(self, message: AgentMessage) -> bool:
        """Envía un mensaje a la cola global."""
        try:
            await self.message_queue.put(message)
            self.logger.debug(f"Message {message.id} from {message.sender_id} to {message.receiver_id} queued.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to queue message {message.id}: {e}")
            return False

    async def _listen_for_messages(self):
        """Bucle para procesar mensajes de la cola y enrutarlos a los agentes."""
        self.logger.info("MessageBus listener loop started.")
        while not self._stop_event.is_set():
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                if message.receiver_id == "all":
                    # Enviar a todos los agentes activos (broadcast)
                    for agent in self.framework.registry.list_all_agents():
                        # Evitar que el remitente reciba su propio broadcast si no es necesario
                        if agent.id != message.sender_id:
                            await agent.message_queue.put(message)
                else:
                    receiver_agent = self.framework.registry.get_agent(message.receiver_id)
                    if receiver_agent:
                        await receiver_agent.message_queue.put(message)
                        self.logger.debug(f"Message {message.id} routed to {receiver_agent.name}.")
                    else:
                        self.logger.warning(f"Receiver agent {message.receiver_id} not found for message {message.id}.")
                        # Opcional: enviar mensaje de error de vuelta al remitente
            except asyncio.TimeoutError:
                pass # No hay mensajes, seguir esperando
            except Exception as e:
                self.logger.error(f"Error processing message in MessageBus: {e}", exc_info=True)
            finally:
                self.message_queue.task_done() # Indicar que la tarea de la cola ha sido procesada
            await asyncio.sleep(0.05) # Pequeña pausa para no monopolizar el CPU
        self.logger.info("MessageBus listener loop stopped.")


class ResourceManager:
    """
    Gestiona los recursos compartidos entre agentes.
    (Placeholder - se extraerá a core/resource_manager.py)
    """
    def __init__(self):
        self.resources: Dict[str, AgentResource] = {}
        self.logger = logging.getLogger("ResourceManager")
        self.logger.info("ResourceManager initialized.")

    async def create_resource(self, resource: AgentResource) -> bool:
        """Crea un nuevo recurso."""
        if resource.id in self.resources:
            self.logger.warning(f"Resource with ID {resource.id} already exists. Use update_resource.")
            return False
        self.resources[resource.id] = resource
        self.logger.info(f"Resource '{resource.name}' (ID: {resource.id}) created.")
        return True

    async def get_resource(self, resource_id: str) -> Optional[AgentResource]:
        """Obtiene un recurso por su ID."""
        return self.resources.get(resource_id)

    async def update_resource(self, resource_id: str, new_data: Dict[str, Any]) -> bool:
        """Actualiza los datos de un recurso existente."""
        resource = self.resources.get(resource_id)
        if resource:
            # Asegurarse de que `data` sea un diccionario si se va a usar `update`
            if isinstance(resource.data, dict) and isinstance(new_data, dict):
                resource.data.update(new_data)
            else:
                resource.data = new_data # Reemplazar si no son diccionarios
            resource.last_modified = datetime.now()
            # Opcional: recalcular checksum si la data es grande y se desea verificar integridad
            self.logger.info(f"Resource '{resource.name}' (ID: {resource_id}) updated.")
            return True
        self.logger.warning(f"Resource with ID {resource_id} not found for update.")
        return False

    async def delete_resource(self, resource_id: str) -> bool:
        """Elimina un recurso por su ID."""
        if resource_id in self.resources:
            del self.resources[resource_id]
            self.logger.info(f"Resource with ID {resource_id} deleted.")
            return True
        self.logger.warning(f"Resource with ID {resource_id} not found for deletion.")
        return False

    def find_resources_by_type(self, resource_type: ResourceType) -> List[AgentResource]:
        """Encuentra recursos por tipo."""
        return [r for r in self.resources.values() if r.type == resource_type]

    def find_resources_by_owner(self, owner_agent_id: str) -> List[AgentResource]:
        """Encuentra recursos por agente propietario."""
        return [r for r in self.resources.values() if r.owner_agent_id == owner_agent_id]

    def list_all_resources(self) -> List[AgentResource]:
        """Lista todos los recursos gestionados."""
        return list(self.resources.values())


class AgentFactory:
    """
    Factoría para crear instancias de agentes.
    (Placeholder - se extraerá a core/agent_factory.py)
    """
    def __init__(self, framework: 'AgentFramework'):
        self.framework = weakref.proxy(framework)
        self.logger = logging.getLogger("AgentFactory")
        self.logger.info("AgentFactory initialized.")

    async def create_agent_instance(self, namespace: str, name: str, agent_class: Type[BaseAgent], framework: 'AgentFramework', initial_params: Optional[Dict[str, Any]] = None) -> Optional[BaseAgent]:
        """Crea y registra una instancia de un agente."""
        try:
            agent = agent_class(name=name, framework=framework)
            agent.namespace = namespace # Sobrescribir namespace si la clase lo define diferente
            
            # Si hay parámetros iniciales específicos, aplicarlos aquí antes de la inicialización
            if initial_params:
                for key, value in initial_params.items():
                    if hasattr(agent, key):
                        setattr(agent, key, value)
                    else:
                        self.logger.warning(f"Initial parameter '{key}' not found on agent {name}.")

            if await agent.initialize():
                await agent.start() # Iniciar el agente después de la inicialización
                self.logger.info(f"Successfully created and started agent: {namespace}.{name} (ID: {agent.id})")
                return agent
            else:
                self.logger.error(f"Failed to initialize agent {namespace}.{name}")
                return None
        except Exception as e:
            self.logger.error(f"Error creating agent {namespace}.{name}: {e}", exc_info=True)
            return None

    @classmethod
    async def create_agent_ecosystem(cls, framework: 'AgentFramework') -> Dict[str, BaseAgent]:
        """
        Crea un ecosistema de agentes predefinidos para la demo.
        Esto se moverá a un archivo de configuración o un módulo de orquestación.
        """
        logging.info("Creating agent ecosystem...")
        agents: Dict[str, BaseAgent] = {}

        # Importar dinámicamente o de forma explícita los agentes especializados
        # Esto es un placeholder; en una refactorización real, los agentes se cargarían de config
        try:
            # Importaciones temporales para la demo, deberían ser gestionadas por un PluginManager o similar
            from specialized_agents import StrategistAgent, WorkflowDesignerAgent, CodeGeneratorAgent, TestGeneratorAgent, BuildAgent # type: ignore
        except ImportError:
            logging.error("Could not import specialized agents. Please ensure 'specialized_agents.py' is available.")
            return agents

        factory = AgentFactory(framework) # Usar la instancia de factory

        # Agentes de Planificación
        strategist_agent = await factory.create_agent_instance(
            "agent.planning.strategist", "strategist", StrategistAgent, framework
        )
        if strategist_agent:
            agents['strategist'] = strategist_agent

        workflow_designer_agent = await factory.create_agent_instance(
            "agent.planning.workflow_designer", "workflow_designer", WorkflowDesignerAgent, framework
        )
        if workflow_designer_agent:
            agents['workflow_designer'] = workflow_designer_agent

        # Agentes de Desarrollo
        code_generator_agent = await factory.create_agent_instance(
            "agent.build.code.generator", "code_generator", CodeGeneratorAgent, framework
        )
        if code_generator_agent:
            agents['code_generator'] = code_generator_agent
            
        build_agent_instance = await factory.create_agent_instance(
            "agent.build.builder", "builder", BuildAgent, framework
        )
        if build_agent_instance:
            agents['build_agent'] = build_agent_instance

        # Agentes de Testing
        test_generator_agent = await factory.create_agent_instance(
            "agent.test.generator", "test_generator", TestGeneratorAgent, framework
        )
        if test_generator_agent:
            agents['test_generator'] = test_generator_agent

        # Asegurar que todos los agentes se iniciaron correctamente
        for agent_name, agent_instance in agents.items():
            if agent_instance and agent_instance.status != AgentStatus.ACTIVE:
                logging.warning(f"Agent {agent_name} did not start correctly. Current status: {agent_instance.status.value}")
            elif not agent_instance:
                 logging.error(f"Agent {agent_name} could not be created.")

        logging.info(f"Agent ecosystem created with {len(agents)} agents.")
        return {k: v for k, v in agents.items() if v is not None} # Filter out None values


# ================================\
# MAIN FRAMEWORK CLASS
# ================================\

class AgentFramework:
    """
    Clase principal del Framework de Agentes Autónomos.
    Orquesta los componentes del sistema y gestiona el ciclo de vida.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("AgentFramework")
        self.config = config or {}

        # Core Components
        self.registry = AgentRegistry() # <--- INSTANCIA DE LA CLASE EXTRAÍDA
        self.message_bus = MessageBus(self) # Pasa self para que pueda acceder al registry
        self.resource_manager = ResourceManager()
        self.agent_factory = AgentFactory(self) # Pasa self para que pueda usar registry y message_bus
        self.is_running = False
        self.health_check_task: Optional[asyncio.Task] = None
        self.logger.info("AgentFramework initialized.")

    async def start(self):
        """Inicia todos los componentes del framework."""
        if self.is_running:
            self.logger.warning("Framework is already running.")
            return

        self.logger.info("Starting AgentFramework components...")
        
        # Iniciar MessageBus primero, ya que es fundamental para la comunicación
        await self.message_bus.start()

        # Iniciar tareas de monitoreo de salud (si aplica)
        self.health_check_task = asyncio.create_task(self._run_health_checks())

        self.is_running = True
        self.logger.info("AgentFramework started successfully.")

    async def stop(self):
        """Detiene todos los componentes del framework y los agentes."""
        if not self.is_running:
            self.logger.warning("Framework is not running.")
            return

        self.logger.info("Stopping AgentFramework components...")

        # Detener todos los agentes registrados
        agents_to_stop = list(self.registry.list_all_agents()) # Obtener una copia para evitar cambios durante la iteración
        for agent in agents_to_stop:
            try:
                await agent.stop()
            except Exception as e:
                self.logger.error(f"Error stopping agent {agent.name}: {e}")

        # Cancelar tareas de salud
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                self.logger.info("Health check task cancelled.")
        
        # Detener MessageBus
        await self.message_bus.stop()

        self.is_running = False
        self.logger.info("AgentFramework stopped.")

    async def _run_health_checks(self):
        """Tarea periódica para verificar la salud de los agentes."""
        try:
            while self.is_running:
                # self.logger.debug("Running health checks...")
                for agent in self.registry.list_all_agents():
                    if (datetime.now() - agent.last_heartbeat).total_seconds() > 60: # Agentes sin heartbeat por 60s
                        if agent.status != AgentStatus.ERROR and agent.status != AgentStatus.TERMINATED:
                            self.logger.warning(f"Agent {agent.name} (ID: {agent.id}) is unresponsive. Setting status to ERROR.")
                            self.registry.update_agent_status(agent.id, AgentStatus.ERROR)
                            # Podríamos añadir lógica para intentar reiniciar el agente
                await asyncio.sleep(30) # Ejecutar cada 30 segundos
        except asyncio.CancelledError:
            self.logger.info("Framework health check task cancelled.")
        except Exception as e:
            self.logger.error(f"Error in framework health check: {e}", exc_info=True)


# ================================\
# DEMO / EXAMPLE USAGE
# ================================\

async def example_usage():
    """Ejemplo de uso del framework."""
    framework = AgentFramework()
    try:
        await framework.start()

        # Crear ecosistema de agentes
        agents = await AgentFactory.create_agent_ecosystem(framework) # Se usa el factory del framework
        
        # Ejemplo de comunicación entre agentes
        strategist = agents.get('strategist')
        workflow_designer = agents.get('workflow_designer')
        
        if strategist and workflow_designer:
            # El estratega solicita un workflow al diseñador
            message_id = await strategist.send_message(
                workflow_designer.id,
                "action.create.workflow",
                {
                    "tasks": ["analyze_requirements", "design_architecture", "implement_features"],
                    "priority": "high"
                }
            )
            
            print(f"Message sent: {message_id}")
            
            # Esperar un poco para ver la comunicación
            await asyncio.sleep(2)
            
            # El estratega puede crear un nuevo agente especializado
            # Necesitamos importar BuildAgent si la vamos a usar aquí directamente
            from specialized_agents import BuildAgent # type: ignore
            new_agent = await strategist.create_agent(
                "agent.test",
                "unit_tester",
                BuildAgent  # Usando BuildAgent como ejemplo
            )
            
            if new_agent:
                print(f"New agent created: {new_agent.id}")
                
            # Listar todos los agentes
            all_agents = framework.registry.list_all_agents()
            print(f"Total agents: {len(all_agents)}")
            for agent in all_agents:
                print(f"  - {agent.namespace}.{agent.name} ({agent.id}) - {agent.status.value}")
        else:
            print("Could not retrieve strategist or workflow_designer agent. Check agent creation.")
            
    finally:
        await framework.stop()

if __name__ == "__main__":
    asyncio.run(example_usage())