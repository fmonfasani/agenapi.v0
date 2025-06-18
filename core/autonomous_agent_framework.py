import asyncio
import uuid
import json
import logging
from typing import Dict, List, Any, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
import weakref
from contextlib import asynccontextmanager
import traceback

# logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CORE FRAMEWORK CLASSES
# ================================

class AgentStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"

class MessageType(Enum):
    COMMAND = "command"
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

class ResourceType(Enum):
    CODE = "code"
    INFRA = "infra"
    WORKFLOW = "workflow"
    UI = "ui"
    DATA = "data"
    TEST = "test"
    SECURITY = "security"
    RELEASE = "release"

@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.INFO
    action: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "sent" # sent, delivered, read, failed
    correlation_id: Optional[str] = None # Para correlacionar requests con responses
    expires_at: Optional[datetime] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "action": self.action,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "correlation_id": self.correlation_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            id=data["id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType(data["message_type"]),
            action=data["action"],
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            status=data["status"],
            correlation_id=data.get("correlation_id"),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
        )

@dataclass
class AgentResource:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    namespace: str = "" # e.g., "resource.data.user_profiles"
    type: ResourceType = ResourceType.DATA
    data: Any = None # Puede ser un diccionario, una cadena, etc.
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    owner_agent_id: Optional[str] = None
    version: str = "1.0.0"
    tags: Dict[str, str] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list) # e.g., ["read", "write"]

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "namespace": self.namespace,
            "type": self.type.value,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner_agent_id": self.owner_agent_id,
            "version": self.version,
            "tags": self.tags,
            "permissions": self.permissions
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            id=data["id"],
            name=data["name"],
            namespace=data["namespace"],
            type=ResourceType(data["type"]),
            data=data["data"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            owner_agent_id=data["owner_agent_id"],
            version=data["version"],
            tags=data["tags"],
            permissions=data["permissions"]
        )

@dataclass
class AgentCapability:
    name: str
    namespace: str # e.g., "agent.skill.coding.generate_code"
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    handler: Callable # Función asíncrona que implementa la capacidad
    permissions_required: Set[str] = field(default_factory=set) # Permisos necesarios para ejecutar esta capacidad

class BaseAgent(ABC):
    def __init__(self, namespace: str, name: str, framework: 'AgentFramework'):
        self.id: str = str(uuid.uuid4())
        self.namespace: str = namespace
        self.name: str = name
        self.status: AgentStatus = AgentStatus.INITIALIZING
        self.created_at: datetime = datetime.now()
        self.last_active_at: datetime = datetime.now()
        self.framework: 'AgentFramework' = weakref.proxy(framework) # Evitar referencia circular
        self.capabilities: List[AgentCapability] = []
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.event_bus_listener_task: Optional[asyncio.Task] = None
        self.resource_ownership: Set[str] = set() # IDs de recursos que este agente posee

    async def initialize(self) -> bool:
        self.status = AgentStatus.ACTIVE
        logger.info(f"Agent {self.namespace}.{self.name} ({self.id}) initialized and set to ACTIVE.")
        self.event_bus_listener_task = asyncio.create_task(self._event_bus_listener())
        return True

    async def _event_bus_listener(self):
        try:
            while True:
                message = await self.message_queue.get()
                self.last_active_at = datetime.now()
                await self.handle_message(message)
                self.message_queue.task_done()
        except asyncio.CancelledError:
            logger.info(f"Agent {self.name} message listener cancelled.")
        except Exception as e:
            logger.error(f"Error in agent {self.name} message listener: {e}")
            self.status = AgentStatus.ERROR

    async def send_message(self, receiver_id: str, action: str, payload: Dict[str, Any], message_type: MessageType = MessageType.COMMAND, correlation_id: Optional[str] = None) -> str:
        message = AgentMessage(
            sender_id=self.id,
            receiver_id=receiver_id,
            message_type=message_type,
            action=action,
            payload=payload,
            correlation_id=correlation_id
        )
        await self.framework.message_bus.send_message(message)
        logger.debug(f"Agent {self.name} sent message {message.id} to {receiver_id} with action {action}.")
        return message.id

    async def receive_message(self, message: AgentMessage):
        await self.message_queue.put(message)
        logger.debug(f"Agent {self.name} received message {message.id} from {message.sender_id}.")

    @abstractmethod
    async def handle_message(self, message: AgentMessage):
        pass

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        capability = next((cap for cap in self.capabilities if cap.name == action or cap.namespace == action), None)
        if capability:
            # Aquí se integraría la seguridad: verificar permisos antes de ejecutar
            # if not self.framework.security_manager.check_permissions(self.id, capability.permissions_required):
            #     raise Exception("Permission denied")
            try:
                self.status = AgentStatus.BUSY
                result = await capability.handler(params)
                self.status = AgentStatus.ACTIVE
                logger.info(f"Agent {self.name} executed action {action}.")
                return result
            except Exception as e:
                logger.error(f"Error executing action {action} for agent {self.name}: {e}\n{traceback.format_exc()}")
                self.status = AgentStatus.ERROR
                return {"error": str(e)}
        else:
            logger.warning(f"Agent {self.name} received request for unknown action: {action}")
            return {"error": f"Unknown action: {action}"}
            
    async def create_agent(self, namespace: str, name: str, agent_class: Type['BaseAgent'], initial_params: Optional[Dict[str, Any]] = None) -> Optional['BaseAgent']:
        """ Permite a un agente crear y registrar un nuevo agente. """
        new_agent = await self.framework.agent_factory.create_agent(namespace, name, agent_class, initial_params)
        if new_agent:
            logger.info(f"Agent {self.name} created new agent: {new_agent.namespace}.{new_agent.name} ({new_agent.id})")
        return new_agent

    async def shutdown(self):
        self.status = AgentStatus.TERMINATED
        if self.event_bus_listener_task:
            self.event_bus_listener_task.cancel()
            try:
                await self.event_bus_listener_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Agent {self.namespace}.{self.name} ({self.id}) shut down.")

class MessageBus:
    def __init__(self, registry: 'AgentRegistry'):
        self.registry = registry
        self.message_log: List[AgentMessage] = [] # Para auditoría y persistencia
        self.message_handlers: Dict[str, Callable] = {} # receiver_id -> handler_func

    async def send_message(self, message: AgentMessage):
        self.message_log.append(message)
        receiver = self.registry.get_agent(message.receiver_id)
        if receiver:
            await receiver.receive_message(message)
            message.status = "delivered"
            logger.debug(f"Message {message.id} delivered to {receiver.name}.")
        else:
            message.status = "failed"
            logger.warning(f"Message {message.id} failed: Receiver {message.receiver_id} not found.")

    def register_handler(self, agent_id: str, handler: Callable):
        self.message_handlers[agent_id] = handler

    def unregister_handler(self, agent_id: str):
        self.message_handlers.pop(agent_id, None)

    def get_message_history(self, agent_id: Optional[str] = None, limit: int = 100) -> List[AgentMessage]:
        if agent_id:
            return [msg for msg in self.message_log if msg.sender_id == agent_id or msg.receiver_id == agent_id][-limit:]
        return self.message_log[-limit:]

class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {} # id -> agent_instance
        self._namespaces: Dict[str, List[str]] = {} # namespace -> [id1, id2, ...]
        logger.info("AgentRegistry initialized.")

    def register_agent(self, agent: BaseAgent):
        if agent.id in self._agents:
            logger.warning(f"Agent with ID {agent.id} already registered. Updating.")
        self._agents[agent.id] = agent
        if agent.namespace not in self._namespaces:
            self._namespaces[agent.namespace] = []
        if agent.id not in self._namespaces[agent.namespace]:
            self._namespaces[agent.namespace].append(agent.id)
        logger.info(f"Agent {agent.namespace}.{agent.name} ({agent.id}) registered.")

    def unregister_agent(self, agent_id: str):
        agent = self._agents.pop(agent_id, None)
        if agent:
            if agent.namespace in self._namespaces:
                self._namespaces[agent.namespace].remove(agent.id)
                if not self._namespaces[agent.namespace]:
                    del self._namespaces[agent.namespace]
            logger.info(f"Agent {agent.name} ({agent_id}) unregistered.")
            return True
        logger.warning(f"Attempted to unregister non-existent agent with ID: {agent_id}.")
        return False

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        return self._agents.get(agent_id)

    def get_agents_by_namespace(self, namespace: str) -> List[BaseAgent]:
        agent_ids = self._namespaces.get(namespace, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def list_all_agents(self) -> List[BaseAgent]:
        return list(self._agents.values())
        
    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        agent = self.get_agent(agent_id)
        return agent.status if agent else None

class ResourceManager:
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._resources: Dict[str, AgentResource] = {} # id -> resource_instance
        self._resource_by_namespace: Dict[str, List[str]] = {} # namespace -> [id1, id2, ...]
        logger.info("ResourceManager initialized.")

    async def create_resource(self, resource: AgentResource) -> bool:
        if resource.id in self._resources:
            logger.warning(f"Resource with ID {resource.id} already exists. Use update_resource.")
            return False
        self._resources[resource.id] = resource
        if resource.namespace not in self._resource_by_namespace:
            self._resource_by_namespace[resource.namespace] = []
        self._resource_by_namespace[resource.namespace].append(resource.id)
        
        if resource.owner_agent_id:
            owner_agent = self.registry.get_agent(resource.owner_agent_id)
            if owner_agent:
                owner_agent.resource_ownership.add(resource.id)
        
        logger.info(f"Resource {resource.name} ({resource.id}) created under namespace {resource.namespace}.")
        return True

    async def get_resource(self, resource_id: str) -> Optional[AgentResource]:
        return self._resources.get(resource_id)

    async def update_resource(self, resource: AgentResource) -> bool:
        if resource.id not in self._resources:
            logger.warning(f"Resource with ID {resource.id} not found for update.")
            return False
        self._resources[resource.id] = resource
        resource.updated_at = datetime.now()
        logger.info(f"Resource {resource.name} ({resource.id}) updated.")
        return True

    async def delete_resource(self, resource_id: str) -> bool:
        resource = self._resources.pop(resource_id, None)
        if resource:
            if resource.namespace in self._resource_by_namespace:
                self._resource_by_namespace[resource.namespace].remove(resource.id)
                if not self._resource_by_namespace[resource.namespace]:
                    del self._resource_by_namespace[resource.namespace]
            
            if resource.owner_agent_id:
                owner_agent = self.registry.get_agent(resource.owner_agent_id)
                if owner_agent and resource.id in owner_agent.resource_ownership:
                    owner_agent.resource_ownership.remove(resource.id)

            logger.info(f"Resource {resource.name} ({resource.id}) deleted.")
            return True
        logger.warning(f"Attempted to delete non-existent resource with ID: {resource_id}.")
        return False
        
    def find_resources_by_owner(self, owner_agent_id: str) -> List[AgentResource]:
        return [res for res in self._resources.values() if res.owner_agent_id == owner_agent_id]

    def find_resources_by_type(self, resource_type: ResourceType) -> List[AgentResource]:
        return [res for res in self._resources.values() if res.type == resource_type]

    def list_all_resources(self) -> List[AgentResource]:
        return list(self._resources.values())

class AgentFactory:
    def __init__(self, framework: 'AgentFramework'):
        self.framework = framework
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}
        logger.info("AgentFactory initialized.")

    def register_agent_class(self, namespace: str, agent_class: Type[BaseAgent]):
        self._agent_classes[namespace] = agent_class
        logger.info(f"Agent class for namespace '{namespace}' registered.")

    async def create_agent(self, namespace: str, name: str, agent_class: Optional[Type[BaseAgent]] = None, initial_params: Optional[Dict[str, Any]] = None) -> Optional[BaseAgent]:
        if agent_class is None:
            agent_class = self._agent_classes.get(namespace)
            if not agent_class:
                logger.error(f"No agent class registered for namespace: {namespace}")
                return None
        
        agent = agent_class(name, self.framework)
        await agent.initialize()
        self.framework.registry.register_agent(agent)
        logger.info(f"Agent {name} of type {namespace} created with ID {agent.id}.")
        return agent
        
    async def create_agent_ecosystem(self, framework: 'AgentFramework') -> Dict[str, BaseAgent]:
        """ Crea un conjunto de agentes predefinidos para una demostración o caso de uso. """
        from specialized_agents import StrategistAgent, WorkflowDesignerAgent, CodeGeneratorAgent, TestGeneratorAgent, BuildAgent # Importación local para evitar circular

        self.register_agent_class("agent.planning.strategist", StrategistAgent)
        self.register_agent_class("agent.design.workflow", WorkflowDesignerAgent)
        self.register_agent_class("agent.development.code_generator", CodeGeneratorAgent)
        self.register_agent_class("agent.development.test_generator", TestGeneratorAgent)
        self.register_agent_class("agent.devops.build", BuildAgent)

        agents = {}
        agents['strategist'] = await self.create_agent("agent.planning.strategist", "MainStrategist")
        agents['workflow_designer'] = await self.create_agent("agent.design.workflow", "SystemWorkflowDesigner")
        agents['code_generator'] = await self.create_agent("agent.development.code_generator", "BackendCodeGenerator")
        agents['test_generator'] = await self.create_agent("agent.development.test_generator", "UnitTestGenerator")
        agents['builder'] = await self.create_agent("agent.devops.build", "CIBuilder")
        
        logger.info(f"Agent ecosystem created with {len(agents)} agents.")
        return agents

class AgentFramework:
    def __init__(self, name: str = "DefaultFramework"):
        self.name = name
        self.registry: AgentRegistry = AgentRegistry()
        self.message_bus: MessageBus = MessageBus(self.registry)
        self.resource_manager: ResourceManager = ResourceManager(self.registry)
        self.agent_factory: AgentFactory = AgentFactory(self)
        self.is_running: bool = False
        self.start_time: datetime = datetime.now()
        logger.info(f"AgentFramework '{self.name}' initialized.")

    async def start(self):
        if self.is_running:
            logger.warning("Framework is already running.")
            return

        self.is_running = True
        logger.info(f"AgentFramework '{self.name}' started.")

    async def stop(self):
        if not self.is_running:
            logger.warning("Framework is not running.")
            return

        logger.info(f"Stopping AgentFramework '{self.name}'...")
        
        # Shut down all agents gracefully
        agents_to_shutdown = list(self.registry.list_all_agents())
        for agent in agents_to_shutdown:
            await agent.shutdown()
            self.registry.unregister_agent(agent.id) # Unregister after shutdown

        self.is_running = False
        logger.info(f"AgentFramework '{self.name}' stopped.")

    @asynccontextmanager
    async def run_until_complete(self):
        await self.start()
        try:
            yield
        finally:
            await self.stop()

# ================================
# DEMO / USAGE EXAMPLE
# ================================

# Definir un agente simple para la demostración
class BuildAgent(BaseAgent):
    def __init__(self, name: str, framework: AgentFramework):
        super().__init__("agent.devops.build", name, framework)
        self.capabilities = [
            AgentCapability(
                name="perform_build",
                namespace="agent.devops.build.perform",
                description="Performs a simulated software build.",
                input_schema={"project_name": "string", "version": "string"},
                output_schema={"status": "string", "build_log": "string"},
                handler=self._perform_build
            )
        ]

    async def handle_message(self, message: AgentMessage):
        logger.info(f"BuildAgent {self.name} received message: {message.action} from {message.sender_id}")
        if message.action == "action.perform.build":
            result = await self.execute_action("perform_build", message.payload)
            await self.send_message(
                message.sender_id,
                "response.perform.build",
                result,
                MessageType.RESPONSE,
                correlation_id=message.id
            )
        else:
            logger.warning(f"BuildAgent {self.name} received unhandled action: {message.action}")

    async def _perform_build(self, params: Dict[str, Any]) -> Dict[str, Any]:
        project_name = params.get("project_name", "unknown")
        version = params.get("version", "1.0.0")
        logger.info(f"Simulating build for {project_name} version {version}...")
        await asyncio.sleep(1) # Simulate work
        build_log = f"Build of {project_name} v{version} completed successfully."
        logger.info(build_log)
        
        # Crear un recurso de tipo BUILD (artefacto)
        build_artifact = AgentResource(
            name=f"{project_name}_build_{version}",
            namespace=f"resource.build.{project_name}",
            type=ResourceType.RELEASE,
            data={"project": project_name, "version": version, "log": build_log, "status": "success"},
            owner_agent_id=self.id,
            tags={"version": version, "status": "success"}
        )
        await self.framework.resource_manager.create_resource(build_artifact)

        return {"status": "success", "build_log": build_log}


async def example_usage():
    framework = AgentFramework("DevOpsFramework")
    try:
        await framework.start()

        # Registrar el tipo de agente BuildAgent con el factory
        framework.agent_factory.register_agent_class("agent.devops.build", BuildAgent)

        # Crear ecosistema de agentes
        agents = await AgentFactory.create_agent_ecosystem(framework)
        
        # Ejemplo de comunicación entre agentes
        strategist = agents['strategist']
        workflow_designer = agents['workflow_designer']
        
        # El estratega solicita un workflow al diseñador
        message_id = await strategist.send_message(
            workflow_designer.id,
            "action.create.workflow",
            {
                "tasks": ["analyze_requirements", "design_architecture", "implement_features"],
                "priority": "high"
            }
        )
        
        logger.info(f"Message sent: {message_id}")
        
        # Esperar un poco para ver la comunicación
        await asyncio.sleep(2)
        
        # El estratega puede crear un nuevo agente especializado
        new_agent = await strategist.create_agent(
            "agent.devops.build",
            "unit_tester",
            BuildAgent  # Usando BuildAgent como ejemplo
        )
        
        if new_agent:
            logger.info(f"New agent created: {new_agent.id}")
            
        # Listar todos los agentes
        all_agents = framework.registry.list_all_agents()
        logger.info(f"Total agents: {len(all_agents)}")
        for agent in all_agents:
            logger.info(f"  - {agent.namespace}.{agent.name} ({agent.id}) - {agent.status.value}")
            
    finally:
        await framework.stop()

if __name__ == "__main__":
    asyncio.run(example_usage())