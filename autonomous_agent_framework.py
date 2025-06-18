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
import weakref
from contextlib import asynccontextmanager
import traceback

# ================================
# CORE FRAMEWORK CLASSES
# ================================

class AgentStatus(Enum):
    """Estados de los agentes"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"

class MessageType(Enum):
    """Tipos de mensajes entre agentes"""
    COMMAND = "command"
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

class ResourceType(Enum):
    """Tipos de recursos"""
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
    """Mensaje entre agentes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.REQUEST
    action: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    response_required: bool = True

@dataclass
class AgentResource:
    """Recurso gestionado por agentes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ResourceType = ResourceType.CODE
    name: str = ""
    namespace: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    owner_agent_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentCapability:
    """Capacidad específica de un agente"""
    name: str
    namespace: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    handler: Callable

# ================================
# AGENT REGISTRY & DISCOVERY
# ================================

class AgentRegistry:
    """Registro centralizado de agentes"""
    
    def __init__(self):
        self._agents: Dict[str, 'BaseAgent'] = {}
        self._namespaces: Dict[str, List[str]] = {}
        self._capabilities: Dict[str, List[AgentCapability]] = {}
        
    def register_agent(self, agent: 'BaseAgent') -> bool:
        """Registrar un agente"""
        if agent.id in self._agents:
            return False
            
        self._agents[agent.id] = agent
        
        # Registrar namespace
        if agent.namespace not in self._namespaces:
            self._namespaces[agent.namespace] = []
        self._namespaces[agent.namespace].append(agent.id)
        
        # Registrar capacidades
        self._capabilities[agent.id] = agent.capabilities
        
        logging.info(f"Agent {agent.id} ({agent.namespace}) registered")
        return True
        
    def unregister_agent(self, agent_id: str) -> bool:
        """Desregistrar un agente"""
        if agent_id not in self._agents:
            return False
            
        agent = self._agents[agent_id]
        
        # Remover de namespace
        if agent.namespace in self._namespaces:
            if agent_id in self._namespaces[agent.namespace]:
                self._namespaces[agent.namespace].remove(agent_id)
                
        # Remover capacidades
        if agent_id in self._capabilities:
            del self._capabilities[agent_id]
            
        del self._agents[agent_id]
        logging.info(f"Agent {agent_id} unregistered")
        return True
        
    def find_agents_by_namespace(self, namespace: str) -> List['BaseAgent']:
        """Encontrar agentes por namespace"""
        if namespace not in self._namespaces:
            return []
        return [self._agents[agent_id] for agent_id in self._namespaces[namespace]]
        
    def find_agents_by_capability(self, capability_name: str) -> List['BaseAgent']:
        """Encontrar agentes por capacidad"""
        matching_agents = []
        for agent_id, capabilities in self._capabilities.items():
            if any(cap.name == capability_name for cap in capabilities):
                matching_agents.append(self._agents[agent_id])
        return matching_agents
        
    def get_agent(self, agent_id: str) -> Optional['BaseAgent']:
        """Obtener agente por ID"""
        return self._agents.get(agent_id)
        
    def list_all_agents(self) -> List['BaseAgent']:
        """Listar todos los agentes"""
        return list(self._agents.values())

# ================================
# MESSAGE BUS & COMMUNICATION
# ================================

class MessageBus:
    """Bus de mensajes para comunicación entre agentes"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
    async def start(self):
        """Iniciar el bus de mensajes"""
        self._running = True
        asyncio.create_task(self._process_messages())
        
    async def stop(self):
        """Detener el bus de mensajes"""
        self._running = False
        
    def subscribe(self, agent_id: str, handler: Callable):
        """Suscribir un agente al bus"""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(handler)
        
    def unsubscribe(self, agent_id: str):
        """Desuscribir un agente del bus"""
        if agent_id in self._subscribers:
            del self._subscribers[agent_id]
            
    async def publish(self, message: AgentMessage):
        """Publicar un mensaje"""
        await self._message_queue.put(message)
        
    async def _process_messages(self):
        """Procesar mensajes en cola"""
        while self._running:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self._deliver_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error processing message: {e}")
                
    async def _deliver_message(self, message: AgentMessage):
        """Entregar mensaje al agente destinatario"""
        if message.receiver_id in self._subscribers:
            for handler in self._subscribers[message.receiver_id]:
                try:
                    await handler(message)
                except Exception as e:
                    logging.error(f"Error delivering message to {message.receiver_id}: {e}")

# ================================
# RESOURCE MANAGER
# ================================

class ResourceManager:
    """Gestor de recursos del framework"""
    
    def __init__(self):
        self._resources: Dict[str, AgentResource] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        
    async def create_resource(self, resource: AgentResource) -> bool:
        """Crear un nuevo recurso"""
        if resource.id in self._resources:
            return False
            
        self._resources[resource.id] = resource
        self._locks[resource.id] = asyncio.Lock()
        logging.info(f"Resource {resource.id} created by agent {resource.owner_agent_id}")
        return True
        
    async def delete_resource(self, resource_id: str, requesting_agent_id: str) -> bool:
        """Eliminar un recurso"""
        if resource_id not in self._resources:
            return False
            
        resource = self._resources[resource_id]
        if resource.owner_agent_id != requesting_agent_id:
            # Verificar permisos adicionales aquí
            return False
            
        del self._resources[resource_id]
        del self._locks[resource_id]
        logging.info(f"Resource {resource_id} deleted by agent {requesting_agent_id}")
        return True
        
    async def get_resource(self, resource_id: str) -> Optional[AgentResource]:
        """Obtener un recurso"""
        return self._resources.get(resource_id)
        
    async def update_resource(self, resource_id: str, data: Dict[str, Any], 
                            requesting_agent_id: str) -> bool:
        """Actualizar un recurso"""
        if resource_id not in self._resources:
            return False
            
        async with self._locks[resource_id]:
            resource = self._resources[resource_id]
            resource.data.update(data)
            resource.updated_at = datetime.now()
            return True
            
    def find_resources_by_type(self, resource_type: ResourceType) -> List[AgentResource]:
        """Encontrar recursos por tipo"""
        return [res for res in self._resources.values() if res.type == resource_type]
        
    def find_resources_by_owner(self, owner_agent_id: str) -> List[AgentResource]:
        """Encontrar recursos por propietario"""
        return [res for res in self._resources.values() if res.owner_agent_id == owner_agent_id]

# ================================
# BASE AGENT CLASS
# ================================

class BaseAgent(ABC):
    """Clase base para todos los agentes"""
    
    def __init__(self, namespace: str, name: str, framework: 'AgentFramework'):
        self.id: str = str(uuid.uuid4())
        self.namespace: str = namespace
        self.name: str = name
        self.status: AgentStatus = AgentStatus.INITIALIZING
        self.framework: 'AgentFramework' = framework
        self.capabilities: List[AgentCapability] = []
        self.created_at: datetime = datetime.now()
        self.last_heartbeat: datetime = datetime.now()
        self.metadata: Dict[str, Any] = {}
        self._tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Auto-registro en el framework
        framework.registry.register_agent(self)
        framework.message_bus.subscribe(self.id, self._handle_message)
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Inicializar el agente"""
        pass
        
    @abstractmethod
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar una acción específica"""
        pass
        
    async def start(self):
        """Iniciar el agente"""
        try:
            success = await self.initialize()
            if success:
                self.status = AgentStatus.ACTIVE
                # Iniciar tareas en background
                self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
                logging.info(f"Agent {self.id} ({self.namespace}.{self.name}) started")
            else:
                self.status = AgentStatus.ERROR
                logging.error(f"Failed to initialize agent {self.id}")
        except Exception as e:
            self.status = AgentStatus.ERROR
            logging.error(f"Error starting agent {self.id}: {e}")
            
    async def stop(self):
        """Detener el agente"""
        self._shutdown_event.set()
        self.status = AgentStatus.TERMINATED
        
        # Cancelar tareas
        for task in self._tasks:
            task.cancel()
            
        # Desregistrar del framework
        self.framework.registry.unregister_agent(self.id)
        self.framework.message_bus.unsubscribe(self.id)
        
        logging.info(f"Agent {self.id} stopped")
        
    async def send_message(self, receiver_id: str, action: str, payload: Dict[str, Any] = None,
                          message_type: MessageType = MessageType.REQUEST) -> str:
        """Enviar mensaje a otro agente"""
        message = AgentMessage(
            sender_id=self.id,
            receiver_id=receiver_id,
            message_type=message_type,
            action=action,
            payload=payload or {}
        )
        
        await self.framework.message_bus.publish(message)
        return message.id
        
    async def broadcast_message(self, action: str, payload: Dict[str, Any] = None,
                              namespace_filter: str = None):
        """Enviar mensaje a múltiples agentes"""
        agents = self.framework.registry.list_all_agents()
        if namespace_filter:
            agents = [a for a in agents if a.namespace == namespace_filter]
            
        for agent in agents:
            if agent.id != self.id:  # No enviarse a sí mismo
                await self.send_message(agent.id, action, payload, MessageType.EVENT)
                
    async def create_agent(self, namespace: str, name: str, agent_class: Type['BaseAgent'],
                          **kwargs) -> Optional['BaseAgent']:
        """Crear un nuevo agente"""
        try:
            new_agent = agent_class(namespace, name, self.framework, **kwargs)
            await new_agent.start()
            
            # Notificar creación
            await self.broadcast_message("agent.created", {
                "agent_id": new_agent.id,
                "namespace": namespace,
                "name": name,
                "creator_id": self.id
            })
            
            return new_agent
        except Exception as e:
            logging.error(f"Error creating agent: {e}")
            return None
            
    async def terminate_agent(self, agent_id: str) -> bool:
        """Terminar un agente"""
        agent = self.framework.registry.get_agent(agent_id)
        if not agent:
            return False
            
        await agent.stop()
        
        # Notificar terminación
        await self.broadcast_message("agent.terminated", {
            "agent_id": agent_id,
            "terminator_id": self.id
        })
        
        return True
        
    async def _handle_message(self, message: AgentMessage):
        """Manejar mensajes recibidos"""
        try:
            self.status = AgentStatus.BUSY
            
            if message.action.startswith("action."):
                # Ejecutar acción
                result = await self.execute_action(
                    message.action.replace("action.", ""),
                    message.payload
                )
                
                # Enviar respuesta si se requiere
                if message.response_required:
                    await self.send_message(
                        message.sender_id,
                        "response",
                        {"result": result, "correlation_id": message.id},
                        MessageType.RESPONSE
                    )
                    
            elif message.action == "ping":
                await self.send_message(
                    message.sender_id,
                    "pong",
                    {"agent_id": self.id, "status": self.status.value},
                    MessageType.RESPONSE
                )
                
            elif message.action == "shutdown":
                await self.stop()
                
            else:
                # Manejar acciones personalizadas
                await self._handle_custom_action(message)
                
        except Exception as e:
            logging.error(f"Error handling message in agent {self.id}: {e}")
            # Enviar mensaje de error
            if message.response_required:
                await self.send_message(
                    message.sender_id,
                    "error",
                    {"error": str(e), "correlation_id": message.id},
                    MessageType.ERROR
                )
        finally:
            self.status = AgentStatus.ACTIVE
            
    async def _handle_custom_action(self, message: AgentMessage):
        """Manejar acciones personalizadas (sobrescribir en subclases)"""
        pass
        
    async def _heartbeat_loop(self):
        """Loop de heartbeat"""
        while not self._shutdown_event.is_set():
            try:
                self.last_heartbeat = datetime.now()
                await self.send_message(
                    "framework.monitor",
                    "heartbeat",
                    {
                        "agent_id": self.id,
                        "status": self.status.value,
                        "timestamp": self.last_heartbeat.isoformat()
                    },
                    MessageType.HEARTBEAT
                )
                await asyncio.sleep(30)  # Heartbeat cada 30 segundos
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Heartbeat error for agent {self.id}: {e}")

# ================================
# SPECIALIZED AGENT TYPES
# ================================

class PlanningAgent(BaseAgent):
    """Agente especializado en planificación"""
    
    def __init__(self, name: str, framework: 'AgentFramework'):
        super().__init__("agent.planning", name, framework)
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="strategist",
                namespace="agent.planning.strategist",
                description="Create strategic plans",
                input_schema={"requirements": "string", "constraints": "object"},
                output_schema={"strategy": "object"},
                handler=self._handle_strategist
            ),
            AgentCapability(
                name="workflow",
                namespace="agent.planning.workflow",
                description="Design workflows",
                input_schema={"tasks": "array"},
                output_schema={"workflow": "object"},
                handler=self._handle_workflow
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "create.strategy":
            return await self._handle_strategist(params)
        elif action == "create.workflow":
            return await self._handle_workflow(params)
        else:
            return {"error": f"Unknown action: {action}"}
            
    async def _handle_strategist(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Implementar lógica de estrategia
        return {
            "strategy": {
                "phases": ["analysis", "design", "implementation", "testing"],
                "timeline": "4 weeks",
                "resources_needed": ["developers", "testers", "infrastructure"]
            }
        }
        
    async def _handle_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Implementar lógica de workflow
        tasks = params.get("tasks", [])
        return {
            "workflow": {
                "steps": [{"task": task, "dependencies": []} for task in tasks],
                "parallel_execution": True
            }
        }

class BuildAgent(BaseAgent):
    """Agente especializado en construcción/desarrollo"""
    
    def __init__(self, name: str, framework: 'AgentFramework'):
        super().__init__("agent.build", name, framework)
        
    async def initialize(self) -> bool:
        self.capabilities = [
            AgentCapability(
                name="code.generator",
                namespace="agent.build.code.generator",
                description="Generate code components",
                input_schema={"specification": "object"},
                output_schema={"code": "string"},
                handler=self._handle_code_generation
            ),
            AgentCapability(
                name="code.refactorer",
                namespace="agent.build.code.refactorer",
                description="Refactor existing code",
                input_schema={"code": "string", "improvements": "array"},
                output_schema={"refactored_code": "string"},
                handler=self._handle_code_refactoring
            )
        ]
        return True
        
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "generate.code":
            return await self._handle_code_generation(params)
        elif action == "refactor.code":
            return await self._handle_code_refactoring(params)
        else:
            return {"error": f"Unknown action: {action}"}
            
    async def _handle_code_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Implementar generación de código
        spec = params.get("specification", {})
        return {
            "code": f"# Generated code for {spec.get('name', 'component')}\nclass GeneratedComponent:\n    pass"
        }
        
    async def _handle_code_refactoring(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Implementar refactoring
        code = params.get("code", "")
        return {
            "refactored_code": f"# Refactored:\n{code}\n# End refactoring"
        }

# ================================
# FRAMEWORK ORCHESTRATOR
# ================================

class AgentFramework:
    """Framework principal para gestión de agentes"""
    
    def __init__(self):
        self.registry = AgentRegistry()
        self.message_bus = MessageBus()
        self.resource_manager = ResourceManager()
        self._running = False
        self._monitor_task = None
        
    async def start(self):
        """Iniciar el framework"""
        await self.message_bus.start()
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_agents())
        logging.info("Agent Framework started")
        
    async def stop(self):
        """Detener el framework"""
        self._running = False
        
        # Detener todos los agentes
        agents = self.registry.list_all_agents()
        for agent in agents:
            await agent.stop()
            
        await self.message_bus.stop()
        
        if self._monitor_task:
            self._monitor_task.cancel()
            
        logging.info("Agent Framework stopped")
        
    async def create_agent(self, agent_class: Type[BaseAgent], name: str, **kwargs) -> BaseAgent:
        """Crear un nuevo agente"""
        agent = agent_class(name, self, **kwargs)
        await agent.start()
        return agent
        
    async def _monitor_agents(self):
        """Monitorear salud de agentes"""
        while self._running:
            try:
                agents = self.registry.list_all_agents()
                now = datetime.now()
                
                for agent in agents:
                    # Verificar si el agente está respondiendo
                    time_since_heartbeat = (now - agent.last_heartbeat).total_seconds()
                    if time_since_heartbeat > 120:  # 2 minutos sin heartbeat
                        logging.warning(f"Agent {agent.id} seems unresponsive")
                        # Intentar reiniciar o marcar como error
                        agent.status = AgentStatus.ERROR
                        
                await asyncio.sleep(60)  # Verificar cada minuto
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Monitor error: {e}")

# ================================
# FACTORY METHODS
# ================================

class AgentFactory:
    """Factory para crear agentes especializados"""
    
    @staticmethod
    def create_planning_agent(framework: AgentFramework, name: str = "planner") -> PlanningAgent:
        return PlanningAgent(name, framework)
        
    @staticmethod
    def create_build_agent(framework: AgentFramework, name: str = "builder") -> BuildAgent:
        return BuildAgent(name, framework)
        
    @staticmethod
    async def create_agent_ecosystem(framework: AgentFramework) -> Dict[str, BaseAgent]:
        """Crear un ecosistema básico de agentes"""
        agents = {}
        
        # Crear agentes de planificación
        agents['strategist'] = AgentFactory.create_planning_agent(framework, "strategist")
        agents['workflow_designer'] = AgentFactory.create_planning_agent(framework, "workflow_designer")
        
        # Crear agentes de construcción
        agents['code_generator'] = AgentFactory.create_build_agent(framework, "code_generator")
        agents['refactorer'] = AgentFactory.create_build_agent(framework, "refactorer")
        
        # Iniciar todos los agentes
        for agent in agents.values():
            await agent.start()
            
        return agents

# ================================
# EXAMPLE USAGE
# ================================

async def example_usage():
    """Ejemplo de uso del framework"""
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear framework
    framework = AgentFramework()
    await framework.start()
    
    try:
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
        
        print(f"Message sent: {message_id}")
        
        # Esperar un poco para ver la comunicación
        await asyncio.sleep(2)
        
        # El estratega puede crear un nuevo agente especializado
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
            
    finally:
        await framework.stop()

if __name__ == "__main__":
    asyncio.run(example_usage())