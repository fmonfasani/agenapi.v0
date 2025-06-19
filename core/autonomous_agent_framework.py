import asyncio
import logging
import uuid
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

# Import configuration
from agentapi.config import ConfigLoader
from agentapi.interfaces.agent_interfaces import BaseAgent, AgentStatus, AgentMessage, MessageType # Keep these for type hints in framework methods
from agentapi.models.framework_models import FrameworkConfig, FrameworkMetrics
from agentapi.models.general_models import Metric, Alert, AlertSeverity # Required for framework level alerts

# Import individual managers (assuming they are in framework/ directory now)
from framework.registry import AgentRegistry
from framework.message_bus import MessageBus
from framework.agent_factory import AgentFactory
from framework.resource_manager import ResourceManager
from framework.security_manager import SecurityManager
from framework.monitoring_manager import MonitoringManager # Will become MonitoringOrchestrator conceptually
from framework.persistence_manager import PersistenceManager
from framework.backup_manager import BackupManager

# Importaciones para LLM y sistema de memoria
from core.cognitive_system import AgentMemorySystem, MemoryType # <--- NUEVO
from openai import AsyncOpenAI # <--- NUEVO

# Import specialized agents for example_usage if they exist
try:
    import agents.specialized_agents
    _SPECIALIZED_AGENTS_AVAILABLE = True
except ImportError:
    _SPECIALIZED_AGENTS_AVAILABLE = False


class BaseAgent:
    """Clase base para todos los agentes del framework."""
    def __init__(self, namespace: str, name: str, framework: Any): # Any para evitar importación circular
        self.id = str(uuid.uuid4())
        self.namespace = namespace
        self.name = name
        self.framework = framework
        self.status = AgentStatus.INITIALIZING
        self.capabilities: List[Any] = [] # List of AgentCapability
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.logger = logging.getLogger(f"Agent.{namespace}.{name}")
        self.last_heartbeat = datetime.now()
        self._running_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # --- Integración de memoria ---
        self.memory: Optional[AgentMemorySystem] = None
        if self.framework and self.framework.config and self.framework.config.cognitive.enabled: # Asumir configuración de memoria
            self.memory = AgentMemorySystem(
                agent_id=self.id,
                config=self.framework.config.cognitive.memory_config,
                persistence_manager=self.framework.persistence_manager # Pasar el persistence_manager si se necesita
            )

        # --- Cliente LLM para capacidades cognitivas ---
        self.llm_client: Optional[AsyncOpenAI] = None
        if self.framework and self.framework.config and self.framework.config.cognitive.enabled:
            # Puedes usar una configuración más granular para el LLM, como modelo, temperatura, etc.
            # Por ahora, un simple cliente OpenAI
            self.llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not os.getenv("OPENAI_API_KEY"):
                self.logger.warning("OPENAI_API_KEY not set. LLM capabilities may be limited.")


    async def initialize(self) -> bool:
        """Inicializa el agente y sus componentes."""
        self.logger.info(f"Agent {self.name} ({self.id}) initializing...")
        if self.memory:
            await self.memory.add_memory(MemoryType.EPISODIC, f"Agent {self.name} initialized successfully.")
        self.status = AgentStatus.ACTIVE
        self.logger.info(f"Agent {self.name} initialized.")
        return True

    async def start(self):
        """Inicia el bucle principal del agente."""
        if self._running_task:
            self.logger.warning(f"Agent {self.name} already running.")
            return

        self.status = AgentStatus.ACTIVE
        self._running_task = asyncio.create_task(self._agent_loop())
        self.logger.info(f"Agent {self.name} started with ID: {self.id}")

    async def stop(self):
        """Detiene el agente."""
        if self._running_task:
            self.status = AgentStatus.TERMINATED
            self._shutdown_event.set()
            self._running_task.cancel()
            try:
                await self._running_task
            except asyncio.CancelledError:
                self.logger.info(f"Agent {self.name} stopped.")
            finally:
                self._running_task = None
        if self.memory:
            await self.memory.shutdown() # Asegurar que la memoria se apague correctamente
        self.logger.info(f"Agent {self.name} ({self.id}) terminated.")

    async def _agent_loop(self):
        """Bucle principal de procesamiento de mensajes del agente."""
        self.logger.info(f"Agent {self.name} starting message loop.")
        while not self._shutdown_event.is_set():
            try:
                # Esperar por mensajes con un timeout para poder chequear el evento de apagado
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                self.logger.debug(f"Agent {self.name} received message: {message.id} ({message.message_type.value}) from {message.sender_id}")
                await self.process_message(message)
                self.message_queue.task_done()
            except asyncio.TimeoutError:
                # No messages, continue checking shutdown event
                pass
            except asyncio.CancelledError:
                self.logger.info(f"Agent {self.name} message loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in agent {self.name} message loop: {e}", exc_info=True)
                self.status = AgentStatus.ERROR
                if self.memory:
                    await self.memory.add_memory(MemoryType.EPISODIC, f"Error in message loop: {e}", {"severity": "critical"})

    async def process_message(self, message: AgentMessage):
        """
        Procesa un mensaje entrante. Los agentes derivados deben sobrescribir esto.
        """
        self.last_heartbeat = datetime.now()
        if self.memory:
            await self.memory.add_memory(MemoryType.EPISODIC, f"Received message from {message.sender_id}: {message.content.get('command', message.content)}", {"message_id": message.id, "sender_id": message.sender_id})

        self.logger.info(f"Agent {self.name} processing message from {message.sender_id}: {message.message_type.value}")
        # Lógica de procesamiento de mensajes. Aquí es donde se conectaría con execute_action.
        if message.message_type == MessageType.COMMAND:
            command = message.content.get("command")
            args = message.content.get("args", {})
            if command:
                await self.execute_action(command, args, message.sender_id)
            else:
                self.logger.warning(f"Received COMMAND message without 'command' field: {message.content}")
        elif message.message_type == MessageType.REQUEST:
            # Implementar lógica para manejar solicitudes y enviar respuestas
            pass # TODO
        elif message.message_type == MessageType.RESPONSE:
            # Implementar lógica para manejar respuestas a solicitudes previas
            pass # TODO
        elif message.message_type == MessageType.EVENT:
            # Los agentes pueden reaccionar a eventos
            pass # TODO
        elif message.message_type == MessageType.HEARTBEAT:
            # Manejar latidos, actualizar el estado
            self.status = AgentStatus.ACTIVE
            self.framework.registry.update_agent_status(self.id, AgentStatus.ACTIVE, self.last_heartbeat)
            self.logger.debug(f"Agent {self.name} heartbeat received and status updated.")
        elif message.message_type == MessageType.ERROR:
            self.logger.error(f"Agent {self.name} received error message from {message.sender_id}: {message.content.get('error_message', 'Unknown Error')}")
            if self.memory:
                await self.memory.add_memory(MemoryType.EPISODIC, f"Received error from {message.sender_id}: {message.content.get('error_message', 'Unknown Error')}", {"severity": "error", "source_agent": message.sender_id})
        else:
            self.logger.warning(f"Agent {self.name} received unsupported message type: {message.message_type.value}")

    async def execute_action(self, action_name: str, args: Dict[str, Any], sender_id: str = "self") -> Any:
        """
        Ejecuta una acción específica basada en las capacidades del agente.
        """
        self.status = AgentStatus.BUSY
        self.logger.info(f"Agent {self.name} executing action: {action_name} with args: {args}")
        if self.memory:
            await self.memory.add_memory(MemoryType.EPISODIC, f"Executing action '{action_name}' with arguments: {args}", {"action": action_name, "args": args})

        try:
            # Buscar la capacidad y ejecutar su handler
            for capability in self.capabilities:
                if capability.name == action_name:
                    result = await capability.handler(args)
                    self.status = AgentStatus.ACTIVE
                    if self.memory:
                        await self.memory.add_memory(MemoryType.EPISODIC, f"Action '{action_name}' completed with result: {result}", {"action": action_name, "result": result})
                    return result
            raise ValueError(f"Action '{action_name}' not found for agent {self.name}.")
        except Exception as e:
            self.logger.error(f"Error executing action {action_name} for agent {self.name}: {e}", exc_info=True)
            self.status = AgentStatus.ERROR
            if self.memory:
                await self.memory.add_memory(MemoryType.EPISODIC, f"Error executing action '{action_name}': {e}", {"action": action_name, "error": str(e), "severity": "critical"})
            raise

    async def send_message(self, receiver_id: str, message_type: MessageType, content: Dict[str, Any]):
        """Envía un mensaje a otro agente o a broadcast."""
        message = AgentMessage(
            sender_id=self.id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content
        )
        if self.framework.message_bus:
            await self.framework.message_bus.send_message(message)
            self.logger.debug(f"Agent {self.name} sent message {message.id} to {receiver_id}.")
            if self.memory:
                await self.memory.add_memory(MemoryType.EPISODIC, f"Sent message to {receiver_id}: {content}", {"message_id": message.id, "receiver_id": receiver_id})
        else:
            self.logger.error("Message bus not available in framework.")

    # --- NUEVOS MÉTODOS COGNITIVOS ---
    async def think_step_by_step(self, prompt: str, max_steps: int = 5) -> str:
        """
        Guía al LLM a pensar paso a paso para resolver una tarea o pregunta.
        Utiliza la memoria para contextualizar la respuesta.
        """
        if not self.llm_client:
            self.logger.warning("LLM client not initialized for think_step_by_step.")
            return "Error: LLM capabilities not available."

        self.status = AgentStatus.BUSY
        thinking_process = []
        final_thought = ""

        # Recuperar memorias relevantes para el prompt
        relevant_memories = await self.memory.retrieve_memories(prompt, num_results=3, memory_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC]) if self.memory else []
        memory_context = "\n".join([f"Memory: {m.content}" for m in relevant_memories])
        if memory_context:
            memory_context = "\n\nRelevant Memories:\n" + memory_context

        initial_system_prompt = (
            "You are an intelligent agent designed to think step-by-step to solve problems. "
            "You will be given a task or question. Break it down into logical steps. "
            "At each step, provide your current thought process and what you plan to do next. "
            "Finally, provide a consolidated answer or plan. "
            "Use the provided relevant memories to inform your thoughts if applicable. If you need more information, state it."
            f"{memory_context}"
        )

        messages = [
            {"role": "system", "content": initial_system_prompt},
            {"role": "user", "content": f"Task: {prompt}\n\nBegin your step-by-step thinking:"}
        ]

        self.logger.info(f"Agent {self.name} starting step-by-step thinking for: {prompt[:50]}...")
        if self.memory:
            await self.memory.add_memory(MemoryType.EPISODIC, f"Starting step-by-step thinking for: {prompt}", {"cognitive_process": "think_step_by_step"})

        for step in range(max_steps):
            try:
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4o", # Puedes hacer esto configurable
                    messages=messages,
                    temperature=0.7,
                    max_tokens=300
                )
                step_thought = response.choices[0].message.content
                thinking_process.append(f"Step {step + 1}:\n{step_thought}\n")
                self.logger.debug(f"Agent {self.name} - Think Step {step+1}: {step_thought[:100]}...")

                messages.append({"role": "assistant", "content": step_thought})
                messages.append({"role": "user", "content": f"Continue thinking for the task: {prompt}"})

                if "Final Answer:" in step_thought or "Plan:" in step_thought or "Conclusion:" in step_thought:
                    final_thought = step_thought
                    break
            except Exception as e:
                self.logger.error(f"Error during step-by-step thinking (LLM call): {e}", exc_info=True)
                thinking_process.append(f"Error during thinking step: {e}")
                break
        
        self.status = AgentStatus.ACTIVE
        if not final_thought:
            final_thought = " ".join(thinking_process) # Concatenate if no explicit final answer
            self.logger.warning(f"No explicit final answer found in step-by-step thinking. Concatenated thoughts.")

        if self.memory:
            await self.memory.add_memory(MemoryType.EPISODIC, f"Completed step-by-step thinking for '{prompt}': {final_thought}", {"cognitive_process": "think_step_by_step", "output": final_thought})
            await self.memory.add_memory(MemoryType.PROCEDURAL, f"Learned thinking process for '{prompt}': {''.join(thinking_process)}")
            
        return final_thought

    async def self_reflect(self, context: str, problem: str) -> str:
        """
        Permite al agente reflexionar sobre una situación o problema,
        identificar errores o mejorar su enfoque.
        """
        if not self.llm_client:
            self.logger.warning("LLM client not initialized for self_reflect.")
            return "Error: LLM capabilities not available."

        self.status = AgentStatus.BUSY
        reflection_prompt = (
            "You are an intelligent agent tasked with self-reflection. "
            "Analyze the provided 'Context' and 'Problem' to identify any issues, "
            "potential improvements, or alternative approaches. "
            "Think critically and propose concrete actions or insights.\n\n"
            f"Context: {context}\n"
            f"Problem/Observation: {problem}\n\n"
            "Your Reflection and Proposed Improvements:"
        )

        self.logger.info(f"Agent {self.name} starting self-reflection for: {problem[:50]}...")
        if self.memory:
            await self.memory.add_memory(MemoryType.EPISODIC, f"Starting self-reflection for problem: {problem}", {"cognitive_process": "self_reflect"})

        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": reflection_prompt}],
                temperature=0.5,
                max_tokens=500
            )
            reflection_output = response.choices[0].message.content
            self.logger.debug(f"Agent {self.name} Reflection: {reflection_output[:200]}...")
            
            if self.memory:
                await self.memory.add_memory(MemoryType.EPISODIC, f"Completed self-reflection for '{problem}': {reflection_output}", {"cognitive_process": "self_reflect", "output": reflection_output})
                await self.memory.add_memory(MemoryType.SEMANTIC, f"Reflective insight on problem: {problem}. Insight: {reflection_output}")

            return reflection_output
        except Exception as e:
            self.logger.error(f"Error during self-reflection (LLM call): {e}", exc_info=True)
            if self.memory:
                await self.memory.add_memory(MemoryType.EPISODIC, f"Error during self-reflection: {e}", {"cognitive_process": "self_reflect", "error": str(e), "severity": "critical"})
            return f"Error during self-reflection: {e}"
        finally:
            self.status = AgentStatus.ACTIVE

# La clase AgentFramework es la que orquesta todo, debería manejar la configuración cognitiva
class AgentFramework:
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration using ConfigLoader
        self.config_loader = ConfigLoader()
        self.config: FrameworkConfig = self.config_loader.load_config(config_path)

        # Configuración de logging
        logging.basicConfig(level=getattr(logging, self.config.logging_level.upper()),
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("AgentFramework")
        self.logger.info(f"AgentFramework '{self.config.name}' (v{self.config.version}) initializing...")

        # Inicialización de managers (orden importante por dependencias)
        self.registry = AgentRegistry()
        self.message_bus = MessageBus(self.registry) # MessageBus necesita el Registry
        self.resource_manager = ResourceManager(self.registry) # ResourceManager necesita el Registry
        
        # Managers con configuración
        self.security_manager = SecurityManager(self.config.security) # Asumiendo que SecurityManager toma config
        self.persistence_manager = PersistenceManager(self.config.persistence) # PersistenceManager toma config
        self.monitoring_manager = MonitoringManager(self.config.monitoring) # MonitoringManager toma config
        self.backup_manager = BackupManager(self.config.backup, self.monitoring_manager, self.persistence_manager) # BackupManager toma config y referencias

        # Agent Factory (necesita una referencia al framework para crear agentes)
        self.agent_factory = AgentFactory(self)
        
        # Colección de agentes activos (instancias)
        self.active_agents: Dict[str, BaseAgent] = {}

        self.metrics = FrameworkMetrics() # Para tracking de métricas internas

        self.logger.info("AgentFramework initialized.")

    async def initialize(self):
        """Inicializa todos los componentes del framework."""
        self.logger.info("Initializing framework components...")
        await self.registry.initialize() # Si el registry tiene inicialización async
        await self.persistence_manager.initialize()
        await self.monitoring_manager.initialize()
        await self.security_manager.initialize() # Si el security_manager tiene inicialización async
        await self.backup_manager.initialize()
        
        # Cargar estado previo si la persistencia está habilitada
        if self.config.persistence.enabled:
            # Aquí podrías cargar información de agentes y recursos desde la persistencia
            # y reconstruir el estado si es necesario.
            # Por ahora, solo registramos los agentes y recursos configurados.
            pass

        # Crear agentes definidos en la configuración
        initial_agents = await self.agent_factory.create_initial_agents_from_config(self.config.agents)
        for agent_id, agent_instance in initial_agents.items():
            self.active_agents[agent_id] = agent_instance
            # El agent_factory ya registra y starta el agente si auto_start es True

        self.logger.info("Framework components initialized.")

    async def start(self):
        """Inicia el framework y todos sus agentes."""
        self.logger.info("Starting AgentFramework...")
        # Los agentes ya están iniciados si auto_start=True en create_initial_agents_from_config
        # Si no, podrías tener un bucle aquí para iniciarlos
        
        # Iniciar monitoreo del framework si está habilitado
        if self.config.monitoring.enabled:
            asyncio.create_task(self._framework_monitoring_loop())

        self.logger.info("AgentFramework started.")

    async def stop(self):
        """Detiene el framework y todos sus agentes."""
        self.logger.info("Stopping AgentFramework...")
        for agent_id, agent in list(self.active_agents.items()):
            await agent.stop()
            self.metrics.total_agents = len(self.active_agents) # Actualizar métrica
        
        await self.monitoring_manager.shutdown()
        await self.backup_manager.shutdown()
        await self.persistence_manager.shutdown()
        await self.security_manager.shutdown()

        self.logger.info("AgentFramework stopped.")

    async def _framework_monitoring_loop(self):
        """Bucle para recolectar y reportar métricas del framework."""
        while True:
            try:
                # Actualizar métricas
                all_agents = await self.registry.list_all_agents()
                active_agents_count = sum(1 for agent in all_agents if agent.status == AgentStatus.ACTIVE)
                self.metrics.total_agents = len(all_agents)
                self.metrics.active_agents = active_agents_count
                # otras métricas...
                
                # Reportar métricas al MonitoringManager
                await self.monitoring_manager.record_metric(
                    Metric(name="framework.total_agents", value=self.metrics.total_agents, unit="count")
                )
                await self.monitoring_manager.record_metric(
                    Metric(name="framework.active_agents", value=self.metrics.active_agents, unit="count")
                )

                self.logger.debug(f"Framework metrics updated: Total Agents={self.metrics.total_agents}, Active Agents={self.metrics.active_agents}")
                
                await asyncio.sleep(self.config.monitoring.collection_interval_seconds)
            except asyncio.CancelledError:
                self.logger.info("Framework monitoring loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in framework monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(self.config.monitoring.collection_interval_seconds) # Avoid tight loop on error

    async def create_agent(self, namespace: str, name: str, agent_class: Type[BaseAgent], creator_agent_id: Optional[str] = None) -> BaseAgent:
        """Crea y registra un nuevo agente dinámicamente."""
        agent = await self.agent_factory.create_agent(namespace, name, agent_class, creator_agent_id)
        if agent:
            self.active_agents[agent.id] = agent
            self.metrics.total_agents += 1
            if agent.status == AgentStatus.ACTIVE:
                self.metrics.active_agents += 1
        return agent

    async def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Obtiene una instancia de agente por su ID."""
        return self.active_agents.get(agent_id)

    async def get_agent_by_name(self, namespace: str, name: str) -> Optional[BaseAgent]:
        """Obtiene una instancia de agente por su namespace y nombre."""
        for agent in self.active_agents.values():
            if agent.namespace == namespace and agent.name == name:
                return agent
        return None

    async def list_all_agents(self) -> List[AgentInfo]:
        """Lista todos los agentes registrados en el framework."""
        return await self.registry.list_all_agents()
    
    async def find_agents_by_capability(self, capability_name: str) -> List[AgentInfo]:
        """Busca agentes que tienen una capacidad específica."""
        return await self.registry.find_agents_by_capability(capability_name)

    async def shutdown(self):
        """Método de apagado principal del framework."""
        await self.stop()


# --- Función de ejemplo de uso actualizada para demostrar capacidades cognitivas ---
async def example_usage():
    sample_config_path = "framework_config.yaml"
    # Crear un archivo de configuración de ejemplo temporal
    sample_config_content = """
    name: "Cognitive Agent Demo Framework"
    version: "1.0.0"
    logging_level: "INFO"
    message_queue_size: 100
    message_timeout: 5
    enable_message_persistence: false

    security:
        enabled: true
        jwt_secret: "super_secret_demo_key"
    persistence:
        enabled: false # Deshabilitar para esta demo si no quieres DB
    monitoring:
        enabled: true
        collection_interval_seconds: 2
        alert_thresholds:
            system.cpu.usage: {"CRITICAL": 90, "WARNING": 75}
    backup:
        enabled: false
    deployment:
        enabled: false
    plugins:
        enabled: false
    
    # NUEVA SECCIÓN DE CONFIGURACIÓN COGNITIVA
    cognitive:
        enabled: true
        llm_model: "gpt-4o" # O "claude-3-sonnet", etc.
        llm_temperature: 0.7
        memory_config:
            chroma_persist_dir: "./framework_chroma_db"
            max_short_term_memory: 10
            embedding_model: "all-MiniLM-L6-v2" # Usar modelo local para demo sin API Key
            # embedding_model: "text-embedding-ada-002" # Para usar OpenAI (requiere OPENAI_API_KEY)

    agents:
        - namespace: "agent.planning"
          name: "StrategistAgent"
          enabled: true
          auto_start: true
          custom_settings:
            description: "An agent capable of defining strategies and proposing workflows."
        - namespace: "agent.cognition"
          name: "ReflectiveAgent"
          enabled: true
          auto_start: true
          custom_settings:
            description: "An agent capable of self-reflection and problem analysis."
    """
    Path(sample_config_path).write_text(sample_config_content)

    framework: Optional[AgentFramework] = None
    try:
        # Inicializar el framework con la configuración
        framework = AgentFramework(config_path=sample_config_path)
        await framework.initialize()
        await framework.start()

        # Dar tiempo para que los agentes se inicialicen y registren
        await asyncio.sleep(2)
        
        strategist: Optional[BaseAgent] = await framework.get_agent_by_name("agent.planning", "StrategistAgent")
        reflective_agent: Optional[BaseAgent] = await framework.get_agent_by_name("agent.cognition", "ReflectiveAgent")

        if strategist and reflective_agent:
            print("\n--- Demostración de Capacidades Cognitivas ---")
            
            # 1. Agente estratega usa think_step_by_step
            print("\n1. StrategistAgent pensando paso a paso para un plan...")
            plan_prompt = "Devise a comprehensive plan to onboard a new engineering team, including technical setup, cultural integration, and initial project assignment."
            strategy_output = await strategist.think_step_by_step(plan_prompt)
            print(f"\nStrategistAgent's Plan:\n{strategy_output}\n")

            # 2. ReflexiveAgent usa self_reflect sobre un problema
            print("\n2. ReflectiveAgent reflexionando sobre un problema de rendimiento...")
            problem_context = "The 'UserService' API has seen a 300% increase in latency over the last 24 hours, especially on 'get_user' endpoint."
            problem_description = "High latency in UserService API, impacting user experience. No recent code deployments."
            reflection_output = await reflective_agent.self_reflect(problem_context, problem_description)
            print(f"\nReflectiveAgent's Reflection:\n{reflection_output}\n")

            # Mostrar algunas memorias del StrategistAgent
            if strategist.memory:
                print(f"\n--- Recent memories for StrategistAgent ({strategist.name}) ---")
                recent_memories = await strategist.memory.get_all_memories(limit=5)
                for i, mem in enumerate(recent_memories):
                    print(f"  {i+1}. [{mem.type.value}] {mem.content[:100]}...")
            
            # Mostrar algunas memorias del ReflectiveAgent
            if reflective_agent.memory:
                print(f"\n--- Recent memories for ReflectiveAgent ({reflective_agent.name}) ---")
                recent_memories = await reflective_agent.memory.get_all_memories(limit=5)
                for i, mem in enumerate(recent_memories):
                    print(f"  {i+1}. [{mem.type.value}] {mem.content[:100]}...")

        else:
            print("Agentes (StrategistAgent o ReflectiveAgent) no encontrados. Asegúrate de que estén definidos en la configuración y se carguen correctamente.")


        print("\n--- Current Agents in Framework ---")
        all_agents = await framework.registry.list_all_agents()
        for agent_info in all_agents:
            print(f"  - {agent_info.namespace}.{agent_info.name} ({agent_info.id}) - {agent_info.status.value}")
                
        await asyncio.sleep(5) # Let the framework run for a bit

    except Exception as e:
        if framework:
            framework.logger.critical(f"An unhandled error occurred in example_usage: {e}", exc_info=True)
        else:
            print(f"An unhandled error occurred before framework initialization: {e}", exc_info=True)
    finally:
        if framework:
            await framework.stop()
        # Clean up the sample config file
        if Path(sample_config_path).exists():
            os.remove(sample_config_path)
            logging.info(f"Cleaned up {sample_config_path}")
        
        # Limpiar directorio de ChromaDB de la demo
        chroma_path = Path("./framework_chroma_db")
        if chroma_path.exists():
            import shutil
            shutil.rmtree(chroma_path)
            logging.info(f"Cleaned up framework ChromaDB directory: {chroma_path}")


if __name__ == "__main__":
    import os
    # Establecer la variable de entorno para OpenAI si se usa
    # os.environ["OPENAI_API_KEY"] = "sk-YOUR_OPENAI_API_KEY" # ¡Importante!

    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("\nFramework simulation interrupted by user.")
    except Exception as e:
        print(f"An error occurred during framework simulation: {e}")