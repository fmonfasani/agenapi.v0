"""
persistence_system.py - Sistema de persistencia para el framework de agentes
"""

import json
import sqlite3
import asyncio
import aiosqlite
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import pickle
import logging
from pathlib import Path

# Importaciones actualizadas
from core.models import AgentMessage, AgentResource, AgentStatus # <-- CAMBIO AQUI
from core.autonomous_agent_framework import BaseAgent, AgentFramework # <-- AGREGADO DE BaseAgent y Framework

# ================================\
# PERSISTENCE INTERFACES
# ================================\

class PersistenceBackend(Enum):
    """Tipos de backend de persistencia"""
    SQLITE = "sqlite"
    JSON_FILE = "json"
    MEMORY = "memory"
    REDIS = "redis"
    POSTGRESQL = "postgresql"

@dataclass
class PersistenceConfig:
    """Configuración de persistencia"""
    backend: PersistenceBackend = PersistenceBackend.SQLITE
    connection_string: str = "framework.db"
    auto_save_interval: int = 60  # segundos
    max_message_history: int = 1000
    enable_compression: bool = False
    backup_enabled: bool = True
    backup_interval: int = 3600  # segundos

class PersistenceInterface(ABC):
    """Interfaz base para sistemas de persistencia"""
    
    @abstractmethod
    async def initialize(self, config: PersistenceConfig) -> bool:
        """Inicializar el sistema de persistencia"""
        pass
        
    @abstractmethod
    async def save_agent_state(self, agent: BaseAgent) -> bool:
        """Guardar el estado de un agente individual (no la instancia completa)"""
        pass
        
    @abstractmethod
    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Cargar el estado de un agente por su ID"""
        pass

    @abstractmethod
    async def save_message(self, message: AgentMessage) -> bool:
        """Guardar un mensaje en el historial"""
        pass

    @abstractmethod
    async def load_messages(self, agent_id: Optional[str] = None, limit: int = 100) -> List[AgentMessage]:
        """Cargar mensajes, opcionalmente filtrados por agente"""
        pass

    @abstractmethod
    async def save_resource(self, resource: AgentResource) -> bool:
        """Guardar un recurso"""
        pass

    @abstractmethod
    async def load_resource(self, resource_id: str) -> Optional[AgentResource]:
        """Cargar un recurso por su ID"""
        pass

    @abstractmethod
    async def delete_resource(self, resource_id: str) -> bool:
        """Eliminar un recurso"""
        pass

    @abstractmethod
    async def save_full_framework_state(self, framework: "AgentFramework") -> bool:
        """Guardar el estado completo del framework (agentes, recursos, etc.)"""
        pass

    @abstractmethod
    async def load_full_framework_state(self, framework: "AgentFramework") -> bool:
        """Cargar el estado completo del framework"""
        pass

    @abstractmethod
    async def cleanup(self, older_than_days: int = 30) -> Dict[str, int]:
        """Limpiar datos antiguos"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Cerrar la conexión/recursos de persistencia"""
        pass


# ================================\
# SQLITE IMPLEMENTATION
# ================================\

class SQLitePersistence(PersistenceInterface):
    """Implementación de persistencia usando SQLite."""
    def __init__(self):
        self.db_path: Path = Path("framework.db")
        self.config: Optional[PersistenceConfig] = None
        self.conn: Optional[aiosqlite.Connection] = None
        self.logger = logging.getLogger("SQLitePersistence")
        self.logger.info("SQLitePersistence backend initialized.")

    async def initialize(self, config: PersistenceConfig) -> bool:
        self.config = config
        self.db_path = Path(config.connection_string)
        try:
            self.conn = await aiosqlite.connect(self.db_path)
            await self._create_tables()
            self.logger.info(f"SQLite database connected and tables ensured at {self.db_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite persistence at {self.db_path}: {e}")
            return False

    async def _create_tables(self):
        """Crea las tablas necesarias si no existen."""
        if not self.conn:
            raise RuntimeError("Database connection not established.")

        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                namespace TEXT,
                name TEXT,
                status TEXT,
                last_heartbeat TEXT,
                state_data BLOB,
                capabilities BLOB
            )
        """)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                sender_id TEXT,
                receiver_id TEXT,
                message_type TEXT,
                payload BLOB,
                timestamp TEXT,
                correlation_id TEXT,
                status TEXT,
                error TEXT
            )
        """)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                id TEXT PRIMARY KEY,
                type TEXT,
                name TEXT,
                namespace TEXT,
                version TEXT,
                data BLOB,
                created_at TEXT,
                last_modified TEXT,
                owner_agent_id TEXT,
                access_permissions BLOB,
                checksum TEXT
            )
        """)
        await self.conn.commit()

    async def save_agent_state(self, agent: BaseAgent) -> bool:
        if not self.conn: return False
        try:
            # Serializar las capacidades de forma sencilla, posiblemente a JSON si son serializables
            # o a una representación simplificada. Para esta demo, lo haremos un JSON de sus nombres.
            capabilities_data = json.dumps([c.name for c in agent.capabilities]) if agent.capabilities else "[]"

            # Serializar la instancia completa del agente (incluyendo atributos adicionales)
            # Esto es un ejemplo. En un sistema de prod, se guardarían atributos clave.
            # NO se guarda toda la instancia si contiene referencias circulares o objetos complejos.
            # Para la demo, lo haremos con pickle, pero es inestable para cambios de clase.
            # La mejor práctica es guardar solo los datos relevantes para reconstruir el estado.
            # Aquí asumimos que los atributos serializables son 'id', 'namespace', 'name', 'status', 'last_heartbeat'
            # y que el resto de datos de estado relevantes deberían estar en un `state_data` dedicado.

            # Crear un diccionario con los atributos serializables del agente
            agent_data_to_save = {
                "id": agent.id,
                "namespace": agent.namespace,
                "name": agent.name,
                "status": agent.status.value,
                "last_heartbeat": agent.last_heartbeat.isoformat(),
                # Aquí se añadirían otros atributos de estado del agente que sean relevantes.
                # Por ejemplo, si un agente tiene un 'memory' o 'knowledge_base'
                "custom_state": {} # Placeholder para estados específicos del agente
            }
            state_data_blob = pickle.dumps(agent_data_to_save) # Usar pickle para la demo, pero JSON es preferible para portabilidad

            await self.conn.execute(
                """
                INSERT OR REPLACE INTO agents 
                (id, namespace, name, status, last_heartbeat, state_data, capabilities) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (agent.id, agent.namespace, agent.name, agent.status.value, 
                 agent.last_heartbeat.isoformat(), state_data_blob, capabilities_data)
            )
            await self.conn.commit()
            self.logger.debug(f"Agent state {agent.id} saved.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save agent state {agent.id}: {e}")
            return False

    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        if not self.conn: return None
        try:
            cursor = await self.conn.execute("SELECT state_data FROM agents WHERE id = ?", (agent_id,))
            row = await cursor.fetchone()
            if row and row[0]:
                state_data_blob = row[0]
                # Deserializar con pickle
                agent_data = pickle.loads(state_data_blob)
                self.logger.debug(f"Agent state {agent_id} loaded.")
                return agent_data
            self.logger.debug(f"Agent state for {agent_id} not found.")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load agent state {agent_id}: {e}")
            return None

    async def save_message(self, message: AgentMessage) -> bool:
        if not self.conn: return False
        try:
            await self.conn.execute(
                """
                INSERT INTO messages 
                (id, sender_id, receiver_id, message_type, payload, timestamp, correlation_id, status, error) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (message.id, message.sender_id, message.receiver_id, message.message_type.value,
                 json.dumps(message.payload), message.timestamp.isoformat(), message.correlation_id,
                 message.status, message.error)
            )
            await self.conn.commit()
            self.logger.debug(f"Message {message.id} saved.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save message {message.id}: {e}")
            return False

    async def load_messages(self, agent_id: Optional[str] = None, limit: int = 100) -> List[AgentMessage]:
        if not self.conn: return []
        messages = []
        try:
            query = "SELECT * FROM messages"
            params = ()
            if agent_id:
                query += " WHERE sender_id = ? OR receiver_id = ?"
                params = (agent_id, agent_id)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params += (limit,)

            cursor = await self.conn.execute(query, params)
            rows = await cursor.fetchall()
            for row in rows:
                try:
                    msg = AgentMessage(
                        id=row[0],
                        sender_id=row[1],
                        receiver_id=row[2],
                        message_type=MessageType(row[3]),
                        payload=json.loads(row[4]),
                        timestamp=datetime.fromisoformat(row[5]),
                        correlation_id=row[6],
                        status=row[7],
                        error=row[8]
                    )
                    messages.append(msg)
                except Exception as e:
                    self.logger.error(f"Error loading message from DB: {row} - {e}")
            self.logger.debug(f"Loaded {len(messages)} messages.")
            return messages
        except Exception as e:
            self.logger.error(f"Failed to load messages: {e}")
            return []

    async def save_resource(self, resource: AgentResource) -> bool:
        if not self.conn: return False
        try:
            await self.conn.execute(
                """
                INSERT OR REPLACE INTO resources 
                (id, type, name, namespace, version, data, created_at, last_modified, owner_agent_id, access_permissions, checksum) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (resource.id, resource.type.value, resource.name, resource.namespace, resource.version,
                 pickle.dumps(resource.data), resource.created_at.isoformat(), resource.last_modified.isoformat(),
                 resource.owner_agent_id, json.dumps(resource.access_permissions), resource.checksum)
            )
            await self.conn.commit()
            self.logger.debug(f"Resource {resource.id} saved.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save resource {resource.id}: {e}")
            return False

    async def load_resource(self, resource_id: str) -> Optional[AgentResource]:
        if not self.conn: return None
        try:
            cursor = await self.conn.execute("SELECT * FROM resources WHERE id = ?", (resource_id,))
            row = await cursor.fetchone()
            if row:
                resource = AgentResource(
                    id=row[0],
                    type=ResourceType(row[1]),
                    name=row[2],
                    namespace=row[3],
                    version=row[4],
                    data=pickle.loads(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    last_modified=datetime.fromisoformat(row[7]),
                    owner_agent_id=row[8],
                    access_permissions=json.loads(row[9]),
                    checksum=row[10]
                )
                self.logger.debug(f"Resource {resource_id} loaded.")
                return resource
            self.logger.debug(f"Resource {resource_id} not found.")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load resource {resource_id}: {e}")
            return None
    
    async def delete_resource(self, resource_id: str) -> bool:
        if not self.conn: return False
        try:
            await self.conn.execute("DELETE FROM resources WHERE id = ?", (resource_id,))
            await self.conn.commit()
            self.logger.debug(f"Resource {resource_id} deleted.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete resource {resource_id}: {e}")
            return False

    async def save_full_framework_state(self, framework: "AgentFramework") -> bool:
        """
        Guarda el estado completo del framework.
        Para un framework real, esto implicaría serializar estados de MessageBus, ResourceManager, etc.
        Para esta demo, nos enfocamos en agentes y recursos gestionados por ellos.
        """
        self.logger.info("Saving full framework state...")
        try:
            # Guardar estado de todos los agentes
            for agent in framework.registry.list_all_agents():
                await self.save_agent_state(agent)
            
            # Guardar estado de todos los recursos
            for resource in framework.resource_manager.list_all_resources():
                await self.save_resource(resource)

            # Opcional: guardar mensajes (ya se guardan al enviarse si se llama save_message)
            self.logger.info("Full framework state saved.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save full framework state: {e}")
            return False

    async def load_full_framework_state(self, framework: "AgentFramework") -> bool:
        """
        Carga el estado completo del framework.
        Requiere que el framework ya esté inicializado con sus componentes.
        Los agentes y recursos se re-crearán en el framework a partir de los datos guardados.
        """
        self.logger.info("Loading full framework state...")
        try:
            # Cargar agentes
            cursor = await self.conn.execute("SELECT id FROM agents")
            agent_ids = [row[0] for row in await cursor.fetchall()]
            
            for agent_id in agent_ids:
                agent_data = await self.load_agent_state(agent_id)
                if agent_data:
                    # Aquí la lógica se vuelve compleja. No podemos "recargar" una instancia BaseAgent
                    # sin saber su clase concreta. Se necesitaría un AgentFactory que pueda
                    # reconstruir agentes a partir de sus datos, o al menos un mapeo.
                    # Para esta demo, solo verificamos que los datos se cargan.
                    # En un sistema real, el AgentFactory tendría un método `reconstruct_agent(data)`.
                    self.logger.debug(f"Loaded agent data for ID: {agent_id}. Needs reconstruction.")
                    # Ejemplo rudimentario de cómo se "reconstruiría"
                    # agent_instance = framework.agent_factory.reconstruct_agent(agent_data)
                    # if agent_instance:
                    #     await framework.registry.register_agent(agent_instance)
            
            # Cargar recursos
            cursor = await self.conn.execute("SELECT id FROM resources")
            resource_ids = [row[0] for row in await cursor.fetchall()]
            for resource_id in resource_ids:
                resource_data = await self.load_resource(resource_id)
                if resource_data:
                    # Añadir el recurso al resource manager del framework
                    await framework.resource_manager.create_resource(resource_data)
                    self.logger.debug(f"Loaded resource: {resource_data.id}")
            
            self.logger.info("Full framework state loaded (reconstruction of agents is simplified).")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load full framework state: {e}")
            return False

    async def cleanup(self, older_than_days: int = 30) -> Dict[str, int]:
        if not self.conn: return {"messages_deleted": 0, "resources_deleted": 0}
        try:
            threshold_date = (datetime.now() - timedelta(days=older_than_days)).isoformat()
            
            cursor = await self.conn.execute("DELETE FROM messages WHERE timestamp < ?", (threshold_date,))
            messages_deleted = cursor.rowcount
            
            # Los recursos probablemente no deberían tener una limpieza automática por antigüedad
            # a menos que sean temporales. Esto es solo para demostración.
            cursor = await self.conn.execute("DELETE FROM resources WHERE last_modified < ?", (threshold_date,))
            resources_deleted = cursor.rowcount

            await self.conn.commit()
            self.logger.info(f"Cleaned up {messages_deleted} messages and {resources_deleted} resources older than {older_than_days} days.")
            return {"messages_deleted": messages_deleted, "resources_deleted": resources_deleted}
        except Exception as e:
            self.logger.error(f"Failed to perform cleanup: {e}")
            return {"messages_deleted": 0, "resources_deleted": 0, "error": str(e)}

    async def close(self) -> None:
        if self.conn:
            await self.conn.close()
            self.conn = None
            self.logger.info("SQLite connection closed.")

# ================================\
# PERSISTENCE MANAGER
# ================================\

class PersistenceManager:
    """
    Gestiona el almacenamiento y recuperación del estado del framework.
    Actúa como una interfaz para diferentes backends de persistencia.
    """
    def __init__(self, framework: "AgentFramework", config: Optional[PersistenceConfig] = None):
        self.framework = framework
        self.config = config or PersistenceConfig()
        self.backend: PersistenceInterface
        self.logger = logging.getLogger("PersistenceManager")
        self._auto_save_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self.is_initialized = False

        self._initialize_backend()
        self.logger.info("PersistenceManager initialized.")

    def _initialize_backend(self):
        """Selecciona e inicializa el backend de persistencia."""
        if self.config.backend == PersistenceBackend.SQLITE:
            self.backend = SQLitePersistence()
        elif self.config.backend == PersistenceBackend.JSON_FILE:
            # TODO: Implement JSONFilePersistence
            raise NotImplementedError("JSON_FILE backend not yet implemented.")
        elif self.config.backend == PersistenceBackend.MEMORY:
            # TODO: Implement InMemoryPersistence (útil para pruebas)
            raise NotImplementedError("MEMORY backend not yet implemented.")
        else:
            raise ValueError(f"Unsupported persistence backend: {self.config.backend}")

    async def initialize(self) -> bool:
        """Inicializa el backend de persistencia seleccionado."""
        if not self.is_initialized:
            success = await self.backend.initialize(self.config)
            if success:
                self.is_initialized = True
                if self.config.auto_save_interval > 0:
                    self._auto_save_task = asyncio.create_task(self._auto_save_loop())
                    self.logger.info(f"Auto-save enabled with interval: {self.config.auto_save_interval} seconds.")
                return True
            else:
                self.logger.error("Failed to initialize persistence backend.")
                return False
        return True

    async def _auto_save_loop(self):
        """Bucle para el auto-guardado periódico."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.config.auto_save_interval)
                self.logger.info("Performing auto-save...")
                await self.save_full_state(self.framework)
            except asyncio.CancelledError:
                self.logger.info("Auto-save task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error during auto-save: {e}")

    async def save_full_state(self, framework: "AgentFramework") -> bool:
        """Guarda el estado completo del framework."""
        if not self.is_initialized:
            self.logger.warning("Persistence not initialized. Cannot save state.")
            return False
        return await self.backend.save_full_framework_state(framework)

    async def load_full_state(self, framework: "AgentFramework") -> bool:
        """Carga el estado completo del framework."""
        if not self.is_initialized:
            self.logger.warning("Persistence not initialized. Cannot load state.")
            return False
        return await self.backend.load_full_framework_state(framework)

    async def close(self) -> None:
        """Cierra el gestor de persistencia y su backend."""
        if self._auto_save_task:
            self._stop_event.set()
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass # Expected
        if self.is_initialized:
            await self.backend.close()
            self.is_initialized = False
        self.logger.info("PersistenceManager closed.")

# ================================\
# PERSISTENCE FACTORY (optional, useful for DI)
# ================================\

class PersistenceFactory:
    """Factoría para crear instancias de PersistenceManager."""
    @staticmethod
    def create_persistence_manager(framework: "AgentFramework", config: Optional[PersistenceConfig] = None) -> PersistenceManager:
        return PersistenceManager(framework, config)

# ================================\
# DEMO
# ================================\

async def persistence_demo():
    """Ejemplo de uso del sistema de persistencia."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("🚀 Starting Persistence System Demo")
    print("="*50)

    # Inicializar Framework (necesario para registry, message_bus, resource_manager)
    framework = AgentFramework()
    await framework.start()

    persistence_config = PersistenceConfig(
        backend=PersistenceBackend.SQLITE,
        connection_string="demo_framework.db",
        auto_save_interval=5 # Guardar cada 5 segundos para la demo
    )
    persistence_manager = PersistenceManager(framework, persistence_config)
    await persistence_manager.initialize()

    # Eliminar archivo de DB anterior para una demo limpia
    if Path(persistence_config.connection_string).exists():
        Path(persistence_config.connection_string).unlink()
        print(f"🗑️ Cleaned up previous database: {persistence_config.connection_string}")
        # Volver a inicializar la conexión si se borró el archivo
        await persistence_manager.close()
        await persistence_manager.initialize()


    print("⚙️ Persistence system initialized. Simulating agent activity...")

    # Simular algunos agentes
    class DemoAgent(BaseAgent):
        def __init__(self, agent_id: str, name: str, framework_instance: AgentFramework):
            super().__init__("demo.agent", name, framework_instance)
            self.id = agent_id
            self.status = AgentStatus.ACTIVE
            self.message_queue = asyncio.Queue()

        async def initialize(self) -> bool:
            self.logger.info(f"DemoAgent {self.name} initialized.")
            return True
        
        async def process_message(self, message):
            self.logger.info(f"DemoAgent {self.name} received message: {message.payload.get('action')}")
            # Simular trabajo
            await asyncio.sleep(0.1)

    agent1 = DemoAgent("pers-agent-001", "PersistAgent1", framework)
    agent2 = DemoAgent("pers-agent-002", "PersistAgent2", framework)
    
    await agent1.initialize()
    await agent2.initialize()

    await agent1.start()
    await agent2.start()

    # Simular intercambio de mensajes para que se persistan
    await agent1.send_message(agent2.id, "greet", {"text": "Hello Agent2!"})
    await agent2.send_message(agent1.id, "reply", {"text": "Hi Agent1!"})
    await asyncio.sleep(1) # Dar tiempo para que se procesen y guarden

    # Simular la creación de un recurso
    test_resource = AgentResource(
        type=ResourceType.DATA,
        name="demo_resource",
        namespace="resource.demo",
        data={"content": "This is a demo resource"},
        owner_agent_id=agent1.id
    )
    await framework.resource_manager.create_resource(test_resource)
    
    print("✅ Simulated agent activity (messages and resources)")

    # Guardar estado completo
    # El auto-save debería encargarse de esto, pero lo hacemos manualmente para la demo
    await persistence_manager.save_full_state(framework)
    print("💾 Framework state saved to persistence")
    
    # Detener framework para simular un reinicio
    print("\n🔄 Stopping framework to simulate restart...")
    await framework.stop()
    print("Framework stopped.")

    # Crear una nueva instancia del framework para cargar el estado
    print("\n🚀 Starting new framework instance to load state...")
    new_framework = AgentFramework()
    await new_framework.start()

    new_persistence_manager = PersistenceManager(new_framework, persistence_config)
    await new_persistence_manager.initialize()

    # Cargar estado
    print("📖 Loading persisted state into new framework instance...")
    load_success = await new_persistence_manager.load_full_state(new_framework)
    
    if load_success:
        print("✅ State loaded successfully.")
        loaded_agents = new_framework.registry.list_all_agents()
        loaded_resources = new_framework.resource_manager.list_all_resources()
        print(f"   • Loaded {len(loaded_agents)} agents (note: agent instances not fully re-hydrated in this demo).")
        print(f"   • Loaded {len(loaded_resources)} resources.")
        
        # Verificar un recurso cargado
        loaded_test_resource = await new_persistence_manager.backend.load_resource(test_resource.id)
        if loaded_test_resource:
            print(f"   • Verified loaded resource '{loaded_test_resource.name}': {loaded_test_resource.data}")
        else:
            print(f"   ❌ Failed to load resource '{test_resource.id}'")

        # Cargar mensajes de un agente
        messages_for_agent1 = await new_persistence_manager.backend.load_messages(agent1.id, limit=5)
        print(f"   • Messages involving '{agent1.name}': {len(messages_for_agent1)}")
        for msg in messages_for_agent1:
            print(f"     - Msg: {msg.sender_id} -> {msg.receiver_id} ({msg.message_type.value}) Payload: {msg.payload}")

    else:
        print("❌ Failed to load state.")


    # Cleanup demo
    print("\n🧹 Running cleanup (removing old messages/resources)...")
    cleanup_result = await new_persistence_manager.backend.cleanup(older_than_days=0) # Borrar todo lo de hoy
    print(f"   • Cleanup completed: {cleanup_result}")
    
    # Cerrar todo
    await new_framework.stop()
    await new_persistence_manager.close()
    print("\n👋 Demo completed - state management demonstrated!")

if __name__ == "__main__":
    asyncio.run(persistence_demo())