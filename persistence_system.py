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

from autonomous_agent_framework import AgentMessage, AgentResource, BaseAgent, AgentStatus

# ================================
# PERSISTENCE INTERFACES
# ================================

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
        """Guardar estado de un agente"""
        pass
        
    @abstractmethod
    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Cargar estado de un agente"""
        pass
        
    @abstractmethod
    async def save_message(self, message: AgentMessage) -> bool:
        """Guardar mensaje"""
        pass
        
    @abstractmethod
    async def load_messages(self, agent_id: str, limit: int = 100) -> List[AgentMessage]:
        """Cargar mensajes de un agente"""
        pass
        
    @abstractmethod
    async def save_resource(self, resource: AgentResource) -> bool:
        """Guardar recurso"""
        pass
        
    @abstractmethod
    async def load_resource(self, resource_id: str) -> Optional[AgentResource]:
        """Cargar recurso"""
        pass
        
    @abstractmethod
    async def save_framework_state(self, state: Dict[str, Any]) -> bool:
        """Guardar estado del framework"""
        pass
        
    @abstractmethod
    async def load_framework_state(self) -> Optional[Dict[str, Any]]:
        """Cargar estado del framework"""
        pass
        
    @abstractmethod
    async def cleanup(self, older_than_days: int = 30) -> bool:
        """Limpiar datos antiguos"""
        pass
        
    @abstractmethod
    async def close(self) -> bool:
        """Cerrar conexiones"""
        pass

# ================================
# SQLITE PERSISTENCE IMPLEMENTATION
# ================================

class SQLitePersistence(PersistenceInterface):
    """Implementación de persistencia usando SQLite"""
    
    def __init__(self):
        self.db_path = None
        self.connection = None
        
    async def initialize(self, config: PersistenceConfig) -> bool:
        """Inicializar base de datos SQLite"""
        self.db_path = config.connection_string
        
        try:
            # Crear tablas
            async with aiosqlite.connect(self.db_path) as db:
                await self._create_tables(db)
                await db.commit()
            
            logging.info(f"SQLite persistence initialized: {self.db_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize SQLite persistence: {e}")
            return False
            
    async def _create_tables(self, db: aiosqlite.Connection):
        """Crear tablas de la base de datos"""
        
        # Tabla de agentes
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata TEXT,
                state_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de mensajes
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                sender_id TEXT NOT NULL,
                receiver_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                action TEXT NOT NULL,
                payload TEXT,
                correlation_id TEXT,
                response_required BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de recursos
        await db.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                namespace TEXT NOT NULL,
                data TEXT,
                owner_agent_id TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de estado del framework
        await db.execute("""
            CREATE TABLE IF NOT EXISTS framework_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Índices para performance
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_receiver ON messages(receiver_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_resources_owner ON resources(owner_agent_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_resources_type ON resources(type)")
        
    async def save_agent_state(self, agent: BaseAgent) -> bool:
        """Guardar estado de un agente"""
        try:
            # Serializar estado del agente
            state_data = {
                "capabilities": [asdict(cap) for cap in agent.capabilities],
                "metadata": agent.metadata,
                "created_at": agent.created_at.isoformat(),
                "last_heartbeat": agent.last_heartbeat.isoformat()
            }
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO agents 
                    (id, namespace, name, status, metadata, state_data, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    agent.id,
                    agent.namespace,
                    agent.name,
                    agent.status.value,
                    json.dumps(agent.metadata),
                    json.dumps(state_data)
                ))
                await db.commit()
                
            return True
            
        except Exception as e:
            logging.error(f"Failed to save agent state {agent.id}: {e}")
            return False
            
    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Cargar estado de un agente"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM agents WHERE id = ?", (agent_id,)
                )
                row = await cursor.fetchone()
                
                if row:
                    return {
                        "id": row["id"],
                        "namespace": row["namespace"],
                        "name": row["name"],
                        "status": row["status"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                        "state_data": json.loads(row["state_data"]) if row["state_data"] else {},
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"]
                    }
                    
        except Exception as e:
            logging.error(f"Failed to load agent state {agent_id}: {e}")
            
        return None
        
    async def save_message(self, message: AgentMessage) -> bool:
        """Guardar mensaje"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO messages 
                    (id, sender_id, receiver_id, message_type, action, payload, 
                     correlation_id, response_required, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.id,
                    message.sender_id,
                    message.receiver_id,
                    message.message_type.value,
                    message.action,
                    json.dumps(message.payload),
                    message.correlation_id,
                    message.response_required,
                    message.timestamp.isoformat()
                ))
                await db.commit()
                
            return True
            
        except Exception as e:
            logging.error(f"Failed to save message {message.id}: {e}")
            return False
            
    async def load_messages(self, agent_id: str, limit: int = 100) -> List[AgentMessage]:
        """Cargar mensajes de un agente"""
        messages = []
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("""
                    SELECT * FROM messages 
                    WHERE sender_id = ? OR receiver_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (agent_id, agent_id, limit))
                
                rows = await cursor.fetchall()
                
                for row in rows:
                    from autonomous_agent_framework import MessageType
                    message = AgentMessage(
                        id=row["id"],
                        sender_id=row["sender_id"],
                        receiver_id=row["receiver_id"],
                        message_type=MessageType(row["message_type"]),
                        action=row["action"],
                        payload=json.loads(row["payload"]) if row["payload"] else {},
                        correlation_id=row["correlation_id"],
                        response_required=bool(row["response_required"]),
                        timestamp=datetime.fromisoformat(row["timestamp"])
                    )
                    messages.append(message)
                    
        except Exception as e:
            logging.error(f"Failed to load messages for agent {agent_id}: {e}")
            
        return messages
        
    async def save_resource(self, resource: AgentResource) -> bool:
        """Guardar recurso"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO resources 
                    (id, type, name, namespace, data, owner_agent_id, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    resource.id,
                    resource.type.value,
                    resource.name,
                    resource.namespace,
                    json.dumps(resource.data),
                    resource.owner_agent_id,
                    json.dumps(resource.metadata)
                ))
                await db.commit()
                
            return True
            
        except Exception as e:
            logging.error(f"Failed to save resource {resource.id}: {e}")
            return False
            
    async def load_resource(self, resource_id: str) -> Optional[AgentResource]:
        """Cargar recurso"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM resources WHERE id = ?", (resource_id,)
                )
                row = await cursor.fetchone()
                
                if row:
                    from autonomous_agent_framework import ResourceType
                    return AgentResource(
                        id=row["id"],
                        type=ResourceType(row["type"]),
                        name=row["name"],
                        namespace=row["namespace"],
                        data=json.loads(row["data"]) if row["data"] else {},
                        owner_agent_id=row["owner_agent_id"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"])
                    )
                    
        except Exception as e:
            logging.error(f"Failed to load resource {resource_id}: {e}")
            
        return None
        
    async def save_framework_state(self, state: Dict[str, Any]) -> bool:
        """Guardar estado del framework"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for key, value in state.items():
                    await db.execute("""
                        INSERT OR REPLACE INTO framework_state (key, value, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    """, (key, json.dumps(value)))
                await db.commit()
                
            return True
            
        except Exception as e:
            logging.error(f"Failed to save framework state: {e}")
            return False
            
    async def load_framework_state(self) -> Optional[Dict[str, Any]]:
        """Cargar estado del framework"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("SELECT key, value FROM framework_state")
                rows = await cursor.fetchall()
                
                state = {}
                for row in rows:
                    state[row["key"]] = json.loads(row["value"])
                    
                return state if state else None
                
        except Exception as e:
            logging.error(f"Failed to load framework state: {e}")
            
        return None
        
    async def cleanup(self, older_than_days: int = 30) -> bool:
        """Limpiar datos antiguos"""
        try:
            cutoff_date = datetime.now().isoformat().split('T')[0]
            # En SQLite, restar días de fecha actual
            
            async with aiosqlite.connect(self.db_path) as db:
                # Limpiar mensajes antiguos
                await db.execute("""
                    DELETE FROM messages 
                    WHERE date(timestamp) < date('now', '-{} days')
                """.format(older_than_days))
                
                # Limpiar agentes inactivos
                await db.execute("""
                    DELETE FROM agents 
                    WHERE status = 'terminated' 
                    AND date(updated_at) < date('now', '-{} days')
                """.format(older_than_days))
                
                await db.commit()
                
            logging.info(f"Cleanup completed: removed data older than {older_than_days} days")
            return True
            
        except Exception as e:
            logging.error(f"Failed to cleanup old data: {e}")
            return False
            
    async def close(self) -> bool:
        """Cerrar conexiones"""
        # SQLite se cierra automáticamente con context managers
        return True

# ================================
# JSON FILE PERSISTENCE
# ================================

class JSONFilePersistence(PersistenceInterface):
    """Implementación de persistencia usando archivos JSON"""
    
    def __init__(self):
        self.data_dir = None
        self.data = {
            "agents": {},
            "messages": {},
            "resources": {},
            "framework_state": {}
        }
        
    async def initialize(self, config: PersistenceConfig) -> bool:
        """Inicializar persistencia con archivos JSON"""
        self.data_dir = Path(config.connection_string)
        self.data_dir.mkdir(exist_ok=True)
        
        # Cargar datos existentes
        await self._load_all_data()
        return True
        
    async def _load_all_data(self):
        """Cargar todos los datos desde archivos"""
        for data_type in self.data.keys():
            file_path = self.data_dir / f"{data_type}.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        self.data[data_type] = json.load(f)
                except Exception as e:
                    logging.error(f"Failed to load {data_type}.json: {e}")
                    
    async def _save_data_type(self, data_type: str):
        """Guardar un tipo específico de datos"""
        file_path = self.data_dir / f"{data_type}.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(self.data[data_type], f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save {data_type}.json: {e}")
            
    async def save_agent_state(self, agent: BaseAgent) -> bool:
        """Guardar estado de un agente"""
        try:
            agent_data = {
                "id": agent.id,
                "namespace": agent.namespace,
                "name": agent.name,
                "status": agent.status.value,
                "metadata": agent.metadata,
                "capabilities": [asdict(cap) for cap in agent.capabilities],
                "created_at": agent.created_at.isoformat(),
                "last_heartbeat": agent.last_heartbeat.isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            self.data["agents"][agent.id] = agent_data
            await self._save_data_type("agents")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save agent state {agent.id}: {e}")
            return False
            
    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Cargar estado de un agente"""
        return self.data["agents"].get(agent_id)
        
    async def save_message(self, message: AgentMessage) -> bool:
        """Guardar mensaje"""
        try:
            message_data = {
                "id": message.id,
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                "message_type": message.message_type.value,
                "action": message.action,
                "payload": message.payload,
                "correlation_id": message.correlation_id,
                "response_required": message.response_required,
                "timestamp": message.timestamp.isoformat()
            }
            
            self.data["messages"][message.id] = message_data
            await self._save_data_type("messages")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save message {message.id}: {e}")
            return False
            
    async def load_messages(self, agent_id: str, limit: int = 100) -> List[AgentMessage]:
        """Cargar mensajes de un agente"""
        messages = []
        
        try:
            # Filtrar mensajes del agente y ordenar por timestamp
            agent_messages = [
                msg for msg in self.data["messages"].values()
                if msg["sender_id"] == agent_id or msg["receiver_id"] == agent_id
            ]
            
            # Ordenar por timestamp descendente y limitar
            agent_messages.sort(key=lambda x: x["timestamp"], reverse=True)
            agent_messages = agent_messages[:limit]
            
            # Convertir a objetos AgentMessage
            for msg_data in agent_messages:
                from autonomous_agent_framework import MessageType
                message = AgentMessage(
                    id=msg_data["id"],
                    sender_id=msg_data["sender_id"],
                    receiver_id=msg_data["receiver_id"],
                    message_type=MessageType(msg_data["message_type"]),
                    action=msg_data["action"],
                    payload=msg_data["payload"],
                    correlation_id=msg_data["correlation_id"],
                    response_required=msg_data["response_required"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"])
                )
                messages.append(message)
                
        except Exception as e:
            logging.error(f"Failed to load messages for agent {agent_id}: {e}")
            
        return messages
        
    async def save_resource(self, resource: AgentResource) -> bool:
        """Guardar recurso"""
        try:
            resource_data = {
                "id": resource.id,
                "type": resource.type.value,
                "name": resource.name,
                "namespace": resource.namespace,
                "data": resource.data,
                "owner_agent_id": resource.owner_agent_id,
                "metadata": resource.metadata,
                "created_at": resource.created_at.isoformat(),
                "updated_at": resource.updated_at.isoformat()
            }
            
            self.data["resources"][resource.id] = resource_data
            await self._save_data_type("resources")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save resource {resource.id}: {e}")
            return False
            
    async def load_resource(self, resource_id: str) -> Optional[AgentResource]:
        """Cargar recurso"""
        try:
            resource_data = self.data["resources"].get(resource_id)
            if resource_data:
                from autonomous_agent_framework import ResourceType
                return AgentResource(
                    id=resource_data["id"],
                    type=ResourceType(resource_data["type"]),
                    name=resource_data["name"],
                    namespace=resource_data["namespace"],
                    data=resource_data["data"],
                    owner_agent_id=resource_data["owner_agent_id"],
                    metadata=resource_data["metadata"],
                    created_at=datetime.fromisoformat(resource_data["created_at"]),
                    updated_at=datetime.fromisoformat(resource_data["updated_at"])
                )
        except Exception as e:
            logging.error(f"Failed to load resource {resource_id}: {e}")
            
        return None
        
    async def save_framework_state(self, state: Dict[str, Any]) -> bool:
        """Guardar estado del framework"""
        try:
            self.data["framework_state"].update(state)
            await self._save_data_type("framework_state")
            return True
        except Exception as e:
            logging.error(f"Failed to save framework state: {e}")
            return False
            
    async def load_framework_state(self) -> Optional[Dict[str, Any]]:
        """Cargar estado del framework"""
        return self.data["framework_state"] if self.data["framework_state"] else None
        
    async def cleanup(self, older_than_days: int = 30) -> bool:
        """Limpiar datos antiguos"""
        try:
            cutoff_date = datetime.now().timestamp() - (older_than_days * 24 * 3600)
            
            # Limpiar mensajes antiguos
            old_messages = [
                msg_id for msg_id, msg in self.data["messages"].items()
                if datetime.fromisoformat(msg["timestamp"]).timestamp() < cutoff_date
            ]
            
            for msg_id in old_messages:
                del self.data["messages"][msg_id]
                
            # Limpiar agentes terminados antiguos
            old_agents = [
                agent_id for agent_id, agent in self.data["agents"].items()
                if (agent["status"] == "terminated" and 
                    datetime.fromisoformat(agent["updated_at"]).timestamp() < cutoff_date)
            ]
            
            for agent_id in old_agents:
                del self.data["agents"][agent_id]
                
            # Guardar cambios
            await self._save_data_type("messages")
            await self._save_data_type("agents")
            
            logging.info(f"Cleanup completed: removed {len(old_messages)} messages and {len(old_agents)} agents")
            return True
            
        except Exception as e:
            logging.error(f"Failed to cleanup old data: {e}")
            return False
            
    async def close(self) -> bool:
        """Cerrar y guardar todos los datos"""
        try:
            for data_type in self.data.keys():
                await self._save_data_type(data_type)
            return True
        except Exception as e:
            logging.error(f"Failed to close persistence: {e}")
            return False

# ================================
# PERSISTENCE MANAGER
# ================================

class PersistenceManager:
    """Gestor principal de persistencia"""
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.backend: PersistenceInterface = None
        self.auto_save_task = None
        
    async def initialize(self) -> bool:
        """Inicializar el sistema de persistencia"""
        
        # Crear backend apropiado
        if self.config.backend == PersistenceBackend.SQLITE:
            self.backend = SQLitePersistence()
        elif self.config.backend == PersistenceBackend.JSON_FILE:
            self.backend = JSONFilePersistence()
        else:
            logging.error(f"Unsupported persistence backend: {self.config.backend}")
            return False
            
        # Inicializar backend
        success = await self.backend.initialize(self.config)
        
        if success and self.config.auto_save_interval > 0:
            # Iniciar auto-save
            self.auto_save_task = asyncio.create_task(self._auto_save_loop())
            
        return success
        
    async def _auto_save_loop(self):
        """Loop de auto-guardado"""
        while True:
            try:
                await asyncio.sleep(self.config.auto_save_interval)
                # Aquí podrías implementar lógica de auto-save específica
                logging.debug("Auto-save triggered")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Auto-save error: {e}")
                
    async def restore_framework_state(self, framework) -> bool:
        """Restaurar estado completo del framework"""
        try:
            # Cargar estado del framework
            framework_state = await self.backend.load_framework_state()
            
            if framework_state:
                logging.info("Restoring framework state from persistence")
                
                # Restaurar agentes si están configurados para auto-start
                if "agents" in framework_state:
                    await self._restore_agents(framework, framework_state["agents"])
                    
                # Restaurar recursos
                if "resources" in framework_state:
                    await self._restore_resources(framework, framework_state["resources"])
                    
                return True
                
        except Exception as e:
            logging.error(f"Failed to restore framework state: {e}")
            
        return False
        
    async def _restore_agents(self, framework, agent_states: Dict[str, Any]):
        """Restaurar agentes desde estado persistido"""
        from specialized_agents import ExtendedAgentFactory
        
        for agent_id, agent_config in agent_states.items():
            try:
                # Cargar estado completo del agente
                agent_state = await self.backend.load_agent_state(agent_id)
                
                if agent_state and agent_state.get("status") != "terminated":
                    # Recrear agente
                    namespace = agent_state["namespace"]
                    name = agent_state["name"]
                    
                    if namespace in ExtendedAgentFactory.AGENT_CLASSES:
                        agent_class = ExtendedAgentFactory.AGENT_CLASSES[namespace]
                        agent = agent_class(name, framework)
                        
                        # Restaurar metadata y estado
                        agent.metadata = agent_state.get("metadata", {})
                        
                        await agent.start()
                        logging.info(f"Restored agent: {agent_id} ({namespace})")
                        
            except Exception as e:
                logging.error(f"Failed to restore agent {agent_id}: {e}")
                
    async def _restore_resources(self, framework, resource_ids: List[str]):
        """Restaurar recursos desde estado persistido"""
        for resource_id in resource_ids:
            try:
                resource = await self.backend.load_resource(resource_id)
                if resource:
                    await framework.resource_manager.create_resource(resource)
                    logging.info(f"Restored resource: {resource_id}")
            except Exception as e:
                logging.error(f"Failed to restore resource {resource_id}: {e}")
                
    async def save_full_state(self, framework) -> bool:
        """Guardar estado completo del framework"""
        try:
            # Guardar estado de todos los agentes
            agents = framework.registry.list_all_agents()
            for agent in agents:
                await self.backend.save_agent_state(agent)
                
            # Guardar todos los recursos
            all_resources = []
            for agent in agents:
                agent_resources = framework.resource_manager.find_resources_by_owner(agent.id)
                all_resources.extend(agent_resources)
                
            for resource in all_resources:
                await self.backend.save_resource(resource)
                
            # Guardar estado general del framework
            framework_state = {
                "agents": {agent.id: {"auto_restart": True} for agent in agents},
                "resources": [resource.id for resource in all_resources],
                "last_save": datetime.now().isoformat()
            }
            
            await self.backend.save_framework_state(framework_state)
            
            logging.info(f"Saved complete framework state: {len(agents)} agents, {len(all_resources)} resources")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save framework state: {e}")
            return False
            
    async def close(self):
        """Cerrar el sistema de persistencia"""
        if self.auto_save_task:
            self.auto_save_task.cancel()
            
        if self.backend:
            await self.backend.close()

# ================================
# PERSISTENCE FACTORY
# ================================

class PersistenceFactory:
    """Factory para crear sistemas de persistencia"""
    
    @staticmethod
    def create_persistence_manager(
        backend: PersistenceBackend = PersistenceBackend.SQLITE,
        connection_string: str = "framework.db",
        **kwargs
    ) -> PersistenceManager:
        """Crear gestor de persistencia"""
        
        config = PersistenceConfig(
            backend=backend,
            connection_string=connection_string,
            **kwargs
        )
        
        return PersistenceManager(config)

# ================================
# EXAMPLE USAGE
# ================================

async def persistence_demo():
    """Demo del sistema de persistencia"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Crear framework con persistencia
    from autonomous_agent_framework import AgentFramework
    from specialized_agents import ExtendedAgentFactory
    
    # Configurar persistencia
    persistence_manager = PersistenceFactory.create_persistence_manager(
        backend=PersistenceBackend.SQLITE,
        connection_string="demo_framework.db",
        auto_save_interval=30
    )
    
    await persistence_manager.initialize()
    
    # Crear framework
    framework = AgentFramework()
    await framework.start()
    
    # Intentar restaurar estado anterior
    restored = await persistence_manager.restore_framework_state(framework)
    if restored:
        print("✅ Framework state restored from persistence")
    else:
        print("ℹ️ No previous state found, starting fresh")
        
        # Crear algunos agentes para demo
        strategist = ExtendedAgentFactory.create_agent("agent.planning.strategist", "strategist", framework)
        generator = ExtendedAgentFactory.create_agent("agent.build.code.generator", "generator", framework)
        
        await strategist.start()
        await generator.start()
        
        print("📝 Created new agents for demo")
    
    # Mostrar agentes activos
    agents = framework.registry.list_all_agents()
    print(f"\n👥 Active agents: {len(agents)}")
    for agent in agents:
        print(f"   • {agent.name} ({agent.namespace}) - {agent.status.value}")
    
    # Simular actividad
    if len(agents) >= 2:
        agent1, agent2 = agents[0], agents[1]
        
        # Intercambio de mensajes
        await agent1.send_message(agent2.id, "test.message", {"data": "persistence demo"})
        
        # Crear recurso
        from autonomous_agent_framework import AgentResource, ResourceType
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
    await persistence_manager.save_full_state(framework)
    print("💾 Framework state saved to persistence")
    
    # Demo de carga de datos
    print("\n📖 Loading persisted data:")
    
    # Cargar mensajes de un agente
    if agents:
        messages = await persistence_manager.backend.load_messages(agents[0].id, limit=5)
        print(f"   • Messages for {agents[0].name}: {len(messages)}")
        
    # Cargar recursos
    all_resources = []
    for agent in agents:
        agent_resources = framework.resource_manager.find_resources_by_owner(agent.id)
        all_resources.extend(agent_resources)
    print(f"   • Total resources: {len(all_resources)}")
    
    # Cleanup demo
    print("\n🧹 Running cleanup (demo - no actual deletion)")
    cleanup_result = await persistence_manager.backend.cleanup(older_than_days=365)  # Very old to avoid deleting demo data
    print(f"   • Cleanup completed: {cleanup_result}")
    
    # Cerrar todo
    await framework.stop()
    await persistence_manager.close()
    print("\n👋 Demo completed - state saved for next run!")

if __name__ == "__main__":
    asyncio.run(persistence_demo())