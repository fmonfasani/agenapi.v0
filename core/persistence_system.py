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

from core.autonomous_agent_framework import AgentMessage, AgentResource, BaseAgent, AgentStatus

# PERSISTENCE INTERFACES


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


# SQLITE PERSISTENCE IMPLEMENTATION


class SQLitePersistence(PersistenceInterface):
    def __init__(self):
        self.db_path = None
        self.connection = None
        self.logger = logging.getLogger("SQLitePersistence")
        
    async def initialize(self, config: PersistenceConfig) -> bool:
        """Inicializar base de datos SQLite"""
        self.db_path = config.connection_string
        
        try:
            # Crear tablas
            async with aiosqlite.connect(self.db_path) as db:
                await self._create_tables(db)
                await db.commit()
            
            self.logger.info(f"SQLite persistence initialized: {self.db_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite persistence: {e}")
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
            self.logger.error(f"Failed to save agent state {agent.id}: {e}")
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
            self.logger.error(f"Failed to load agent state {agent_id}: {e}")
            
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
            self.logger.error(f"Failed to save message {message.id}: {e}")
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
                    from core.autonomous_agent_framework import MessageType
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
            self.logger.error(f"Failed to load messages for agent {agent_id}: {e}")
            
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
            self.logger.error(f"Failed to save resource {resource.id}: {e}")
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
                    from core.autonomous_agent_framework import ResourceType
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
            self.logger.error(f"Failed to load resource {resource_id}: {e}")
            
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
            self.logger.error(f"Failed to save framework state: {e}")
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
            self.logger.error(f"Failed to load framework state: {e}")
            
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
                
            self.logger.info(f"Cleanup completed: removed data older than {older_than_days} days")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
            return False
            
    async def close(self) -> bool:
        """Cerrar conexiones"""
        # SQLite se cierra automáticamente con context managers
        return True


# JSON FILE PERSISTENCE


class JSONFilePersistence(PersistenceInterface):
    def __init__(self):
        self.data_dir = None
        self.data = {
            "agents": {},
            "messages": {},
            "resources": {},
            "framework_state": {}
        }
        self.logger = logging.getLogger("JSONFilePersistence")
        
    async def initialize(self, config: PersistenceConfig) -> bool:
        """Inicializar persistencia con archivos JSON"""
        self.data_dir = Path(config.connection_string)
        self.data_dir.mkdir(exist_ok=True)
        
        # Cargar datos existentes
        await self._load_all_data()
        self.logger.info(f"JSON File persistence initialized: {self.data_dir}")
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
                    self.logger.error(f"Failed to load {data_type}.json: {e}")
                    
    async def _save_data_type(self, data_type: str):
        """Guardar un tipo específico de datos"""
        file_path = self.data_dir / f"{data_type}.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(self.data[data_type], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save {data_type}.json: {e}")
            
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
            self.logger.error(f"Failed to save agent state {agent.id}: {e}")
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
            self.logger.error(f"Failed to save message {message.id}: {e}")
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
                from core.autonomous_agent_framework import MessageType
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
            self.logger.error(f"Failed to load messages for agent {agent_id}: {e}")
            
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
            self.logger.error(f"Failed to save resource {resource.id}: {e}")
            return False
            
    async def load_resource(self, resource_id: str) -> Optional[AgentResource]:
        """Cargar recurso"""
        try:
            resource_data = self.data["resources"].get(resource_id)
            if resource_data:
                from core.autonomous_agent_framework import ResourceType
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
            self.logger.error(f"Failed to load resource {resource_id}: {e}")
            
        return None
        
    async def save_framework_state(self, state: Dict[str, Any]) -> bool:
        """Guardar estado del framework"""
        try:
            self.data["framework_state"].update(state)
            await self._save_data_type("framework_state")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save framework state: {e}")
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
            
            self.logger.info(f"Cleanup completed: removed {len(old_messages)} messages and {len(old_agents)} agents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
            return False
            
    async def close(self) -> bool:
        """Cerrar y guardar todos los datos"""
        try:
            for data_type in self.data.keys():
                await self._save_data_type(data_type)
            return True
        except Exception as e:
            self.logger.error(f"Failed to close persistence: {e}")
            return False


# PERSISTENCE MANAGER


class PersistenceManager:
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.backend: PersistenceInterface = None
        self.auto_save_task = None
        self.logger = logging.getLogger("PersistenceManager")
        
    async def initialize(self) -> bool:
        """Inicializar el sistema de persistencia"""
        
        # Crear backend apropiado
        if self.config.backend == PersistenceBackend.SQLITE:
            self.backend = SQLitePersistence()
        elif self.config.backend == PersistenceBackend.JSON_FILE:
            self.backend = JSONFilePersistence()
        else:
            self.logger.error(f"Unsupported persistence backend: {self.config.backend}")
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
                self.logger.debug("Auto-save triggered")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Auto-save error: {e}")
                
    async def restore_framework_state(self, framework) -> bool:
        """Restaurar estado completo del framework"""
        try:
            # Cargar estado del framework
            framework_state = await self.backend.load_framework_state()
            
            if framework_state:
                self.logger.info("Restoring framework state from persistence")
                
                # Restaurar agentes si están configurados para auto-start
                if "agents" in framework_state:
                    await self._restore_agents(framework, framework_state["agents"])
                    
                # Restaurar recursos
                if "resources" in framework_state:
                    await self._restore_resources(framework, framework_state["resources"])
                    
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to restore framework state: {e}")
            
        return False
        
    async def _restore_agents(self, framework, agent_states: Dict[str, Any]):
        """Restaurar agentes desde estado persistido"""
        from core.specialized_agents import ExtendedAgentFactory # Se asume la existencia de esta clase
        
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
                        self.logger.info(f"Restored agent: {agent_id} ({namespace})")
                        
            except Exception as e:
                self.logger.error(f"Failed to restore agent {agent_id}: {e}")
                
    async def _restore_resources(self, framework, resource_ids: List[str]):
        """Restaurar recursos desde estado persistido"""
        for resource_id in resource_ids:
            try:
                resource = await self.backend.load_resource(resource_id)
                if resource:
                    await framework.resource_manager.create_resource(resource)
                    self.logger.info(f"Restored resource: {resource_id}")
            except Exception as e:
                self.logger.error(f"Failed to restore resource {resource_id}: {e}")
                
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
            
            self.logger.info(f"Saved complete framework state: {len(agents)} agents, {len(all_resources)} resources")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save full framework state: {e}")
            return False

    async def export_state_to_directory(self, output_dir: Path) -> bool:
        """Exporta el estado completo del framework a un directorio."""
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Exportar agentes
            agents_dir = output_dir / "agents"
            agents_dir.mkdir(exist_ok=True)
            for agent in self.framework.registry.list_all_agents():
                agent_state = await self.backend.load_agent_state(agent.id)
                if agent_state:
                    with open(agents_dir / f"{agent.id}.json", "w") as f:
                        json.dump(agent_state, f, indent=2)
            self.logger.info(f"Exported {len(self.framework.registry.list_all_agents())} agent states.")

            # Exportar recursos
            resources_dir = output_dir / "resources"
            resources_dir.mkdir(exist_ok=True)
            for resource in self.framework.resource_manager.list_all_resources():
                res_data = {
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
                with open(resources_dir / f"{resource.id}.json", "w") as f:
                    json.dump(res_data, f, indent=2)
            self.logger.info(f"Exported {len(self.framework.resource_manager.list_all_resources())} resources.")

            # Exportar estado general del framework (referencias a agentes/recursos)
            framework_state_meta_path = output_dir / "framework_state_meta.json"
            framework_state_meta = {
                "agents": {a.id: {"name": a.name, "namespace": a.namespace} for a in self.framework.registry.list_all_agents()},
                "resources": {r.id: {"name": r.name, "type": r.type.value} for r in self.framework.resource_manager.list_all_resources()},
                "last_export": datetime.now().isoformat()
            }
            with open(framework_state_meta_path, "w") as f:
                json.dump(framework_state_meta, f, indent=2)
            self.logger.info("Exported framework state metadata.")
            
            return True
        except Exception as e:
            self.logger.error(f"Error exporting framework state to directory {output_dir}: {e}")
            return False

    async def import_state_from_directory(self, input_dir: Path) -> bool:
        """Importa el estado completo del framework desde un directorio."""
        try:
            # Importar estado general del framework (metadata)
            framework_state_meta_path = input_dir / "framework_state_meta.json"
            if not framework_state_meta_path.exists():
                self.logger.error(f"Framework state metadata file not found at {framework_state_meta_path}")
                return False

            with open(framework_state_meta_path, "r") as f:
                framework_state_meta = json.load(f)
            self.logger.info("Importing framework state metadata.")

            # Importar agentes
            agents_dir = input_dir / "agents"
            if agents_dir.exists():
                for agent_file in agents_dir.glob("*.json"):
                    try:
                        with open(agent_file, "r") as f:
                            agent_data = json.load(f)
                        # Recrear el agente en el registro del framework y persistir su estado
                        # Esto asume que el AgentManager/Factory sabe cómo recrear el agente desde sus datos.
                        # Para una implementación completa, se requeriría más lógica de hidratación.
                        self.logger.debug(f"Importing agent: {agent_data.get('id')} - {agent_data.get('name')}")
                        # self.framework.agent_manager.create_agent_from_persisted_data(agent_data) # Esto necesitaría un método en AgentManager
                        # Por ahora, solo guardamos en la persistencia.
                        await self.backend.save_agent_state_from_dict(agent_data) # Necesita un método que reciba dict
                    except Exception as e:
                        self.logger.error(f"Error importing agent from {agent_file}: {e}")
                self.logger.info(f"Imported agents from {agents_dir}.")

            # Importar recursos
            resources_dir = input_dir / "resources"
            if resources_dir.exists():
                for resource_file in resources_dir.glob("*.json"):
                    try:
                        with open(resource_file, "r") as f:
                            resource_data = json.load(f)
                        # Recrear el objeto AgentResource y persistirlo
                        from core.autonomous_agent_framework import ResourceType
                        resource_obj = AgentResource(
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
                        await self.backend.save_resource(resource_obj)
                    except Exception as e:
                        self.logger.error(f"Error importing resource from {resource_file}: {e}")
                self.logger.info(f"Imported resources from {resources_dir}.")
            
            self.logger.info(f"Successfully imported state from directory {input_dir}.")
            return True
        except Exception as e:
            self.logger.error(f"Error importing framework state from directory {input_dir}: {e}")
            return False

    async def close(self) -> bool:
        """Cerrar el sistema de persistencia y tareas."""
        if self.auto_save_task:
            self.auto_save_task.cancel()
            try:
                await self.auto_save_task
            except asyncio.CancelledError:
                pass
        return await self.backend.close()

# Esto es parte de la refactorización anterior. Necesitaría un AgentFramework para ser un argumento
# para save_full_state, restore_framework_state, _restore_agents, _restore_resources, etc.

# Ejemplo de uso (simulado para la refactorización, esto iría en un archivo de demo/ejemplo)
async def example_usage():
    logger = logging.getLogger("PersistenceDemo")
    # Este ejemplo asume un AgentFramework simulado para evitar dependencias circulares
    # y para que el código sea directamente ejecutable para la demo de persistencia.
    class MockAgent:
        def __init__(self, agent_id, namespace, name, status, capabilities=None, metadata=None, created_at=None, last_heartbeat=None):
            self.id = agent_id
            self.namespace = namespace
            self.name = name
            self.status = status
            self.capabilities = capabilities if capabilities is not None else []
            self.metadata = metadata if metadata is not None else {}
            self.created_at = created_at if created_at is not None else datetime.now()
            self.last_heartbeat = last_heartbeat if last_heartbeat is not None else datetime.now()
            self.inbox = asyncio.Queue() # Dummy for serialization
            self.outbox = asyncio.Queue() # Dummy for serialization

    class MockResourceManager:
        def __init__(self):
            self._resources = {}
        def list_all_resources(self):
            return list(self._resources.values())
        async def create_resource(self, resource):
            self._resources[resource.id] = resource
            return True
        def find_resources_by_owner(self, owner_id):
            return [res for res in self._resources.values() if res.owner_agent_id == owner_id]

    class MockAgentRegistry:
        def __init__(self):
            self._agents = {}
        def list_all_agents(self):
            return list(self._agents.values())
        def get_agent(self, agent_id):
            return self._agents.get(agent_id)

    class MockFramework:
        def __init__(self):
            self.registry = MockAgentRegistry()
            self.resource_manager = MockResourceManager()

    mock_framework = MockFramework()

    config = PersistenceConfig(backend=PersistenceBackend.SQLITE, connection_string="demo_framework.db", auto_save_interval=2)
    persistence_manager = PersistenceManager(config)
    await persistence_manager.initialize()

    # Simular algunos agentes y recursos
    agent1 = MockAgent("agent1_id", "namespace.test", "TestAgent1", AgentStatus.ACTIVE,
                       metadata={"config_key": "value1"}, created_at=datetime.now())
    agent2 = MockAgent("agent2_id", "namespace.test", "TestAgent2", AgentStatus.TERMINATED,
                       metadata={"config_key": "value2"}, created_at=datetime.now() - timedelta(days=50))
    mock_framework.registry._agents["agent1_id"] = agent1
    mock_framework.registry._agents["agent2_id"] = agent2

    resource1 = AgentResource(id="res1", type=ResourceType.DATA, name="data1", namespace="data.prod",
                              data={"value": 100}, owner_agent_id="agent1_id",
                              created_at=datetime.now(), updated_at=datetime.now())
    resource2 = AgentResource(id="res2", type=ResourceType.CODE, name="script1", namespace="code.util",
                              data={"content": "print('hello')"}, owner_agent_id="agent1_id",
                              created_at=datetime.now(), updated_at=datetime.now())
    await mock_framework.resource_manager.create_resource(resource1)
    await mock_framework.resource_manager.create_resource(resource2)


    # Demo de guardado
    logger.info("💾 Saving full framework state...")
    save_success = await persistence_manager.save_full_state(mock_framework)
    logger.info(f"Save successful: {save_success}")

    # Simular una "limpieza" en memoria del framework para restaurar
    mock_framework.registry._agents = {}
    mock_framework.resource_manager._resources = {}
    logger.info("🧹 Simulated clearing framework state in memory.")
    logger.info(f"Agents in memory after clear: {len(mock_framework.registry.list_all_agents())}")
    logger.info(f"Resources in memory after clear: {len(mock_framework.resource_manager.list_all_resources())}")

    # Demo de restauración
    logger.info("🔄 Restoring framework state...")
    restore_success = await persistence_manager.restore_framework_state(mock_framework)
    logger.info(f"Restore successful: {restore_success}")
    logger.info(f"Agents in memory after restore: {len(mock_framework.registry.list_all_agents())}")
    logger.info(f"Resources in memory after restore: {len(mock_framework.resource_manager.list_all_resources())}")

    # Verificar si los agentes y recursos restaurados son accesibles y correctos (básico)
    restored_agent1 = mock_framework.registry.get_agent("agent1_id")
    if restored_agent1:
        logger.info(f"Restored Agent 1: {restored_agent1.name} (Status: {restored_agent1.status.value})")
    else:
        logger.warning("Agent 1 not restored.")
    
    restored_resource1 = await persistence_manager.backend.load_resource("res1") # Se carga del backend no del resource_manager restaurado
    if restored_resource1:
        logger.info(f"Restored Resource 1 data: {restored_resource1.data}")
    else:
        logger.warning("Resource 1 not restored.")

    # Demo de carga de mensajes para un agente (sin mensajes guardados en este flujo, solo demo de la función)
    logger.info("💬 Loading messages for agent1_id (expect empty for this demo)...")
    messages = await persistence_manager.backend.load_messages("agent1_id", limit=5)
    logger.info(f"Loaded {len(messages)} messages for agent1_id.")

    # Demo de limpieza
    logger.info("🧹 Running cleanup (removing agents terminated 50+ days ago)...")
    cleanup_success = await persistence_manager.backend.cleanup(older_than_days=40)
    logger.info(f"Cleanup successful: {cleanup_success}")

    # Verificar si el agente terminado fue eliminado
    restored_agent2_after_cleanup = await persistence_manager.backend.load_agent_state("agent2_id")
    if restored_agent2_after_cleanup:
        logger.warning("Agent 2 still exists after cleanup (expected removal).")
    else:
        logger.info("Agent 2 successfully removed by cleanup.")

    await persistence_manager.close()
    logger.info("👋 Demo completed and persistence closed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(example_usage())