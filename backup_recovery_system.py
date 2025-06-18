"""
backup_recovery_system.py - Sistema de backup y recuperación ante desastres
"""

import asyncio
import json
import gzip
import tarfile
import shutil
import os
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import tempfile
import boto3
from azure.storage.blob import BlobServiceClient
import paramiko

from autonomous_agent_framework import AgentFramework, BaseAgent, AgentResource
from persistence_system import PersistenceManager

# ================================
# BACKUP MODELS
# ================================

class BackupType(Enum):
    """Tipos de backup"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    """Estados de backup"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"

class StorageBackend(Enum):
    """Backends de almacenamiento"""
    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    SSH = "ssh"
    FTP = "ftp"

@dataclass
class BackupMetadata:
    """Metadatos de backup"""
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime
    completed_at: Optional[datetime]
    size_bytes: int
    checksum: str
    file_path: str
    storage_backend: StorageBackend
    compression: bool
    encryption: bool
    agent_count: int
    resource_count: int
    framework_version: str
    description: str = ""
    error_message: Optional[str] = None

@dataclass
class RestorePoint:
    """Punto de restauración"""
    restore_id: str
    backup_id: str
    created_at: datetime
    description: str
    agents_snapshot: Dict[str, Any]
    resources_snapshot: Dict[str, Any]
    framework_state: Dict[str, Any]
    verified: bool = False

# ================================
# BACKUP ENGINE
# ================================

class BackupEngine:
    """Motor de backup del framework"""
    
    def __init__(self, framework: AgentFramework, persistence_manager: PersistenceManager):
        self.framework = framework
        self.persistence_manager = persistence_manager
        self.backup_dir = Path("./backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.backup_history: List[BackupMetadata] = []
        self.max_backups = 50
        self.compression_enabled = True
        self.encryption_key = None
        
    def set_encryption_key(self, key: str):
        """Configurar clave de encriptación"""
        self.encryption_key = key
        
    async def create_full_backup(self, description: str = "") -> BackupMetadata:
        """Crear backup completo"""
        backup_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.FULL,
            status=BackupStatus.PENDING,
            created_at=datetime.now(),
            completed_at=None,
            size_bytes=0,
            checksum="",
            file_path="",
            storage_backend=StorageBackend.LOCAL,
            compression=self.compression_enabled,
            encryption=self.encryption_key is not None,
            agent_count=0,
            resource_count=0,
            framework_version="1.0.0",
            description=description
        )
        
        try:
            backup_metadata.status = BackupStatus.RUNNING
            logging.info(f"Starting full backup: {backup_id}")
            
            # Crear directorio temporal para el backup
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Recopilar datos del framework
                framework_data = await self._collect_framework_data()
                backup_metadata.agent_count = len(framework_data["agents"])
                backup_metadata.resource_count = len(framework_data["resources"])
                
                # Guardar datos en archivos JSON
                with open(temp_path / "framework_state.json", "w") as f:
                    json.dump(framework_data["framework_state"], f, indent=2, default=str)
                    
                with open(temp_path / "agents.json", "w") as f:
                    json.dump(framework_data["agents"], f, indent=2, default=str)
                    
                with open(temp_path / "resources.json", "w") as f:
                    json.dump(framework_data["resources"], f, indent=2, default=str)
                    
                with open(temp_path / "messages.json", "w") as f:
                    json.dump(framework_data["messages"], f, indent=2, default=str)
                    
                # Incluir datos de persistencia si existen
                if hasattr(self.persistence_manager, 'backend'):
                    await self._backup_persistence_data(temp_path)
                    
                # Crear archivo de metadatos
                metadata_dict = asdict(backup_metadata)
                with open(temp_path / "backup_metadata.json", "w") as f:
                    json.dump(metadata_dict, f, indent=2, default=str)
                    
                # Crear archivo tar (comprimido si está habilitado)
                backup_filename = f"{backup_id}.tar"
                if self.compression_enabled:
                    backup_filename += ".gz"
                    
                backup_path = self.backup_dir / backup_filename
                
                if self.compression_enabled:
                    with tarfile.open(backup_path, "w:gz") as tar:
                        tar.add(temp_path, arcname=backup_id)
                else:
                    with tarfile.open(backup_path, "w") as tar:
                        tar.add(temp_path, arcname=backup_id)
                        
                # Encriptar si está habilitado
                if self.encryption_key:
                    encrypted_path = self._encrypt_file(backup_path)
                    backup_path.unlink()  # Eliminar archivo sin encriptar
                    backup_path = encrypted_path
                    
                # Calcular checksum
                checksum = self._calculate_checksum(backup_path)
                
                # Actualizar metadatos
                backup_metadata.status = BackupStatus.COMPLETED
                backup_metadata.completed_at = datetime.now()
                backup_metadata.size_bytes = backup_path.stat().st_size
                backup_metadata.checksum = checksum
                backup_metadata.file_path = str(backup_path)
                
            self.backup_history.append(backup_metadata)
            self._cleanup_old_backups()
            
            logging.info(f"Full backup completed: {backup_id} ({backup_metadata.size_bytes} bytes)")
            return backup_metadata
            
        except Exception as e:
            backup_metadata.status = BackupStatus.FAILED
            backup_metadata.error_message = str(e)
            backup_metadata.completed_at = datetime.now()
            
            logging.error(f"Full backup failed: {backup_id} - {e}")
            return backup_metadata
            
    async def create_incremental_backup(self, base_backup_id: str, description: str = "") -> BackupMetadata:
        """Crear backup incremental"""
        backup_id = f"incr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Buscar backup base
        base_backup = self._find_backup(base_backup_id)
        if not base_backup:
            raise ValueError(f"Base backup not found: {base_backup_id}")
            
        backup_metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.INCREMENTAL,
            status=BackupStatus.PENDING,
            created_at=datetime.now(),
            completed_at=None,
            size_bytes=0,
            checksum="",
            file_path="",
            storage_backend=StorageBackend.LOCAL,
            compression=self.compression_enabled,
            encryption=self.encryption_key is not None,
            agent_count=0,
            resource_count=0,
            framework_version="1.0.0",
            description=f"Incremental from {base_backup_id}. {description}"
        )
        
        try:
            backup_metadata.status = BackupStatus.RUNNING
            logging.info(f"Starting incremental backup: {backup_id}")
            
            # Obtener cambios desde el backup base
            changes = await self._get_changes_since_backup(base_backup)
            
            if not changes["has_changes"]:
                backup_metadata.status = BackupStatus.COMPLETED
                backup_metadata.completed_at = datetime.now()
                backup_metadata.description += " (no changes)"
                logging.info(f"No changes detected for incremental backup: {backup_id}")
                return backup_metadata
                
            # Crear archivo de cambios
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Guardar solo los cambios
                with open(temp_path / "changes.json", "w") as f:
                    json.dump(changes, f, indent=2, default=str)
                    
                with open(temp_path / "base_backup_id.txt", "w") as f:
                    f.write(base_backup_id)
                    
                # Crear archivo tar
                backup_filename = f"{backup_id}.tar"
                if self.compression_enabled:
                    backup_filename += ".gz"
                    
                backup_path = self.backup_dir / backup_filename
                
                if self.compression_enabled:
                    with tarfile.open(backup_path, "w:gz") as tar:
                        tar.add(temp_path, arcname=backup_id)
                else:
                    with tarfile.open(backup_path, "w") as tar:
                        tar.add(temp_path, arcname=backup_id)
                        
                # Calcular checksum
                checksum = self._calculate_checksum(backup_path)
                
                # Actualizar metadatos
                backup_metadata.status = BackupStatus.COMPLETED
                backup_metadata.completed_at = datetime.now()
                backup_metadata.size_bytes = backup_path.stat().st_size
                backup_metadata.checksum = checksum
                backup_metadata.file_path = str(backup_path)
                backup_metadata.agent_count = len(changes.get("changed_agents", {}))
                backup_metadata.resource_count = len(changes.get("changed_resources", {}))
                
            self.backup_history.append(backup_metadata)
            logging.info(f"Incremental backup completed: {backup_id}")
            return backup_metadata
            
        except Exception as e:
            backup_metadata.status = BackupStatus.FAILED
            backup_metadata.error_message = str(e)
            backup_metadata.completed_at = datetime.now()
            
            logging.error(f"Incremental backup failed: {backup_id} - {e}")
            return backup_metadata
            
    async def create_snapshot(self, description: str = "") -> RestorePoint:
        """Crear snapshot/punto de restauración en memoria"""
        restore_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Recopilar estado actual
            framework_data = await self._collect_framework_data()
            
            restore_point = RestorePoint(
                restore_id=restore_id,
                backup_id="",  # No hay archivo físico
                created_at=datetime.now(),
                description=description,
                agents_snapshot=framework_data["agents"],
                resources_snapshot=framework_data["resources"],
                framework_state=framework_data["framework_state"]
            )
            
            logging.info(f"Snapshot created: {restore_id}")
            return restore_point
            
        except Exception as e:
            logging.error(f"Failed to create snapshot: {e}")
            raise
            
    async def _collect_framework_data(self) -> Dict[str, Any]:
        """Recopilar todos los datos del framework"""
        
        # Datos de agentes
        agents = self.framework.registry.list_all_agents()
        agents_data = {}
        
        for agent in agents:
            agents_data[agent.id] = {
                "id": agent.id,
                "name": agent.name,
                "namespace": agent.namespace,
                "status": agent.status.value,
                "created_at": agent.created_at.isoformat(),
                "last_heartbeat": agent.last_heartbeat.isoformat(),
                "metadata": agent.metadata,
                "capabilities": [
                    {
                        "name": cap.name,
                        "namespace": cap.namespace,
                        "description": cap.description,
                        "input_schema": cap.input_schema,
                        "output_schema": cap.output_schema
                    }
                    for cap in agent.capabilities
                ]
            }
            
        # Datos de recursos
        all_resources = []
        for agent in agents:
            agent_resources = self.framework.resource_manager.find_resources_by_owner(agent.id)
            all_resources.extend(agent_resources)
            
        resources_data = {}
        for resource in all_resources:
            resources_data[resource.id] = {
                "id": resource.id,
                "name": resource.name,
                "type": resource.type.value,
                "namespace": resource.namespace,
                "data": resource.data,
                "owner_agent_id": resource.owner_agent_id,
                "metadata": resource.metadata,
                "created_at": resource.created_at.isoformat(),
                "updated_at": resource.updated_at.isoformat()
            }
            
        # Estado del framework
        framework_state = {
            "backup_timestamp": datetime.now().isoformat(),
            "total_agents": len(agents),
            "total_resources": len(all_resources),
            "framework_version": "1.0.0",
            "registry_state": {
                "namespaces": {
                    ns: [agent.id for agent in agents_list]
                    for ns, agents_list in self.framework.registry._namespaces.items()
                }
            }
        }
        
        # Mensajes recientes (si hay persistencia)
        messages_data = {}
        if hasattr(self.persistence_manager, 'backend'):
            try:
                # Obtener mensajes recientes de cada agente
                for agent in agents:
                    recent_messages = await self.persistence_manager.backend.load_messages(agent.id, limit=100)
                    messages_data[agent.id] = [
                        {
                            "id": msg.id,
                            "sender_id": msg.sender_id,
                            "receiver_id": msg.receiver_id,
                            "message_type": msg.message_type.value,
                            "action": msg.action,
                            "payload": msg.payload,
                            "timestamp": msg.timestamp.isoformat(),
                            "correlation_id": msg.correlation_id,
                            "response_required": msg.response_required
                        }
                        for msg in recent_messages
                    ]
            except Exception as e:
                logging.warning(f"Could not backup messages: {e}")
                
        return {
            "agents": agents_data,
            "resources": resources_data,
            "framework_state": framework_state,
            "messages": messages_data
        }
        
    async def _backup_persistence_data(self, backup_path: Path):
        """Backup de datos de persistencia"""
        try:
            if hasattr(self.persistence_manager.backend, 'db_path'):
                # SQLite database
                db_path = Path(self.persistence_manager.backend.db_path)
                if db_path.exists():
                    shutil.copy2(db_path, backup_path / "framework.db")
                    
            elif hasattr(self.persistence_manager.backend, 'data_dir'):
                # JSON files
                data_dir = self.persistence_manager.backend.data_dir
                if data_dir.exists():
                    shutil.copytree(data_dir, backup_path / "persistence_data")
                    
        except Exception as e:
            logging.warning(f"Could not backup persistence data: {e}")
            
    async def _get_changes_since_backup(self, base_backup: BackupMetadata) -> Dict[str, Any]:
        """Obtener cambios desde un backup base"""
        
        # Cargar datos del backup base
        base_data = await self._load_backup_data(base_backup)
        
        # Obtener datos actuales
        current_data = await self._collect_framework_data()
        
        changes = {
            "has_changes": False,
            "changed_agents": {},
            "new_agents": {},
            "deleted_agents": {},
            "changed_resources": {},
            "new_resources": {},
            "deleted_resources": {},
            "framework_state_changes": {}
        }
        
        # Comparar agentes
        base_agents = base_data.get("agents", {})
        current_agents = current_data.get("agents", {})
        
        # Agentes nuevos
        for agent_id, agent_data in current_agents.items():
            if agent_id not in base_agents:
                changes["new_agents"][agent_id] = agent_data
                changes["has_changes"] = True
                
        # Agentes eliminados
        for agent_id in base_agents:
            if agent_id not in current_agents:
                changes["deleted_agents"][agent_id] = base_agents[agent_id]
                changes["has_changes"] = True
                
        # Agentes modificados
        for agent_id, current_agent in current_agents.items():
            if agent_id in base_agents:
                base_agent = base_agents[agent_id]
                if self._agent_changed(base_agent, current_agent):
                    changes["changed_agents"][agent_id] = {
                        "before": base_agent,
                        "after": current_agent
                    }
                    changes["has_changes"] = True
                    
        # Comparar recursos (similar lógica)
        base_resources = base_data.get("resources", {})
        current_resources = current_data.get("resources", {})
        
        for resource_id, resource_data in current_resources.items():
            if resource_id not in base_resources:
                changes["new_resources"][resource_id] = resource_data
                changes["has_changes"] = True
            elif self._resource_changed(base_resources[resource_id], resource_data):
                changes["changed_resources"][resource_id] = {
                    "before": base_resources[resource_id],
                    "after": resource_data
                }
                changes["has_changes"] = True
                
        return changes
        
    def _agent_changed(self, agent1: Dict[str, Any], agent2: Dict[str, Any]) -> bool:
        """Verificar si un agente ha cambiado"""
        # Comparar campos importantes (excluyendo timestamps dinámicos)
        compare_fields = ["name", "namespace", "status", "metadata"]
        
        for field in compare_fields:
            if agent1.get(field) != agent2.get(field):
                return True
                
        return False
        
    def _resource_changed(self, resource1: Dict[str, Any], resource2: Dict[str, Any]) -> bool:
        """Verificar si un recurso ha cambiado"""
        # Comparar campos importantes
        compare_fields = ["name", "type", "namespace", "data", "metadata"]
        
        for field in compare_fields:
            if resource1.get(field) != resource2.get(field):
                return True
                
        return False
        
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calcular checksum SHA256 de un archivo"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
                
        return sha256_hash.hexdigest()
        
    def _encrypt_file(self, file_path: Path) -> Path:
        """Encriptar archivo (implementación simplificada)"""
        # En producción, usar una librería de criptografía robusta
        from cryptography.fernet import Fernet
        
        # Generar clave desde la clave configurada
        key = Fernet.generate_key()  # En producción, derivar de self.encryption_key
        cipher = Fernet(key)
        
        encrypted_path = file_path.with_suffix(file_path.suffix + ".encrypted")
        
        with open(file_path, "rb") as f:
            encrypted_data = cipher.encrypt(f.read())
            
        with open(encrypted_path, "wb") as f:
            f.write(encrypted_data)
            
        return encrypted_path
        
    def _cleanup_old_backups(self):
        """Limpiar backups antiguos"""
        if len(self.backup_history) > self.max_backups:
            # Ordenar por fecha y eliminar los más antiguos
            self.backup_history.sort(key=lambda x: x.created_at)
            
            backups_to_remove = self.backup_history[:-self.max_backups]
            
            for backup in backups_to_remove:
                try:
                    if Path(backup.file_path).exists():
                        Path(backup.file_path).unlink()
                        logging.info(f"Deleted old backup: {backup.backup_id}")
                except Exception as e:
                    logging.warning(f"Could not delete backup {backup.backup_id}: {e}")
                    
            self.backup_history = self.backup_history[-self.max_backups:]
            
    def _find_backup(self, backup_id: str) -> Optional[BackupMetadata]:
        """Encontrar backup por ID"""
        for backup in self.backup_history:
            if backup.backup_id == backup_id:
                return backup
        return None
        
    async def _load_backup_data(self, backup_metadata: BackupMetadata) -> Dict[str, Any]:
        """Cargar datos de un backup"""
        backup_path = Path(backup_metadata.file_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
        # Desencriptar si es necesario
        if backup_metadata.encryption:
            # Implementar desencriptación
            pass
            
        # Extraer archivo tar
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            with tarfile.open(backup_path, "r:gz" if backup_metadata.compression else "r") as tar:
                tar.extractall(temp_path)
                
            # Cargar datos JSON
            backup_data_dir = temp_path / backup_metadata.backup_id
            
            data = {}
            
            for json_file in ["framework_state.json", "agents.json", "resources.json", "messages.json"]:
                file_path = backup_data_dir / json_file
                if file_path.exists():
                    with open(file_path) as f:
                        data[json_file.replace(".json", "")] = json.load(f)
                        
            return data
            
    def get_backup_history(self) -> List[BackupMetadata]:
        """Obtener historial de backups"""
        return sorted(self.backup_history, key=lambda x: x.created_at, reverse=True)

# ================================
# RECOVERY ENGINE
# ================================

class RecoveryEngine:
    """Motor de recuperación"""
    
    def __init__(self, framework: AgentFramework, persistence_manager: PersistenceManager,
                 backup_engine: BackupEngine):
        self.framework = framework
        self.persistence_manager = persistence_manager
        self.backup_engine = backup_engine
        
    async def restore_from_backup(self, backup_id: str, selective_restore: Dict[str, bool] = None) -> bool:
        """Restaurar desde backup"""
        
        backup_metadata = self.backup_engine._find_backup(backup_id)
        if not backup_metadata:
            raise ValueError(f"Backup not found: {backup_id}")
            
        if backup_metadata.status != BackupStatus.COMPLETED:
            raise ValueError(f"Backup is not in completed state: {backup_metadata.status}")
            
        selective_restore = selective_restore or {
            "agents": True,
            "resources": True,
            "framework_state": True,
            "messages": True
        }
        
        try:
            logging.info(f"Starting restore from backup: {backup_id}")
            
            # Verificar integridad del backup
            if not await self._verify_backup_integrity(backup_metadata):
                raise ValueError("Backup integrity verification failed")
                
            # Cargar datos del backup
            backup_data = await self.backup_engine._load_backup_data(backup_metadata)
            
            # Detener agentes actuales
            await self._stop_all_agents()
            
            # Restaurar componentes selectivamente
            if selective_restore.get("framework_state", True):
                await self._restore_framework_state(backup_data.get("framework_state", {}))
                
            if selective_restore.get("agents", True):
                await self._restore_agents(backup_data.get("agents", {}))
                
            if selective_restore.get("resources", True):
                await self._restore_resources(backup_data.get("resources", {}))
                
            if selective_restore.get("messages", True):
                await self._restore_messages(backup_data.get("messages", {}))
                
            logging.info(f"Restore completed successfully: {backup_id}")
            return True
            
        except Exception as e:
            logging.error(f"Restore failed: {backup_id} - {e}")
            raise
            
    async def restore_from_snapshot(self, restore_point: RestorePoint) -> bool:
        """Restaurar desde snapshot en memoria"""
        try:
            logging.info(f"Starting restore from snapshot: {restore_point.restore_id}")
            
            # Detener agentes actuales
            await self._stop_all_agents()
            
            # Restaurar desde snapshot
            await self._restore_framework_state(restore_point.framework_state)
            await self._restore_agents(restore_point.agents_snapshot)
            await self._restore_resources(restore_point.resources_snapshot)
            
            logging.info(f"Snapshot restore completed: {restore_point.restore_id}")
            return True
            
        except Exception as e:
            logging.error(f"Snapshot restore failed: {e}")
            raise
            
    async def _verify_backup_integrity(self, backup_metadata: BackupMetadata) -> bool:
        """Verificar integridad del backup"""
        backup_path = Path(backup_metadata.file_path)
        
        if not backup_path.exists():
            return False
            
        # Verificar checksum
        current_checksum = self.backup_engine._calculate_checksum(backup_path)
        
        if current_checksum != backup_metadata.checksum:
            logging.error(f"Backup checksum mismatch: expected {backup_metadata.checksum}, got {current_checksum}")
            return False
            
        # Verificar que el archivo se puede abrir
        try:
            with tarfile.open(backup_path, "r:gz" if backup_metadata.compression else "r") as tar:
                tar.getnames()  # Verificar que se puede leer
            return True
        except Exception as e:
            logging.error(f"Backup file corruption detected: {e}")
            return False
            
    async def _stop_all_agents(self):
        """Detener todos los agentes"""
        agents = self.framework.registry.list_all_agents()
        
        for agent in agents:
            try:
                await agent.stop()
            except Exception as e:
                logging.warning(f"Error stopping agent {agent.id}: {e}")
                
    async def _restore_framework_state(self, framework_state: Dict[str, Any]):
        """Restaurar estado del framework"""
        # En una implementación completa, restaurarías configuraciones específicas
        logging.info("Framework state restored")
        
    async def _restore_agents(self, agents_data: Dict[str, Any]):
        """Restaurar agentes"""
        from specialized_agents import ExtendedAgentFactory
        
        for agent_id, agent_data in agents_data.items():
            try:
                # Crear agente
                namespace = agent_data["namespace"]
                name = agent_data["name"]
                
                if namespace in ExtendedAgentFactory.AGENT_CLASSES:
                    agent_class = ExtendedAgentFactory.AGENT_CLASSES[namespace]
                    agent = agent_class(name, self.framework)
                    
                    # Restaurar metadata
                    agent.metadata = agent_data.get("metadata", {})
                    
                    # Iniciar agente
                    await agent.start()
                    
                    logging.info(f"Restored agent: {agent_id} ({namespace})")
                    
            except Exception as e:
                logging.error(f"Failed to restore agent {agent_id}: {e}")
                
    async def _restore_resources(self, resources_data: Dict[str, Any]):
        """Restaurar recursos"""
        from autonomous_agent_framework import AgentResource, ResourceType
        
        for resource_id, resource_data in resources_data.items():
            try:
                # Verificar que el agente propietario existe
                owner_id = resource_data["owner_agent_id"]
                owner_agent = self.framework.registry.get_agent(owner_id)
                
                if owner_agent:
                    resource = AgentResource(
                        id=resource_data["id"],
                        type=ResourceType(resource_data["type"]),
                        name=resource_data["name"],
                        namespace=resource_data["namespace"],
                        data=resource_data["data"],
                        owner_agent_id=resource_data["owner_agent_id"],
                        metadata=resource_data.get("metadata", {}),
                        created_at=datetime.fromisoformat(resource_data["created_at"]),
                        updated_at=datetime.fromisoformat(resource_data["updated_at"])
                    )
                    
                    await self.framework.resource_manager.create_resource(resource)
                    logging.info(f"Restored resource: {resource_id}")
                    
            except Exception as e:
                logging.error(f"Failed to restore resource {resource_id}: {e}")
                
    async def _restore_messages(self, messages_data: Dict[str, Any]):
        """Restaurar mensajes"""
        if not hasattr(self.persistence_manager, 'backend'):
            return
            
        for agent_id, messages in messages_data.items():
            try:
                for msg_data in messages:
                    from autonomous_agent_framework import AgentMessage, MessageType
                    
                    message = AgentMessage(
                        id=msg_data["id"],
                        sender_id=msg_data["sender_id"],
                        receiver_id=msg_data["receiver_id"],
                        message_type=MessageType(msg_data["message_type"]),
                        action=msg_data["action"],
                        payload=msg_data["payload"],
                        timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                        correlation_id=msg_data.get("correlation_id"),
                        response_required=msg_data.get("response_required", True)
                    )
                    
                    await self.persistence_manager.backend.save_message(message)
                    
                logging.info(f"Restored {len(messages)} messages for agent {agent_id}")
                
            except Exception as e:
                logging.error(f"Failed to restore messages for agent {agent_id}: {e}")

# ================================
# DISASTER RECOVERY ORCHESTRATOR
# ================================

class DisasterRecoveryOrchestrator:
    """Orquestador de recuperación ante desastres"""
    
    def __init__(self, framework: AgentFramework, persistence_manager: PersistenceManager):
        self.framework = framework
        self.persistence_manager = persistence_manager
        self.backup_engine = BackupEngine(framework, persistence_manager)
        self.recovery_engine = RecoveryEngine(framework, persistence_manager, self.backup_engine)
        
        # Configuración de backups automáticos
        self.auto_backup_enabled = False
        self.backup_interval_hours = 6
        self.backup_task = None
        
        # Snapshots en memoria para recuperación rápida
        self.restore_points: List[RestorePoint] = []
        self.max_restore_points = 10
        
    async def start_auto_backup(self):
        """Iniciar backups automáticos"""
        self.auto_backup_enabled = True
        self.backup_task = asyncio.create_task(self._auto_backup_loop())
        logging.info(f"Auto-backup started (interval: {self.backup_interval_hours}h)")
        
    async def stop_auto_backup(self):
        """Detener backups automáticos"""
        self.auto_backup_enabled = False
        if self.backup_task:
            self.backup_task.cancel()
        logging.info("Auto-backup stopped")
        
    async def _auto_backup_loop(self):
        """Loop de backups automáticos"""
        while self.auto_backup_enabled:
            try:
                await asyncio.sleep(self.backup_interval_hours * 3600)
                
                if self.auto_backup_enabled:
                    await self.create_scheduled_backup()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Auto-backup error: {e}")
                await asyncio.sleep(300)  # Esperar 5 minutos antes de reintentar
                
    async def create_scheduled_backup(self) -> BackupMetadata:
        """Crear backup programado"""
        description = f"Scheduled backup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return await self.backup_engine.create_full_backup(description)
        
    async def create_restore_point(self, description: str = "") -> RestorePoint:
        """Crear punto de restauración"""
        restore_point = await self.backup_engine.create_snapshot(description)
        
        self.restore_points.append(restore_point)
        
        # Mantener límite de puntos de restauración
        if len(self.restore_points) > self.max_restore_points:
            self.restore_points.pop(0)
            
        return restore_point
        
    async def disaster_recovery_plan(self, scenario: str) -> Dict[str, Any]:
        """Ejecutar plan de recuperación ante desastres"""
        
        recovery_plans = {
            "agent_failure": self._recover_from_agent_failure,
            "data_corruption": self._recover_from_data_corruption,
            "system_crash": self._recover_from_system_crash,
            "complete_failure": self._recover_from_complete_failure
        }
        
        if scenario not in recovery_plans:
            raise ValueError(f"Unknown disaster scenario: {scenario}")
            
        logging.info(f"Executing disaster recovery plan: {scenario}")
        
        try:
            result = await recovery_plans[scenario]()
            logging.info(f"Disaster recovery completed: {scenario}")
            return {"success": True, "scenario": scenario, "result": result}
            
        except Exception as e:
            logging.error(f"Disaster recovery failed: {scenario} - {e}")
            return {"success": False, "scenario": scenario, "error": str(e)}
            
    async def _recover_from_agent_failure(self) -> Dict[str, Any]:
        """Recuperar de fallo de agentes"""
        
        # Identificar agentes con problemas
        agents = self.framework.registry.list_all_agents()
        failed_agents = [agent for agent in agents if agent.status.value in ["error", "terminated"]]
        
        recovery_actions = []
        
        for agent in failed_agents:
            try:
                # Intentar reiniciar agente
                await agent.start()
                recovery_actions.append(f"Restarted agent: {agent.id}")
                
            except Exception as e:
                recovery_actions.append(f"Failed to restart agent {agent.id}: {e}")
                
        return {"failed_agents": len(failed_agents), "actions": recovery_actions}
        
    async def _recover_from_data_corruption(self) -> Dict[str, Any]:
        """Recuperar de corrupción de datos"""
        
        # Buscar backup más reciente válido
        backups = self.backup_engine.get_backup_history()
        valid_backup = None
        
        for backup in backups:
            if backup.status == BackupStatus.COMPLETED:
                if await self.recovery_engine._verify_backup_integrity(backup):
                    valid_backup = backup
                    break
                    
        if not valid_backup:
            raise ValueError("No valid backup found for data recovery")
            
        # Restaurar desde backup
        await self.recovery_engine.restore_from_backup(
            valid_backup.backup_id,
            {"agents": True, "resources": True, "messages": True, "framework_state": False}
        )
        
        return {"restored_from": valid_backup.backup_id, "backup_date": valid_backup.created_at.isoformat()}
        
    async def _recover_from_system_crash(self) -> Dict[str, Any]:
        """Recuperar de crash del sistema"""
        
        # Usar punto de restauración más reciente si está disponible
        if self.restore_points:
            latest_restore_point = self.restore_points[-1]
            await self.recovery_engine.restore_from_snapshot(latest_restore_point)
            
            return {
                "restored_from": "restore_point",
                "restore_point_id": latest_restore_point.restore_id,
                "restore_point_date": latest_restore_point.created_at.isoformat()
            }
        else:
            # Usar backup más reciente
            return await self._recover_from_data_corruption()
            
    async def _recover_from_complete_failure(self) -> Dict[str, Any]:
        """Recuperar de fallo completo"""
        
        # Buscar backup completo más reciente
        backups = self.backup_engine.get_backup_history()
        full_backup = None
        
        for backup in backups:
            if (backup.backup_type == BackupType.FULL and 
                backup.status == BackupStatus.COMPLETED):
                if await self.recovery_engine._verify_backup_integrity(backup):
                    full_backup = backup
                    break
                    
        if not full_backup:
            raise ValueError("No valid full backup found for complete recovery")
            
        # Restauración completa
        await self.recovery_engine.restore_from_backup(full_backup.backup_id)
        
        return {
            "restored_from": full_backup.backup_id,
            "backup_date": full_backup.created_at.isoformat(),
            "backup_type": full_backup.backup_type.value
        }
        
    def get_recovery_status(self) -> Dict[str, Any]:
        """Obtener estado del sistema de recuperación"""
        
        backups = self.backup_engine.get_backup_history()
        
        return {
            "auto_backup_enabled": self.auto_backup_enabled,
            "backup_interval_hours": self.backup_interval_hours,
            "total_backups": len(backups),
            "latest_backup": backups[0].created_at.isoformat() if backups else None,
            "restore_points": len(self.restore_points),
            "latest_restore_point": self.restore_points[-1].created_at.isoformat() if self.restore_points else None,
            "backup_engine_status": "active" if self.backup_engine else "inactive"
        }

# ================================
# EXAMPLE USAGE
# ================================

async def backup_recovery_demo():
    """Demo del sistema de backup y recovery"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("💾 Backup & Recovery System Demo")
    print("="*60)
    
    # Crear framework y componentes
    from autonomous_agent_framework import AgentFramework
    from specialized_agents import ExtendedAgentFactory
    from persistence_system import PersistenceFactory, PersistenceBackend
    
    framework = AgentFramework()
    await framework.start()
    
    # Configurar persistencia
    persistence_manager = PersistenceFactory.create_persistence_manager(
        backend=PersistenceBackend.SQLITE,
        connection_string="demo_backup.db"
    )
    await persistence_manager.initialize()
    
    # Crear algunos agentes
    strategist = ExtendedAgentFactory.create_agent("agent.planning.strategist", "strategist", framework)
    generator = ExtendedAgentFactory.create_agent("agent.build.code.generator", "generator", framework)
    
    await strategist.start()
    await generator.start()
    
    # Crear algunos recursos
    from autonomous_agent_framework import AgentResource, ResourceType
    test_resource = AgentResource(
        type=ResourceType.CODE,
        name="demo_code",
        namespace="resource.demo",
        data={"code": "print('Hello, World!')", "language": "python"},
        owner_agent_id=strategist.id
    )
    await framework.resource_manager.create_resource(test_resource)
    
    # Crear sistema de backup y recovery
    dr_orchestrator = DisasterRecoveryOrchestrator(framework, persistence_manager)
    
    print(f"\n✅ Initial setup completed")
    print(f"   Agents: {len(framework.registry.list_all_agents())}")
    print(f"   Resources: 1")
    
    # Demo 1: Crear backup completo
    print(f"\n1. Creating full backup...")
    full_backup = await dr_orchestrator.backup_engine.create_full_backup("Demo full backup")
    
    if full_backup.status == BackupStatus.COMPLETED:
        print(f"   ✅ Full backup created: {full_backup.backup_id}")
        print(f"   📁 Size: {full_backup.size_bytes} bytes")
        print(f"   🏷️ Agents: {full_backup.agent_count}, Resources: {full_backup.resource_count}")
    else:
        print(f"   ❌ Backup failed: {full_backup.error_message}")
        
    # Demo 2: Crear punto de restauración
    print(f"\n2. Creating restore point...")
    restore_point = await dr_orchestrator.create_restore_point("Demo restore point")
    print(f"   ✅ Restore point created: {restore_point.restore_id}")
    
    # Demo 3: Simular cambios
    print(f"\n3. Simulating changes...")
    
    # Crear otro agente
    tester = ExtendedAgentFactory.create_agent("agent.test.generator", "tester", framework)
    await tester.start()
    
    # Crear otro recurso
    test_resource2 = AgentResource(
        type=ResourceType.TEST,
        name="demo_test",
        namespace="resource.test",
        data={"test_code": "assert True", "framework": "pytest"},
        owner_agent_id=tester.id
    )
    await framework.resource_manager.create_resource(test_resource2)
    
    print(f"   ✅ Added 1 agent and 1 resource")
    print(f"   📊 Current state: {len(framework.registry.list_all_agents())} agents")
    
    # Demo 4: Crear backup incremental
    print(f"\n4. Creating incremental backup...")
    incremental_backup = await dr_orchestrator.backup_engine.create_incremental_backup(
        full_backup.backup_id, "Demo incremental backup"
    )
    
    if incremental_backup.status == BackupStatus.COMPLETED:
        print(f"   ✅ Incremental backup created: {incremental_backup.backup_id}")
        print(f"   📁 Size: {incremental_backup.size_bytes} bytes")
    
    # Demo 5: Simular disaster recovery
    print(f"\n5. Disaster recovery simulation...")
    
    # Simular fallo de sistema restaurando desde punto anterior
    print(f"   📉 Simulating system crash...")
    
    recovery_result = await dr_orchestrator.disaster_recovery_plan("system_crash")
    
    if recovery_result["success"]:
        print(f"   ✅ Recovery successful")
        print(f"   📊 Current state: {len(framework.registry.list_all_agents())} agents")
    else:
        print(f"   ❌ Recovery failed: {recovery_result['error']}")
    
    # Demo 6: Estado del sistema
    print(f"\n6. Recovery system status:")
    status = dr_orchestrator.get_recovery_status()
    
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Demo 7: Historial de backups
    print(f"\n7. Backup history:")
    backup_history = dr_orchestrator.backup_engine.get_backup_history()
    
    for i, backup in enumerate(backup_history[:5], 1):
        print(f"   {i}. {backup.backup_id} ({backup.backup_type.value}) - {backup.status.value}")
        print(f"      Created: {backup.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"      Size: {backup.size_bytes} bytes")
    
    # Cleanup
    await framework.stop()
    await persistence_manager.close()
    
    print(f"\n✅ Backup & Recovery demo completed!")

if __name__ == "__main__":
    asyncio.run(backup_recovery_demo())